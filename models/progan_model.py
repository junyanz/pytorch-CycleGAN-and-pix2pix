import torch

from .base_model import BaseModel
from . import networks
from torch.nn import functional as F

try:
    from apex import amp
except ImportError:
    print("Please install NVIDIA Apex for safe mixed precision if you want to use non default --opt_level")


class ProGanModel(BaseModel):
    """
    This is an implementation of the paper "Progressive Growing of GANs": https://arxiv.org/abs/1710.10196.
    Model requires dataset of type dataset_mode='single', generator netG='progan', discriminator netD='progan'.
    ngf and ndf controlls dimensions of the backbone.
    Network G is a master-generator (for eval) and network C (stands for current) is a current trainable generator.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(netG='progan', netD='progan', dataset_mode='single', beta1=0., ngf=512, ndf=512)
        parser.add_argument('--z_dim', type=int, default=32, help='random noise dim')
        parser.add_argument('--max_steps', type=int, default=6, help='steps of growing')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.z_dim = opt.z_dim
        self.max_steps = opt.max_steps
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'C']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.z_dim, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      init_weights=False)
        self.netC = networks.define_G(opt.z_dim, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      init_weights=False)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                          init_weights=False)

        assert opt.beta1 == 0
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers.append(self.optimizer_C)
            self.optimizers.append(self.optimizer_D)

            if opt.apex:
                [self.netC, self.netD], [self.optimizer_C, self.optimizer_D] = amp.initialize(
                    [self.netC, self.netD], [self.optimizer_C, self.optimizer_D], opt_level=opt.opt_level, num_losses=2)

        self.make_data_parallel()

        # inner counters
        self.total_steps = opt.n_epochs + opt.n_epochs_decay + 1
        self.step = 1
        self.iter = 0
        self.alpha = 0.

        assert self.total_steps > 12
        assert self.opt.crop_size % 2 ** self.max_steps == 0

        # set fusing
        self.netG.eval()
        self.accumulate(0)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_B = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = F.interpolate(self.real_B, size=(4 * 2 ** self.step, 4 * 2 ** self.step), mode='bilinear')
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        batch_size = self.real_B.size(0)
        z = torch.randn((batch_size, self.z_dim, self.opt.crop_size // (2 ** self.max_steps),
                         self.opt.crop_size // (2 ** self.max_steps)),
                        device=self.device)
        self.fake_B = self.netC(z, step=self.step, alpha=self.alpha)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B.detach(), step=self.step, alpha=self.alpha)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_B = self.real_B
        pred_real = self.netD(real_B, step=self.step, alpha=self.alpha)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.opt.apex:
            with amp.scale_loss(self.loss_D, self.optimizer_D, loss_id=0) as loss_D_scaled:
                loss_D_scaled.backward()
        else:
            self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B, step=self.step, alpha=self.alpha)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G = self.loss_G_GAN

        if self.opt.apex:
            with amp.scale_loss(self.loss_G, self.optimizer_C, loss_id=1) as loss_G_scaled:
                loss_G_scaled.backward()
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update generator C
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_C.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_C.step()  # udpate G's weights
        self.accumulate()  # fuse params

    def update_inners_counters(self):
        """
        Update counters of iterations
        """
        self.iter += 1
        self.alpha = min(1, (2 / (self.total_steps // self.max_steps)) * self.iter)
        if self.iter > self.total_steps // self.max_steps:
            self.alpha = 0
            self.iter = 0
            self.step += 1

            if self.step > self.max_steps:
                self.alpha = 1
                self.step = self.max_steps

    def accumulate(self, decay=0.999):
        """
        Accumulate weights from self.C to self.G with decay
        @param decay decay
        """
        par1 = dict(self.netG.named_parameters())
        par2 = dict(self.netC.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def update_learning_rate(self):
        super(ProGanModel, self).update_learning_rate()
        self.update_inners_counters()

import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class HairColorGANModel(BaseModel):
    """
    This class implements a variation of the cycleGAN model with just one generator and one
    discriminator for changing the hair color of people in images.Both the generator and the discriminator 
    take a hair color as additional input in addition to an image. For "real" images this hair color matches
    the hair color in the image. (It is the average of the pixels' color values.) 
    
    The model training requires '--dataset_mode hair' dataset.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(dataset_mode='hair')
        parser.set_defaults(update_html_freq=400)
        parser.set_defaults(print_freq=40)
        parser.set_defaults(display_freq=400)
        parser.set_defaults(norm='batch')
        parser.set_defaults(batch_size=8)
        if is_train:
            parser.add_argument('--lambda_cyc', type=float, default=5.0, help='weight for cycle loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the hairColorGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G', 'cycle', 'idt']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','fake_B','target_color']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt = G(concat(real_B,orig_color_B))
            self.visual_names = ['real_A','orig_color_A','fake_B','target_color', 'rec_A']                  
            self.visual_names.append('real_B')
            self.visual_names.append('orig_color_B')
            self.visual_names.append('idt')

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        self.netG = networks.define_G(6, 3, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(6, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.orig_color_A = input['orig_color_A_img'].to(self.device)
        self.orig_color_B = input['orig_color_B_img'].to(self.device)
        self.target_color = input['target_hair_color_img'].to(self.device)
        self.image_paths = input['path']

        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        image_and_target_color = torch.cat((self.real_A, self.target_color), 1)
        self.fake_B = self.netG(image_and_target_color)  

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        fake_and_target = torch.cat((self.fake_B, self.target_color), 1)
        fake_from_pool = self.fake_pool.query(fake_and_target)
        
        real_with_real_color = torch.cat((self.real_B, self.orig_color_B), 1)
        
        self.loss_D = self.backward_D_basic(self.netD, real_with_real_color, fake_from_pool)

    def backward_G(self):
        """Calculate the loss for generator G"""
        lambda_cyc = self.opt.lambda_cyc
        lambda_idt = self.opt.lambda_identity
        # Identity loss
        if lambda_idt > 0:
            # G should be identity if target hair color is original hair color.
            real_with_real_color = torch.cat((self.real_B, self.orig_color_B), 1)
            self.idt = self.netG(real_with_real_color)
            self.loss_idt = self.criterionIdt(self.idt, self.real_B) * lambda_cyc * lambda_idt
        else:
            self.loss_idt = 0
        
        #calculate self.recA:
        fake_and_orig_color = torch.cat((self.fake_B, self.orig_color_A), 1)
        self.rec_A = self.netG(fake_and_orig_color)
        
        # GAN loss D(G(A))
        fake_and_target = torch.cat((self.fake_B, self.target_color), 1)
        self.loss_G = self.criterionGAN(self.netD(fake_and_target), True)
        # Forward cycle loss || G(G(A)) - A||
        self.loss_cycle = self.criterionCycle(self.rec_A, self.real_A) * lambda_cyc
        # combined loss and calculate gradients
        self.loss_G_total = self.loss_G + self.loss_cycle + self.loss_idt
        self.loss_G_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()             # calculate gradients for G
        self.optimizer_G.step()       # update G's weights
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D's gradients to zero
        self.backward_D()      # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

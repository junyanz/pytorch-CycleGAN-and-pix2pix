import torch
import torch.nn as nn

from .imgs_networks import ResnetGenerator, UnetGenerator, NLayerDiscriminator, PixelDiscriminator
from .avsg_networks import SceneGenerator, SceneDiscriminator
from models.helper_func import get_norm_layer, init_net
###############################################################################
# Helper Functions
###############################################################################




def define_G(opt, gpu_ids=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    if gpu_ids is None:
        gpu_ids = []
    use_dropout = not opt.no_dropout
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if opt.netG == 'resnet_9blocks':
        net = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif opt.netG == 'resnet_6blocks':
        net = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif opt.netG == 'unet_128':
        net = UnetGenerator(opt.input_nc, opt.output_nc, 7, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif opt.netG == 'unet_256':
        net = UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif opt.netG == 'SceneGenerator':
        net = SceneGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)
    return init_net(net, opt.init_type, opt.init_gain, gpu_ids)


def define_D(opt, discriminator_in_nc=1,  gpu_ids=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    if gpu_ids is None:
        gpu_ids = []
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if opt.netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(discriminator_in_nc, opt.ndf, n_layers=3, norm_layer=norm_layer)
    elif opt.netD == 'n_layers':  # more options
        net = NLayerDiscriminator(discriminator_in_nc, opt.ndf, opt.n_layers_D, norm_layer=norm_layer)
    elif opt.netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(discriminator_in_nc, opt.ndf, norm_layer=norm_layer)
    elif opt.netD == 'SceneDiscriminator':
        net = SceneDiscriminator(opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % opt.netD)
    return init_net(net, opt.init_type, opt.init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise ValueError('Invalid gan_mode')
        return loss


def cal_gradient_penalty(netD, conditioning, real_samp, fake_samp, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_samp (tensor array)    -- real images
        fake_samp (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_samp
        elif type == 'fake':
            interpolatesv = fake_samp
        elif type == 'mixed':
            alpha = torch.rand(real_samp.shape[0], 1, device=device)
            alpha = alpha.expand(real_samp.shape[0], real_samp.nelement() // real_samp.shape[0]).contiguous().view(*real_samp.shape)
            interpolatesv = alpha * real_samp + ((1 - alpha) * fake_samp)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(conditioning, interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_samp.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty
    else:
        return 0.0


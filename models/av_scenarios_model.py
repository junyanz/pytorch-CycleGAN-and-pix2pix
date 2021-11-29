"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks


class AvScenariosModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')

        if is_train:
            parser.set_defaults(gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt, is_image_data=False)  # call the initialization method of BaseModel

        # specify the training losses you want to print out.
        # The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks
        self.netG = networks.define_G(opt, self.gpu_ids)
        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            discriminator_in_nc = opt.input_nc + opt.output_nc
            self.netD = networks.define_D(opt, discriminator_in_nc, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
             # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """

        self.real_map = input['map_feat']
        self.real_agents = input['agents_feat']
        # TODO: to device

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # generate the output of the generator given the input map
        self.fake_agents,  self.recounstructed_map = self.netG(self.real_map)

    def backward_D(self):
        # Feed fake (generated) agents to discriminator and calculate its prediction loss
        # we use conditional GANs; we need to feed both input and output to the discriminator
        map_and_fake_agents = torch.cat((self.real_map, self.fake_agents), 1)
        pred_fake = self.netD(map_and_fake_agents.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Feed real (loaded from data) agents to discriminator and calculate its prediction loss
        map_and_real_agents = torch.cat((self.real_map, self.real_agents), 1)
        pred_real = self.netD(map_and_real_agents)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #  the generator should fool the discriminator
        map_and_fake_agents = torch.cat((self.real_map, self.fake_agents), 1)
        pred_fake = self.netD(map_and_fake_agents.detach())
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, we want G(map) = map, since the generator acts also as an encoder-decoder for the map
        self.loss_G_L1 = self.criterionL1(self.recounstructed_map, self.real_map) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # update G's weights

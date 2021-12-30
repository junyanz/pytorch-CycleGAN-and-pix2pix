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
from avsg_utils import pre_process_scene_data
from models.networks import cal_gradient_penalty
#########################################################################################


class AvsgModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """


        # ~~~~  Data
        parser.add_argument('--data_eval', type=str, default='', help='Path for evaluation dataset file')

        parser.add_argument('--augmentation_type', type=str, default='rotate_and_translate',
                            help=" 'none' | 'rotate_and_translate' | 'Gaussian_data' ")


        # ~~~~  Map features
        parser.add_argument('--polygon_name_order', type=list,
                            default=['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks'], help='')
        parser.add_argument('--closed_polygon_types', type=list,
                            default=['crosswalks'], help='')
        parser.add_argument('--max_points_per_poly', type=int, default=20,
                            help='Maximal number of points per polygon element')
        # ~~~~  Agents features
        parser.add_argument('--agent_feat_vec_coord_labels',
                            default=['centroid_x',  # [0]  Real number
                                     'centroid_y',  # [1]  Real number
                                     'yaw_cos',  # [2]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'yaw_sin',  # [3]  in range [-1,1],  sin(yaw)^2 + cos(yaw)^2 = 1
                                     'extent_length',  # [4] Real positive
                                     'extent_width',  # [5] Real positive
                                     'speed',  # [6] Real non-negative
                                     'is_CAR',  # [7] 0 or 1
                                     'is_CYCLIST',  # [8] 0 or 1
                                     'is_PEDESTRIAN',  # [9]  0 or 1
                                     ],
                            type=list)
        parser.add_argument('--max_num_agents', type=int, default=4, help=' number of agents in a scene')

        # ~~~~  General model settings
        if is_train:
            parser.set_defaults(gan_mode='wgangp',  # 'the type of GAN objective. [vanilla| lsgan | wgangp].
                                # vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
                                netD='SceneDiscriminator',
                                netG='SceneGenerator')
            parser.add_argument('--agents_decoder_model', type=str,
                                default='MLP')  # | 'MLP' | 'LSTM'
            # parser.add_argument('--num_samples_pack', type=int,
            #                     default=1)  # accumulate  m samples to classify with D (as in PacGAN)


        if is_train:
            # ~~~~  Training optimization settings
            parser.set_defaults(
                n_epochs=1000,
                lr=0.02,
                lr_policy='constant',  # [linear | step | plateau | cosine | constant]
                lr_decay_iters=1000,  # if lr_policy==step'
                lr_decay_factor=0.9,  # if lr_policy==step'
                num_threads=0,  # threads for loading data, can increase to 4 for faster run if no mem issues
            )
            parser.add_argument('--lambda_reconstruct', type=float, default=1., help='weight for reconstruct_loss ')
            parser.add_argument('--lambda_gp', type=float, default=100., help='weight for gradient penalty in WGANGP')
            parser.add_argument('--reconstruct_loss_type', type=str, default='MSE', help=" 'L1' | 'MSE' ")

            # ~~~~ general model settings
            parser.add_argument('--dim_agent_noise', type=int, default=16, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_map', type=int, default=32, help='Scene latent noise dimension')
            parser.add_argument('--n_point_net_layers', type=int, default=3, help='PointNet layers number')
            parser.add_argument('--use_layer_norm', type=int, default=1, help='0 or 1')

            # ~~~~ map encoder settings
            parser.add_argument('--dim_latent_polygon_elem', type=int, default=8, help='')
            parser.add_argument('--dim_latent_polygon_type', type=int, default=16, help='')
            parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='')
            parser.add_argument('--n_conv_layers_polygon', type=int, default=3, help='')
            parser.add_argument('--n_layers_poly_types_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_sets_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_scene_embedder_out', type=int, default=3, help='')

            # ~~~~ discriminator encoder settings
            parser.add_argument('--dim_discr_agents_enc', type=int, default=16, help='')
            parser.add_argument('--n_discr_out_mlp_layers', type=int, default=3, help='')
            parser.add_argument('--n_discr_pointnet_layers', type=int, default=3, help='')

            # ~~~~   Agents decoder options
            parser.add_argument('--agents_dec_in_layers', type=int, default=3, help='')
            parser.add_argument('--agents_dec_out_layers', type=int, default=3, help='')
            parser.add_argument('--agents_dec_n_stacked_rnns', type=int, default=3, help='')
            parser.add_argument('--agents_dec_dim_hid', type=int, default=512, help='')
            parser.add_argument('--agents_dec_use_bias', type=int, default=1)
            parser.add_argument('--agents_dec_mlp_n_layers', type=int, default=4)
            parser.add_argument('--gru_attn_layers', type=int, default=3, help='')

            # ~~~~ Display settings
            parser.set_defaults(
                display_freq=200,
                update_html_freq=200,
                display_id=0)
            parser.add_argument('--vis_n_maps', type=int, default=2, help='')
            parser.add_argument('--vis_n_generator_runs', type=int, default=3, help='')
            parser.add_argument('--stats_n_maps', type=int, default=10, help='')

        return parser

    #########################################################################################

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options
///
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt, is_image_data=False)  # call the initialization method of BaseModel
        opt.device = self.device
        self.use_wandb = opt.use_wandb
        self.polygon_name_order = opt.polygon_name_order
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(self.agent_feat_vec_coord_labels)
        self.opt = opt
        # specify the training losses you want to print out.
        # The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_reconstruct', 'D_real', 'D_fake', 'D_grad_penalty']

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks
        self.netG = networks.define_G(opt, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if opt.reconstruct_loss_type == 'L1':
                self.criterion_reconstruct = torch.nn.L1Loss()
            elif opt.reconstruct_loss_type == 'MSE':
                self.criterion_reconstruct = torch.nn.MSELoss()
            else:
                raise NotImplementedError
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
            self.gan_mode = opt.gan_mode
            self.lambda_gp = opt.lambda_gp
            ## Debug
            # print('calculating the statistics (mean & std) of the agents features...')
            # from avsg_utils import calc_agents_feats_stats
            # print(calc_agents_feats_stats(dataset, opt.agent_feat_vec_coord_labels, opt.device, opt.num_agents))
            ##
    #########################################################################################
    # def is_sample_valid(self, scene_data):
    #     assert isinstance(scene_data, dict)  # assume batch_size == 1, where the sample is a dict of one scene
    #     agents_feat = scene_data['agents_feat']
    #     # if there are too few agents in the scene - skip it
    #     return len(agents_feat) <= self.num_agents

    #########################################################################################
    def set_input(self, scene_data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_agents, map_feat, self.conditioning = pre_process_scene_data(scene_data, self.opt)
        #########################################################################################

        # self.conditioning = []
        # self.real_agents = []
        # # for i, scene_data in enumerate(data_buffer):
        #     real_agents, map_feat, conditioning = pre_process_scene_data(scene_data, self.opt)
        #     self.real_agents.append(real_agents)
        #     self.conditioning.append(conditioning)

    #########################################################################################

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # generate the output of the generator given the input map
        self.fake_agents = self.netG(self.conditioning)

    #########################################################################################

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # we use conditional GANs; we need to feed both input and output to the discriminator


        # Feed fake agents to discriminator and calculate its prediction loss
        fake_agents_detached = self.fake_agents.detach()  # stop backprop to the generator by detaching
        d_out_for_fake = self.netD(self.conditioning, fake_agents_detached)

        # the loss is 0 if D correctly classify as fake
        self.loss_D_fake = self.criterionGAN(prediction=d_out_for_fake, target_is_real=False)

        # Feed real (loaded from data) agents to discriminator and calculate its prediction loss
        pred_is_real_for_real = self.netD(self.conditioning, self.real_agents)

        # the loss is 0 if D correctly classify as not fake
        self.loss_D_real = self.criterionGAN(prediction=pred_is_real_for_real, target_is_real=True)


        self.loss_D_grad_penalty = cal_gradient_penalty(self.netD, self.conditioning, self.real_agents,
                                                        fake_agents_detached, self)

        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.lambda_gp * self.loss_D_grad_penalty
        self.loss_D.backward()

    #########################################################################################

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #  the generator should fool the discriminator
        d_out_for_fake = self.netD(self.conditioning, self.fake_agents)
        # G wants to make D wrongly classify the fake sample (make D output "True")
        self.loss_G_GAN = self.criterionGAN(prediction=d_out_for_fake, target_is_real=True)

        # Second, we want G(map) = map, since the generator acts also as an encoder-decoder for the map
        self.loss_G_reconstruct = self.criterion_reconstruct(self.fake_agents, self.real_agents)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_reconstruct * self.opt.lambda_reconstruct

        self.loss_G.backward()

    #########################################################################################

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False)  # disable backprop for G
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)  # enable backprop for G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    #########################################################################################


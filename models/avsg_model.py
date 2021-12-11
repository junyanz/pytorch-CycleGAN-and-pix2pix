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
import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
from visualization_utils import visualize_scene_feat


#########################################################################################
def agents_feat_vecs_to_dicts(agents_feat_vecs):
    agents_feat_dicts = []
    n_agents = agents_feat_vecs.shape[0]
    for i_agent in range(n_agents):
        agent_feat_vec = agents_feat_vecs[i_agent]
        agent_feat_dict = ({'centroid': agent_feat_vec[:2],
                            'yaw': torch.atan2(agent_feat_vec[3], agent_feat_vec[2]),
                            'extent': agent_feat_vec[4:6],
                            'speed': agent_feat_vec[6],
                            'agent_label_id': torch.argmax(agent_feat_vec[7:10])})
        for key in agent_feat_dict.keys():
            agent_feat_dict[key] = agent_feat_dict[key].detach().cpu().numpy()
        agents_feat_dicts.append(agent_feat_dict)
    return agents_feat_dicts


#########################################################################################


def agents_feat_dicts_to_vecs(agent_feat_vec_coord_labels, agents_feat_dicts, device):
    dim_agent_feat_vec = len(agent_feat_vec_coord_labels)
    assert agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                           'extent_length', 'extent_width', 'speed',
                                           'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']
    agents_feat_vecs = []
    n_agents = len(agents_feat_dicts)
    for agent_dict in agents_feat_dicts:
        agent_feat_vec = torch.zeros(dim_agent_feat_vec, device=device)
        assert agent_dict['centroid'].shape == torch.Size([1, 2])
        agent_feat_vec[0] = agent_dict['centroid'][0, 0]
        agent_feat_vec[1] = agent_dict['centroid'][0, 1]
        agent_feat_vec[2] = torch.cos(agent_dict['yaw'])
        agent_feat_vec[3] = torch.sin(agent_dict['yaw'])
        assert agent_dict['extent'].shape == torch.Size([1, 2])
        agent_feat_vec[4] = agent_dict['extent'][0, 0]
        agent_feat_vec[5] = agent_dict['extent'][0, 1]
        agent_feat_vec[6] = agent_dict['speed']
        # agent type ['CAR', 'CYCLIST', 'PEDESTRIAN'] is represented in one-hot encoding
        agent_feat_vec[7] = agent_dict['agent_label_id'] == 0
        agent_feat_vec[8] = agent_dict['agent_label_id'] == 1
        agent_feat_vec[9] = agent_dict['agent_label_id'] == 2
        assert agent_feat_vec[7:].sum() == 1
        agents_feat_vecs.append(agent_feat_vec)
    agents_feat_vecs = torch.stack(agents_feat_vecs)
    return agents_feat_vecs


#########################################################################################


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
        parser.add_argument('--polygon_name_order', type=list,
                            default=['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks'], help='')
        parser.add_argument('--closed_polygon_types', type=list,
                            default=['crosswalks'], help='')

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
        parser.set_defaults(netG='SceneGenerator')
        if is_train:
            parser.set_defaults(gan_mode='vanilla', netD='SceneDiscriminator')
            parser.set_defaults(lr=0.002, lr_policy='step', lr_decay_iters=1000)

            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--dim_latent_scene_noise', type=int, default=256, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_map', type=int, default=256, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_all_agents', type=int, default=256, help='')
            parser.add_argument('--dim_latent_scene', type=int, default=512, help='')
            parser.add_argument('--dim_agents_decoder_hid', type=int, default=512, help='')
            parser.add_argument('--dim_latent_polygon_elem', type=int, default=64, help='')
            parser.add_argument('--dim_latent_polygon_type', type=int, default=128, help='')
            parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='')
            parser.add_argument('--n_conv_layers_polygon', type=int, default=3, help='')
            parser.add_argument('--n_point_net_layers', type=int, default=3, help='PointNet layers number')
            parser.add_argument('--max_points_per_poly', type=int, default=20,
                                help='Maximal number of points per polygon element')
            parser.add_argument('--max_num_agents', type=int, default=5,
                                help='Maximal number of agents in a scene')
            parser.add_argument('--min_num_agents', type=int, default=2,
                                help='Minimal number of agents in a scene')
        return parser

    #########################################################################################

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt, is_image_data=False)  # call the initialization method of BaseModel
        self.polygon_name_order = opt.polygon_name_order
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.max_num_agents = opt.max_num_agents
        self.min_num_agents = opt.min_num_agents

        # specify the training losses you want to print out.
        # The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']  # = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['fake_agents', 'real_agents']

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        opt.device = self.device

        # define networks
        self.netG = networks.define_G(opt, self.gpu_ids)
        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            discriminator_in_nc = opt.dim_latent_scene
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

    #########################################################################################

    def set_input(self, scene_data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        assert isinstance(scene_data, dict)  # assume batch_size == 1, where the sample is a dict of one scene

        # Map features - Move to device
        map_feat = dict()
        for poly_type in self.polygon_name_order:
            map_feat[poly_type] = []
            poly_elems = scene_data['map_feat'][poly_type]
            map_feat[poly_type] = [poly_elem.to(self.device) for poly_elem in poly_elems]

        # Agent features -
        #
        agent_dists_to_ego = [np.linalg.norm(agent_dict['centroid'][0, :]) for agent_dict in scene_data['agents_feat']]

        # Change to vector form, Move to device
        agents_feat = []
        agents_feat_vecs = agents_feat_dicts_to_vecs(self.agent_feat_vec_coord_labels,
                                                     scene_data['agents_feat'],
                                                     self.device)
        agents_dists_order = np.argsort(agent_dists_to_ego)
        n_agents_to_use = np.random.randint(low=self.min_num_agents, high=self.max_num_agents+1)
        agents_feat_vecs = agents_feat_vecs[agents_dists_order[:n_agents_to_use]]
        self.real_map = map_feat
        self.real_agents = agents_feat_vecs
    #########################################################################################

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # generate the output of the generator given the input map
        self.fake_agents = self.netG(self.real_map)

    #########################################################################################

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # Feed fake (generated) agents to discriminator and calculate its prediction loss
        # we use conditional GANs; we need to feed both input and output to the discriminator
        # stop backprop to the generator by detaching fake_B
        fake_agents_detached = self.fake_agents.detach()
        pred_fake = self.netD(self.real_map, fake_agents_detached)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Feed real (loaded from data) agents to discriminator and calculate its prediction loss
        pred_real = self.netD(self.real_map, self.real_agents)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    #########################################################################################

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #  the generator should fool the discriminator
        pred_fake = self.netD(self.real_map, self.fake_agents)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G = self.loss_G_GAN
        self.loss_G.backward()
        # # Second, we want G(map) = map, since the generator acts also as an encoder-decoder for the map
        # self.loss_G_L1 = self.criterionL1(self.reconstructed_map, self.real_map) * self.opt.lambda_L1
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1

    #########################################################################################

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    #########################################################################################

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                # Generate image:
                agents_feat_vecs_ = getattr(self, name)
                agents_feat_dicts = agents_feat_vecs_to_dicts(agents_feat_vecs_)
                visual_ret[name] = visualize_scene_feat(agents_feat_dicts, self.real_map)
        return visual_ret
    #########################################################################################

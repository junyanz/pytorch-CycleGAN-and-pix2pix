import torch
from models.base_model import BaseModel
from models import avsg_networks
from avsg_utils import pre_process_scene_data

class AvsgCheckMapEncModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--data_eval', type=str, default='', help='Path for evaluation dataset file')

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
        parser.add_argument('--num_agents', type=int, default=4, help=' number of agents in a scene')

        # ~~~~  Data processing
        parser.add_argument('--augmentation_type', type=str, default='rotate_and_translate',
                            help=" 'none' | 'rotate_and_translate' | 'Gaussian_data' ")

        # ~~~~  General model settings
        if is_train:
            parser.set_defaults(gan_mode='vanilla',  # 'the type of GAN objective. [vanilla| lsgan | wgangp].
                                # vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
                                netD='SceneDiscriminator',
                                netG='SceneGenerator')
            parser.add_argument('--agents_decoder_model', type=str,
                                default='MLP')  # | 'MLP' | 'LSTM'
            parser.add_argument('--use_layer_norm', type=int, default=1, help='0 or 1')

        if is_train:
            # ~~~~  Training optimization settings
            parser.set_defaults(
                n_epochs=1000,
                lr=0.02,
                lr_policy='constant',  # [linear | step | plateau | cosine | constant]
                lr_decay_iters=1000,  # if lr_policy==step'
                lr_decay_factor=0.9,  # if lr_policy==step'
            )
            parser.add_argument('--lambda_reconstruct', type=float, default=100., help='weight for L1 loss')
            parser.add_argument('--lambda_gp', type=float, default=100., help='weight for gradient penalty in WGANGP')

            # ~~~~ general model settings
            parser.add_argument('--dim_agent_noise', type=int, default=16, help='Scene latent noise dimension')
            parser.add_argument('--dim_latent_map', type=int, default=32, help='Scene latent noise dimension')
            parser.add_argument('--n_point_net_layers', type=int, default=3, help='PointNet layers number')

            # ~~~~ map encoder settings
            parser.add_argument('--dim_latent_polygon_elem', type=int, default=8, help='')
            parser.add_argument('--dim_latent_polygon_type', type=int, default=16, help='')
            parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='')
            parser.add_argument('--n_conv_layers_polygon', type=int, default=3, help='')
            parser.add_argument('--n_layers_poly_types_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_sets_aggregator', type=int, default=3, help='')
            parser.add_argument('--n_layers_scene_embedder_out', type=int, default=3, help='')

            # ~~~~ Display settings
            parser.set_defaults(
                display_freq=200,
                update_html_freq=200,
                display_id=0)
            parser.add_argument('--vis_n_maps', type=int, default=2, help='')
            parser.add_argument('--vis_n_generator_runs', type=int, default=4, help='')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt, is_image_data=False)

        opt.device = self.device
        self.polygon_name_order = opt.polygon_name_order
        self.map_enc = avsg_networks.MapEncoder(opt)
        # out layer, in case of scalar regression:
        self.out_layer = torch.nn.Linear(in_features=opt.dim_latent_map, out_features=1, device=self.device)

        self.loss_criterion = torch.nn.L1Loss()
        print('Map encoder parameters: ', [p[0] for p in self.map_enc.named_parameters()])
        self.optimizer = torch.optim.Adam(self.map_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)

    def set_input(self, scene_data):
        assert isinstance(scene_data, dict)  # assume batch_size == 1, where the sample is a dict of one scene
        is_valid, real_agents, conditioning = pre_process_scene_data(scene_data, self.opt)
        # if there are too few agents in the scene - skip it
        if not is_valid:
            return False
        self.conditioning = conditioning
        self.real_agents = real_agents
        self.map_feat = conditioning['map_feat']

        # the task -  scalar regression of the number of lanes:
        n_lane_mid_elem = len(self.map_feat['lanes_mid'])
        self.ground_truth = torch.ones(1, device=self.device) * n_lane_mid_elem

    def forward(self):
        map_latent = self.map_enc(self.map_feat)
        self.prediction = self.out_layer(map_latent)

    def backward(self):
        self.loss = self.loss_criterion(self.prediction, self.ground_truth)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

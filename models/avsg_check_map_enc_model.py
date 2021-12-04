import torch
from models.base_model import BaseModel
from models import avsg_networks

class AvsgCheckMapEncModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """

        parser.add_argument('--data_eval', type=str, default='', help='Path for evaluation dataset file')
        parser.add_argument('--polygon_name_order', type=list,
                            default=['lanes_mid', 'lanes_left', 'lanes_right', 'crosswalks'], help='')
        parser.set_defaults(gan_mode='vanilla')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--dim_latent_scene_noise', type=int, default=256, help='Scene latent noise dimension')
        parser.add_argument('--dim_latent_map', type=int, default=256, help='Scene latent noise dimension')
        parser.add_argument('--dim_latent_polygon_elem', type=int, default=64, help='Scene latent noise dimension')
        parser.add_argument('--dim_latent_polygon_type', type=int, default=128, help='Scene latent noise dimension')
        parser.add_argument('--kernel_size_conv_polygon', type=int, default=5, help='Scene latent noise dimension')
        parser.add_argument('--n_conv_layers_polygon', type=int, default=3, help='Scene latent noise dimension')
        parser.add_argument('--n_point_net_layers', type=int, default=3, help='PointNet layers number')
        parser.add_argument('--max_points_per_poly', type=int, default=20,
                            help='Maximal number of points per polygon element')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt, is_image_data=False)
        self.map_enc = avsg_networks.MapEncoder(opt)
        # out layer, in case of scalar regression:
        self.out_layer = torch.nn.Linear(in_features=opt.dim_latent_map, out_features=1)

        self.loss_criterion = torch.nn.L1Loss()
        print('Map encoder parameters: ', [p[0] for p in self.map_enc.named_parameters()])
        self.optimizer = torch.optim.Adam(self.map_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)

    def set_input(self, scene_data):
        map_feat = scene_data['map_feat']
        self.map_feat = map_feat
        n_lane_mid_elem = len(map_feat['lanes_mid'])
        # the task -  scalar regression of the number of lanes:
        self.ground_truth = n_lane_mid_elem

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

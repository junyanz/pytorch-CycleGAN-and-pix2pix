import torch
from models.base_model import BaseModel
from models import avsg_networks

class AvsgCheckMapEncModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        """

        parser.add_argument('--data_eval', type=str, default='', help='Path for evaluation dataset file')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.map_enc = avsg_networks.MapEncoder(opt)
        self.loss_criterion = torch.nn.L1Loss()
        print('Map encoder parameters: ', [p[0] for p in self.map_enc.named_parameters()])
        self.optimizer = torch.optim.Adam(self.map_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)

    def set_input(self, map_feat):
        self.map_feat = map_feat
        self.ground_truth = None

    def forward(self):
        self.prediction = None

    def backward(self):
        self.loss = self.loss_criterion(self.prediction, self.ground_truth)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

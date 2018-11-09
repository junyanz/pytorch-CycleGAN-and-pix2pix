from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

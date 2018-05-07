from torch.autograd import Variable
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)


def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        assert(opt.align_data == False)
        model = CycleGANModel()
    if opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        assert(opt.align_data == True)
        model = Pix2PixModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

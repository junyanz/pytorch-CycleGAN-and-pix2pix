import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.cycle_gan_model import CycleGANModel

class CycleGANModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.generators = []
        self.netD_A  = None
        self.netD_B = None


class GeneratorPair:

    def __init__(self):
        pass
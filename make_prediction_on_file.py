import argparse

import numpy as np
import torch
from data import create_dataset
from models import create_model

# need to run in the pytorch pix2pix repo
from options.test_options import TestOptions
from PIL import Image


def normalize_image(arr):
    arr = arr - 0.5
    arr = arr / arr.max()
    return arr


def pop_module_before_model(cp):
    for key in list(cp.keys()):
        cp["module." + key] = cp[key]
        cp.pop(key)
    return cp


def normalized_image_to_8bit(arr):
    if arr.min() < 0:
        arr = arr - arr.min()
    arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)


fname = "/Users/zhelinisivanesan/Downloads/20230720T072408000_visual_40_hotsat1_chip_3_3.png"
model_path = "/Users/zhelinisivanesan/Downloads/first_pass_s2_nir.pth"

opt = TestOptions()

opt.dataroot = "./imgs/"
opt.name = "facades_pix2pix"
opt.gpu_ids = []
opt.checkpoints_dir = "./checkpoints"
opt.model = "pix2pix"
opt.input_nc = 3
opt.output_nc = 3
opt.ngf = 64
opt.ndf = 64
opt.netD = "basic"
opt.netG = "unet_256"
opt.n_layers_D = 3
opt.norm = "batch"
opt.init_type = "normal"
opt.init_gain = 0.02
opt.no_dropout = False
opt.dataset_mode = "aligned"
opt.direction = "AtoB"
opt.serial_batches = True
opt.num_threads = 0
opt.batch_size = 1
opt.load_size = 256
opt.crop_size = 256
opt.max_dataset_size = float("inf")
opt.preprocess = "resize_and_crop"
opt.no_flip = True
opt.display_winsize = 256
opt.epoch = "latest"
opt.load_iter = 0
opt.verbose = False
opt.suffix = ""
opt.use_wandb = False
opt.wandb_project_name = "CycleGAN-and-pix2pix"
opt.results_dir = "./results/"
opt.aspect_ratio = 1.0
opt.phase = "test"
opt.eval = False
opt.num_test = 50
opt.isTrain = False
opt.display_id = -1

model = create_model(opt)  # create a model given opt.model and other options
device = torch.device("mps")

cp = torch.load(model_path)


# cp2 = pop_module_before_model(cp)
model.netG.load_state_dict(cp)
# model.netG.to(device)

im = np.array(Image.open(fname))[:, :256].astype(float)

stacked = normalize_image(np.stack((im, im, im)) / 255)

stacked = torch.from_numpy(np.expand_dims(stacked, axis=0))
stacked = stacked.type(torch.FloatTensor)

o = model.netG(stacked)
res = o[0, 0].cpu().detach().numpy()


Image.fromarray(normalized_image_to_8bit(res)).save("res.png")

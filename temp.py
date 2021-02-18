import torch
import torch.nn as nn
from models.networks3d import Unet3dPatchGenerator

import os
import glob
import numpy as np


model = Unet3dPatchGenerator(1, 1, nn.BatchNorm3d)
PATH = '/ocean/projects/asc170022p/rohit33/cyclegan_models/checkpoints/copd_emphysema/5_net_G_A.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
model.eval()

patch_idx = 100
patch_dir = os.path.join('/ocean/projects/asc170022p/rohit33/emphysemapatches/lo/', str(patch_idx))
patch_files = glob.glob(patch_dir + '/*.npy')

patch_arr = np.load(patch_files[1])
patch_tensor = torch.from_numpy(patch_arr)
patch_tensor = patch_tensor.unsqueeze(0)
patch_tensor = patch_tensor.unsqueeze(0)
fake_A = model(patch_tensor, torch.tensor(patch_idx))
fake_A_arr = fake_A.squeeze(0).squeeze(0).detach().numpy()
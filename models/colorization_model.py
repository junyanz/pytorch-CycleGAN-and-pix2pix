from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np


class ColorizationModel(Pix2PixModel):
    def name(self):
        return 'ColorizationModel'

    def __init__(self, opt):
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']

    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L: L channel images (range: [-1, 1], torch tensor array)
            AB: ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb: rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)
        self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)

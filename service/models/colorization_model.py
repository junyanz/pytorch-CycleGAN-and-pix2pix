from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # used for lab2rgb
import numpy as np


class ColorizationModel(Pix2PixModel):
    """This is a subclass of Pix2PixModel for image colorization (black & white image -> colorful images).

    The model training requires '-dataset_model colorization' dataset.
    It trains a pix2pix model, mapping from L channel to ab channels in Lab color space.
    By default, the colorization dataset will automatically set '--input_nc 1' and '--output_nc 2'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use 'colorization' dataset for this model.
        See the original pix2pix paper (https://arxiv.org/pdf/1611.07004.pdf) and colorization results (Figure 9 in the paper)
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        For visualization, we set 'visual_names' as 'real_A' (input real image),
        'real_B_rgb' (ground truth RGB image), and 'fake_B_rgb' (predicted RGB image)
        We convert the Lab image 'real_B' (inherited from Pix2pixModel) to a RGB image 'real_B_rgb'.
        we convert the Lab image 'fake_B' (inherited from Pix2pixModel) to a RGB image 'fake_B_rgb'.
        """
        # reuse the pix2pix model
        Pix2PixModel.__init__(self, opt)
        # specify the images to be visualized.
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb']

    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
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

import os

import PIL
import numpy as np
import torch
from skimage.exposure import rescale_intensity
from skimage import io

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from pathlib import Path


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A")  # create a path '/path/to/data/train/A'
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B")  # create a path '/path/to/data/train/B'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/train/A'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/train/B'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        additional_targets = {"tissue_mask": "mask"} if opt.masks_path else None
        self.transform_A = get_transform(grayscale=(input_nc == 1), size=opt.crop_size, additional_targets=additional_targets, stage=opt.phase)
        self.transform_B = get_transform(grayscale=(output_nc == 1), size=opt.crop_size, additional_targets=additional_targets, stage=opt.phase)
        self.masks_path = Path(opt.masks_path) if opt.masks_path else None

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--masks_path", type=str, default=None, help="optional path containing <image_stem>_mask.png tissue masks")
        return parser

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        A_img = rescale_intensity(
            image=np.array(A_img).astype('uint8'),
            in_range=(0, np.iinfo('uint8').max),
            out_range=(0, 1)
        )
        B_img = rescale_intensity(
            image=np.array(B_img).astype('uint8'),
            in_range=(0, np.iinfo('uint8').max),
            out_range=(0, 1)
        )

        A_pad_mask = np.ones(A_img.shape[:2], dtype='uint8')
        B_pad_mask = np.ones(B_img.shape[:2], dtype='uint8')

        # apply image transformation
        if self.masks_path:
            A_tissue_mask = _load_tissue_mask(self.masks_path, A_path)
            B_tissue_mask = _load_tissue_mask(self.masks_path, B_path)
            A = self.transform_A(image=A_img, mask=A_pad_mask, tissue_mask=A_tissue_mask)
            B = self.transform_B(image=B_img, mask=B_pad_mask, tissue_mask=B_tissue_mask)
        else:
            A = self.transform_A(image=A_img, mask=A_pad_mask)
            B = self.transform_B(image=B_img, mask=B_pad_mask)

        item = {"A": A['image'], "B": B['image'], "A_paths": A_path, "B_paths": B_path, "A_pad_mask": A['mask'], "B_pad_mask": B['mask']}
        if self.masks_path:
            item["A_mask"] = A["tissue_mask"].float().unsqueeze(0)
            item["B_mask"] = B["tissue_mask"].float().unsqueeze(0)
        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def _load_tissue_mask(masks_path: Path, image_path: str) -> np.ndarray:
    mask_path = masks_path / f"{Path(image_path).stem}_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: expected {mask_path}")

    mask = io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype('uint8')

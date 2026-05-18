import os

import PIL
import numpy as np
import torch
from skimage.exposure import rescale_intensity

from data.base_dataset import BaseDataset, get_transform, crop, pad_if_needed, crop_pad
from data.image_folder import make_dataset
from PIL import Image
import random


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
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(grayscale=(input_nc == 1))
        self.transform_B = get_transform(grayscale=(output_nc == 1))

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

        A_img, A_pre_pad_shape = self._crop_pad_img(img=A_img)
        B_img, B_pre_pad_shape = self._crop_pad_img(img=B_img)

        A_img = rescale_intensity(
            image=A_img.astype('uint8'),
            in_range=(0, np.iinfo('uint8').max),
            out_range=(0, 1)
        )
        B_img = rescale_intensity(
            image=B_img.astype('uint8'),
            in_range=(0, np.iinfo('uint8').max),
            out_range=(0, 1)
        )

        A_pad_mask = _construct_pad_mask(
            img=A_img,
            pre_pad_h=A_pre_pad_shape[0],
            pre_pad_w=A_pre_pad_shape[1]
        )
        B_pad_mask = _construct_pad_mask(
            img=B_img,
            pre_pad_h=B_pre_pad_shape[0],
            pre_pad_w=B_pre_pad_shape[1]
        )

        # apply image transformation
        A = self.transform_A(image=A_img, mask=A_pad_mask.astype('uint8'))
        B = self.transform_B(image=B_img, mask=B_pad_mask.astype('uint8'))

        return {"A": A['image'], "B": B['image'], "A_paths": A_path, "B_paths": B_path, "A_pad_mask": A['mask'], "B_pad_mask": B['mask']}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


    def _crop_pad_img(self, img: PIL.Image) -> tuple[np.ndarray, tuple[int, ...]]:
        crop_h, crop_w = self.opt.crop_size, self.opt.crop_size
        h, w = img.size
        max_y = max(h - crop_h, 0)
        max_x = max(w - crop_w, 0)
        crop_y = np.random.randint(0, max_y + 1)
        crop_x = np.random.randint(0, max_x + 1)
        img, pre_pad_shape = crop_pad(
            array=np.array(img),
            y=crop_y,
            x=crop_x,
            desired_h=self.opt.load_size,
            desired_w=self.opt.load_size
        )
        return img, pre_pad_shape


def _construct_pad_mask(img: np.ndarray, pre_pad_w: int, pre_pad_h: int):
    pad_mask = np.zeros(img.shape).astype('bool')
    H, W = pad_mask.shape
    pad_top = (H - pre_pad_h) // 2
    pad_left = (W - pre_pad_w) // 2
    pad_mask[pad_top:pad_top + pre_pad_h, pad_left:pad_left + pre_pad_w] = 1.0
    return pad_mask

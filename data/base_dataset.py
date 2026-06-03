"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import random

import albumentations
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(grayscale=False, convert=True, size=None, additional_targets=None, stage="train"):
    transform_list = []

    if stage != "val":
        transform_list.append(albumentations.SquareSymmetry())
        transform_list.append(albumentations.RandomScale(scale_range=(-0.25, 0.0)))
    if size is not None:
        transform_list.extend([
            albumentations.LongestMaxSize(max_size=size),
            albumentations.PadIfNeeded(
                min_height=size,
                min_width=size,
                fill=0,
                fill_mask=0,
            ),
        ])

    if convert:
        if grayscale:
            transform_list += [albumentations.Normalize((0.5,), (0.5,), max_pixel_value=1.0)]
        else:
            transform_list += [albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list += [albumentations.ToTensorV2()]
    return albumentations.Compose(transform_list, additional_targets=additional_targets)


def __transforms2pil_resize(method):
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print("The image size needs to be a multiple of 4. " "The loaded image size was (%d, %d), so it was adjusted to " "(%d, %d). This adjustment will be done to all images " "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def pad_if_needed(array: np.ndarray, desired_h: int, desired_w: int, channels_last: bool = False, mode: str = 'constant') -> np.ndarray:
    if array.ndim == 2:
        h, w = array.shape
    else:
        if channels_last:
            h, w = array.shape[:-1]
        else:
            h, w = array.shape[-2:]
    pad_h = max(desired_h - h, 0)
    pad_w = max(desired_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return array
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if array.ndim == 2:
        return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    elif array.ndim == 3:
        if channels_last:
            return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode)
        else:
            return np.pad(array, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    else:
        raise ValueError


def crop(array: np.ndarray, y: int, x: int, crop_h: int, crop_w: int, channels_last: bool = False) -> np.ndarray:
    if array.ndim == 2:
        array = array[y:y + crop_h, x:x + crop_w]
    elif array.ndim == 3:
        if channels_last:
            array = array[y:y+crop_h, x:x+crop_w, :]
        else:
            array = array[:, y:y+crop_h, x:x+crop_w]
    else:
        raise ValueError
    return array

def crop_pad(array: np.ndarray, y: int, x: int, desired_h: int, desired_w: int, channels_last: bool = True, pad_mode: str = 'constant') -> tuple[np.ndarray, tuple[int, ...]]:
    """Crop, then pad if needed."""
    array = crop(array=array, y=y, x=x, crop_h=desired_h, crop_w=desired_w, channels_last=channels_last)
    pre_pad_shape = array.shape
    array = pad_if_needed(array=array, desired_h=desired_h, desired_w=desired_w, channels_last=channels_last, mode=pad_mode)
    return array, pre_pad_shape

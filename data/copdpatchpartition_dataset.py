"""Dataset class for separating patients of GOLD scores 0 and 5

The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import numpy as np
import os
from os import path as osp
from glob import glob
import torch
import pandas as pd
from collections import OrderedDict
from monai.transforms import Compose, RandGaussianNoise, Rand3DElastic, RandAdjustContrast
# from data.image_folder import make_dataset
# from PIL import Image

def getlastpart(x):
    return x.split('/')[-1].split('.')[0]

NUM_PATCHES = 581
def one_hot(idx):
    a = torch.zeros(NUM_PATCHES)
    a[idx] = 1
    return a

class CopdpatchpartitionDataset(BaseDataset):
    """A dataset that is separated by low and high levels of emphysema or vessels"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        #parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        #parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values

        # Here, subroot will refer to the file index
        parser.add_argument('--subroot', type=str, required=True)
        parser.add_argument('--patchlocations', type=str, default='/ocean/projects/asc170022p/rohit33/patch_locations.npy')
        parser.add_argument('--patchfloat', type=int, default=0, help='Do we use continuous values for patch locations?')
        parser.add_argument('--augment', type=int, default=0, help='Do we want to augment the dataset? (Using MONAI)')
        parser.add_argument('--partitions', type=int, default=5, help='Number of partitions to train on.')
        return parser


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # Load the file containing patient info

        # Extra parameters for augmentations and location representation
        self.patchlocations = opt.patchlocations
        self.patchfloat = opt.patchfloat
        self.augment = opt.augment
        self.partitions = opt.partitions

        # If augmentations enabled, then use the MONAI transformations
        self.transform_re = Rand3DElastic(mode='bilinear', prob=1.0,
                             sigma_range=(8, 12),
                             magnitude_range=(0, 1024 + 240),  # [-1024, 240] -> [0, 1024+240]
                             spatial_size=(32, 32, 32),
                             translate_range=(12, 12, 12),
                             rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                             scale_range=(0.1, 0.1, 0.1),
                             padding_mode='border',
                             #device='cuda:0'
                             )
        self.transform_rgn = RandGaussianNoise(prob=0.25, mean=0.0, std=50)
        self.transform_rac = RandAdjustContrast(prob=0.25)

        # If we want to use x,y,z then load it first
        if self.patchfloat:
            self.patchlocations = np.load(self.patchlocations)*1.0
            mmax = self.patchlocations.max(0)
            self.patchlocations /= mmax
            self.patchlocations = 2*self.patchlocations - 1         # [581, 3] of range [-1, 1]
        else:
            self.patchlocations = None

        # Path to file index
        subroot = osp.join(self.root, self.opt.subroot)

        # Keep patches for low and high
        self.low_patches = []
        self.high_patches = []

        # For all indices, read the contents, and add to patches
        textfiles = glob(osp.join(subroot, '*.txt'))

        for idxfile in textfiles:
            patchid = int(getlastpart(idxfile))  # Get patch index
            with open(idxfile, 'r') as fi:
                values = fi.readlines()
                sids = [getlastpart(x) for x in values]

            # Given patient IDs and patch index, put it into correct bin
            Numpatches = len(sids)
            for i in range(Numpatches):
                partid = int((i*1.0/Numpatches) / (1.0/self.partitions))
                # Keep in low or high
                if partid < self.partitions - 2:
                    self.low_patches.append((sids[i], patchid, partid))
                if partid > 1:
                    self.high_patches.append((sids[i], patchid, partid))

        # Init empty caches for both datasets
        self.lo_size = len(self.low_patches)
        self.hi_size = len(self.high_patches)

        self.cache = OrderedDict()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        loidx = index%self.lo_size
        hiidx = index%self.hi_size

        # Get values
        sid_A, label_A, part_A = self.low_patches[loidx]
        sid_B, label_B, part_B = self.high_patches[hiidx]

        # Get filenames
        lofile = osp.join(self.root, 'patch_fragmented', '{}'.format(label_A), '{}.npy'.format(sid_A))
        hifile = osp.join(self.root, 'patch_fragmented', '{}'.format(label_B), '{}.npy'.format(sid_B))

        data_A = self.load_patch(lofile)
        data_B = self.load_patch(hifile)

        # convert to x, y, z
        if self.patchfloat:
            label_A = torch.FloatTensor(self.patchlocations[label_A])
            label_B = torch.FloatTensor(self.patchlocations[label_B])

        return {
                'A': data_A,
                'B': data_B,

                'A_patchidx': (label_A, part_A),
                'B_patchidx': (label_B, part_B),

                'path': self.root,

                'A_paths': lofile,
                'B_paths': hifile
            }


    def transform_patch(self, Patch):
        '''
        Just scale the patch to normalize it
        '''
        patch = Patch + 0
        # MONAI augmentations
        if self.augment:
            patch = self.transform_re(patch)
            patch = self.transform_rgn(patch)
            patch = self.transform_rac(patch)

        # Normalize it
        m = -1024
        M = 240
        norm = (patch - m)/(M - m)
        norm = 2*norm - 1
        return norm


    def load_patch(self, filename):
        '''
        Load a patch from image (or cache defined by LRU)
        '''
        if self.cache.get(filename) is None:
            data = np.load(filename)
            self.cache[filename] = data
            self.cache.move_to_end(filename)

        patch = self.cache[filename] + 0
        patch = self.transform_patch(patch[None])

        # Discard item from cache if its getting too big
        if len(self.cache) > self.opt.cache_capacity:
            self.cache.popitem(last=False)

        return patch


    def __len__(self):
        """Return the total number of images."""
        return max(self.lo_size, self.hi_size)



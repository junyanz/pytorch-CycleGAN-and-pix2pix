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
import pandas as pd
from collections import OrderedDict
# from data.image_folder import make_dataset
# from PIL import Image

NUM_PATCHES = 581
def one_hot(idx):
    a = torch.zeros(NUM_PATCHES)
    a[idx] = 1
    return a

class CopdpatchlabelsDataset(BaseDataset):
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
        parser.add_argument('--subroot', type=str, required=True,)
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
        subroot = osp.join(self.root, self.opt.subroot)
        lo_files = []
        hi_files = []
        for r, dirs, files in os.walk(osp.join(subroot, 'lo')):
            files = filter(lambda x: x.endswith('npy'), files)
            files = map(lambda x: osp.join(r, x), files)
            lo_files.extend(list(files))

        for r, dirs, files in os.walk(osp.join(subroot, 'hi')):
            files = filter(lambda x: x.endswith('npy'), files)
            files = map(lambda x: osp.join(r, x), files)
            hi_files.extend(list(files))

        self.lo_files = lo_files
        self.hi_files = hi_files

        # Set sizes so that we can sample as per other datasets
        self.lo_size = len(self.lo_files)
        self.hi_size = len(self.hi_files)
        # Init empty caches for both datasets
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

        lofile = self.lo_files[loidx]
        hifile = self.hi_files[hiidx]

        data_A = self.load_patch(lofile)
        data_B = self.load_patch(hifile)

        # Get patch ids
        label_A = int(lofile.split('/')[-2])
        label_B = int(hifile.split('/')[-2])

        #path = 'temp{}'.format(index)    # needs to be a string
        #data_A = self.load_patch(imgidx % self.size0, patchidx, 0)
        #data_B = self.load_patch(imgidx % self.size4, patchidx, 4)
        return {
                'A': data_A,
                'B': data_B,

                'A_patchidx': label_A,
                'B_patchidx': label_B,

                'path': self.root,

                'A_paths': lofile,
                'B_paths': hifile
            }


    def transform_patch(self, patch):
        '''
        Just scale the patch to normalize it
        '''
        m = -1024
        M = 240
        norm = (patch - m)/(M - m)
        norm = 2*norm - 1
        return norm


    def load_patch(self, filename):
        '''
        Load a patch from image (or cache defined by LRU)
        '''
        if cache.get(filename) is None:
            data = np.load(filename)
            cache[filename] = data
            cache.move_to_end(filename)

        patch = cache[filename] + 0
        patch = self.transform_patch(patch)

        # Discard item from cache if its getting too big
        if len(cache) > self.opt.cache_capacity:
            cache.popitem(last=False)
        return patch[None]


    def __len__(self):
        """Return the total number of images."""
        return max(self.lo_size, self.hi_size)

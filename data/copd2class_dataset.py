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

class Copd2classDataset(BaseDataset):
    """A dataset that contains the GOLD score 0 and 4"""
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
        self.patientdatafile = osp.join(self.root, 'Final10000_Phase1_Rev_28oct16.txt')
        data = pd.read_csv(self.patientdatafile, delimiter='\t')
        sid0 = set(list(data[data['finalGold'] == 0.0]['sid']))
        sid4 = set(list(data[data['finalGold'] == 4.0]['sid']))
        # Sort files into either sid0 or sid4
        allfiles = osp.join(self.root, 'patch')
        for r, dirs, files in os.walk(allfiles):
            files = list(map(lambda x: osp.join(r, x), files))
            files = list(filter(lambda x: x.endswith('npy'), files))
            break
        # Given these files, sort them up
        self.gold0files = []
        self.gold4files = []
        for f in files:
            sid = f.split('/')[-1].split('_')[0]   # Get subject id
            # Add it to either set depending on which gold score it belongs to
            if sid in sid0:
                self.gold0files.append(f)
            elif sid in sid4:
                self.gold4files.append(f)

        # Set sizes so that we can sample as per other datasets
        self.size0 = len(self.gold0files)
        self.size4 = len(self.gold4files)

        # Init empty caches for both datasets
        self.cache0 = OrderedDict()
        self.cache4 = OrderedDict()


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
        patchidx = index%NUM_PATCHES
        imgidx = int(index//NUM_PATCHES)

        path = 'temp{}'.format(index)    # needs to be a string
        data_A = self.load_patch(imgidx % self.size0, patchidx, 0)
        data_B = self.load_patch(imgidx % self.size4, patchidx, 4)
        return {'A': data_A, 'B': data_B, 'path': path, 'A_paths': path, 'B_paths': path}


    def transform_patch(self, patch):
        '''
        Just scale the patch to normalize it
        '''
        m = -1024
        M = 240
        norm = (patch - m)/(M - m)
        norm = 2*norm - 1
        return norm


    def load_patch(self, imgidx, patchidx, goldidx):
        '''
        Load a patch from image (or cache defined by LRU)
        '''
        # Load filelist and cache given gold score
        filelist = self.gold0files if goldidx == 0 else self.gold4files
        cache = self.cache0 if goldidx == 0 else self.cache4
        # Get filename
        filename = filelist[imgidx]
        # Get cache contents or load it in
        if cache.get(filename) is None:
            data = np.load(filename)
            cache[filename] = data
            cache.move_to_end(filename)

        patch = cache[filename][patchidx] + 0
        patch = self.transform_patch(patch)

        # Discard item from cache if its getting too big
        if len(cache) > self.opt.cache_capacity:
            cache.popitem(last=False)
        return patch[None]


    def __len__(self):
        """Return the total number of images."""
        return NUM_PATCHES*max(self.size0, self.size4)

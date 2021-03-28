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
try:
    from data.base_dataset import BaseDataset, get_transform
except:
    from base_dataset import BaseDataset, get_transform
import numpy as np
import os
from os import path as osp
import pandas as pd
import math
from collections import OrderedDict
from glob import glob
from collections import namedtuple
# from data.image_folder import make_dataset
# from PIL import Image


class Copdslice2classDataset(BaseDataset):
    """ A 2 class dataset to separate high and low emphysema """
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
        parser.add_argument('--use_nan', type=int, default=1, help='Do we use nan values? (If yes, then they are added to both classes)')
        parser.add_argument('--frac', type=float, default=0.2)
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

        # Load the directories containing patient info and slices
        self.patchindexdir = osp.join(self.root, 'emphysemaindex')
        self.slicesdir = osp.join(self.root, 'slices')

        self.lofiles = []
        self.hifiles = []

        for file in glob(osp.join(self.patchindexdir, "*.txt")):
            # Get slice ID
            sliceid = int(file.split('/')[-1].split('.')[0])
            sid, emph = np.loadtxt(file, dtype=str, delimiter=',').T
            emph = emph.astype(float)

            # Group by nan and not nan
            nanidx = np.isnan(emph)
            sid_notnan, emph_notnan = sid[~nanidx], emph[~nanidx]
            sid_nan = sid[nanidx]

            N_total = len(emph)
            N_notnan = sum(~nanidx)

            # Get top and bottom fractions
            for i in range(math.ceil(opt.frac * N_notnan)):
                self.lofiles.append((sid_notnan[i], emph_notnan[i], sliceid))

            for i in range(math.ceil(opt.frac * N_notnan)):
                self.hifiles.append((sid_notnan[N_notnan - 1 - i], emph_notnan[N_notnan - 1 - i], sliceid))

            # Check if nans are to be included
            if opt.use_nan:
                for sid in sid_nan:
                    self.lofiles.append((sid, np.nan, sliceid))
                    self.hifiles.append((sid, np.nan, sliceid))

        # Set sizes so that we can sample as per other datasets
        self.losize = len(self.lofiles)
        self.hisize = len(self.hifiles)

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
        lofile, loval, loslice = self.lofiles[index%self.losize]
        hifile, hival, hislice = self.hifiles[index%self.hisize]

        lofile = os.path.join(self.slicesdir, '{}_{}.npy'.format(lofile, str(loslice)))
        hifile = os.path.join(self.slicesdir, '{}_{}.npy'.format(hifile, str(hislice)))

        data_A = np.load(lofile)
        data_B = np.load(hifile)

        data_A = self.transform_patch(data_A)
        data_B = self.transform_patch(data_B)

        print(loval, hival, np.min(data_A), np.max(data_A), np.min(data_B), np.max(data_B))

        return {'A': data_A,
                'B': data_B,

                'A_patchidx': loslice,
                'B_patchidx': hislice,

                'path': '',
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

    def __len__(self):
        """Return the total number of images."""
        return max(self.losize, self.hisize)


if __name__ == "__main__":
    Args = namedtuple('Args', 'frac use_nan dataroot')
    args = Args(frac=0.2, use_nan=0, dataroot='/ocean/projects/asc170022p/rohit33/COPDslices')
    ds = Copdslice2classDataset(args)
    print(len(ds))
    for i in range(5):
        J = (ds[np.random.randint(len(ds))])
        print(J['A_paths'], J['B_paths'])

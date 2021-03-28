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
from glob import glob
import torch
import pandas as pd
from collections import OrderedDict
from monai.transforms import Compose, RandGaussianNoise, Rand3DElastic, RandAdjustContrast
from collections import namedtuple


class CopdslicepartitionDataset(BaseDataset):
    """ A partition dataset to separate high and low emphysema """
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
        parser.add_argument('--use_nan', type=int, default=0, help='Do we use nan values? (If yes, then they are added to both classes)')
        parser.add_argument('--patchfloat', type=int, default=0, help='Do we use continuous values for patch locations?')
        parser.add_argument('--partitions', type=int, default=5)
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

        assert opt.partitions >= 4, 'Should have at least 4 partitions'
        self.partitions = opt.partitions

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

            # Get part ID, and add to respective ids
            for i in range(N_notnan):
                partid = int((i*1.0/N_notnan) * self.partitions)
                if partid > 1:
                    self.hifiles.append((sid_notnan[i], emph_notnan[i], sliceid, partid))
                if partid < self.partitions - 2:
                    self.lofiles.append((sid_notnan[i], emph_notnan[i], sliceid, partid))

            # Check if nans are to be included
            if opt.use_nan:
                for sid in sid_nan:
                    for part in range(self.partitions-2):
                        self.lofiles.append((sid, np.nan, sliceid, partid))
                    for part in range(2, self.partitions):
                        self.hifiles.append((sid, np.nan, sliceid, partid))

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
        lofile, loval, loslice, lopart = self.lofiles[index%self.losize]
        hifile, hival, hislice, hipart = self.hifiles[index%self.hisize]

        lofile = os.path.join(self.slicesdir, '{}_{}.npy'.format(lofile, str(loslice)))
        hifile = os.path.join(self.slicesdir, '{}_{}.npy'.format(hifile, str(hislice)))

        data_A = np.load(lofile)
        data_B = np.load(hifile)

        data_A = self.pad(self.transform_patch(data_A))
        data_B = self.pad(self.transform_patch(data_B))

        print(loval, hival, np.min(data_A), np.max(data_A), np.min(data_B), np.max(data_B))

        return {'A': torch.FloatTensor(data_A[None] + 0),
                'B': torch.FloatTensor(data_B[None] + 0),

                'A_patchidx': (loslice, lopart),
                'B_patchidx': (hislice, hipart),

                'path': '',
                'A_paths': lofile,
                'B_paths': hifile
        }

    def pad(self, img):
        H, W = img.shape
        H, W = min(H, 448), min(W, 448)
        outimg = np.zeros((448, 448)) + img[-1, -1]
        outimg[:H, :W] = img[:H, :W] + 0
        return outimg

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
    Args = namedtuple('Args', 'frac use_nan dataroot partitions')
    args = Args(frac=0.2, use_nan=0, dataroot='/ocean/projects/asc170022p/rohit33/COPDslices', partitions=5)
    ds = CopdslicepartitionDataset(args)
    print(len(ds))
    for i in range(5):
        J = (ds[np.random.randint(len(ds))])
        print(J['A_paths'], J['B_paths'], J['A'].shape, J['B'].shape, J['A'].shape, J['B'].shape)
        print(J['A_patchidx'], J['B_patchidx'])
        print()

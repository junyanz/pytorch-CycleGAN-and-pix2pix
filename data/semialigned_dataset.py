import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision
import numpy as np

class SemiAlignedDataset(BaseDataset):
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

        #unaligned data
        self.unaligned_dir_A = os.path.join(opt.dataroot,'unaligned', opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.unaligned_dir_B = os.path.join(opt.dataroot, 'unaligned', opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.unaligned_A_paths = sorted(make_dataset(self.unaligned_dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.unaligned_B_paths = sorted(make_dataset(self.unaligned_dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.unaligned_A_size = len(self.unaligned_A_paths)  # get the size of dataset A
        self.unaligned_B_size = len(self.unaligned_B_paths)  # get the size of dataset B

        #aligned data
        self.aligned_dir_A = os.path.join(opt.dataroot,'aligned', opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.aligned_dir_B = os.path.join(opt.dataroot, 'aligned', opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.aligned_A_paths = sorted(make_dataset(self.aligned_dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.aligned_B_paths = sorted(make_dataset(self.aligned_dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.aligned_A_size = len(self.aligned_A_paths)  # get the size of dataset A
        self.aligned_B_size = len(self.aligned_B_paths)  # get the size of dataset B

        #create a dict to easily map pairs
        #when we call __getitem__ for aligned, we will sample from aligned_A_paths
        self.aligned_glossary = {}
        for im in self.aligned_B_paths:
            label = im.split('/')[-2]
            if not label in self.aligned_glossary:
                self.aligned_glossary[label] = [im]
            else:
                self.aligned_glossary[label].append(im)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))




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
        flip = random.randint(0, 1)

        if flip == 0: #unaligned
            A_path = self.unaligned_A_paths[index % self.unaligned_A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.unaligned_B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.unaligned_B_size - 1)
            B_path = self.unaligned_B_paths[index_B]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

        else: #aligned
            A_path = self.aligned_A_paths[index % self.aligned_A_size]
            label = A_path.split('/')[-2]
            aligned_B_paths = self.aligned_glossary[label]
            if self.opt.serial_batches:   # make sure index is within then range
                print ("here")
                index_B = index % len(aligned_B_paths)
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, len(aligned_B_paths) - 1)
            B_path = aligned_B_paths[index_B]

            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            # A_img = torchvision.transforms.functional.crop(A_img, top = 300 , left =0, height = 632-300 , width = 312)
            # B_img = torchvision.transforms.functional.crop(B_img, top = 300 , left =0, height = 632-300 , width = 312)

            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.unaligned_A_size, self.unaligned_B_size)

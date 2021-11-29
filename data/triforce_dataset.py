import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

from data.ss_image_dataset import SSImageDataset, AugmentFlag, ZoomLevelFlag, ImageToTensorTransform


class TriforceDataset(BaseDataset):
    """
    This dataset class can load the triforce datasets.

    It requires two sub-datasets, each with a different set of images. (E.g. 2 different consoles).
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        split = opt.tf_split
        console_a = opt.tf_console_a
        console_b = opt.tf_console_b

        dataset_maker = lambda console: SSImageDataset(root_dirs=self.root,
                                                       image_sets=['internetarchive',
                                                                   'mobygames',
                                                                   'superfamicomdotorg',
                                                                   'vgm'],
                                                       train=opt.isTrain,
                                                       train_pct=split,
                                                       consoles=[console],
                                                       augments_allowed=AugmentFlag.AllowFlipBoth,
                                                       zoom_levels=ZoomLevelFlag.CleanSet,
                                                       # transform=ImageToTensorTransform(),
                                                       exclude_noisy_files=False)
        self.dataset_a = dataset_maker(console_a)
        self.dataset_b = dataset_maker(console_b)

        max_data_size = min(max(len(self.dataset_a), len(self.dataset_b)), opt.max_dataset_size)
        self.A_paths = sorted([p for p in self.dataset_a.get_filenames()])[
                       :max_data_size]  # load images from '/path/to/data/trainA'
        self.B_paths = sorted([p for p in self.dataset_b.get_filenames()])[
                       :max_data_size]  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), method=Image.NEAREST)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), method=Image.NEAREST)

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
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

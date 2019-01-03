import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # we manually crop and flip in __getitem__ to make sure we apply the same crop and flip for image A and B
        # we disable the cropping and flipping in the function get_transform
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1), crop=False, flip=False)
        self.transform_B = get_transform(opt, grayscale=(output_nc == 1), crop=False, flip=False)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary of A, B, A_paths and B_paths
            A(tensor) - - an image in the input domain
            B(tensor) - - its corresponding image in the target domain
            A_paths(str) - - image paths
            B_paths(str) - - image paths
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        # apply the same cropping to both A and B
        if 'crop' in self.opt.preprocess:
            x, y, h, w = transforms.RandomCrop.get_params(A, output_size=[self.opt.crop_size, self.opt.crop_size])
            A = A.crop((x, y, w, h))
            B = B.crop((x, y, w, h))
        # apply the same flipping to both A and B
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT)
            B = B.transpose(Image.FLIP_LEFT_RIGHT)
        # call standard transformation function
        A = self.transform_A(A)
        B = self.transform_B(B)
        print(AB_path, index)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

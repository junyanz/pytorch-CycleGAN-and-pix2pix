import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class HairTestModeDataset(BaseDataset):
    """A dataset class for dataset of pairs {portrait image, target hair color}.

    It assumes that the directory '/path/to/data/test' contains image pairs in the form of {portrait image,color}.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_images_hair = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.image_hair_paths = sorted(make_dataset(self.dir_images_hair, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an input image
            B (tensor) - - a dummy tensor that is not used for test mode
            orig_color_A_img (tensor) - - a dummy tensor that is not used for test mode
            orig_color_B_img (tensor) - - a dummy tensor that is not used for test mode
            target_hair_color_img (tensor) - - the target hair color for the input image
            path (str) - - image path
        """
        # read a image given a random integer index
        path = self.image_hair_paths[index]
        img_and_hair = Image.open(path).convert('RGB')
        # split img_and_hair image into two images (one of them of the target hair color)
        w, h = img_and_hair.size
        w2 = int(w / 2)
        img = img_and_hair.crop((0, 0, w2, h))
        hair = img_and_hair.crop((w2, 0, w, h))

        A = self.transform(img)
        target_hair_color_img = self.transform(hair)
         
        dummy = torch.zeros(1,1,1,1)

        return {'A': A, 'B': dummy, 'orig_color_A_img': dummy, 'orig_color_B_img': dummy, 'target_hair_color_img':target_hair_color_img, 'path': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_hair_paths)

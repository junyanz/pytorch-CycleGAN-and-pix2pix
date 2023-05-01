import torchvision
import torchvision.transforms as T
import PIL
import skimage.exposure
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision
import torch
import numpy as np
import torchvision.transforms as T
from skimage import io
import tifffile

class rawDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.transforms_he =  get_transform(train= opt.train, size=256, HE_IF = "he")
        self.transforms_if = get_transform(train=opt.train, size=256 , HE_IF = "if")
        print(f"training? {opt.train}")
        self.imgs = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_he"))) if ".tif" in logo_name]  
))# HE
        self.targets = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_if"))) if ".tif" in logo_name]  
))# IF  
    def __getitem__(self, index):
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        img = tifffile.imread(img_path)
        target = tifffile.imread(targets_path)
        target = skimage.util.img_as_float32(target)#**ADDED**
        img = np.moveaxis(img, 0, 2)
        target = np.moveaxis(target, 0, 2)
        img, target = self.transforms_he(img), self.transforms_if(target)
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
    
    def __len__(self):
        return len(self.imgs)
    
def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    transforms.append(T.ToTensor())

    if train==1:
        print("training data!!")
        transforms.append(T.Resize((size,size)))
        # Adding transformation to both inputs
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        print("testing data!!")
        transforms.append(T.Resize((size,size)))
        
    return T.Compose(transforms)

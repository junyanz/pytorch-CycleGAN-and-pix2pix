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

class adapterfullDataset(BaseDataset):
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
        self.transforms_he =  get_transform(train= True, size=256, HE_IF = "he")
        self.transforms_if = get_transform(train=True, size=256 , HE_IF = "if")
        self.imgs = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_he"))) if ".tif" in logo_name]  
))# HE
        self.targets = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_if"))) if ".tif" in logo_name]  
))# IF 
    """
    def __getitem__(self, index):
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        img = io.imread(img_path)
        img = (np.array(img)/255).astype("float32")
        img = torch.as_tensor(img)
        img = img.permute((2,0,1))
        target = io.imread(targets_path)
        target = (target/(256.*256.-1.))
        target = torch.as_tensor(target.astype("float32"))
        #print(img.shape, target.shape) 
        img, target = self.transforms_he(img), self.transforms_if(target)
        #return {'A': img, 'B': target, 'A_paths': img_path, 'B_paths': targets_path}
        #print(img.shape, target.shape)
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
    def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    #transforms.append(T.ToTensor())
    if train:
        if HE_IF=="he":
            transforms.append(T.Resize((size,size)))
        elif HE_IF=="if":
            transforms.append(T.Resize((size,size)))
        else:
            transforms.append(T.Resize((size,size)))
    # Adding transformation to both inputs
    
    transforms.append(T.RandomHorizontalFlip(0.5))
    #
        
    return T.Compose(transforms)
    """
    def __getitem__(self, index):
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        img = tifffile.imread(img_path)
        img = skimage.util.img_as_float32(img)#**ADDED**
        #img = torch.as_tensor(img)
        target = tifffile.imread(targets_path)
        target = skimage.util.img_as_float32(target)#**ADDED**
        # Normalize images
        CHANNELS = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18)
        img = np.moveaxis(img, 0, 2)
        #CHANNELS = range(19)
        target = np.dstack([
            skimage.exposure.rescale_intensity(
                target[c],
                in_range=(np.percentile(target[c], 1), np.percentile(target[c], 99.9)),#**ADDED**: reduce clipping to 1%
                out_range=(0, 1)
            ) 
            for c in CHANNELS
        ]).astype(np.float32)#**ADDED**
        
        #target = torch.as_tensor(target)
        #img = torch.as_tensor(img.astype(np.float32))
        #target = torch.as_tensor(target.astype(np.float32))
        #print("bef: ",img.shape, target.shape)
        img, target = self.transforms_he(img), self.transforms_if(target)
        #print("after", img.shape, target.shape)
        #return {'A': img, 'B': target, 'A_paths': img_path, 'B_paths': targets_path}
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
    
    def __len__(self):
        return len(self.imgs)
#"""
def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        if HE_IF=="he":
            transforms.append(T.Resize((size,size)))
        elif HE_IF=="if":
            transforms.append(T.Resize((size,size)))
        else:
            transforms.append(T.Resize((size,size)))
        transforms.append(T.RandomHorizontalFlip(0.5))
        
    return T.Compose(transforms)
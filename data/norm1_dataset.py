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

class norm1Dataset(BaseDataset):
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
    def __getitem__(self, index):
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        img = tifffile.imread(img_path)
        img = skimage.util.img_as_float32(img)#**ADDED**
        img = np.moveaxis(img, 0, 2)
        #img = torch.as_tensor(img)
        target = tifffile.imread(targets_path)
        target = skimage.util.img_as_float32(target)#**ADDED**
        # Normalize images

        CHANNELS = range(19)
        # CRC01 
        maxi = [0.2724814,  0.0690039, 0.016164,   0.05094079, 0.03598041, 0.00516317,
                 0.01510232, 0.01042182, 0.0339626,  0.01665093, 0.00781439, 0.01586902,
                 0.02887617, 0.0281076,  0.03916262, 0.01380907, 0.02480303, 0.06438798,
                 0.01835778]
        mini = [0.01051875, 0.00338394, 0.00239182, 0.0028123,  0.00282186, 0.0017648,
                 0.00256854, 0.00107948, 0.00212739, 0.00301781, 0.00221916, 0.00282591,
                 0.00089257, 0.00047432, 0.00176231, 0.0026303,  0.00265461, 0.00235595,
                 0.00118385
                ]
        #print(target.shape)
        target = np.dstack([
            skimage.exposure.rescale_intensity(
                target[c],
                in_range=(mini[c], maxi[c]),
                out_range=(0, 1)
            ) 
            for c in CHANNELS
        ]).astype(np.float32)    
        
        img, target = self.transforms_he(img), self.transforms_if(target)# HWC
        
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
    def __len__(self):
        return len(self.imgs)
    
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
        
    return T.Compose(transforms)

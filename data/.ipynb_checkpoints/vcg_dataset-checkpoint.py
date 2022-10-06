import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision
import torch
import numpy as np
import torchvision.transforms as T
from skimage import io
class vcgDataset(BaseDataset):
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
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.transforms_he =  get_transform(train= True, size=256, HE_IF = "he")
        self.transforms_if = get_transform(train=True, size=256 , HE_IF = "if")
        self.imgs = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_he"))) if ".tif" in logo_name]  
))# HE
        self.targets = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_if"))) if ".tif" in logo_name]  
))# IF  
    def __getitem__(self, index):

                
        ## Data paths 
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        mask_path = os.path.join(self.root, "mask", self.targets[index])
        img = io.imread(img_path)
        img = (np.array(img)/255).astype("float32")
        img = torch.as_tensor(img)
        #img = torch.permute(img, (2, 0, 1))
        img = img.permute((2,0,1))
        target = io.imread(targets_path)
        
        channels = [0,3,17] # visual markers
        target = np.array([target[ch,:,:] for ch in channels]).astype("float32")
        
        target = (target/(256.*256.-1.))
        target = torch.as_tensor(target.astype("float32"))
        mask = io.imread(mask_path)
        mask = torch.as_tensor(mask.astype("float32"))
        img, target = self.transforms_he(img), self.transforms_if(target)
        #return img, target
        #return {'A': img, 'B': target[:3,:,:], 'A_paths': img_path, 'B_paths': targets_path}
        return {'A': img, 'B': target, 'A_paths': img_path, 'B_paths': targets_path}
        
    def __len__(self):
        return len(self.imgs)
def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    
    if train:
        if HE_IF=="he":
            #transforms.append(T.Normalize(mean=[0.0663, 0.0628, 0.0703], std=[3.6709, 3.4846, 3.8646])),
            transforms.append(T.Resize((size,size)))
            #transforms.append(T.ColorJitter(brightness=.5, hue=.3))
        elif HE_IF=="if":
            """
            transforms.append(T.Normalize(
                mean=[1.0794, 0.1006, 0.0737, 0.0806, 0.0862, 0.0694, 0.0795, 0.0680, 0.0776,
                        0.1135, 0.0700, 0.1048, 0.0942, 0.0571, 0.3018, 0.0965, 0.0870, 0.4430,
                        0.0651], 
                std=[74.1644,  5.5260,  4.0343,  4.4169,  4.7434,  3.7954,  4.3549,  3.7224,
                     4.2502,  6.2776,  3.8380,  5.8004,  5.3099,  3.4354, 20.0305,  5.3270,
                     4.8978, 28.0817,  3.5912])),
            """
            transforms.append(T.Resize((size,size)))
        else:
            transforms.append(T.Resize((size,size)))
            
        ## TODO: add here other augmentations
        #transforms.append(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
        #transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.Lambda(lambda x: x[None])),
        #transforms.append(T.ToTensor())
    return T.Compose(transforms)

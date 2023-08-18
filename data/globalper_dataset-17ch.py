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

class globalperDataset(BaseDataset):
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
        target = tifffile.imread(targets_path)# 0 22850
        img = np.moveaxis(img, 0, 2)
        CHANNELS = range(19)

        target = skimage.util.img_as_float32(target)#**ADDED**
        target = np.delete(target, [5, 10], axis=0)
        CHANNELS   = range(len(target))
        marker = ['Hoechst','AF1','CD31','CD45','CD68','CD4',
            'FOXP3','CD8a','CD45RO','PD-L1','CD3e','CD163',
            'E-Cadherin','PD-1','Ki-67','Pan-CK','SMA']
        FOXP3 = 0.004
        PD_L1 = 0.004
        E_CADHERIN=0.004
        PAN_CK = 0.005
        threshold = [0, 0 , 0 , 0 , 0 ,  0.004, FOXP3,  0 , 0  ,  PD_L1  ,  0 ,  0,  E_CADHERIN ,  0,   0,    PAN_CK,    0]
        val_lower = [0.0075446152706507945, 0.003629292572446398, 0.0029421623111127336, 0.0032673615865327333, 0.003105331579194418, 0.0030575714542808705, 0.0026497724921902875, 0.0028753122741884984, 0.0036289414492961377,  0.00345574265440891, 0.0025111015143016785, 0.0021671432928285664, 0.0025467569248282585, 0.0030193113044666765, 0.0030956476865261046, 0.0030234318537183413, 0.00217884851425964]
        val_upper = [0.2265105275027782, 0.014934057374848634, 0.01727550464754221, 0.049639684407353196, 0.031028947067013278,      0.01819653593757534, 0.009791264213451343, 0.019860039003553712, 0.020504618506112984, 0.014383675656430854, 0.022218277600566532, 0.03350917936382925, 0.022567319253225353, 0.008643826065390162, 0.01750753829050882, 0.03975980690939506, 0.01822751117716048]
        target = np.dstack([
            skimage.exposure.rescale_intensity(
                target[c],
                in_range=(val_lower[c], val_upper[c]),#**ADDED**: reduce clipping to 1%
                out_range=(0, 1)
            ) * (target[c] > threshold[c])  
            for c in CHANNELS
        ]).astype(np.float32)#**ADDED**
        
        img, target = self.transforms_he(img), self.transforms_if(target)
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
        
    def __len__(self):
        return len(self.imgs)
    
def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    transforms.append(T.ToTensor())

    if train==1:
        transforms.append(T.Resize((size,size)))
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        transforms.append(T.Resize((size,size)))
        
    return T.Compose(transforms)

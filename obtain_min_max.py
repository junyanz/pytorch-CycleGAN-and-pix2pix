store_path = "data"
SAVER = True #False
import os
#root = "/home/sebasmos/Desktop/DATASETS/pix2pix/val"
#root = "/n/pfister_lab2/Lab/vcg_biology/ORION/ORION-PATCH/C1-C40-patches/CRC06"
root = "/net/coxfs01/srv/export/coxfs01/pfister_lab2/share_root/Lab/scajas/DATASETS/DATASET_pix2pix_merged/DATAFULL/train"

os.listdir(root)

import copy
import os
import torchvision
import torchvision.transforms as T
import skimage.exposure
import torch
from PIL import Image

import torch
import torch.utils.data
import torchvision


#import transforms as T
import torchvision.transforms as visionT
import pdb
import numpy as np
import cv2

from skimage import io

import glob

import random
import tifffile
import pickle
import time
import matplotlib.pyplot as plt


if not os.path.exists(store_path):
    os.makedirs(store_path)
else:
    print('folder already exists')

def plot_imgs(imgs, titles):
    """
    Generate visualization of list of arrays
    :param imgs: list of arrays, each numpy array is an image of size (width, height)
    :param titles: list of titles [string]
    """
    # create figure
    fig = plt.figure(figsize=(50, 50))
    # loop over images
    for i in range(len(imgs)):
        fig.add_subplot(4, 4, i + 1)
        plt.imshow(imgs[i])
        plt.title(str(titles[i]))
        plt.axis("off")

import torchvision
import torchvision.transforms as T
#import transforms as T # Custom version 
import PIL
import skimage.exposure

class Dataloader_vcg_sizer(torch.utils.data.Dataset): 
    def __init__(self, root, 
                 Check_files = False, 
                 mask_flag=False, 
                 augment=False, 
                 transforms_he=None, 
                 transforms_if=None):
        self.root = root
        self.Check_files = Check_files
        self.augment = augment
        self.transforms_he = transforms_he
        self.transforms_if = transforms_if
        self.mask_flag = mask_flag
        self.imgs = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(root, "img_he"))) if ".tif" in logo_name]  
))# HE
        self.targets = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(root, "img_if"))) if ".tif" in logo_name]  
))# IF  
        #self.masks = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(root, "mask"))) if ".tif" in 
    def __getitem__(self, idx):
        ## Data paths 
        # https://syspharm.slack.com/archives/C02SC9VS7AA/p1663364492474239 
        img_path = os.path.join(self.root, "img_he", self.imgs[idx])        
        targets_path = os.path.join(self.root, "img_if", self.targets[idx])
        #mask_path = os.path.join(self.root, "mask", self.targets[idx])
        #print(targets_path)
        # Read images
        img = tifffile.imread(img_path)
        target = tifffile.imread(targets_path)#.astype("float32")
        target = skimage.util.img_as_float32(target)
        # Normalize images
        #CHANNELS = (0, 3, 17)
        img = np.moveaxis(img, 0, 2)
        channels = range(19)#[0,3,17] # visual markers
        target = np.array([target[ch,:,:] for ch in channels]).astype("float32")
        target = target.transpose(1,2,0)
        #target=torch.as_tensor(target.astype("float32"))

        #print(target.shape)
        """
        target = np.dstack([
            skimage.exposure.rescale_intensity(
                target[c],
                in_range=(np.percentile(target[c], 1), np.percentile(target[c], 99.9)),
                out_range=(0, 1)
            ) 
            for c in CHANNELS
        ]).astype(np.float32)
        """
        if self.augment is not None:        
            img, target = self.transforms_he(img), self.transforms_if(target)
            return img, target
        
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

transforms_he =  get_transform(train= True, size=256, HE_IF = "he")
transforms_if = get_transform(train=True, size=256 , HE_IF = "if")

dataset_test = Dataloader_vcg_sizer(root, 
                                    Check_files = False, 
                                    mask_flag= False, 
                                    augment=True, 
                                    transforms_he=transforms_he, 
                                    transforms_if=transforms_if)

print("Dataset size: ", len(dataset_test), "HE-IF pairs")

dataloader = dataset_test 
num_batches = 0

dict_HE = {"min": np.ones((3,1)), "max":np.zeros((3,1))}


def get_minmax(x):
    n = x.shape[0]
    minimo = np.ones((n,1))
    maximo = np.zeros((n,1))
    for ch in range(x.shape[0]):
            if np.array(x[ch,:,:].min())<minimo[ch,:]:
                minimo[ch,:] = x[ch,:,:].min()
            if np.array(x[ch,:,:].max())>maximo[ch,:]:
                maximo[ch,:] = x[ch,:,:].max()
    return minimo, maximo

def calculate_minmax(dataloader):
    _he_min=0
    _he_max=0
    _if_min=0
    _if_max=0
    batches = 0
    for idx in range(len(dataloader)):
        img, target = dataset_test[idx]
        #print(f"Range for HE]-> [{img.min(), img.max()}] - Shape: [{img.shape}]")
        #print(f"Range for IF]-> [{target.min(), target.max()}] - Shape: [{target.shape}]")
        minimoHE, maximoHE = get_minmax(img)
        minimoIF, maximoIF = get_minmax(target)#target.min(axis=(1,2)), target.max(axis=(1,2))#
        _he_min +=minimoHE
        _he_max +=maximoHE
        _if_min +=minimoIF
        _if_max +=maximoIF
        batches+=1
    return _if_min / batches, _if_max / batches 
#import json
#out_file = open("minmax.json","w")

min,max = calculate_minmax(dataloader)
print("Method 1".center(60,"-"))
print("IF MAX: ",max.T)
print("IF MIN: ", min.T )
#print(json.dumps(f))
#json.dump(f, "minmax.json", ident=6)
#out_file.close()
maximoIF = np.mean([np.array(dataset_test[idx][1]).max(axis=(1,2)) for idx in range(len(dataloader))], axis=0)
minIF = np.mean([np.array(dataset_test[idx][1]).min(axis=(1,2)) for idx in range(len(dataloader))], axis=0)
print("Method 2".center(60,"-"))
print("IF MAX: ",maximoIF)
print("IF MIN: ", minIF )
"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import skimage
from skimage import io
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): 
            image_tensor = input_image.data

        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy arra
        if image_numpy.shape[0] == 1: 
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        
        if np.absolute(image_numpy.max())>1.0 or np.absolute(image_numpy.min())>1.0:
            
            image_numpy = (image_numpy-image_numpy.min())/(image_numpy.max()-image_numpy.min())
            
            image_numpy = skimage.util.img_as_ubyte(image_numpy)
        else:
            image_numpy = skimage.util.img_as_ubyte(image_numpy)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.astype(imtype)
    return image_numpy
def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    print("Saving image on path ", image_path, "and shape: ", image_numpy.shape)
    #image_numpy = np.dstack([
    #            skimage.exposure.rescale_intensity(image_numpy[:,:,c], out_range=(0, 255)) 
    #            for c in range(image_numpy.shape[2])
    #        ])
    io.imsave(image_path, image_numpy)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

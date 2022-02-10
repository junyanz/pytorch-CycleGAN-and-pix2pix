"""
@Author: David Wang
@Function: This file is used to remove all the noises in the background and convert the image to Grayscale for CycleGAN outputted sketches.
"""
import numpy as np
from PIL import Image
from PIL import ImageOps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sketch_image', type=str, help="input sketches")
parser.add_argument('--ground_image', type=str, help="ground truth images")
parser.add_argument('--output_path', type=str, help="output image path")
opts=parser.parse_args()

src = np.array(ImageOps.grayscale(Image.open(opts.sketch_image)))      #sketch
mask = np.array(ImageOps.grayscale(Image.open(opts.ground_image)))     #ground truth
alpha = 0.01                                                 #threshold

#create binary mask, -1 for white areas, 0 for black areas
mask = np.ceil(mask/(-255) - alpha)
#cut out only the white areas of the sketch
mask = (src) * mask
#find the difference between mask value and 255
mask = mask + 255
#add the difference to the original sketch, making all masked areas white
dst = src + mask
#save image
Image.fromarray(dst.astype(np.uint8)).save(opts.output_path)

# HED batch processing script; modified from https://github.com/s9xie/hed/blob/master/examples/hed/HED-tutorial.ipynb
# Step 1: download the hed repo: https://github.com/s9xie/hed
# Step 2: download the models and protoxt, and put them under {caffe_root}/examples/hed/
# Step 3: put this script under {caffe_root}/examples/hed/
# Step 4: run the following script:
#       python batch_hed.py --images_dir=/data/to/path/photos/ --hed_mat_dir=/data/to/path/hed_mat_files/
# The code sometimes crashes after computation is done. Error looks like "Check failed: ... driver shutting down". You can just kill the job.
# For large images, it will produce gpu memory issue. Therefore, you better resize the images before running this script.
# Step 5: run the MATLAB post-processing script "PostprocessHED.m"


import caffe
import numpy as np
from PIL import Image
import os
import argparse
import sys
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(description='batch proccesing: photos->edges')
    parser.add_argument('--caffe_root', dest='caffe_root', help='caffe root', default='../../', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel', default='./hed_pretrained_bsds.caffemodel', type=str)
    parser.add_argument('--prototxt', dest='prototxt', help='caffe prototxt file', default='./deploy.prototxt', type=str)
    parser.add_argument('--images_dir', dest='images_dir', help='directory to store input photos', type=str)
    parser.add_argument('--hed_mat_dir', dest='hed_mat_dir', help='directory to store output hed edges in mat file', type=str)
    parser.add_argument('--border', dest='border', help='padding border', type=int, default=128)
    parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id', type=int, default=1)
    args = parser.parse_args()
    return args


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
# Make sure that caffe is on the python path:
caffe_root = args.caffe_root   # this file is expected to be in {caffe_root}/examples/hed/
sys.path.insert(0, caffe_root + 'python')


if not os.path.exists(args.hed_mat_dir):
    print('create output directory %s' % args.hed_mat_dir)
    os.makedirs(args.hed_mat_dir)

imgList = os.listdir(args.images_dir)
nImgs = len(imgList)
print('#images = %d' % nImgs)

caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)
# load net
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
# pad border
border = args.border

for i in range(nImgs):
    if i % 500 == 0:
        print('processing image %d/%d' % (i, nImgs))
    im = Image.open(os.path.join(args.images_dir, imgList[i]))

    in_ = np.array(im, dtype=np.float32)
    in_ = np.pad(in_, ((border, border), (border, border), (0, 0)), 'reflect')

    in_ = in_[:, :, 0:3]
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_.transpose((2, 0, 1))
    # remove the following two lines if testing with cpu

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
    # get rid of the border
    fuse = fuse[border:-border, border:-border]
    # save hed file to the disk
    name, ext = os.path.splitext(imgList[i])
    sio.savemat(os.path.join(args.hed_mat_dir, name + '.mat'), {'edge_predict': fuse})

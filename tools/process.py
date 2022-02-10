"""
@article: pix2pix2016,
@title:Image-to-Image Translation with Conditional Adversarial Networks
@author: Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A
@journal= arxiv
@year= 2016

@Note: Most parts of this code was written in the original implementation of Pix2Pix.
       We, Sketch2Fashion team only made a small adjustment in resize() and
       wrote canny_edge function.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import tempfile
import subprocess
import tensorflow as tf
import numpy as np
import tfimage as im
import threading
import time
import multiprocessing
from skimage import io
from skimage import feature
import os
edge_pool = None


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--operation", required=True, choices=["grayscale", "resize", "blank", "combine", "edges", "canny_edges"])
parser.add_argument("--workers", type=int, default=1, help="number of workers")
# resize
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
# combine
parser.add_argument("--b_dir", type=str, help="path to folder containing B images for combine operation")
a = parser.parse_args()

"""
Author: Vy Thai
Funtion: resize() take an non-square input image shaped(762,1100) and adding white padding to the left and right 
size to make it square before resizing to 256x256
@arg src: image 
@return: new resized image
"""
def resize(src):
    dst = np.pad(src, ((0, 0), (169, 169), (0,0)), 'constant', constant_values= (255,255))
    height, width, _ = dst.shape

    if height != width:
        size = max(height, width)
        oh = (size - height) // 2
        ow = (size - width) // 2
        dst = im.pad(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

    assert(dst.shape[0] == dst.shape[1])

    #Reference to the oiginal Pix2Pix code for the following section
    size, _, _ = dst.shape
    if size > a.size:
        dst = im.downscale(images=dst, size=[a.size, a.size])
    elif size < a.size:
        dst = im.upscale(images=dst, size=[a.size, a.size])
    return dst

"""
Author: Vy Thai
Funtion: canny_edge() take a folder of images to generate Canny edge sketches of them and save in output path
@arg src: image
@return: Canny-edge sketch
"""
def canny_edges(src):
#calculate average RGB values to detect if the dresses are light or dark colored.
    average = np.average(src)
    #threshold 1
    if average > 0.94 and average <= 0.96:
        sigma = 1
        canny_edge = feature.canny(src, sigma)
        canny_edge = np.invert(canny_edge)
        return canny_edge
    elif average > 0.96 and average <= 0.97:
        sigma = 0.6
        canny_edge = feature.canny(src, sigma)
        canny_edge = np.invert(canny_edge)
        return canny_edge
    elif average > 0.97:
        sigma = 0.4
        canny_edge = feature.canny(src, sigma)
        canny_edge = np.invert(canny_edge)
        return canny_edge
        
def blank(src):
    height, width, _ = src.shape
    if height != width:
        raise Exception("non-square image")

    image_size = width
    size = int(image_size * 0.3)
    offset = int(image_size / 2 - size / 2)

    dst = src
    dst[offset:offset + size,offset:offset + size,:] = np.ones([size, size, 3])
    return dst


def combine(src, src_path):
    if a.b_dir is None:
        raise Exception("missing b_dir")

    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(a.b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")
    
    # convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)


def grayscale(src):
    return im.grayscale_to_rgb(images=im.rgb_to_grayscale(images=src))


net = None
def run_caffe(src):
    # lazy load caffe and create net
    global net
    if net is None:
        # don't require caffe unless we are doing edge detection
        os.environ["GLOG_minloglevel"] = "2" # disable logging from caffe
        import caffe
        # using this requires using the docker image or assembling a bunch of dependencies
        # and then changing these hardcoded paths
        net = caffe.Net("/opt/caffe/examples/hed/deploy.prototxt", "/opt/caffe/hed_pretrained_bsds.caffemodel", caffe.TEST)
        
    net.blobs["data"].reshape(1, *src.shape)
    net.blobs["data"].data[...] = src
    net.forward()
    return net.blobs["sigmoid-fuse"].data[0][0,:,:]

    
def edges(src):
    # based on https://github.com/phillipi/pix2pix/blob/master/scripts/edges/batch_hed.py
    # and https://github.com/phillipi/pix2pix/blob/master/scripts/edges/PostprocessHED.m
    import scipy.io
    src = src * 255
    border = 128 # put a padding around images since edge detection seems to detect edge of image
    src = src[:,:,:3] # remove alpha channel if present
    src = np.pad(src, ((border, border), (border, border), (0,0)), "reflect")
    src = src[:,:,::-1]
    src -= np.array((104.00698793,116.66876762,122.67891434))
    src = src.transpose((2, 0, 1))

    # [height, width, channels] => [batch, channel, height, width]
    fuse = edge_pool.apply(run_caffe, [src])
    fuse = fuse[border:-border, border:-border]

    with tempfile.NamedTemporaryFile(suffix=".png") as png_file, tempfile.NamedTemporaryFile(suffix=".mat") as mat_file:
        scipy.io.savemat(mat_file.name, {"input": fuse})
        
        octave_code = r"""
E = 1-load(input_path).input;
E = imresize(E, [image_width,image_width]);
E = 1 - E;
E = single(E);
[Ox, Oy] = gradient(convTri(E, 4), 1);
[Oxx, ~] = gradient(Ox, 1);
[Oxy, Oyy] = gradient(Oy, 1);
O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
E = edgesNmsMex(E, O, 1, 5, 1.01, 1);
E = double(E >= max(eps, threshold));
E = bwmorph(E, 'thin', inf);
E = bwareaopen(E, small_edge);
E = 1 - E;
E = uint8(E * 255);
imwrite(E, output_path);
"""

        config = dict(
            input_path="'%s'" % mat_file.name,
            output_path="'%s'" % png_file.name,
            image_width=256,
            threshold=25.0/255.0,
            small_edge=5,
        )

        args = ["octave"]
        for k, v in config.items():
            args.extend(["--eval", "%s=%s;" % (k, v)])

        args.extend(["--eval", octave_code])
        try:
            subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("octave failed")
            print("returncode:", e.returncode)
            print("output:", e.output)
            raise
        return im.load(png_file.name)


def process(src_path, dst_path):
    src = im.load(src_path)

    if a.operation == "grayscale":
        dst = grayscale(src)
    elif a.operation == "resize":
        dst = resize(src)
    elif a.operation == "blank":
        dst = blank(src)
    elif a.operation == "combine":
        dst = combine(src, src_path)
    elif a.operation == "edges":
        dst = edges(src)
    elif a.operation == "canny_edge":
        dst = canny_edges(src)
    else:
        raise Exception("invalid operation")

    im.save(dst, dst_path)


complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in im.find(a.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)
    
    print("skipping %d files that already exist" % skipped)
            
    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()
    
    if a.operation == "edges":
        # use a multiprocessing pool for this operation so it can use multiple CPUs
        # create the pool before we launch processing threads
        global edge_pool
        edge_pool = multiprocessing.Pool(a.workers)

    if a.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path)
                complete()
    else:
        queue = tf.train.input_producer(zip(src_paths, dst_paths), shuffle=False, num_epochs=1)
        dequeue_op = queue.dequeue()

        def worker(coord):
            with sess.as_default():
                while not coord.should_stop():
                    try:
                        src_path, dst_path = sess.run(dequeue_op)
                    except tf.errors.OutOfRangeError:
                        coord.request_stop()
                        break

                    process(src_path, dst_path)
                    complete()

        # init epoch counter for the queue
        local_init_op = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(local_init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(a.workers):
                t = threading.Thread(target=worker, args=(coord,))
                t.start()
                threads.append(t)
            
            try:
                coord.join(threads)
            except KeyboardInterrupt:
                coord.request_stop()
                coord.join(threads)

main()

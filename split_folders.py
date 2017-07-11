import math
import os

from data.image_folder import make_dataset
from util import util

a_path = "/home/local2/yhasson/first-person-action-recognition/data/blender-renders/annots/rgb"
b_path = "/home/local2/yhasson/datasets/gun_all"
cycle_dataset_path = "/home/local2/yhasson/pytorch-CycleGAN-and-pix2pix/datasets/hands-test/"

# Create subdirectories
dest_names = ['trainA', 'valA', 'testA', 'trainB', 'valB', 'testB']
dest_paths = [os.path.join(cycle_dataset_path, dest_name)
              for dest_name in dest_names]
for dest_path in dest_paths:
    util.mkdir(dest_path)


def _symlink_files(imgs, dest_path):
    """
    :param imgs: list of original image paths
    :param dest_path: destination folder path
    """
    for img in imgs:
        img_name = img.split('/')[-1]
        dest_img = os.path.join(dest_path, img_name)
        util.symlink(img, dest_img)


def create_cycle_dataset(root, dest_path, train_prop=0.9, val_prop=0.05,
                         data_prefix="a", file_ext='.png'):
    imgs = make_dataset(root)
    imgs = [img for img in imgs if img.endswith(file_ext)]
    idx_end_train = math.floor(train_prop * len(imgs))
    idx_end_val = math.floor((train_prop + val_prop) * len(imgs))

    print('{0} training samples, {1} valid samples, {2} test samples'.format(idx_end_train,
                                                                             idx_end_val - idx_end_train,
                                                                             len(imgs) - idx_end_val))

    # Symlink training files
    train_imgs = imgs[0:idx_end_train]
    if data_prefix == "a":
        dest_path = dest_paths[0]
    else:
        dest_path = dest_paths[3]
    _symlink_files(train_imgs, dest_path)

    # Symlink validation files
    valid_imgs = imgs[idx_end_train:idx_end_val]
    if data_prefix == "a":
        dest_path = dest_paths[1]
    else:
        dest_path = dest_paths[4]
    _symlink_files(valid_imgs, dest_path)

    # Symlink test files
    test_imgs = imgs[idx_end_val:len(imgs)]
    if data_prefix == "a":
        dest_path = dest_paths[2]
    else:
        dest_path = dest_paths[5]
    _symlink_files(test_imgs, dest_path)

create_cycle_dataset(a_path, dest_paths, data_prefix="a")
create_cycle_dataset(b_path, dest_paths, data_prefix="b", file_ext='.jpg')

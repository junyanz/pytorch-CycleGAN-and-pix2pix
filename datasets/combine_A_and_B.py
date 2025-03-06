import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool

def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1)
    im_B = cv2.imread(path_B, 1)
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)

def process_images_multiprocessing(fold_A, fold_B, fold_AB, num_imgs, use_AB):
    splits = os.listdir(fold_A)
    pool = Pool()
    
    for sp in splits:
        img_fold_A = os.path.join(fold_A, sp)
        img_fold_B = os.path.join(fold_B, sp)
        img_list = os.listdir(img_fold_A)
        if use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]
        
        num_imgs = min(num_imgs, len(img_list))
        img_fold_AB = os.path.join(fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        
        for n in range(num_imgs):
            name_A = img_list[n]
            path_A = os.path.join(img_fold_A, name_A)
            name_B = name_A.replace('_A.', '_B.') if use_AB else name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A.replace('_A.', '.') if use_AB else name_A
                path_AB = os.path.join(img_fold_AB, name_AB)
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
    
    pool.close()
    pool.join()

def process_images_single(fold_A, fold_B, fold_AB, num_imgs, use_AB):
    splits = os.listdir(fold_A)
    
    for sp in splits:
        img_fold_A = os.path.join(fold_A, sp)
        img_fold_B = os.path.join(fold_B, sp)
        img_list = os.listdir(img_fold_A)
        if use_AB:
            img_list = [img_path for img_path in img_list if '_A.' in img_path]
        
        num_imgs = min(num_imgs, len(img_list))
        img_fold_AB = os.path.join(fold_AB, sp)
        if not os.path.isdir(img_fold_AB):
            os.makedirs(img_fold_AB)
        
        for n in range(num_imgs):
            name_A = img_list[n]
            path_A = os.path.join(img_fold_A, name_A)
            name_B = name_A.replace('_A.', '_B.') if use_AB else name_A
            path_B = os.path.join(img_fold_B, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_AB = name_A.replace('_A.', '.') if use_AB else name_A
                path_AB = os.path.join(img_fold_AB, name_AB)
                im_A = cv2.imread(path_A, 1)
                im_B = cv2.imread(path_B, 1)
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--fold_A', type=str, required=True, help='Input directory for image A')
    parser.add_argument('--fold_B', type=str, required=True, help='Input directory for image B')
    parser.add_argument('--fold_AB', type=str, required=True, help='Output directory')
    parser.add_argument('--num_imgs', type=int, default=1000000, help='Number of images')
    parser.add_argument('--use_AB', action='store_true', help='(0001_A, 0001_B) to (0001_AB)')
    parser.add_argument('--no_multiprocessing', action='store_true', help='Use single CPU execution')
    args = parser.parse_args()

    if args.no_multiprocessing:
        process_images_single(args.fold_A, args.fold_B, args.fold_AB, args.num_imgs, args.use_AB)
    else:
        process_images_multiprocessing(args.fold_A, args.fold_B, args.fold_AB, args.num_imgs, args.use_AB)

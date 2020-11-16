from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help="path to folder containing images")
parser.add_argument("--train_frac", type=float, default=0.8, help="percentage of images to use for training set")
parser.add_argument("--test_frac", type=float, default=0.0, help="percentage of images to use for test set")
parser.add_argument("--sort", action="store_true", help="if set, sort the images instead of shuffling them")
a = parser.parse_args()


def main():
    random.seed(0)

    files = glob.glob(os.path.join(a.dir, "*.png"))
    files.sort()

    assignments = []
    assignments.extend(["train"] * int(a.train_frac * len(files)))
    assignments.extend(["test"] * int(a.test_frac * len(files)))
    assignments.extend(["val"] * int(len(files) - len(assignments)))

    if not a.sort:
        random.shuffle(assignments)

    for name in ["train", "val", "test"]:
        if name in assignments:
            d = os.path.join(a.dir, name)
            if not os.path.exists(d):
                os.makedirs(d)

    print(len(files), len(assignments))
    for inpath, assignment in zip(files, assignments):
        outpath = os.path.join(a.dir, assignment, os.path.basename(inpath))
        print(inpath, "->", outpath)
        os.rename(inpath, outpath)

main()

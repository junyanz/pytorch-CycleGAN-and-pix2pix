#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          #
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, #
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

from os import walk, path, makedirs
from shutil import copy2 as cp


def get_files(base_dir):
    # Minmizing (halfing) list to just pre image files
    base_dir = path.join(base_dir, "images")
    files = [f for f in next(walk(base_dir))[2] if "pre" in f]

    return files


def move_files(files, base_dir, output_dir):
    for filename in files:
        disaster = filename.split("_")[0]

        # If the output directory and disater name do not exist make the directory
        if not path.isdir(path.join(output_dir, disaster)):
            makedirs(path.join(output_dir, disaster))

        # Check if the images directory exists
        if not path.isdir(path.join(output_dir, disaster, "images")):
            # If not create it
            makedirs(path.join(output_dir, disaster, "images"))

        # Move the pre and post image to the images directory under the disaster name
        cp(
            path.join(base_dir, "images", filename),
            path.join(output_dir, disaster, "images", filename),
        )
        post_file = filename.replace("_pre_", "_post_")
        cp(
            path.join(base_dir, "images", post_file),
            path.join(output_dir, disaster, "images", post_file),
        )

        # Check if the label directory exists
        if not path.isdir(path.join(output_dir, disaster, "labels")):
            # If not create it
            makedirs(path.join(output_dir, disaster, "labels"))

        pre_label_file = filename.replace("png", "json")
        # Move the pre and post label files to the labels directory under the disaster name
        cp(
            path.join(base_dir, "labels", pre_label_file),
            path.join(output_dir, disaster, "labels", pre_label_file),
        )
        post_label_file = pre_label_file.replace("_pre_", "_post_")
        cp(
            path.join(base_dir, "labels", post_label_file),
            path.join(output_dir, disaster, "labels", post_label_file),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="split_into_disasters.py: Splits files under a single directory (with images/ and labels/ into directory of disasters/images|labels for the base submission pipeline (copies files)"
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="/path/to/dataset/train",
        help="Full path to the train (or any other directory) with /images and /labels",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="/path/to/output/xBD",
        help="Full path to the output root dataset directory, will create disaster/images|labels under this directory",
    )

    args = parser.parse_args()

    files = get_files(args.input)
    print("Start splitting into disasters")
    move_files(files, args.input, args.output)
    print("Finished splitting into disasters")

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import os
import sys
import time
import shutil
import shlex

INPUT_DIR = os.path.abspath("../data")
OUTPUT_DIR = os.path.expanduser("~/data/pix2pix/test")


def main():
    start = time.time()

    images = {
        "affinelayer": "affinelayer/pix2pix-tensorflow:v3",
        # "py2-tensorflow": "tensorflow/tensorflow:1.4.1-gpu",
        # "py3-tensorflow": "tensorflow/tensorflow:1.4.1-gpu-py3",
    }

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for image_name, image in images.items():
        def run(cmd):
            docker = "docker"
            if sys.platform.startswith("linux"):
                docker = "nvidia-docker"

            prefix = [docker, "run", "--rm", "--volume", os.getcwd() + ":/prj", "--volume", INPUT_DIR + ":/input", "--volume", os.path.join(OUTPUT_DIR, image_name) + ":/output","--workdir", "/prj", "--env", "PYTHONUNBUFFERED=x", "--volume", "/tmp/cuda-cache:/cuda-cache", "--env", "CUDA_CACHE_PATH=/cuda-cache", image]
            args = prefix + shlex.split(cmd)
            print(" ".join(args))
            subprocess.check_call(args)

        run("python tools/process.py --input_dir /input/pusheen/original --operation resize --output_dir /output/process_resize")
        if image_name == "affinelayer":
            run("python tools/process.py --input_dir /output/process_resize --operation edges --output_dir /output/process_edges")

        for direction in ["AtoB", "BtoA"]:
            for dataset in ["facades", "maps"]:
                name = dataset + "_" + direction
                run("python pix2pix.py --mode train --input_dir /input/official/%s/train --output_dir /output/%s_train --display_freq 1 --max_steps 1 --which_direction %s --seed 0" % (dataset, name, direction))
                run("python pix2pix.py --mode test --input_dir /input/official/%s/val --output_dir /output/%s_test --display_freq 1 --max_steps 1 --checkpoint /output/%s_train --seed 0" % (dataset, name, name))

            dataset = "color-lab"
            name = dataset + "_" + direction
            run("python pix2pix.py --mode train --input_dir /input/%s/train --output_dir /output/%s_train --display_freq 1 --max_steps 1 --which_direction %s --lab_colorization --seed 0" % (dataset, name, direction))
            run("python pix2pix.py --mode test --input_dir /input/%s/val --output_dir /output/%s_test --display_freq 1 --max_steps 1 --checkpoint /output/%s_train --seed 0" % (dataset, name, name))

        # using pretrained model
        # for dataset, direction in [("facades", "BtoA")]:
        #     name = dataset + "_" + direction
        #     run("python pix2pix.py --mode test --output_dir test/%s_pretrained_test --input_dir /input/official/%s/val --max_steps 100 --which_direction %s --seed 0 --checkpoint /input/pretrained/%s" % (name, dataset, direction, name))
        #     run("python pix2pix.py --mode export --output_dir test/%s_pretrained_export --checkpoint /input/pretrained/%s" % (name, name))

    print("elapsed", int(time.time() - start))


main()

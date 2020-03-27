import re
from collections import OrderedDict
from random import shuffle
from os import mkdir
from shutil import copyfile
import pathlib
import os
import random
import argparse


def create_label_entry(label, split, name, labels_output):
    with open(labels_output, "a") as myFile:
        for image_name in split:
            new_filename = re.sub('.png','_'+str(label)+'.png',image_name)
            new_filename = './datasets/satellite_AB_imagetable/AB/test/' + new_filename
            #re.sub('./data/satellite_AB/', './datasets/satellite_AB_generated/', new_filename)
            print('new filename', new_filename)
            #print(new_filename + ' ' + str(label) + '\n')
            myFile.write(new_filename + ' ' + str(label) + '\n')

def copy_file_into_A(label, split, name, output_path):
    i = 0
    for image_name in split:
        filename = re.sub('./data/satellite_AB/AB/train/', '', image_name)
        new_filename = re.sub('.png','_'+str(label)+'.png',filename)
        new_path = os.path.join(output_path, new_filename)
        old_path = os.path.join(path_to_ABtrain, filename)
        try:
            pass
            print('copy from ', old_path, ' to ', new_path)
            copyfile(old_path, new_path)
        except:
            i += 1
            #print('didnt copy: old path', old_path, 'new_path', new_path)
    print('Didnt copy ', i, ' of label ', label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot Stat Logs')
    parser.add_argument('--path_to_ABtrain',
                        required=False,
                        default='datasets/satellite_AB/AB/train',
                        help="Relative path to the AB/train zeros")
    parser.add_argument('--rate',
                        required=True,
                        type=int,
                        help="Rate at which to use zeros: 1: use every zero, 2: use every second zero, 3: use every third zero")
    parser.add_argument('--original_labels_file',
                        required=True,
                        type=str,
                        help="Relative path to original labels file")
    parser.add_argument('--output_path',
                        required=True,
                        type=str,
                        help="Relative path to output")
    parser.add_argument('--labels_output',
                        required=True,
                        type=str,
                        help="Relative path to labels_output")

    args = parser.parse_args()
    path_to_ABtrain = args.path_to_ABtrain
    rate = args.rate
    name = 'inputtable'
    original_labels_file = args.original_labels_file
    output_path = args.output_path
    labels_output = args.labels_output

    image_names = ['005b347d-54f3-4c3d-99af-2b0be4b9e0c2.png',
                   '000219bd-5e7a-4407-9486-55860990b504.png',
                   '0040469a-2df4-4320-8d35-a1c5eee41769.png',
                   'fefcb9bf-57ba-4e32-ac50-c5654e19a6e6.png',
                   'ff3d1306-2f37-47ea-b4df-09972d6a672b.png']

    labels = range(0,24)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    for label in labels:
        create_label_entry(label, image_names, name, labels_output)
        copy_file_into_A(label, image_names, name, output_path)




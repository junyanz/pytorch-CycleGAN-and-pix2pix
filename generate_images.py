import re
from collections import OrderedDict
from random import shuffle
from os import mkdir
from shutil import copyfile
import pathlib
import os
import random
import argparse


random.seed(483745342)
labels_dict = {}

disaster_type_names = {
    0: 'flooding-0',
    1: 'flooding-1',
    2: 'flooding-2',
    3: 'flooding-3',
    4: 'fire-0',
    5: 'fire-1',
    6: 'fire-2',
    7: 'fire-3',
    8: 'wind-0',
    9: 'wind-1',
    10: 'wind-2',
    11: 'wind-3',
    12: 'tsunami-0',
    13: 'tsunami-1',
    14: 'tsunami-2',
    15: 'tsunami-3',
    16: 'earthquake-0',
    17: 'earthquake-1',
    18: 'earthquake-2',
    19: 'earthquake-3',
    20: 'volcano-0',
    21: 'volcano-1',
    22: 'volcano-2',
    23: 'volcano-3'
}

disaster_type_mapping = {
    'flooding': [1, 2, 3],
    'fire': [5, 6, 7],
    'wind': [9, 10, 11],
    'tsunami': [13, 14, 15],
    'earthquake': [17, 18, 19],
    'volcano': [21, 22, 23]
}

zero_labels = {
    'flooding': 0,
    'fire': 4,
    'wind': 8,
    'tsunami': 12,
    'earthquake': 16,
    'volcano': 20
}

disaster_type_mapping_incl0 = {
    'flooding': [0, 1, 2, 3],
    'fire': [4, 5, 6, 7],
    'wind': [8, 9, 10, 11],
    'tsunami': [12, 13, 14, 15],
    'earthquake': [16, 17, 18, 19],
    'volcano': [20, 21, 22, 23]
}

disaster_levels = {
    0: [0, 4, 8, 12, 16, 20],
    1: [1, 5, 9, 13, 17, 21],
    2: [2, 6, 10, 14, 18, 22],
    3: [3, 7, 11, 15, 19, 23],
}

with open('datasets/satellite_AB_labels.txt', 'r') as fp:
    labels = fp.read()
    labels = re.split('\n', labels)
    for line in labels:
        if len(line)>0:
            line_split = re.split(' ', line)
            path = line_split[0]
            label = line_split[1]
            if int(label) in labels_dict:
                labels_dict[int(label)].append(path)
            else:
                labels_dict[int(label)] = [path]

labels_dict_full = labels_dict

#Show distribution of labels:
dict_levels = {}

for level, labels in disaster_levels.items():
    length_train = 0
    length_test = 0
    length_val = 0
    length_total = 0
    for label in labels:
        for element in labels_dict_full[label]:
            if '/train/' in element:
                length_train += 1
            if '/test/' in element:
                length_val += 1
            if '/val/' in element:
                length_test += 1
            length_total += 1
    dict_levels[level] = [length_total, length_train, length_val, length_test]

print(dict_levels)

def list_zeros_to_be_used(rate, disaster_type_mapping, labels_dict_full):
    zero_list = []
    for key in disaster_type_mapping:
        zero_list.extend(list(labels_dict_full[zero_labels[key]]))
    random.Random(483745342).shuffle(zero_list)

    #print(len(zero_list))
    zeros_in_train = []
    for zero in zero_list:
        if '/train/' in zero:
            zeros_in_train.append(zero)
    used_zeros = zeros_in_train[:int(len(zeros_in_train)/rate)]
    return used_zeros

def match_zero_to_class(splits_length, to_be_used_zeros, non_zero_classes, disaster_type_mapping, labels_dict_full, zeros_for_label):
    for label in non_zero_classes:
        zeros_for_label[label] = to_be_used_zeros[0:splits_length]
        print(len(zeros_for_label[label]), ' zeros will be used for label', label)
        to_be_used_zeros = to_be_used_zeros[splits_length:]
        print(len(to_be_used_zeros), ' zeros are left over')
    return zeros_for_label

def create_label_entry(label, split, name, textfile_name):
    with open(textfile_name, "a") as myFile:
        for image_name in split:
            new_filename = re.sub('.png','_'+str(label)+'.png',image_name)
            myFile.write(new_filename + ' ' + str(label) + '\n')

def copy_file_into_A(label, split, name, path_generated):
    i = 0
    for image_name in split:
        filename = re.sub('./datasets/satellite_AB/AB/train/', '', image_name)
        new_filename = re.sub('.png','_'+str(label)+'.png',filename)
        new_path = os.path.join(path_generated, new_filename)
        old_path = os.path.join(origin_path, filename)
        try:
            copyfile(old_path, new_path)
        except:
            i += 1
    print('Didnt copy ', i, ' of label ', label)

def non_zero_classes_fct(disaster_type_mapping):
    non_zero_classes = []
    for list_labels in disaster_type_mapping.values():
        print('disaster_type_mapping', list_labels)
        non_zero_classes.extend(list_labels)
    return non_zero_classes

def create_input(rate, name):
    #we create images for all classes that are not zero.
    #These are:
    print(disaster_type_mapping)
    non_zero_classes = non_zero_classes_fct(disaster_type_mapping)
    print(non_zero_classes)
    path_generated = './datasets/satellite_AB_generated_' + name + '/AB/test'
    pathlib.Path(path_generated).mkdir(parents=True, exist_ok=True)
    zeros_for_label = {}
    to_be_used_zeros = list_zeros_to_be_used(rate, disaster_type_mapping, labels_dict_full)
    print(len(to_be_used_zeros), ' zeros will be used')
    splits_length = int(len(to_be_used_zeros) / len(non_zero_classes))
    print('one split has length ', splits_length)
    splits_dict = match_zero_to_class(splits_length, to_be_used_zeros, non_zero_classes, disaster_type_mapping, labels_dict_full, zeros_for_label)
    print(splits_dict.keys())
    textfile_name = 'datasets/generated_labels_' + name + '.txt'
    open(textfile_name, 'w').close()
    for label, split in splits_dict.items():
        create_label_entry(label, split, name, textfile_name)
        copy_file_into_A(label, split, name, path_generated)

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

    rate_names = {
        1: 'every',
        2: 'every_second',
        3: 'every_third'
        }

    args = parser.parse_args()
    #output = args.output
    origin_path = args.path_to_ABtrain
    rate = args.rate
    name = rate_names[rate]

    create_input(rate, name)


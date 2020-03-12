import re
from collections import OrderedDict
from random import shuffle
from os import mkdir
from shutil import copyfile
import pathlib
import os

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
                #print(labels_dict[label])

#print(labels_dict)

labels_dict = OrderedDict(sorted(labels_dict.items()))

total = 0
for key in labels_dict:
    print(disaster_type_names[int(key)], key, len(labels_dict[key]))
    total += len(labels_dict[key])

for key, value in disaster_type_mapping.items():
    num_img = []
    for label in value:
        num_img.append(len(labels_dict[label]))
    max_zeros = max(num_img)
    zero_images = list(labels_dict[zero_labels[key]])
    shuffle(zero_images)
    labels_dict[zero_labels[key]] = zero_images[:max_zeros]


keeping = sum(labels_dict.values(), [])
print(len(keeping), total, len(keeping)/total)

# pathlib.Path('./datasets/satellite_AB_reduced/AB/train').mkdir(parents=True, exist_ok=True)
# pathlib.Path('./datasets/satellite_AB_reduced/AB/test').mkdir(parents=True, exist_ok=True)
# pathlib.Path('./datasets/satellite_AB_reduced/AB/val').mkdir(parents=True, exist_ok=True)

# with open('datasets/satellite_AB_labels.txt', 'r') as fp:
#     labels = fp.read()
#     labels = re.split('\n', labels)
#     for line in labels:
#         if len(line)>0:
#             line_split = re.split(' ', line)
#             path = line_split[0]
#             if path in keeping and os.path.isfile(path):
#                 path_new = re.sub('satellite_AB', 'satellite_AB_reduced', path)
#                 print(path_new)
#                 copyfile(path, path_new)



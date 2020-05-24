import os
from random import shuffle
import shutil
import argparse
from pathlib import Path


# NOT USED - but we keep it since we might need it in the future
# TODO: rewrite this function to actually use 'input dir', 'output_dir' and 'satellite_labels' variables.
# TODO: change how split is determined
def split_into_train_val_test(input_dir, output_dir, satellite_labels):
    # FIRST THIS
    mypath = './data/' + folder + '_AB/B/'
    identifier_list = [f for f in os.listdir(mypath) if os.path.isfile(mypath + f) and f.endswith('.png')]
    shuffle(identifier_list)
    print(len(identifier_list))

    with open('./data/' + folder + '_labels.txt', 'r') as f:
        lines = f.readlines()
        file_label_dict = {}

        # Create mapping from file to label
        for line in lines:
            filepath, label = line.replace('\n', '').split(' ')
            file = filepath.split('/')[-1]
            file_label_dict[file] = label

    training = identifier_list[int((0.4 * len(identifier_list))):]
    test = identifier_list[0:int((0.2 * len(identifier_list)))]
    val = identifier_list[int((0.2 * len(identifier_list))): int((0.4 * len(identifier_list)))]

    os.makedirs(mypath + '/train/', exist_ok=True)
    os.makedirs(mypath + '/test/', exist_ok=True)
    os.makedirs(mypath + '/val/', exist_ok=True)

    with open('./data/' + folder + '_AB_labels.txt', 'w') as f:
        for file in training:
            shutil.move(mypath + file, mypath + 'train/' + file)
            f.write(mypath + 'train/' + file + ' ' + str(file_label_dict[file]) + '\n')
        for file in test:
            shutil.move(mypath + file, mypath + 'test/' + file)
            f.write(mypath + 'test/' + file + ' ' + str(file_label_dict[file]) + '\n')
        for file in val:
            shutil.move(mypath + file, mypath + 'val/' + file)
            f.write(mypath + 'val/' + file + ' ' + str(file_label_dict[file]) + '\n')

    # THEN THIS
    path_train = 'data/' + folder + '_AB/B/train/'
    path_test = 'data/' + folder + '_AB/B/test/'
    path_val = 'data/' + folder + '_AB/B/val/'
    path_post = 'data/' + folder + '_AB/A/'

    os.makedirs(path_post + '/train/', exist_ok=True)
    os.makedirs(path_post + '/test/', exist_ok=True)
    os.makedirs(path_post + '/val/', exist_ok=True)

    identifier_list_train = [f for f in os.listdir(path_train) if os.path.isfile(path_train + f) and f.endswith('.png')]
    print(len(identifier_list_train))
    identifier_list_test = [f for f in os.listdir(path_test) if os.path.isfile(path_test + f) and f.endswith('.png')]
    identifier_list_val = [f for f in os.listdir(path_val) if os.path.isfile(path_val + f) and f.endswith('.png')]

    for file in identifier_list_train:
        shutil.move(path_post + file, path_post + '/train/' + file)
    for file in identifier_list_test:
        shutil.move(path_post + file, path_post + '/test/' + file)
    for file in identifier_list_val:
        shutil.move(path_post + file, path_post + '/val/' + file)


def move_to_single_category(input_dir, output_dir, satellite_labels, output_labels_file, output_category):
    Path(output_dir + '/A/' + output_category).mkdir(parents=True, exist_ok=True)
    Path(output_dir + '/B/' + output_category).mkdir(parents=True, exist_ok=True)

    with open(satellite_labels, 'r') as f:
        lines = f.readlines()
        file_label_dict = {}

        # Create mapping from file to label
        for line in lines:
            filepath, label = line.replace('\n', '').split(' ')
            file = filepath.split('/')[-1]
            file_label_dict[file] = label

            try:
                shutil.move(input_dir + '/A/' + file, output_dir + '/A/' + output_category + '/' + file)
                shutil.move(input_dir + '/B/' + file, output_dir + '/B/' + output_category + '/' + file)

                with open(output_labels_file, 'a') as f:
                    f.write(output_dir + '/B/' + output_category + '/' + file + ' ' + str(file_label_dict[file]) + '\n')
            except:
                print('Didnt move ', file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/ABdata",
                        help="Path to directory that contains pre and post folder")
    parser.add_argument('--output_dir',
                        required=True,
                        metavar='/path/to/ABtrainvaltest',
                        help="Path to new directory to save A B images")
    parser.add_argument('--satellite_labels',
                        required=True,
                        metavar='/path/to/satellite_labels',
                        help="Path to satellite labels file which describes the data")
    parser.add_argument('--output_labels_file',
                        required=True,
                        metavar='/path/to/outputlabelsfile',
                        help="Path to AB labels file")
    parser.add_argument('--single_category_output',
                        default=False,
                        required=True,
                        help="Specify if all files should go into train, test or val")
    parser.add_argument('--output_category',
                        required=False,
                        help="Specify if all files should go into train, test or val")
    args = parser.parse_args()

    # labels_folder, labels_file = args.output_labels_file.split('/')

    print('Started split_train_val')
    # This option is not used for now - split_into_train_val_test will need refactoring
    if args.single_category_output == False:
        print('Randomly splitting the data')
        split_into_train_val_test(args.input_dir, args.output_dir, args.satellite_labels)
    else:
        print('Move all files into', args.output_category)
        move_to_single_category(args.input_dir, args.output_dir, args.satellite_labels, args.output_labels_file, args.output_category)
    print('Finished split_train_val')

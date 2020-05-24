import shutil
from pathlib import Path
import argparse
import os

disaster_type_mapping = {
    'flooding-0': 0,
    'flooding-1': 1,
    'flooding-2': 2,
    'flooding-3': 3,
    'fire-0': 4,
    'fire-1': 5,
    'fire-2': 6,
    'fire-3': 7,
    'wind-0': 8,
    'wind-1': 9,
    'wind-2': 10,
    'wind-3': 11,
    'tsunami-0': 12,
    'tsunami-1': 13,
    'tsunami-2': 14,
    'tsunami-3': 15,
    'earthquake-0': 16,
    'earthquake-1': 17,
    'earthquake-2': 18,
    'earthquake-3': 19,
    'volcano-0': 20,
    'volcano-1': 21,
    'volcano-2': 22,
    'volcano-3': 23
}


def create_labels(input_dir, output_dir, output_dir_csv, train_csv):
    Path(output_dir + '/A/').mkdir(parents=True, exist_ok=True)
    Path(output_dir + '/B/').mkdir(parents=True, exist_ok=True)
    Path(output_dir_csv).mkdir(parents=True, exist_ok=True)

    with open(train_csv, 'r') as f:
        lines = f.readlines()

        # Write filenames and labels to textfile
        with open(os.path.join(output_dir_csv, 'satellite_labels.txt'), 'w') as labels_file:
            for line in lines[1:]:
                idx, image, label, disaster_type = line.split(',')
                image = image.replace('\n', '')
                img_filename, img_suffix = image.split('_')
                disaster_type = disaster_type.replace('\n', '')

                disaster_label = str(disaster_type_mapping[disaster_type + '-' + label])

                # Split images into folders A and B
                if img_suffix.startswith('pre'):
                    src_path = os.path.join(input_dir, 'pre', image)
                    dest_path = output_dir + '/A/' + img_filename + '.png'
                    shutil.copy(src_path, dest_path)
                else:
                    src_path = os.path.join(input_dir, 'post', image)
                    dest_path = output_dir + '/B/' + img_filename + '.png'
                    shutil.copy(src_path, dest_path)
                    labels_file.write(dest_path + ' ' + disaster_label + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/prepostdata",
                        help="Path to directory that contains pre and post folder")
    parser.add_argument('--output_dir',
                        required=True,
                        metavar='/path/to/ABdataoutput',
                        help="Path to new directory to save A B images")
    parser.add_argument('--train_csv',
                        required=True,
                        metavar='/path/to/train_csv',
                        help="Path to train csv")
    parser.add_argument('--output_dir_csv',
                        required=True,
                        metavar='/path/to/xBD_output_csv',
                        help="Path to new directory to save csv")

    args = parser.parse_args()

    print('Started created_satellite_labels')
    create_labels(args.input_dir, args.output_dir, args.output_dir_csv, args.train_csv)
    print('Finished created_satellite_labels')

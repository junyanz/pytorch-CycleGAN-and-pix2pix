import numpy as np
import pandas as pd
import os
import argparse
import logging
import json
import cv2
import shapely.wkt
import shapely

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Configurations
NUM_WORKERS = 4
NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 0.0001
RANDOM_SEED = 123
LOG_STEP = 150

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into
            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.
            Returns:
                numpy array: .
    """

    height, width, _ = img_array.shape

    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    # Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_data(input_path, output_path, output_csv_path, val_split_pct):
    """Process Raw Data into
        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.
        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form
    """
    x_data = []
    y_data = []
    disaster_type = []

    disasters = [folder for folder in os.listdir(input_path) if not folder.startswith('.')]
    disaster_paths = ([input_path + '/' + d + '/images' for d in disasters])
    image_paths = []
    image_paths.extend([(disaster_path + "/" + pic) for pic in os.listdir(disaster_path)] for disaster_path in disaster_paths)
    img_paths = np.concatenate(image_paths)

    for img_path in tqdm(img_paths):
        if '.png' in img_path:
            img_obj = Image.open(img_path)
            img_array = np.array(img_obj)

            # Get corresponding label for the current image
            label_path = img_path.replace('png', 'json').replace('/images/', '/labels/')
            label_file = open(label_path)
            label_data = json.load(label_file)
            image_name = label_data['metadata']['img_name']
            # image_identifier = re.sub('_pre_disaster.png', '', re.sub('_post_disaster.png','', image_name))
            if 'pre' in image_name:
                prepost = 'pre'
            elif 'post' in image_name:
                prepost = 'post'

            for feat in label_data['features']['xy']:
                # only images post-disaster will have damage type
                try:
                    damage_type = feat['properties']['subtype']
                except:  # pre-disaster damage is default no-damage
                    damage_type = "no-damage"
                    # continue

                poly_uuid = feat['properties']['uid'] + '_' + prepost + ".png"

                polygon_geom = shapely.wkt.loads(feat['wkt'])
                polygon_pts = np.array(list(polygon_geom.exterior.coords))
                try:
                    poly_img = process_img(img_array, polygon_pts, 0.8)
                    Path(os.path.join(output_path, prepost)).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(output_path + '/' + prepost + '/' + poly_uuid, poly_img)
                    x_data.append(poly_uuid)
                    y_data.append(damage_intensity_encoding[damage_type])
                    disaster_type.append(label_data['metadata']['disaster_type'])
                except:
                    pass

    Path(output_csv_path).mkdir(parents=True, exist_ok=True)
    output_train_csv_path = os.path.join(output_csv_path, "train.csv")

    if(val_split_pct > 0):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=val_split_pct)
        data_array_train = {'uuid': x_train, 'labels': y_train}
        data_array_test = {'uuid': x_test, 'labels': y_test}
        output_test_csv_path = os.path.join(output_csv_path, "test.csv")
        df_train = pd.DataFrame(data_array_train)
        df_test = pd.DataFrame(data_array_test)
        df_train.to_csv(output_train_csv_path)
        df_test.to_csv(output_test_csv_path)
    else:
        data_array = {'uuid': x_data, 'labels': y_data, 'disaster_type': disaster_type}
        df = pd.DataFrame(data=data_array)
        df = df[['uuid', 'labels', 'disaster_type']]
        df.to_csv(output_train_csv_path)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/xBD_input",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--output_dir',
                        required=True,
                        metavar='/path/to/xBD_output',
                        help="Path to new directory to save images")
    parser.add_argument('--output_dir_csv',
                        required=True,
                        metavar='/path/to/xBD_output_csv',
                        help="Path to new directory to save csv")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.0,
                        metavar='Percentage to split validation',
                        help="Percentage to split ")
    args = parser.parse_args()

    logging.info("Started process_data_xbd")
    process_data(args.input_dir, args.output_dir, args.output_dir_csv, float(args.val_split_pct))
    logging.info("Finished process_data_xbd")


if __name__ == '__main__':
    main()

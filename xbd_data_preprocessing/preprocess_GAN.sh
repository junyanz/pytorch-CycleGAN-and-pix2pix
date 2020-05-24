#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p path_to_dataset -f input_folder_name -c output_category"
   echo -e "\t-p: The path to the train/val/test xBD dataset eg. ~/Downloads/test_images_labels_targets"
   echo -e "\t-f: The name of the folder inside path_to_dataset, usually one of train/val/test"
   echo -e "\t-c: One of train/val/test"
   exit 1 # Exit script after printing help
}

while getopts "p:f:c:" opt
do
   case "$opt" in
      p ) path_to_dataset="$OPTARG" ;;
      f ) input_folder_name="$OPTARG" ;;
      c ) output_category="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$path_to_dataset" ] || [ -z "$input_folder_name" ] || [ -z "$output_category" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$path_to_dataset"
echo "$input_folder_name"
echo "$output_category"

XBD_DISASTER_SPLIT=$path_to_dataset/xBD_disaster_split
XBD_POLYGONS=$path_to_dataset/xBD_polygons
XBD_POLYGONS_CSV=$path_to_dataset/xBD_polygons_csv
XBD_POLYGONS_AB=$path_to_dataset/xBD_polygons_AB
XBD_POLYGONS_AB_CSV=$path_to_dataset/xBD_polygons_AB_csv
XBD_POLYGONS_SPLIT=$path_to_dataset/xBD_polygons_split

set -e

python split_into_disasters.py \
--input $path_to_dataset/$input_folder_name \
--output $XBD_DISASTER_SPLIT

python process_data_xbd.py \
--input_dir $XBD_DISASTER_SPLIT \
--output_dir $XBD_POLYGONS \
--output_dir_csv $XBD_POLYGONS_CSV

rm -r $XBD_DISASTER_SPLIT

python create_satellite_labels.py \
--input_dir $XBD_POLYGONS \
--output_dir $XBD_POLYGONS_SPLIT \
--output_dir_csv $XBD_POLYGONS_AB_CSV \
--train_csv $XBD_POLYGONS_CSV/train.csv

rm -r $XBD_POLYGONS
rm -r $XBD_POLYGONS_CSV

python split_to_AB.py \
--input_dir $XBD_POLYGONS_SPLIT \
--output_dir $XBD_POLYGONS_AB \
--satellite_labels $XBD_POLYGONS_AB_CSV/satellite_labels.txt \
--single_category_output True \
--output_category $output_category \
--output_labels_file $XBD_POLYGONS_AB_CSV/satellite_AB_labels.txt

rm -r $XBD_POLYGONS_SPLIT
rm -r $XBD_POLYGONS_AB_CSV/satellite_labels.txt

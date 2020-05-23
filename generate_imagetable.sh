source .env/bin/activate
python gantablepreparation.py \
--rate 3 \
--path_to_ABtrain datasets/satellite_AB/AB/train \
--original_labels_file datasets/satellite_AB_labels.txt \
--output_path datasets/satellite_AB_imagetable/AB/test \
--labels_output datasets/imagetable_labels.txt

sed -i 's/train/test/g' datasets/imagetable_labels.txt

set -ex
python test.py \
--dataroot ./datasets/satellite_AB_imagetable/AB \
--name trained_GAN \
--model pix2pix \
--netG unet_128 \
--direction AtoB \
--dataset_mode aligned \
--norm batch \
--load_size 142 \
--crop_size 128 \
--input_nc 3 \
--output_nc 3 \
--labels_file ./datasets/imagetable_labels.txt \
--eval

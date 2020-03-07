set -ex
python test.py --dataroot ./datasets/mnist0123/AB \
--name mnist0123_pix2pix_bs8 \
--model pix2pix \
--netG unet_128 \
--direction AtoB \
--dataset_mode aligned \
--norm batch \
--load_size 142 \
--crop_size 128 \
--input_nc 3 \
--output_nc 3 \
--num_test 300 \

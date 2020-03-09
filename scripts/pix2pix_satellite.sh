set -ex
# python train.py --dataroot ./datasets/mnist0123/AB --name mnist0123_pix2pix_bs8 --model pix2pix --direction AtoB --input_nc 3 --output_nc 3 --batch_size 8 --n_epochs 100 --n_epochs_decay 100 --model pix2pix --no_flip --load_size 84 --crop_size 64
python train.py --dataroot ./datasets/satellite/AB --name satellite_pix2pix_bs8 --model pix2pix --direction AtoB --input_nc 3 --output_nc 3 --batch_size 8 --n_epochs 100 --n_epochs_decay 100 --model pix2pix --no_flip --netG unet_128 --load_size 142 --crop_size 128

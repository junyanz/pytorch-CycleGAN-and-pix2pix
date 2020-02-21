set -ex
python train.py --dataroot ./datasets/mnist_4channel/AB --name mnist_4channel_pix2pix_bs8 --model pix2pix --netG unet_128 --direction AtoB --input_nc 4 --output_nc 4 --batch_size 8 --n_epochs 100 --n_epochs_decay 100

#--preprocess none

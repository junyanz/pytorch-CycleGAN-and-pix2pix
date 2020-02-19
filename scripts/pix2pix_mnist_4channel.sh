set -ex
python test.py --dataroot ./datasets/mnist_4channel/AB --name mnist_4channel_pix2pix_bs8 --model pix2pix --direction AtoB

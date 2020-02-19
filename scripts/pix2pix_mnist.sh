set -ex
python test.py --dataroot ./datasets/mnist150/AB --name mnist150_pix2pix_bs8 --model pix2pix --direction AtoB

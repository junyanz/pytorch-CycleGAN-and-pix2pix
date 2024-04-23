set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout

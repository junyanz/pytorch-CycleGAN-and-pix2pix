set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0

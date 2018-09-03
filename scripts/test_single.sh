set -ex
python test.py --dataroot ./datasets/facades/testB/ --name facades_pix2pix --model test --netG unet_256 --which_direction BtoA --dataset_mode single --norm batch

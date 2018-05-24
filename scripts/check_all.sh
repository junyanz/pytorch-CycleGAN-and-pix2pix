set -ex
DOWNLOAD=${1}
echo 'apply a pretrained cyclegan model'
if [ ${DOWNLOAD} -eq 1 ]
then
  bash pretrained_models/download_cyclegan_model.sh horse2zebra
  bash ./datasets/download_cyclegan_dataset.sh horse2zebra
fi
python test.py --dataroot datasets/horse2zebra/testA --checkpoints_dir ./checkpoints/ --name horse2zebra_pretrained --no_dropout --model test --dataset_mode single --loadSize 256

echo 'apply a pretrained pix2pix model'
if [ ${DOWNLOAD} -eq 1 ]
then
  bash pretrained_models/download_pix2pix_model.sh facades_label2photo
  bash ./datasets/download_pix2pix_dataset.sh facades
fi
python test.py --dataroot ./datasets/facades/ --which_direction BtoA --model pix2pix --name facades_label2photo_pretrained --dataset_mode aligned --which_model_netG unet_256 --norm batch


echo 'cyclegan train (1 epoch) and test'
if [ ${DOWNLOAD} -eq 1 ]
then
  bash ./datasets/download_cyclegan_dataset.sh maps
fi
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout --niter 1 --niter_decay 0 --max_dataset_size 100 --save_latest_freq 100
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout


echo 'pix2pix train (1 epoch) and test'
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 1 --niter_decay 0 --save_latest_freq 400
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch

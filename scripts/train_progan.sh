set -ex
python train.py --dataroot ./datasets/facades --name progan_train --model progan --pool_size 50  --dataset_mode single --batch_size 1  --crop_size 256 --load_size 300 --preprocess resize_and_crop

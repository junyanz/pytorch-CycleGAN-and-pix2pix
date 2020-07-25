set -ex
python train.py --dataroot ./datasets/facades --name progan_facades --model progan --pool_size 10  --dataset_mode single --batch_size 1 --crop_size 256 --preprocess crop

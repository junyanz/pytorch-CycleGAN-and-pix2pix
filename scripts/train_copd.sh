#!/bin/bash
source activate venv
python train3d.py \
    --dataroot /ocean/projects/asc170022p/rohit33 \
    --dataset_mode copd2class \
    --gpu_ids 0 \
    --sampler copd2class \
    --batch_size 512 \
    --lambda_identity 0 \
    --netG unet3d_32 \
    --netD basic3d \
    --input_nc 1 --output_nc 1 \
    --norm batch3d \
    --name copd

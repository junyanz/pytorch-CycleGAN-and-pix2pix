#!/bin/bash
#source activate venv
source ~/.bashrc
conda activate venv
python train3d.py \
    --dataroot /ocean/projects/asc170022p/rohit33 \
    --dataset_mode copdpatchlabels \
    --gpu_ids 0 \
    --model cycle_gan_patch \
    --batch_size 500 \
    --save_latest_freq 40000 \
    --lambda_identity 0.05 \
    --display_freq 20 \
    --netG unet3d_patch \
    --netD basic3d_patch \
    --input_nc 1  \
    --output_nc 1 \
    --norm batch3d \
    --subroot emphysemapatches \
    --name copd_emphysema

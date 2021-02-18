#!/bin/bash
#source activate venv
source ~/.bashrc
conda activate venv
python train3d.py \
    --dataroot /ocean/projects/asc170022p/rohit33 \
    --dataset_mode copdpatchlabels \
    --gpu_ids 0,1 \
    --model cycle_gan_patch \
    --patchfloat 1 \
    --augment 1 \
    --batch_size 100 \
    --save_latest_freq 40 \
    --lambda_identity 0.2 \
    --display_freq 40 \
    --netG resnet3d_9blocks \
    --netD patchgan_3d \
    --input_nc 1  \
    --output_nc 1 \
    --norm batch3d \
    --subroot emphysemapatches \
    --name copd_emphysema_resnet9

#--name copd_emphysema_float
#--netD basic3d_patch_float \
#--netG unet3d_patch_float \

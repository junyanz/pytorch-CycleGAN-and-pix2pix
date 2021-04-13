#!/bin/bash
#source activate venv
# Batch size 20 works for 2 GPUs
# This script is for a smaller batch size

source ~/.bashrc
conda activate venv
python train.py \
    --dataroot /ocean/projects/asc170022p/rohit33/COPDslices/ \
    --dataset_mode copdslice2class \
    --gpu_ids 0,1 \
    --model cycle_gan_patch \
    --patchfloat 0 \
    --batch_size 15 \
    --num_threads 15 \
    --save_latest_freq 1500 \
    --lambda_identity 0.5 \
    --display_freq 1500 \
    --pool_size 1000 \
    --netG resnet_6blocks_2parts \
    --netD basic_2parts \
    --ngf 32 \
    --input_nc 1  \
    --output_nc 1 \
    --norm batch \
    --name copd_emphysema_slice_resnet9_2parts_small

#--name copd_emphysema_float
#--netD basic3d_patch_float \
#--netG unet3d_patch_float \
#--batch_size 100 \     batch size 100 for 2 GPUs of 32GB each
#--netG resnet3d_9blocks \

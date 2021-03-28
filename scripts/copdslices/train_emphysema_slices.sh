#!/bin/bash
#source activate venv
# Batch size 20 works for 2 GPUs

source ~/.bashrc
conda activate venv
python train.py \
    --dataroot /ocean/projects/asc170022p/rohit33/COPDslices/ \
    --dataset_mode copdslice2class \
    # --gpu_ids 0,1 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --model cycle_gan_patch \
    --patchfloat 0 \
    # --batch_size 20 \
    --batch_size 80 \
    --num_threads 16 \
    #--save_latest_freq 300 \
    --save_latest_freq 60 \
    --lambda_identity 0.2 \
    #--display_freq 300 \
    --display_freq 60 \
    --lambda_identity 0.2 \
    --pool_size 1000 \
    --netG resnet_6blocks_2parts \
    --netD basic_2parts \
    --ngf 32 \
    --input_nc 1  \
    --output_nc 1 \
    --norm batch \
    --name copd_emphysema_slice_resnet9_2parts

#--name copd_emphysema_float
#--netD basic3d_patch_float \
#--netG unet3d_patch_float \
#--batch_size 100 \     batch size 100 for 2 GPUs of 32GB each
#--netG resnet3d_9blocks \

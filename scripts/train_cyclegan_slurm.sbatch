#!/bin/bash
#SBATCH --job-name=maps_cyclegan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -ex

cd "$SLURM_SUBMIT_DIR"

source /lustre/disk/home/users/mfaizan/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-img2img

python train.py \
    --dataroot ./datasets/maps \
    --name maps_cyclegan_slurm \
    --model cycle_gan \
    --pool_size 50 \
    --no_dropout \
    --use_wandb

#!/bin/bash
#SBATCH --job-name=maps_cyclegan_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -ex

cd "$SLURM_SUBMIT_DIR"

source /lustre/disk/home/users/mfaizan/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-img2img

python test.py \
    --dataroot ./datasets/maps \
    --name maps_cyclegan_slurm \
    --model cycle_gan \
    --no_dropout \
    --phase test \
    --num_test 1098 \
    --serial_batches \
    --eval \
    --results_dir /lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/maps_cyclegan_slurm_evaluation/test_results

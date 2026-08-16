#!/bin/bash
#SBATCH --job-name=build_9mo_dataset
#SBATCH --partition=gpu
#SBATCH --output=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/brain_masked_cropped_2mo_dataset/logs/build_9mo_%j.out
#SBATCH --error=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/brain_masked_cropped_2mo_dataset/logs/build_9mo_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "============================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "Node          : $SLURMD_NODENAME"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "============================================================"

REPO_DIR="/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix"
JOB_DIR="$REPO_DIR/data_preprocessing/brain_masked_cropped_2mo_dataset"
# Same output root as the 2mo build -- 9mo subjects get their own subject-id subdirectories
# (distinct IDs, e.g. ICC105A vs ICC105), so this just adds to the existing tree, no collisions.
OUT_ROOT="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/brain_masked_cropped_normalized_to_common_space"
mkdir -p "$JOB_DIR/logs"

source /lustre/disk/home/users/mfaizan/miniconda3/etc/profile.d/conda.sh
conda activate moco

echo "Conda env     : $CONDA_DEFAULT_ENV"
echo "Python        : $(which python)"
echo ""

cd "$REPO_DIR"
python -u data_preprocessing/brain_masked_cropped_2mo_dataset/build_dataset.py \
    --runs-root /lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/normalized_to_common_space \
    --out-root "$OUT_ROOT" \
    --out-csv "$JOB_DIR/build_log_9mo.csv" \
    --age-group 9mo \
    --workers "$SLURM_CPUS_PER_TASK"

echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Job finished successfully."

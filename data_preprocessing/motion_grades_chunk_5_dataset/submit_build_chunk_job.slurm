#!/bin/bash
#SBATCH --job-name=build_motion_grades
#SBATCH --partition=gpu
#SBATCH --output=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset/logs/build_%j.out
#SBATCH --error=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset/logs/build_%j.err
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
JOB_DIR="$REPO_DIR/data_preprocessing/motion_grades_chunk_5_dataset"
OUT_ROOT="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/motion_grades_chunk_5_dataset"
mkdir -p "$JOB_DIR/logs"

source /lustre/disk/home/users/mfaizan/miniconda3/etc/profile.d/conda.sh
conda activate moco

echo "Conda env     : $CONDA_DEFAULT_ENV"
echo "Python        : $(which python)"
echo ""

cd "$REPO_DIR"
python -u data_preprocessing/motion_grades_chunk_5_dataset/build_chunk_dataset.py \
    --out-root "$OUT_ROOT" \
    --out-csv "$JOB_DIR/chunk_metadata.csv" \
    --workers "$SLURM_CPUS_PER_TASK"

echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Job finished successfully."

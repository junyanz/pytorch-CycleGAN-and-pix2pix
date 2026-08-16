#!/bin/bash
#SBATCH --job-name=run_bboxes
#SBATCH --partition=gpu
#SBATCH --output=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/effective_bounding_box_analysis/logs/run_bboxes_%j.out
#SBATCH --error=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/effective_bounding_box_analysis/logs/run_bboxes_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "============================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "Node          : $SLURMD_NODENAME"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "============================================================"

REPO_DIR="/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix"
OUT_DIR="$REPO_DIR/data_preprocessing/effective_bounding_box_analysis"
mkdir -p "$OUT_DIR/logs"

source /lustre/disk/home/users/mfaizan/miniconda3/etc/profile.d/conda.sh
conda activate moco

echo "Conda env     : $CONDA_DEFAULT_ENV"
echo "Python        : $(which python)"
echo ""

cd "$REPO_DIR"
python -u data_preprocessing/effective_bounding_box_analysis/compute_run_bboxes.py \
    --runs-root /lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/normalized_to_common_space \
    --out-csv "$OUT_DIR/run_bounding_boxes.csv"

echo "------------------------------------------------------------"
echo "End time : $(date)"
echo "Job finished successfully."

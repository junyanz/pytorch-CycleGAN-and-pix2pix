#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -p seas_gpu_requeue
#SBATCH -n 8 # number of cores
#SBATCH --mem 128000 # memory pool for all cores
#SBATCH --gres=gpu:2 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH --constraint=cc5.2
#SBATCH -o slurm/_cellsegm_train_%j.%N.out # STDOUT
#SBATCH -e slurm/_cellsegm_train_%j.%N.err # STDERR
#module load Anaconda3/5.0.1-fasrc01
#module load python/3.6.3-fasrc01
#module load cuda/10.1.243-fasrc01
module load Java/1.8.0_201
module load cuda/10.1.243-fasrc01
module load gcc
echo "java 1.8.0, gcc and cuda 10.1.2.43 loaded"
source activate /n/home09/scajasordonez/anaconda3/envs/pytorch-CycleGAN-and-pix2pix

nvcc --version
echo "Loaded"

# Run
### Ablation Study ###
# python -u -m torch.distributed.launch --master_port=9520 --nproc_per_node=2 --use_env cellsegm_train.py --world-size 2 --epochs 1 --lr 0.0001
#python -u -m torch.distributed.launch --master_port=9525 --nproc_per_node=2 --use_env cellsegm_train.py --world-size 2 --epochs 1 --lr 0.0001
python ../train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_augmented_horiz_v2_500_epochs --dataset_mode vcg --model pix2pix --output_nc 19 --direction BtoA --use_wandb --n_epochs 500

######

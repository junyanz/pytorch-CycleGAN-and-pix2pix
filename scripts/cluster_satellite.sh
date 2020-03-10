#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=7500  # memory in Mb
#SBATCH --time=0-08:00:00


export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}datasets/

# Activate the relevant virtual environment:
rsync -ua /home/${STUDENT_ID}/pytorch-CycleGAN-and-pix2pix/tars/AB.tar.gz "${DATASET_DIR}"
echo $DATASET_DIR
tar -xzf "${DATASET_DIR}/AB.tar.gz" -C "${DATASET_DIR}AB"

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

sh /home/${STUDENT_ID}/pytorch-CycleGAN-and-pix2pix/scripts/pix2pix_satellite_cluster.sh ${DATASET_DIR}AB

#python train_evaluate_emnist_classification_system.py --filepath_to_arguments_json_file experiment_configs/emnist_tutorial_config.json

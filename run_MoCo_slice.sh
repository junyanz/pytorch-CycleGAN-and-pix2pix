#!/usr/bin/env bash

LOG_DIR="./logs/ssl_exp"
EXPERIMENT="moco_slice_resnet18_lungmask_fa"

mkdir -p ${LOG_DIR}

python train_MoCo_slice.py \
  --exp-name=${EXPERIMENT} \
  --world-size=1 \
  --rank=0 \
  --dist-url='tcp://localhost:10001' \
  --dist-backend='nccl' \
  --npgus-per-node=4 \
  --workers-slice=20 \
  --epochs=20 \
  --start-epoch=0 \
  --resume='' \
  --print-freq=10 \
  --seed=0 \
  --num-slice=379 \
  --batch-size=512 \
  --lr=0.01 \
  --rep-dim-slice=512 \
  --moco-dim-slice=256 \
  --moco-k-slice=4096 \
  --moco-m-slice=0.999 \
  --moco-t-slice=0.2 \
  --augmentation='fa' \
  --slice-size=224 \
  --mask-threshold=0.05 \
  --sample-prop=1.0 \
  --mask-imputation >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1
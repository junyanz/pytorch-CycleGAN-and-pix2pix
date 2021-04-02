#!/usr/bin/env bash

LOG_DIR="./logs/ssl_exp"
EXPERIMENT="moco_slice_affine_encoder_512_128"

mkdir -p ${LOG_DIR}

python train_MoCo_slice.py \
  --exp-name=${EXPERIMENT} \
  --world-size=1 \
  --rank=0 \
  --dist-url='tcp://localhost:10001' \
  --dist-backend='nccl' \
  --npgus-per-node=4 \
  --workers-slice=16 \
  --epochs=10 \
  --start-epoch=0 \
  --resume='' \
  --print-freq=10 \
  --seed=0 \
  --num-slice=379 \
  --batch-size=128 \
  --lr=0.3 \
  --rep-dim-slice=512 \
  --moco-dim-slice=128 \
  --moco-k-slice=4096 \
  --moco-m-slice=0.999 \
  --moco-t-slice=0.2 \
  --transform-type='affine' \
  --slice-size=447 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1
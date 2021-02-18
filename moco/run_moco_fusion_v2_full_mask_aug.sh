#!/usr/bin/env bash

set -x

LOG_DIR="./logs"
EXPERIMENT="custom_encoder_fusion_full_v2_mask_aug"

mkdir -p ${LOG_DIR}
#rm ${LOG_DIR}/${EXPERIMENT}.log

export MKL_THREADING_LAYER=GNU

python main_moco_fusion_full_v2_cont_aug.py \
  --workers-patch=8 \
  --workers-graph=4 \
  --epochs=50 \
  --batch-size-patch=128 \
  --batch-size-graph=16 \
  --lr=0.03 \
  --start-epoch=0 \
  --print-freq=20 \
  --resume="" \
  --resume-graph="" \
  --world-size=1 \
  --num-sub-epoch=0 \
  --rank=0 \
  --dist-url="tcp://localhost:10001" \
  --dist-backend="nccl" \
  --seed=0 \
  --multiprocessing-distributed \
  --num-patch=581 \
  --root-dir="../data/patch_data_32_6_reg_mask/" \
  --fold=5 \
  --moco-dim=128 \
  --moco-k-patch=4096 \
  --moco-k-graph=4096 \
  --moco-m=0.999 \
  --moco-t=0.2 \
  --mlp \
  --cos \
  --exp-name="${EXPERIMENT}"
# >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1

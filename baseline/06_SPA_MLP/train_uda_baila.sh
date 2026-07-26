#!/usr/bin/env bash

# SPA on BailA: bailA_2 (source) -> bailA_1 (target).
python train_uda.py \
  --pl spa \
  --tar_par 0.2 \
  --svd_par 1.0 \
  --max_epoch 5 \
  --method DANNE \
  --dset bailA \
  --net mlp \
  --s 0 \
  --t 1 \
  --output logs/uda/ \
  --worker 3 \
  --momentum 0.3 \
  --laplac laplac1 \
  --ap gauss \
  --batch_size 32 \
  --seeds 1 2 3 4 5 \
  --ifsvd \
  --gpu_id 1

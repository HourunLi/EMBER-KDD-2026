#!/usr/bin/env bash

# Strict domain-generalization protocol:
#   train/model-select on syn-2 only, then frozen test on syn-1.
python fairvgnn_domain.py \
  --source_dataset='syn-2' \
  --target_dataset='syn-1' \
  --encoder='GCN' \
  --prop='spmm' \
  --epochs=500 \
  --gpu=0

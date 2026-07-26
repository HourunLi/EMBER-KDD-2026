#!/usr/bin/env bash

# Strict domain-generalization protocol:
#   train/model-select on germanA_2 only, then frozen test on all of germanA_1.
python fairvgnn_domain.py \
  --source_dataset='germanA_2' \
  --target_dataset='germanA_1' \
  --encoder='GCN' \
  --epochs=500 \
  --gpu=0 
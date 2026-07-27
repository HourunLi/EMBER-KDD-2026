#!/usr/bin/env bash

# Strict domain-generalization protocol:
#   train/model-select on pokec_z only, then frozen test on pokec_n.
python fairvgnn_domain.py \
  --source_dataset='pokec_z' \
  --target_dataset='pokec_n' \
  --encoder='GCN' \
  --prop='spmm' \
  --epochs=100 \
  --gpu=6 \
  --runs=1

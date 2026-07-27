#!/usr/bin/env bash

# Strict domain-generalization protocol:
#   train/model-select on bailA_2 only, then frozen test on all of bailA_1.
python fairvgnn_domain.py \
  --source_dataset=bailA_2 \
  --target_dataset=bailA_1 \
  --encoder=GCN \
  --clip_e=1 \
  --d_epochs=5 \
  --g_epochs=10 \
  --c_epochs=10 \
  --c_lr=0.01 \
  --e_lr=0.001 \
  --ratio=1 \
  --epochs=300 \
  --gpu=2
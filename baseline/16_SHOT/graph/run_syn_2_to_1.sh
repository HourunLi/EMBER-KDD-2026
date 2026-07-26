#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/syn-2_to_syn-1}"

mkdir -p "${OUTPUT_DIR}"
RESULT_FILES=()

for SEED in 1 2 3 4 5; do
  SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
  SOURCE_DIR="${SEED_DIR}/source"
  TARGET_DIR="${SEED_DIR}/target"
  SOURCE_CHECKPOINT="${SOURCE_DIR}/source_model.pt"
  mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}"

  echo "============================================================"
  echo "Running syn-2 -> syn-1 strict SHOT with seed=${SEED}"
  echo "============================================================"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" source \
    --csv "${REPO_ROOT}/dataset/syn/syn-2_feat.csv" \
    --edges "${REPO_ROOT}/dataset/syn/syn-2_edges.txt" \
    --label_file "${REPO_ROOT}/dataset/syn/syn-2_label.txt" \
    --headerless_features \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --experiment "syn-2_to_syn-1_strict_shot_gcn" \
    --source_domain "syn-2" \
    --id_column "node_index" \
    --label_column "label" \
    --sensitive_column "sensitive" \
    --negative_label "0" \
    --positive_label "1" \
    --sensitive_group_zero "0" \
    --sensitive_group_one "1" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 500 \
    --lr 0.01 \
    --hidden_dim 128 \
    --encoder_dim 128 \
    --bottleneck_dim 256 \
    --dropout 0.5 \
    --smooth 0.1

  # The standalone target label/sensitive paths are passed for final metrics,
  # but their contents are read only after the adapted checkpoint is saved.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" target \
    --csv "${REPO_ROOT}/dataset/syn/syn-1_feat.csv" \
    --edges "${REPO_ROOT}/dataset/syn/syn-1_edges.txt" \
    --label_file "${REPO_ROOT}/dataset/syn/syn-1_label.txt" \
    --sensitive_file "${REPO_ROOT}/dataset/syn/syn-1_sens.txt" \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${TARGET_DIR}" \
    --target_domain "syn-1" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 15 \
    --batch_size 100000 \
    --lr 0.01 \
    --cls_par 0.3 \
    --ent_par 1.0 \
    --threshold 0 \
    --pseudo_refine_rounds 2

  RESULT_FILES+=("${TARGET_DIR}/results.json")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" aggregate \
  --results "${RESULT_FILES[@]}" \
  --output "${OUTPUT_DIR}/summary_5seeds.json" \
  --expected_runs 5

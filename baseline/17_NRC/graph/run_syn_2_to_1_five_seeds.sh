#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/syn-2_to_syn-1_nrc_gcn}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-300}"
TARGET_EPOCHS="${TARGET_EPOCHS:-15}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-64}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-0}"
NEIGHBOR_QUERY_CHUNK_SIZE="${NEIGHBOR_QUERY_CHUNK_SIZE:-64}"

SOURCE_FEATURES="${REPO_ROOT}/dataset/syn/syn-2_feat.csv"
SOURCE_EDGES="${REPO_ROOT}/dataset/syn/syn-2_edges.txt"
SOURCE_LABELS="${REPO_ROOT}/dataset/syn/syn-2_label.txt"
TARGET_FEATURES="${REPO_ROOT}/dataset/syn/syn-1_feat.csv"
TARGET_EDGES="${REPO_ROOT}/dataset/syn/syn-1_edges.txt"
TARGET_LABELS="${REPO_ROOT}/dataset/syn/syn-1_label.txt"
TARGET_SENSITIVE="${REPO_ROOT}/dataset/syn/syn-1_sens.txt"

mkdir -p "${OUTPUT_ROOT}"
RESULT_FILES=()

for SEED in 1 2 3 4 5; do
  RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
  SOURCE_DIR="${RUN_DIR}/source"
  TARGET_DIR="${RUN_DIR}/target"
  EVALUATION_DIR="${RUN_DIR}/evaluation"
  SOURCE_CHECKPOINT="${SOURCE_DIR}/source_model.pt"
  PREDICTION_PAYLOAD="${TARGET_DIR}/evaluation_payload.pt"
  RESULT_FILE="${EVALUATION_DIR}/results.json"

  mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}" "${EVALUATION_DIR}"

  echo "============================================================"
  echo "Seed ${SEED}: stage 1/3 - supervised syn-2 training"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" source \
    --dataset syn \
    --csv "${SOURCE_FEATURES}" \
    --edges "${SOURCE_EDGES}" \
    --label_file "${SOURCE_LABELS}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs "${SOURCE_EPOCHS}" \
    --lr 0.01 \
    --hidden_dim 128 \
    --encoder_dim 128 \
    --bottleneck_dim 256 \
    --dropout 0.5 \
    --smooth 0.1 \
    --train_ratio 0.8 \
    --validation_ratio 0.1

  echo "============================================================"
  echo "Seed ${SEED}: stage 2/3 - label-free syn-1 NRC adaptation"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" target \
    --dataset syn \
    --csv "${TARGET_FEATURES}" \
    --edges "${TARGET_EDGES}" \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${TARGET_DIR}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs "${TARGET_EPOCHS}" \
    --batch_size "${TARGET_BATCH_SIZE}" \
    --lr 0.001 \
    --K 5 \
    --KK 5 \
    --r 0.1 \
    --epsilon 1e-5 \
    --memory_bank_size "${MEMORY_BANK_SIZE}" \
    --neighbor_query_chunk_size "${NEIGHBOR_QUERY_CHUNK_SIZE}"

  echo "============================================================"
  echo "Seed ${SEED}: stage 3/3 - final label/sensitive evaluation"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" evaluate \
    --dataset syn \
    --label_file "${TARGET_LABELS}" \
    --sensitive_file "${TARGET_SENSITIVE}" \
    --prediction_payload "${PREDICTION_PAYLOAD}" \
    --output "${RESULT_FILE}" \
    --export_dir "${EVALUATION_DIR}"

  RESULT_FILES+=("${RESULT_FILE}")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" summarize \
  --results "${RESULT_FILES[@]}" \
  --output "${OUTPUT_ROOT}/five_seed_summary.json"

echo "============================================================"
echo "All five Syn seeds completed."
echo "Summary: ${OUTPUT_ROOT}/five_seed_summary.json"
echo "============================================================"

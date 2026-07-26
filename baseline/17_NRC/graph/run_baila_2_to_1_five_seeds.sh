#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/bailA_2_to_bailA_1_nrc_gcn}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-300}"
TARGET_EPOCHS="${TARGET_EPOCHS:-15}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-64}"

SOURCE_CSV="${REPO_ROOT}/dataset/bailA/bailA_2.csv"
SOURCE_EDGES="${REPO_ROOT}/dataset/bailA/bailA_2_edges.txt"
TARGET_CSV="${REPO_ROOT}/dataset/bailA/bailA_1.csv"
TARGET_EDGES="${REPO_ROOT}/dataset/bailA/bailA_1_edges.txt"

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
  echo "Seed ${SEED}: stage 1/3 - supervised source training"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" source \
    --dataset bailA \
    --csv "${SOURCE_CSV}" \
    --edges "${SOURCE_EDGES}" \
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
  echo "Seed ${SEED}: stage 2/3 - label-free target NRC adaptation"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" target \
    --dataset bailA \
    --csv "${TARGET_CSV}" \
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
    --epsilon 1e-5

  echo "============================================================"
  echo "Seed ${SEED}: stage 3/3 - final RECID/WHITE evaluation"
  echo "============================================================"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" evaluate \
    --dataset bailA \
    --csv "${TARGET_CSV}" \
    --prediction_payload "${PREDICTION_PAYLOAD}" \
    --output "${RESULT_FILE}" \
    --export_dir "${EVALUATION_DIR}"

  RESULT_FILES+=("${RESULT_FILE}")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/baila_nrc.py" summarize \
  --results "${RESULT_FILES[@]}" \
  --output "${OUTPUT_ROOT}/five_seed_summary.json"

echo "============================================================"
echo "All five seeds completed."
echo "Summary: ${OUTPUT_ROOT}/five_seed_summary.json"
echo "============================================================"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/bailA_2_to_bailA_1}"

mkdir -p "${OUTPUT_DIR}"
RESULT_FILES=()

for SEED in 1 2 3 4 5; do
  SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
  SOURCE_DIR="${SEED_DIR}/source"
  TARGET_DIR="${SEED_DIR}/target"
  SOURCE_CHECKPOINT="${SOURCE_DIR}/source_model.pt"
  mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}"

  echo "============================================================"
  echo "Running BailA_2 -> BailA_1 strict SHOT with seed=${SEED}"
  echo "============================================================"

  # Stage 1 may access BailA_2 features, graph edges, and RECID labels.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_shot.py" source \
    --csv "${REPO_ROOT}/dataset/bailA/bailA_2.csv" \
    --edges "${REPO_ROOT}/dataset/bailA/bailA_2_edges.txt" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 1000 \
    --lr 0.01 \
    --hidden_dim 128 \
    --encoder_dim 128 \
    --bottleneck_dim 256 \
    --dropout 0.5 \
    --smooth 0.1

  # Stage 2 is a new process. It accepts no source CSV/edge path. Target RECID
  # and WHITE are loaded only after all SHOT updates, for final evaluation.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/baila_shot.py" target \
    --csv "${REPO_ROOT}/dataset/bailA/bailA_1.csv" \
    --edges "${REPO_ROOT}/dataset/bailA/bailA_1_edges.txt" \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${TARGET_DIR}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 15 \
    --batch_size 256 \
    --lr 0.01 \
    --cls_par 0.3 \
    --ent_par 1.0 \
    --threshold 0 \
    --pseudo_refine_rounds 2

  RESULT_FILES+=("${TARGET_DIR}/results.json")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/baila_shot.py" aggregate \
  --results "${RESULT_FILES[@]}" \
  --output "${OUTPUT_DIR}/summary_5seeds.json" \
  --expected_runs 5

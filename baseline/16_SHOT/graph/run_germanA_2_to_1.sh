#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/germanA_2_to_germanA_1}"

mkdir -p "${OUTPUT_DIR}"
RESULT_FILES=()

for SEED in 1 2 3 4 5; do
  SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
  SOURCE_DIR="${SEED_DIR}/source"
  TARGET_DIR="${SEED_DIR}/target"
  SOURCE_CHECKPOINT="${SOURCE_DIR}/source_model.pt"
  mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}"

  echo "============================================================"
  echo "Running germanA_2 -> germanA_1 strict SHOT with seed=${SEED}"
  echo "============================================================"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" source \
    --csv "${REPO_ROOT}/dataset/germanA/germanA_2.csv" \
    --edges "${REPO_ROOT}/dataset/germanA/germanA_2_edges.txt" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --experiment "germanA_2_to_germanA_1_strict_shot_gcn" \
    --source_domain "germanA_2" \
    --id_column "user_id" \
    --label_column "GoodCustomer" \
    --sensitive_column "Gender" \
    --negative_label=-1 \
    --positive_label=1 \
    --sensitive_group_zero "Female" \
    --sensitive_group_one "Male" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 500 \
    --lr 0.01 \
    --hidden_dim 128 \
    --encoder_dim 128 \
    --bottleneck_dim 256 \
    --dropout 0.5 \
    --smooth 0.1

  # This process accepts no germanA_2 CSV/edge path. GoodCustomer and Gender
  # are read from germanA_1 only after all target adaptation updates.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" target \
    --csv "${REPO_ROOT}/dataset/germanA/germanA_1.csv" \
    --edges "${REPO_ROOT}/dataset/germanA/germanA_1_edges.txt" \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${TARGET_DIR}" \
    --target_domain "germanA_1" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 15 \
    --batch_size 64 \
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

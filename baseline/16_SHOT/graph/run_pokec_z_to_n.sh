#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pokec_z_to_pokec_n}"

mkdir -p "${OUTPUT_DIR}"
RESULT_FILES=()

for SEED in 1 2 3 4 5; do
  SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
  SOURCE_DIR="${SEED_DIR}/source"
  TARGET_DIR="${SEED_DIR}/target"
  SOURCE_CHECKPOINT="${SOURCE_DIR}/source_model.pt"
  mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}"

  echo "============================================================"
  echo "Running pokec_z -> pokec_n strict SHOT with seed=${SEED}"
  echo "============================================================"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" source \
    --csv "${REPO_ROOT}/dataset/pokec_z/pokec_z.csv" \
    --edges "${REPO_ROOT}/dataset/pokec_z/pokec_z_edges.txt" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --experiment "pokec_z_to_pokec_n_strict_shot_gcn" \
    --source_domain "pokec_z" \
    --id_column "user_id" \
    --label_column "I_am_working_in_field" \
    --sensitive_column "region" \
    --negative_label "0" \
    --positive_label "1" \
    --additional_positive_labels "2" "3" "4" \
    --ignored_labels=-1 \
    --ignore_index=-1 \
    --sensitive_group_zero "0" \
    --sensitive_group_one "1" \
    --edges_use_node_ids \
    --align_target_features_by_name \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --epochs 1000 \
    --lr 0.01 \
    --hidden_dim 128 \
    --encoder_dim 128 \
    --bottleneck_dim 256 \
    --dropout 0.5 \
    --smooth 0.1

  # This process receives only the source artifact and the unlabeled target
  # graph inputs. Target labels/region are read after the adapted model is saved.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gcn_shot.py" target \
    --csv "${REPO_ROOT}/dataset/pokec_n/pokec_n.csv" \
    --edges "${REPO_ROOT}/dataset/pokec_n/pokec_n_edges.txt" \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${TARGET_DIR}" \
    --target_domain "pokec_n" \
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

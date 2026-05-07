#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
CONFIG_PATH="${2:-configs/grpo_qwen25_1_5b.yaml}"
ADAPTER_PATH="${3:-}"
K_VALUES="${4:-4 8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

EVAL_DIR="./outputs/eval"
mkdir -p "${EVAL_DIR}"

if [[ "${MODE}" == "grpo" && -z "${ADAPTER_PATH}" ]]; then
  ADAPTER_PATH="$(ls -d ./outputs/qwen25_1_5b_grpo_gsm8k/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ "${MODE}" == "grpo" && -z "${ADAPTER_PATH}" ]]; then
  echo "No GRPO checkpoint found. Pass adapter path explicitly as the third argument." >&2
  exit 1
fi

for K in ${K_VALUES}; do
  OUTPUT_FILE="${EVAL_DIR}/${MODE}_passk_majk_k${K}.json"
  if [[ "${MODE}" == "baseline" ]]; then
    python -m src.eval.eval_passk_majk \
      --config "${CONFIG_PATH}" \
      --output_file "${OUTPUT_FILE}" \
      --k "${K}"
  elif [[ "${MODE}" == "grpo" ]]; then
    python -m src.eval.eval_passk_majk \
      --config "${CONFIG_PATH}" \
      --adapter_path "${ADAPTER_PATH}" \
      --output_file "${OUTPUT_FILE}" \
      --k "${K}"
  else
    echo "Usage: bash scripts/eval_passk_majk.sh [baseline|grpo] [config_path] [adapter_path] [\"4 8\"]" >&2
    exit 1
  fi
done

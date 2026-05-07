#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
CONFIG_PATH="${2:-configs/grpo_qwen25_1_5b.yaml}"
ADAPTER_PATH="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

EVAL_DIR="./outputs/eval"
mkdir -p "${EVAL_DIR}"

if [[ "${MODE}" == "baseline" ]]; then
  uv run --active python -m src.eval.eval_acc1 \
    --config "${CONFIG_PATH}" \
    --output_file "${EVAL_DIR}/baseline_acc1.jsonl"
elif [[ "${MODE}" == "grpo" ]]; then
  if [[ -z "${ADAPTER_PATH}" ]]; then
    ADAPTER_PATH="$(ls -d ./outputs/qwen25_1_5b_grpo_gsm8k/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${ADAPTER_PATH}" ]]; then
    echo "No GRPO checkpoint found. Pass adapter path explicitly as the third argument." >&2
    exit 1
  fi
  uv run --active python -m src.eval.eval_acc1 \
    --config "${CONFIG_PATH}" \
    --adapter_path "${ADAPTER_PATH}" \
    --output_file "${EVAL_DIR}/grpo_acc1.jsonl"
else
  echo "Usage: bash scripts/eval_acc1.sh [baseline|grpo] [config_path] [adapter_path]" >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/grpo_qwen25_1_5b.yaml}"
RESUME_PATH="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -n "${RESUME_PATH}" ]]; then
  uv run --active python -m src.train.train_grpo \
    --config "${CONFIG_PATH}" \
    --resume_from_checkpoint "${RESUME_PATH}"
else
  uv run --active python -m src.train.train_grpo --config "${CONFIG_PATH}"
fi

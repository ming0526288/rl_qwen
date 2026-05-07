#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/grpo_qwen25_1_5b.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
python -m src.train.train_grpo --config "${CONFIG_PATH}"

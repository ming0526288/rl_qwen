from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.config import ensure_dir, load_yaml_config, resolve_paths
from src.utils.logging_utils import setup_logger


def model_already_exists(model_dir: Path) -> bool:
    return model_dir.exists() and any(model_dir.iterdir())


def download_model(config_path: str) -> None:
    config = load_yaml_config(config_path)
    paths = resolve_paths(config, config_path)
    logger = setup_logger("download_model", paths["log_dir"])

    model_name = config["paths"]["model_name"]
    model_dir = ensure_dir(paths["model_dir"])
    if model_already_exists(model_dir):
        logger.info("Model directory already contains files. Skipping download: %s", model_dir)
        return

    logger.info("Downloading model %s to %s", model_name, model_dir)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("Model download complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen model from Hugging Face.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_model(args.config)


if __name__ == "__main__":
    main()

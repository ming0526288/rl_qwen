from __future__ import annotations

import argparse

from src.data.prepare_gsm8k import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess GSM8K.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()

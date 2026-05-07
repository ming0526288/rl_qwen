from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from src.utils.answer_extract import extract_ground_truth_from_gsm8k
from src.utils.config import ensure_dir, load_yaml_config, resolve_paths
from src.utils.logging_utils import setup_logger


PROMPT_TEMPLATE = """请解答下面的数学题。你可以给出必要的推理过程，但最后必须严格使用如下格式输出最终答案：

Answer: <数字>

题目：
{question}
"""


def build_record(example: dict[str, Any], split: str, index: int) -> dict[str, str]:
    question = example["question"].strip()
    answer = example["answer"].strip()
    return {
        "id": f"{split}-{index}",
        "question": question,
        "answer": answer,
        "ground_truth": extract_ground_truth_from_gsm8k(answer),
        "prompt": PROMPT_TEMPLATE.format(question=question),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(config_path: str) -> None:
    config = load_yaml_config(config_path)
    paths = resolve_paths(config, config_path)
    logger = setup_logger("prepare_gsm8k", paths["log_dir"])

    dataset_dir = ensure_dir(paths["dataset_dir"])
    logger.info("Downloading openai/gsm8k (config=main) to %s", dataset_dir)
    dataset = load_dataset("openai/gsm8k", "main")

    train_records = [
        build_record(example, "train", index)
        for index, example in enumerate(tqdm(dataset["train"], desc="Processing train"))
    ]
    test_records = [
        build_record(example, "test", index)
        for index, example in enumerate(tqdm(dataset["test"], desc="Processing test"))
    ]

    write_jsonl(train_records, paths["train_file"])
    write_jsonl(test_records, paths["test_file"])
    write_jsonl(test_records[:200], paths["test_200_file"])
    logger.info(
        "Saved train=%d, test=%d, test_200=%d",
        len(train_records),
        len(test_records),
        min(200, len(test_records)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare GSM8K as jsonl files.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()

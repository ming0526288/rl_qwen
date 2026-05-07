from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_row(model_name: str, acc1: dict[str, Any], passk: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Acc@1": acc1.get("Acc@1"),
        "Format Pass": acc1.get("Format Pass Rate"),
        "Avg Reward": acc1.get("Avg Reward"),
        "Avg Length": acc1.get("Avg Length"),
        "Overlong Rate": acc1.get("Overlong Rate"),
        "Invalid Rate": acc1.get("Invalid Answer Rate"),
        "Pass@4": passk.get("Pass@K") if passk.get("K") == 4 else None,
        "Maj@4": passk.get("Maj@K") if passk.get("K") == 4 else None,
    }


def summarize(eval_dir: str) -> tuple[Path, Path]:
    eval_path = Path(eval_dir)
    baseline_acc1 = read_json(eval_path / "baseline_acc1.summary.json")
    grpo_acc1 = read_json(eval_path / "grpo_acc1.summary.json")
    baseline_pass4 = read_json(eval_path / "baseline_passk_majk_k4.json")
    grpo_pass4 = read_json(eval_path / "grpo_passk_majk_k4.json")

    rows = [
        build_row("Qwen2.5-1.5B-Instruct", baseline_acc1, baseline_pass4),
        build_row("Qwen2.5-1.5B-Instruct + GRPO", grpo_acc1, grpo_pass4),
    ]
    frame = pd.DataFrame(rows)

    csv_path = eval_path / "results_summary.csv"
    md_path = eval_path / "results_summary.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")
    return md_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline vs GRPO evaluation results.")
    parser.add_argument("--eval_dir", required=True, help="Directory containing evaluation summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path, csv_path = summarize(args.eval_dir)
    print(f"Saved markdown summary to {md_path}")
    print(f"Saved csv summary to {csv_path}")


if __name__ == "__main__":
    main()

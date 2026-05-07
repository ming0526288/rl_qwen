from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.gsm8k_reward import compute_gsm8k_reward
from src.utils.config import ensure_dir, load_yaml_config, resolve_paths
from src.utils.logging_utils import setup_logger


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(model_path: Path, adapter_path: Path | None = None) -> tuple[Any, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, int(generated.shape[-1])


def evaluate(
    model_path: str,
    adapter_path: str | None,
    test_file: str,
    output_file: str,
    reward_cfg: dict[str, float],
    max_new_tokens: int,
) -> dict[str, float]:
    logger = setup_logger("eval_acc1")
    model, tokenizer = load_model(Path(model_path), Path(adapter_path) if adapter_path else None)
    examples = load_jsonl(Path(test_file))
    output_path = Path(output_file)
    ensure_dir(output_path.parent)

    results = []
    for example in tqdm(examples, desc="Evaluating Acc@1"):
        model_output, completion_length = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            max_new_tokens=max_new_tokens,
        )
        reward = compute_gsm8k_reward(
            completion_text=model_output,
            ground_truth=example["ground_truth"],
            completion_length=completion_length,
            answer_correct=float(reward_cfg["answer_correct"]),
            format_correct=float(reward_cfg["format_correct"]),
            overlong_512=float(reward_cfg["overlong_512"]),
            overlong_768=float(reward_cfg["overlong_768"]),
        )
        record = {
            "id": example["id"],
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "model_output": model_output,
            "extracted_answer": reward.extracted_answer,
            "correct": reward.correct,
            "format_ok": reward.format_ok,
            "completion_length": completion_length,
            "reward": reward.reward,
        }
        results.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_acc1(results)
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved detailed results to %s", output_path)
    logger.info("Saved summary to %s", summary_path)
    return summary


def summarize_acc1(results: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(results), 1)
    correct = sum(1 for item in results if item["correct"])
    format_ok = sum(1 for item in results if item["format_ok"])
    rewards = [float(item["reward"]) for item in results]
    lengths = [int(item["completion_length"]) for item in results]
    overlong = sum(1 for item in results if int(item["completion_length"]) > 512)
    invalid = sum(1 for item in results if item["extracted_answer"] is None)
    return {
        "Acc@1": correct / total,
        "Format Pass Rate": format_ok / total,
        "Avg Reward": sum(rewards) / total,
        "Avg Length": sum(lengths) / total,
        "Overlong Rate": overlong / total,
        "Invalid Answer Rate": invalid / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K Acc@1.")
    parser.add_argument("--config", default="configs/grpo_qwen25_1_5b.yaml", help="Path to YAML config.")
    parser.add_argument("--model_path", default=None, help="Base model path.")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--test_file", default=None, help="Path to test jsonl.")
    parser.add_argument("--output_file", default=None, help="Path to save per-sample results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    paths = resolve_paths(config, args.config)
    output_file = args.output_file or str(paths["eval_dir"] / "acc1_results.jsonl")
    evaluate(
        model_path=args.model_path or str(paths["model_dir"]),
        adapter_path=args.adapter_path,
        test_file=args.test_file or str(paths["test_file"]),
        output_file=output_file,
        reward_cfg=config["reward"],
        max_new_tokens=int(config["eval"]["max_new_tokens"]),
    )


if __name__ == "__main__":
    main()

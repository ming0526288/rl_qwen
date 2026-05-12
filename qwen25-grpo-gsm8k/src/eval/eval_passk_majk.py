from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.gsm8k_reward import compute_gsm8k_reward
from src.utils.answer_extract import answers_equal
from src.utils.config import ensure_dir, load_yaml_config, resolve_paths
from src.utils.logging_utils import setup_logger


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(model_path: Path, adapter_path: Path | None = None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def sample_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[list[tuple[str, int]]]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_only = outputs[:, inputs["input_ids"].shape[-1] :]
    texts = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
    pad_token_id = tokenizer.pad_token_id
    lengths = generated_only.ne(pad_token_id).sum(dim=1).tolist()
    flat_samples = [(text, int(length)) for text, length in zip(texts, lengths, strict=False)]
    return [flat_samples[i : i + k] for i in range(0, len(flat_samples), k)]


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def evaluate_passk_majk(
    model_path: str,
    adapter_path: str | None,
    test_file: str,
    output_file: str,
    reward_cfg: dict[str, float],
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    logger = setup_logger("eval_passk_majk")
    model, tokenizer = load_model(Path(model_path), Path(adapter_path) if adapter_path else None)
    examples = load_jsonl(Path(test_file))

    per_example = []
    for start in tqdm(range(0, len(examples), batch_size), desc=f"Evaluating Pass@{k}/Maj@{k}"):
        batch_examples = examples[start : start + batch_size]
        batch_samples = sample_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=[example["prompt"] for example in batch_examples],
            k=k,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        for example, samples in zip(batch_examples, batch_samples, strict=False):
            rewards = [
                compute_gsm8k_reward(
                    completion_text=text,
                    ground_truth=example["ground_truth"],
                    completion_length=completion_length,
                    answer_correct=float(reward_cfg["answer_correct"]),
                    format_correct=float(reward_cfg["format_correct"]),
                    overlong_512=float(reward_cfg["overlong_512"]),
                    overlong_768=float(reward_cfg["overlong_768"]),
                )
                for text, completion_length in samples
            ]
            extracted_answers = [item.extracted_answer for item in rewards]
            correct_count = sum(item.correct for item in rewards)
            maj_answer = majority_vote(extracted_answers)
            record = {
                "id": example["id"],
                "ground_truth": example["ground_truth"],
                "pass": correct_count > 0,
                "maj": answers_equal(maj_answer, example["ground_truth"]),
                "correct_count": correct_count,
                "unique_answer_count": len({answer for answer in extracted_answers if answer is not None}),
                "answers": extracted_answers,
            }
            per_example.append(record)

    total = max(len(per_example), 1)
    summary = {
        "K": k,
        "Pass@K": sum(item["pass"] for item in per_example) / total,
        "Maj@K": sum(item["maj"] for item in per_example) / total,
        "Avg Correct Count": sum(item["correct_count"] for item in per_example) / total,
        "Avg Unique Answer Count": sum(item["unique_answer_count"] for item in per_example) / total,
        "per_example": per_example,
    }

    output_path = Path(output_file)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved Pass@K/Maj@K results to %s", output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pass@K and Maj@K on GSM8K.")
    parser.add_argument("--config", default="configs/grpo_qwen25_1_5b.yaml", help="Path to YAML config.")
    parser.add_argument("--model_path", default=None, help="Base model path.")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--test_file", default=None, help="Path to test jsonl.")
    parser.add_argument("--output_file", default=None, help="Path to save summary json.")
    parser.add_argument("--k", type=int, required=True, help="Number of samples per question.")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of prompts to evaluate per batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    paths = resolve_paths(config, args.config)
    output_file = args.output_file or str(paths["eval_dir"] / f"passk_majk_k{args.k}.json")
    evaluate_passk_majk(
        model_path=args.model_path or str(paths["model_dir"]),
        adapter_path=args.adapter_path,
        test_file=args.test_file or str(paths["test_file"]),
        output_file=output_file,
        reward_cfg=config["reward"],
        k=args.k,
        temperature=float(config["eval"]["temperature"]),
        top_p=float(config["eval"]["top_p"]),
        max_new_tokens=int(config["eval"]["max_new_tokens"]),
        batch_size=int(args.batch_size or config["eval"].get("batch_size", 8)),
    )


if __name__ == "__main__":
    main()

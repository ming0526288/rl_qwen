from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)

from src.rewards.gsm8k_reward import RewardBreakdown, compute_gsm8k_reward
from src.utils.config import ensure_dir, load_yaml_config, resolve_paths
from src.utils.logging_utils import MetricLogger, StepTimer, get_gpu_metrics, setup_logger
from src.utils.seed import set_seed


def import_grpo_components() -> tuple[type[Any], type[Any]]:
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "Failed to import GRPOTrainer from trl. Please install a recent TRL version "
            "that includes GRPOTrainer, for example `pip install -U trl`."
        ) from exc
    return GRPOConfig, GRPOTrainer


@dataclass
class RewardAccumulator:
    reward_mean: float = 0.0
    reward_std: float = 0.0
    correct_rate: float = 0.0
    format_rate: float = 0.0
    reward_answer: float = 0.0
    reward_format: float = 0.0
    reward_length_penalty: float = 0.0
    completion_length: float = 0.0
    sample_count: int = 0
    token_count: int = 0

    def update(self, items: list[RewardBreakdown], completion_lengths: list[int]) -> None:
        rewards = np.array([item.reward for item in items], dtype=float)
        self.reward_mean = float(rewards.mean()) if len(rewards) else 0.0
        self.reward_std = float(rewards.std()) if len(rewards) else 0.0
        self.correct_rate = float(np.mean([item.correct for item in items])) if items else 0.0
        self.format_rate = float(np.mean([item.format_ok for item in items])) if items else 0.0
        self.reward_answer = float(np.mean([item.answer_reward for item in items])) if items else 0.0
        self.reward_format = float(np.mean([item.format_reward for item in items])) if items else 0.0
        self.reward_length_penalty = float(np.mean([item.length_penalty for item in items])) if items else 0.0
        self.completion_length = float(np.mean(completion_lengths)) if completion_lengths else 0.0
        self.sample_count = len(items)
        self.token_count = int(sum(completion_lengths))

    def to_log_dict(self) -> dict[str, float]:
        return {
            "train/reward_mean": self.reward_mean,
            "train/reward_std": self.reward_std,
            "train/correct_rate": self.correct_rate,
            "train/format_rate": self.format_rate,
            "train/reward_answer": self.reward_answer,
            "train/reward_format": self.reward_format,
            "train/reward_length_penalty": self.reward_length_penalty,
            "train/completion_length": self.completion_length,
        }


class RewardFunction:
    def __init__(self, tokenizer: AutoTokenizer, reward_cfg: dict[str, float]) -> None:
        self.tokenizer = tokenizer
        self.reward_cfg = reward_cfg
        self.accumulator = RewardAccumulator()
        self.__name__ = "gsm8k_rule_reward"

    def __call__(
        self,
        prompts: list[str],
        completions: list[str] | list[list[dict[str, str]]],
        ground_truth: list[str],
        completion_ids: list[list[int]] | None = None,
        **_: Any,
    ) -> list[float]:
        del prompts
        normalized_completions = [self._completion_to_text(item) for item in completions]
        completion_lengths = self._get_completion_lengths(normalized_completions, completion_ids)

        details = [
            compute_gsm8k_reward(
                completion_text=text,
                ground_truth=truth,
                completion_length=length,
                answer_correct=float(self.reward_cfg["answer_correct"]),
                format_correct=float(self.reward_cfg["format_correct"]),
                overlong_512=float(self.reward_cfg["overlong_512"]),
                overlong_768=float(self.reward_cfg["overlong_768"]),
            )
            for text, truth, length in zip(normalized_completions, ground_truth, completion_lengths, strict=False)
        ]
        self.accumulator.update(details, completion_lengths)
        return [detail.reward for detail in details]

    @staticmethod
    def _completion_to_text(item: str | list[dict[str, str]]) -> str:
        if isinstance(item, str):
            return item
        return "".join(part.get("content", "") for part in item)

    def _get_completion_lengths(
        self,
        completions: list[str],
        completion_ids: list[list[int]] | None,
    ) -> list[int]:
        if completion_ids is not None:
            return [len(ids) for ids in completion_ids]
        return [
            len(self.tokenizer(completion, add_special_tokens=False)["input_ids"])
            for completion in completions
        ]


class MetricsCallback(TrainerCallback):
    def __init__(self, reward_fn: RewardFunction, metric_logger: MetricLogger) -> None:
        self.reward_fn = reward_fn
        self.metric_logger = metric_logger
        self.step_timer = StepTimer()

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        del control, kwargs
        logs = logs or {}
        step_time = self.step_timer.tick()
        reward_logs = self.reward_fn.accumulator.to_log_dict()

        sample_count = max(self.reward_fn.accumulator.sample_count, 1)
        token_count = max(self.reward_fn.accumulator.token_count, 1)
        perf_logs = {
            "train/step_time": step_time,
            "train/samples_per_second": sample_count / max(step_time, 1e-6),
            "train/tokens_per_second": token_count / max(step_time, 1e-6),
        }

        kl_value = logs.get("kl", logs.get("objective/kl", 0.0))
        entropy_value = logs.get("entropy", logs.get("objective/entropy", 0.0))
        clip_ratio = logs.get("clip_ratio", logs.get("policy/clipfrac_avg", 0.0))

        merged = {
            "train/loss": float(logs.get("loss", 0.0)),
            "train/learning_rate": float(logs.get("learning_rate", 0.0)),
            "train/grad_norm": float(logs.get("grad_norm", 0.0)),
            "train/global_step": float(state.global_step),
            "train/kl": float(kl_value or 0.0),
            "train/entropy": float(entropy_value or 0.0),
            "train/clip_ratio": float(clip_ratio or 0.0),
            **reward_logs,
            **perf_logs,
            **get_gpu_metrics(),
        }
        self.metric_logger.log(merged, step=state.global_step)


def load_model_and_tokenizer(model_path: Path, bf16: bool) -> tuple[Any, AutoTokenizer]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. Run `python scripts/download_model.py --config ...` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    )
    return model, tokenizer


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def copy_config_snapshot(config_path: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    shutil.copy2(config_path, output_dir / Path(config_path).name)


def build_training_args(config: dict[str, Any], output_dir: Path, logger: Any) -> Any:
    GRPOConfig, _ = import_grpo_components()
    training_cfg = config["training"]
    raw_kwargs = dict(
        output_dir=str(output_dir),
        bf16=bool(training_cfg["bf16"]),
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        max_steps=int(training_cfg["max_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        max_prompt_length=int(training_cfg["max_prompt_length"]),
        max_completion_length=int(training_cfg["max_completion_length"]),
        num_generations=int(training_cfg["num_generations"]),
        temperature=float(training_cfg["temperature"]),
        top_p=float(training_cfg["top_p"]),
        beta=float(training_cfg["beta"]),
        remove_unused_columns=False,
        log_on_each_node=False,
        report_to="none",
        run_name="grpo-train",
    )
    if int(training_cfg.get("eval_steps", 0)) > 0:
        raw_kwargs["eval_strategy"] = "no"
    if "epsilon" in training_cfg:
        raw_kwargs["epsilon"] = float(training_cfg["epsilon"])

    signature = inspect.signature(GRPOConfig.__init__)
    supported_keys = set(signature.parameters.keys())
    filtered_kwargs = {key: value for key, value in raw_kwargs.items() if key in supported_keys}
    dropped_keys = sorted(set(raw_kwargs.keys()) - set(filtered_kwargs.keys()))
    if dropped_keys:
        logger.warning(
            "Current TRL GRPOConfig does not support these config fields and they will be ignored: %s",
            ", ".join(dropped_keys),
        )
    try:
        return GRPOConfig(**filtered_kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Failed to initialize GRPOConfig after filtering unsupported arguments. "
            "Please check your installed TRL version and compare it with the project config."
        ) from exc


def train(config_path: str, resume_from_checkpoint: str | None = None) -> None:
    config = load_yaml_config(config_path)
    paths = resolve_paths(config, config_path)
    logger = setup_logger("train_grpo", paths["log_dir"])
    set_seed(int(config["project"]["seed"]))

    output_dir = ensure_dir(paths["output_dir"])
    ensure_dir(paths["eval_dir"])
    copy_config_snapshot(config_path, output_dir)

    dataset_file = paths["train_file"]
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {dataset_file}. Run `python scripts/download_dataset.py --config ...` first."
        )

    model, tokenizer = load_model_and_tokenizer(paths["model_dir"], bool(config["training"]["bf16"]))
    train_dataset = load_dataset("json", data_files=str(dataset_file), split="train")

    reward_fn = RewardFunction(tokenizer=tokenizer, reward_cfg=config["reward"])
    metric_logger = MetricLogger(
        backend=str(config["training"].get("report_to", "none")),
        project=str(config["project"]["name"]),
        run_name="grpo-train",
    )
    callback = MetricsCallback(reward_fn=reward_fn, metric_logger=metric_logger)

    try:
        training_args = build_training_args(config, output_dir, logger)
        _, GRPOTrainer = import_grpo_components()

        trainer_kwargs = {
            "model": model,
            "reward_funcs": reward_fn,
            "args": training_args,
            "train_dataset": train_dataset,
            "callbacks": [callback],
        }
        trainer_kwargs["processing_class"] = tokenizer
        if bool(config["lora"]["enabled"]):
            trainer_kwargs["peft_config"] = build_lora_config(config["lora"])

        try:
            trainer = GRPOTrainer(**trainer_kwargs)
        except TypeError as exc:
            if "processing_class" in trainer_kwargs:
                trainer_kwargs.pop("processing_class")
                trainer_kwargs["tokenizer"] = tokenizer
                try:
                    trainer = GRPOTrainer(**trainer_kwargs)
                except TypeError as nested_exc:
                    raise RuntimeError(
                        "The installed TRL version exposes a different GRPOTrainer signature. "
                        "Please check the README compatibility note and upgrade TRL."
                    ) from nested_exc
            else:
                raise RuntimeError(
                    "The installed TRL version exposes a different GRPOTrainer signature. "
                    "Please check the README compatibility note and upgrade TRL."
                ) from exc

        logger.info("Starting GRPO training with %d samples", len(train_dataset))
        if resume_from_checkpoint is not None:
            checkpoint_path = Path(resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
            logger.info("Resuming training from checkpoint: %s", checkpoint_path)
            train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path))
        else:
            train_result = trainer.train()
        trainer.save_model()
        trainer.save_state()

        summary = {
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "train_loss": float(train_result.training_loss),
            "reward_mean": reward_fn.accumulator.reward_mean,
            "reward_std": reward_fn.accumulator.reward_std,
        }
        with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Training complete. Summary saved to %s", output_dir / "train_summary.json")
    finally:
        metric_logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA + GRPO training on GSM8K.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional checkpoint path for resuming interrupted training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()

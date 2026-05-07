from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any

import torch


def setup_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class MetricLogger:
    def __init__(
        self,
        backend: str = "none",
        project: str | None = None,
        run_name: str | None = None,
        local_metrics_path: str | Path | None = None,
    ):
        self.backend = (backend or "none").lower()
        self.project = project
        self.run_name = run_name
        self.client = None
        self.local_metrics_path = Path(local_metrics_path) if local_metrics_path is not None else None

        if self.local_metrics_path is not None:
            self.local_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "wandb":
            import wandb

            self.client = wandb
            self.client.init(project=project, name=run_name)
        elif self.backend == "swanlab":
            import swanlab

            self.client = swanlab
            self.client.init(project=project, experiment_name=run_name)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        serializable_metrics = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        }
        if step is not None:
            serializable_metrics["_step"] = int(step)

        if self.local_metrics_path is not None:
            with self.local_metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serializable_metrics, ensure_ascii=False) + "\n")

        if self.backend == "none" or self.client is None:
            return
        if self.backend == "wandb":
            self.client.log(metrics, step=step)
        elif self.backend == "swanlab":
            self.client.log(metrics, step=step)

    def finish(self) -> None:
        if self.backend == "wandb" and self.client is not None:
            self.client.finish()
        elif self.backend == "swanlab" and self.client is not None:
            self.client.finish()


class StepTimer:
    def __init__(self) -> None:
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed


def get_gpu_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "system/gpu_memory_allocated": 0.0,
            "system/gpu_memory_reserved": 0.0,
        }
    return {
        "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
        "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
    }

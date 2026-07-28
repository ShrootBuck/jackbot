from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from jackbot.training.config import TrainConfig


@dataclass(slots=True)
class RunLogger:
    run: Any | None
    started_at: float
    log_checkpoints: bool = True
    last_steps: int = 0
    last_time: float = 0.0

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        now = time.perf_counter()
        elapsed = max(1e-6, now - (self.last_time or self.started_at))
        step_delta = step - self.last_steps
        process = psutil.Process()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        payload = {
            **metrics,
            "resources/ram_mb": process.memory_info().rss / (1024 * 1024),
            "resources/system_ram_percent": memory.percent,
            "resources/swap_mb": swap.used / (1024 * 1024),
            "throughput/steps_per_sec": step_delta / elapsed,
        }
        self.last_steps = step
        self.last_time = now
        if self.run is not None:
            self.run.log(payload, step=step)
        else:
            compact = " ".join(
                f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in payload.items()
            )
            print(f"[step {step}] {compact}")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def log_checkpoint(
        self,
        path: Path,
        aliases: list[str],
        metadata: dict[str, Any],
    ) -> None:
        if self.run is None or not self.log_checkpoints:
            return
        import wandb

        artifact = wandb.Artifact(
            name=f"checkpoint-{self.run.id}-{path.stem}",
            type="model",
            metadata=_json_safe_metadata(metadata),
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)


def make_logger(config: TrainConfig, initial_steps: int = 0) -> RunLogger:
    if not config.use_wandb:
        return RunLogger(
            run=None,
            started_at=time.perf_counter(),
            log_checkpoints=False,
            last_steps=initial_steps,
        )
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        id=config.wandb_run_id,
        resume=config.wandb_resume if config.wandb_run_id else None,
        name=config.wandb_run_name,
        group=config.wandb_group,
        job_type=config.wandb_job_type,
        tags=config.wandb_tags or None,
        config=config.to_dict(),
    )
    _define_wandb_metrics(run)
    return RunLogger(
        run=run,
        started_at=time.perf_counter(),
        log_checkpoints=config.wandb_log_checkpoints,
        last_steps=initial_steps,
    )


def _define_wandb_metrics(run: Any) -> None:
    run.define_metric("train/update")
    for pattern in (
        "train/*",
        "arena/*",
        "league/*",
        "opponent/*",
        "throughput/*",
        "resources/*",
        "time/*",
    ):
        run.define_metric(pattern, step_metric="train/update")


def _json_safe_metadata(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_metadata(item) for item in value]
    return value

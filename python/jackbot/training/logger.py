from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import psutil

from jackbot.training.config import TrainConfig


@dataclass(slots=True)
class RunLogger:
    run: Any | None
    started_at: float
    last_steps: int = 0
    last_time: float = 0.0

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        now = time.perf_counter()
        elapsed = max(1e-6, now - (self.last_time or self.started_at))
        step_delta = step - self.last_steps
        payload = {
            **metrics,
            "system/ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
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


def make_logger(config: TrainConfig) -> RunLogger:
    if not config.use_wandb:
        return RunLogger(run=None, started_at=time.perf_counter())
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        config=config.to_dict(),
    )
    return RunLogger(run=run, started_at=time.perf_counter())

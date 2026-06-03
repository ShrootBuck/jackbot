from __future__ import annotations

import math
from pathlib import Path

from jackbot.training.logger import RunLogger, _define_wandb_metrics, _json_safe_metadata


def test_json_safe_metadata_replaces_nonfinite_floats() -> None:
    metadata = _json_safe_metadata(
        {
            "best_score": float("-inf"),
            "nested": [1.0, float("inf"), math.nan],
            "path": Path("checkpoints/jackbot_latest.pt"),
        }
    )

    assert metadata == {
        "best_score": None,
        "nested": [1.0, None, None],
        "path": "checkpoints/jackbot_latest.pt",
    }


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float | int], int]] = []
        self.defined: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def log(self, payload: dict[str, float | int], step: int) -> None:
        self.logged.append((payload, step))

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.defined.append((args, kwargs))


def test_logger_puts_resource_metrics_under_resources() -> None:
    run = _FakeRun()
    logger = RunLogger(run=run, started_at=1.0, last_time=1.0)

    logger.log({"train/update": 1}, step=32)

    payload, step = run.logged[0]
    assert step == 32
    assert "resources/ram_mb" in payload
    assert "resources/system_ram_percent" in payload
    assert "resources/swap_mb" in payload
    assert not any(key.startswith("system/") for key in payload)


def test_wandb_resource_metrics_use_train_update_axis() -> None:
    run = _FakeRun()

    _define_wandb_metrics(run)

    assert (("train/update",), {}) in run.defined
    assert (("resources/*",), {"step_metric": "train/update"}) in run.defined

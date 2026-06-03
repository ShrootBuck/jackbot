from __future__ import annotations

from collections.abc import Iterable
import random
from pathlib import Path

import numpy as np
import torch

from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig
from jackbot.training.model import JackbotNet


def default_checkpoint_path() -> Path:
    if DEFAULT_BEST_CHECKPOINT.exists():
        return DEFAULT_BEST_CHECKPOINT

    checkpoints_dir = DEFAULT_BEST_CHECKPOINT.parent
    best = _newest_checkpoint(checkpoints_dir.glob("**/jackbot_best.pt"))
    if best is not None:
        return best

    latest = checkpoints_dir / "jackbot_latest.pt"
    if latest.exists():
        return latest

    nested_latest = _newest_checkpoint(checkpoints_dir.glob("**/jackbot_latest.pt"))
    return nested_latest if nested_latest is not None else DEFAULT_BEST_CHECKPOINT


def save_checkpoint(
    path: Path,
    model: JackbotNet,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    update: int,
    global_steps: int,
    best_score: float,
    completed_games: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "update": update,
        "global_steps": global_steps,
        "best_score": best_score,
        "completed_games": completed_games,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: JackbotNet,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[TrainConfig, int, int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng(checkpoint.get("rng", {}))
    return (
        TrainConfig.from_dict(checkpoint["config"]),
        int(checkpoint["update"]),
        int(checkpoint["global_steps"]),
        float(checkpoint.get("best_score", float("-inf"))),
        int(checkpoint.get("completed_games", 0)),
    )


def load_checkpoint_config(path: Path, device: torch.device | str = "cpu") -> TrainConfig:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return TrainConfig.from_dict(checkpoint["config"])


def _restore_rng(state: dict[str, object]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    mps_state = state.get("mps")
    if mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(mps_state)


def _newest_checkpoint(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)

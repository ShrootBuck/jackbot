from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig, default_oracle_checkpoint
from jackbot.training.model import JackbotNet
from jackbot.training.runtime import CPU_DEVICE


@dataclass(slots=True)
class LoadedCheckpoint:
    config: TrainConfig
    update: int
    global_steps: int
    best_score: float
    completed_games: int
    environment_state: dict[str, object] | None
    training_state: dict[str, object]
    rng_state: dict[str, object]


def default_checkpoint_path() -> Path:
    incumbent = default_oracle_checkpoint()
    if incumbent.exists():
        return incumbent

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
    environment_state: dict[str, object] | None = None,
    training_state: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "model": model.state_dict(),
        "model_spec": {
            "feature_schema": model.feature_schema,
            "centralized_critic": model.centralized_critic,
            "hidden_size": config.hidden_size,
            "critic_hidden_size": config.critic_hidden_size,
        },
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "update": update,
        "global_steps": global_steps,
        "best_score": best_score,
        "completed_games": completed_games,
        "environment_state": environment_state,
        "training_state": training_state or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: JackbotNet,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    config = checkpoint_config_from_payload(checkpoint)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng_state = checkpoint.get("rng", {})
    if restore_rng:
        restore_rng_state(rng_state)
    return LoadedCheckpoint(
        config=config,
        update=int(checkpoint["update"]),
        global_steps=int(checkpoint["global_steps"]),
        best_score=float(checkpoint.get("best_score", float("-inf"))),
        completed_games=int(checkpoint.get("completed_games", 0)),
        environment_state=checkpoint.get("environment_state"),
        training_state=dict(checkpoint.get("training_state", {})),
        rng_state=dict(rng_state),
    )


def load_checkpoint_config(path: Path) -> TrainConfig:
    checkpoint = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    return checkpoint_config_from_payload(checkpoint)


def load_model_weights(path: Path, model: JackbotNet) -> TrainConfig:
    checkpoint = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    source_config = checkpoint_config_from_payload(checkpoint)
    source_state = checkpoint["model"]
    destination_keys = set(model.state_dict())
    source_keys = set(source_state)
    unexpected = sorted(source_keys - destination_keys)
    missing = sorted(destination_keys - source_keys)
    allowed_prefixes = (
        "history_adapter.",
        "action_consequence_adapter.",
        "central_value_delta.",
    )
    invalid_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
    if unexpected or invalid_missing:
        raise ValueError(
            "incompatible warm-start checkpoint: "
            f"unexpected={unexpected}, missing={invalid_missing}"
        )
    result = model.load_state_dict(source_state, strict=False)
    if sorted(result.unexpected_keys) != unexpected or sorted(result.missing_keys) != missing:
        raise RuntimeError("warm-start state validation disagreed with PyTorch load result")
    for key in missing:
        tensor = model.state_dict()[key]
        if key.endswith(".weight") and not torch.count_nonzero(tensor).item() == 0:
            if key.startswith("central_value_delta.0."):
                continue
            raise ValueError(f"new warm-start parameter {key} is not zero-initialized")
        if key.endswith(".bias") and torch.count_nonzero(tensor).item() != 0:
            raise ValueError(f"new warm-start parameter {key} is not zero-initialized")
    return source_config


def checkpoint_config_from_payload(checkpoint: dict[str, object]) -> TrainConfig:
    config = TrainConfig.from_dict(checkpoint["config"])
    version = int(checkpoint.get("checkpoint_version", 1))
    if version == 1:
        if config.feature_schema != "base_v1" or config.centralized_critic:
            raise ValueError("legacy checkpoint cannot declare enhanced model features")
        return config
    if version != 2:
        raise ValueError(f"unsupported checkpoint version {version}")
    expected = {
        "feature_schema": config.feature_schema,
        "centralized_critic": config.centralized_critic,
        "hidden_size": config.hidden_size,
        "critic_hidden_size": config.critic_hidden_size,
    }
    if checkpoint.get("model_spec") != expected:
        raise ValueError("checkpoint model_spec does not match its config")
    return config


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_rng_state(state: dict[str, object]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)


def _newest_checkpoint(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)

from __future__ import annotations

import os

import torch

from jackbot.training.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    load_checkpoint_config,
    save_checkpoint,
)
from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig
from jackbot.training.model import JackbotNet


def test_checkpoint_roundtrip(tmp_path) -> None:
    config = TrainConfig.smoke()
    model = JackbotNet(config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        config,
        update=3,
        global_steps=123,
        best_score=0.75,
        completed_games=42,
    )
    loaded_config, update, global_steps, best_score, completed_games = load_checkpoint(
        path,
        model,
        optimizer,
    )

    assert loaded_config.hidden_size == config.hidden_size
    assert update == 3
    assert global_steps == 123
    assert best_score == 0.75
    assert completed_games == 42

    metadata_config = load_checkpoint_config(path)
    assert metadata_config.hidden_size == config.hidden_size
    assert metadata_config.total_updates == config.total_updates


def test_default_checkpoint_path_finds_newest_nested_best(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "checkpoints" / "old-run" / "jackbot_best.pt"
    new = tmp_path / "checkpoints" / "new-run" / "jackbot_best.pt"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.touch()
    new.touch()
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert (
        default_checkpoint_path()
        == DEFAULT_BEST_CHECKPOINT.parent / "new-run" / "jackbot_best.pt"
    )

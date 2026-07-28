from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from jackbot.training.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    load_checkpoint_config,
    load_model_weights,
    save_checkpoint,
)
from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig
from jackbot.training.model import JackbotNet
from jackbot.training.model import make_model


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
    loaded = load_checkpoint(
        path,
        model,
        optimizer,
    )

    assert loaded.config.hidden_size == config.hidden_size
    assert loaded.update == 3
    assert loaded.global_steps == 123
    assert loaded.best_score == 0.75
    assert loaded.completed_games == 42

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


def test_default_checkpoint_path_prefers_promoted_champion(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    promoted = tmp_path / "checkpoints" / "champions" / "promoted.pt"
    promoted.parent.mkdir(parents=True)
    promoted.touch()
    best = tmp_path / "checkpoints" / "jackbot_best.pt"
    best.touch()

    assert default_checkpoint_path() == Path("checkpoints/champions/promoted.pt")


def test_weight_only_init_adds_only_zero_initialized_candidate_branches(tmp_path) -> None:
    source_config = TrainConfig.smoke()
    source_model = make_model(source_config)
    optimizer = torch.optim.AdamW(source_model.parameters())
    source = tmp_path / "source.pt"
    save_checkpoint(source, source_model, optimizer, source_config, 3, 100, 0.5)
    candidate_config = TrainConfig.smoke()
    candidate_config.feature_schema = "a1h1"
    candidate_config.centralized_critic = True
    candidate = make_model(candidate_config)

    loaded_config = load_model_weights(source, candidate)

    assert loaded_config.feature_schema == "base_v1"
    source_state = source_model.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in source_state.items():
        assert torch.equal(candidate_state[key], value)
    assert torch.count_nonzero(candidate.history_adapter.weight).item() == 0
    assert torch.count_nonzero(candidate.action_consequence_adapter.weight).item() == 0
    assert torch.count_nonzero(candidate.central_value_delta[-1].weight).item() == 0


def test_checkpoint_loader_rejects_unknown_schema_version(tmp_path) -> None:
    config = TrainConfig.smoke()
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "future.pt"
    save_checkpoint(path, model, optimizer, config, 1, 32, 0.5)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["checkpoint_version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_checkpoint_config(path)

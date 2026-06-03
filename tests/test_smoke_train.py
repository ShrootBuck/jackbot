from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig
from jackbot.training.gauntlet import parse_args as parse_gauntlet_args
from jackbot.training.train import _config_for_resume, config_from_args, parse_args, train


def test_tiny_smoke_train(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.checkpoint_dir = tmp_path
    config.eval_games = 1

    train(config)

    assert (tmp_path / config.latest_name).exists()


def test_resume_config_keeps_checkpoint_training_settings_by_default() -> None:
    loaded = TrainConfig(
        total_updates=3_000,
        eval_games=256,
        league_enabled=True,
        checkpoint_dir=Path("checkpoints/rulefix-run"),
    )
    requested = TrainConfig()

    merged = _config_for_resume(loaded, requested)

    assert merged.total_updates == 3_000
    assert merged.eval_games == 256
    assert merged.league_enabled is True
    assert merged.checkpoint_dir == Path("checkpoints/rulefix-run")


def test_serious_profile_bakes_in_long_run_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["jackbot-train", "--profile", "serious"])

    config = config_from_args(parse_args())

    assert config.num_envs == 1024
    assert config.total_updates == 3_000
    assert config.eval_interval_updates == 50
    assert config.eval_games == 256
    assert config.league_enabled is True


def test_gauntlet_defaults_to_best_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jackbot-gauntlet"])

    args = parse_gauntlet_args()

    assert args.checkpoint == [DEFAULT_BEST_CHECKPOINT]


def test_resume_config_allows_explicit_safe_overrides() -> None:
    loaded = TrainConfig(total_updates=3_000, wandb_mode="online")
    requested = TrainConfig(total_updates=4_000, wandb_mode="offline")

    merged = _config_for_resume(loaded, requested)

    assert merged.total_updates == 4_000
    assert merged.wandb_mode == "offline"


def test_resume_config_rejects_hidden_size_mismatch() -> None:
    loaded = TrainConfig(hidden_size=2_048)
    requested = TrainConfig(hidden_size=4_096)

    with pytest.raises(ValueError, match="Cannot resume hidden_size"):
        _config_for_resume(loaded, requested)

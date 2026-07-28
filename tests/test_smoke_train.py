from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jackbot.training.checkpoint import checkpoint_sha256, save_checkpoint
from jackbot.training.config import DEFAULT_BEST_CHECKPOINT, TrainConfig
from jackbot.training.gauntlet import parse_args as parse_gauntlet_args
from jackbot.training.train import _config_for_resume, config_from_args, parse_args, train
from jackbot.training.model import make_model


def test_tiny_smoke_train(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.checkpoint_dir = tmp_path
    config.eval_games = 1

    train(config)

    assert (tmp_path / config.latest_name).exists()


def test_candidate_can_warm_start_enhanced_model_with_fresh_counters(tmp_path) -> None:
    source_config = TrainConfig.smoke()
    source_model = make_model(source_config)
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source = tmp_path / "source.pt"
    save_checkpoint(source, source_model, source_optimizer, source_config, 7, 999, 0.8)
    candidate = TrainConfig.smoke()
    candidate.feature_schema = "a1h1"
    candidate.centralized_critic = True
    candidate.shaping_start = 0.0
    candidate.checkpoint_dir = tmp_path / "candidate"
    candidate.eval_games = 1

    train(candidate, init_from=source)

    checkpoint = torch.load(
        candidate.checkpoint_dir / candidate.latest_name,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["update"] == candidate.total_updates
    assert checkpoint["global_steps"] == candidate.total_updates * candidate.num_envs * candidate.rollout_len
    assert checkpoint["config"]["parent_checkpoint"] == str(source)
    assert torch.count_nonzero(checkpoint["model"]["central_value_delta.2.weight"]).item() > 0


def test_interrupted_training_resumes_exactly(tmp_path) -> None:
    source_config = TrainConfig.smoke()
    source_model = make_model(source_config)
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source = tmp_path / "source-league.pt"
    save_checkpoint(source, source_model, source_optimizer, source_config, 1, 32, 0.5)
    base = TrainConfig.smoke()
    base.total_updates = 4
    base.anneal_lr = False
    base.eval_interval_updates = 10_000
    base.checkpoint_interval_updates = 1
    base.league_enabled = True
    base.league_baselines = ["heuristic"]
    base.league_opponents = [str(source)]
    base.league_auto_checkpoints = False
    base.league_deterministic_opponents = False
    base.parent_checkpoint = str(source)
    base.parent_checkpoint_sha256 = checkpoint_sha256(source)
    full = replace(base, checkpoint_dir=tmp_path / "full")
    split = replace(base, total_updates=2, checkpoint_dir=tmp_path / "split")

    train(full, init_from=source)
    train(split, init_from=source)
    train(
        replace(base, checkpoint_dir=split.checkpoint_dir),
        resume=split.checkpoint_dir / split.latest_name,
    )

    full_checkpoint = torch.load(
        full.checkpoint_dir / full.latest_name,
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        split.checkpoint_dir / split.latest_name,
        map_location="cpu",
        weights_only=False,
    )
    assert full_checkpoint["global_steps"] == resumed_checkpoint["global_steps"]
    for key, value in full_checkpoint["model"].items():
        assert torch.equal(value, resumed_checkpoint["model"][key]), key


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


def test_genius_profile_bakes_in_champion_finetune_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["jackbot-train", "--profile", "genius"])

    config = config_from_args(parse_args())

    assert config.hidden_size == 2048
    assert config.total_updates == 12_000
    assert config.lr == 1e-4
    assert config.shaping_start == 0.0
    assert config.eval_interval_updates == 100
    assert config.eval_games == 512
    assert config.milestone_interval_updates == 250
    assert config.league_enabled is True
    assert config.league_deterministic_opponents is False


def test_smoke_profile_can_parse_genius_without_long_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["jackbot-train", "--smoke", "--profile", "genius"])

    config = config_from_args(parse_args())

    assert config.total_updates == 2
    assert config.num_envs == 8


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

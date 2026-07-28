from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jackbot.training import candidate_screen
from jackbot.training.checkpoint import save_checkpoint
from jackbot.training.config import TrainConfig
from jackbot.training.model import make_model


def test_screen_arms_use_equal_transition_budgets() -> None:
    budgets = {
        spec.num_envs * spec.rollout_len * candidate_screen.SCREEN_UPDATES
        for spec in candidate_screen.SCREEN_ARMS.values()
    }

    assert budgets == {13_107_200}
    assert candidate_screen.SCREEN_ARMS["A"].feature_schema == "a1"
    assert candidate_screen.SCREEN_ARMS["H"].feature_schema == "h1"
    assert candidate_screen.SCREEN_ARMS["V"].centralized_critic is True


def test_candidate_config_is_a_clean_weight_only_screen(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        candidate_screen,
        "load_checkpoint_config",
        lambda path: TrainConfig(hidden_size=64),
    )
    source = tmp_path / "source.pt"
    args = _args()

    config = candidate_screen.candidate_config(
        "T",
        args,
        source,
        "source-hash",
        tmp_path / "T",
    )

    assert config.total_updates == 200
    assert config.num_envs == 512
    assert config.rollout_len == 128
    assert config.gamma == 0.999
    assert config.gae_lambda == 0.99
    assert config.anneal_lr is False
    assert config.weight_decay == 0.0
    assert config.shaping_start == 0.0
    assert config.parent_checkpoint == str(source)


def test_manifest_mismatch_fails_closed(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    candidate_screen.ensure_manifest(path, {"screen": 1})

    with pytest.raises(ValueError, match="manifest mismatch"):
        candidate_screen.ensure_manifest(path, {"screen": 2})


def test_final_checkpoint_archive_is_immutable_and_idempotent(tmp_path) -> None:
    latest = tmp_path / "latest.pt"
    final = tmp_path / "final.pt"
    torch.save({"update": 200}, latest)

    candidate_screen.archive_final_checkpoint(latest, final, 200)
    candidate_screen.archive_final_checkpoint(latest, final, 200)

    assert final.exists()
    assert latest.stat().st_ino == final.stat().st_ino


def test_arm_checkpoint_validation_binds_config_source_and_steps(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        candidate_screen,
        "load_checkpoint_config",
        lambda path: TrainConfig(hidden_size=64),
    )
    source = tmp_path / "source.pt"
    config = candidate_screen.candidate_config(
        "C",
        _args(),
        source,
        "source-hash",
        tmp_path / "C",
    )
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "candidate.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        config,
        update=200,
        global_steps=200 * config.num_envs * config.rollout_len,
        best_score=0.5,
        environment_state={"games": []},
    )

    assert candidate_screen.arm_checkpoint_valid(
        checkpoint,
        200,
        config,
        "source-hash",
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["global_steps"] += 1
    torch.save(payload, checkpoint)
    assert not candidate_screen.arm_checkpoint_valid(
        checkpoint,
        200,
        config,
        "source-hash",
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        screen_id="test-screen",
        updates=200,
        development_eval_games=8,
        final_eval_games=8,
        wandb_project="jackbot",
        wandb_mode="disabled",
        smoke=False,
    )

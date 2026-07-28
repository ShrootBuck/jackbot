from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jackbot.training import league as league_module
from jackbot.training.config import TrainConfig
from jackbot.training.env import TensorBatch


class _FakePolicy:
    def __init__(self, name: str, feature_schema: str = "base_v1") -> None:
        self.name = name
        self.feature_schema = feature_schema

    def actions(self, tensors, deterministic: bool = True):
        raise NotImplementedError


def test_league_refresh_reuses_loaded_checkpoint_opponents(monkeypatch) -> None:
    config = TrainConfig.smoke()
    config.league_baselines = []
    config.league_auto_checkpoints = False
    first = Path("first.pt")
    second = Path("second.pt")
    specs = [{first}, {first, second}]
    loaded: list[Path] = []

    def fake_checkpoint_specs(self):
        return specs.pop(0)

    def fake_load_model_policy(path):
        loaded.append(path)
        return _FakePolicy(path.name), config

    monkeypatch.setattr(league_module.LeaguePool, "_checkpoint_specs", fake_checkpoint_specs)
    monkeypatch.setattr(league_module, "load_model_policy", fake_load_model_policy)
    monkeypatch.setattr(league_module, "_release_stale_opponents", lambda: None)

    league = league_module.LeaguePool(config, torch.device("cpu"))
    league.refresh()

    assert loaded == [first, second]
    assert [opponent.name for opponent in league.opponents] == ["first.pt", "second.pt"]


def test_league_reloads_an_overwritten_checkpoint(monkeypatch, tmp_path) -> None:
    config = TrainConfig.smoke()
    config.league_baselines = []
    config.league_auto_checkpoints = False
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"first")
    config.league_opponents = [str(checkpoint)]
    loaded: list[Path] = []

    def fake_load_model_policy(path):
        loaded.append(path)
        return _FakePolicy(path.name), config

    monkeypatch.setattr(league_module, "load_model_policy", fake_load_model_policy)
    monkeypatch.setattr(league_module, "_release_stale_opponents", lambda: None)

    league = league_module.LeaguePool(config, torch.device("cpu"))
    checkpoint.write_bytes(b"second-version")
    league.refresh()

    assert loaded == [checkpoint, checkpoint]


def test_league_weights_are_category_totals(monkeypatch) -> None:
    config = TrainConfig.smoke()
    config.league_baselines = ["random", "heuristic"]
    config.league_auto_checkpoints = False
    config.league_self_play_weight = 0.55
    config.league_baseline_weight = 0.05
    config.league_checkpoint_weight = 0.40
    first = Path("first.pt")
    second = Path("second.pt")

    monkeypatch.setattr(
        league_module.LeaguePool,
        "_checkpoint_specs",
        lambda self: {first, second},
    )
    monkeypatch.setattr(
        league_module,
        "load_model_policy",
        lambda path: (_FakePolicy(path.name), config),
    )
    monkeypatch.setattr(league_module, "_release_stale_opponents", lambda: None)

    league = league_module.LeaguePool(config, torch.device("cpu"))

    assert league.weights == [0.025, 0.025, 0.2, 0.2]
    assert config.league_self_play_weight + sum(league.weights) == 1.0


def test_slice_tensor_batch_keeps_only_selected_ragged_rows() -> None:
    batch = _tensor_batch()
    mask = torch.tensor([True, False, True])

    selected = league_module._slice_tensor_batch(batch, mask)

    assert selected.obs[:, 0].tolist() == [0.0, 2.0]
    assert selected.action_offsets.tolist() == [0, 2, 5]
    assert selected.action_features[:, 0].tolist() == [0.0, 1.0, 3.0, 4.0, 5.0]
    assert selected.env_ids.tolist() == [0, 0, 1, 1, 1]


def test_required_league_checkpoint_failure_is_fatal(monkeypatch, tmp_path) -> None:
    config = TrainConfig.smoke()
    config.league_baselines = []
    config.league_auto_checkpoints = False
    required = tmp_path / "required.pt"
    config.league_opponents = [str(required)]
    monkeypatch.setattr(
        league_module,
        "load_model_policy",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )

    with pytest.raises(RuntimeError, match="required league opponent"):
        league_module.LeaguePool(config, torch.device("cpu"))


def test_league_reports_union_schema(monkeypatch) -> None:
    config = TrainConfig.smoke()
    config.league_baselines = []
    config.league_auto_checkpoints = False
    enhanced = Path("enhanced.pt")
    config.league_opponents = [str(enhanced)]
    monkeypatch.setattr(
        league_module,
        "load_model_policy",
        lambda path: (_FakePolicy(path.name, "h1"), config),
    )
    monkeypatch.setattr(league_module, "_release_stale_opponents", lambda: None)

    league = league_module.LeaguePool(config, torch.device("cpu"))

    assert league.feature_schema == "h1"


def _tensor_batch() -> TensorBatch:
    rows = 3
    return TensorBatch(
        obs=torch.arange(rows, dtype=torch.float32).unsqueeze(1),
        action_features=torch.arange(6, dtype=torch.float32).unsqueeze(1),
        action_offsets=torch.tensor([0, 2, 3, 6]),
        env_ids=torch.tensor([0, 0, 1, 2, 2, 2]),
        current_players=torch.arange(rows),
        belief_targets=torch.zeros(rows, 1),
        public_history=torch.zeros(rows, 0),
        action_consequences=torch.zeros(6, 0),
        rewards=torch.zeros(rows),
        team_rewards=torch.zeros(rows, 2),
        dones=torch.zeros(rows, dtype=torch.bool),
        winners=torch.full((rows,), -1),
        winning_move_players=torch.full((rows,), -1),
        acting_players=torch.arange(rows),
        acting_teams=torch.arange(rows) % 2,
        game_lengths=torch.ones(rows, dtype=torch.long),
    )

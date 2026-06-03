from __future__ import annotations

from pathlib import Path

import torch

from jackbot.training import league as league_module
from jackbot.training.config import TrainConfig


class _FakePolicy:
    def __init__(self, name: str) -> None:
        self.name = name

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

    def fake_load_model_policy(path, device):
        loaded.append(path)
        return _FakePolicy(path.name), config

    monkeypatch.setattr(league_module.LeaguePool, "_checkpoint_specs", fake_checkpoint_specs)
    monkeypatch.setattr(league_module, "load_model_policy", fake_load_model_policy)
    monkeypatch.setattr(league_module, "_release_accelerator_cache", lambda device: None)

    league = league_module.LeaguePool(config, torch.device("cpu"))
    league.refresh()

    assert loaded == [first, second]
    assert [opponent.name for opponent in league.opponents] == ["first.pt", "second.pt"]

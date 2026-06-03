from __future__ import annotations

from pathlib import Path

import torch

from jackbot.training import policies as policies_module
from jackbot.training.config import TrainConfig
from jackbot.training.policies import load_model_policy


class _FakeModel:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.device: torch.device | None = None
        self.evaluated = False

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.state = state

    def to(self, device: torch.device) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True


def test_load_model_policy_loads_checkpoint_on_cpu(monkeypatch) -> None:
    calls: list[object] = []

    def fake_load(path, map_location, weights_only):
        calls.append(map_location)
        return {
            "config": TrainConfig.smoke().to_dict(),
            "model": {"weight": object()},
            "optimizer": {"large_state": object()},
        }

    monkeypatch.setattr(policies_module.torch, "load", fake_load)
    monkeypatch.setattr(policies_module, "JackbotNet", _FakeModel)

    policy, _ = load_model_policy(Path("checkpoint.pt"), torch.device("mps"))

    assert calls == ["cpu"]
    assert policy.model.device == torch.device("mps")
    assert policy.model.evaluated is True

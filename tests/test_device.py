from __future__ import annotations

import torch

from jackbot.training.device import _benchmark, choose_device


def test_benchmark_enables_grad_inside_no_grad() -> None:
    with torch.no_grad():
        assert _benchmark(torch.device("cpu")) >= 0.0


def test_auto_device_avoids_mps_for_ragged_training(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert choose_device("auto") == torch.device("cpu")

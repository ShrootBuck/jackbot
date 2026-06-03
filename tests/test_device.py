from __future__ import annotations

import torch

from jackbot.training.device import _benchmark


def test_benchmark_enables_grad_inside_no_grad() -> None:
    with torch.no_grad():
        assert _benchmark(torch.device("cpu")) >= 0.0

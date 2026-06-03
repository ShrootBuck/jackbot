from __future__ import annotations

from functools import cache
import time

import torch


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("mps") if _mps_beats_cpu() else torch.device("cpu")


@cache
def _mps_beats_cpu() -> bool:
    try:
        cpu_time = _benchmark(torch.device("cpu"))
        mps_time = _benchmark(torch.device("mps"))
    except RuntimeError:
        return False
    return mps_time < cpu_time * 0.90


def _benchmark(device: torch.device) -> float:
    with torch.enable_grad():
        layer = torch.nn.Sequential(
            torch.nn.Linear(222, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
        ).to(device)
        x = torch.randn(4096, 222, device=device)
        for _ in range(5):
            layer(x).sum().backward()
            layer.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            layer(x).sum().backward()
            layer.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - start

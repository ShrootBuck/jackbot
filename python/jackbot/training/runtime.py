from __future__ import annotations

import torch


CPU_DEVICE = torch.device("cpu")


def training_device() -> torch.device:
    return CPU_DEVICE

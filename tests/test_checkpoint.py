from __future__ import annotations

import torch

from jackbot.training.checkpoint import load_checkpoint, save_checkpoint
from jackbot.training.config import TrainConfig
from jackbot.training.model import JackbotNet


def test_checkpoint_roundtrip(tmp_path) -> None:
    config = TrainConfig.smoke()
    model = JackbotNet(config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, config, update=3, global_steps=123, best_score=0.75)
    loaded_config, update, global_steps, best_score = load_checkpoint(
        path,
        model,
        optimizer,
    )

    assert loaded_config.hidden_size == config.hidden_size
    assert update == 3
    assert global_steps == 123
    assert best_score == 0.75

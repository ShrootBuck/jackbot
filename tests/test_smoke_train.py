from __future__ import annotations

from jackbot.training.config import TrainConfig
from jackbot.training.train import train


def test_tiny_smoke_train(tmp_path) -> None:
    config = TrainConfig.smoke()
    config.checkpoint_dir = tmp_path
    config.device = "cpu"
    config.eval_games = 1

    train(config)

    assert (tmp_path / config.latest_name).exists()

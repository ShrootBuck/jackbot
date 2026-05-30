from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    seed: int = 1
    num_envs: int = 512
    rollout_len: int = 64
    total_updates: int = 1_000
    ppo_epochs: int = 3
    minibatch_size: int = 4_096
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    anneal_lr: bool = True
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    belief_coef: float = 0.05
    max_grad_norm: float = 1.0
    hidden_size: int = 2048
    shaping_start: float = 1.0
    shaping_decay_fraction: float = 0.40
    device: str = "auto"
    checkpoint_dir: Path = Path("checkpoints")
    latest_name: str = "jackbot_latest.pt"
    best_name: str = "jackbot_best.pt"
    checkpoint_interval_updates: int = 25
    milestone_interval_updates: int = 50
    checkpoint_interval_seconds: float = 15 * 60
    eval_interval_updates: int = 25
    eval_games: int = 64
    wandb_project: str = "jackbot"
    wandb_mode: str = "online"
    use_wandb: bool = True

    @classmethod
    def smoke(cls) -> "TrainConfig":
        return cls(
            num_envs=8,
            rollout_len=4,
            total_updates=2,
            ppo_epochs=1,
            minibatch_size=16,
            hidden_size=64,
            checkpoint_interval_updates=1,
            milestone_interval_updates=10_000,
            checkpoint_interval_seconds=0.0,
            eval_interval_updates=1,
            eval_games=4,
            use_wandb=False,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrainConfig":
        config = dict(data)
        if "checkpoint_dir" in config:
            config["checkpoint_dir"] = Path(str(config["checkpoint_dir"]))
        return cls(**config)


def shaping_scale(config: TrainConfig, update: int) -> float:
    decay_updates = max(1, int(config.total_updates * config.shaping_decay_fraction))
    progress = min(1.0, update / decay_updates)
    return config.shaping_start * (1.0 - progress)

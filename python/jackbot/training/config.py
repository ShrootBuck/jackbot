from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_BEST_CHECKPOINT = Path("checkpoints/jackbot_best.pt")


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
    checkpoint_dir: Path = Path("checkpoints")
    latest_name: str = "jackbot_latest.pt"
    best_name: str = "jackbot_best.pt"
    checkpoint_interval_updates: int = 25
    milestone_interval_updates: int = 50
    checkpoint_interval_seconds: float = 15 * 60
    eval_interval_updates: int = 25
    eval_games: int = 64
    eval_opponents: list[str] = field(default_factory=lambda: ["random", "heuristic"])
    eval_num_envs: int = 32
    eval_max_steps_per_game: int = 2_000
    league_enabled: bool = False
    league_baselines: list[str] = field(default_factory=lambda: ["heuristic", "random"])
    league_opponents: list[str] = field(default_factory=list)
    league_auto_checkpoints: bool = True
    league_max_checkpoints: int = 8
    league_self_play_weight: float = 1.0
    league_baseline_weight: float = 1.0
    league_checkpoint_weight: float = 2.0
    league_refresh_interval_updates: int = 25
    league_deterministic_opponents: bool = True
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

    @classmethod
    def serious(cls) -> "TrainConfig":
        return cls(
            num_envs=1024,
            total_updates=3_000,
            eval_interval_updates=50,
            eval_games=256,
            league_enabled=True,
            league_baselines=["heuristic", "random"],
            league_auto_checkpoints=True,
            league_deterministic_opponents=True,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TrainConfig":
        config = dict(data)
        config.pop("device", None)
        if "checkpoint_dir" in config:
            config["checkpoint_dir"] = Path(str(config["checkpoint_dir"]))
        return cls(**config)


def shaping_scale(config: TrainConfig, update: int) -> float:
    decay_updates = max(1, int(config.total_updates * config.shaping_decay_fraction))
    progress = min(1.0, update / decay_updates)
    return config.shaping_start * (1.0 - progress)

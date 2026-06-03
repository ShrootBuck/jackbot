from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.config import TrainConfig
from jackbot.training.model import JackbotNet
from jackbot.training.policies import BaselinePolicy, ModelPolicy, evaluate_match
from jackbot.training.runtime import CPU_DEVICE, training_device


@dataclass(slots=True)
class EvalResult:
    games: int
    even_wins: int
    odd_wins: int

    @property
    def even_win_rate(self) -> float:
        return self.even_wins / max(1, self.games)


@torch.no_grad()
def evaluate_model(
    model: JackbotNet,
    games: int,
    opponent: str,
    seed: int,
    device: torch.device,
    num_envs: int = 32,
    max_steps_per_game: int = 2_000,
) -> EvalResult:
    model.eval()
    result = evaluate_match(
        ModelPolicy("model", model),
        BaselinePolicy(opponent),
        games,
        seed,
        device,
        num_envs=num_envs,
        max_steps_per_game=max_steps_per_game,
    )
    return EvalResult(games=games, even_wins=result.even_wins, odd_wins=result.odd_wins)


def load_model(path: Path) -> tuple[JackbotNet, TrainConfig]:
    checkpoint = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    config = TrainConfig.from_dict(checkpoint["config"])
    model = JackbotNet(config.hidden_size).to(CPU_DEVICE)
    model.load_state_dict(checkpoint["model"])
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Jackbot checkpoint.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()

    checkpoint = args.checkpoint or default_checkpoint_path()
    device = training_device()
    model, _ = load_model(checkpoint)
    for opponent in ["random", "heuristic"]:
        result = evaluate_model(model, args.games, opponent, args.seed, device)
        print(
            f"{opponent}: P1+P3 win rate {result.even_win_rate:.3f} "
            f"({result.even_wins}/{result.games})"
        )


if __name__ == "__main__":
    main()

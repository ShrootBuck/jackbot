from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from jackbot.training.config import TrainConfig
from jackbot.training.device import choose_device
from jackbot.training.env import TensorBatch, heuristic_actions, make_env, random_actions, to_tensors
from jackbot.training.model import JackbotNet


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
    env = make_env(min(num_envs, games), seed)
    batch = env.reset(seed)
    even_wins = 0
    odd_wins = 0
    completed = 0
    steps = 0

    while completed < games and steps < games * max_steps_per_game:
        tensors = to_tensors(batch, device)
        model_actions, _, _ = model.act(
            tensors.obs,
            tensors.action_features,
            tensors.action_offsets,
            deterministic=True,
        )
        baseline = _baseline_actions(tensors, opponent)
        model_turn = (tensors.current_players % 2) == 0
        actions = torch.where(model_turn, model_actions, baseline)
        batch = env.step(actions.cpu().numpy(), 0.0)
        steps += int(tensors.current_players.numel())
        winners = batch["winners"]
        for winner in winners:
            if winner == 0:
                even_wins += 1
                completed += 1
            elif winner == 1:
                odd_wins += 1
                completed += 1
            if completed >= games:
                break

    odd_wins += games - completed
    return EvalResult(games=games, even_wins=even_wins, odd_wins=odd_wins)


def _baseline_actions(tensors: TensorBatch, opponent: str) -> torch.Tensor:
    if opponent == "random":
        return random_actions(tensors)
    if opponent == "heuristic":
        return heuristic_actions(tensors)
    raise ValueError(f"unknown opponent {opponent!r}")


def load_model(path: Path, device: torch.device) -> tuple[JackbotNet, TrainConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig.from_dict(checkpoint["config"])
    model = JackbotNet(config.hidden_size).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Jackbot checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    for opponent in ["random", "heuristic"]:
        result = evaluate_model(model, args.games, opponent, args.seed, device)
        print(
            f"{opponent}: P1+P3 win rate {result.even_win_rate:.3f} "
            f"({result.even_wins}/{result.games})"
        )


if __name__ == "__main__":
    main()

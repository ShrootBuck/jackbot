from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from jackbot.training.config import TrainConfig
from jackbot.training.device import choose_device
from jackbot.training.env import TensorBatch, heuristic_actions, make_env, random_actions, to_tensors
from jackbot.training.model import JackbotNet


class Policy(Protocol):
    name: str

    def actions(self, tensors: TensorBatch, deterministic: bool = True) -> torch.Tensor:
        ...


@dataclass(slots=True)
class BaselinePolicy:
    name: str

    def actions(self, tensors: TensorBatch, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        if self.name == "random":
            return random_actions(tensors)
        if self.name == "heuristic":
            return heuristic_actions(tensors)
        raise ValueError(f"unknown baseline policy {self.name!r}")


@dataclass(slots=True)
class ModelPolicy:
    name: str
    model: JackbotNet

    @torch.no_grad()
    def actions(self, tensors: TensorBatch, deterministic: bool = True) -> torch.Tensor:
        actions, _, _ = self.model.act(
            tensors.obs,
            tensors.action_features,
            tensors.action_offsets,
            deterministic=deterministic,
        )
        return actions


@dataclass(slots=True)
class MatchResult:
    even_policy: str
    odd_policy: str
    games: int
    even_wins: int
    odd_wins: int
    unfinished: int

    @property
    def even_win_rate(self) -> float:
        return self.even_wins / max(1, self.games)

    @property
    def odd_win_rate(self) -> float:
        return self.odd_wins / max(1, self.games)


@dataclass(slots=True)
class CandidateOpponentResult:
    candidate: str
    opponent: str
    games_per_side: int
    candidate_even: MatchResult
    candidate_odd: MatchResult

    @property
    def candidate_wins(self) -> int:
        return self.candidate_even.even_wins + self.candidate_odd.odd_wins

    @property
    def candidate_points(self) -> float:
        draws = self.candidate_even.unfinished + self.candidate_odd.unfinished
        return self.candidate_wins + 0.5 * draws

    @property
    def games(self) -> int:
        return self.candidate_even.games + self.candidate_odd.games

    @property
    def win_rate(self) -> float:
        return self.candidate_points / max(1, self.games)

    @property
    def ci95_radius(self) -> float:
        p = self.win_rate
        return 1.96 * math.sqrt(p * (1.0 - p) / max(1, self.games))


@dataclass(slots=True)
class GauntletResult:
    candidate: str
    opponents: list[CandidateOpponentResult]

    @property
    def score(self) -> float:
        if not self.opponents:
            return 0.0
        return sum(result.win_rate for result in self.opponents) / len(self.opponents)

    @property
    def total_games(self) -> int:
        return sum(result.games for result in self.opponents)


def load_model_policy(
    checkpoint_path: Path,
    device: torch.device,
    name: str | None = None,
) -> tuple[ModelPolicy, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainConfig.from_dict(checkpoint["config"])
    model = JackbotNet(config.hidden_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return ModelPolicy(name=name or checkpoint_path.stem, model=model), config


def policy_from_spec(spec: str, device: torch.device) -> Policy:
    if spec in {"random", "heuristic"}:
        return BaselinePolicy(spec)
    policy, _ = load_model_policy(Path(spec), device)
    return policy


@torch.no_grad()
def evaluate_match(
    even_policy: Policy,
    odd_policy: Policy,
    games: int,
    seed: int,
    device: torch.device,
    num_envs: int = 32,
    max_steps_per_game: int = 2_000,
    deterministic: bool = True,
) -> MatchResult:
    even_wins = 0
    odd_wins = 0
    completed = 0
    steps = 0
    env = make_env(min(num_envs, games), seed)
    batch = env.reset(seed)

    while completed < games and steps < games * max_steps_per_game:
        tensors = to_tensors(batch, device)
        even_actions = even_policy.actions(tensors, deterministic=deterministic)
        odd_actions = odd_policy.actions(tensors, deterministic=deterministic)
        even_turn = (tensors.current_players % 2) == 0
        actions = torch.where(even_turn, even_actions, odd_actions)
        batch = env.step(actions.cpu().numpy(), 0.0)
        steps += int(tensors.current_players.numel())

        for winner in batch["winners"]:
            if winner == 0:
                even_wins += 1
                completed += 1
            elif winner == 1:
                odd_wins += 1
                completed += 1
            if completed >= games:
                break

    unfinished = games - completed
    return MatchResult(
        even_policy=even_policy.name,
        odd_policy=odd_policy.name,
        games=games,
        even_wins=even_wins,
        odd_wins=odd_wins,
        unfinished=unfinished,
    )


def evaluate_candidate_against(
    candidate: Policy,
    opponent: Policy,
    games_per_side: int,
    seed: int,
    device: torch.device,
    num_envs: int = 32,
    max_steps_per_game: int = 2_000,
    deterministic: bool = True,
) -> CandidateOpponentResult:
    as_even = evaluate_match(
        candidate,
        opponent,
        games_per_side,
        seed,
        device,
        num_envs=num_envs,
        max_steps_per_game=max_steps_per_game,
        deterministic=deterministic,
    )
    as_odd = evaluate_match(
        opponent,
        candidate,
        games_per_side,
        seed + 1_000_003,
        device,
        num_envs=num_envs,
        max_steps_per_game=max_steps_per_game,
        deterministic=deterministic,
    )
    return CandidateOpponentResult(
        candidate=candidate.name,
        opponent=opponent.name,
        games_per_side=games_per_side,
        candidate_even=as_even,
        candidate_odd=as_odd,
    )


def run_gauntlet(
    candidate: Policy,
    opponents: list[Policy],
    games_per_side: int,
    seed: int,
    device: torch.device,
    num_envs: int = 32,
    max_steps_per_game: int = 2_000,
    deterministic: bool = True,
) -> GauntletResult:
    results = [
        evaluate_candidate_against(
            candidate,
            opponent,
            games_per_side,
            seed + index * 2_000_011,
            device,
            num_envs=num_envs,
            max_steps_per_game=max_steps_per_game,
            deterministic=deterministic,
        )
        for index, opponent in enumerate(opponents)
    ]
    return GauntletResult(candidate=candidate.name, opponents=results)


def gauntlet_metrics(result: GauntletResult, prefix: str = "arena") -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}/score": result.score,
        f"{prefix}/total_games": result.total_games,
    }
    for opponent_result in result.opponents:
        key = opponent_result.opponent.replace("/", "_")
        metrics[f"{prefix}/{key}_win_rate"] = opponent_result.win_rate
        metrics[f"{prefix}/{key}_ci95"] = opponent_result.ci95_radius
        metrics[f"{prefix}/{key}_games"] = opponent_result.games
    return metrics


def default_device(requested: str) -> torch.device:
    return choose_device(requested)

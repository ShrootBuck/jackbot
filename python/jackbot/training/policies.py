from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from jackbot.training.checkpoint import checkpoint_config_from_payload
from jackbot.training.config import TrainConfig, union_feature_schemas
from jackbot.training.env import TensorBatch, heuristic_actions, make_env, random_actions, to_tensors
from jackbot.training.model import JackbotNet, make_model
from jackbot.training.runtime import CPU_DEVICE


class Policy(Protocol):
    name: str

    @property
    def feature_schema(self) -> str:
        ...

    def actions(self, tensors: TensorBatch, deterministic: bool = True) -> torch.Tensor:
        ...


@dataclass(slots=True)
class BaselinePolicy:
    name: str

    @property
    def feature_schema(self) -> str:
        return "base_v1"

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

    @property
    def feature_schema(self) -> str:
        return self.model.feature_schema

    @torch.no_grad()
    def actions(self, tensors: TensorBatch, deterministic: bool = True) -> torch.Tensor:
        actions, _, _ = self.model.act(
            tensors.obs,
            tensors.action_features,
            tensors.action_offsets,
            deterministic=deterministic,
            public_history=tensors.public_history,
            action_consequences=tensors.action_consequences,
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
    def ci95_lower(self) -> float:
        lower, _ = _wilson_interval(self.win_rate, self.games)
        return lower

    @property
    def ci95_upper(self) -> float:
        _, upper = _wilson_interval(self.win_rate, self.games)
        return upper

    @property
    def ci95_radius(self) -> float:
        p = self.win_rate
        return max(p - self.ci95_lower, self.ci95_upper - p)


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
    name: str | None = None,
) -> tuple[ModelPolicy, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=CPU_DEVICE, weights_only=False)
    config = checkpoint_config_from_payload(checkpoint)
    rng_state = torch.get_rng_state()
    try:
        model = make_model(config)
        model.load_state_dict(checkpoint["model"])
        model.to(CPU_DEVICE)
        model.eval()
    finally:
        torch.set_rng_state(rng_state)
    return ModelPolicy(name=name or checkpoint_path.stem, model=model), config


def policy_from_spec(spec: str) -> Policy:
    if spec in {"random", "heuristic"}:
        return BaselinePolicy(spec)
    policy, _ = load_model_policy(Path(spec))
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
    rng_state = torch.get_rng_state()
    try:
        return _evaluate_match(
            even_policy,
            odd_policy,
            games,
            seed,
            device,
            num_envs,
            max_steps_per_game,
            deterministic,
        )
    finally:
        torch.set_rng_state(rng_state)


def _evaluate_match(
    even_policy: Policy,
    odd_policy: Policy,
    games: int,
    seed: int,
    device: torch.device,
    num_envs: int,
    max_steps_per_game: int,
    deterministic: bool,
) -> MatchResult:
    even_wins = 0
    odd_wins = 0
    unfinished = 0
    feature_schema = union_feature_schemas(even_policy.feature_schema, odd_policy.feature_schema)
    seed_stride = 0x9E37_79B9
    torch.manual_seed(seed)

    for start in range(0, games, num_envs):
        chunk_size = min(num_envs, games - start)
        chunk_seed = (seed + start * seed_stride) % (1 << 64)
        env = make_env(chunk_size, chunk_seed, feature_schema)
        batch = env.reset(chunk_seed)
        active = torch.ones(chunk_size, dtype=torch.bool)

        for _ in range(max_steps_per_game):
            if not bool(active.any().item()):
                break
            tensors = to_tensors(batch, device)
            even_actions = even_policy.actions(tensors, deterministic=deterministic)
            odd_actions = odd_policy.actions(tensors, deterministic=deterministic)
            even_turn = (tensors.current_players % 2) == 0
            actions = torch.where(even_turn, even_actions, odd_actions)
            batch = env.step(actions.cpu().numpy(), 0.0)

            for env_index, winner in enumerate(batch["winners"]):
                if not active[env_index] or winner < 0:
                    continue
                if winner == 0:
                    even_wins += 1
                else:
                    odd_wins += 1
                active[env_index] = False

        unfinished += int(active.sum().item())
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
        seed,
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


def _wilson_interval(rate: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    denominator = 1.0 + z * z / games
    center = (rate + z * z / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


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

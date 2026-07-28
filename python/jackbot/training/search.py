from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jackbot import PlayGame
from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.env import to_tensors
from jackbot.training.config import union_feature_schemas
from jackbot.training.policies import ModelPolicy, Policy, load_model_policy, policy_from_spec
from jackbot.training.runtime import training_device


@dataclass(slots=True)
class ActionScore:
    action_id: int
    label: str
    prior: float
    score: float
    wins: float
    rollouts: int


@torch.no_grad()
def rank_actions(
    game: PlayGame,
    model_policy: ModelPolicy,
    opponent: Policy,
    device: torch.device,
    rollouts: int = 32,
    max_steps: int = 500,
    rollout_deterministic: bool = False,
) -> list[ActionScore]:
    batch = game.batch()
    tensors = to_tensors(batch, device)
    root_player = int(tensors.current_players[0].item())
    root_team = root_player % 2
    output = model_policy.model(
        tensors.obs,
        tensors.action_features,
        tensors.action_offsets,
        public_history=tensors.public_history,
        action_consequences=tensors.action_consequences,
    )
    priors = output.log_probs.exp().detach().cpu().tolist()
    labels = game.legal_action_labels()
    scores = []

    for index, (action_id, label) in enumerate(labels):
        wins = 0.0
        for _ in range(rollouts):
            child = game.copy()
            child.step(action_id)
            wins += _rollout_score(
                child,
                root_team,
                model_policy,
                opponent,
                device,
                max_steps=max_steps,
                deterministic=rollout_deterministic,
            )
        scores.append(
            ActionScore(
                action_id=action_id,
                label=label,
                prior=float(priors[index]),
                score=wins / max(1, rollouts),
                wins=wins,
                rollouts=rollouts,
            )
        )

    return sorted(scores, key=lambda item: (item.score, item.prior), reverse=True)


def search_policy_target(scores: list[ActionScore], temperature: float = 0.10) -> list[float]:
    if not scores:
        return []
    if temperature <= 0.0:
        best = max(range(len(scores)), key=lambda index: scores[index].score)
        return [1.0 if index == best else 0.0 for index in range(len(scores))]
    raw = torch.tensor([score.score for score in scores], dtype=torch.float32)
    probs = torch.softmax(raw / temperature, dim=0)
    return [float(value) for value in probs.tolist()]


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or default_checkpoint_path()
    device = training_device()
    model_policy, _ = load_model_policy(checkpoint)
    opponent = model_policy if args.opponent == "self" else policy_from_spec(args.opponent)
    game = PlayGame(
        args.seed,
        union_feature_schemas(model_policy.feature_schema, opponent.feature_schema),
    )
    scores = rank_actions(
        game,
        model_policy,
        opponent,
        device,
        rollouts=args.rollouts,
        max_steps=args.max_steps,
        rollout_deterministic=args.deterministic_rollouts,
    )
    for rank, score in enumerate(scores[: args.top], start=1):
        print(
            f"{rank:>2}. {score.score:.3f} prior={score.prior:.3f} "
            f"wins={score.wins:g}/{score.rollouts} {score.label}"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(score) for score in scores], indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank legal moves with rollout search.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opponent", default="self", help="self, random, heuristic, or checkpoint path.")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--deterministic-rollouts", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def _rollout_score(
    game: PlayGame,
    root_team: int,
    model_policy: ModelPolicy,
    opponent: Policy,
    device: torch.device,
    max_steps: int,
    deterministic: bool,
) -> float:
    for _ in range(max_steps):
        winner = game.winner()
        if winner is not None:
            return 1.0 if winner == root_team else 0.0

        batch = game.batch()
        tensors = to_tensors(batch, device)
        current_team = int(tensors.current_players[0].item()) % 2
        policy = model_policy if current_team == root_team else opponent
        action = int(policy.actions(tensors, deterministic=deterministic)[0].item())
        game.step(action)

    return 0.5


def example_from_search(
    game: PlayGame,
    scores: list[ActionScore],
    temperature: float,
) -> dict[str, Any]:
    batch = game.batch()
    by_id = {score.action_id: score for score in scores}
    ordered = [by_id[action_id] for action_id, _ in game.legal_action_labels()]
    target = search_policy_target(ordered, temperature)
    value = sum(prob * score.score for prob, score in zip(target, ordered, strict=True))
    best = max(ordered, key=lambda score: score.score) if ordered else None
    return {
        "obs": batch["obs"][0].tolist(),
        "action_features": batch["action_features"].tolist(),
        "public_history": batch["public_history"][0].tolist(),
        "action_consequences": batch["action_consequences"].tolist(),
        "feature_schema": batch["feature_schema"],
        "target_probs": target,
        "value": value,
        "best_action": best.action_id if best is not None else None,
        "scores": [asdict(score) for score in ordered],
    }


if __name__ == "__main__":
    main()

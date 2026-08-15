from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jackbot import BatchEnv, PlayGame
from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.env import to_tensors
from jackbot.training.config import union_feature_schemas
from jackbot.training.policies import ModelPolicy, Policy, load_model_policy, policy_from_spec
from jackbot.training.runtime import training_device


MAX_ROLLOUT_BATCH = 2_048


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
    return rank_action_groups(
        [game],
        model_policy,
        opponent,
        device,
        rollouts=rollouts,
        max_steps=max_steps,
        rollout_deterministic=rollout_deterministic,
    )[0]


@torch.no_grad()
def rank_action_groups(
    games: Sequence[PlayGame],
    model_policy: ModelPolicy,
    opponent: Policy,
    device: torch.device,
    rollouts: int = 32,
    max_steps: int = 500,
    rollout_deterministic: bool = False,
) -> list[list[ActionScore]]:
    """Rank several roots in one ragged neural-policy batch.

    Advisor roots differ only in their sampled hidden cards. Batching all of
    their counterfactual games avoids thousands of batch-size-one MLP calls.
    Trajectories that reach the depth limit bootstrap from the learned value.
    """
    if rollouts <= 0:
        raise ValueError("rollouts must be positive")
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if not games:
        return []

    feature_schema = union_feature_schemas(model_policy.feature_schema, opponent.feature_schema)
    root_batch = _load_game_batch(games, feature_schema)
    root_tensors = to_tensors(root_batch, device)
    root_output = model_policy.model(
        root_tensors.obs,
        root_tensors.action_features,
        root_tensors.action_offsets,
        public_history=root_tensors.public_history,
        action_consequences=root_tensors.action_consequences,
    )
    flat_priors = root_output.log_probs.exp().detach().cpu().tolist()
    root_offsets = root_tensors.action_offsets.cpu().tolist()
    root_teams = torch.remainder(root_tensors.current_players, 2).cpu().numpy()
    labels_by_game = [game.legal_action_labels() for game in games]
    if all(len(labels) <= 1 for labels in labels_by_game):
        return [
            [
                ActionScore(
                    action_id=labels[0][0],
                    label=labels[0][1],
                    prior=float(flat_priors[root_offsets[game_index]]),
                    score=0.5,
                    wins=0.0,
                    rollouts=0,
                )
            ]
            if labels
            else []
            for game_index, labels in enumerate(labels_by_game)
        ]
    wins_by_game = [np.zeros(len(labels), dtype=np.float64) for labels in labels_by_game]

    child_states: list[dict[str, Any]] = []
    lane_games: list[int] = []
    lane_actions: list[int] = []
    lane_root_teams: list[int] = []
    for game_index, (game, labels) in enumerate(zip(games, labels_by_game, strict=True)):
        if root_offsets[game_index + 1] - root_offsets[game_index] != len(labels):
            raise RuntimeError("root policy offsets disagree with legal action labels")
        root_team = int(root_teams[game_index])
        for action_index, (action_id, _) in enumerate(labels):
            for _ in range(rollouts):
                child = game.copy()
                child.step(action_id)
                winner = child.winner()
                if winner is None:
                    child_states.append(child.state())
                    lane_games.append(game_index)
                    lane_actions.append(action_index)
                    lane_root_teams.append(root_team)
                elif winner == root_team:
                    wins_by_game[game_index][action_index] += 1.0

    if child_states:
        for start in range(0, len(child_states), MAX_ROLLOUT_BATCH):
            end = start + MAX_ROLLOUT_BATCH
            _finish_batched_rollouts(
                child_states[start:end],
                lane_games[start:end],
                lane_actions[start:end],
                lane_root_teams[start:end],
                wins_by_game,
                feature_schema,
                model_policy,
                opponent,
                device,
                max_steps,
                rollout_deterministic,
            )

    grouped_scores: list[list[ActionScore]] = []
    for game_index, labels in enumerate(labels_by_game):
        scores = []
        prior_start = root_offsets[game_index]
        for action_index, (action_id, label) in enumerate(labels):
            wins = float(wins_by_game[game_index][action_index])
            scores.append(
                ActionScore(
                    action_id=action_id,
                    label=label,
                    prior=float(flat_priors[prior_start + action_index]),
                    score=wins / rollouts,
                    wins=wins,
                    rollouts=rollouts,
                )
            )
        grouped_scores.append(
            sorted(scores, key=lambda item: (item.score, item.prior), reverse=True)
        )
    return grouped_scores


def _load_game_batch(
    games: Sequence[PlayGame],
    feature_schema: str,
) -> dict[str, np.ndarray]:
    env = BatchEnv(len(games), 0, feature_schema)
    return env.load_state(
        {
            "base_seed": 0,
            "episodes": [0] * len(games),
            "game_steps": [0] * len(games),
            "feature_schema": feature_schema,
            "games": [game.state() for game in games],
        }
    )


def _finish_batched_rollouts(
    child_states: list[dict[str, Any]],
    lane_games: list[int],
    lane_actions: list[int],
    lane_root_teams: list[int],
    wins_by_game: list[np.ndarray],
    feature_schema: str,
    model_policy: ModelPolicy,
    opponent: Policy,
    device: torch.device,
    max_steps: int,
    deterministic: bool,
) -> None:
    lane_count = len(child_states)
    env = BatchEnv(lane_count, 0, feature_schema)
    batch = env.load_state(
        {
            "base_seed": 0,
            "episodes": [0] * lane_count,
            "game_steps": [0] * lane_count,
            "feature_schema": feature_schema,
            "games": child_states,
        }
    )
    active = np.ones(lane_count, dtype=np.bool_)
    game_indices = np.asarray(lane_games, dtype=np.int64)
    action_indices = np.asarray(lane_actions, dtype=np.int64)
    root_teams = np.asarray(lane_root_teams, dtype=np.int64)
    root_team_tensor = torch.as_tensor(root_teams, device=device)

    for _ in range(max_steps):
        if not bool(active.any()):
            break
        tensors = to_tensors(batch, device)
        model_actions = model_policy.actions(tensors, deterministic=deterministic)
        if opponent is model_policy:
            opponent_actions = model_actions
        else:
            opponent_actions = opponent.actions(tensors, deterministic=deterministic)
        current_teams = torch.remainder(tensors.current_players, 2)
        actions = torch.where(current_teams == root_team_tensor, model_actions, opponent_actions)
        batch = env.step(actions.cpu().numpy(), 0.0)

        winners = batch["winners"]
        finished = active & (winners >= 0)
        won = finished & (winners == root_teams)
        for game_index, action_index in zip(
            game_indices[won], action_indices[won], strict=True
        ):
            wins_by_game[int(game_index)][int(action_index)] += 1.0
        active[finished] = False

    if bool(active.any()):
        leaf_tensors = to_tensors(batch, device)
        leaf_values = model_policy.model.value(
            leaf_tensors.obs,
            public_history=leaf_tensors.public_history,
        )
        leaf_teams = torch.remainder(leaf_tensors.current_players, 2)
        # Values are zero-sum returns for the team whose player is observing
        # the leaf. Flip opponent leaves into the root team's perspective and
        # map the expected [-1, 1] return to win points in [0, 1].
        signed_values = torch.where(
            leaf_teams == root_team_tensor,
            leaf_values,
            -leaf_values,
        )
        leaf_scores = ((signed_values.clamp(-1.0, 1.0) + 1.0) * 0.5).cpu().numpy()
        for game_index, action_index, leaf_score in zip(
            game_indices[active],
            action_indices[active],
            leaf_scores[active],
            strict=True,
        ):
            wins_by_game[int(game_index)][int(action_index)] += float(leaf_score)


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
            f"points={score.wins:g}/{score.rollouts} {score.label}"
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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jackbot import ACTION_SIZE, BELIEF_SIZE, OBS_SIZE, BatchEnv


ACTION_TYPE_START = 53
ACTION_TYPE_ENTER = ACTION_TYPE_START
ACTION_TYPE_MOVE = ACTION_TYPE_START + 1
ACTION_TYPE_SPLIT = ACTION_TYPE_START + 2
ACTION_TYPE_SWAP = ACTION_TYPE_START + 3
ACTION_TYPE_SKIP = ACTION_TYPE_START + 4
ACTION_TYPE_BURN = ACTION_TYPE_START + 5
ACTION_STEPS = 90
ACTION_STARTS_IN_BASE = 92
ACTION_ENDS_IN_HOME = 93
ACTION_CAPTURES = 94
ACTION_BULLDOZER = 95


@dataclass(slots=True)
class TensorBatch:
    obs: torch.Tensor
    action_features: torch.Tensor
    action_offsets: torch.Tensor
    env_ids: torch.Tensor
    current_players: torch.Tensor
    belief_targets: torch.Tensor
    rewards: torch.Tensor
    team_rewards: torch.Tensor
    dones: torch.Tensor
    winners: torch.Tensor
    acting_players: torch.Tensor
    acting_teams: torch.Tensor
    game_lengths: torch.Tensor


def make_env(num_envs: int, seed: int) -> BatchEnv:
    return BatchEnv(num_envs, seed)


def to_tensors(batch: dict[str, np.ndarray], device: torch.device) -> TensorBatch:
    obs = _float(batch["obs"], device)
    action_features = _float(batch["action_features"], device)
    belief_targets = _float(batch["belief_targets"], device)
    if obs.shape[-1] != OBS_SIZE:
        raise ValueError(f"expected obs size {OBS_SIZE}, got {obs.shape[-1]}")
    if action_features.shape[-1] != ACTION_SIZE:
        raise ValueError(f"expected action size {ACTION_SIZE}, got {action_features.shape[-1]}")
    if belief_targets.shape[-1] != BELIEF_SIZE:
        raise ValueError(f"expected belief size {BELIEF_SIZE}, got {belief_targets.shape[-1]}")
    return TensorBatch(
        obs=obs,
        action_features=action_features,
        action_offsets=_long(batch["action_offsets"], device),
        env_ids=_long(batch["env_ids"], device),
        current_players=_long(batch["current_players"], device),
        belief_targets=belief_targets,
        rewards=_float(batch["rewards"], device),
        team_rewards=_float(batch["team_rewards"], device),
        dones=torch.as_tensor(batch["dones"], device=device, dtype=torch.bool),
        winners=_long(batch["winners"], device),
        acting_players=_long(batch["acting_players"], device),
        acting_teams=_long(batch["acting_teams"], device),
        game_lengths=_long(batch["game_lengths"], device),
    )


def random_actions(batch: TensorBatch) -> torch.Tensor:
    counts = batch.action_offsets[1:] - batch.action_offsets[:-1]
    return torch.floor(torch.rand_like(counts, dtype=torch.float32) * counts.float()).long()


def heuristic_actions(batch: TensorBatch) -> torch.Tensor:
    scores = torch.zeros(batch.action_features.shape[0], device=batch.action_features.device)
    features = batch.action_features
    scores += features[:, ACTION_ENDS_IN_HOME] * 100.0
    scores += features[:, ACTION_STARTS_IN_BASE] * 60.0
    scores += features[:, ACTION_CAPTURES] * 20.0
    scores += features[:, ACTION_BULLDOZER] * 5.0
    scores += features[:, ACTION_STEPS].clamp_min(0.0) * 8.0
    scores += features[:, ACTION_TYPE_SKIP] * 4.0
    scores -= features[:, ACTION_TYPE_BURN] * 30.0
    return _segment_argmax(scores, batch.action_offsets)


def _segment_argmax(scores: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    actions = []
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        actions.append(int(torch.argmax(scores[start:end]).item()))
    return torch.tensor(actions, device=scores.device, dtype=torch.long)


def _float(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _long(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.long)

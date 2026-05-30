from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jackbot.training.config import TrainConfig
from jackbot.training.env import TensorBatch, to_tensors
from jackbot.training.model import JackbotNet


@dataclass(slots=True)
class Rollout:
    obs: torch.Tensor
    action_features: torch.Tensor
    action_offsets: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    belief_targets: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    team_rewards: torch.Tensor
    dones: torch.Tensor
    game_lengths: torch.Tensor
    action_starts: torch.Tensor
    action_ends: torch.Tensor


@dataclass(slots=True)
class UpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    belief_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    mean_reward: float
    completed_games: int
    game_length_mean: float


def collect_rollout(
    env,
    model: JackbotNet,
    batch: dict[str, np.ndarray],
    config: TrainConfig,
    device: torch.device,
    shaping_scale: float,
) -> tuple[Rollout, dict[str, np.ndarray]]:
    obs_parts: list[torch.Tensor] = []
    belief_parts: list[torch.Tensor] = []
    action_feature_parts: list[torch.Tensor] = []
    offset_parts: list[torch.Tensor] = []
    actions_parts: list[torch.Tensor] = []
    log_prob_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    team_reward_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    game_length_parts: list[torch.Tensor] = []
    acting_team_parts: list[torch.Tensor] = []

    model.eval()
    for _ in range(config.rollout_len):
        tensors = to_tensors(batch, device)
        with torch.no_grad():
            actions, log_probs, values = model.act(
                tensors.obs,
                tensors.action_features,
                tensors.action_offsets,
                deterministic=False,
            )

        next_batch = env.step(actions.cpu().numpy(), float(shaping_scale))
        next_tensors = to_tensors(next_batch, device)

        obs_parts.append(tensors.obs)
        belief_parts.append(tensors.belief_targets)
        action_feature_parts.append(tensors.action_features)
        offset_parts.append(tensors.action_offsets)
        actions_parts.append(actions)
        log_prob_parts.append(log_probs)
        value_parts.append(values)
        reward_parts.append(next_tensors.rewards)
        team_reward_parts.append(next_tensors.team_rewards)
        done_parts.append(next_tensors.dones)
        game_length_parts.append(next_tensors.game_lengths)
        acting_team_parts.append(next_tensors.acting_teams)

        batch = next_batch

    rollout = _assemble_rollout(
        obs_parts,
        belief_parts,
        action_feature_parts,
        offset_parts,
        actions_parts,
        log_prob_parts,
        value_parts,
        reward_parts,
        team_reward_parts,
        done_parts,
        game_length_parts,
        acting_team_parts,
        config.gamma,
    )
    return rollout, batch


def ppo_update(
    model: JackbotNet,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: TrainConfig,
) -> UpdateStats:
    model.train()
    stats = []
    row_count = rollout.obs.shape[0]
    indices = torch.randperm(row_count, device=rollout.obs.device)

    for _ in range(config.ppo_epochs):
        for batch_indices in indices.split(config.minibatch_size):
            mb = _slice_rollout(rollout, batch_indices)
            new_log_probs, values, belief_logits, entropies, _ = model.evaluate_actions(
                mb.obs,
                mb.action_features,
                mb.action_offsets,
                mb.actions,
            )
            ratio = (new_log_probs - mb.old_log_probs).exp()
            unclipped = ratio * mb.advantages
            clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * mb.advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, mb.returns)
            belief_loss = F.binary_cross_entropy_with_logits(
                belief_logits,
                mb.belief_targets,
            )
            entropy = entropies.mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.belief_coef * belief_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            approx_kl = (mb.old_log_probs - new_log_probs).mean().detach()
            clip_fraction = ((ratio - 1.0).abs() > config.clip).float().mean().detach()
            stats.append(
                (
                    loss.detach(),
                    policy_loss.detach(),
                    value_loss.detach(),
                    belief_loss.detach(),
                    entropy.detach(),
                    approx_kl,
                    clip_fraction,
                )
            )

    stacked = torch.stack([torch.stack(items) for items in stats])
    means = stacked.mean(dim=0).cpu().tolist()
    completed_game_lengths = rollout.game_lengths[rollout.game_lengths > 0]
    completed_games = int(completed_game_lengths.numel())
    game_length_mean = (
        float(completed_game_lengths.float().mean().item()) if completed_games else float("nan")
    )
    return UpdateStats(
        loss=means[0],
        policy_loss=means[1],
        value_loss=means[2],
        belief_loss=means[3],
        entropy=means[4],
        approx_kl=means[5],
        clip_fraction=means[6],
        mean_reward=float(rollout.team_rewards.mean().item()),
        completed_games=completed_games,
        game_length_mean=game_length_mean,
    )


def _assemble_rollout(
    obs_parts: list[torch.Tensor],
    belief_parts: list[torch.Tensor],
    action_feature_parts: list[torch.Tensor],
    offset_parts: list[torch.Tensor],
    actions_parts: list[torch.Tensor],
    log_prob_parts: list[torch.Tensor],
    value_parts: list[torch.Tensor],
    reward_parts: list[torch.Tensor],
    team_reward_parts: list[torch.Tensor],
    done_parts: list[torch.Tensor],
    game_length_parts: list[torch.Tensor],
    acting_team_parts: list[torch.Tensor],
    gamma: float,
) -> Rollout:
    env_count = obs_parts[0].shape[0]
    device = obs_parts[0].device
    obs = torch.cat(obs_parts, dim=0)
    belief_targets = torch.cat(belief_parts, dim=0)
    actions = torch.cat(actions_parts, dim=0)
    old_log_probs = torch.cat(log_prob_parts, dim=0)
    old_values = torch.cat(value_parts, dim=0)
    team_rewards = torch.stack(team_reward_parts, dim=0)
    dones = torch.stack(done_parts, dim=0)
    game_lengths = torch.stack(game_length_parts, dim=0)
    acting_teams = torch.stack(acting_team_parts, dim=0)

    returns = _team_returns(team_rewards, dones, acting_teams, gamma).reshape(-1)
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

    action_features, action_offsets = _concat_action_segments(action_feature_parts, offset_parts)
    starts = action_offsets[:-1]
    ends = action_offsets[1:]
    assert obs.shape[0] == env_count * len(obs_parts)

    return Rollout(
        obs=obs,
        action_features=action_features,
        action_offsets=action_offsets,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        belief_targets=belief_targets,
        returns=returns,
        advantages=advantages,
        team_rewards=team_rewards,
        dones=dones,
        game_lengths=game_lengths,
        action_starts=starts,
        action_ends=ends,
    )


def _team_returns(
    team_rewards: torch.Tensor,
    dones: torch.Tensor,
    acting_teams: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    steps, envs, _ = team_rewards.shape
    future = torch.zeros(envs, 2, device=team_rewards.device)
    returns = torch.zeros(steps, envs, device=team_rewards.device)
    for step in range(steps - 1, -1, -1):
        keep_future = (~dones[step]).float().unsqueeze(-1)
        future = team_rewards[step] + gamma * future * keep_future
        returns[step] = future.gather(1, acting_teams[step].unsqueeze(-1)).squeeze(-1)
    return returns


def _concat_action_segments(
    action_feature_parts: list[torch.Tensor],
    offset_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    action_features = []
    offsets = [0]
    total = 0
    for features, offsets_for_step in zip(action_feature_parts, offset_parts, strict=True):
        action_features.append(features)
        counts = offsets_for_step[1:] - offsets_for_step[:-1]
        for count in counts.tolist():
            total += int(count)
            offsets.append(total)
    return torch.cat(action_features, dim=0), torch.tensor(
        offsets,
        device=action_feature_parts[0].device,
        dtype=torch.long,
    )


def _slice_rollout(rollout: Rollout, rows: torch.Tensor) -> Rollout:
    features = []
    offsets = [0]
    actions = []
    total = 0
    for row in rows.tolist():
        start = int(rollout.action_starts[row].item())
        end = int(rollout.action_ends[row].item())
        features.append(rollout.action_features[start:end])
        total += end - start
        offsets.append(total)
        actions.append(int(rollout.actions[row].item()))

    action_features = torch.cat(features, dim=0)
    action_offsets = torch.tensor(offsets, device=rollout.obs.device, dtype=torch.long)
    row_actions = torch.tensor(actions, device=rollout.obs.device, dtype=torch.long)
    return Rollout(
        obs=rollout.obs[rows],
        action_features=action_features,
        action_offsets=action_offsets,
        actions=row_actions,
        old_log_probs=rollout.old_log_probs[rows],
        old_values=rollout.old_values[rows],
        belief_targets=rollout.belief_targets[rows],
        returns=rollout.returns[rows],
        advantages=rollout.advantages[rows],
        team_rewards=rollout.team_rewards,
        dones=rollout.dones,
        game_lengths=rollout.game_lengths,
        action_starts=action_offsets[:-1],
        action_ends=action_offsets[1:],
    )

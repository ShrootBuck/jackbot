from __future__ import annotations

import torch

from jackbot.training.ppo import _normalize_advantages, _team_gae_returns


def test_team_gae_bootstraps_unfinished_rollout() -> None:
    rewards = torch.tensor([[[2.0, 0.0]]])
    dones = torch.tensor([[False]])
    acting_teams = torch.tensor([[0]])
    values = torch.tensor([[1.0]])
    bootstrap_values = torch.tensor([5.0])
    bootstrap_acting_teams = torch.tensor([0])

    returns, advantages = _team_gae_returns(
        rewards,
        dones,
        acting_teams,
        values,
        bootstrap_values,
        bootstrap_acting_teams,
        gamma=0.5,
        gae_lambda=0.95,
    )

    assert torch.allclose(advantages, torch.tensor([[3.5]]))
    assert torch.allclose(returns, torch.tensor([[4.5]]))


def test_team_gae_masks_terminal_bootstrap() -> None:
    rewards = torch.tensor([[[3.0, -3.0]]])
    dones = torch.tensor([[True]])
    acting_teams = torch.tensor([[0]])
    values = torch.tensor([[1.0]])
    bootstrap_values = torch.tensor([100.0])
    bootstrap_acting_teams = torch.tensor([0])

    returns, advantages = _team_gae_returns(
        rewards,
        dones,
        acting_teams,
        values,
        bootstrap_values,
        bootstrap_acting_teams,
        gamma=0.5,
        gae_lambda=0.95,
    )

    assert torch.allclose(advantages, torch.tensor([[2.0]]))
    assert torch.allclose(returns, torch.tensor([[3.0]]))


def test_advantage_normalization_uses_only_learner_rows() -> None:
    advantages = torch.tensor([1.0, 3.0, 100.0])
    learn_mask = torch.tensor([True, True, False])

    normalized = _normalize_advantages(advantages, learn_mask)

    assert torch.allclose(normalized[:2], torch.tensor([-1.0, 1.0]))

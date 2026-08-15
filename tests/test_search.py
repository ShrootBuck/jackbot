from __future__ import annotations

import pytest
import torch

from jackbot import PlayGame
from jackbot.training.model import JackbotNet
from jackbot.training.policies import ModelPolicy
from jackbot.training.search import rank_action_groups, rank_actions


def test_grouped_search_matches_separate_deterministic_roots() -> None:
    games = [PlayGame(1), PlayGame(2)]
    policy = ModelPolicy("tiny", JackbotNet(hidden_size=32))
    device = torch.device("cpu")

    grouped = rank_action_groups(
        games,
        policy,
        policy,
        device,
        rollouts=2,
        max_steps=10,
        rollout_deterministic=True,
    )
    separate = [
        rank_actions(
            game,
            policy,
            policy,
            device,
            rollouts=2,
            max_steps=10,
            rollout_deterministic=True,
        )
        for game in games
    ]

    for grouped_scores, separate_scores in zip(grouped, separate, strict=True):
        assert [score.action_id for score in grouped_scores] == [
            score.action_id for score in separate_scores
        ]
        assert [score.score for score in grouped_scores] == pytest.approx(
            [score.score for score in separate_scores], abs=1e-6
        )
        assert [score.prior for score in grouped_scores] == pytest.approx(
            [score.prior for score in separate_scores], abs=1e-6
        )


def test_grouped_search_validates_budgets() -> None:
    policy = ModelPolicy("tiny", JackbotNet(hidden_size=32))
    game = PlayGame(1)

    with pytest.raises(ValueError, match="rollouts"):
        rank_action_groups([game], policy, policy, torch.device("cpu"), rollouts=0)
    with pytest.raises(ValueError, match="max_steps"):
        rank_action_groups([game], policy, policy, torch.device("cpu"), max_steps=-1)

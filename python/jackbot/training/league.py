from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from jackbot.training.config import TrainConfig
from jackbot.training.policies import BaselinePolicy, ModelPolicy, Policy, load_model_policy


@dataclass(slots=True)
class LeagueAssignments:
    learner_teams: torch.Tensor
    opponent_indices: torch.Tensor
    self_play: torch.Tensor


class LeaguePool:
    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.opponents: list[Policy] = []
        self.weights: list[float] = []
        self._checkpoint_paths: set[Path] = set()
        self._last_refresh_update = -1
        self.refresh(force=True)

    @property
    def names(self) -> list[str]:
        return [opponent.name for opponent in self.opponents]

    def maybe_refresh(self, update: int) -> None:
        interval = max(1, self.config.league_refresh_interval_updates)
        if update == 0 or update - self._last_refresh_update >= interval:
            self.refresh()
            self._last_refresh_update = update

    def refresh(self, force: bool = False) -> None:
        baseline_specs = list(dict.fromkeys(self.config.league_baselines))
        checkpoint_paths = self._checkpoint_specs()
        if not force and checkpoint_paths == self._checkpoint_paths:
            return

        opponents: list[Policy] = []
        weights: list[float] = []
        for baseline in baseline_specs:
            if baseline:
                opponents.append(BaselinePolicy(baseline))
                weights.append(self.config.league_baseline_weight)

        for path in checkpoint_paths:
            try:
                policy, _ = load_model_policy(path, self.device)
            except (FileNotFoundError, RuntimeError, KeyError, ValueError):
                continue
            opponents.append(policy)
            weights.append(self.config.league_checkpoint_weight)

        self.opponents = opponents
        self.weights = weights
        self._checkpoint_paths = checkpoint_paths

    def sample(self, env_count: int, device: torch.device) -> LeagueAssignments:
        choices: list[int] = [-1]
        weights = [max(0.0, self.config.league_self_play_weight)]
        for index, weight in enumerate(self.weights):
            choices.append(index)
            weights.append(max(0.0, weight))
        if sum(weights) <= 0.0:
            choices = [-1]
            weights = [1.0]

        opponent_indices = random.choices(choices, weights=weights, k=env_count)
        learner_teams = torch.randint(0, 2, (env_count,), device=device, dtype=torch.long)
        opponent_tensor = torch.tensor(opponent_indices, device=device, dtype=torch.long)
        return LeagueAssignments(
            learner_teams=learner_teams,
            opponent_indices=opponent_tensor,
            self_play=opponent_tensor < 0,
        )

    def _checkpoint_specs(self) -> set[Path]:
        paths = {Path(spec) for spec in self.config.league_opponents if spec}
        if self.config.league_auto_checkpoints:
            checkpoint_dir = self.config.checkpoint_dir
            candidates = []
            candidates.extend(checkpoint_dir.glob("epoch_*.pt"))
            best = checkpoint_dir / self.config.best_name
            if best.exists():
                candidates.append(best)
            candidates = sorted(
                {path for path in candidates if path.exists()},
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            paths.update(candidates[: max(0, self.config.league_max_checkpoints)])
        return paths


@torch.no_grad()
def league_actions(
    model_policy: ModelPolicy,
    tensors,
    assignments: LeagueAssignments | None,
    league: LeaguePool | None,
    deterministic_opponents: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model_actions, old_log_probs, values = model_policy.model.act(
        tensors.obs,
        tensors.action_features,
        tensors.action_offsets,
        deterministic=False,
    )
    if assignments is None or league is None:
        return model_actions, old_log_probs, values, torch.ones_like(model_actions, dtype=torch.bool)

    current_teams = tensors.current_players % 2
    learner_turn = current_teams == assignments.learner_teams
    learn_mask = assignments.self_play | learner_turn
    opponent_turn = ~assignments.self_play & ~learner_turn
    actions = model_actions.clone()

    for index, opponent in enumerate(league.opponents):
        mask = opponent_turn & (assignments.opponent_indices == index)
        if bool(mask.any().item()):
            opponent_actions = opponent.actions(tensors, deterministic=deterministic_opponents)
            actions[mask] = opponent_actions[mask]

    return actions, old_log_probs, values, learn_mask

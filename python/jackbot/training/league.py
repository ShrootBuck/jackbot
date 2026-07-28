from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from jackbot.training.checkpoint import checkpoint_sha256
from jackbot.training.config import TrainConfig, union_feature_schemas
from jackbot.training.env import TensorBatch
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
        self._checkpoint_signatures: dict[Path, tuple[int, int] | None] = {}
        self._checkpoint_opponents: dict[Path, ModelPolicy] = {}
        self._baseline_specs: list[str] = []
        self._feature_schema = "base_v1"
        self._initialized = False
        self._last_refresh_update = -1
        self.refresh(force=True)

    @property
    def names(self) -> list[str]:
        return [opponent.name for opponent in self.opponents]

    @property
    def feature_schema(self) -> str:
        return self._feature_schema

    def maybe_refresh(self, update: int) -> None:
        interval = max(1, self.config.league_refresh_interval_updates)
        if update == 0 or update - self._last_refresh_update >= interval:
            self.refresh()
            self._last_refresh_update = update

    def refresh(self, force: bool = False) -> None:
        baseline_specs = list(dict.fromkeys(self.config.league_baselines))
        checkpoint_paths = self._checkpoint_specs()
        checkpoint_signatures = {path: _checkpoint_signature(path) for path in checkpoint_paths}
        if (
            not force
            and baseline_specs == self._baseline_specs
            and checkpoint_signatures == self._checkpoint_signatures
        ):
            return

        opponents: list[Policy] = []
        checkpoint_opponents: dict[Path, ModelPolicy] = {}
        required_paths = {Path(spec) for spec in self.config.league_opponents if spec}
        for baseline in baseline_specs:
            if baseline:
                opponents.append(BaselinePolicy(baseline))

        baseline_count = len(opponents)

        for path in sorted(checkpoint_paths):
            policy = None
            if not force and self._checkpoint_signatures.get(path) == checkpoint_signatures[path]:
                policy = self._checkpoint_opponents.get(path)
            if policy is None:
                try:
                    if (
                        self.config.parent_checkpoint is not None
                        and self.config.parent_checkpoint_sha256 is not None
                        and path.resolve() == Path(self.config.parent_checkpoint).resolve()
                        and checkpoint_sha256(path) != self.config.parent_checkpoint_sha256
                    ):
                        raise ValueError("parent checkpoint hash changed")
                    policy, _ = load_model_policy(path)
                except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
                    if path in required_paths:
                        raise RuntimeError(f"required league opponent failed to load: {path}") from error
                    continue
            checkpoint_opponents[path] = policy
            opponents.append(policy)

        checkpoint_count = len(opponents) - baseline_count
        baseline_weight = self.config.league_baseline_weight / max(1, baseline_count)
        checkpoint_weight = self.config.league_checkpoint_weight / max(1, checkpoint_count)

        loaded_schema = union_feature_schemas(
            *(opponent.feature_schema for opponent in opponents),
        )
        if self._initialized:
            expanded_schema = union_feature_schemas(self._feature_schema, loaded_schema)
            if expanded_schema != self._feature_schema:
                raise RuntimeError(
                    "league refresh introduced a wider feature schema; restart training"
                )
        else:
            self._feature_schema = union_feature_schemas(
                self.config.feature_schema,
                loaded_schema,
            )
            self._initialized = True

        self.opponents = opponents
        self.weights = [baseline_weight] * baseline_count + [checkpoint_weight] * checkpoint_count
        self._baseline_specs = baseline_specs
        self._checkpoint_paths = set(checkpoint_opponents)
        self._checkpoint_signatures = {
            path: checkpoint_signatures[path] for path in checkpoint_opponents
        }
        self._checkpoint_opponents = checkpoint_opponents
        _release_stale_opponents()

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


def _release_stale_opponents() -> None:
    gc.collect()


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
        public_history=tensors.public_history,
        action_consequences=tensors.action_consequences,
        privileged_hands=tensors.belief_targets,
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
            opponent_batch = _slice_tensor_batch(tensors, mask)
            opponent_actions = opponent.actions(
                opponent_batch,
                deterministic=deterministic_opponents,
            )
            actions[mask] = opponent_actions

    return actions, old_log_probs, values, learn_mask


def _checkpoint_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _slice_tensor_batch(batch: TensorBatch, mask: torch.Tensor) -> TensorBatch:
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    starts = batch.action_offsets[rows]
    ends = batch.action_offsets[rows + 1]
    counts = ends - starts
    action_rows = torch.cat(
        [
            torch.arange(start, end, device=batch.action_features.device)
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
        ]
    )
    offsets = torch.cat(
        [
            torch.zeros(1, device=batch.action_offsets.device, dtype=torch.long),
            counts.cumsum(dim=0),
        ]
    )
    env_ids = torch.repeat_interleave(
        torch.arange(rows.numel(), device=batch.env_ids.device),
        counts,
    )
    return TensorBatch(
        obs=batch.obs[rows],
        action_features=batch.action_features[action_rows],
        action_offsets=offsets,
        env_ids=env_ids,
        current_players=batch.current_players[rows],
        belief_targets=batch.belief_targets[rows],
        public_history=batch.public_history[rows],
        action_consequences=batch.action_consequences[action_rows],
        rewards=batch.rewards[rows],
        team_rewards=batch.team_rewards[rows],
        dones=batch.dones[rows],
        winners=batch.winners[rows],
        winning_move_players=batch.winning_move_players[rows],
        acting_players=batch.acting_players[rows],
        acting_teams=batch.acting_teams[rows],
        game_lengths=batch.game_lengths[rows],
    )

from __future__ import annotations

import numpy as np

from jackbot import ACTION_SIZE, BELIEF_SIZE, OBS_SIZE, BatchEnv


def test_batch_reset_shapes() -> None:
    env = BatchEnv(4, 1)
    batch = env.reset(1)

    assert batch["obs"].shape == (4, OBS_SIZE)
    assert batch["belief_targets"].shape == (4, BELIEF_SIZE)
    assert batch["action_features"].shape[1] == ACTION_SIZE
    assert batch["action_offsets"].shape == (5,)
    assert batch["action_offsets"][0] == 0
    assert batch["action_offsets"][-1] == len(batch["action_features"])
    assert len(batch["env_ids"]) == len(batch["action_features"])


def test_batch_step_shapes_and_offsets() -> None:
    env = BatchEnv(4, 1)
    batch = env.reset(1)
    counts = batch["action_offsets"][1:] - batch["action_offsets"][:-1]
    actions = np.zeros(4, dtype=np.int64)
    assert np.all(counts > 0)

    next_batch = env.step(actions, 1.0)

    assert next_batch["obs"].shape == (4, OBS_SIZE)
    assert next_batch["team_rewards"].shape == (4, 2)
    assert next_batch["dones"].shape == (4,)
    assert next_batch["game_lengths"].shape == (4,)
    assert next_batch["acting_teams"].shape == (4,)
    assert next_batch["action_offsets"][-1] == len(next_batch["action_features"])

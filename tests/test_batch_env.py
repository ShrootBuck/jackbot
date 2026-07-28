from __future__ import annotations

import numpy as np

from jackbot import (
    ACTION_CONSEQUENCE_SIZE,
    ACTION_SIZE,
    BELIEF_SIZE,
    OBS_SIZE,
    PUBLIC_HISTORY_SIZE,
    BatchEnv,
    PlayGame,
)


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
    assert batch["winning_move_players"].shape == (4,)
    assert np.all(batch["winning_move_players"] == -1)
    assert batch["public_history"].shape == (4, 0)
    assert batch["action_consequences"].shape == (len(batch["action_features"]), 0)


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
    assert next_batch["winners"].shape == (4,)
    assert next_batch["winning_move_players"].shape == (4,)
    assert next_batch["game_lengths"].shape == (4,)
    assert next_batch["acting_teams"].shape == (4,)
    assert next_batch["action_offsets"][-1] == len(next_batch["action_features"])


def test_play_game_exposes_single_game_batch() -> None:
    game = PlayGame(1)
    batch = game.batch()

    assert batch["obs"].shape == (1, OBS_SIZE)
    assert batch["action_features"].shape[1] == ACTION_SIZE
    assert batch["action_offsets"].tolist()[0] == 0
    assert batch["action_offsets"].tolist()[-1] == len(game.legal_action_labels())
    assert batch["winning_move_players"].tolist() == [-1]
    assert "P1 to act" in game.turn_text()


def test_play_game_state_round_trip_and_action_details() -> None:
    game = PlayGame(3)
    state = game.state()
    restored = PlayGame.from_state(state)

    assert restored.state() == state
    details = restored.legal_action_details()
    assert len(details) == len(restored.legal_action_labels())
    assert {"id", "card", "kind", "label"} <= set(details[0])


def test_enhanced_schema_preserves_base_features_and_adds_public_inputs() -> None:
    base = BatchEnv(2, 7, "base_v1").reset(7)
    enhanced_env = BatchEnv(2, 7, "a1h1")
    enhanced = enhanced_env.reset(7)

    assert np.array_equal(base["obs"], enhanced["obs"])
    assert np.array_equal(base["action_features"], enhanced["action_features"])
    assert enhanced["public_history"].shape == (2, PUBLIC_HISTORY_SIZE)
    assert enhanced["action_consequences"].shape == (
        len(enhanced["action_features"]),
        ACTION_CONSEQUENCE_SIZE,
    )
    assert not np.any(enhanced["public_history"])

    advanced = enhanced_env.step(np.zeros(2, dtype=np.int64), 0.0)
    assert np.any(advanced["public_history"])


def test_play_game_state_round_trip_preserves_public_history() -> None:
    game = PlayGame(4, "h1")
    game.step(0)
    state = game.state()
    restored = PlayGame.from_state(state, "h1")

    assert restored.state()["public_history"] == state["public_history"]
    assert np.array_equal(restored.batch()["public_history"], game.batch()["public_history"])


def test_batch_environment_state_restores_exact_trajectories() -> None:
    first = BatchEnv(3, 19, "a1h1")
    batch = first.reset(19)
    for _ in range(6):
        batch = first.step(np.zeros(3, dtype=np.int64), 0.0)
    state = first.state()
    second = BatchEnv(3, 999, "a1h1")
    restored = second.load_state(state)

    for key in (
        "obs",
        "action_features",
        "action_offsets",
        "belief_targets",
        "public_history",
        "action_consequences",
    ):
        assert np.array_equal(batch[key], restored[key])

    first_next = first.step(np.zeros(3, dtype=np.int64), 0.0)
    second_next = second.step(np.zeros(3, dtype=np.int64), 0.0)
    for key in first_next:
        if isinstance(first_next[key], np.ndarray):
            assert np.array_equal(first_next[key], second_next[key])

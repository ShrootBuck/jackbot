from __future__ import annotations

import torch

from jackbot import PlayGame
from jackbot.training.advisor import (
    AdvisorSession,
    advisor_checkpoint,
    apply_advisor_preset,
    apply_action,
    card_label,
    legal_action_details,
    parse_card,
    parse_cards,
    rank_advisor_actions,
    sample_hidden_state,
)
from jackbot.training.model import JackbotNet
from jackbot.training.policies import ModelPolicy


def test_card_parsing_and_labels() -> None:
    assert card_label(parse_card("AS")) == "AS"
    assert card_label(parse_card("td")) == "10D"
    assert parse_cards("AS, 10D 4H") == [39, 22, 29]


def test_session_json_round_trip() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.hand_sizes[0] = 4
    session.push_history()

    restored = AdvisorSession.from_dict(session.to_dict())

    assert restored.to_dict() == session.to_dict()


def test_hidden_sampling_is_deterministic_and_builds_play_game() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.hand_sizes[0] = 4

    first = sample_hidden_state(session, 123)
    second = sample_hidden_state(session, 123)

    assert first == second
    game = PlayGame.from_state(first)
    assert game.batch()["obs"].shape[0] == 1


def test_apply_visible_action_updates_session() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.hand_sizes[0] = 4
    ace = parse_card("AS")
    details = legal_action_details(session, ace, 5)
    enter = next(detail for detail in details if detail["kind"] == "enter")

    apply_action(session, enter["id"], ace, None, 5)

    assert session.current_player == 1
    assert session.hand_sizes[0] == 3
    assert ace in session.discard
    assert ace not in session.my_hand
    assert any(marble[0] == 0 and marble[2] == "track" for marble in session.marbles)


def test_advisor_ranking_shape() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.hand_sizes[0] = 4
    model = JackbotNet(hidden_size=32)
    policy = ModelPolicy("tiny", model)

    recommendations = rank_advisor_actions(
        session,
        policy,
        torch.device("cpu"),
        samples=1,
        rollouts_per_sample=1,
        max_steps=1,
        seed=7,
    )

    assert recommendations
    assert recommendations[0].rollouts == 1
    assert recommendations[0].label


def test_oracle_preset_sets_heavier_defaults_and_oracle_checkpoint(monkeypatch, tmp_path) -> None:
    promoted = tmp_path / "promoted.pt"
    promoted.write_bytes(b"checkpoint")
    args = type(
        "Args",
        (),
        {
            "preset": "oracle",
            "checkpoint": None,
            "samples": None,
            "rollouts_per_sample": None,
            "max_steps": None,
        },
    )()

    monkeypatch.setattr(
        "jackbot.training.advisor.default_oracle_checkpoint",
        lambda: promoted,
    )

    apply_advisor_preset(args)

    assert args.samples == 64
    assert args.rollouts_per_sample == 4
    assert args.max_steps == 700
    assert advisor_checkpoint(args) == promoted


def test_advisor_manual_overrides_survive_preset() -> None:
    args = type(
        "Args",
        (),
        {
            "preset": "oracle",
            "checkpoint": None,
            "samples": 8,
            "rollouts_per_sample": None,
            "max_steps": 50,
        },
    )()

    apply_advisor_preset(args)

    assert args.samples == 8
    assert args.rollouts_per_sample == 4
    assert args.max_steps == 50

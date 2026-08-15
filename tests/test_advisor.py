from __future__ import annotations

import pytest
import torch

from jackbot import PlayGame
from jackbot.training.advisor import (
    AdvisorSession,
    QuitAdvisor,
    advisor_checkpoint,
    apply_advisor_preset,
    apply_action,
    card_label,
    effective_search_samples,
    ensure_current_hand,
    legal_action_details,
    load_session,
    parse_card,
    parse_cards,
    prompt_startup,
    rank_advisor_actions,
    sample_hidden_state,
    validate_session_card_counts,
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


def test_startup_prompt_accepts_quit(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _label: "q")

    with pytest.raises(QuitAdvisor):
        prompt_startup("start: ")


def test_updated_hand_is_saved_before_search(monkeypatch, tmp_path) -> None:
    path = tmp_path / "session.json"
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    monkeypatch.setattr("builtins.input", lambda _label: "2S 3D 5H 9C")

    ensure_current_hand(session, path)

    assert load_session(path).my_hand == parse_cards("2S 3D 5H 9C")


def test_old_session_infers_unknown_discard_count() -> None:
    session = AdvisorSession.from_dict(
        {
            "user_seat": 0,
            "deck_remaining": 36,
            "discard": [],
            "hand_sizes": [4, 3, 4, 4],
        }
    )

    assert session.unknown_discard_count == 1
    validate_session_card_counts(session)


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


def test_advisor_ranking_is_reproducible_without_consuming_global_rng() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.hand_sizes[0] = 4
    policy = ModelPolicy("tiny", JackbotNet(hidden_size=32))
    rng_before = torch.get_rng_state()

    first = rank_advisor_actions(
        session, policy, torch.device("cpu"), 2, 1, 2, seed=71
    )
    second = rank_advisor_actions(
        session, policy, torch.device("cpu"), 2, 1, 2, seed=71
    )

    assert first == second
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_forced_advisor_action_skips_rollouts() -> None:
    session = AdvisorSession.new(0)
    session.hand_sizes[0] = 0
    session.deck_remaining = 40
    model = JackbotNet(hidden_size=32)
    policy = ModelPolicy("tiny", model)

    recommendations = rank_advisor_actions(
        session,
        policy,
        torch.device("cpu"),
        samples=2,
        rollouts_per_sample=3,
        max_steps=20,
        seed=7,
    )

    assert len(recommendations) == 1
    assert recommendations[0].rollouts == 0
    assert recommendations[0].score == 0.5


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

    assert args.samples == 128
    assert args.rollouts_per_sample == 2
    assert args.max_steps == 64
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
    assert args.rollouts_per_sample == 2
    assert args.max_steps == 50


def test_god_preset_spends_more_test_time_compute() -> None:
    args = type(
        "Args",
        (),
        {
            "preset": "god",
            "checkpoint": None,
            "samples": None,
            "rollouts_per_sample": None,
            "max_steps": None,
        },
    )()

    apply_advisor_preset(args)

    assert args.samples == 512
    assert args.rollouts_per_sample == 2
    assert args.max_steps == 128


def test_search_samples_adapt_only_to_pathological_branching() -> None:
    assert effective_search_samples(512, 2, 1) == 1
    assert effective_search_samples(512, 2, 30) == 512
    assert effective_search_samples(512, 2, 152) == 107


def test_unknown_skip_discard_is_removed_from_hidden_state() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("QD AS 7H 4C")
    session.hand_sizes[0] = 4
    queen = parse_card("QD")
    details = legal_action_details(session, queen, 9)
    skip = next(detail for detail in details if detail["kind"] == "skip")

    apply_action(session, skip["id"], queen, None, 9)
    state = sample_hidden_state(session, 10)

    assert session.unknown_discard_count == 1
    assert session.public_history[-1]["kind"] == "skip"
    assert session.public_history[-1]["forced_discard"]["card"] is None
    assert len(state["discard"]) == len(session.discard) + 1
    assert sum(len(hand) for hand in state["hands"]) + len(state["deck"]) + len(
        state["discard"]
    ) == 52


def test_advisor_reshuffles_discard_before_next_shoe() -> None:
    session = AdvisorSession.new(0)
    ace = parse_card("AS")
    session.current_player = 0
    session.deal_round_index = 0
    session.deck_remaining = 0
    session.discard = [card for card in range(52) if card != ace]
    session.hand_sizes = [1, 0, 0, 0]
    session.my_hand = [ace]
    validate_session_card_counts(session)
    details = legal_action_details(session, ace, 11)
    enter = next(detail for detail in details if detail["kind"] == "enter")

    apply_action(session, enter["id"], ace, None, 11)

    assert session.discard == []
    assert session.unknown_discard_count == 0
    assert session.hand_sizes == [4, 4, 4, 4]
    assert session.deck_remaining == 36
    validate_session_card_counts(session)


def test_observed_burn_does_not_depend_on_random_hidden_filler() -> None:
    session = AdvisorSession.new(0)
    session.my_hand = parse_cards("AS KD 7H 4C")
    session.current_player = 1
    session.marbles[4] = [1, 0, "track", 10]
    card = parse_card("JC")

    details = legal_action_details(session, card, 1)

    assert any(detail["kind"] == "burn" for detail in details)
    burn = next(detail for detail in details if detail["kind"] == "burn")
    assert len(burn["_synthetic_hand"]) == 4
    apply_action(
        session,
        burn["id"],
        card,
        None,
        burn["_synthetic_seed"],
        burn["_synthetic_hand"],
    )
    assert session.hand_sizes[1] == 3
    assert card in session.discard


def test_skipping_user_requires_and_reports_the_known_discard() -> None:
    session = AdvisorSession.new(1)
    session.my_hand = parse_cards("AS KD 7H 4C")
    queen = parse_card("QD")
    details = legal_action_details(session, queen, 17)
    skip = next(detail for detail in details if detail["kind"] == "skip")

    with pytest.raises(ValueError, match="known hand"):
        apply_action(session, skip["id"], queen, None, skip["_synthetic_seed"])

    discarded = parse_card("7H")
    outcome = apply_action(
        session,
        skip["id"],
        queen,
        discarded,
        skip["_synthetic_seed"],
    )

    assert "discarded 7H" in outcome
    assert discarded not in session.my_hand
    assert session.unknown_discard_count == 0

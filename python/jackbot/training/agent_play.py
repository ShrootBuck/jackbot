from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from jackbot.training.advisor import (
    ADVISOR_PRESETS,
    AdvisorRecommendation,
    AdvisorSession,
    apply_action,
    card_label,
    cards_label,
    effective_search_samples,
    legal_action_details,
    load_session,
    parse_card,
    parse_cards,
    parse_seat,
    player_label,
    print_recommendations,
    print_session_summary,
    rank_advisor_actions,
    save_session,
    stable_detail_label,
    validate_session_card_counts,
)
from jackbot.training.policies import load_model_policy
from jackbot.training.runtime import training_device


DEFAULT_SESSION = Path(".jackbot-agent.json")
DEFAULT_CHECKPOINT = Path("checkpoints/champions/promoted.pt")


def session_payload(session: AdvisorSession) -> dict[str, Any]:
    board: dict[str, dict[str, str]] = {}
    for owner, index, kind, value in session.marbles:
        if kind == "base":
            location = "base"
        elif kind == "track":
            location = f"d{value}"
        elif kind == "home":
            location = f"home{value + 1}"
        else:
            location = f"{kind}{value}"
        board.setdefault(player_label(owner), {})[f"P{owner + 1}m{index + 1}"] = location
    return {
        "turn": session.turn_index,
        "current_player": player_label(session.current_player),
        "user_seat": player_label(session.user_seat),
        "hand": [card_label(card) for card in session.my_hand],
        "hand_sizes": {
            player_label(player): size for player, size in enumerate(session.hand_sizes)
        },
        "board": board,
        "winner": None if session.winner is None else f"team_{int(session.winner) + 1}",
    }


def detail_payload(index: int, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "move": index,
        "id": int(detail["id"]),
        "card": None
        if detail.get("card") is None
        else card_label(int(detail["card"])),
        "label": stable_detail_label(detail),
    }


def recommendation_payload(
    rank: int,
    recommendation: AdvisorRecommendation,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": recommendation.action_id,
        "label": recommendation.label,
        "score": recommendation.score,
        "prior": recommendation.prior,
        "points": recommendation.wins,
        "rollouts": recommendation.rollouts,
    }


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def load_checked_session(path: Path) -> AdvisorSession:
    if not path.is_file():
        raise ValueError(
            f"no live session at {path}; run `./scripts/agent_play.sh new "
            "--seat P1 --hand \"AS 2D 3H 4S\"` first"
        )
    session = load_session(path)
    validate_session_card_counts(session)
    return session


def save_checked_session(path: Path, session: AdvisorSession) -> None:
    validate_session_card_counts(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_session(path, session)


def action_details(
    session: AdvisorSession,
    raw_card: str | None,
    seed: int,
) -> tuple[int | None, list[dict[str, Any]]]:
    player = session.current_player
    if session.winner is not None:
        raise ValueError("the saved game is already over")
    if session.hand_sizes[player] == 0:
        if raw_card is not None:
            raise ValueError(f"{player_label(player)} has no cards; omit --card for the pass")
        card_id = None
    elif raw_card is None:
        if player != session.user_seat:
            raise ValueError(
                f"{player_label(player)} is acting; supply the revealed card with --card"
            )
        card_id = None
    else:
        card_id = parse_card(raw_card)

    details = legal_action_details(session, card_id, seed + session.turn_index)
    if card_id is not None:
        details = [detail for detail in details if detail.get("card") == card_id]
    if not details:
        card_text = "" if card_id is None else f" for {card_label(card_id)}"
        raise ValueError(
            f"no legal visible moves{card_text}; run status and check the synced game"
        )
    return card_id, details


def command_new(args: argparse.Namespace) -> None:
    if args.session.exists() and not args.force:
        raise ValueError(
            f"{args.session} already exists; use status, or new --force to replace it"
        )
    hand = parse_cards(args.hand)
    if len(hand) != 4:
        raise ValueError("a new game starts with exactly four cards in each hand")
    session = AdvisorSession.new(parse_seat(args.seat))
    session.my_hand = hand
    save_checked_session(args.session, session)
    if args.json:
        emit_json({"ok": True, "command": "new", "state": session_payload(session)})
    else:
        print(
            f"New game: you are {player_label(session.user_seat)} with "
            f"{cards_label(session.my_hand)}."
        )
        print(f"Saved to {args.session}. Next: {player_label(session.current_player)}.")


def command_status(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    if args.json:
        emit_json({"ok": True, "command": "status", "state": session_payload(session)})
    else:
        print_session_summary(session)
        print(f"Session: {args.session}")


def command_hand(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    hand = parse_cards(args.cards)
    expected = session.hand_sizes[session.user_seat]
    if len(hand) != expected:
        raise ValueError(
            f"{player_label(session.user_seat)} must have {expected} cards now, "
            f"not {len(hand)}"
        )
    duplicates = set(hand) & set(session.discard)
    if duplicates:
        labels = " ".join(sorted(card_label(card) for card in duplicates))
        raise ValueError(f"known hand duplicates discarded cards: {labels}")
    session.push_history()
    session.my_hand = hand
    save_checked_session(args.session, session)
    if args.json:
        emit_json({"ok": True, "command": "hand", "state": session_payload(session)})
    else:
        print(f"Hand set to {cards_label(hand)}. Saved to {args.session}.")


def command_moves(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    _, details = action_details(session, args.card, args.seed)
    moves = [detail_payload(index, detail) for index, detail in enumerate(details, 1)]
    if args.json:
        emit_json(
            {
                "ok": True,
                "command": "moves",
                "player": player_label(session.current_player),
                "moves": moves,
                "state": session_payload(session),
            }
        )
    else:
        print(f"{player_label(session.current_player)} legal visible moves:")
        for move in moves:
            print(f"  {move['move']:>2}. id={move['id']}: {move['label']}")
        print("Record one with: ./scripts/agent_play.sh observe --move N [--card CARD]")


def select_detail(args: argparse.Namespace, details: list[dict[str, Any]]) -> dict[str, Any]:
    if args.move is not None:
        if not 1 <= args.move <= len(details):
            raise ValueError(f"--move must be between 1 and {len(details)}")
        return details[args.move - 1]
    matches = [detail for detail in details if int(detail["id"]) == args.action_id]
    if not matches:
        legal_ids = ", ".join(str(detail["id"]) for detail in details)
        raise ValueError(f"action id {args.action_id} is not legal; legal ids: {legal_ids}")
    return matches[0]


def command_observe(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    card_id, details = action_details(session, args.card, args.seed)
    detail = select_detail(args, details)
    forced_discard = None if args.discard is None else parse_card(args.discard)
    used_card = detail.get("card")
    if used_card is not None:
        used_card = int(used_card)
    elif card_id is not None:
        used_card = card_id
    outcome = apply_action(
        session,
        int(detail["id"]),
        used_card,
        forced_discard,
        int(detail.get("_synthetic_seed", args.seed + session.turn_index)),
        detail.get("_synthetic_hand"),
    )
    save_checked_session(args.session, session)
    payload = {
        "ok": True,
        "command": "observe",
        "applied": stable_detail_label(detail),
        "outcome": outcome,
        "state": session_payload(session),
    }
    if args.json:
        emit_json(payload)
    else:
        print(f"Applied: {payload['applied']}")
        if outcome:
            print(outcome)
        if session.winner is None:
            print(f"Saved. Next: {player_label(session.current_player)}.")
        else:
            print(f"Saved. Game over: team {int(session.winner) + 1}.")


def command_undo(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    for _ in range(args.steps):
        if not session.undo():
            raise ValueError("not enough saved moves to undo")
    save_checked_session(args.session, session)
    if args.json:
        emit_json({"ok": True, "command": "undo", "state": session_payload(session)})
    else:
        print(
            f"Undid {args.steps} move(s). Next: "
            f"{player_label(session.current_player)}. Saved."
        )


def command_advise(args: argparse.Namespace) -> None:
    session = load_checked_session(args.session)
    if session.current_player != session.user_seat:
        raise ValueError(
            f"it is {player_label(session.current_player)}'s turn; record that move first"
        )
    expected = session.hand_sizes[session.user_seat]
    if len(session.my_hand) != expected:
        raise ValueError(
            f"your saved hand has {len(session.my_hand)} cards but should have {expected}; "
            "run the hand command"
        )
    _, details = action_details(session, None, args.seed)
    preset = ADVISOR_PRESETS[args.preset]
    samples = preset.samples if args.samples is None else args.samples
    rollouts_per_sample = (
        preset.rollouts_per_sample
        if args.rollouts_per_sample is None
        else args.rollouts_per_sample
    )
    max_steps = preset.max_steps if args.max_steps is None else args.max_steps
    samples = effective_search_samples(samples, rollouts_per_sample, len(details))

    if not args.checkpoint.is_file():
        raise ValueError(f"checkpoint not found: {args.checkpoint}")
    model_policy, _ = load_model_policy(args.checkpoint)
    started = time.perf_counter()
    recommendations = rank_advisor_actions(
        session,
        model_policy,
        training_device(),
        samples=samples,
        rollouts_per_sample=rollouts_per_sample,
        max_steps=max_steps,
        seed=args.seed + session.turn_index * 1_000_003,
    )
    elapsed = time.perf_counter() - started
    if not recommendations:
        raise ValueError("the model returned no legal recommendations")

    applied: dict[str, Any] | None = None
    if args.apply_rank is not None:
        if not 1 <= args.apply_rank <= len(recommendations):
            raise ValueError(f"--apply-rank must be between 1 and {len(recommendations)}")
        chosen = recommendations[args.apply_rank - 1]
        detail_by_id = {int(detail["id"]): detail for detail in details}
        detail = detail_by_id[chosen.action_id]
        outcome = apply_action(
            session,
            chosen.action_id,
            detail.get("card"),
            None,
            int(detail.get("_synthetic_seed", args.seed + session.turn_index)),
            detail.get("_synthetic_hand"),
        )
        save_checked_session(args.session, session)
        applied = {"rank": args.apply_rank, "id": chosen.action_id, "outcome": outcome}

    top = min(args.top, len(recommendations))
    payload = {
        "ok": True,
        "command": "advise",
        "checkpoint": str(args.checkpoint),
        "preset": args.preset,
        "elapsed_seconds": elapsed,
        "samples": samples,
        "rollouts_per_sample": rollouts_per_sample,
        "max_steps": max_steps,
        "recommendations": [
            recommendation_payload(rank, recommendation)
            for rank, recommendation in enumerate(recommendations[:top], 1)
        ],
        "applied": applied,
        "state": session_payload(session),
    }
    if args.json:
        emit_json(payload)
    else:
        print(
            f"Advice for {player_label(session.user_seat)} "
            f"({samples * rollouts_per_sample} rollouts/move, "
            f"{max_steps}-ply, {elapsed:.1f}s):"
        )
        print_recommendations(recommendations, top)
        if applied is None:
            print(
                "After making a move: ./scripts/agent_play.sh observe "
                "--id ACTION_ID"
            )
        else:
            print(f"Applied rank {applied['rank']} and saved. {applied['outcome']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-interactive, session-backed Jackbot interface for coding agents.",
    )
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preset", choices=sorted(ADVISOR_PRESETS), default="god")
    parser.add_argument("--seed", type=int, default=400_000)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new = subparsers.add_parser("new", aliases=["init"], help="start a saved game")
    new.add_argument("--seat", required=True, help="P1, P2, P3, or P4")
    new.add_argument("--hand", required=True, help='quoted cards, e.g. "AS KD 7H 4C"')
    new.add_argument("--force", action="store_true", help="replace an existing session")
    new.set_defaults(handler=command_new)

    status = subparsers.add_parser("status", aliases=["show"], help="show saved game state")
    status.set_defaults(handler=command_status)

    hand = subparsers.add_parser("hand", aliases=["deal"], help="set your known current hand")
    hand.add_argument("cards", help='quoted cards, e.g. "AS KD 7H 4C"')
    hand.set_defaults(handler=command_hand)

    moves = subparsers.add_parser(
        "moves",
        aliases=["legal"],
        help="list legal visible interpretations without changing state",
    )
    moves.add_argument("--card", help="revealed card; required for an opponent turn")
    moves.set_defaults(handler=command_moves)

    observe = subparsers.add_parser(
        "observe",
        aliases=["apply", "play"],
        help="record one listed move, save, and exit",
    )
    observe.add_argument("--card", help="revealed card; required for an opponent turn")
    selector = observe.add_mutually_exclusive_group(required=True)
    selector.add_argument("--move", type=int, help="1-based number from the moves command")
    selector.add_argument("--id", dest="action_id", type=int, help="stable legal action id")
    observe.add_argument(
        "--discard",
        help="card discarded by the skipped player; omit when it was hidden",
    )
    observe.set_defaults(handler=command_observe)

    undo = subparsers.add_parser("undo", help="undo saved visible moves")
    undo.add_argument("--steps", type=int, default=1)
    undo.set_defaults(handler=command_undo)

    advise = subparsers.add_parser("advise", help="rank your legal moves and exit")
    advise.add_argument("--top", type=int, default=5)
    advise.add_argument("--samples", type=int)
    advise.add_argument("--rollouts-per-sample", type=int)
    advise.add_argument("--max-steps", type=int)
    advise.add_argument(
        "--apply-rank",
        type=int,
        help="also record this ranked recommendation after search",
    )
    advise.set_defaults(handler=command_advise)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "steps", 1) <= 0:
        parser.error("--steps must be positive")
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        if args.json:
            emit_json({"ok": False, "error": str(error)})
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

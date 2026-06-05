from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from jackbot import PlayGame
from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.config import default_oracle_checkpoint
from jackbot.training.policies import ModelPolicy, load_model_policy
from jackbot.training.runtime import training_device
from jackbot.training.search import rank_actions


NUM_PLAYERS = 4
MARBLES_PER_PLAYER = 4
HAND_CYCLE = (4, 4, 5)
FULL_DECK = list(range(52))
RANK_LABELS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RANK_IDS = {
    "A": 0,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "T": 9,
    "J": 10,
    "Q": 11,
    "K": 12,
}
SUIT_OFFSETS = {"C": 0, "D": 13, "H": 26, "S": 39}
SUIT_LABELS = ("C", "D", "H", "S")


class RestartTurn(Exception):
    pass


class QuitAdvisor(Exception):
    pass


@dataclass(slots=True)
class AdvisorRecommendation:
    action_id: int
    label: str
    prior: float
    score: float
    wins: float
    rollouts: int


@dataclass(frozen=True, slots=True)
class AdvisorPreset:
    samples: int
    rollouts_per_sample: int
    max_steps: int


ADVISOR_PRESETS = {
    "default": AdvisorPreset(samples=16, rollouts_per_sample=2, max_steps=400),
    "oracle": AdvisorPreset(samples=64, rollouts_per_sample=4, max_steps=700),
}


@dataclass(slots=True)
class AdvisorSession:
    user_seat: int
    current_player: int = 0
    turn_index: int = 0
    deal_round_index: int = 1
    deck_remaining: int = 36
    discard: list[int] = field(default_factory=list)
    hand_sizes: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    my_hand: list[int] = field(default_factory=list)
    marbles: list[list[Any]] = field(default_factory=lambda: initial_marbles())
    winner: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls, user_seat: int) -> AdvisorSession:
        return cls(user_seat=user_seat)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdvisorSession:
        return cls(
            user_seat=int(data["user_seat"]),
            current_player=int(data.get("current_player", 0)),
            turn_index=int(data.get("turn_index", 0)),
            deal_round_index=int(data.get("deal_round_index", 1)),
            deck_remaining=int(data.get("deck_remaining", 36)),
            discard=[int(card) for card in data.get("discard", [])],
            hand_sizes=[int(size) for size in data.get("hand_sizes", [4, 4, 4, 4])],
            my_hand=[int(card) for card in data.get("my_hand", [])],
            marbles=[list(marble) for marble in data.get("marbles", initial_marbles())],
            winner=data.get("winner"),
            history=[dict(item) for item in data.get("history", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.core_dict()
        data["history"] = self.history
        return data

    def core_dict(self) -> dict[str, Any]:
        return {
            "user_seat": self.user_seat,
            "current_player": self.current_player,
            "turn_index": self.turn_index,
            "deal_round_index": self.deal_round_index,
            "deck_remaining": self.deck_remaining,
            "discard": list(self.discard),
            "hand_sizes": list(self.hand_sizes),
            "my_hand": list(self.my_hand),
            "marbles": [list(marble) for marble in self.marbles],
            "winner": self.winner,
        }

    def push_history(self) -> None:
        self.history.append(self.core_dict())
        self.history = self.history[-100:]

    def undo(self) -> bool:
        if not self.history:
            return False
        restored = AdvisorSession.from_dict(self.history.pop())
        history = self.history
        self.user_seat = restored.user_seat
        self.current_player = restored.current_player
        self.turn_index = restored.turn_index
        self.deal_round_index = restored.deal_round_index
        self.deck_remaining = restored.deck_remaining
        self.discard = restored.discard
        self.hand_sizes = restored.hand_sizes
        self.my_hand = restored.my_hand
        self.marbles = restored.marbles
        self.winner = restored.winner
        self.history = history
        return True


def initial_marbles() -> list[list[Any]]:
    return [
        [owner, index, "base", 0]
        for owner in range(NUM_PLAYERS)
        for index in range(MARBLES_PER_PLAYER)
    ]


def parse_card(text: str) -> int:
    raw = text.strip().upper()
    if not raw:
        raise ValueError("empty card")
    suit = raw[-1]
    rank = raw[:-1]
    if suit not in SUIT_OFFSETS or rank not in RANK_IDS:
        raise ValueError(f"bad card {text!r}; use AS, 10H, TD, etc.")
    return SUIT_OFFSETS[suit] + RANK_IDS[rank]


def parse_cards(text: str) -> list[int]:
    cards = []
    for chunk in text.replace(",", " ").split():
        cards.append(parse_card(chunk))
    if len(set(cards)) != len(cards):
        raise ValueError("hand contains a duplicate card")
    return cards


def card_label(card_id: int) -> str:
    return f"{RANK_LABELS[card_id % 13]}{SUIT_LABELS[card_id // 13]}"


def cards_label(cards: list[int]) -> str:
    if not cards:
        return "(empty)"
    return " ".join(card_label(card) for card in cards)


def player_label(player: int) -> str:
    return f"P{player + 1}"


def stable_marble_label(owner: int, index: int) -> str:
    return f"P{owner + 1}m{index + 1}"


def team_label(team: int) -> str:
    return "P1 + P3" if team == 0 else "P2 + P4"


def session_engine_state(
    session: AdvisorSession,
    hands: list[list[int]],
    deck: list[int],
    rng_state: int,
) -> dict[str, Any]:
    return {
        "rng_state": rng_state,
        "deck": deck,
        "discard": list(session.discard),
        "hands": hands,
        "marbles": [tuple(marble) for marble in session.marbles],
        "current_player": session.current_player,
        "turn_index": session.turn_index,
        "deal_round_index": session.deal_round_index,
        "winner": session.winner,
    }


def sample_hidden_state(session: AdvisorSession, seed: int) -> dict[str, Any]:
    if len(session.my_hand) != session.hand_sizes[session.user_seat]:
        raise ValueError(
            f"{player_label(session.user_seat)} hand has {len(session.my_hand)} cards, "
            f"but session says {session.hand_sizes[session.user_seat]}"
        )

    rng = random.Random(seed)
    known_cards = list(session.discard) + list(session.my_hand)
    assert_unique_known_cards(known_cards)
    unknown = [card for card in FULL_DECK if card not in set(known_cards)]
    rng.shuffle(unknown)

    hands: list[list[int]] = []
    for player in range(NUM_PLAYERS):
        if player == session.user_seat:
            hands.append(list(session.my_hand))
            continue
        hands.append(draw_cards(unknown, session.hand_sizes[player]))

    deck = list(unknown)
    rng.shuffle(deck)
    return session_engine_state(session, hands, deck, seed)


def synthetic_state_for_action(
    session: AdvisorSession,
    card_id: int | None,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    reserved = list(session.discard) + list(session.my_hand)
    if card_id is not None and card_id not in reserved:
        reserved.append(card_id)
    assert_unique_known_cards(reserved)
    unknown = [card for card in FULL_DECK if card not in set(reserved)]
    rng.shuffle(unknown)

    hands: list[list[int]] = []
    for player in range(NUM_PLAYERS):
        count = session.hand_sizes[player]
        if player == session.user_seat and session.my_hand:
            hand = list(session.my_hand)
            if player == session.current_player and card_id is not None and card_id not in hand:
                hand = [card_id] + draw_cards(unknown, max(0, count - 1))
            elif len(hand) < count:
                hand = hand + draw_cards(unknown, count - len(hand))
            else:
                hand = hand[:count]
        elif player == session.current_player and card_id is not None:
            hand = [card_id] + draw_cards(unknown, max(0, count - 1))
        else:
            hand = draw_cards(unknown, count)
        hands.append(hand)

    deck = list(unknown)
    rng.shuffle(deck)
    return session_engine_state(session, hands, deck, seed)


def draw_cards(deck: list[int], count: int) -> list[int]:
    if count < 0:
        raise ValueError("card count cannot be negative")
    if len(deck) < count:
        raise ValueError("not enough unknown cards left to sample a hidden state")
    drawn = deck[:count]
    del deck[:count]
    return drawn


def assert_unique_known_cards(cards: list[int]) -> None:
    if len(set(cards)) != len(cards):
        labels = " ".join(card_label(card) for card in cards)
        raise ValueError(f"known cards contain a duplicate: {labels}")


@torch.no_grad()
def rank_advisor_actions(
    session: AdvisorSession,
    model_policy: ModelPolicy,
    device: torch.device,
    samples: int,
    rollouts_per_sample: int,
    max_steps: int,
    seed: int,
) -> list[AdvisorRecommendation]:
    aggregates: dict[int, dict[str, Any]] = {}
    for sample_index in range(samples):
        state = sample_hidden_state(session, seed + sample_index * 100_003)
        game = PlayGame.from_state(state)
        details = {int(item["id"]): dict(item) for item in game.legal_action_details()}
        scores = rank_actions(
            game,
            model_policy,
            model_policy,
            device,
            rollouts=rollouts_per_sample,
            max_steps=max_steps,
            rollout_deterministic=False,
        )
        for score in scores:
            aggregate = aggregates.setdefault(
                score.action_id,
                {
                    "label": stable_detail_label(details[score.action_id]),
                    "prior_sum": 0.0,
                    "prior_count": 0,
                    "wins": 0.0,
                    "rollouts": 0,
                },
            )
            aggregate["prior_sum"] += score.prior
            aggregate["prior_count"] += 1
            aggregate["wins"] += score.wins
            aggregate["rollouts"] += score.rollouts

    recommendations = [
        AdvisorRecommendation(
            action_id=action_id,
            label=data["label"],
            prior=data["prior_sum"] / max(1, data["prior_count"]),
            score=data["wins"] / max(1, data["rollouts"]),
            wins=data["wins"],
            rollouts=data["rollouts"],
        )
        for action_id, data in aggregates.items()
    ]
    return sorted(recommendations, key=lambda item: (item.score, item.prior), reverse=True)


def legal_action_details(session: AdvisorSession, card_id: int | None, seed: int) -> list[dict[str, Any]]:
    game = PlayGame.from_state(synthetic_state_for_action(session, card_id, seed))
    return [dict(item) for item in game.legal_action_details()]


def apply_action(
    session: AdvisorSession,
    action_id: int,
    card_id: int | None,
    forced_discard: int | None,
    seed: int,
) -> str:
    game = PlayGame.from_state(synthetic_state_for_action(session, card_id, seed))
    details = {int(item["id"]): dict(item) for item in game.legal_action_details()}
    if action_id not in details:
        raise ValueError(f"action {action_id} is not legal in the synced state")
    detail = details[action_id]
    if card_id is not None and detail.get("card") != card_id:
        raise ValueError("selected action does not use the revealed card")

    acting_player = session.current_player
    next_player = (acting_player + 1) % NUM_PLAYERS
    session.push_history()
    outcome_text = game.step(action_id)
    state = game.state()
    session.marbles = [list(marble) for marble in state["marbles"]]
    session.current_player = int(state["current_player"])
    session.turn_index = int(state["turn_index"])
    session.winner = state["winner"]

    played_card = detail.get("card")
    if played_card is not None:
        played_card = int(played_card)
        session.discard.append(played_card)
        if session.hand_sizes[acting_player] > 0:
            session.hand_sizes[acting_player] -= 1
        if acting_player == session.user_seat:
            remove_known_hand_card(session, played_card)

    if detail["kind"] == "skip" and session.hand_sizes[next_player] > 0:
        session.hand_sizes[next_player] -= 1
        if forced_discard is not None:
            session.discard.append(forced_discard)
            if next_player == session.user_seat:
                remove_known_hand_card(session, forced_discard)

    if session.winner is None and all(size == 0 for size in session.hand_sizes):
        deal_size = HAND_CYCLE[session.deal_round_index % len(HAND_CYCLE)]
        session.deal_round_index = (session.deal_round_index + 1) % len(HAND_CYCLE)
        session.hand_sizes = [deal_size] * NUM_PLAYERS
        session.deck_remaining = max(0, session.deck_remaining - deal_size * NUM_PLAYERS)
        session.my_hand = []

    return outcome_text


def remove_known_hand_card(session: AdvisorSession, card_id: int) -> None:
    try:
        session.my_hand.remove(card_id)
    except ValueError:
        pass


def save_session(path: Path, session: AdvisorSession) -> None:
    path.write_text(json.dumps(session.to_dict(), indent=2) + "\n")


def load_session(path: Path) -> AdvisorSession:
    return AdvisorSession.from_dict(json.loads(path.read_text()))


def parse_seat(raw: str) -> int:
    text = raw.strip().upper()
    if text.startswith("P"):
        text = text[1:]
    seat = int(text) - 1
    if not 0 <= seat < NUM_PLAYERS:
        raise ValueError("seat must be P1, P2, P3, or P4")
    return seat


def print_session_summary(session: AdvisorSession) -> None:
    print()
    print(f"Turn {session.turn_index} | {player_label(session.current_player)} to act")
    print(
        "Hand sizes: "
        + " ".join(
            f"{player_label(player)}={session.hand_sizes[player]}" for player in range(NUM_PLAYERS)
        )
    )
    print(f"You: {player_label(session.user_seat)} ({team_label(session.user_seat % 2)})")
    print(f"Your hand: {cards_label(session.my_hand)}")
    print("Board:")
    for player in range(NUM_PLAYERS):
        parts = []
        for owner, index, kind, value in session.marbles:
            if owner == player:
                parts.append(f"{stable_marble_label(owner, index)}={location_label(kind, value)}")
        print(f"  {player_label(player)}: {'  '.join(parts)}")


def location_label(kind: str, value: int) -> str:
    if kind == "base":
        return "base"
    if kind == "track":
        return f"d{value}"
    if kind == "home":
        return f"home{value + 1}"
    return f"{kind}{value}"


def prompt_checked(label: str, session: AdvisorSession, session_path: Path) -> str:
    try:
        raw = input(label).strip()
    except EOFError as exc:
        raise QuitAdvisor from exc
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise QuitAdvisor
    if lowered in {"u", "undo"}:
        if session.undo():
            save_session(session_path, session)
            print("Undid the last applied move.")
        else:
            print("Nothing to undo.")
        raise RestartTurn
    return raw


def prompt_cards(label: str, session: AdvisorSession, session_path: Path) -> list[int]:
    while True:
        raw = prompt_checked(label, session, session_path)
        try:
            return parse_cards(raw)
        except ValueError as error:
            print(error)


def prompt_optional_card(label: str, session: AdvisorSession, session_path: Path) -> int | None:
    while True:
        raw = prompt_checked(label, session, session_path)
        if raw == "":
            return None
        try:
            return parse_card(raw)
        except ValueError as error:
            print(error)


def prompt_seat() -> int:
    while True:
        raw = input("Your seat (P1-P4): ").strip()
        try:
            return parse_seat(raw)
        except (ValueError, TypeError) as error:
            print(error)


def create_or_resume_session(path: Path) -> AdvisorSession:
    if path.exists():
        raw = input(f"Resume {path}? [Y/n] ").strip().lower()
        if raw not in {"n", "no"}:
            return load_session(path)

    user_seat = prompt_seat()
    session = AdvisorSession.new(user_seat)
    hand = parse_cards(input("Your current hand: ").strip())
    session.my_hand = hand
    session.hand_sizes[user_seat] = len(hand)
    save_session(path, session)
    return session


def ensure_current_hand(session: AdvisorSession, session_path: Path) -> None:
    expected = session.hand_sizes[session.user_seat]
    current = cards_label(session.my_hand)
    raw = prompt_checked(
        f"Your hand [{current}; Enter keeps, or type cards]: ",
        session,
        session_path,
    )
    if raw == "" and len(session.my_hand) == expected:
        return
    if raw == "" and len(session.my_hand) != expected:
        print(f"Stored hand has {len(session.my_hand)} cards, expected {expected}.")
        session.my_hand = prompt_cards("Your current hand: ", session, session_path)
    else:
        session.my_hand = parse_cards(raw)
    session.hand_sizes[session.user_seat] = len(session.my_hand)


def print_recommendations(recommendations: list[AdvisorRecommendation], top: int) -> None:
    print("Advice:")
    for rank, recommendation in enumerate(recommendations[:top], start=1):
        print(
            f"  {rank:>2}. score={recommendation.score:.3f} "
            f"prior={recommendation.prior:.3f} "
            f"wins={recommendation.wins:g}/{recommendation.rollouts} "
            f"id={recommendation.action_id}: {recommendation.label}"
        )


def prompt_forced_discard(
    session: AdvisorSession,
    detail: dict[str, Any],
    session_path: Path,
) -> int | None:
    if detail["kind"] != "skip":
        return None
    skipped_player = (session.current_player + 1) % NUM_PLAYERS
    if session.hand_sizes[skipped_player] <= 0:
        return None
    return prompt_optional_card(
        f"{player_label(skipped_player)} discarded by skip (blank if unknown): ",
        session,
        session_path,
    )


def handle_user_turn(
    session: AdvisorSession,
    model_policy: ModelPolicy,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    ensure_current_hand(session, args.session)
    recommendations = rank_advisor_actions(
        session,
        model_policy,
        device,
        samples=args.samples,
        rollouts_per_sample=args.rollouts_per_sample,
        max_steps=args.max_steps,
        seed=args.seed + session.turn_index * 1_000_003,
    )
    if not recommendations:
        print("No legal recommendations.")
        return

    print_recommendations(recommendations, args.top)
    details = {
        int(item["id"]): dict(item)
        for item in legal_action_details(session, None, args.seed + session.turn_index)
    }
    while True:
        raw = prompt_checked(
            "Apply [Enter=#1, rank number, `id N`, `l`, `undo`, `q`]: ",
            session,
            args.session,
        )
        if raw == "":
            chosen = recommendations[0]
            break
        if raw.lower() in {"l", "legal"}:
            print_legal_details(details.values())
            continue
        if raw.lower().startswith("id "):
            try:
                action_id = int(raw.split(maxsplit=1)[1])
            except ValueError:
                print("Use `id N`, where N is a legal action id.")
                continue
            match = [item for item in recommendations if item.action_id == action_id]
            if not match:
                print("That action id is not in the ranked recommendations.")
                continue
            chosen = match[0]
            break
        try:
            rank = int(raw)
        except ValueError:
            print("Use Enter, a rank number, `id N`, `l`, `undo`, or `q`.")
            continue
        if 1 <= rank <= len(recommendations):
            chosen = recommendations[rank - 1]
            break
        print("Rank out of range.")

    detail = details[chosen.action_id]
    forced_discard = prompt_forced_discard(session, detail, args.session)
    outcome = apply_action(
        session,
        chosen.action_id,
        detail.get("card"),
        forced_discard,
        args.seed + session.turn_index,
    )
    print(outcome)
    save_session(args.session, session)


def handle_observed_turn(session: AdvisorSession, args: argparse.Namespace) -> None:
    player = session.current_player
    if session.hand_sizes[player] == 0:
        card_id = None
        print(f"{player_label(player)} has no cards; pass should be the only legal action.")
    else:
        card_id = prompt_optional_card(
            f"{player_label(player)} revealed card: ",
            session,
            args.session,
        )
        if card_id is None:
            print("Need the revealed card unless the player has no cards.")
            return

    details = legal_action_details(session, card_id, args.seed + session.turn_index)
    if card_id is not None:
        details = [detail for detail in details if detail.get("card") == card_id]
    if not details:
        print("No legal moves match that card. The synced state is probably wrong.")
        return

    print_legal_details(details)
    while True:
        raw = prompt_checked("Which move happened? ", session, args.session)
        try:
            index = int(raw)
        except ValueError:
            print("Type the listed number, `undo`, or `q`.")
            continue
        if 1 <= index <= len(details):
            detail = details[index - 1]
            break
        print("Move number out of range.")

    forced_discard = prompt_forced_discard(session, detail, args.session)
    outcome = apply_action(
        session,
        int(detail["id"]),
        card_id,
        forced_discard,
        args.seed + session.turn_index,
    )
    print(outcome)
    save_session(args.session, session)


def print_legal_details(details: Any) -> None:
    print("Legal visible moves:")
    for index, detail in enumerate(details, start=1):
        print(f"  {index:>2}. id={detail['id']}: {stable_detail_label(dict(detail))}")


def stable_detail_label(detail: dict[str, Any]) -> str:
    card = detail.get("card")
    card_text = "no-card" if card is None else card_label(int(card))
    kind = detail["kind"]
    if kind == "enter":
        return f"{card_text}: spawn {stable_marble_label(detail['owner'], detail['marble'])}"
    if kind == "move":
        bulldozer = " bulldozer" if detail.get("bulldozer") else ""
        return (
            f"{card_text}: move {stable_marble_label(detail['owner'], detail['marble'])} "
            f"{detail['direction']} {detail['steps']}{bulldozer}"
        )
    if kind == "split":
        first = stable_marble_label(detail["owner"], detail["marble"])
        second = stable_marble_label(detail["second_owner"], detail["second_marble"])
        return (
            f"{card_text}: split 7: move {first} forward {detail['steps']}, "
            f"then move {second} forward {detail['second_steps']}"
        )
    if kind == "swap":
        source = stable_marble_label(detail["owner"], detail["marble"])
        target = stable_marble_label(detail["target_owner"], detail["target_marble"])
        return f"{card_text}: swap {source} with {target}"
    if kind == "skip":
        return f"{card_text}: skip next player / random discard"
    if kind == "burn":
        return f"{card_text}: burn"
    return "pass (no cards)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-game Jackbot advisor.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(ADVISOR_PRESETS), default="default")
    parser.add_argument("--session", type=Path, default=Path(".jackbot-advisor.json"))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--rollouts-per-sample", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=400_000)
    args = parser.parse_args()
    return apply_advisor_preset(args)


def apply_advisor_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = ADVISOR_PRESETS[args.preset]
    if args.samples is None:
        args.samples = preset.samples
    if args.rollouts_per_sample is None:
        args.rollouts_per_sample = preset.rollouts_per_sample
    if args.max_steps is None:
        args.max_steps = preset.max_steps
    return args


def advisor_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.preset == "oracle":
        return default_oracle_checkpoint()
    return default_checkpoint_path()


def main() -> None:
    args = parse_args()
    checkpoint = advisor_checkpoint(args)
    device = training_device()
    model_policy, _ = load_model_policy(checkpoint)
    session = create_or_resume_session(args.session)

    print("Jackbot advisor. Commands inside prompts: `undo`, `q`.")
    while session.winner is None:
        try:
            print_session_summary(session)
            if session.current_player == session.user_seat:
                handle_user_turn(session, model_policy, device, args)
            else:
                handle_observed_turn(session, args)
        except RestartTurn:
            continue
        except ValueError as error:
            print(error)

    print(f"Game over. Winner: {team_label(int(session.winner))}")


if __name__ == "__main__":
    main()

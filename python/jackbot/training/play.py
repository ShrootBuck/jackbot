from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jackbot import PlayGame
from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.device import choose_device
from jackbot.training.env import to_tensors
from jackbot.training.evaluate import load_model


TEAM_NAMES = {
    0: "P1 + P3",
    1: "P2 + P4",
}


@torch.no_grad()
def model_action(model: torch.nn.Module, game: PlayGame, device: torch.device) -> int:
    tensors = to_tensors(game.batch(), device)
    actions, _, _ = model.act(
        tensors.obs,
        tensors.action_features,
        tensors.action_offsets,
        deterministic=True,
    )
    return int(actions[0].item())


def play_game(
    checkpoint: Path,
    human_team: int,
    seed: int,
    device_name: str,
    model_delay: bool,
) -> None:
    device = choose_device(device_name)
    model, _ = load_model(checkpoint, device)
    model.eval()
    game = PlayGame(seed)

    print(f"Jackbot play | checkpoint: {checkpoint}")
    print(f"You are {TEAM_NAMES[human_team]}; model is {TEAM_NAMES[1 - human_team]}.")
    print("Type a move number, `l` to list legal moves, or `q` to quit.\n")

    while game.winner() is None:
        print(game.turn_text())
        print()

        current_team = game.current_player % 2
        if current_team == human_team:
            action = prompt_human_action(game)
            if action is None:
                return
        else:
            action = model_action(model, game, device)
            label = action_label(game, action)
            print(f"Model chooses {action}: {label}")
            if model_delay:
                input("Press Enter to apply.")

        print(game.step(action))
        print()

    winner = game.winner()
    if winner is not None:
        print(f"Game over. Winner: {TEAM_NAMES[winner]}")


def prompt_human_action(game: PlayGame) -> int | None:
    print_legal_actions(game)
    valid_actions = {action for action, _ in game.legal_action_labels()}

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return None

        if raw in {"q", "quit", "exit"}:
            return None
        if raw in {"l", "legal"}:
            print_legal_actions(game)
            continue
        if raw == "":
            continue

        try:
            action = int(raw)
        except ValueError:
            print("Not a move number. Use `l` for legal moves or `q` to quit.")
            continue

        if action not in valid_actions:
            print(f"Invalid move {action}. Use `l` to list legal moves.")
            continue
        return action


def print_legal_actions(game: PlayGame) -> None:
    print("Legal moves:")
    for action, label in game.legal_action_labels():
        print(f"  {action:>2}: {label}")


def action_label(game: PlayGame, action: int) -> str:
    for candidate, label in game.legal_action_labels():
        if candidate == action:
            return label
    return "<unknown move>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a checkpoint in the terminal.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--human-team",
        choices=["odd", "even"],
        default="odd",
        help="odd means you play P2+P4; even means you play P1+P3",
    )
    parser.add_argument(
        "--pause-model",
        action="store_true",
        help="wait for Enter before applying each model move",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or default_checkpoint_path()
    human_team = 1 if args.human_team == "odd" else 0
    play_game(checkpoint, human_team, args.seed, args.device, args.pause_model)


if __name__ == "__main__":
    main()

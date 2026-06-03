from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jackbot import PlayGame
from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.device import choose_device
from jackbot.training.evaluate import load_model
from jackbot.training.play import action_label, model_action


PLAYER_NAMES = ("P1", "P2", "P3", "P4")
TEAM_NAMES = {
    0: "P1 + P3",
    1: "P2 + P4",
}


@torch.no_grad()
def watch_game(
    checkpoints: list[Path],
    seed: int,
    device_name: str,
    pause: bool,
    max_turns: int | None,
    show_legal: bool,
) -> None:
    device = choose_device(device_name)
    models = []
    for checkpoint in checkpoints:
        model, _ = load_model(checkpoint, device)
        model.eval()
        models.append(model)

    game = PlayGame(seed)
    print("Jackbot watch")
    print(f"Seed: {seed}")
    if len(checkpoints) == 1:
        print(f"All players use: {checkpoints[0]}")
    else:
        for player, checkpoint in enumerate(checkpoints):
            print(f"{PLAYER_NAMES[player]} uses: {checkpoint}")
    print()

    turns = 0
    while game.winner() is None:
        if max_turns is not None and turns >= max_turns:
            print(f"Stopped after {max_turns} turns.")
            return

        print(game.turn_text())
        if show_legal:
            print("Legal moves:")
            for action, label in game.legal_action_labels():
                print(f"  {action:>2}: {label}")

        current_player = game.current_player
        model = models[current_player if len(models) == 4 else 0]
        action = model_action(model, game, device)
        label = action_label(game, action)
        print(f"\n{PLAYER_NAMES[current_player]} chooses {action}: {label}")

        if pause:
            input("Play that on the board, then press Enter.")

        print(game.step(action))
        print()
        turns += 1

    winner = game.winner()
    if winner is not None:
        print(f"Game over. Winner: {TEAM_NAMES[winner]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch model-controlled Jackbot players one move at a time."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="one checkpoint for all players, four checkpoints for P1 P2 P3 P4, or omitted for jackbot_best.pt",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pause", action="store_true", help="run without waiting after moves")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--show-legal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoints:
        args.checkpoints = [default_checkpoint_path()]
    if len(args.checkpoints) not in {1, 4}:
        raise SystemExit("Pass either 1 checkpoint or exactly 4 checkpoints.")
    watch_game(
        args.checkpoints,
        seed=args.seed,
        device_name=args.device,
        pause=not args.no_pause,
        max_turns=args.max_turns,
        show_legal=args.show_legal,
    )


if __name__ == "__main__":
    main()

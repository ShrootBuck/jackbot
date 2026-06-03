from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.device import choose_device
from jackbot.training.policies import (
    GauntletResult,
    load_model_policy,
    policy_from_spec,
    run_gauntlet,
)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    opponents = [policy_from_spec(spec, device) for spec in args.opponent]
    results: list[GauntletResult] = []

    for index, checkpoint in enumerate(args.checkpoint):
        candidate, _ = load_model_policy(checkpoint, device)
        result = run_gauntlet(
            candidate,
            opponents,
            args.games,
            args.seed + index * 10_000_019,
            device,
            num_envs=args.num_envs,
            max_steps_per_game=args.max_steps_per_game,
            deterministic=not args.stochastic,
        )
        results.append(result)
        _print_result(result)

    payload = [_result_to_dict(result) for result in results]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a side-swapped checkpoint gauntlet.")
    parser.add_argument("checkpoint", nargs="*", type=Path)
    parser.add_argument(
        "--opponent",
        action="append",
        default=None,
        help="Opponent spec: random, heuristic, or a checkpoint path. Can be repeated.",
    )
    parser.add_argument("--games", type=int, default=512, help="Games per side per opponent.")
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps-per-game", type=int, default=2_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Sample model actions.")
    parser.add_argument("--json", type=Path, default=None, help="Write machine-readable results.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        args.checkpoint = [default_checkpoint_path()]
    if args.opponent is None:
        args.opponent = ["random", "heuristic"]
    return args


def _print_result(result: GauntletResult) -> None:
    print(f"{result.candidate}: score={result.score:.3f} games={result.total_games}")
    for opponent in result.opponents:
        print(
            "  "
            f"vs {opponent.opponent}: win_rate={opponent.win_rate:.3f} "
            f"+/-{opponent.ci95_radius:.3f} "
            f"points={opponent.candidate_points:g}/{opponent.games} "
            f"wins={opponent.candidate_wins}"
        )


def _result_to_dict(result: GauntletResult) -> dict[str, Any]:
    data = asdict(result)
    data["score"] = result.score
    data["total_games"] = result.total_games
    for opponent in data["opponents"]:
        wins = opponent["candidate_even"]["even_wins"] + opponent["candidate_odd"]["odd_wins"]
        draws = opponent["candidate_even"]["unfinished"] + opponent["candidate_odd"]["unfinished"]
        games = opponent["candidate_even"]["games"] + opponent["candidate_odd"]["games"]
        points = wins + 0.5 * draws
        win_rate = points / max(1, games)
        opponent["candidate_wins"] = wins
        opponent["candidate_points"] = points
        opponent["games"] = games
        opponent["win_rate"] = win_rate
        opponent["ci95_radius"] = 1.96 * (win_rate * (1.0 - win_rate) / max(1, games)) ** 0.5
    return data


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jackbot.training.checkpoint import default_checkpoint_path
from jackbot.training.config import (
    CHAMPIONS_DIR,
    CURRENT_CHAMPION_CHECKPOINT,
    DISTILLED_DIR,
    OLD_CHAMPION_CHECKPOINT,
    PROMOTED_CHAMPION_CHECKPOINT,
)
from jackbot.training.policies import (
    BaselinePolicy,
    GauntletResult,
    MatchResult,
    Policy,
    load_model_policy,
    run_gauntlet,
)
from jackbot.training.runtime import training_device


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    random_floor: float = 0.99
    heuristic_floor: float = 0.965
    champion_lower_ci_floor: float = 0.55


@dataclass(frozen=True, slots=True)
class BenchmarkCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


def main() -> None:
    args = parse_args()
    device = training_device()
    thresholds = BenchmarkThresholds(
        random_floor=args.random_floor,
        heuristic_floor=args.heuristic_floor,
        champion_lower_ci_floor=args.champion_lower_ci_floor,
    )
    payload = []

    for index, checkpoint in enumerate(args.checkpoint):
        opponents = args.opponent or discover_benchmark_opponents(
            candidate=checkpoint,
            champions_dir=args.champions_dir,
            distilled_dir=args.distilled_dir,
            checkpoint_dir=args.checkpoint_dir,
            milestone_count=args.milestones,
        )
        promotion_opponent = args.promotion_opponent or infer_promotion_opponent(checkpoint)
        result = run_benchmark(
            checkpoint,
            opponents,
            seed=args.seed + index * 10_000_019,
            games=args.games,
            num_envs=args.num_envs,
            max_steps_per_game=args.max_steps_per_game,
        )
        checks = benchmark_checks(result, promotion_opponent, thresholds)
        passed = all(check.passed for check in checks)
        print_benchmark(result, checks, passed)
        payload.append(benchmark_payload(result, checks, passed))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.print_json:
        print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hard Jackbot champion benchmark.")
    parser.add_argument("checkpoint", nargs="*", type=Path)
    parser.add_argument(
        "--opponent",
        action="append",
        default=None,
        help="Override the default ladder. Can be random, heuristic, or a checkpoint path.",
    )
    parser.add_argument("--promotion-opponent", default=None)
    parser.add_argument("--games", type=int, default=512, help="Games per side per opponent.")
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps-per-game", type=int, default=2_000)
    parser.add_argument("--champions-dir", type=Path, default=CHAMPIONS_DIR)
    parser.add_argument("--distilled-dir", type=Path, default=DISTILLED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--milestones", type=int, default=4)
    parser.add_argument("--random-floor", type=float, default=0.99)
    parser.add_argument("--heuristic-floor", type=float, default=0.965)
    parser.add_argument("--champion-lower-ci-floor", type=float, default=0.55)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        args.checkpoint = [default_checkpoint_path()]
    return args


def discover_benchmark_opponents(
    candidate: Path,
    champions_dir: Path = CHAMPIONS_DIR,
    distilled_dir: Path = DISTILLED_DIR,
    checkpoint_dir: Path = Path("checkpoints"),
    milestone_count: int = 4,
) -> list[str]:
    specs = ["random", "heuristic"]
    paths: list[Path] = []
    paths.extend(_known_champions())
    if champions_dir.exists():
        paths.extend(sorted(champions_dir.glob("*.pt")))
    if distilled_dir.exists():
        paths.extend(sorted(distilled_dir.glob("*.pt")))
    paths.extend(_latest_milestones(checkpoint_dir, milestone_count))

    candidate_key = _resolve_if_exists(candidate)
    for path in _dedupe_existing(paths):
        if _resolve_if_exists(path) == candidate_key:
            continue
        specs.append(str(path))
    return specs


def infer_promotion_opponent(candidate: Path) -> str | None:
    candidate_key = _resolve_if_exists(candidate)
    for path in _promotion_candidates():
        if path.exists() and _resolve_if_exists(path) != candidate_key:
            return str(path)
    return None


def run_benchmark(
    checkpoint: Path,
    opponent_specs: list[str],
    seed: int,
    games: int,
    num_envs: int,
    max_steps_per_game: int,
) -> GauntletResult:
    candidate, _ = load_model_policy(checkpoint, name=str(checkpoint))
    opponents = [_policy_from_spec(spec) for spec in opponent_specs]
    return run_gauntlet(
        candidate,
        opponents,
        games,
        seed,
        training_device(),
        num_envs=num_envs,
        max_steps_per_game=max_steps_per_game,
    )


def benchmark_checks(
    result: GauntletResult,
    promotion_opponent: str | None,
    thresholds: BenchmarkThresholds = BenchmarkThresholds(),
) -> list[BenchmarkCheck]:
    by_opponent = {opponent.opponent: opponent for opponent in result.opponents}
    checks: list[BenchmarkCheck] = []

    if "random" in by_opponent:
        random = by_opponent["random"]
        checks.append(
            BenchmarkCheck(
                name="random_floor",
                passed=random.win_rate >= thresholds.random_floor,
                value=random.win_rate,
                threshold=thresholds.random_floor,
                detail="win rate vs random",
            )
        )
    if "heuristic" in by_opponent:
        heuristic = by_opponent["heuristic"]
        checks.append(
            BenchmarkCheck(
                name="heuristic_floor",
                passed=heuristic.win_rate >= thresholds.heuristic_floor,
                value=heuristic.win_rate,
                threshold=thresholds.heuristic_floor,
                detail="win rate vs heuristic",
            )
        )
    if promotion_opponent is not None and promotion_opponent in by_opponent:
        champion = by_opponent[promotion_opponent]
        lower_ci = champion.win_rate - champion.ci95_radius
        checks.append(
            BenchmarkCheck(
                name="champion_lower_ci",
                passed=lower_ci > thresholds.champion_lower_ci_floor,
                value=lower_ci,
                threshold=thresholds.champion_lower_ci_floor,
                detail=f"lower 95% CI vs {promotion_opponent}",
            )
        )
    return checks


def print_benchmark(result: GauntletResult, checks: list[BenchmarkCheck], passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{result.candidate}: {status} score={result.score:.3f} games={result.total_games}")
    for opponent in result.opponents:
        print(
            "  "
            f"vs {opponent.opponent}: win_rate={opponent.win_rate:.3f} "
            f"+/-{opponent.ci95_radius:.3f} "
            f"points={opponent.candidate_points:g}/{opponent.games}"
        )
    for check in checks:
        check_status = "pass" if check.passed else "fail"
        print(f"  [{check_status}] {check.name}: {check.value:.3f} threshold={check.threshold:.3f}")


def benchmark_payload(
    result: GauntletResult,
    checks: list[BenchmarkCheck],
    passed: bool,
) -> dict[str, Any]:
    return {
        **_result_to_dict(result),
        "passed": passed,
        "checks": [asdict(check) for check in checks],
    }


def _policy_from_spec(spec: str) -> Policy:
    if spec in {"random", "heuristic"}:
        return BaselinePolicy(spec)
    policy, _ = load_model_policy(Path(spec), name=spec)
    return policy


def _result_to_dict(result: GauntletResult) -> dict[str, Any]:
    data = asdict(result)
    data["score"] = result.score
    data["total_games"] = result.total_games
    for opponent in data["opponents"]:
        match_even = MatchResult(**opponent["candidate_even"])
        match_odd = MatchResult(**opponent["candidate_odd"])
        wins = match_even.even_wins + match_odd.odd_wins
        draws = match_even.unfinished + match_odd.unfinished
        games = match_even.games + match_odd.games
        points = wins + 0.5 * draws
        win_rate = points / max(1, games)
        ci95 = 1.96 * (win_rate * (1.0 - win_rate) / max(1, games)) ** 0.5
        opponent["candidate_wins"] = wins
        opponent["candidate_points"] = points
        opponent["games"] = games
        opponent["win_rate"] = win_rate
        opponent["ci95_radius"] = ci95
    return data


def _latest_milestones(checkpoint_dir: Path, count: int) -> list[Path]:
    if count <= 0 or not checkpoint_dir.exists():
        return []
    return sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:count]


def _promotion_candidates() -> list[Path]:
    return [
        CURRENT_CHAMPION_CHECKPOINT,
        Path("checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt"),
        PROMOTED_CHAMPION_CHECKPOINT,
        Path("checkpoints/wandb_best_s23pmvby_update1325/jackbot_best.pt"),
        OLD_CHAMPION_CHECKPOINT,
    ]


def _known_champions() -> list[Path]:
    return [
        PROMOTED_CHAMPION_CHECKPOINT,
        Path("checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt"),
        CURRENT_CHAMPION_CHECKPOINT,
        Path("checkpoints/wandb_best_s23pmvby_update1325/jackbot_best.pt"),
        OLD_CHAMPION_CHECKPOINT,
    ]


def _dedupe_existing(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_if_exists(path: Path) -> Path:
    return path.resolve() if path.exists() else path


if __name__ == "__main__":
    main()

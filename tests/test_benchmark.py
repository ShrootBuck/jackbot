from __future__ import annotations

from pathlib import Path

from jackbot.training.benchmark import (
    BenchmarkThresholds,
    benchmark_checks,
    discover_benchmark_opponents,
)
from jackbot.training.policies import CandidateOpponentResult, GauntletResult, MatchResult


def test_benchmark_checks_require_baselines_and_champion_lower_ci() -> None:
    result = GauntletResult(
        candidate="candidate.pt",
        opponents=[
            _opponent("random", wins=1000, games=1000),
            _opponent("heuristic", wins=970, games=1000),
            _opponent("champion.pt", wins=620, games=1000),
        ],
    )

    checks = benchmark_checks(
        result,
        "champion.pt",
        BenchmarkThresholds(
            random_floor=0.99,
            heuristic_floor=0.965,
            champion_win_rate_floor=0.57,
            champion_lower_ci_floor=0.525,
        ),
    )

    assert [check.name for check in checks] == [
        "random_floor",
        "heuristic_floor",
        "champion_win_rate",
        "champion_lower_ci",
    ]
    assert all(check.passed for check in checks)


def test_benchmark_checks_fail_weak_champion_margin() -> None:
    result = GauntletResult(
        candidate="candidate.pt",
        opponents=[
            _opponent("random", wins=1000, games=1000),
            _opponent("heuristic", wins=970, games=1000),
            _opponent("champion.pt", wins=540, games=1000),
        ],
    )

    checks = benchmark_checks(result, "champion.pt")

    assert checks[-1].name == "champion_lower_ci"
    assert checks[-1].passed is False


def test_benchmark_checks_fail_closed_when_required_opponents_are_missing() -> None:
    result = GauntletResult(candidate="candidate.pt", opponents=[])

    checks = benchmark_checks(result, "champion.pt")

    assert [check.name for check in checks] == [
        "random_floor",
        "heuristic_floor",
        "champion_win_rate",
        "champion_lower_ci",
    ]
    assert not any(check.passed for check in checks)


def test_benchmark_default_ladder_discovers_checkpoints_without_candidate(tmp_path) -> None:
    champions = tmp_path / "champions"
    distilled = tmp_path / "distilled"
    checkpoint_dir = tmp_path / "checkpoints"
    champions.mkdir()
    distilled.mkdir()
    checkpoint_dir.mkdir()
    candidate = champions / "candidate.pt"
    candidate.write_bytes(b"candidate")
    champion = champions / "champion.pt"
    champion.write_bytes(b"champion")
    distilled_model = distilled / "distilled.pt"
    distilled_model.write_bytes(b"distilled")
    milestone = checkpoint_dir / "epoch_0250.pt"
    milestone.write_bytes(b"milestone")

    specs = discover_benchmark_opponents(
        candidate,
        champions_dir=champions,
        distilled_dir=distilled,
        checkpoint_dir=checkpoint_dir,
        milestone_count=1,
    )

    assert specs[:2] == ["random", "heuristic"]
    assert str(candidate) not in specs
    assert str(champion) in specs
    assert str(distilled_model) in specs
    assert str(milestone) in specs


def _opponent(name: str, wins: int, games: int) -> CandidateOpponentResult:
    even_games = games // 2
    odd_games = games - even_games
    even_wins = wins // 2
    odd_wins = wins - even_wins
    return CandidateOpponentResult(
        candidate="candidate.pt",
        opponent=name,
        games_per_side=even_games,
        candidate_even=MatchResult(
            even_policy="candidate.pt",
            odd_policy=name,
            games=even_games,
            even_wins=even_wins,
            odd_wins=even_games - even_wins,
            unfinished=0,
        ),
        candidate_odd=MatchResult(
            even_policy=name,
            odd_policy="candidate.pt",
            games=odd_games,
            even_wins=odd_games - odd_wins,
            odd_wins=odd_wins,
            unfinished=0,
        ),
    )

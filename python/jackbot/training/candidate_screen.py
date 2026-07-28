from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from jackbot.training.benchmark import _result_to_dict, run_benchmark
from jackbot.training.checkpoint import (
    checkpoint_sha256,
    load_checkpoint_config,
)
from jackbot.training.config import TrainConfig
from jackbot.training.train import train


DEFAULT_SCREEN_ID = "pzrnunoa-screen-v1"
DEFAULT_SOURCE = Path("checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt")
KNOWN_SOURCE_SHA256 = "09a0c5a9da9938e03e92ec2f2a4c9cb4f5aadcc9e45ffeded96339ddb095d658"
SCREEN_UPDATES = 200
TRANSITIONS_PER_UPDATE = 65_536


@dataclass(frozen=True, slots=True)
class ArmSpec:
    feature_schema: str = "base_v1"
    centralized_critic: bool = False
    num_envs: int = 1_024
    rollout_len: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95


SCREEN_ARMS = {
    "C": ArmSpec(),
    "A": ArmSpec(feature_schema="a1"),
    "H": ArmSpec(feature_schema="h1"),
    "AH": ArmSpec(feature_schema="a1h1"),
    "T": ArmSpec(num_envs=512, rollout_len=128, gamma=0.999, gae_lambda=0.99),
    "V": ArmSpec(centralized_critic=True),
}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    arms = parse_arms(args.arms)
    screen_root = (args.checkpoint_root / args.screen_id).resolve()
    run_root = (args.run_root / args.screen_id).resolve()
    source_hash = checkpoint_sha256(source)
    manifest = build_manifest(args, source, source_hash, arms)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    if args.worker_arm is not None:
        ensure_manifest(run_root / "manifest.json", manifest)
        run_worker(args, source, source_hash, args.worker_arm, screen_root)
        return

    run_root.mkdir(parents=True, exist_ok=True)
    lock = (run_root / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another launcher is already running for {args.screen_id}") from error
    ensure_manifest(run_root / "manifest.json", manifest)

    for arm in arms:
        arm_dir = screen_root / arm
        expected_config = candidate_config(arm, args, source, source_hash, arm_dir)
        result_path = run_root / arm / "direct_c0.json"
        final_path = arm_dir / f"final_update_{args.updates:04d}.pt"
        selected_path = arm_dir / "selected_best.pt"
        if (
            arm_checkpoint_valid(final_path, args.updates, expected_config, source_hash)
            and arm_checkpoint_valid(selected_path, None, expected_config, source_hash)
            and arm_result_valid(result_path, selected_path, final_path, source_hash)
        ):
            print(f"{arm}: complete; skipping")
            continue
        if not arm_checkpoint_valid(final_path, args.updates, expected_config, source_hash):
            if final_path.exists():
                raise ValueError(f"invalid immutable final checkpoint: {final_path}")
            launch_worker(args, arm)
            latest = arm_dir / "jackbot_latest.pt"
            archive_final_checkpoint(latest, final_path, args.updates)
            if not arm_checkpoint_valid(final_path, args.updates, expected_config, source_hash):
                raise ValueError(f"worker produced an invalid final checkpoint: {final_path}")
        if selected_path.exists() and not arm_checkpoint_valid(
            selected_path,
            None,
            expected_config,
            source_hash,
        ):
            raise ValueError(f"invalid immutable selected checkpoint: {selected_path}")
        if not selected_path.exists():
            archive_checkpoint(arm_dir / "jackbot_best.pt", selected_path)
            if not arm_checkpoint_valid(selected_path, None, expected_config, source_hash):
                raise ValueError(f"worker produced an invalid selected checkpoint: {selected_path}")
        if result_path.exists() and not arm_result_valid(
            result_path,
            selected_path,
            final_path,
            source_hash,
        ):
            raise ValueError(f"invalid existing result: {result_path}")
        if not result_path.exists():
            evaluate_arm(
                args,
                arm,
                selected_path,
                final_path,
                source,
                source_hash,
                result_path,
            )
        update_aggregate_results(run_root, arms)
    update_aggregate_results(run_root, arms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the resumable Jackbot candidate screen.")
    parser.add_argument("--screen-id", default=DEFAULT_SCREEN_ID)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--arms", default=",".join(SCREEN_ARMS))
    parser.add_argument("--updates", type=int, default=SCREEN_UPDATES)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/candidate_screens"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/candidate_screens"))
    parser.add_argument("--wandb-project", default="jackbot")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--development-eval-games", type=int, default=128)
    parser.add_argument("--final-eval-games", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run tiny end-to-end arm checks.")
    parser.add_argument("--worker-arm", choices=SCREEN_ARMS, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_arms(raw: str) -> list[str]:
    arms = [arm.strip().upper() for arm in raw.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in SCREEN_ARMS]
    if unknown:
        raise ValueError(f"unknown screen arms: {', '.join(unknown)}")
    if not arms:
        raise ValueError("at least one screen arm is required")
    return list(dict.fromkeys(arms))


def candidate_config(
    arm: str,
    args: argparse.Namespace,
    source: Path,
    source_hash: str,
    checkpoint_dir: Path,
) -> TrainConfig:
    spec = SCREEN_ARMS[arm]
    source_config = load_checkpoint_config(source)
    smoke = getattr(args, "smoke", False)
    return TrainConfig(
        feature_schema=spec.feature_schema,
        centralized_critic=spec.centralized_critic,
        critic_hidden_size=256,
        seed=1,
        num_envs=8 if smoke else spec.num_envs,
        rollout_len=4 if smoke else spec.rollout_len,
        total_updates=args.updates,
        ppo_epochs=1 if smoke else 3,
        minibatch_size=16 if smoke else 4_096,
        gamma=spec.gamma,
        gae_lambda=spec.gae_lambda,
        clip=0.2,
        lr=1e-4,
        weight_decay=0.0,
        anneal_lr=False,
        entropy_coef=0.01,
        value_coef=0.5,
        belief_coef=0.05,
        max_grad_norm=1.0,
        hidden_size=source_config.hidden_size,
        shaping_start=0.0,
        shaping_decay_fraction=0.0,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_updates=1 if smoke else 25,
        milestone_interval_updates=10_000 if smoke else 50,
        checkpoint_interval_seconds=15 * 60,
        eval_interval_updates=1 if smoke else 50,
        eval_games=1 if smoke else args.development_eval_games,
        eval_seed=60_000,
        eval_opponents=[str(source)],
        eval_num_envs=64,
        eval_max_steps_per_game=2_000,
        league_enabled=True,
        league_baselines=["heuristic"],
        league_opponents=[str(source)],
        league_auto_checkpoints=False,
        league_max_checkpoints=1,
        league_self_play_weight=0.55,
        league_baseline_weight=0.05,
        league_checkpoint_weight=0.40,
        league_refresh_interval_updates=25,
        league_deterministic_opponents=False,
        parent_checkpoint=str(source),
        parent_checkpoint_sha256=source_hash,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_run_id=f"{args.screen_id}-{arm.lower()}",
        wandb_run_name=f"{args.screen_id}-{arm}",
        wandb_resume="never",
        wandb_group=args.screen_id,
        wandb_job_type="candidate-screen",
        wandb_tags=["candidate-screen", arm.lower()],
        wandb_log_checkpoints=False,
        use_wandb=not smoke,
    )


def build_manifest(
    args: argparse.Namespace,
    source: Path,
    source_hash: str,
    arms: list[str],
) -> dict[str, Any]:
    if source == DEFAULT_SOURCE.resolve() and source_hash != KNOWN_SOURCE_SHA256:
        raise ValueError(
            f"frozen C0 hash changed: expected {KNOWN_SOURCE_SHA256}, got {source_hash}"
        )
    return {
        "schema_version": 1,
        "screen_id": args.screen_id,
        "source": {"path": str(source), "sha256": source_hash},
        "code_sha256": source_code_sha256(Path.cwd()),
        "checkpoint_root": str((args.checkpoint_root / args.screen_id).resolve()),
        "run_root": str((args.run_root / args.screen_id).resolve()),
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "updates": args.updates,
        "transitions_per_update": 32 if args.smoke else TRANSITIONS_PER_UPDATE,
        "smoke": args.smoke,
        "development_eval_games_per_side": args.development_eval_games,
        "final_eval_games_per_side": args.final_eval_games,
        "development_seed": 60_000,
        "screen_selection_seed": 170_000,
        "arms": {
            arm: manifest_arm(args, arm)
            for arm in arms
        },
    }


def manifest_arm(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    spec = asdict(SCREEN_ARMS[arm])
    if args.smoke:
        spec["num_envs"] = 8
        spec["rollout_len"] = 4
    spec["transitions"] = args.updates * spec["num_envs"] * spec["rollout_len"]
    spec["wandb_run_id"] = f"{args.screen_id}-{arm.lower()}"
    return spec


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError(f"screen manifest mismatch at {path}; use a new --screen-id")
        return
    atomic_write_json(path, manifest)


def run_worker(
    args: argparse.Namespace,
    source: Path,
    source_hash: str,
    arm: str,
    screen_root: Path,
) -> None:
    checkpoint_dir = screen_root / arm
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    worker_lock = (checkpoint_dir / ".worker.lock").open("w")
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another worker is already running for arm {arm}") from error
    if checkpoint_sha256(source) != source_hash:
        raise ValueError("C0 changed after the screen manifest was created")
    expected = candidate_config(arm, args, source, source_hash, checkpoint_dir)
    latest = checkpoint_dir / expected.latest_name
    attempt = time.time_ns()
    if latest.exists():
        stored = load_checkpoint_config(latest)
        if config_fingerprint(stored) != config_fingerprint(expected):
            raise ValueError(f"{arm} checkpoint config does not match the screen manifest")
        update = checkpoint_update(latest)
        stored.wandb_run_id = f"{expected.wandb_run_id}-r{update:04d}-{attempt}"
        stored.wandb_run_name = f"{expected.wandb_run_name} resume {update}"
        stored.wandb_resume = "never"
        train(stored, resume=latest)
    else:
        expected.wandb_run_id = f"{expected.wandb_run_id}-a{attempt}"
        expected.wandb_run_name = f"{expected.wandb_run_name} attempt"
        train(expected, init_from=source)
    if checkpoint_sha256(source) != source_hash:
        raise ValueError("C0 changed while the candidate worker was running")


def launch_worker(args: argparse.Namespace, arm: str) -> None:
    command = [
        sys.executable,
        "-m",
        "jackbot.training.candidate_screen",
        "--worker-arm",
        arm,
        "--screen-id",
        args.screen_id,
        "--source",
        str(args.source),
        "--arms",
        args.arms,
        "--updates",
        str(args.updates),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--run-root",
        str(args.run_root),
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--development-eval-games",
        str(args.development_eval_games),
        "--final-eval-games",
        str(args.final_eval_games),
    ]
    if args.smoke:
        command.append("--smoke")
    print(f"{arm}: launching candidate worker", flush=True)
    started = time.time()
    return_code = -1
    try:
        completed = subprocess.run(command, check=True)
        return_code = completed.returncode
    finally:
        attempt_path = args.run_root.resolve() / args.screen_id / arm / "worker_attempts.json"
        attempt_payload = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
        attempts = list(attempt_payload.get("attempts", []))
        attempts.append(
            {
                "started_at": started,
                "duration_seconds": time.time() - started,
                "return_code": return_code,
            }
        )
        atomic_write_json(attempt_path, {"attempts": attempts})


def archive_final_checkpoint(latest: Path, final: Path, expected_update: int) -> None:
    if not final_checkpoint_valid(latest, expected_update):
        raise ValueError(f"latest checkpoint {latest} is not complete")
    if final.exists():
        if checkpoint_sha256(final) != checkpoint_sha256(latest):
            raise FileExistsError(f"immutable final checkpoint conflict at {final}")
        return
    os.link(latest, final)


def archive_checkpoint(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    if destination.exists():
        if checkpoint_sha256(destination) != checkpoint_sha256(source):
            raise FileExistsError(f"immutable checkpoint conflict at {destination}")
        return
    os.link(source, destination)


def final_checkpoint_valid(path: Path, expected_update: int) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("update", -1)) == expected_update


def arm_checkpoint_valid(
    path: Path,
    expected_update: int | None,
    expected_config: TrainConfig,
    source_hash: str,
) -> bool:
    if not path.exists():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        update = int(checkpoint["update"])
        stored_config = TrainConfig.from_dict(checkpoint["config"])
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    if expected_update is not None and update != expected_update:
        return False
    if not 0 < update <= expected_config.total_updates:
        return False
    if int(checkpoint.get("global_steps", -1)) != (
        update * expected_config.num_envs * expected_config.rollout_len
    ):
        return False
    if config_fingerprint(stored_config) != config_fingerprint(expected_config):
        return False
    if stored_config.parent_checkpoint_sha256 != source_hash:
        return False
    if (
        checkpoint.get("checkpoint_version") != 2
        or "optimizer" not in checkpoint
        or checkpoint.get("environment_state") is None
    ):
        return False
    expected_model_spec = {
        "feature_schema": expected_config.feature_schema,
        "centralized_critic": expected_config.centralized_critic,
        "hidden_size": expected_config.hidden_size,
        "critic_hidden_size": expected_config.critic_hidden_size,
    }
    return checkpoint.get("model_spec") == expected_model_spec


def arm_result_valid(
    path: Path,
    selected: Path,
    final: Path,
    source_hash: str,
) -> bool:
    if not path.exists() or not selected.exists() or not final.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("checkpoint_sha256") == checkpoint_sha256(selected)
        and payload.get("final_checkpoint_sha256") == checkpoint_sha256(final)
        and payload.get("source_sha256") == source_hash
    )


def checkpoint_update(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("update", -1))


def evaluate_arm(
    args: argparse.Namespace,
    arm: str,
    candidate: Path,
    final: Path,
    source: Path,
    source_hash: str,
    result_path: Path,
) -> None:
    print(f"{arm}: evaluating development-selected checkpoint directly against C0")
    if checkpoint_sha256(source) != source_hash:
        raise ValueError("C0 changed before final candidate evaluation")
    evaluation_started = time.perf_counter()
    result = run_benchmark(
        candidate,
        [str(source)],
        seed=170_000,
        games=args.final_eval_games,
        num_envs=64,
        max_steps_per_game=2_000,
    )
    if checkpoint_sha256(source) != source_hash:
        raise ValueError("C0 changed during final candidate evaluation")
    payload = {
        "arm": arm,
        "checkpoint": str(candidate.resolve()),
        "checkpoint_sha256": checkpoint_sha256(candidate),
        "final_checkpoint": str(final.resolve()),
        "final_checkpoint_sha256": checkpoint_sha256(final),
        "source": str(source),
        "source_sha256": source_hash,
        "seed": 170_000,
        "games_per_side": args.final_eval_games,
        "training_wall_seconds": worker_wall_seconds(args, arm),
        "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
        **_result_to_dict(result),
    }
    atomic_write_json(result_path, payload)


def update_aggregate_results(run_root: Path, arms: list[str]) -> None:
    results = {}
    for arm in arms:
        result_path = run_root / arm / "direct_c0.json"
        if result_path.exists():
            results[arm] = json.loads(result_path.read_text())
    atomic_write_json(run_root / "results.json", {"arms": results})


def worker_wall_seconds(args: argparse.Namespace, arm: str) -> float:
    path = args.run_root.resolve() / args.screen_id / arm / "worker_attempts.json"
    if not path.exists():
        return 0.0
    payload = json.loads(path.read_text())
    return sum(float(attempt["duration_seconds"]) for attempt in payload.get("attempts", []))


def config_fingerprint(config: TrainConfig) -> str:
    data = config.to_dict()
    for key in ("wandb_run_id", "wandb_run_name", "wandb_resume"):
        data.pop(key, None)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_code_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = []
    for pattern in (
        "python/**/*.py",
        "crates/**/*.rs",
        "crates/**/Cargo.toml",
        "scripts/*.sh",
    ):
        paths.extend(root.glob(pattern))
    paths.extend(
        path
        for path in (root / "Cargo.toml", root / "Cargo.lock", root / "pyproject.toml", root / "uv.lock")
        if path.exists()
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    import jackbot._jackbot as native_extension

    extension_path = Path(native_extension.__file__)
    digest.update(b"native-extension")
    digest.update(extension_path.read_bytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()

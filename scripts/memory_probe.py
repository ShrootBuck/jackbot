#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import shutil
import tempfile
import time
from pathlib import Path

import psutil
import torch

from jackbot.training.checkpoint import default_checkpoint_path, load_checkpoint_config
from jackbot.training.config import TrainConfig
from jackbot.training.device import choose_device
from jackbot.training.env import make_env, random_actions, to_tensors
from jackbot.training.league import LeaguePool
from jackbot.training.model import JackbotNet
from jackbot.training.policies import load_model_policy
from jackbot.training.ppo import collect_rollout, ppo_update


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    print(f"mode={args.mode} device={device}")
    print(
        "label\titer\trss_mb\tvms_mb\tuss_mb\tsys_used_mb\tswap_used_mb\tswap_pct"
        "\tmps_current_mb\tmps_driver_mb"
    )

    if args.mode == "load-policy":
        probe_load_policy(args, device)
    elif args.mode == "league-refresh":
        probe_league_refresh(args, device)
    elif args.mode == "ppo-updates":
        probe_ppo_updates(args, device)
    elif args.mode == "tensor-transfer":
        probe_tensor_transfer(args, device)
    elif args.mode == "env-random":
        probe_env_loop(args, device, use_model=False)
    elif args.mode == "env-model":
        probe_env_loop(args, device, use_model=True, forward_only=False)
    elif args.mode == "env-forward":
        probe_env_loop(args, device, use_model=True, forward_only=True)
    elif args.mode == "forward-fixed":
        probe_forward_fixed(args, device)
    else:
        raise ValueError(f"unknown mode {args.mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Jackbot training memory behavior.")
    parser.add_argument(
        "mode",
        choices=[
            "load-policy",
            "league-refresh",
            "ppo-updates",
            "tensor-transfer",
            "env-random",
            "env-model",
            "env-forward",
            "forward-fixed",
        ],
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--league-max-checkpoints", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--league", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--log-phases", action="store_true")
    parser.add_argument("--deterministic-actions", action="store_true")
    return parser.parse_args()


def probe_load_policy(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint = _checkpoint_path(args)
    snapshot("start", 0, device)
    for iteration in range(1, args.iterations + 1):
        policy, _ = load_model_policy(checkpoint, device)
        del policy
        release(device)
        if _should_log(iteration, args.log_every):
            snapshot("load_policy", iteration, device)


def probe_league_refresh(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint = _checkpoint_path(args)
    with tempfile.TemporaryDirectory(prefix="jackbot-league-memory-") as temp:
        checkpoint_dir = Path(temp)
        config = _league_config(args, checkpoint, checkpoint_dir)
        _link_checkpoint(checkpoint, checkpoint_dir / "epoch_0000.pt")
        league = LeaguePool(config, device)
        snapshot("start", 0, device)
        for iteration in range(1, args.iterations + 1):
            path = checkpoint_dir / f"epoch_{iteration:04d}.pt"
            _link_checkpoint(checkpoint, path)
            now = time.time() + iteration
            os.utime(path, (now, now))
            league.refresh()
            release(device)
            if _should_log(iteration, args.log_every):
                snapshot("league_refresh", iteration, device)
        del league
        release(device)
        snapshot("after_drop_league", args.iterations, device)


def probe_ppo_updates(args: argparse.Namespace, device: torch.device) -> None:
    config = TrainConfig(
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        total_updates=args.iterations,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        use_wandb=False,
    )
    model = JackbotNet(config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    env = make_env(config.num_envs, config.seed)
    batch = env.reset(config.seed)
    league = None

    if args.league:
        checkpoint = _checkpoint_path(args)
        with tempfile.TemporaryDirectory(prefix="jackbot-ppo-league-memory-") as temp:
            checkpoint_dir = Path(temp)
            for index in range(args.league_max_checkpoints):
                _link_checkpoint(checkpoint, checkpoint_dir / f"epoch_{index:04d}.pt")
            config.league_enabled = True
            config.league_max_checkpoints = args.league_max_checkpoints
            config.checkpoint_dir = checkpoint_dir
            league = LeaguePool(config, device)
            batch = _run_ppo_loop(args, config, model, optimizer, env, batch, league, device)
    else:
        batch = _run_ppo_loop(args, config, model, optimizer, env, batch, league, device)

    del batch, env, optimizer, model, league
    release(device)
    snapshot("after_drop_training", args.iterations, device)


def probe_tensor_transfer(args: argparse.Namespace, device: torch.device) -> None:
    env = make_env(args.num_envs, 1)
    batch = env.reset(1)
    snapshot("start", 0, device)
    for iteration in range(1, args.iterations + 1):
        tensors = to_tensors(batch, device)
        del tensors
        release(device)
        if _should_log(iteration, args.log_every):
            snapshot("tensor_transfer", iteration, device)
    del batch, env
    release(device)
    snapshot("after_drop_transfer", args.iterations, device)


def probe_env_loop(
    args: argparse.Namespace,
    device: torch.device,
    use_model: bool,
    forward_only: bool = False,
) -> None:
    env = make_env(args.num_envs, 1)
    batch = env.reset(1)
    model = JackbotNet(args.hidden_size).to(device) if use_model else None
    snapshot("start", 0, device)
    for iteration in range(1, args.iterations + 1):
        tensors = to_tensors(batch, device)
        with torch.no_grad():
            if model is None or forward_only:
                if model is not None:
                    output = model(tensors.obs, tensors.action_features, tensors.action_offsets)
                    del output
                actions = random_actions(tensors)
            else:
                actions, _, _ = model.act(
                    tensors.obs,
                    tensors.action_features,
                    tensors.action_offsets,
                    deterministic=args.deterministic_actions,
                )
        action_array = actions.cpu().numpy()
        batch = env.step(action_array, 0.0)
        del tensors, actions, action_array
        release(device)
        if _should_log(iteration, args.log_every):
            label = "env_forward" if forward_only else "env_model" if use_model else "env_random"
            snapshot(label, iteration, device)
    del batch, env, model
    release(device)
    snapshot("after_drop_env", args.iterations, device)


def probe_forward_fixed(args: argparse.Namespace, device: torch.device) -> None:
    env = make_env(args.num_envs, 1)
    batch = env.reset(1)
    tensors = to_tensors(batch, device)
    model = JackbotNet(args.hidden_size).to(device)
    snapshot("start", 0, device)
    for iteration in range(1, args.iterations + 1):
        with torch.no_grad():
            output = model(tensors.obs, tensors.action_features, tensors.action_offsets)
        del output
        release(device)
        if _should_log(iteration, args.log_every):
            snapshot("forward_fixed", iteration, device)
    del tensors, batch, env, model
    release(device)
    snapshot("after_drop_fixed", args.iterations, device)


def _run_ppo_loop(
    args: argparse.Namespace,
    config: TrainConfig,
    model: JackbotNet,
    optimizer: torch.optim.Optimizer,
    env,
    batch,
    league: LeaguePool | None,
    device: torch.device,
):
    snapshot("start", 0, device)
    for update in range(args.iterations):
        if league is not None:
            league.maybe_refresh(update)
        rollout, batch, _ = collect_rollout(
            env,
            model,
            batch,
            config,
            device,
            shaping_scale=0.0,
            league=league,
        )
        iteration = update + 1
        if args.log_phases and _should_log(iteration, args.log_every):
            snapshot("after_collect", iteration, device)
        if args.skip_ppo:
            del rollout
        else:
            stats = ppo_update(model, optimizer, rollout, config)
            if args.log_phases and _should_log(iteration, args.log_every):
                snapshot("after_ppo", iteration, device)
            del rollout, stats
        release(device)
        if _should_log(iteration, args.log_every):
            label = "collect_only" if args.skip_ppo else "ppo_update"
            snapshot(label, iteration, device)
    return batch


def _league_config(args: argparse.Namespace, checkpoint: Path, checkpoint_dir: Path) -> TrainConfig:
    try:
        config = load_checkpoint_config(checkpoint)
    except Exception:
        config = TrainConfig(hidden_size=args.hidden_size)
    config.checkpoint_dir = checkpoint_dir
    config.league_enabled = True
    config.league_baselines = []
    config.league_auto_checkpoints = True
    config.league_max_checkpoints = args.league_max_checkpoints
    return config


def _checkpoint_path(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint or default_checkpoint_path()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def _device(requested: str) -> torch.device:
    return choose_device(requested) if requested == "auto" else torch.device(requested)


def _link_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _should_log(iteration: int, log_every: int) -> bool:
    return iteration == 1 or iteration % max(1, log_every) == 0


def snapshot(label: str, iteration: int, device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    process = psutil.Process()
    memory = process.memory_info()
    full = process.memory_full_info()
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    current_mps = 0.0
    driver_mps = 0.0
    if torch.backends.mps.is_available():
        current_mps = torch.mps.current_allocated_memory() / (1024 * 1024)
        driver_allocated = getattr(torch.mps, "driver_allocated_memory", None)
        if driver_allocated is not None:
            driver_mps = driver_allocated() / (1024 * 1024)
    print(
        f"{label}\t{iteration}\t"
        f"{memory.rss / (1024 * 1024):.1f}\t"
        f"{memory.vms / (1024 * 1024):.1f}\t"
        f"{getattr(full, 'uss', 0) / (1024 * 1024):.1f}\t"
        f"{virtual.used / (1024 * 1024):.1f}\t"
        f"{swap.used / (1024 * 1024):.1f}\t"
        f"{swap.percent:.1f}\t"
        f"{current_mps:.1f}\t"
        f"{driver_mps:.1f}",
        flush=True,
    )


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

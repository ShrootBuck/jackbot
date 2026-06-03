from __future__ import annotations

import argparse
from dataclasses import fields, replace
import random
import time
from pathlib import Path

import numpy as np
import torch

from jackbot.training.checkpoint import load_checkpoint, load_checkpoint_config, save_checkpoint
from jackbot.training.config import TrainConfig, shaping_scale
from jackbot.training.device import choose_device
from jackbot.training.env import make_env
from jackbot.training.league import LeaguePool
from jackbot.training.logger import make_logger
from jackbot.training.model import JackbotNet
from jackbot.training.policies import ModelPolicy, gauntlet_metrics, policy_from_spec, run_gauntlet
from jackbot.training.ppo import collect_rollout, ppo_update


def train(config: TrainConfig, resume: Path | None = None) -> None:
    if resume is not None:
        config = _config_for_resume(load_checkpoint_config(resume), config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = choose_device(config.device)
    print(f"Using device: {device}")
    model = JackbotNet(config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    start_update = 0
    global_steps = 0
    best_score = float("-inf")
    completed_games = 0
    cumulative_winning_team_counts = [0, 0]
    cumulative_winning_move_player_counts = [0, 0, 0, 0]
    if resume is not None:
        _, start_update, global_steps, best_score, completed_games = load_checkpoint(
            resume,
            model,
            optimizer,
            device,
        )
        print(f"Resumed {resume} at update {start_update}, global_steps={global_steps}")

    env = make_env(config.num_envs, config.seed)
    batch = env.reset(config.seed)
    logger = make_logger(config)
    league = LeaguePool(config, device) if config.league_enabled else None
    last_checkpoint = time.monotonic()

    try:
        for update in range(start_update, config.total_updates):
            update_started = time.perf_counter()
            lr = _learning_rate_for_update(config, update)
            _set_learning_rate(optimizer, lr)
            if league is not None:
                league.maybe_refresh(update)
            scale = shaping_scale(config, update)
            rollout, batch, timing_metrics = collect_rollout(
                env,
                model,
                batch,
                config,
                device,
                scale,
                league,
            )
            ppo_started = time.perf_counter()
            stats = ppo_update(model, optimizer, rollout, config)
            _sync_device(device)
            timing_metrics["time/ppo_update_sec"] = time.perf_counter() - ppo_started
            del rollout
            global_steps += config.num_envs * config.rollout_len
            completed_games += stats.completed_games
            for team, count in enumerate(stats.winning_team_counts):
                cumulative_winning_team_counts[team] += count
            for player, count in enumerate(stats.winning_move_player_counts):
                cumulative_winning_move_player_counts[player] += count

            metrics = {
                "train/loss": stats.loss,
                "train/policy_loss": stats.policy_loss,
                "train/value_loss": stats.value_loss,
                "train/belief_loss": stats.belief_loss,
                "train/entropy": stats.entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_fraction": stats.clip_fraction,
                "train/mean_team_reward": stats.mean_reward,
                "train/completed_games": completed_games,
                "train/game_length_mean": stats.game_length_mean,
                "train/shaping_scale": scale,
                "train/lr": lr,
                "train/update": update + 1,
                **_winner_metrics(
                    stats.winning_team_counts,
                    stats.winning_move_player_counts,
                    cumulative_winning_team_counts,
                    cumulative_winning_move_player_counts,
                ),
                **timing_metrics,
            }

            if (update + 1) % config.eval_interval_updates == 0:
                eval_started = time.perf_counter()
                arena = run_gauntlet(
                    ModelPolicy("current", model),
                    _arena_opponents(config, device),
                    config.eval_games,
                    config.seed + 20_000 + update,
                    device,
                    num_envs=config.eval_num_envs,
                    max_steps_per_game=config.eval_max_steps_per_game,
                )
                _sync_device(device)
                metrics["time/eval_sec"] = time.perf_counter() - eval_started
                metrics.update(gauntlet_metrics(arena))
                score = arena.score
                if score > best_score:
                    best_score = score
                    best_path = config.checkpoint_dir / config.best_name
                    checkpoint_started = time.perf_counter()
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        config,
                        update + 1,
                        global_steps,
                        best_score,
                        completed_games,
                    )
                    logger.log_checkpoint(
                        best_path,
                        ["best", f"update-{update + 1}"],
                        _checkpoint_metadata(config, update + 1, global_steps, best_score, completed_games),
                    )
                    metrics["time/checkpoint_sec"] = metrics.get("time/checkpoint_sec", 0.0) + (
                        time.perf_counter() - checkpoint_started
                    )

            should_save_by_update = (update + 1) % config.checkpoint_interval_updates == 0
            should_save_by_time = (
                config.checkpoint_interval_seconds <= 0
                or time.monotonic() - last_checkpoint >= config.checkpoint_interval_seconds
            )
            if should_save_by_update or should_save_by_time:
                latest_path = config.checkpoint_dir / config.latest_name
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    config,
                    update + 1,
                    global_steps,
                    best_score,
                    completed_games,
                )
                logger.log_checkpoint(
                    latest_path,
                    ["latest", f"update-{update + 1}"],
                    _checkpoint_metadata(config, update + 1, global_steps, best_score, completed_games),
                )
                metrics["time/checkpoint_sec"] = metrics.get("time/checkpoint_sec", 0.0) + (
                    time.perf_counter() - checkpoint_started
                )
                last_checkpoint = time.monotonic()

            if (update + 1) % config.milestone_interval_updates == 0:
                milestone_path = config.checkpoint_dir / f"epoch_{update + 1:04d}.pt"
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    milestone_path,
                    model,
                    optimizer,
                    config,
                    update + 1,
                    global_steps,
                    best_score,
                    completed_games,
                )
                logger.log_checkpoint(
                    milestone_path,
                    ["milestone", f"update-{update + 1}"],
                    _checkpoint_metadata(config, update + 1, global_steps, best_score, completed_games),
                )
                metrics["time/checkpoint_sec"] = metrics.get("time/checkpoint_sec", 0.0) + (
                    time.perf_counter() - checkpoint_started
                )

            metrics["time/update_total_sec"] = time.perf_counter() - update_started
            logger.log(metrics, global_steps)
    finally:
        save_checkpoint(
            config.checkpoint_dir / config.latest_name,
            model,
            optimizer,
            config,
            min(config.total_updates, update + 1 if "update" in locals() else start_update),
            global_steps,
            best_score,
            completed_games,
        )
        logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jackbot with PPO self-play.")
    parser.add_argument("--smoke", action="store_true", help="Run the tiny smoke-training config.")
    parser.add_argument(
        "--profile",
        choices=["default", "serious"],
        default="default",
        help="Training defaults to start from before applying explicit overrides.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--belief-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--shaping-start", type=float, default=None)
    parser.add_argument("--shaping-decay-fraction", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr-anneal", action="store_true")
    parser.add_argument("--no-lr-anneal", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--latest-name", default=None)
    parser.add_argument("--best-name", default=None)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)
    parser.add_argument("--milestone-interval-updates", type=int, default=None)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=None)
    parser.add_argument("--eval-interval-updates", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument("--eval-max-steps-per-game", type=int, default=None)
    parser.add_argument("--eval-opponent", action="append", default=None)
    parser.add_argument("--league", action="store_true")
    parser.add_argument("--league-opponent", action="append", default=None)
    parser.add_argument("--league-baselines", default=None)
    parser.add_argument("--league-auto-checkpoints", action="store_true")
    parser.add_argument("--no-league-auto-checkpoints", action="store_true")
    parser.add_argument("--league-max-checkpoints", type=int, default=None)
    parser.add_argument("--league-self-play-weight", type=float, default=None)
    parser.add_argument("--league-baseline-weight", type=float, default=None)
    parser.add_argument("--league-checkpoint-weight", type=float, default=None)
    parser.add_argument("--league-refresh-interval-updates", type=int, default=None)
    parser.add_argument("--deterministic-league-opponents", action="store_true")
    parser.add_argument("--stochastic-league-opponents", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    if args.smoke:
        config = TrainConfig.smoke()
    elif args.profile == "serious":
        config = TrainConfig.serious()
    else:
        config = TrainConfig()
    if args.updates is not None:
        config.total_updates = args.updates
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.rollout_len is not None:
        config.rollout_len = args.rollout_len
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.minibatch_size is not None:
        config.minibatch_size = args.minibatch_size
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.gae_lambda is not None:
        config.gae_lambda = args.gae_lambda
    if args.clip is not None:
        config.clip = args.clip
    if args.lr is not None:
        config.lr = args.lr
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.entropy_coef is not None:
        config.entropy_coef = args.entropy_coef
    if args.value_coef is not None:
        config.value_coef = args.value_coef
    if args.belief_coef is not None:
        config.belief_coef = args.belief_coef
    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.shaping_start is not None:
        config.shaping_start = args.shaping_start
    if args.shaping_decay_fraction is not None:
        config.shaping_decay_fraction = args.shaping_decay_fraction
    if args.device is not None:
        config.device = args.device
    if args.lr_anneal:
        config.anneal_lr = True
    if args.no_lr_anneal:
        config.anneal_lr = False
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_mode is not None:
        config.wandb_mode = args.wandb_mode
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.latest_name is not None:
        config.latest_name = args.latest_name
    if args.best_name is not None:
        config.best_name = args.best_name
    if args.checkpoint_interval_updates is not None:
        config.checkpoint_interval_updates = args.checkpoint_interval_updates
    if args.milestone_interval_updates is not None:
        config.milestone_interval_updates = args.milestone_interval_updates
    if args.checkpoint_interval_seconds is not None:
        config.checkpoint_interval_seconds = args.checkpoint_interval_seconds
    if args.eval_interval_updates is not None:
        config.eval_interval_updates = args.eval_interval_updates
    if args.eval_games is not None:
        config.eval_games = args.eval_games
    if args.eval_num_envs is not None:
        config.eval_num_envs = args.eval_num_envs
    if args.eval_max_steps_per_game is not None:
        config.eval_max_steps_per_game = args.eval_max_steps_per_game
    if args.eval_opponent is not None:
        config.eval_opponents = args.eval_opponent
    if args.league:
        config.league_enabled = True
    if args.league_opponent is not None:
        config.league_opponents = args.league_opponent
    if args.league_baselines is not None:
        config.league_baselines = [item for item in args.league_baselines.split(",") if item]
    if args.league_auto_checkpoints:
        config.league_auto_checkpoints = True
    if args.no_league_auto_checkpoints:
        config.league_auto_checkpoints = False
    if args.league_max_checkpoints is not None:
        config.league_max_checkpoints = args.league_max_checkpoints
    if args.league_self_play_weight is not None:
        config.league_self_play_weight = args.league_self_play_weight
    if args.league_baseline_weight is not None:
        config.league_baseline_weight = args.league_baseline_weight
    if args.league_checkpoint_weight is not None:
        config.league_checkpoint_weight = args.league_checkpoint_weight
    if args.league_refresh_interval_updates is not None:
        config.league_refresh_interval_updates = args.league_refresh_interval_updates
    if args.deterministic_league_opponents:
        config.league_deterministic_opponents = True
    if args.stochastic_league_opponents:
        config.league_deterministic_opponents = False
    return config


def _config_for_resume(loaded: TrainConfig, requested: TrainConfig) -> TrainConfig:
    default = TrainConfig()
    merged = replace(loaded)
    runtime_fields = {"device", "use_wandb", "wandb_mode", "wandb_project"}

    for field in fields(TrainConfig):
        name = field.name
        requested_value = getattr(requested, name)
        if name == "hidden_size":
            if requested_value != default.hidden_size and requested_value != loaded.hidden_size:
                raise ValueError(
                    f"Cannot resume hidden_size={loaded.hidden_size} checkpoint "
                    f"with hidden_size={requested_value}."
                )
            continue
        if name in runtime_fields or requested_value != getattr(default, name):
            setattr(merged, name, requested_value)
    return merged


def main() -> None:
    args = parse_args()
    train(config_from_args(args), args.resume)


def _learning_rate_for_update(config: TrainConfig, update: int) -> float:
    if not config.anneal_lr:
        return config.lr
    progress = update / max(1, config.total_updates)
    return config.lr * max(0.0, 1.0 - progress)


def _set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _winner_metrics(
    winning_team_counts: tuple[int, int],
    winning_move_player_counts: tuple[int, int, int, int],
    cumulative_winning_team_counts: list[int],
    cumulative_winning_move_player_counts: list[int],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    update_games = sum(winning_team_counts)
    total_games = sum(cumulative_winning_team_counts)

    for team, label in enumerate(("even_p1_p3", "odd_p2_p4")):
        count = winning_team_counts[team]
        cumulative_count = cumulative_winning_team_counts[team]
        metrics[f"train/winning_team_{label}_count"] = count
        metrics[f"train/winning_team_{label}_rate"] = (
            count / update_games if update_games else float("nan")
        )
        metrics[f"train/winning_team_{label}_total"] = cumulative_count
        metrics[f"train/winning_team_{label}_cumulative_rate"] = (
            cumulative_count / total_games if total_games else float("nan")
        )

    for player, label in enumerate(("p1", "p2", "p3", "p4")):
        count = winning_move_player_counts[player]
        cumulative_count = cumulative_winning_move_player_counts[player]
        metrics[f"train/winning_move_player_{label}_count"] = count
        metrics[f"train/winning_move_player_{label}_rate"] = (
            count / update_games if update_games else float("nan")
        )
        metrics[f"train/winning_move_player_{label}_total"] = cumulative_count
        metrics[f"train/winning_move_player_{label}_cumulative_rate"] = (
            cumulative_count / total_games if total_games else float("nan")
        )

    return metrics


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _arena_opponents(config: TrainConfig, device: torch.device):
    return [policy_from_spec(spec, device) for spec in config.eval_opponents]


def _checkpoint_metadata(
    config: TrainConfig,
    update: int,
    global_steps: int,
    best_score: float,
    completed_games: int,
) -> dict[str, object]:
    return {
        "update": update,
        "global_steps": global_steps,
        "best_score": best_score,
        "completed_games": completed_games,
        "hidden_size": config.hidden_size,
        "league_enabled": config.league_enabled,
        "eval_games_per_side": config.eval_games,
        "eval_opponents": list(config.eval_opponents),
    }


if __name__ == "__main__":
    main()

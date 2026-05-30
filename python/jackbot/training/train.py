from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from jackbot.training.checkpoint import load_checkpoint, save_checkpoint
from jackbot.training.config import TrainConfig, shaping_scale
from jackbot.training.device import choose_device
from jackbot.training.env import make_env
from jackbot.training.evaluate import evaluate_model
from jackbot.training.logger import make_logger
from jackbot.training.model import JackbotNet
from jackbot.training.ppo import collect_rollout, ppo_update


def train(config: TrainConfig, resume: Path | None = None) -> None:
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
    if resume is not None:
        loaded_config, start_update, global_steps, best_score = load_checkpoint(
            resume,
            model,
            optimizer,
            device,
        )
        config.hidden_size = loaded_config.hidden_size
        print(f"Resumed {resume} at update {start_update}, global_steps={global_steps}")

    env = make_env(config.num_envs, config.seed)
    batch = env.reset(config.seed)
    logger = make_logger(config)
    last_checkpoint = time.monotonic()

    try:
        for update in range(start_update, config.total_updates):
            scale = shaping_scale(config, update)
            rollout, batch = collect_rollout(env, model, batch, config, device, scale)
            stats = ppo_update(model, optimizer, rollout, config)
            global_steps += config.num_envs * config.rollout_len

            metrics = {
                "train/loss": stats.loss,
                "train/policy_loss": stats.policy_loss,
                "train/value_loss": stats.value_loss,
                "train/belief_loss": stats.belief_loss,
                "train/entropy": stats.entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_fraction": stats.clip_fraction,
                "train/mean_team_reward": stats.mean_reward,
                "train/shaping_scale": scale,
                "train/update": update + 1,
            }

            if (update + 1) % config.eval_interval_updates == 0:
                random_eval = evaluate_model(
                    model,
                    config.eval_games,
                    "random",
                    config.seed + 20_000 + update,
                    device,
                )
                heuristic_eval = evaluate_model(
                    model,
                    config.eval_games,
                    "heuristic",
                    config.seed + 30_000 + update,
                    device,
                )
                metrics["arena/random_win_rate"] = random_eval.even_win_rate
                metrics["arena/heuristic_win_rate"] = heuristic_eval.even_win_rate
                score = (random_eval.even_win_rate + heuristic_eval.even_win_rate) / 2.0
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        config.checkpoint_dir / config.best_name,
                        model,
                        optimizer,
                        config,
                        update + 1,
                        global_steps,
                        best_score,
                    )

            should_save_by_update = (update + 1) % config.checkpoint_interval_updates == 0
            should_save_by_time = (
                config.checkpoint_interval_seconds <= 0
                or time.monotonic() - last_checkpoint >= config.checkpoint_interval_seconds
            )
            if should_save_by_update or should_save_by_time:
                save_checkpoint(
                    config.checkpoint_dir / config.latest_name,
                    model,
                    optimizer,
                    config,
                    update + 1,
                    global_steps,
                    best_score,
                )
                last_checkpoint = time.monotonic()

            if (update + 1) % config.milestone_interval_updates == 0:
                save_checkpoint(
                    config.checkpoint_dir / f"epoch_{update + 1:04d}.pt",
                    model,
                    optimizer,
                    config,
                    update + 1,
                    global_steps,
                    best_score,
                )

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
        )
        logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jackbot with PPO self-play.")
    parser.add_argument("--smoke", action="store_true", help="Run the tiny smoke-training config.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig.smoke() if args.smoke else TrainConfig()
    if args.updates is not None:
        config.total_updates = args.updates
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.rollout_len is not None:
        config.rollout_len = args.rollout_len
    if args.device is not None:
        config.device = args.device
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_mode is not None:
        config.wandb_mode = args.wandb_mode
    return config


def main() -> None:
    args = parse_args()
    train(config_from_args(args), args.resume)


if __name__ == "__main__":
    main()

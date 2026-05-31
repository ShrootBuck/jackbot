from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from jackbot import PlayGame
from jackbot.training.checkpoint import save_checkpoint
from jackbot.training.config import TrainConfig
from jackbot.training.device import choose_device
from jackbot.training.env import to_tensors
from jackbot.training.model import JackbotNet
from jackbot.training.ops import segment_env_ids
from jackbot.training.policies import ModelPolicy, load_model_policy, policy_from_spec
from jackbot.training.search import example_from_search, rank_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and train on search-improved targets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Write JSONL search targets.")
    collect.add_argument("checkpoint", type=Path)
    collect.add_argument("output", type=Path)
    collect.add_argument("--positions", type=int, default=128)
    collect.add_argument("--seed", type=int, default=90_000)
    collect.add_argument("--warmup-steps", type=int, default=20)
    collect.add_argument("--rollouts", type=int, default=16)
    collect.add_argument("--max-steps", type=int, default=400)
    collect.add_argument("--temperature", type=float, default=0.10)
    collect.add_argument("--opponent", default="self")
    collect.add_argument("--device", default="auto")

    train = subparsers.add_parser("train", help="Distill a checkpoint from JSONL targets.")
    train.add_argument("input", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--base-checkpoint", type=Path, default=None)
    train.add_argument("--hidden-size", type=int, default=2048)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--value-coef", type=float, default=0.25)
    train.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.command == "collect":
        collect_targets(args)
    else:
        train_distillation(args)


def collect_targets(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model_policy, _ = load_model_policy(args.checkpoint, device)
    opponent = model_policy if args.opponent == "self" else policy_from_spec(args.opponent, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    game_seed = args.seed
    game = PlayGame(game_seed)
    with args.output.open("w") as handle:
        while written < args.positions:
            _advance_game(game, model_policy, device, args.warmup_steps)
            if game.winner() is not None:
                game_seed += 1
                game = PlayGame(game_seed)
                continue

            scores = rank_actions(
                game,
                model_policy,
                opponent,
                device,
                rollouts=args.rollouts,
                max_steps=args.max_steps,
                rollout_deterministic=False,
            )
            handle.write(json.dumps(example_from_search(game, scores, args.temperature)) + "\n")
            written += 1
            _advance_game(game, model_policy, device, 1)
            if game.winner() is not None:
                game_seed += 1
                game = PlayGame(game_seed)


def train_distillation(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    examples = _read_examples(args.input)
    if args.base_checkpoint is not None:
        base_policy, config = load_model_policy(args.base_checkpoint, device)
        model = base_policy.model
    else:
        config = TrainConfig(hidden_size=args.hidden_size)
        model = JackbotNet(config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        random.shuffle(examples)
        losses = []
        for batch in _chunks(examples, args.batch_size):
            obs, action_features, action_offsets, target_probs, value_targets = _batch_examples(
                batch,
                device,
            )
            output = model(obs, action_features, action_offsets)
            env_ids = segment_env_ids(action_offsets)
            policy_loss = torch.zeros(obs.shape[0], device=device)
            policy_loss.scatter_add_(0, env_ids, -(target_probs * output.log_probs))
            policy_loss = policy_loss.mean()
            value_loss = F.mse_loss(output.values, value_targets)
            loss = policy_loss + args.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = sum(losses) / max(1, len(losses))
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    save_checkpoint(
        args.output,
        model,
        optimizer,
        config,
        update=0,
        global_steps=0,
        best_score=0.0,
        completed_games=0,
    )


@torch.no_grad()
def _advance_game(game: PlayGame, policy: ModelPolicy, device: torch.device, steps: int) -> None:
    for _ in range(steps):
        if game.winner() is not None:
            return
        tensors = to_tensors(game.batch(), device)
        action = int(policy.actions(tensors, deterministic=False)[0].item())
        game.step(action)


def _read_examples(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chunks(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _batch_examples(
    examples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.tensor([example["obs"] for example in examples], device=device, dtype=torch.float32)
    features = []
    probs = []
    offsets = [0]
    total = 0
    values = []
    for example in examples:
        action_features = example["action_features"]
        target_probs = example["target_probs"]
        features.extend(action_features)
        probs.extend(target_probs)
        total += len(action_features)
        offsets.append(total)
        values.append(float(example["value"]) * 2.0 - 1.0)

    return (
        obs,
        torch.tensor(features, device=device, dtype=torch.float32),
        torch.tensor(offsets, device=device, dtype=torch.long),
        torch.tensor(probs, device=device, dtype=torch.float32),
        torch.tensor(values, device=device, dtype=torch.float32),
    )


if __name__ == "__main__":
    main()

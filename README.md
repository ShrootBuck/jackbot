# jackbot
A Jackaroo bot trained with self-play reinforcement learning

## Current focus

The repo starts with the Rust game engine before any training code. That is
intentional: house rules should be nailed down in the engine before PyTorch
learns anything from it.

## Engine

```bash
cargo test
```

The engine crate lives in `crates/jackbot-engine` and owns:

- deterministic seeded games
- card/deck/deal state
- configurable V1 rule knobs
- dynamic legal actions
- legal player observations with hidden hands/deck order excluded
- replay by seed plus action IDs

## Manual smoke check

You can play a generated engine game in the terminal:

```bash
cargo run -p jackbot-engine --example play -- --seed 1
```

It prints the current player's hand, board state, and numbered legal moves. Type
the move number to apply it, `h` for help, or `q` to quit. This is the easiest
way to sanity-check the rules without reading Rust. For now, it is an
engine-generated game; a later advisor CLI can sync against a real table by
letting you enter observed cards and moves.

You can also run the final-advisor-shaped smoke test with a random policy:

```bash
cargo run -p jackbot-engine --example random_advisor -- --seed 1
```

This picks one random legal move for every player. Press Enter to apply the
suggestion on the engine and mirror it on your physical board. Type `l` whenever
you want to see the full legal move list.

Terminal marble labels are sorted by each player's progress toward home:
`#1` is closest to finished, then `#2`, and so on. The stable engine marble ID is
shown in parentheses, like `P1#2(m4)`, so debugging and replay still have fixed
identities.

## Current house-rule model

- Internal players are `0..3`; user-facing tools should show them as `P1..P4`.
- Teams are `P1 + P3` vs `P2 + P4`.
- The main track has 76 spaces. Spawn points are 19 spaces apart, meaning 18
  spaces between spawn points, not counting the spawn spaces themselves.
- Track distance is stored relative to each marble's owner. Distance `0` is
  that player's spawn. Distance `74` is the home-entry square two spaces behind
  spawn; moving forward from there enters home slot `0`.
- Home has 4 slots. Home movement is forward-only and exact-count.
- A marble on its own spawn is a hard blockade: nobody can pass, kill, or swap
  it, but its owner can move it forward normally.
- Burning is forced-only. If any legal play exists, burn actions are not legal.
- Deal cycle is 4 cards, 4 cards, 5 cards, then reshuffle.
- Finished players keep using their own hand, but their own-piece cards control
  their partner's marbles.

## Later advisor flow

The practical first UI should be a CLI/TUI, not a big GUI. It should let you
enter visible game events as they happen:

```bash
jackbot hand AS 7D 10H 4C
jackbot play p2 burn 9S
jackbot play p3 move p3m1 6
jackbot advise
```

The wrapper will keep the Rust engine state synchronized, then ask the trained
model for ranked legal actions. The answer should be phrased in human terms,
like "play 7D, move P1 marble 2 forward 3, then P1 marble 0 forward 4."

## Later training layer

When the rules are stable, the Python side should use `uv`, PyTorch, and
Weights & Biases (`wandb`) instead of TensorBoard. W&B should track training
reward, policy loss, value loss, entropy, win rate by baseline, engine steps/sec,
GPU/CPU utilization, temperatures when available, RAM usage, and checkpoint
metadata.

The realistic baseline training machine is an old M1 MacBook Pro, not an
8x-H100 cluster. The strategy should stay the same either way: keep memory
bounded, stream short rollouts, checkpoint aggressively, and let slower hardware
run longer. Bigger GPUs should reduce wall-clock time, not require a different
algorithm or a giant RAM-hungry replay setup.

Long training runs need explicit checkpointing. The training script should:

- overwrite `checkpoints/jackbot_latest.pt` on a fixed interval so training can
  resume after a crash or reboot
- save `checkpoints/jackbot_best.pt` whenever arena win rate or Elo reaches a
  new high
- archive milestone checkpoints like `checkpoints/epoch_050.pt` so bad RL runs
  can roll back before catastrophic forgetting
- log each checkpoint path and evaluation score to W&B

For under-the-bed/server runs, keep the machine awake, plugged in, and on a hard
surface with real airflow. RL training is sustained CPU/GPU load; carpet is not
a cooling strategy, it is a slow-motion hardware crime scene.

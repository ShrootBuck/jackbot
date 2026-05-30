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
`#1` is closest to finished, then `#2`, and so on. The CLI hides stable engine
marble IDs because seeded replays are fast enough when we need to debug exact
state.

## Board visualization

![Board](Board.png)

`Board.png` contains embedded Excalidraw metadata. Open it at https://excalidraw.com to interact with the board—you can move marbles to play out scenarios visually.

## Current house-rule model

- Internal players are `0..3`; user-facing tools should show them as `P1..P4`.
- Player labels go clockwise in turn order: `P1 -> P2 -> P3 -> P4 -> P1`.
  For a real game, choose `P1` as you or the starting seat, then label the rest
  clockwise.
- Teams are `P1 + P3` vs `P2 + P4`.
- The main track has 76 spaces. Spawn points are 19 spaces apart, meaning 18
  spaces between spawn points, not counting the spawn spaces themselves.
- Track distance is stored relative to each marble's owner. Distance `0` is
  that player's spawn. Distance `74` is the home-entry square two spaces behind
  spawn; moving forward from there enters home slot `0`.
- Home has 4 slots. Home movement is forward-only and exact-count.
- A marble on its own spawn is a hard blockade: nobody can pass, kill, or swap
  it. Even the owner cannot use a Jack to swap it off spawn; it can only move
  by legal movement cards such as forward movement or backward 4.
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

The Python side uses `uv`, PyTorch, PyO3/maturin, and Weights & Biases (`wandb`)
instead of TensorBoard. W&B tracks training reward, policy loss, value loss,
belief loss, entropy, win rate by baseline, engine steps/sec, RAM usage, and
checkpoint metadata.

Set up the pinned Python 3.12 environment and build the Rust extension:

```bash
uv sync --group dev
```

The helper scripts do the same thing with `RUSTFLAGS="-C target-cpu=native"` so
Rust compiles for the exact CPU on the current Mac:

```bash
./scripts/bootstrap.sh
./scripts/verify.sh
./scripts/smoke_train.sh
./scripts/short_train.sh
./scripts/train.sh
./scripts/resume.sh
./scripts/eval.sh
```

`target-cpu=native` is mainly for the Rust engine and PyO3 extension. It does
not magically turn PyTorch into MLX; the model still uses PyTorch's selected
device (`mps` or `cpu`) at runtime.

Run `./scripts/bootstrap.sh` after pulling new code or changing Rust files; it
forces a reinstall of the local `jackbot` package so the compiled extension is
rebuilt with the native CPU flag.

Run the full test suite:

```bash
cargo test
uv run pytest
```

Run the tiny end-to-end smoke trainer:

```bash
uv run jackbot-train --smoke
```

Run a real first training job locally:

```bash
uv run jackbot-train --wandb-mode online
```

For a full production run on the dedicated Mac, keep the machine awake with
`caffeinate`:

```bash
caffeinate -dimsu ./scripts/train.sh
```

Resume after a crash/reboot:

```bash
uv run jackbot-train --resume checkpoints/jackbot_latest.pt
```

Or keep the Mac awake while resuming:

```bash
caffeinate -dimsu ./scripts/resume.sh
```

Evaluate a checkpoint:

```bash
uv run jackbot-eval checkpoints/jackbot_best.pt --games 64
```

The realistic baseline training machine is an old M1 MacBook Pro, not an
8x-H100 cluster. The strategy should stay the same either way: keep memory
bounded, stream short rollouts, checkpoint aggressively, and let slower hardware
run longer. Bigger GPUs should reduce wall-clock time, not require a different
algorithm or a giant RAM-hungry replay setup.

The V1 trainer is a PPO self-play loop with a feed-forward MLP. The Rust engine
returns legal observations, legal dynamic actions, both-team rewards, and hidden
hand targets for the auxiliary belief head. Python never reimplements rules; it
just scores the legal action list Rust gives it.

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

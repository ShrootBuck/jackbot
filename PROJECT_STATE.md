# Project State

Last updated: 2026-08-15

## Shipped Champion

The August 15 local deadline funnel is complete. The promoted policy is:

```text
checkpoints/champions/promoted.pt
SHA-256 bbd29763fa752b17dd4152570d8b7cf605d640031d705c64a2d4c28c66b319cf
```

It is an exact copy of
`checkpoints/candidate_screens/pzrnunoa-screen-v1/A/epoch_0200.pt`.
The model uses the `a1` feature schema: deterministic post-action
marble-board consequences for every legal action.

The frozen prior champion C0 remains:

```text
checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt
SHA-256 09a0c5a9da9938e03e92ec2f2a4c9cb4f5aadcc9e45ffeded96339ddb095d658
```

On a fresh seed-180000 side-swapped match, A/200 beat C0 723/1024
(70.61%). On the untouched seed-70000 promotion bank it passed all release
gates:

- random: 2048/2048 (100.00%)
- heuristic: 2013/2048 (98.29%)
- C0: 1438/2048 (70.21%), Wilson 95% lower bound 68.20%

The complete immutable result is
`runs/deadline_20260815/final_benchmark.json`.

## Table Play

The one-command launcher is:

```bash
./scripts/play_tonight.sh
```

For a coding agent that must not hold a TTY open, use:

```bash
./scripts/agent_play.sh
```

Its `new`, `status`, `moves`, `observe`, `hand`, `advise`, and
`undo` subcommands load and save `.jackbot-agent.json` independently.
Agents must list legal `moves` before recording an ambiguous narration rather
than guessing a marble. `--json` provides structured one-line responses.

It fails closed if `checkpoints/champions/promoted.pt` is unavailable. The
default `god` advisor preset spends substantial test-time compute:

- 512 hidden-hand determinizations
- 2 batched rollouts per determination
- 128-ply cutoff with learned value bootstrap
- adaptive 32,768-total-rollout branch cap
- forced-move fast path and deterministic per-turn sampling

Fresh-root tests favored the 128-ply configuration overall. Across two
independent A/200 root banks it scored about 61.1% on independent holdout
rollouts, versus 57.6% for the raw policy. Prior-policy mixing did not help.
512 x 2 was marginally stronger than the cheaper 128 x 2 and 1024 x 1 variants
in the direct compute comparison, so it is the shipped preset.

The advisor supports saving/resuming a session and cleanly handles `q` or
Ctrl-C. Test-time search improves decisions without changing the promoted
checkpoint.

## Deadline Funnel Outcome

A/200 is the only policy improvement that replicated strongly. The corrected
control, longer-rollout PPO, privileged critic, independent PPO continuations,
combined representation, checkpoint soups, and two independent
search-distillation banks all failed to beat A/200 on fresh validation.
Development-only wins were rejected rather than promoted.

Detailed experiment results are in the ignored file
`runs/deadline_20260815/NOTES.md`.

## Evaluation Rules

Champion decisions use hard side-swapped matches. Random and heuristic results
are sanity floors, not strength claims. The release gate is:

- at least 99% versus random
- at least 94% versus heuristic
- at least 57% versus the prior champion
- Wilson 95% lower bound above 52.5% versus the prior champion

Required opponents fail closed. Never replace `promoted.pt` with a run-local
best checkpoint without passing a fresh promotion bank.

## Machine and Future Training

All deadline work ran locally on the current Apple Silicon MacBook Pro. PyTorch
training stays on CPU because ragged legal-action tensors are not a useful MPS
fit. Rust entrypoints should keep
`RUSTFLAGS="-C target-cpu=native"` through the repo scripts.

For a future long run, the blessed entrypoint remains:

```bash
./scripts/bootstrap.sh
caffeinate -dimsu ./scripts/train_god.sh
```

`train_god.sh` runs the `genius` profile and discovers the promoted champion
for warm start. Use `--resume PATH` only to continue a specific interrupted
run. Generated checkpoints, datasets, W&B downloads, and benchmark JSON remain
out of git.

# Project State

Last updated: 2026-06-05

## Current Goal

Build a stronger Jackbot model and advisor for practical Jackaroo table play.
The point is to beat friends, not build a generic RL platform.

## Machine Assumption

Training runs on the same Apple Silicon MacBook Pro every time. Existing
artifacts should be reused in place. Do not copy or rename checkpoints unless
the existing file is missing or Zayd explicitly asks for a new archive.

Known useful local checkpoint paths:

- `checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt`
- `checkpoints/wandb_best_s23pmvby_update1325/jackbot_best.pt`
- `checkpoints/champions/promoted.pt` only when a new benchmark winner has been
  intentionally promoted

## Current Training Flow

Use the single blessed script:

```bash
./scripts/bootstrap.sh
caffeinate -dimsu ./scripts/train_god.sh
```

`train_god.sh` runs the `genius` profile and auto-resumes from the first existing
champion checkpoint it finds. Pass `--resume PATH` only when intentionally
overriding that discovery.

## Current Evaluation Flow

Use `jackbot-benchmark` for champion decisions. A new candidate should not be
called better just because it crushes random/heuristic; those baselines are
mostly saturated. Promotion should clear the champion lower-confidence gate
against the prior best while preserving random/heuristic floors.


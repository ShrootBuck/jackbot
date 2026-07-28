# Project State

Last updated: 2026-07-28

## Current Goal

Build a stronger Jackbot model and advisor for practical Jackaroo table play.
The point is to beat friends, not build a generic RL platform.

The existing Rust engine is the canonical table ruleset. Do not block training
on comparisons with external Jackaroo variants unless Zayd reports a concrete
table mismatch.

## Machine Assumption

Training runs on the same Apple Silicon MacBook Pro every time. Existing
artifacts should be reused in place. Do not copy or rename checkpoints unless
the existing file is missing or Zayd explicitly asks for a new archive.

Known useful local checkpoint paths:

- `checkpoints/wandb_best_pzrnunoa_update3000/jackbot_best.pt`
- `checkpoints/wandb_best_s23pmvby_update1325/jackbot_best.pt`
- `checkpoints/champions/promoted.pt` only when a new benchmark winner has been
  intentionally promoted

Treat `wandb_best_pzrnunoa_update3000` as the frozen provisional incumbent
`C0` until a candidate is intentionally promoted. Default play/advisor tools
prefer `promoted.pt`, then this known incumbent, rather than silently serving an
unpromoted run-local `jackbot_best.pt`.

## Current Training Flow

The eventual long-run entrypoint remains:

```bash
./scripts/bootstrap.sh
caffeinate -dimsu ./scripts/train_god.sh
```

`train_god.sh` runs the `genius` profile and auto-resumes from the first existing
champion checkpoint it finds. Pass `--resume PATH` only when intentionally
overriding that discovery.

Do not launch the old monolithic 12,000-update plan. The approved execution
order is:

1. Verify the benchmark/league/advisor and candidate-screen code.
2. Run the warm-started candidate tournament, 200 updates per arm.
3. Inspect hard side-swapped results through W&B/artifacts and extend only the
   top two candidates.
4. Run week-scale compute only for the replicated winner.
5. Train explicit exploiters and improve table-time search after the winning
   policy architecture is known.

The user should only need to launch two or three supplied scripts. Do not ask
the user to interpret training charts or manually select checkpoints.

## Current Evaluation Flow

Use `jackbot-benchmark` for champion decisions. A new candidate should not be
called better just because it crushes random/heuristic; those baselines are
mostly saturated. Promotion should clear the champion lower-confidence gate
against the prior best while preserving random/heuristic floors.

Current hard-promotion defaults are a direct win rate of at least 57% against
the incumbent and a Wilson 95% lower bound above 52.5%, with 99% random and 94%
heuristic floors. Required opponents fail closed. Candidate and incumbent use
the same seed bank for both side assignments.

The genius profile now selects run-local best snapshots directly against the
incumbent. Its league category mass is 55% current self-play, 40% frozen
checkpoints, and 5% heuristic; adding checkpoint files must not increase total
checkpoint probability. Overwritten checkpoint paths must be reloaded, and
frozen policies should infer only on their assigned environment subbatches.

## Approved Candidate Funnel

The first screen is implemented by `./scripts/screen_candidates.sh`. Every arm
uses weights-only initialization from `C0`, a fresh AdamW optimizer, local
update/step counters, zero shaping, and 65,536 transitions per update. Never use
`--resume C0` to fork an arm; `--resume` is only for continuing an interrupted
arm.

Frozen C0 SHA-256:

```text
09a0c5a9da9938e03e92ec2f2a4c9cb4f5aadcc9e45ffeded96339ddb095d658
```

The six 200-update arms are:

- `C`: corrected base control
- `A`: deterministic post-action marble-board features
- `H`: last eight legal public action/card records
- `AH`: both representation additions
- `T`: 512 environments x 128 steps, gamma 0.999, lambda 0.99
- `V`: a training-only privileged residual critic using exact hidden-hand targets

Keep shaping disabled. Prefer direct champion Elo per wall-clock over weak
baseline scores. New adapters and the privileged critic's final layer start at
zero, so every migrated arm is bit-identical to C0 before its first update. The
screen totals 78,643,200 environment transitions before evaluation.

Preflight:

```bash
./scripts/bootstrap.sh
./scripts/verify.sh
uv run jackbot-screen --dry-run
```

Launch or resume the full screen with the exact same command:

```bash
caffeinate -dimsu ./scripts/screen_candidates.sh
```

Outputs live under `checkpoints/candidate_screens/pzrnunoa-screen-v1/` and
`runs/candidate_screens/pzrnunoa-screen-v1/`. Each arm has an immutable final,
an immutable development-selected best, a deterministic W&B group with unique
attempt IDs, and a direct fixed-seed C0 result. Screen selection uses seed bank
170,000; the 70,000 promotion bank remains untouched. Generated outputs stay
out of git.

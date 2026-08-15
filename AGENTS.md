# Jackbot Agent Notes

This is a greenfield research project to beat friends at Jackaroo. It is
not a platform, not infra for Google, and not worth enterprise-shaped ceremony.
Prefer the smallest real improvement to model strength, eval quality, or
table-play usability.

## Project Reality

- The training server is always the same Apple Silicon MacBook Pro.
- Assume existing local artifacts are still where previous runs put them.
- Do not standardize checkpoint locations just to look tidy. Use the actual path
  that exists, especially W&B downloads under `checkpoints/wandb_*`.
- The blessed long-run entrypoint is `./scripts/train_god.sh`.
- Use `RUSTFLAGS="-C target-cpu=native"` through the repo scripts; PyTorch runs
  on CPU because the ragged legal-action tensors are not a good MPS fit here.

## Working Rules

- Read `PROJECT_STATE.md` before changing training/eval/advisor behavior.
- Update `PROJECT_STATE.md` when you materially change the current training
  plan, checkpoint assumptions, benchmark ladder, or remote run instructions.
- For remote run status, use W&B/API and artifact metadata, not local-process
  guessing.
- For model-strength claims, prefer hard side-swapped benchmark results over
  random/heuristic charts, since those are nearly saturated.
- Generated checkpoints, W&B downloads, datasets, and benchmark JSON belong out
  of git unless explicitly requested.

## Live Table Agent Interface

When Zayd narrates a live game, do not open or preserve an interactive TTY.
Use the persistent one-shot interface in `./scripts/agent_play.sh`:

- `new --seat P1 --hand "AS KD 7H 4C"` starts a game.
- `status` reads the current state.
- `moves --card 5H` lists interpretations of an opponent's revealed move.
- `observe --card 5H --move 2` records the selected interpretation and exits.
- `advise` runs God-search when it is Zayd's turn.
- `observe --id 17` records the move Zayd actually made.
- `hand "AS KD 7H 4C"` records Zayd's cards after a new deal.
- `undo` repairs the last incorrectly recorded move.

Every mutating command saves to `.jackbot-agent.json`. Run `moves` before
`observe` when a narration could map to multiple legal actions; never guess
which marble moved. `--json` before the subcommand is available for structured
agent output.

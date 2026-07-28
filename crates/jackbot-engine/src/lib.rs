//! Jackbot's Rust game engine.
//!
//! This crate intentionally starts with the game/rules layer only. The Python
//! RL stack should treat this crate as the only source of truth for legal
//! actions, observations, state transitions, and deterministic replay.

mod cards;
mod game;
mod rng;
mod rules;

pub use cards::{Card, Rank, Suit};
pub use game::{
    ActionKind, Direction, Game, GameError, GameState, LegalAction, MarbleLocation, MarbleState,
    MoveLeg, NUM_PLAYERS, Observation, ObservedMarble, PUBLIC_HISTORY_LIMIT, PublicAction,
    PublicDiscard, StepEvents, StepOutcome, Team,
};
pub use rules::{RewardWeights, Rules};

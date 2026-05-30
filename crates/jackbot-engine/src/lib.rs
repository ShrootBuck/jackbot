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
    ActionKind, Direction, Game, GameError, LegalAction, MarbleLocation, MoveLeg, NUM_PLAYERS,
    Observation, ObservedMarble, StepEvents, StepOutcome, Team,
};
pub use rules::{RewardWeights, Rules};

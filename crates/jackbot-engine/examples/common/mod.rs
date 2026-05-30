use std::io::{self, Write};

use jackbot_engine::{
    ActionKind, Card, Direction, LegalAction, MarbleLocation, MoveLeg, NUM_PLAYERS, Observation,
    StepOutcome, Team,
};

pub struct MarbleLabels {
    labels: [[usize; 4]; NUM_PLAYERS],
}

impl MarbleLabels {
    pub fn from_observation(observation: &Observation, home_entry_distance: u8) -> Self {
        let mut labels = [[0; 4]; NUM_PLAYERS];

        for (player, player_labels) in labels.iter_mut().enumerate().take(NUM_PLAYERS) {
            let mut owned = observation
                .marbles
                .iter()
                .filter(|marble| marble.owner == player)
                .collect::<Vec<_>>();
            owned.sort_by(|left, right| {
                progress_score(right.location, home_entry_distance)
                    .cmp(&progress_score(left.location, home_entry_distance))
                    .then_with(|| left.index.cmp(&right.index))
            });

            for (ordinal, marble) in owned.iter().enumerate() {
                player_labels[marble.index] = ordinal + 1;
            }
        }

        Self { labels }
    }

    pub fn marble_label(&self, player: usize, marble_index: usize) -> String {
        format!(
            "P{}#{}(m{})",
            player + 1,
            self.labels[player][marble_index],
            marble_index + 1
        )
    }

    fn ordinal(&self, player: usize, marble_index: usize) -> usize {
        self.labels[player][marble_index]
    }
}

pub fn parse_seed() -> u64 {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--seed" {
            let Some(seed) = args.next() else {
                panic!("--seed needs a number");
            };
            return seed.parse().expect("--seed must be a number");
        }
        if let Some(seed) = arg.strip_prefix("--seed=") {
            return seed.parse().expect("--seed must be a number");
        }
    }
    1
}

pub fn print_turn(observation: &Observation, labels: &MarbleLabels) {
    println!(
        "Turn {} | {} to act | hand: {}",
        observation.turn_index,
        player_label(observation.current_player),
        cards(&observation.own_hand)
    );
    println!(
        "Hand sizes: P1={} P2={} P3={} P4={} | deck left: {}",
        observation.hand_sizes[0],
        observation.hand_sizes[1],
        observation.hand_sizes[2],
        observation.hand_sizes[3],
        observation.deck_remaining
    );
    print_board(observation, labels);
}

pub fn print_legal_actions(legal_actions: &[LegalAction], labels: &MarbleLabels) {
    println!("All legal moves:");
    for action in legal_actions {
        println!("  {:>2}: {}", action.id, action_label(action, labels));
    }
}

pub fn print_outcome(outcome: &StepOutcome) {
    let card = outcome
        .played_card
        .map_or_else(|| "no card".to_string(), |card| card.to_string());
    println!(
        "{} played {card}; next: {}",
        player_label(outcome.current_player),
        player_label(outcome.next_player)
    );

    if let Some((player, card)) = outcome.forced_discard {
        println!(
            "{} was skipped and randomly discarded {card}.",
            player_label(player)
        );
    }

    if outcome.events.entered_from_base > 0 {
        println!("Entered from base.");
    }
    if outcome.events.forward_steps > 0 {
        println!(
            "Moved forward {} total step(s).",
            outcome.events.forward_steps
        );
    }
    if outcome.events.backward_steps > 0 {
        println!(
            "Moved backward {} total step(s).",
            outcome.events.backward_steps
        );
    }
    if outcome.events.entered_home > 0 {
        println!("Entered home.");
    }
    if outcome.events.captures > 0 {
        println!("Captured {} marble(s).", outcome.events.captures);
    }
    if outcome.events.burns > 0 {
        println!("Burned a card.");
    }
    if let Some(winner) = outcome.winner {
        println!("Winner: {}", team_label(winner));
    }
}

pub fn action_label(action: &LegalAction, labels: &MarbleLabels) -> String {
    let card = action
        .card
        .map_or_else(|| "no-card".to_string(), |card| card.to_string());
    match &action.kind {
        ActionKind::Enter {
            owner,
            marble_index,
        } => format!(
            "{card}: spawn {}",
            labels.marble_label(*owner, *marble_index)
        ),
        ActionKind::Move {
            owner,
            marble_index,
            steps,
            direction,
            bulldozer,
        } => {
            let bulldozer_label = if *bulldozer { " bulldozer" } else { "" };
            format!(
                "{card}: move {} {} {steps}{bulldozer_label}",
                labels.marble_label(*owner, *marble_index),
                direction_label(*direction)
            )
        }
        ActionKind::SplitSeven { first, second } => {
            format!(
                "{card}: split 7: {}, then {}",
                leg_label(*first, labels),
                leg_label(*second, labels)
            )
        }
        ActionKind::Swap {
            owner,
            marble_index,
            target_player,
            target_marble_index,
        } => format!(
            "{card}: swap {} with {}",
            labels.marble_label(*owner, *marble_index),
            labels.marble_label(*target_player, *target_marble_index)
        ),
        ActionKind::SkipNext => format!("{card}: skip next player / random discard"),
        ActionKind::Burn => format!("{card}: burn"),
        ActionKind::PassNoCards => "pass (no cards)".to_string(),
    }
}

pub fn prompt(label: &str) -> Option<String> {
    print!("{label}");
    io::stdout().flush().ok()?;

    let mut input = String::new();
    if io::stdin().read_line(&mut input).ok()? == 0 {
        return None;
    }
    Some(input)
}

pub fn player_label(player: usize) -> String {
    format!("P{}", player + 1)
}

pub fn team_label(team: Team) -> &'static str {
    match team {
        Team::Even => "P1 + P3",
        Team::Odd => "P2 + P4",
    }
}

fn print_board(observation: &Observation, labels: &MarbleLabels) {
    for player in 0..NUM_PLAYERS {
        let mut marbles = observation
            .marbles
            .iter()
            .filter(|marble| marble.owner == player)
            .map(|marble| {
                (
                    labels.ordinal(player, marble.index),
                    format!(
                        "#{}(m{})={}",
                        labels.ordinal(player, marble.index),
                        marble.index + 1,
                        location_label(marble.location, marble.track_index)
                    ),
                )
            })
            .collect::<Vec<_>>();
        marbles.sort_by_key(|(ordinal, _)| *ordinal);
        let labels = marbles
            .into_iter()
            .map(|(_, label)| label)
            .collect::<Vec<_>>();
        println!("{}: {}", player_label(player), labels.join("  "));
    }
}

fn leg_label(leg: MoveLeg, labels: &MarbleLabels) -> String {
    format!(
        "move {} forward {}",
        labels.marble_label(leg.owner, leg.marble_index),
        leg.steps
    )
}

fn cards(cards: &[Card]) -> String {
    if cards.is_empty() {
        return "(empty)".to_string();
    }
    cards
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn location_label(location: MarbleLocation, track_index: Option<u8>) -> String {
    match location {
        MarbleLocation::Base => "base".to_string(),
        MarbleLocation::Track { distance } => {
            let absolute = track_index.expect("track marble should have track index");
            format!("track d{distance}/abs{absolute}")
        }
        MarbleLocation::Home { slot } => format!("home{}", slot + 1),
    }
}

fn direction_label(direction: Direction) -> &'static str {
    match direction {
        Direction::Forward => "forward",
        Direction::Backward => "backward",
    }
}

fn progress_score(location: MarbleLocation, home_entry_distance: u8) -> i16 {
    match location {
        MarbleLocation::Base => -100,
        MarbleLocation::Track { distance } if distance <= home_entry_distance => {
            i16::from(distance)
        }
        MarbleLocation::Track { distance } => -10 - i16::from(distance - home_entry_distance),
        MarbleLocation::Home { slot } => 100 + i16::from(slot),
    }
}

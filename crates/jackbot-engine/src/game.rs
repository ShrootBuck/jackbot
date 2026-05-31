use std::fmt;

use crate::{Card, Rank, Rules, Suit, rng::SmallRng};

pub const NUM_PLAYERS: usize = 4;
pub const MARBLES_PER_PLAYER: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Team {
    Even,
    Odd,
}

impl Team {
    pub fn of_player(player: usize) -> Self {
        if player.is_multiple_of(2) {
            Team::Even
        } else {
            Team::Odd
        }
    }

    pub fn index(self) -> usize {
        match self {
            Team::Even => 0,
            Team::Odd => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarbleLocation {
    Base,
    Track { distance: u8 },
    Home { slot: u8 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Marble {
    owner: usize,
    index: usize,
    location: MarbleLocation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedMarble {
    pub owner: usize,
    pub index: usize,
    pub location: MarbleLocation,
    pub track_index: Option<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveLeg {
    pub owner: usize,
    pub marble_index: usize,
    pub steps: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActionKind {
    Enter {
        owner: usize,
        marble_index: usize,
    },
    Move {
        owner: usize,
        marble_index: usize,
        steps: u8,
        direction: Direction,
        bulldozer: bool,
    },
    SplitSeven {
        first: MoveLeg,
        second: MoveLeg,
    },
    Swap {
        owner: usize,
        marble_index: usize,
        target_player: usize,
        target_marble_index: usize,
    },
    SkipNext,
    Burn,
    PassNoCards,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalAction {
    pub id: usize,
    pub player: usize,
    pub card_index: Option<usize>,
    pub card: Option<Card>,
    pub kind: ActionKind,
    pub features: ActionFeatures,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionFeatures {
    pub card_id: Option<usize>,
    pub card_rank_value: Option<u8>,
    pub action_type: u8,
    pub source_owner: Option<usize>,
    pub source_marble: Option<usize>,
    pub target_owner: Option<usize>,
    pub target_marble: Option<usize>,
    pub second_owner: Option<usize>,
    pub second_marble: Option<usize>,
    pub steps: i8,
    pub second_steps: i8,
    pub starts_in_base: bool,
    pub ends_in_home: bool,
    pub captures: bool,
    pub bulldozer: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Observation {
    pub player: usize,
    pub current_player: usize,
    pub turn_index: u64,
    pub own_hand: Vec<Card>,
    pub hand_sizes: [usize; NUM_PLAYERS],
    pub discard_counts: [u8; 52],
    pub deck_remaining: usize,
    pub marbles: Vec<ObservedMarble>,
    pub winner: Option<Team>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepEvents {
    pub acting_player: usize,
    pub acting_team: Team,
    pub entered_from_base: u8,
    pub forward_steps: u8,
    pub backward_steps: u8,
    pub entered_home: u8,
    pub captures: u8,
    pub burns: u8,
    pub skipped_player: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOutcome {
    pub current_player: usize,
    pub next_player: usize,
    pub played_card: Option<Card>,
    pub forced_discard: Option<(usize, Card)>,
    pub winner: Option<Team>,
    pub events: StepEvents,
    pub team_rewards: [f32; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameError {
    GameAlreadyFinished,
    InvalidPlayer(usize),
    InvalidActionId { id: usize, legal_count: usize },
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::GameAlreadyFinished => write!(f, "game is already finished"),
            GameError::InvalidPlayer(player) => write!(f, "invalid player index {player}"),
            GameError::InvalidActionId { id, legal_count } => {
                write!(
                    f,
                    "invalid action id {id}; {legal_count} legal actions exist"
                )
            }
        }
    }
}

impl std::error::Error for GameError {}

#[derive(Clone, Debug)]
pub struct Game {
    rules: Rules,
    rng: SmallRng,
    deck: Vec<Card>,
    discard: Vec<Card>,
    hands: [Vec<Card>; NUM_PLAYERS],
    marbles: Vec<Marble>,
    current_player: usize,
    turn_index: u64,
    deal_round_index: usize,
    winner: Option<Team>,
}

#[derive(Clone, Debug)]
struct MoveSimulation {
    destination: MarbleLocation,
    captures: Vec<usize>,
    forward_steps: u8,
    backward_steps: u8,
    entered_home: u8,
}

impl Game {
    pub fn new(seed: u64, rules: Rules) -> Self {
        rules.validate();
        let mut rng = SmallRng::new(seed);
        let mut deck = Card::standard_deck();
        rng.shuffle(&mut deck);

        let marbles = (0..NUM_PLAYERS)
            .flat_map(|owner| {
                (0..MARBLES_PER_PLAYER).map(move |index| Marble {
                    owner,
                    index,
                    location: MarbleLocation::Base,
                })
            })
            .collect();

        let mut game = Self {
            rules,
            rng,
            deck,
            discard: Vec::new(),
            hands: std::array::from_fn(|_| Vec::new()),
            marbles,
            current_player: 0,
            turn_index: 0,
            deal_round_index: 0,
            winner: None,
        };
        game.deal_round();
        game
    }

    pub fn rules(&self) -> &Rules {
        &self.rules
    }

    pub fn current_player(&self) -> usize {
        self.current_player
    }

    pub fn turn_index(&self) -> u64 {
        self.turn_index
    }

    pub fn winner(&self) -> Option<Team> {
        self.winner
    }

    pub fn legal_actions(&self) -> Vec<LegalAction> {
        if self.winner.is_some() {
            return Vec::new();
        }

        let player = self.current_player;
        if self.hands[player].is_empty() {
            return vec![self.make_action(0, player, None, None, ActionKind::PassNoCards)];
        }

        let mut actions = Vec::new();
        let controlled_player = self.controlled_player(player);

        for (card_index, card) in self.hands[player].iter().copied().enumerate() {
            match card.rank {
                Rank::Ace => {
                    self.add_enter_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                    );
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        1,
                        Direction::Forward,
                        false,
                    );
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        11,
                        Direction::Forward,
                        false,
                    );
                }
                Rank::King => {
                    self.add_enter_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                    );
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        13,
                        Direction::Forward,
                        true,
                    );
                }
                Rank::Jack => {
                    self.add_swap_actions(&mut actions, player, card_index, card, controlled_player)
                }
                Rank::Queen => {
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        12,
                        Direction::Forward,
                        false,
                    );
                    if matches!(card.suit, Suit::Diamonds | Suit::Hearts)
                        && !self.hands[self.next_player(player)].is_empty()
                    {
                        actions.push(self.make_action(
                            actions.len(),
                            player,
                            Some(card_index),
                            Some(card),
                            ActionKind::SkipNext,
                        ));
                    }
                }
                Rank::Ten => {
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        10,
                        Direction::Forward,
                        false,
                    );
                    if !self.hands[self.next_player(player)].is_empty() {
                        actions.push(self.make_action(
                            actions.len(),
                            player,
                            Some(card_index),
                            Some(card),
                            ActionKind::SkipNext,
                        ));
                    }
                }
                Rank::Four => self.add_move_actions(
                    &mut actions,
                    player,
                    card_index,
                    card,
                    controlled_player,
                    4,
                    Direction::Backward,
                    false,
                ),
                Rank::Five => {
                    for marble in self
                        .marbles
                        .iter()
                        .filter(|marble| !matches!(marble.location, MarbleLocation::Base))
                    {
                        if self.is_protected(marble) && marble.owner != controlled_player {
                            continue;
                        }
                        if self
                            .simulate_move(
                                &self.marbles,
                                marble.owner,
                                marble.index,
                                5,
                                Direction::Forward,
                                false,
                            )
                            .is_some()
                        {
                            actions.push(self.make_action(
                                actions.len(),
                                player,
                                Some(card_index),
                                Some(card),
                                ActionKind::Move {
                                    owner: marble.owner,
                                    marble_index: marble.index,
                                    steps: 5,
                                    direction: Direction::Forward,
                                    bulldozer: false,
                                },
                            ));
                        }
                    }
                }
                Rank::Seven => {
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        7,
                        Direction::Forward,
                        false,
                    );
                    self.add_split_seven_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                    );
                }
                Rank::Two | Rank::Three | Rank::Six | Rank::Eight | Rank::Nine => {
                    self.add_move_actions(
                        &mut actions,
                        player,
                        card_index,
                        card,
                        controlled_player,
                        card.rank.pip_value(),
                        Direction::Forward,
                        false,
                    );
                }
            }
        }

        if actions.is_empty() {
            for (card_index, card) in self.hands[player].iter().copied().enumerate() {
                actions.push(self.make_action(
                    actions.len(),
                    player,
                    Some(card_index),
                    Some(card),
                    ActionKind::Burn,
                ));
            }
        }

        actions
    }

    pub fn step(&mut self, action_id: usize) -> Result<StepOutcome, GameError> {
        if self.winner.is_some() {
            return Err(GameError::GameAlreadyFinished);
        }

        let actions = self.legal_actions();
        let action = actions
            .get(action_id)
            .cloned()
            .ok_or(GameError::InvalidActionId {
                id: action_id,
                legal_count: actions.len(),
            })?;

        let previous = self.current_player;
        let mut played_card = None;
        if let Some(card_index) = action.card_index {
            let card = self.hands[previous].remove(card_index);
            debug_assert_eq!(Some(card), action.card);
            self.discard.push(card);
            played_card = Some(card);
        }

        let mut events = StepEvents {
            acting_player: previous,
            acting_team: Team::of_player(previous),
            entered_from_base: 0,
            forward_steps: 0,
            backward_steps: 0,
            entered_home: 0,
            captures: 0,
            burns: 0,
            skipped_player: None,
        };

        let forced_discard = self.apply_action(&action, &mut events);
        self.winner = self.compute_winner();
        let team_rewards = self.team_rewards(&events, self.winner);

        self.current_player = match action.kind {
            ActionKind::SkipNext if forced_discard.is_some() => {
                self.next_player(self.next_player(previous))
            }
            _ => self.next_player(previous),
        };
        self.turn_index += 1;

        if self.winner.is_none() && self.hands.iter().all(Vec::is_empty) {
            self.deal_round();
        }

        Ok(StepOutcome {
            current_player: previous,
            next_player: self.current_player,
            played_card,
            forced_discard,
            winner: self.winner,
            events,
            team_rewards,
        })
    }

    pub fn observation(&self, player: usize) -> Result<Observation, GameError> {
        if player >= NUM_PLAYERS {
            return Err(GameError::InvalidPlayer(player));
        }

        let mut discard_counts = [0; 52];
        for card in &self.discard {
            discard_counts[card.id()] += 1;
        }

        Ok(Observation {
            player,
            current_player: self.current_player,
            turn_index: self.turn_index,
            own_hand: self.hands[player].clone(),
            hand_sizes: std::array::from_fn(|p| self.hands[p].len()),
            discard_counts,
            deck_remaining: self.deck.len(),
            marbles: self.observed_marbles(),
            winner: self.winner,
        })
    }

    pub fn hidden_hand_targets(&self, player: usize) -> Result<[[u8; 52]; 3], GameError> {
        if player >= NUM_PLAYERS {
            return Err(GameError::InvalidPlayer(player));
        }

        let mut targets = [[0; 52]; 3];
        for (slot, target) in targets.iter_mut().enumerate() {
            let other_player = (player + slot + 1) % NUM_PLAYERS;
            for card in &self.hands[other_player] {
                target[card.id()] = 1;
            }
        }

        Ok(targets)
    }

    fn add_enter_actions(
        &self,
        actions: &mut Vec<LegalAction>,
        player: usize,
        card_index: usize,
        card: Card,
        owner: usize,
    ) {
        for marble in self.owned_marbles(owner) {
            if self.can_enter(&self.marbles, owner, marble.index).is_some() {
                actions.push(self.make_action(
                    actions.len(),
                    player,
                    Some(card_index),
                    Some(card),
                    ActionKind::Enter {
                        owner,
                        marble_index: marble.index,
                    },
                ));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_move_actions(
        &self,
        actions: &mut Vec<LegalAction>,
        player: usize,
        card_index: usize,
        card: Card,
        owner: usize,
        steps: u8,
        direction: Direction,
        bulldozer: bool,
    ) {
        for marble in self.owned_marbles(owner) {
            if self
                .simulate_move(
                    &self.marbles,
                    owner,
                    marble.index,
                    steps,
                    direction,
                    bulldozer,
                )
                .is_some()
            {
                actions.push(self.make_action(
                    actions.len(),
                    player,
                    Some(card_index),
                    Some(card),
                    ActionKind::Move {
                        owner,
                        marble_index: marble.index,
                        steps,
                        direction,
                        bulldozer,
                    },
                ));
            }
        }
    }

    fn add_swap_actions(
        &self,
        actions: &mut Vec<LegalAction>,
        player: usize,
        card_index: usize,
        card: Card,
        owner: usize,
    ) {
        for source in self.owned_marbles(owner) {
            if !matches!(source.location, MarbleLocation::Track { .. }) || self.is_protected(source)
            {
                continue;
            }

            for target in &self.marbles {
                if source.owner == target.owner && source.index == target.index {
                    continue;
                }
                if !matches!(target.location, MarbleLocation::Track { .. })
                    || self.is_protected(target)
                {
                    continue;
                }

                actions.push(self.make_action(
                    actions.len(),
                    player,
                    Some(card_index),
                    Some(card),
                    ActionKind::Swap {
                        owner,
                        marble_index: source.index,
                        target_player: target.owner,
                        target_marble_index: target.index,
                    },
                ));
            }
        }
    }

    fn add_split_seven_actions(
        &self,
        actions: &mut Vec<LegalAction>,
        player: usize,
        card_index: usize,
        card: Card,
        owner: usize,
    ) {
        let marble_indices: Vec<usize> = self.owned_marbles(owner).map(|m| m.index).collect();
        for first_marble in &marble_indices {
            for second_marble in &marble_indices {
                if first_marble == second_marble {
                    continue;
                }
                for first_steps in 1..=6 {
                    let second_steps = 7 - first_steps;
                    let first = MoveLeg {
                        owner,
                        marble_index: *first_marble,
                        steps: first_steps,
                    };
                    let second = MoveLeg {
                        owner,
                        marble_index: *second_marble,
                        steps: second_steps,
                    };
                    if self.can_apply_split(first, second) {
                        actions.push(self.make_action(
                            actions.len(),
                            player,
                            Some(card_index),
                            Some(card),
                            ActionKind::SplitSeven { first, second },
                        ));
                    }
                }
            }
        }
    }

    fn make_action(
        &self,
        id: usize,
        player: usize,
        card_index: Option<usize>,
        card: Option<Card>,
        kind: ActionKind,
    ) -> LegalAction {
        let features = self.action_features(card, &kind);
        LegalAction {
            id,
            player,
            card_index,
            card,
            kind,
            features,
        }
    }

    fn action_features(&self, card: Option<Card>, kind: &ActionKind) -> ActionFeatures {
        let mut source_owner = None;
        let mut source_marble = None;
        let mut target_owner = None;
        let mut target_marble = None;
        let mut second_owner = None;
        let mut second_marble = None;
        let mut steps = 0;
        let mut second_steps = 0;
        let mut starts_in_base = false;
        let mut ends_in_home = false;
        let mut captures = false;
        let mut bulldozer = false;

        let action_type = match *kind {
            ActionKind::Enter {
                owner,
                marble_index,
            } => {
                source_owner = Some(owner);
                source_marble = Some(marble_index);
                starts_in_base = true;
                captures = self
                    .can_enter(&self.marbles, owner, marble_index)
                    .is_some_and(|captures| !captures.is_empty());
                0
            }
            ActionKind::Move {
                owner,
                marble_index,
                steps: move_steps,
                direction,
                bulldozer: is_bulldozer,
            } => {
                source_owner = Some(owner);
                source_marble = Some(marble_index);
                steps = match direction {
                    Direction::Forward => move_steps as i8,
                    Direction::Backward => -(move_steps as i8),
                };
                bulldozer = is_bulldozer;
                if let Some(simulation) = self.simulate_move(
                    &self.marbles,
                    owner,
                    marble_index,
                    move_steps,
                    direction,
                    is_bulldozer,
                ) {
                    ends_in_home = matches!(simulation.destination, MarbleLocation::Home { .. });
                    captures = !simulation.captures.is_empty();
                }
                1
            }
            ActionKind::SplitSeven { first, second } => {
                source_owner = Some(first.owner);
                source_marble = Some(first.marble_index);
                second_owner = Some(second.owner);
                second_marble = Some(second.marble_index);
                steps = first.steps as i8;
                second_steps = second.steps as i8;
                ends_in_home = self
                    .simulate_split(first, second)
                    .is_some_and(|(_, entered_home, _)| entered_home > 0);
                captures = self
                    .simulate_split(first, second)
                    .is_some_and(|(_, _, captures)| captures > 0);
                2
            }
            ActionKind::Swap {
                owner,
                marble_index,
                target_player,
                target_marble_index,
            } => {
                source_owner = Some(owner);
                source_marble = Some(marble_index);
                target_owner = Some(target_player);
                target_marble = Some(target_marble_index);
                3
            }
            ActionKind::SkipNext => 4,
            ActionKind::Burn => 5,
            ActionKind::PassNoCards => 6,
        };

        ActionFeatures {
            card_id: card.map(Card::id),
            card_rank_value: card.map(|c| c.rank.pip_value()),
            action_type,
            source_owner,
            source_marble,
            target_owner,
            target_marble,
            second_owner,
            second_marble,
            steps,
            second_steps,
            starts_in_base,
            ends_in_home,
            captures,
            bulldozer,
        }
    }

    fn apply_action(
        &mut self,
        action: &LegalAction,
        events: &mut StepEvents,
    ) -> Option<(usize, Card)> {
        match action.kind {
            ActionKind::Enter {
                owner,
                marble_index,
            } => {
                let captures = self
                    .can_enter(&self.marbles, owner, marble_index)
                    .expect("legal enter action remains legal");
                for captured_idx in captures {
                    self.marbles[captured_idx].location = MarbleLocation::Base;
                    events.captures += 1;
                }
                self.marble_mut(owner, marble_index).location =
                    MarbleLocation::Track { distance: 0 };
                events.entered_from_base = 1;
                None
            }
            ActionKind::Move {
                owner,
                marble_index,
                steps,
                direction,
                bulldozer,
            } => {
                let simulation = self
                    .simulate_move(
                        &self.marbles,
                        owner,
                        marble_index,
                        steps,
                        direction,
                        bulldozer,
                    )
                    .expect("legal move action remains legal");
                self.apply_move_simulation(owner, marble_index, simulation, events);
                None
            }
            ActionKind::SplitSeven { first, second } => {
                let first_simulation = self
                    .simulate_move(
                        &self.marbles,
                        first.owner,
                        first.marble_index,
                        first.steps,
                        Direction::Forward,
                        false,
                    )
                    .expect("legal split first leg remains legal");
                self.apply_move_simulation(
                    first.owner,
                    first.marble_index,
                    first_simulation,
                    events,
                );

                let second_simulation = self
                    .simulate_move(
                        &self.marbles,
                        second.owner,
                        second.marble_index,
                        second.steps,
                        Direction::Forward,
                        false,
                    )
                    .expect("legal split second leg remains legal");
                self.apply_move_simulation(
                    second.owner,
                    second.marble_index,
                    second_simulation,
                    events,
                );
                None
            }
            ActionKind::Swap {
                owner,
                marble_index,
                target_player,
                target_marble_index,
            } => {
                self.swap_marbles(owner, marble_index, target_player, target_marble_index);
                None
            }
            ActionKind::SkipNext => {
                let skipped_player = self.next_player(action.player);
                if self.hands[skipped_player].is_empty() {
                    return None;
                }
                let card_index = self.rng.gen_range(self.hands[skipped_player].len());
                let card = self.hands[skipped_player].remove(card_index);
                self.discard.push(card);
                events.skipped_player = Some(skipped_player);
                Some((skipped_player, card))
            }
            ActionKind::Burn => {
                events.burns = 1;
                None
            }
            ActionKind::PassNoCards => None,
        }
    }

    fn apply_move_simulation(
        &mut self,
        owner: usize,
        marble_index: usize,
        simulation: MoveSimulation,
        events: &mut StepEvents,
    ) {
        for captured_idx in &simulation.captures {
            self.marbles[*captured_idx].location = MarbleLocation::Base;
        }
        self.marble_mut(owner, marble_index).location = simulation.destination;
        events.forward_steps += simulation.forward_steps;
        events.backward_steps += simulation.backward_steps;
        events.entered_home += simulation.entered_home;
        events.captures += simulation.captures.len() as u8;
    }

    fn can_apply_split(&self, first: MoveLeg, second: MoveLeg) -> bool {
        self.simulate_split(first, second).is_some()
    }

    fn simulate_split(&self, first: MoveLeg, second: MoveLeg) -> Option<(Vec<Marble>, u8, u8)> {
        let mut marbles = self.marbles.clone();
        let first_simulation = self.simulate_move(
            &marbles,
            first.owner,
            first.marble_index,
            first.steps,
            Direction::Forward,
            false,
        )?;
        Self::apply_simulation_to_marbles(
            &mut marbles,
            first.owner,
            first.marble_index,
            first_simulation.clone(),
        );

        let second_simulation = self.simulate_move(
            &marbles,
            second.owner,
            second.marble_index,
            second.steps,
            Direction::Forward,
            false,
        )?;

        let entered_home = first_simulation.entered_home + second_simulation.entered_home;
        let captures =
            first_simulation.captures.len() as u8 + second_simulation.captures.len() as u8;
        Some((marbles, entered_home, captures))
    }

    fn apply_simulation_to_marbles(
        marbles: &mut [Marble],
        owner: usize,
        marble_index: usize,
        simulation: MoveSimulation,
    ) {
        for captured_idx in &simulation.captures {
            marbles[*captured_idx].location = MarbleLocation::Base;
        }
        let moving_idx = Self::marble_vec_index_in(marbles, owner, marble_index);
        marbles[moving_idx].location = simulation.destination;
    }

    fn simulate_move(
        &self,
        marbles: &[Marble],
        owner: usize,
        marble_index: usize,
        steps: u8,
        direction: Direction,
        bulldozer: bool,
    ) -> Option<MoveSimulation> {
        if steps == 0 {
            return None;
        }
        match direction {
            Direction::Forward => {
                self.simulate_forward(marbles, owner, marble_index, steps, bulldozer)
            }
            Direction::Backward => {
                if bulldozer {
                    return None;
                }
                self.simulate_backward(marbles, owner, marble_index, steps)
            }
        }
    }

    fn simulate_forward(
        &self,
        marbles: &[Marble],
        owner: usize,
        marble_index: usize,
        steps: u8,
        bulldozer: bool,
    ) -> Option<MoveSimulation> {
        let moving_idx = Self::marble_vec_index_in(marbles, owner, marble_index);
        let mut location = marbles[moving_idx].location;
        let mut captures = Vec::new();
        let mut entered_home = 0;

        for _ in 0..steps {
            location = match location {
                MarbleLocation::Base => return None,
                MarbleLocation::Home { slot } => {
                    let next_slot = slot + 1;
                    if next_slot >= self.rules.home_len {
                        return None;
                    }
                    MarbleLocation::Home { slot: next_slot }
                }
                MarbleLocation::Track { distance } => {
                    if distance == self.rules.home_entry_distance() {
                        if self.home_slot_occupied(marbles, owner, 0, Some(moving_idx)) {
                            return None;
                        }
                        entered_home = 1;
                        MarbleLocation::Home { slot: 0 }
                    } else {
                        let next_distance = (distance + 1) % self.rules.track_len;
                        let absolute = self.track_index(owner, next_distance);
                        if let Some(occupant_idx) = self.track_occupant_in(marbles, absolute) {
                            if occupant_idx != moving_idx
                                && self.is_protected(&marbles[occupant_idx])
                            {
                                return None;
                            }
                            if bulldozer && occupant_idx != moving_idx {
                                push_unique(&mut captures, occupant_idx);
                            }
                        }
                        MarbleLocation::Track {
                            distance: next_distance,
                        }
                    }
                }
            };
        }

        match location {
            MarbleLocation::Track { distance } => {
                let absolute = self.track_index(owner, distance);
                if let Some(occupant_idx) = self.track_occupant_in(marbles, absolute)
                    && occupant_idx != moving_idx
                {
                    if self.is_protected(&marbles[occupant_idx]) {
                        return None;
                    }
                    push_unique(&mut captures, occupant_idx);
                }
            }
            MarbleLocation::Home { slot } => {
                if self.home_slot_occupied(marbles, owner, slot, Some(moving_idx)) {
                    return None;
                }
            }
            MarbleLocation::Base => return None,
        }

        Some(MoveSimulation {
            destination: location,
            captures,
            forward_steps: steps,
            backward_steps: 0,
            entered_home,
        })
    }

    fn simulate_backward(
        &self,
        marbles: &[Marble],
        owner: usize,
        marble_index: usize,
        steps: u8,
    ) -> Option<MoveSimulation> {
        let moving_idx = Self::marble_vec_index_in(marbles, owner, marble_index);
        let mut distance = match marbles[moving_idx].location {
            MarbleLocation::Base | MarbleLocation::Home { .. } => return None,
            MarbleLocation::Track { distance } => distance,
        };

        for _ in 0..steps {
            distance = if distance == 0 {
                self.rules.track_len - 1
            } else {
                distance - 1
            };
            let absolute = self.track_index(owner, distance);
            if let Some(occupant_idx) = self.track_occupant_in(marbles, absolute)
                && occupant_idx != moving_idx
                && self.is_protected(&marbles[occupant_idx])
            {
                return None;
            }
        }

        let mut captures = Vec::new();
        let absolute = self.track_index(owner, distance);
        if let Some(occupant_idx) = self.track_occupant_in(marbles, absolute)
            && occupant_idx != moving_idx
        {
            if self.is_protected(&marbles[occupant_idx]) {
                return None;
            }
            captures.push(occupant_idx);
        }

        Some(MoveSimulation {
            destination: MarbleLocation::Track { distance },
            captures,
            forward_steps: 0,
            backward_steps: steps,
            entered_home: 0,
        })
    }

    fn can_enter(
        &self,
        marbles: &[Marble],
        owner: usize,
        marble_index: usize,
    ) -> Option<Vec<usize>> {
        let marble = marbles
            .iter()
            .find(|m| m.owner == owner && m.index == marble_index)?;
        if marble.location != MarbleLocation::Base {
            return None;
        }

        let spawn = self.rules.spawn_index(owner);
        if let Some(occupant_idx) = self.track_occupant_in(marbles, spawn) {
            if self.is_protected(&marbles[occupant_idx]) {
                return None;
            }
            return Some(vec![occupant_idx]);
        }

        Some(Vec::new())
    }

    fn swap_marbles(
        &mut self,
        owner: usize,
        marble_index: usize,
        target_player: usize,
        target_marble_index: usize,
    ) {
        let source_idx = self.marble_vec_index(owner, marble_index);
        let target_idx = self.marble_vec_index(target_player, target_marble_index);

        let source_abs = self.absolute_track_index(&self.marbles[source_idx]);
        let target_abs = self.absolute_track_index(&self.marbles[target_idx]);

        let source_new_distance = self.relative_distance(owner, target_abs);
        let target_new_distance = self.relative_distance(target_player, source_abs);

        self.marbles[source_idx].location = MarbleLocation::Track {
            distance: source_new_distance,
        };
        self.marbles[target_idx].location = MarbleLocation::Track {
            distance: target_new_distance,
        };
    }

    fn deal_round(&mut self) {
        self.refill_deck_if_needed();
        let hand_size = self.rules.hand_cycle[self.deal_round_index % self.rules.hand_cycle.len()];
        for player in 0..NUM_PLAYERS {
            while self.hands[player].len() < hand_size {
                self.refill_deck_if_needed();
                let Some(card) = self.deck.pop() else {
                    break;
                };
                self.hands[player].push(card);
            }
        }
        self.deal_round_index = (self.deal_round_index + 1) % self.rules.hand_cycle.len();
    }

    fn refill_deck_if_needed(&mut self) {
        if !self.deck.is_empty() {
            return;
        }
        if self.discard.is_empty() {
            return;
        }
        self.deck.append(&mut self.discard);
        self.rng.shuffle(&mut self.deck);
    }

    fn controlled_player(&self, player: usize) -> usize {
        let partner = self.partner(player);
        if self.player_finished(player) && !self.player_finished(partner) {
            partner
        } else {
            player
        }
    }

    fn partner(&self, player: usize) -> usize {
        (player + 2) % NUM_PLAYERS
    }

    fn next_player(&self, player: usize) -> usize {
        (player + 1) % NUM_PLAYERS
    }

    fn player_finished(&self, player: usize) -> bool {
        self.owned_marbles(player)
            .all(|m| matches!(m.location, MarbleLocation::Home { .. }))
    }

    fn compute_winner(&self) -> Option<Team> {
        for team in [Team::Even, Team::Odd] {
            let all_home = self
                .marbles
                .iter()
                .filter(|m| Team::of_player(m.owner) == team)
                .all(|m| matches!(m.location, MarbleLocation::Home { .. }));
            if all_home {
                return Some(team);
            }
        }
        None
    }

    fn team_rewards(&self, events: &StepEvents, winner: Option<Team>) -> [f32; 2] {
        let weights = &self.rules.reward_weights;
        let mut rewards = [0.0, 0.0];

        rewards[events.acting_team.index()] += f32::from(events.entered_from_base)
            * weights.enter_from_base
            + f32::from(events.forward_steps) * weights.forward_step
            + f32::from(events.entered_home) * weights.enter_home
            + f32::from(events.captures) * weights.capture
            + f32::from(events.burns) * weights.burn;

        if let Some(winning_team) = winner {
            rewards[winning_team.index()] += weights.terminal_win;
            let losing_team = match winning_team {
                Team::Even => Team::Odd,
                Team::Odd => Team::Even,
            };
            rewards[losing_team.index()] += weights.terminal_loss;
        }

        rewards
    }

    fn owned_marbles(&self, owner: usize) -> impl Iterator<Item = &Marble> {
        self.marbles.iter().filter(move |m| m.owner == owner)
    }

    fn marble_mut(&mut self, owner: usize, index: usize) -> &mut Marble {
        self.marbles
            .iter_mut()
            .find(|m| m.owner == owner && m.index == index)
            .expect("legal action references an existing marble")
    }

    fn marble_vec_index(&self, owner: usize, index: usize) -> usize {
        Self::marble_vec_index_in(&self.marbles, owner, index)
    }

    fn marble_vec_index_in(marbles: &[Marble], owner: usize, index: usize) -> usize {
        marbles
            .iter()
            .position(|m| m.owner == owner && m.index == index)
            .expect("legal action references an existing marble")
    }

    fn is_protected(&self, marble: &Marble) -> bool {
        self.rules.protect_spawn_blockades
            && marble.location == (MarbleLocation::Track { distance: 0 })
    }

    fn track_occupant_in(&self, marbles: &[Marble], track_index: u8) -> Option<usize> {
        marbles.iter().position(|marble| {
            let MarbleLocation::Track { distance } = marble.location else {
                return false;
            };
            self.track_index(marble.owner, distance) == track_index
        })
    }

    fn home_slot_occupied(
        &self,
        marbles: &[Marble],
        owner: usize,
        slot: u8,
        except_idx: Option<usize>,
    ) -> bool {
        marbles.iter().enumerate().any(|(idx, marble)| {
            Some(idx) != except_idx
                && marble.owner == owner
                && marble.location == (MarbleLocation::Home { slot })
        })
    }

    fn absolute_track_index(&self, marble: &Marble) -> u8 {
        let MarbleLocation::Track { distance } = marble.location else {
            panic!("absolute_track_index requires a track marble");
        };
        self.track_index(marble.owner, distance)
    }

    fn track_index(&self, owner: usize, distance: u8) -> u8 {
        (self.rules.spawn_index(owner) + distance) % self.rules.track_len
    }

    fn relative_distance(&self, owner: usize, absolute: u8) -> u8 {
        (absolute + self.rules.track_len - self.rules.spawn_index(owner)) % self.rules.track_len
    }

    fn observed_marbles(&self) -> Vec<ObservedMarble> {
        self.marbles
            .iter()
            .map(|marble| {
                let track_index = match marble.location {
                    MarbleLocation::Track { distance } => {
                        Some(self.track_index(marble.owner, distance))
                    }
                    _ => None,
                };
                ObservedMarble {
                    owner: marble.owner,
                    index: marble.index,
                    location: marble.location,
                    track_index,
                }
            })
            .collect()
    }
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: Rank) -> Card {
        Card::new(rank, Suit::Clubs)
    }

    fn suited(rank: Rank, suit: Suit) -> Card {
        Card::new(rank, suit)
    }

    fn game_with_hand(rank: Rank) -> Game {
        let mut game = Game::new(7, Rules::canonical_v1());
        game.hands = std::array::from_fn(|_| Vec::new());
        game.hands[0] = vec![card(rank)];
        game.current_player = 0;
        game
    }

    #[test]
    fn house_board_geometry_matches_clarified_rules() {
        let rules = Rules::canonical_v1();

        assert_eq!(rules.track_len, 76);
        assert_eq!(rules.spawn_index(0), 0);
        assert_eq!(rules.spawn_index(1), 19);
        assert_eq!(rules.spawn_index(2), 38);
        assert_eq!(rules.spawn_index(3), 57);
        assert_eq!(rules.home_entry_distance(), 74);
    }

    #[test]
    fn deal_cycle_is_four_four_five_then_reshuffle() {
        let mut game = Game::new(7, Rules::canonical_v1());

        assert!(game.hands.iter().all(|hand| hand.len() == 4));
        assert_eq!(game.deck.len(), 36);

        game.hands = std::array::from_fn(|_| Vec::new());
        game.deal_round();
        assert!(game.hands.iter().all(|hand| hand.len() == 4));
        assert_eq!(game.deck.len(), 20);

        game.hands = std::array::from_fn(|_| Vec::new());
        game.deal_round();
        assert!(game.hands.iter().all(|hand| hand.len() == 5));
        assert_eq!(game.deck.len(), 0);
    }

    #[test]
    fn entry_card_can_enter_from_base() {
        let game = game_with_hand(Rank::Ace);

        let actions = game.legal_actions();

        assert_eq!(actions.len(), MARBLES_PER_PLAYER);
        assert!(
            actions
                .iter()
                .all(|a| matches!(a.kind, ActionKind::Enter { .. }))
        );
    }

    #[test]
    fn forced_play_blocks_voluntary_burns() {
        let mut game = Game::new(7, Rules::canonical_v1());
        game.hands = std::array::from_fn(|_| Vec::new());
        game.hands[0] = vec![card(Rank::Ace), card(Rank::Two)];
        game.current_player = 0;

        let actions = game.legal_actions();

        assert!(!actions.iter().any(|a| matches!(a.kind, ActionKind::Burn)));
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, ActionKind::Enter { .. }))
        );
    }

    #[test]
    fn skip_next_blocks_burning_own_card() {
        let mut game = Game::new(7, Rules::canonical_v1());
        game.hands = std::array::from_fn(|_| Vec::new());
        game.hands[0] = vec![suited(Rank::Queen, Suit::Hearts), card(Rank::Two)];
        game.hands[1] = vec![card(Rank::Three)];
        game.current_player = 0;

        let actions = game.legal_actions();

        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, ActionKind::SkipNext))
        );
        assert!(!actions.iter().any(|a| matches!(a.kind, ActionKind::Burn)));
    }

    #[test]
    fn burn_choices_exist_only_when_no_card_is_playable() {
        let game = game_with_hand(Rank::Two);

        let actions = game.legal_actions();

        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0].kind, ActionKind::Burn));
    }

    #[test]
    fn moving_onto_any_unprotected_piece_sends_it_to_base() {
        let mut game = game_with_hand(Rank::Two);
        game.marbles[0].location = MarbleLocation::Track { distance: 3 };
        game.marbles[1].location = MarbleLocation::Track { distance: 5 };
        game.hands[0] = vec![card(Rank::Two)];

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Move {
                        owner: 0,
                        marble_index: 0,
                        steps: 2,
                        direction: Direction::Forward,
                        ..
                    }
                )
            })
            .expect("expected self-capture move")
            .id;

        let outcome = game.step(action_id).unwrap();

        assert_eq!(
            game.marbles[0].location,
            MarbleLocation::Track { distance: 5 }
        );
        assert_eq!(game.marbles[1].location, MarbleLocation::Base);
        assert_eq!(outcome.events.forward_steps, 2);
        assert_eq!(outcome.events.captures, 1);
    }

    #[test]
    fn home_entry_is_two_spaces_behind_spawn_and_forward_only() {
        let mut game = game_with_hand(Rank::Ace);
        game.marbles[0].location = MarbleLocation::Track { distance: 74 };
        game.hands[0] = vec![card(Rank::Ace)];

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Move {
                        owner: 0,
                        marble_index: 0,
                        steps: 1,
                        direction: Direction::Forward,
                        ..
                    }
                )
            })
            .expect("expected home entry move")
            .id;

        let outcome = game.step(action_id).unwrap();

        assert_eq!(game.marbles[0].location, MarbleLocation::Home { slot: 0 });
        assert_eq!(outcome.events.entered_home, 1);
    }

    #[test]
    fn forward_moves_branch_into_home_instead_of_passing_entry() {
        let mut game = game_with_hand(Rank::Five);
        game.marbles[0].location = MarbleLocation::Track { distance: 72 };
        game.hands[0] = vec![card(Rank::Five)];

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Move {
                        owner: 0,
                        marble_index: 0,
                        steps: 5,
                        direction: Direction::Forward,
                        ..
                    }
                )
            })
            .expect("expected forward five into home")
            .id;

        let outcome = game.step(action_id).unwrap();

        assert_eq!(game.marbles[0].location, MarbleLocation::Home { slot: 2 });
        assert_eq!(outcome.events.entered_home, 1);
    }

    #[test]
    fn four_can_back_up_near_home_but_cannot_leave_home_backwards() {
        let mut game = game_with_hand(Rank::Four);
        game.marbles[0].location = MarbleLocation::Track { distance: 0 };

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Move {
                        owner: 0,
                        marble_index: 0,
                        steps: 4,
                        direction: Direction::Backward,
                        ..
                    }
                )
            })
            .expect("expected backward four")
            .id;
        game.step(action_id).unwrap();
        assert_eq!(
            game.marbles[0].location,
            MarbleLocation::Track { distance: 72 }
        );

        game.hands = std::array::from_fn(|_| Vec::new());
        game.hands[game.current_player] = vec![card(Rank::Four)];
        game.marbles[4].location = MarbleLocation::Home { slot: 1 };

        let actions = game.legal_actions();
        assert!(!actions.iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 1,
                    direction: Direction::Backward,
                    ..
                }
            )
        }));
    }

    #[test]
    fn home_moves_must_not_overshoot_or_share_pockets() {
        let mut game = game_with_hand(Rank::Three);
        game.marbles[0].location = MarbleLocation::Home { slot: 2 };

        let actions = game.legal_actions();

        assert!(
            actions
                .iter()
                .all(|a| !matches!(a.kind, ActionKind::Move { .. }))
        );

        game.hands[0] = vec![card(Rank::Ace)];
        game.marbles[0].location = MarbleLocation::Home { slot: 1 };
        game.marbles[1].location = MarbleLocation::Home { slot: 2 };
        let actions = game.legal_actions();
        assert!(!actions.iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 0,
                    marble_index: 0,
                    steps: 1,
                    ..
                }
            )
        }));
    }

    #[test]
    fn blockade_blocks_passing_landing_spawning_and_swapping() {
        let mut game = game_with_hand(Rank::King);
        game.marbles[0].location = MarbleLocation::Track { distance: 0 };
        game.marbles[4].location = MarbleLocation::Track { distance: 56 };
        game.hands[0] = vec![card(Rank::King)];

        assert!(!game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 1,
                    marble_index: 0,
                    ..
                }
            )
        }));

        game.hands[0] = vec![card(Rank::Jack)];
        assert!(!game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Swap {
                    target_player: 0,
                    target_marble_index: 0,
                    ..
                }
            )
        }));

        game.hands[0] = vec![card(Rank::Ace)];
        assert!(!game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Enter {
                    owner: 0,
                    marble_index: 1
                }
            )
        }));
    }

    #[test]
    fn jack_cannot_swap_own_spawn_blockade() {
        let mut game = game_with_hand(Rank::Jack);
        game.marbles[0].location = MarbleLocation::Track { distance: 0 };
        game.marbles[4].location = MarbleLocation::Track { distance: 10 };

        assert!(!game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Swap {
                    owner: 0,
                    marble_index: 0,
                    ..
                }
            )
        }));
    }

    #[test]
    fn owner_can_move_blockade_off_spawn() {
        let mut game = game_with_hand(Rank::Two);
        game.marbles[0].location = MarbleLocation::Track { distance: 0 };

        assert!(game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 0,
                    marble_index: 0,
                    steps: 2,
                    direction: Direction::Forward,
                    ..
                }
            )
        }));
    }

    #[test]
    fn spawning_kills_unprotected_enemy_on_spawn() {
        let mut game = game_with_hand(Rank::Ace);
        game.marbles[4].location = MarbleLocation::Track { distance: 57 };

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Enter {
                        owner: 0,
                        marble_index: 0
                    }
                )
            })
            .expect("expected enter action")
            .id;

        game.step(action_id).unwrap();

        assert_eq!(
            game.marbles[0].location,
            MarbleLocation::Track { distance: 0 }
        );
        assert_eq!(game.marbles[4].location, MarbleLocation::Base);
    }

    #[test]
    fn king_bulldozer_kills_passed_pieces_and_destination() {
        let mut game = game_with_hand(Rank::King);
        game.marbles[0].location = MarbleLocation::Track { distance: 10 };
        game.marbles[4].location = MarbleLocation::Track { distance: 72 };
        game.marbles[8].location = MarbleLocation::Track { distance: 61 };
        game.hands[0] = vec![card(Rank::King)];

        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|a| {
                matches!(
                    a.kind,
                    ActionKind::Move {
                        owner: 0,
                        marble_index: 0,
                        steps: 13,
                        bulldozer: true,
                        ..
                    }
                )
            })
            .expect("expected bulldozer")
            .id;

        let outcome = game.step(action_id).unwrap();

        assert_eq!(
            game.marbles[0].location,
            MarbleLocation::Track { distance: 23 }
        );
        assert_eq!(game.marbles[4].location, MarbleLocation::Base);
        assert_eq!(game.marbles[8].location, MarbleLocation::Base);
        assert_eq!(outcome.events.captures, 2);
    }

    #[test]
    fn five_can_move_any_piece_forward() {
        let mut game = game_with_hand(Rank::Five);
        game.marbles[8].location = MarbleLocation::Track { distance: 10 };

        assert!(game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 2,
                    marble_index: 0,
                    steps: 5,
                    direction: Direction::Forward,
                    ..
                }
            )
        }));
    }

    #[test]
    fn five_cannot_move_someone_elses_spawn_blockade() {
        let mut game = game_with_hand(Rank::Five);
        game.marbles[8].location = MarbleLocation::Track { distance: 0 };

        assert!(!game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 2,
                    marble_index: 0,
                    steps: 5,
                    direction: Direction::Forward,
                    ..
                }
            )
        }));
    }

    #[test]
    fn five_can_move_own_spawn_blockade() {
        let mut game = game_with_hand(Rank::Five);
        game.marbles[0].location = MarbleLocation::Track { distance: 0 };

        assert!(game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 0,
                    marble_index: 0,
                    steps: 5,
                    direction: Direction::Forward,
                    ..
                }
            )
        }));
    }

    #[test]
    fn seven_can_move_normally_or_split_two_distinct_marbles_in_order() {
        let mut game = game_with_hand(Rank::Seven);
        game.marbles[0].location = MarbleLocation::Track { distance: 10 };
        game.marbles[1].location = MarbleLocation::Track { distance: 20 };

        let actions = game.legal_actions();

        assert!(actions.iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 0,
                    marble_index: 0,
                    steps: 7,
                    ..
                }
            )
        }));
        assert!(actions.iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::SplitSeven {
                    first: MoveLeg {
                        owner: 0,
                        marble_index: 0,
                        steps: 3,
                    },
                    second: MoveLeg {
                        owner: 0,
                        marble_index: 1,
                        steps: 4,
                    }
                }
            )
        }));
    }

    #[test]
    fn ten_and_red_queen_can_reveal_discard_and_skip_next_player() {
        let mut game = game_with_hand(Rank::Ten);
        game.marbles[0].location = MarbleLocation::Track { distance: 10 };
        game.hands[0] = vec![card(Rank::Ten)];
        game.hands[1] = vec![card(Rank::Two), card(Rank::Three)];

        let skip_id = game
            .legal_actions()
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::SkipNext))
            .expect("expected ten skip")
            .id;
        let outcome = game.step(skip_id).unwrap();

        assert_eq!(outcome.next_player, 2);
        assert_eq!(outcome.forced_discard.map(|(p, _)| p), Some(1));
        assert_eq!(game.hands[1].len(), 1);

        let mut queen_game = game_with_hand(Rank::Queen);
        queen_game.hands[0] = vec![suited(Rank::Queen, Suit::Hearts)];
        queen_game.hands[1] = vec![card(Rank::Two)];
        assert!(
            queen_game
                .legal_actions()
                .iter()
                .any(|a| matches!(a.kind, ActionKind::SkipNext))
        );

        queen_game.hands[0] = vec![suited(Rank::Queen, Suit::Spades)];
        assert!(
            !queen_game
                .legal_actions()
                .iter()
                .any(|a| matches!(a.kind, ActionKind::SkipNext))
        );
    }

    #[test]
    fn skip_is_illegal_when_next_player_has_no_cards() {
        let mut game = game_with_hand(Rank::Ten);
        game.hands[0] = vec![card(Rank::Ten)];
        game.hands[1] = Vec::new();

        assert!(
            !game
                .legal_actions()
                .iter()
                .any(|a| matches!(a.kind, ActionKind::SkipNext))
        );
    }

    #[test]
    fn finished_player_moves_partner_for_own_piece_cards() {
        let mut game = game_with_hand(Rank::Two);
        for marble in game.marbles.iter_mut().filter(|m| m.owner == 0) {
            marble.location = MarbleLocation::Home {
                slot: marble.index as u8,
            };
        }
        game.marbles[8].location = MarbleLocation::Track { distance: 10 };

        assert!(game.legal_actions().iter().any(|a| {
            matches!(
                a.kind,
                ActionKind::Move {
                    owner: 2,
                    marble_index: 0,
                    steps: 2,
                    ..
                }
            )
        }));
    }

    #[test]
    fn observation_hides_other_hands_and_deck_order() {
        let game = Game::new(42, Rules::canonical_v1());

        let obs = game.observation(0).unwrap();

        assert_eq!(obs.own_hand, game.hands[0]);
        assert_eq!(obs.hand_sizes[1], game.hands[1].len());
        assert_eq!(obs.deck_remaining, game.deck.len());
        assert!(!obs.own_hand.iter().any(|c| game.hands[1].contains(c)));
    }

    #[test]
    fn hidden_hand_targets_are_training_only() {
        let game = Game::new(42, Rules::canonical_v1());

        let targets = game.hidden_hand_targets(0).unwrap();

        for (slot, target) in targets.iter().enumerate() {
            let player = slot + 1;
            for card in &game.hands[player] {
                assert_eq!(target[card.id()], 1);
            }
            for card in &game.hands[0] {
                assert_eq!(target[card.id()], 0);
            }
        }
    }

    #[test]
    fn deterministic_replay_from_seed_and_action_ids() {
        let mut first = Game::new(12345, Rules::canonical_v1());
        let mut second = Game::new(12345, Rules::canonical_v1());

        for _ in 0..25 {
            let legal = first.legal_actions();
            if legal.is_empty() {
                break;
            }
            let action_id = legal.len() / 2;
            let first_outcome = first.step(action_id).unwrap();
            let second_outcome = second.step(action_id).unwrap();

            assert_eq!(first_outcome, second_outcome);
            assert_eq!(first.observed_marbles(), second.observed_marbles());
            assert_eq!(first.current_player(), second.current_player());
            assert_eq!(first.winner(), second.winner());
        }
    }
}

use crate::Rank;

#[derive(Clone, Debug)]
pub struct RewardWeights {
    pub terminal_win: f32,
    pub terminal_loss: f32,
    pub enter_from_base: f32,
    pub forward_step: f32,
    pub enter_home: f32,
    pub capture: f32,
    pub burn: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            terminal_win: 1.0,
            terminal_loss: -1.0,
            enter_from_base: 0.02,
            forward_step: 0.001,
            enter_home: 0.05,
            capture: 0.02,
            burn: -0.005,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rules {
    pub track_len: u8,
    pub home_len: u8,
    pub spawn_spacing: u8,
    pub home_entry_back_steps: u8,
    pub hand_cycle: Vec<usize>,
    pub enter_ranks: Vec<Rank>,
    pub swap_ranks: Vec<Rank>,
    pub protect_spawn_blockades: bool,
    pub reward_weights: RewardWeights,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            track_len: 76,
            home_len: 4,
            spawn_spacing: 19,
            home_entry_back_steps: 2,
            hand_cycle: vec![4, 4, 5],
            enter_ranks: vec![Rank::Ace, Rank::King],
            swap_ranks: vec![Rank::Jack],
            protect_spawn_blockades: true,
            reward_weights: RewardWeights::default(),
        }
    }
}

impl Rules {
    pub fn canonical_v1() -> Self {
        Self::default()
    }

    pub fn validate(&self) {
        assert!(self.track_len >= 16, "track_len must be at least 16");
        assert_eq!(
            u16::from(self.spawn_spacing) * 4,
            u16::from(self.track_len),
            "four player spawns must divide the track evenly"
        );
        assert!(self.home_len > 0, "home_len must be positive");
        assert!(
            self.home_entry_back_steps > 0,
            "home_entry_back_steps must be positive"
        );
        assert!(
            self.home_entry_back_steps < self.track_len,
            "home entry must be on the main track"
        );
        assert!(!self.hand_cycle.is_empty(), "hand_cycle must not be empty");
        assert!(
            self.hand_cycle.iter().all(|size| *size > 0),
            "hand sizes must be positive"
        );
    }

    pub fn spawn_index(&self, player: usize) -> u8 {
        (player as u8) * self.spawn_spacing
    }

    pub fn home_entry_distance(&self) -> u8 {
        self.track_len - self.home_entry_back_steps
    }

    pub fn can_enter_from_base(&self, rank: Rank) -> bool {
        self.enter_ranks.contains(&rank)
    }

    pub fn can_swap(&self, rank: Rank) -> bool {
        self.swap_ranks.contains(&rank)
    }
}

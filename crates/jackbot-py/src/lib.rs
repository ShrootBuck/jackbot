use jackbot_engine::{
    Game, LegalAction, MarbleLocation, NUM_PLAYERS, Observation, Rules, StepEvents, Team,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

pub const OBS_SIZE: usize = 222;
pub const ACTION_SIZE: usize = 96;
pub const BELIEF_SIZE: usize = 156;
const MARBLES_PER_PLAYER: usize = 4;
const NONE_INDEX: usize = 4;

#[pyclass]
pub struct BatchEnv {
    envs: Vec<Game>,
    base_seed: u64,
    episodes: Vec<u64>,
}

struct BatchArrays {
    obs: Vec<Vec<f32>>,
    action_features: Vec<Vec<f32>>,
    action_offsets: Vec<i64>,
    env_ids: Vec<i64>,
    current_players: Vec<i64>,
    belief_targets: Vec<Vec<f32>>,
}

#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=1))]
    pub fn new(num_envs: usize, seed: u64) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        Ok(Self {
            envs: make_envs(num_envs, seed),
            base_seed: seed,
            episodes: vec![0; num_envs],
        })
    }

    #[getter]
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    #[pyo3(signature = (seed=None))]
    pub fn reset(&mut self, py: Python<'_>, seed: Option<u64>) -> PyResult<Py<PyAny>> {
        if let Some(seed) = seed {
            self.base_seed = seed;
        }
        self.envs = make_envs(self.envs.len(), self.base_seed);
        self.episodes.fill(0);

        let arrays = self.snapshot()?;
        batch_to_dict(
            py,
            arrays,
            vec![0.0; self.envs.len()],
            vec![vec![0.0, 0.0]; self.envs.len()],
            vec![false; self.envs.len()],
            vec![-1; self.envs.len()],
            vec![0; self.envs.len()],
            vec![0; self.envs.len()],
        )
    }

    #[pyo3(signature = (action_indices, shaping_scale=1.0))]
    pub fn step(
        &mut self,
        py: Python<'_>,
        action_indices: &Bound<'_, PyAny>,
        shaping_scale: f32,
    ) -> PyResult<Py<PyAny>> {
        let action_indices: Vec<usize> = action_indices.call_method0("tolist")?.extract()?;
        if action_indices.len() != self.envs.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} action indices, got {}",
                self.envs.len(),
                action_indices.len()
            )));
        }

        let mut rewards = Vec::with_capacity(self.envs.len());
        let mut team_rewards = Vec::with_capacity(self.envs.len());
        let mut dones = Vec::with_capacity(self.envs.len());
        let mut winners = Vec::with_capacity(self.envs.len());
        let mut acting_players = Vec::with_capacity(self.envs.len());
        let mut acting_teams = Vec::with_capacity(self.envs.len());

        for (env_index, action_index) in action_indices.into_iter().enumerate() {
            let game = &mut self.envs[env_index];
            let legal_count = game.legal_actions().len();
            if action_index >= legal_count {
                return Err(PyValueError::new_err(format!(
                    "env {env_index} action index {action_index} is invalid; {legal_count} legal actions exist"
                )));
            }

            let acting_player = game.current_player();
            let acting_team = Team::of_player(acting_player).index();
            let outcome = game
                .step(action_index)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let scaled_team_rewards = scaled_team_rewards(
                game.rules(),
                &outcome.events,
                outcome.winner,
                shaping_scale,
            );
            let winner = outcome.winner.map_or(-1, |team| team.index() as i64);
            let done = outcome.winner.is_some();

            rewards.push(scaled_team_rewards[acting_team]);
            team_rewards.push(vec![scaled_team_rewards[0], scaled_team_rewards[1]]);
            dones.push(done);
            winners.push(winner);
            acting_players.push(acting_player as i64);
            acting_teams.push(acting_team as i64);

            if done {
                self.episodes[env_index] += 1;
                let seed = episode_seed(self.base_seed, env_index, self.episodes[env_index]);
                self.envs[env_index] = Game::new(seed, Rules::canonical_v1());
            }
        }

        let arrays = self.snapshot()?;
        batch_to_dict(
            py,
            arrays,
            rewards,
            team_rewards,
            dones,
            winners,
            acting_players,
            acting_teams,
        )
    }
}

#[pymodule]
fn _jackbot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchEnv>()?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ACTION_SIZE", ACTION_SIZE)?;
    m.add("BELIEF_SIZE", BELIEF_SIZE)?;
    Ok(())
}

impl BatchEnv {
    fn snapshot(&self) -> PyResult<BatchArrays> {
        let mut obs = Vec::with_capacity(self.envs.len());
        let mut action_features = Vec::new();
        let mut action_offsets = Vec::with_capacity(self.envs.len() + 1);
        let mut env_ids = Vec::new();
        let mut current_players = Vec::with_capacity(self.envs.len());
        let mut belief_targets = Vec::with_capacity(self.envs.len());

        action_offsets.push(0);
        for (env_index, game) in self.envs.iter().enumerate() {
            let player = game.current_player();
            current_players.push(player as i64);
            let observation = game
                .observation(player)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            obs.push(encode_observation(&observation, game.rules()));

            let legal_actions = game.legal_actions();
            for action in &legal_actions {
                action_features.push(encode_action_features(player, action));
                env_ids.push(env_index as i64);
            }
            action_offsets.push(action_features.len() as i64);

            let targets = game
                .hidden_hand_targets(player)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            belief_targets.push(flatten_targets(targets));
        }

        Ok(BatchArrays {
            obs,
            action_features,
            action_offsets,
            env_ids,
            current_players,
            belief_targets,
        })
    }
}

fn make_envs(num_envs: usize, seed: u64) -> Vec<Game> {
    (0..num_envs)
        .map(|env_index| Game::new(episode_seed(seed, env_index, 0), Rules::canonical_v1()))
        .collect()
}

fn episode_seed(base_seed: u64, env_index: usize, episode: u64) -> u64 {
    base_seed
        .wrapping_add((env_index as u64).wrapping_mul(0x9e37_79b9))
        .wrapping_add(episode.wrapping_mul(1_000_003))
}

#[allow(clippy::too_many_arguments)]
fn batch_to_dict(
    py: Python<'_>,
    arrays: BatchArrays,
    rewards: Vec<f32>,
    team_rewards: Vec<Vec<f32>>,
    dones: Vec<bool>,
    winners: Vec<i64>,
    acting_players: Vec<i64>,
    acting_teams: Vec<i64>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("obs", array_f32_2(py, arrays.obs)?)?;
    dict.set_item("action_features", array_f32_2(py, arrays.action_features)?)?;
    dict.set_item("action_offsets", array_i64_1(py, arrays.action_offsets)?)?;
    dict.set_item("env_ids", array_i64_1(py, arrays.env_ids)?)?;
    dict.set_item("current_players", array_i64_1(py, arrays.current_players)?)?;
    dict.set_item("belief_targets", array_f32_2(py, arrays.belief_targets)?)?;
    dict.set_item("rewards", array_f32_1(py, rewards)?)?;
    dict.set_item("team_rewards", array_f32_2(py, team_rewards)?)?;
    dict.set_item("dones", array_bool_1(py, dones)?)?;
    dict.set_item("winners", array_i64_1(py, winners)?)?;
    dict.set_item("acting_players", array_i64_1(py, acting_players)?)?;
    dict.set_item("acting_teams", array_i64_1(py, acting_teams)?)?;
    Ok(dict.into_any().unbind())
}

fn numpy_array(py: Python<'_>, data: impl for<'py> IntoPyObject<'py>, dtype: &str) -> PyResult<Py<PyAny>> {
    let numpy = PyModule::import(py, "numpy")?;
    let array = numpy.call_method1("asarray", (data,))?;
    let array = array.call_method1("astype", (dtype,))?;
    Ok(array.unbind())
}

fn array_f32_1(py: Python<'_>, data: Vec<f32>) -> PyResult<Py<PyAny>> {
    numpy_array(py, data, "float32")
}

fn array_f32_2(py: Python<'_>, data: Vec<Vec<f32>>) -> PyResult<Py<PyAny>> {
    numpy_array(py, data, "float32")
}

fn array_i64_1(py: Python<'_>, data: Vec<i64>) -> PyResult<Py<PyAny>> {
    numpy_array(py, data, "int64")
}

fn array_bool_1(py: Python<'_>, data: Vec<bool>) -> PyResult<Py<PyAny>> {
    numpy_array(py, data, "bool")
}

fn encode_observation(observation: &Observation, rules: &Rules) -> Vec<f32> {
    let mut values = Vec::with_capacity(OBS_SIZE);

    let mut hand = [0.0; 52];
    for card in &observation.own_hand {
        hand[card.id()] = 1.0;
    }
    values.extend(hand);

    values.extend(
        observation
            .discard_counts
            .iter()
            .map(|count| f32::from(*count).min(4.0) / 4.0),
    );

    for offset in 0..NUM_PLAYERS {
        let player = (observation.player + offset) % NUM_PLAYERS;
        values.push(observation.hand_sizes[player] as f32 / 5.0);
    }
    values.push(observation.deck_remaining as f32 / 52.0);
    values.push((observation.turn_index as f32 / 512.0).min(1.0));

    for owner_offset in 0..NUM_PLAYERS {
        let owner = (observation.player + owner_offset) % NUM_PLAYERS;
        for marble_index in 0..MARBLES_PER_PLAYER {
            let marble = observation
                .marbles
                .iter()
                .find(|marble| marble.owner == owner && marble.index == marble_index)
                .expect("observation contains every marble");
            match marble.location {
                MarbleLocation::Base => {
                    values.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                }
                MarbleLocation::Track { distance } => {
                    let absolute = f32::from(marble.track_index.expect("track marble has index"))
                        / f32::from(rules.track_len - 1);
                    values.extend([
                        0.0,
                        1.0,
                        0.0,
                        f32::from(distance) / f32::from(rules.track_len - 1),
                        absolute,
                        0.0,
                        if distance == 0 { 1.0 } else { 0.0 },
                    ]);
                }
                MarbleLocation::Home { slot } => {
                    values.extend([
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        f32::from(slot) / f32::from(rules.home_len - 1),
                        0.0,
                    ]);
                }
            }
        }
    }

    debug_assert_eq!(values.len(), OBS_SIZE);
    values
}

fn encode_action_features(player: usize, action: &LegalAction) -> Vec<f32> {
    let mut values = Vec::with_capacity(ACTION_SIZE);
    let mut card = [0.0; 52];
    if let Some(card_id) = action.features.card_id {
        card[card_id] = 1.0;
    }
    values.extend(card);
    values.push(
        action
            .features
            .card_rank_value
            .map_or(0.0, |rank| f32::from(rank) / 13.0),
    );

    push_one_hot(&mut values, action.features.action_type as usize, 7);
    push_optional_player(&mut values, player, action.features.source_owner);
    push_optional_index(&mut values, action.features.source_marble);
    push_optional_player(&mut values, player, action.features.target_owner);
    push_optional_index(&mut values, action.features.target_marble);
    push_optional_player(&mut values, player, action.features.second_owner);
    push_optional_index(&mut values, action.features.second_marble);
    values.push(f32::from(action.features.steps) / 13.0);
    values.push(f32::from(action.features.second_steps) / 7.0);
    values.push(if action.features.starts_in_base { 1.0 } else { 0.0 });
    values.push(if action.features.ends_in_home { 1.0 } else { 0.0 });
    values.push(if action.features.captures { 1.0 } else { 0.0 });
    values.push(if action.features.bulldozer { 1.0 } else { 0.0 });

    debug_assert_eq!(values.len(), ACTION_SIZE);
    values
}

fn push_optional_player(values: &mut Vec<f32>, perspective: usize, player: Option<usize>) {
    let index = player.map_or(NONE_INDEX, |player| relative_player(perspective, player));
    push_one_hot(values, index, 5);
}

fn push_optional_index(values: &mut Vec<f32>, index: Option<usize>) {
    push_one_hot(values, index.unwrap_or(NONE_INDEX), 5);
}

fn push_one_hot(values: &mut Vec<f32>, index: usize, size: usize) {
    for item in 0..size {
        values.push(if item == index { 1.0 } else { 0.0 });
    }
}

fn relative_player(perspective: usize, player: usize) -> usize {
    (player + NUM_PLAYERS - perspective) % NUM_PLAYERS
}

fn flatten_targets(targets: [[u8; 52]; 3]) -> Vec<f32> {
    targets
        .into_iter()
        .flat_map(|target| target.into_iter().map(f32::from))
        .collect()
}

fn scaled_team_rewards(
    rules: &Rules,
    events: &StepEvents,
    winner: Option<Team>,
    shaping_scale: f32,
) -> [f32; 2] {
    let weights = &rules.reward_weights;
    let mut rewards = [0.0, 0.0];
    rewards[events.acting_team.index()] += shaping_scale
        * (f32::from(events.entered_from_base) * weights.enter_from_base
            + f32::from(events.forward_steps) * weights.forward_step
            + f32::from(events.entered_home) * weights.enter_home
            + f32::from(events.captures) * weights.capture
            + f32::from(events.burns) * weights.burn);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_encoder_has_fixed_size() {
        let game = Game::new(1, Rules::canonical_v1());
        let observation = game.observation(game.current_player()).unwrap();

        assert_eq!(encode_observation(&observation, game.rules()).len(), OBS_SIZE);
    }

    #[test]
    fn action_encoder_has_fixed_size() {
        let game = Game::new(1, Rules::canonical_v1());
        let legal_actions = game.legal_actions();

        assert!(!legal_actions.is_empty());
        assert_eq!(
            encode_action_features(game.current_player(), &legal_actions[0]).len(),
            ACTION_SIZE
        );
    }

    #[test]
    fn batch_offsets_match_legal_action_count() {
        let env = BatchEnv::new(3, 1).unwrap();
        let arrays = env.snapshot().unwrap();

        assert_eq!(arrays.obs.len(), 3);
        assert_eq!(arrays.action_offsets.len(), 4);
        assert_eq!(
            *arrays.action_offsets.last().unwrap() as usize,
            arrays.action_features.len()
        );
        assert_eq!(arrays.env_ids.len(), arrays.action_features.len());
    }
}

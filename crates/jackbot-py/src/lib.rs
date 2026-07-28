use jackbot_engine::{
    ActionKind, Card, Direction, Game, GameState, LegalAction, MarbleLocation, MarbleState,
    MoveLeg, NUM_PLAYERS, Observation, PUBLIC_HISTORY_LIMIT, PublicAction, PublicDiscard, Rules,
    StepEvents, StepOutcome, Team,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyList, PyModule};

pub const OBS_SIZE: usize = 222;
pub const ACTION_SIZE: usize = 96;
pub const BELIEF_SIZE: usize = 156;
pub const PUBLIC_HISTORY_RECORD_SIZE: usize = 65;
pub const PUBLIC_HISTORY_SIZE: usize = PUBLIC_HISTORY_LIMIT * PUBLIC_HISTORY_RECORD_SIZE;
pub const ACTION_CONSEQUENCE_SIZE: usize = NUM_PLAYERS * MARBLES_PER_PLAYER * 7;
const MARBLES_PER_PLAYER: usize = 4;
const NONE_INDEX: usize = 4;

#[derive(Clone, Copy)]
struct FeatureSchema {
    name: &'static str,
    public_history: bool,
    action_consequences: bool,
}

impl FeatureSchema {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "base_v1" => Ok(Self {
                name: "base_v1",
                public_history: false,
                action_consequences: false,
            }),
            "a1" => Ok(Self {
                name: "a1",
                public_history: false,
                action_consequences: true,
            }),
            "h1" => Ok(Self {
                name: "h1",
                public_history: true,
                action_consequences: false,
            }),
            "a1h1" => Ok(Self {
                name: "a1h1",
                public_history: true,
                action_consequences: true,
            }),
            _ => Err(PyValueError::new_err(format!(
                "unknown feature schema {name:?}; expected base_v1, a1, h1, or a1h1"
            ))),
        }
    }

    fn history_size(self) -> usize {
        if self.public_history {
            PUBLIC_HISTORY_SIZE
        } else {
            0
        }
    }

    fn consequence_size(self) -> usize {
        if self.action_consequences {
            ACTION_CONSEQUENCE_SIZE
        } else {
            0
        }
    }
}

#[pyclass]
pub struct BatchEnv {
    envs: Vec<Game>,
    base_seed: u64,
    episodes: Vec<u64>,
    game_steps: Vec<u64>,
    feature_schema: FeatureSchema,
}

#[pyclass]
pub struct PlayGame {
    game: Game,
    feature_schema: FeatureSchema,
}

struct BatchArrays {
    obs: Vec<f32>,
    action_features: Vec<f32>,
    action_offsets: Vec<i64>,
    env_ids: Vec<i64>,
    current_players: Vec<i64>,
    belief_targets: Vec<f32>,
    public_history: Vec<f32>,
    action_consequences: Vec<f32>,
    env_count: usize,
    action_count: usize,
    feature_schema: FeatureSchema,
}

#[pymethods]
impl PlayGame {
    #[new]
    #[pyo3(signature = (seed=1, feature_schema="base_v1"))]
    pub fn new(seed: u64, feature_schema: &str) -> PyResult<Self> {
        Ok(Self {
            game: Game::new(seed, Rules::canonical_v1()),
            feature_schema: FeatureSchema::parse(feature_schema)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (state, feature_schema="base_v1"))]
    pub fn from_state(state: &Bound<'_, PyAny>, feature_schema: &str) -> PyResult<Self> {
        let state = game_state_from_py(state)?;
        Ok(Self {
            game: Game::from_state(state, Rules::canonical_v1())
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
            feature_schema: FeatureSchema::parse(feature_schema)?,
        })
    }

    pub fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        game_state_to_py(py, self.game.state())
    }

    #[getter]
    pub fn current_player(&self) -> usize {
        self.game.current_player()
    }

    #[getter]
    pub fn turn_index(&self) -> u64 {
        self.game.turn_index()
    }

    pub fn winner(&self) -> Option<usize> {
        self.game.winner().map(Team::index)
    }

    pub fn copy(&self) -> Self {
        Self {
            game: self.game.clone(),
            feature_schema: self.feature_schema,
        }
    }

    pub fn batch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let arrays = snapshot_game(&self.game, 0, self.feature_schema)?;
        batch_to_dict(
            py,
            arrays,
            vec![0.0],
            vec![0.0, 0.0],
            vec![false],
            vec![-1],
            vec![-1],
            vec![0],
            vec![0],
            vec![0],
        )
    }

    pub fn turn_text(&self) -> PyResult<String> {
        let player = self.game.current_player();
        let observation = self
            .game
            .observation(player)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let labels =
            MarbleLabels::from_observation(&observation, self.game.rules().home_entry_distance());
        Ok(turn_text(&observation, &labels))
    }

    pub fn legal_action_labels(&self) -> PyResult<Vec<(usize, String)>> {
        let player = self.game.current_player();
        let observation = self
            .game
            .observation(player)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let labels =
            MarbleLabels::from_observation(&observation, self.game.rules().home_entry_distance());
        Ok(self
            .game
            .legal_actions()
            .iter()
            .map(|action| (action.id, action_label(action, &labels)))
            .collect())
    }

    pub fn legal_action_details(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let player = self.game.current_player();
        let observation = self
            .game
            .observation(player)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let labels =
            MarbleLabels::from_observation(&observation, self.game.rules().home_entry_distance());
        let details = PyList::empty(py);
        for action in self.game.legal_actions() {
            details.append(action_detail(py, &action, &labels)?)?;
        }
        Ok(details.into_any().unbind())
    }

    pub fn step(&mut self, action_id: usize) -> PyResult<String> {
        let outcome = self
            .game
            .step(action_id)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(outcome_text(&outcome))
    }
}

#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=1, feature_schema="base_v1"))]
    pub fn new(num_envs: usize, seed: u64, feature_schema: &str) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        Ok(Self {
            envs: make_envs(num_envs, seed),
            base_seed: seed,
            episodes: vec![0; num_envs],
            game_steps: vec![0; num_envs],
            feature_schema: FeatureSchema::parse(feature_schema)?,
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
        self.game_steps.fill(0);

        let arrays = self.snapshot()?;
        batch_to_dict(
            py,
            arrays,
            vec![0.0; self.envs.len()],
            vec![0.0; self.envs.len() * 2],
            vec![false; self.envs.len()],
            vec![-1; self.envs.len()],
            vec![-1; self.envs.len()],
            vec![0; self.envs.len()],
            vec![0; self.envs.len()],
            vec![0; self.envs.len()],
        )
    }

    pub fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("base_seed", self.base_seed)?;
        dict.set_item("episodes", self.episodes.clone())?;
        dict.set_item("game_steps", self.game_steps.clone())?;
        dict.set_item("feature_schema", self.feature_schema.name)?;
        let games = PyList::empty(py);
        for game in &self.envs {
            games.append(game_state_to_py(py, game.state())?)?;
        }
        dict.set_item("games", games)?;
        Ok(dict.into_any().unbind())
    }

    pub fn load_state(&mut self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dict = state.cast::<PyDict>()?;
        let schema_name: String = required_item(dict, "feature_schema")?.extract()?;
        let schema = FeatureSchema::parse(&schema_name)?;
        if schema.name != self.feature_schema.name {
            return Err(PyValueError::new_err(format!(
                "environment state uses {}, environment uses {}",
                schema.name, self.feature_schema.name
            )));
        }
        let episodes: Vec<u64> = required_item(dict, "episodes")?.extract()?;
        let game_steps: Vec<u64> = required_item(dict, "game_steps")?.extract()?;
        let raw_games = required_item(dict, "games")?;
        let game_items = raw_games.cast::<PyList>()?;
        if episodes.len() != self.envs.len()
            || game_steps.len() != self.envs.len()
            || game_items.len() != self.envs.len()
        {
            return Err(PyValueError::new_err(
                "environment state length does not match num_envs",
            ));
        }
        let games = game_items
            .iter()
            .map(|item| {
                let state = game_state_from_py(&item)?;
                Game::from_state(state, Rules::canonical_v1())
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.base_seed = required_item(dict, "base_seed")?.extract()?;
        self.episodes = episodes;
        self.game_steps = game_steps;
        self.envs = games;

        let arrays = self.snapshot()?;
        batch_to_dict(
            py,
            arrays,
            vec![0.0; self.envs.len()],
            vec![0.0; self.envs.len() * 2],
            vec![false; self.envs.len()],
            vec![-1; self.envs.len()],
            vec![-1; self.envs.len()],
            vec![0; self.envs.len()],
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
        let action_indices = extract_action_indices(py, action_indices)?;
        if action_indices.len() != self.envs.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} action indices, got {}",
                self.envs.len(),
                action_indices.len()
            )));
        }

        let mut rewards = Vec::with_capacity(self.envs.len());
        let mut team_rewards = Vec::with_capacity(self.envs.len() * 2);
        let mut dones = Vec::with_capacity(self.envs.len());
        let mut winners = Vec::with_capacity(self.envs.len());
        let mut winning_move_players = Vec::with_capacity(self.envs.len());
        let mut acting_players = Vec::with_capacity(self.envs.len());
        let mut acting_teams = Vec::with_capacity(self.envs.len());
        let mut game_lengths = Vec::with_capacity(self.envs.len());

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
            self.game_steps[env_index] += 1;
            let scaled_team_rewards =
                scaled_team_rewards(game.rules(), &outcome.events, outcome.winner, shaping_scale);
            let winner = outcome.winner.map_or(-1, |team| team.index() as i64);
            let done = outcome.winner.is_some();
            let winning_move_player = if done { acting_player as i64 } else { -1 };

            rewards.push(scaled_team_rewards[acting_team]);
            team_rewards.extend(scaled_team_rewards);
            dones.push(done);
            winners.push(winner);
            winning_move_players.push(winning_move_player);
            acting_players.push(acting_player as i64);
            acting_teams.push(acting_team as i64);
            game_lengths.push(if done {
                self.game_steps[env_index] as i64
            } else {
                0
            });

            if done {
                self.episodes[env_index] += 1;
                let seed = episode_seed(self.base_seed, env_index, self.episodes[env_index]);
                self.envs[env_index] = Game::new(seed, Rules::canonical_v1());
                self.game_steps[env_index] = 0;
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
            winning_move_players,
            acting_players,
            acting_teams,
            game_lengths,
        )
    }
}

#[pymodule]
fn _jackbot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchEnv>()?;
    m.add_class::<PlayGame>()?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ACTION_SIZE", ACTION_SIZE)?;
    m.add("BELIEF_SIZE", BELIEF_SIZE)?;
    m.add("PUBLIC_HISTORY_SIZE", PUBLIC_HISTORY_SIZE)?;
    m.add("ACTION_CONSEQUENCE_SIZE", ACTION_CONSEQUENCE_SIZE)?;
    Ok(())
}

impl BatchEnv {
    fn snapshot(&self) -> PyResult<BatchArrays> {
        let mut obs = Vec::with_capacity(self.envs.len() * OBS_SIZE);
        let mut action_features = Vec::new();
        let mut public_history =
            Vec::with_capacity(self.envs.len() * self.feature_schema.history_size());
        let mut action_consequences = Vec::new();
        let mut action_offsets = Vec::with_capacity(self.envs.len() + 1);
        let mut env_ids = Vec::new();
        let mut current_players = Vec::with_capacity(self.envs.len());
        let mut belief_targets = Vec::with_capacity(self.envs.len() * BELIEF_SIZE);
        let mut action_count = 0;

        action_offsets.push(0);
        for (env_index, game) in self.envs.iter().enumerate() {
            let player = game.current_player();
            current_players.push(player as i64);
            let observation = game
                .observation(player)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            obs.extend(encode_observation(&observation, game.rules()));
            if self.feature_schema.public_history {
                public_history.extend(encode_public_history(&observation));
            }

            let legal_actions = game.legal_actions();
            for action in &legal_actions {
                action_features.extend(encode_action_features(player, action));
                if self.feature_schema.action_consequences {
                    action_consequences.extend(encode_action_consequences(
                        player,
                        &game.action_result_marbles(action),
                        game.rules(),
                    ));
                }
                env_ids.push(env_index as i64);
            }
            action_count += legal_actions.len();
            action_offsets.push(action_count as i64);

            let targets = game
                .hidden_hand_targets(player)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            belief_targets.extend(flatten_targets(targets));
        }

        Ok(BatchArrays {
            obs,
            action_features,
            action_offsets,
            env_ids,
            current_players,
            belief_targets,
            public_history,
            action_consequences,
            env_count: self.envs.len(),
            action_count,
            feature_schema: self.feature_schema,
        })
    }
}

fn snapshot_game(
    game: &Game,
    env_index: usize,
    feature_schema: FeatureSchema,
) -> PyResult<BatchArrays> {
    let player = game.current_player();
    let observation = game
        .observation(player)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let legal_actions = game.legal_actions();
    let targets = game
        .hidden_hand_targets(player)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let action_count = legal_actions.len();

    Ok(BatchArrays {
        obs: encode_observation(&observation, game.rules()),
        action_features: legal_actions
            .iter()
            .flat_map(|action| encode_action_features(player, action))
            .collect(),
        action_offsets: vec![0, action_count as i64],
        env_ids: vec![env_index as i64; action_count],
        current_players: vec![player as i64],
        belief_targets: flatten_targets(targets),
        public_history: if feature_schema.public_history {
            encode_public_history(&observation)
        } else {
            Vec::new()
        },
        action_consequences: if feature_schema.action_consequences {
            legal_actions
                .iter()
                .flat_map(|action| {
                    encode_action_consequences(
                        player,
                        &game.action_result_marbles(action),
                        game.rules(),
                    )
                })
                .collect()
        } else {
            Vec::new()
        },
        env_count: 1,
        action_count,
        feature_schema,
    })
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
    team_rewards: Vec<f32>,
    dones: Vec<bool>,
    winners: Vec<i64>,
    winning_move_players: Vec<i64>,
    acting_players: Vec<i64>,
    acting_teams: Vec<i64>,
    game_lengths: Vec<i64>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let env_count = arrays.env_count;
    let action_count = arrays.action_count;
    dict.set_item(
        "obs",
        numpy_array(py, arrays.obs, "float32", &[env_count, OBS_SIZE])?,
    )?;
    dict.set_item(
        "action_features",
        numpy_array(
            py,
            arrays.action_features,
            "float32",
            &[action_count, ACTION_SIZE],
        )?,
    )?;
    dict.set_item(
        "action_offsets",
        numpy_array(py, arrays.action_offsets, "int64", &[env_count + 1])?,
    )?;
    dict.set_item(
        "env_ids",
        numpy_array(py, arrays.env_ids, "int64", &[action_count])?,
    )?;
    dict.set_item(
        "current_players",
        numpy_array(py, arrays.current_players, "int64", &[env_count])?,
    )?;
    dict.set_item(
        "belief_targets",
        numpy_array(
            py,
            arrays.belief_targets,
            "float32",
            &[env_count, BELIEF_SIZE],
        )?,
    )?;
    dict.set_item(
        "public_history",
        numpy_array(
            py,
            arrays.public_history,
            "float32",
            &[env_count, arrays.feature_schema.history_size()],
        )?,
    )?;
    dict.set_item(
        "action_consequences",
        numpy_array(
            py,
            arrays.action_consequences,
            "float32",
            &[action_count, arrays.feature_schema.consequence_size()],
        )?,
    )?;
    dict.set_item("feature_schema", arrays.feature_schema.name)?;
    dict.set_item(
        "rewards",
        numpy_array(py, rewards, "float32", &[env_count])?,
    )?;
    dict.set_item(
        "team_rewards",
        numpy_array(py, team_rewards, "float32", &[env_count, 2])?,
    )?;
    dict.set_item("dones", numpy_bool_array(py, dones, &[env_count])?)?;
    dict.set_item("winners", numpy_array(py, winners, "int64", &[env_count])?)?;
    dict.set_item(
        "winning_move_players",
        numpy_array(py, winning_move_players, "int64", &[env_count])?,
    )?;
    dict.set_item(
        "acting_players",
        numpy_array(py, acting_players, "int64", &[env_count])?,
    )?;
    dict.set_item(
        "acting_teams",
        numpy_array(py, acting_teams, "int64", &[env_count])?,
    )?;
    dict.set_item(
        "game_lengths",
        numpy_array(py, game_lengths, "int64", &[env_count])?,
    )?;
    Ok(dict.into_any().unbind())
}

fn game_state_from_py(state: &Bound<'_, PyAny>) -> PyResult<GameState> {
    let dict = state.cast::<PyDict>()?;
    let rng_state = optional_item(dict, "rng_state")?
        .map(|value| value.extract::<u64>())
        .transpose()?
        .unwrap_or(1);
    let deck = card_ids_to_cards(required_item(dict, "deck")?.extract()?)?;
    let discard = card_ids_to_cards(required_item(dict, "discard")?.extract()?)?;
    let raw_hands: Vec<Vec<usize>> = required_item(dict, "hands")?.extract()?;
    if raw_hands.len() != NUM_PLAYERS {
        return Err(PyValueError::new_err(format!(
            "expected {NUM_PLAYERS} hands, got {}",
            raw_hands.len()
        )));
    }
    let hands_vec = raw_hands
        .into_iter()
        .map(card_ids_to_cards)
        .collect::<PyResult<Vec<_>>>()?;
    let hands: [Vec<Card>; NUM_PLAYERS] = hands_vec.try_into().map_err(|hands: Vec<_>| {
        PyValueError::new_err(format!("expected {NUM_PLAYERS} hands, got {}", hands.len()))
    })?;
    let raw_marbles: Vec<(usize, usize, String, u8)> = required_item(dict, "marbles")?.extract()?;
    let marbles = raw_marbles
        .into_iter()
        .map(|(owner, index, kind, value)| {
            let location = match kind.as_str() {
                "base" => MarbleLocation::Base,
                "track" => MarbleLocation::Track { distance: value },
                "home" => MarbleLocation::Home { slot: value },
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown marble location {kind:?}"
                    )));
                }
            };
            Ok(MarbleState {
                owner,
                index,
                location,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let winner = optional_item(dict, "winner")?
        .map(|value| value.extract::<Option<usize>>())
        .transpose()?
        .flatten()
        .map(team_from_index)
        .transpose()?;
    let public_history = match optional_item(dict, "public_history")? {
        Some(value) => value
            .cast::<PyList>()?
            .iter()
            .map(|item| public_action_from_py(&item))
            .collect::<PyResult<Vec<_>>>()?,
        None => Vec::new(),
    };

    Ok(GameState {
        rng_state,
        deck,
        discard,
        hands,
        marbles,
        current_player: required_item(dict, "current_player")?.extract()?,
        turn_index: required_item(dict, "turn_index")?.extract()?,
        deal_round_index: required_item(dict, "deal_round_index")?.extract()?,
        public_history,
        winner,
    })
}

fn game_state_to_py(py: Python<'_>, state: GameState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("rng_state", state.rng_state)?;
    dict.set_item("deck", card_ids(&state.deck))?;
    dict.set_item("discard", card_ids(&state.discard))?;
    dict.set_item(
        "hands",
        state
            .hands
            .iter()
            .map(|hand| card_ids(hand))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "marbles",
        state
            .marbles
            .iter()
            .map(|marble| {
                let (kind, value) = location_parts(marble.location);
                (marble.owner, marble.index, kind, value)
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("current_player", state.current_player)?;
    dict.set_item("turn_index", state.turn_index)?;
    dict.set_item("deal_round_index", state.deal_round_index)?;
    let public_history = PyList::empty(py);
    for action in &state.public_history {
        public_history.append(public_action_to_py(py, action)?)?;
    }
    dict.set_item("public_history", public_history)?;
    dict.set_item("winner", state.winner.map(Team::index))?;
    Ok(dict.into_any().unbind())
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("state missing `{key}`")))
}

fn optional_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key)
}

fn card_ids_to_cards(ids: Vec<usize>) -> PyResult<Vec<Card>> {
    ids.into_iter()
        .map(|id| {
            Card::from_id(id).ok_or_else(|| PyValueError::new_err(format!("invalid card id {id}")))
        })
        .collect()
}

fn card_ids(cards: &[Card]) -> Vec<usize> {
    cards.iter().map(|card| card.id()).collect()
}

fn team_from_index(index: usize) -> PyResult<Team> {
    match index {
        0 => Ok(Team::Even),
        1 => Ok(Team::Odd),
        _ => Err(PyValueError::new_err(format!("invalid team index {index}"))),
    }
}

fn location_parts(location: MarbleLocation) -> (&'static str, u8) {
    match location {
        MarbleLocation::Base => ("base", 0),
        MarbleLocation::Track { distance } => ("track", distance),
        MarbleLocation::Home { slot } => ("home", slot),
    }
}

fn public_action_from_py(value: &Bound<'_, PyAny>) -> PyResult<PublicAction> {
    let dict = value.cast::<PyDict>()?;
    let card = optional_usize(dict, "card")?
        .map(|id| {
            Card::from_id(id).ok_or_else(|| PyValueError::new_err(format!("invalid card id {id}")))
        })
        .transpose()?;
    let forced_discard = match optional_item(dict, "forced_discard")? {
        Some(value) if !value.is_none() => {
            let discard = value.cast::<PyDict>()?;
            let discard_card = optional_usize(discard, "card")?
                .map(|id| {
                    Card::from_id(id)
                        .ok_or_else(|| PyValueError::new_err(format!("invalid card id {id}")))
                })
                .transpose()?;
            Some(PublicDiscard {
                player: required_item(discard, "player")?.extract()?,
                card: discard_card,
            })
        }
        _ => None,
    };
    Ok(PublicAction {
        player: required_item(dict, "player")?.extract()?,
        card,
        kind: action_kind_from_py(dict)?,
        forced_discard,
    })
}

fn public_action_to_py<'py>(
    py: Python<'py>,
    action: &PublicAction,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("player", action.player)?;
    dict.set_item("card", action.card.map(Card::id))?;
    set_action_kind_fields(&dict, &action.kind)?;
    if let Some(discard) = &action.forced_discard {
        let forced = PyDict::new(py);
        forced.set_item("player", discard.player)?;
        forced.set_item("card", discard.card.map(Card::id))?;
        dict.set_item("forced_discard", forced)?;
    } else {
        dict.set_item("forced_discard", py.None())?;
    }
    Ok(dict)
}

fn action_kind_from_py(dict: &Bound<'_, PyDict>) -> PyResult<ActionKind> {
    let kind: String = required_item(dict, "kind")?.extract()?;
    match kind.as_str() {
        "enter" => Ok(ActionKind::Enter {
            owner: required_item(dict, "owner")?.extract()?,
            marble_index: required_item(dict, "marble")?.extract()?,
        }),
        "move" => {
            let direction: String = required_item(dict, "direction")?.extract()?;
            let direction = match direction.as_str() {
                "forward" => Direction::Forward,
                "backward" => Direction::Backward,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown move direction {direction:?}"
                    )));
                }
            };
            Ok(ActionKind::Move {
                owner: required_item(dict, "owner")?.extract()?,
                marble_index: required_item(dict, "marble")?.extract()?,
                steps: required_item(dict, "steps")?.extract()?,
                direction,
                bulldozer: optional_item(dict, "bulldozer")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(false),
            })
        }
        "split" => Ok(ActionKind::SplitSeven {
            first: MoveLeg {
                owner: required_item(dict, "owner")?.extract()?,
                marble_index: required_item(dict, "marble")?.extract()?,
                steps: required_item(dict, "steps")?.extract()?,
            },
            second: MoveLeg {
                owner: required_item(dict, "second_owner")?.extract()?,
                marble_index: required_item(dict, "second_marble")?.extract()?,
                steps: required_item(dict, "second_steps")?.extract()?,
            },
        }),
        "swap" => Ok(ActionKind::Swap {
            owner: required_item(dict, "owner")?.extract()?,
            marble_index: required_item(dict, "marble")?.extract()?,
            target_player: required_item(dict, "target_owner")?.extract()?,
            target_marble_index: required_item(dict, "target_marble")?.extract()?,
        }),
        "skip" => Ok(ActionKind::SkipNext),
        "burn" => Ok(ActionKind::Burn),
        "pass" => Ok(ActionKind::PassNoCards),
        _ => Err(PyValueError::new_err(format!(
            "unknown public action kind {kind:?}"
        ))),
    }
}

fn set_action_kind_fields(dict: &Bound<'_, PyDict>, kind: &ActionKind) -> PyResult<()> {
    match kind {
        ActionKind::Enter {
            owner,
            marble_index,
        } => {
            dict.set_item("kind", "enter")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
        }
        ActionKind::Move {
            owner,
            marble_index,
            steps,
            direction,
            bulldozer,
        } => {
            dict.set_item("kind", "move")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
            dict.set_item("steps", *steps)?;
            dict.set_item("direction", direction_label(*direction))?;
            dict.set_item("bulldozer", *bulldozer)?;
        }
        ActionKind::SplitSeven { first, second } => {
            dict.set_item("kind", "split")?;
            dict.set_item("owner", first.owner)?;
            dict.set_item("marble", first.marble_index)?;
            dict.set_item("steps", first.steps)?;
            dict.set_item("second_owner", second.owner)?;
            dict.set_item("second_marble", second.marble_index)?;
            dict.set_item("second_steps", second.steps)?;
        }
        ActionKind::Swap {
            owner,
            marble_index,
            target_player,
            target_marble_index,
        } => {
            dict.set_item("kind", "swap")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
            dict.set_item("target_owner", *target_player)?;
            dict.set_item("target_marble", *target_marble_index)?;
        }
        ActionKind::SkipNext => dict.set_item("kind", "skip")?,
        ActionKind::Burn => dict.set_item("kind", "burn")?,
        ActionKind::PassNoCards => dict.set_item("kind", "pass")?,
    }
    Ok(())
}

fn optional_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    optional_item(dict, key)?
        .map(|value| value.extract::<Option<usize>>())
        .transpose()
        .map(Option::flatten)
}

fn action_detail<'py>(
    py: Python<'py>,
    action: &LegalAction,
    labels: &MarbleLabels,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", action.id)?;
    dict.set_item("card", action.card.map(Card::id))?;
    dict.set_item("label", action_label(action, labels))?;
    match &action.kind {
        ActionKind::Enter {
            owner,
            marble_index,
        } => {
            dict.set_item("kind", "enter")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
        }
        ActionKind::Move {
            owner,
            marble_index,
            steps,
            direction,
            bulldozer,
        } => {
            dict.set_item("kind", "move")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
            dict.set_item("steps", *steps)?;
            dict.set_item("direction", direction_label(*direction))?;
            dict.set_item("bulldozer", *bulldozer)?;
        }
        ActionKind::SplitSeven { first, second } => {
            dict.set_item("kind", "split")?;
            dict.set_item("owner", first.owner)?;
            dict.set_item("marble", first.marble_index)?;
            dict.set_item("steps", first.steps)?;
            dict.set_item("second_owner", second.owner)?;
            dict.set_item("second_marble", second.marble_index)?;
            dict.set_item("second_steps", second.steps)?;
        }
        ActionKind::Swap {
            owner,
            marble_index,
            target_player,
            target_marble_index,
        } => {
            dict.set_item("kind", "swap")?;
            dict.set_item("owner", *owner)?;
            dict.set_item("marble", *marble_index)?;
            dict.set_item("target_owner", *target_player)?;
            dict.set_item("target_marble", *target_marble_index)?;
        }
        ActionKind::SkipNext => {
            dict.set_item("kind", "skip")?;
        }
        ActionKind::Burn => {
            dict.set_item("kind", "burn")?;
        }
        ActionKind::PassNoCards => {
            dict.set_item("kind", "pass")?;
        }
    }
    Ok(dict)
}

fn numpy_array<T>(
    py: Python<'_>,
    data: Vec<T>,
    dtype: &str,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    let numpy = PyModule::import(py, "numpy")?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * std::mem::size_of::<T>(),
        )
    };
    let buffer = PyByteArray::new(py, bytes);
    let array = numpy.call_method1("frombuffer", (buffer, dtype))?;
    let reshaped = array.call_method1("reshape", (shape.to_vec(),))?;
    Ok(reshaped.unbind())
}

fn numpy_bool_array(py: Python<'_>, data: Vec<bool>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    let bytes = data.into_iter().map(u8::from).collect::<Vec<_>>();
    numpy_array(py, bytes, "bool", shape)
}

fn extract_action_indices(
    py: Python<'_>,
    action_indices: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    if let Ok(buffer) = PyBuffer::<i64>::get(action_indices) {
        if let Some(slice) = buffer.as_slice(py) {
            return slice
                .iter()
                .map(|value| {
                    usize::try_from(value.get())
                        .map_err(|_| PyValueError::new_err("action index must be non-negative"))
                })
                .collect();
        }
        return buffer
            .to_vec(py)?
            .into_iter()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("action index must be non-negative"))
            })
            .collect();
    }
    if let Ok(buffer) = PyBuffer::<i32>::get(action_indices) {
        if let Some(slice) = buffer.as_slice(py) {
            return slice
                .iter()
                .map(|value| {
                    usize::try_from(value.get())
                        .map_err(|_| PyValueError::new_err("action index must be non-negative"))
                })
                .collect();
        }
        return buffer
            .to_vec(py)?
            .into_iter()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("action index must be non-negative"))
            })
            .collect();
    }
    action_indices.extract()
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
            push_marble_location(&mut values, marble.location, rules);
        }
    }

    debug_assert_eq!(values.len(), OBS_SIZE);
    values
}

fn encode_public_history(observation: &Observation) -> Vec<f32> {
    let mut values = Vec::with_capacity(PUBLIC_HISTORY_SIZE);
    for action in observation
        .public_history
        .iter()
        .rev()
        .take(PUBLIC_HISTORY_LIMIT)
    {
        values.push(1.0);
        push_one_hot(
            &mut values,
            relative_player(observation.player, action.player),
            NUM_PLAYERS,
        );
        push_card_rank_and_suit(&mut values, action.card);

        let (action_type, primary, secondary, steps, second_steps, bulldozer) = match &action.kind {
            ActionKind::Enter {
                owner,
                marble_index,
            } => (0, Some((*owner, *marble_index)), None, 0, 0, false),
            ActionKind::Move {
                owner,
                marble_index,
                steps,
                direction,
                bulldozer,
            } => (
                1,
                Some((*owner, *marble_index)),
                None,
                match direction {
                    Direction::Forward => *steps as i8,
                    Direction::Backward => -(*steps as i8),
                },
                0,
                *bulldozer,
            ),
            ActionKind::SplitSeven { first, second } => (
                2,
                Some((first.owner, first.marble_index)),
                Some((second.owner, second.marble_index)),
                first.steps as i8,
                second.steps as i8,
                false,
            ),
            ActionKind::Swap {
                owner,
                marble_index,
                target_player,
                target_marble_index,
            } => (
                3,
                Some((*owner, *marble_index)),
                Some((*target_player, *target_marble_index)),
                0,
                0,
                false,
            ),
            ActionKind::SkipNext => (4, None, None, 0, 0, false),
            ActionKind::Burn => (5, None, None, 0, 0, false),
            ActionKind::PassNoCards => (6, None, None, 0, 0, false),
        };
        push_one_hot(&mut values, action_type, 7);
        push_history_marble(&mut values, observation.player, primary);
        push_history_marble(&mut values, observation.player, secondary);
        values.push(f32::from(steps) / 13.0);
        values.push(f32::from(second_steps) / 7.0);
        values.push(if bulldozer { 1.0 } else { 0.0 });
        push_card_rank_and_suit(
            &mut values,
            action
                .forced_discard
                .as_ref()
                .and_then(|discard| discard.card),
        );
    }
    values.resize(PUBLIC_HISTORY_SIZE, 0.0);
    debug_assert_eq!(values.len(), PUBLIC_HISTORY_SIZE);
    values
}

fn encode_action_consequences(player: usize, marbles: &[MarbleState], rules: &Rules) -> Vec<f32> {
    let mut values = Vec::with_capacity(ACTION_CONSEQUENCE_SIZE);
    for owner_offset in 0..NUM_PLAYERS {
        let owner = (player + owner_offset) % NUM_PLAYERS;
        for marble_index in 0..MARBLES_PER_PLAYER {
            let marble = marbles
                .iter()
                .find(|marble| marble.owner == owner && marble.index == marble_index)
                .expect("action result contains every marble");
            push_marble_location(&mut values, marble.location, rules);
        }
    }
    debug_assert_eq!(values.len(), ACTION_CONSEQUENCE_SIZE);
    values
}

fn push_marble_location(values: &mut Vec<f32>, location: MarbleLocation, rules: &Rules) {
    match location {
        MarbleLocation::Base => values.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        MarbleLocation::Track { distance } => values.extend([
            0.0,
            1.0,
            0.0,
            f32::from(distance) / f32::from(rules.track_len - 1),
            0.0,
            0.0,
            if distance == 0 { 1.0 } else { 0.0 },
        ]),
        MarbleLocation::Home { slot } => values.extend([
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            f32::from(slot) / f32::from(rules.home_len - 1),
            0.0,
        ]),
    }
}

fn push_card_rank_and_suit(values: &mut Vec<f32>, card: Option<Card>) {
    if let Some(card) = card {
        push_one_hot(values, card.id() % 13, 13);
        push_one_hot(values, card.id() / 13, 4);
    } else {
        values.extend([0.0; 17]);
    }
}

fn push_history_marble(values: &mut Vec<f32>, perspective: usize, marble: Option<(usize, usize)>) {
    if let Some((owner, index)) = marble {
        push_one_hot(values, relative_player(perspective, owner), NUM_PLAYERS);
        push_one_hot(values, index, MARBLES_PER_PLAYER);
    } else {
        values.extend([0.0; NUM_PLAYERS + MARBLES_PER_PLAYER]);
    }
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
    values.push(if action.features.starts_in_base {
        1.0
    } else {
        0.0
    });
    values.push(if action.features.ends_in_home {
        1.0
    } else {
        0.0
    });
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

struct MarbleLabels {
    labels: [[usize; MARBLES_PER_PLAYER]; NUM_PLAYERS],
}

impl MarbleLabels {
    fn from_observation(observation: &Observation, home_entry_distance: u8) -> Self {
        let mut labels = [[0; MARBLES_PER_PLAYER]; NUM_PLAYERS];

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

    fn marble_label(&self, player: usize, marble_index: usize) -> String {
        format!(
            "P{} marble #{}",
            player + 1,
            self.labels[player][marble_index]
        )
    }

    fn ordinal(&self, player: usize, marble_index: usize) -> usize {
        self.labels[player][marble_index]
    }
}

fn turn_text(observation: &Observation, labels: &MarbleLabels) -> String {
    let mut lines = vec![
        format!(
            "Turn {} | {} to act | hand: {}",
            observation.turn_index,
            player_label(observation.current_player),
            cards(&observation.own_hand)
        ),
        format!(
            "Hand sizes: P1={} P2={} P3={} P4={} | deck left: {}",
            observation.hand_sizes[0],
            observation.hand_sizes[1],
            observation.hand_sizes[2],
            observation.hand_sizes[3],
            observation.deck_remaining
        ),
    ];
    lines.extend(board_lines(observation, labels));
    lines.join("\n")
}

fn board_lines(observation: &Observation, labels: &MarbleLabels) -> Vec<String> {
    (0..NUM_PLAYERS)
        .map(|player| {
            let mut marbles = observation
                .marbles
                .iter()
                .filter(|marble| marble.owner == player)
                .map(|marble| {
                    (
                        labels.ordinal(player, marble.index),
                        format!(
                            "#{}={}",
                            labels.ordinal(player, marble.index),
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
            format!("{}: {}", player_label(player), labels.join("  "))
        })
        .collect()
}

fn action_label(action: &LegalAction, labels: &MarbleLabels) -> String {
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

fn outcome_text(outcome: &StepOutcome) -> String {
    let card = outcome
        .played_card
        .map_or_else(|| "no card".to_string(), |card| card.to_string());
    let mut lines = vec![format!(
        "{} played {card}; next: {}",
        player_label(outcome.current_player),
        player_label(outcome.next_player)
    )];

    if let Some((player, card)) = outcome.forced_discard {
        lines.push(format!(
            "{} was skipped and randomly discarded {card}.",
            player_label(player)
        ));
    }

    if outcome.events.entered_from_base > 0 {
        lines.push("Entered from base.".to_string());
    }
    if outcome.events.forward_steps > 0 {
        lines.push(format!(
            "Moved forward {} total step(s).",
            outcome.events.forward_steps
        ));
    }
    if outcome.events.backward_steps > 0 {
        lines.push(format!(
            "Moved backward {} total step(s).",
            outcome.events.backward_steps
        ));
    }
    if outcome.events.entered_home > 0 {
        lines.push("Entered home.".to_string());
    }
    if outcome.events.captures > 0 {
        lines.push(format!("Captured {} marble(s).", outcome.events.captures));
    }
    if outcome.events.burns > 0 {
        lines.push("Burned a card.".to_string());
    }
    if let Some(winner) = outcome.winner {
        lines.push(format!("Winner: {}", team_label(winner)));
    }

    lines.join("\n")
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
        MarbleLocation::Track { distance: 0 } => {
            let absolute = track_index.expect("track marble should have track index");
            format!("spawn/b{absolute}")
        }
        MarbleLocation::Track { distance } => {
            let absolute = track_index.expect("track marble should have track index");
            format!("track d{distance}/b{absolute}")
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

fn player_label(player: usize) -> String {
    format!("P{}", player + 1)
}

fn team_label(team: Team) -> &'static str {
    match team {
        Team::Even => "P1 + P3",
        Team::Odd => "P2 + P4",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_encoder_has_fixed_size() {
        let game = Game::new(1, Rules::canonical_v1());
        let observation = game.observation(game.current_player()).unwrap();

        assert_eq!(
            encode_observation(&observation, game.rules()).len(),
            OBS_SIZE
        );
    }

    #[test]
    fn observation_encoder_does_not_expose_absolute_board_indexes() {
        let mut game = (1..100)
            .map(|seed| Game::new(seed, Rules::canonical_v1()))
            .find(|game| {
                game.legal_actions()
                    .iter()
                    .any(|action| matches!(action.kind, ActionKind::Enter { .. }))
            })
            .expect("expected a seed with an opening enter action");
        let action_id = game
            .legal_actions()
            .into_iter()
            .find(|action| matches!(action.kind, ActionKind::Enter { .. }))
            .unwrap()
            .id;
        game.step(action_id).unwrap();

        let observation = game.observation(game.current_player()).unwrap();
        let encoded = encode_observation(&observation, game.rules());
        let marble_features_start = 52 + 52 + NUM_PLAYERS + 1 + 1;
        let mut saw_track_marble = false;

        for marble in 0..(NUM_PLAYERS * MARBLES_PER_PLAYER) {
            let offset = marble_features_start + marble * 7;
            let is_track = encoded[offset + 1];
            if is_track == 1.0 {
                saw_track_marble = true;
                assert_eq!(encoded[offset + 4], 0.0);
            }
        }

        assert!(saw_track_marble);
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
        let env = BatchEnv::new(3, 1, "base_v1").unwrap();
        let arrays = env.snapshot().unwrap();

        assert_eq!(arrays.obs.len(), 3 * OBS_SIZE);
        assert_eq!(arrays.action_offsets.len(), 4);
        assert_eq!(
            *arrays.action_offsets.last().unwrap() as usize,
            arrays.action_count
        );
        assert_eq!(
            arrays.action_features.len(),
            arrays.action_count * ACTION_SIZE
        );
        assert_eq!(arrays.env_ids.len(), arrays.action_count);
    }

    #[test]
    fn enhanced_feature_schemas_have_fixed_sizes() {
        for (schema, history_size, consequence_size) in [
            ("base_v1", 0, 0),
            ("a1", 0, ACTION_CONSEQUENCE_SIZE),
            ("h1", PUBLIC_HISTORY_SIZE, 0),
            ("a1h1", PUBLIC_HISTORY_SIZE, ACTION_CONSEQUENCE_SIZE),
        ] {
            let env = BatchEnv::new(2, 1, schema).unwrap();
            let arrays = env.snapshot().unwrap();
            assert_eq!(arrays.public_history.len(), 2 * history_size);
            assert_eq!(
                arrays.action_consequences.len(),
                arrays.action_count * consequence_size
            );
        }
    }

    #[test]
    fn public_history_encoder_records_applied_action() {
        let mut game = Game::new(1, Rules::canonical_v1());
        let action = game.legal_actions()[0].id;
        game.step(action).unwrap();
        let observation = game.observation(game.current_player()).unwrap();
        let history = encode_public_history(&observation);

        assert_eq!(history.len(), PUBLIC_HISTORY_SIZE);
        assert_eq!(history[0], 1.0);
        assert!(history.iter().any(|value| *value != 0.0));
    }
}

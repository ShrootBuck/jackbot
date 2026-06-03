mod common;

use common::{
    MarbleLabels, action_label, parse_seed, print_legal_actions, print_outcome, print_turn, prompt,
    team_label,
};
use jackbot_engine::{Game, Rules};

fn main() {
    let seed = parse_seed();
    let mut game = Game::new(seed, Rules::canonical_v1());
    let mut policy_rng = ExampleRng::new(seed ^ 0xa11c_e5eed);

    println!("Jackbot random advisor");
    println!("Seed: {seed}");
    println!("Engine-only example; suggestions are random legal moves.");
    println!(
        "Press Enter to apply the suggested move, `l` to list all legal moves, or `q` to quit.\n"
    );

    loop {
        if let Some(winner) = game.winner() {
            println!("Game over. Winner: {}", team_label(winner));
            break;
        }

        let player = game.current_player();
        let observation = game
            .observation(player)
            .expect("current player should always be valid");
        let labels =
            MarbleLabels::from_observation(&observation, game.rules().home_entry_distance());
        let legal_actions = game.legal_actions();
        let recommended_index = policy_rng.gen_range(legal_actions.len());
        let recommended = &legal_actions[recommended_index];

        print_turn(&observation, &labels);
        println!("Random advice:");
        println!("  {}", action_label(recommended, &labels));

        loop {
            let Some(input) = prompt("apply? ") else {
                return;
            };
            match input.trim() {
                "" | "y" | "yes" => match game.step(recommended.id) {
                    Ok(outcome) => {
                        print_outcome(&outcome);
                        println!();
                        break;
                    }
                    Err(error) => {
                        println!("{error}");
                        return;
                    }
                },
                "l" | "legal" => {
                    print_legal_actions(&legal_actions, &labels);
                }
                "q" | "quit" | "exit" => return,
                _ => {
                    println!("Use Enter to apply, `l` to list legal moves, or `q` to quit.");
                }
            }
        }
    }
}

struct ExampleRng {
    state: u64,
}

impl ExampleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn gen_range(&mut self, upper_exclusive: usize) -> usize {
        assert!(
            upper_exclusive > 0,
            "random advisor needs at least one action"
        );
        (self.next_u64() as usize) % upper_exclusive
    }
}

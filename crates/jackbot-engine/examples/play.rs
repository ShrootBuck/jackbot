mod common;

use common::{
    MarbleLabels, action_label, parse_seed, player_label, print_legal_actions, print_outcome,
    print_turn, prompt, team_label,
};
use jackbot_engine::{Game, Rules};

fn main() {
    let seed = parse_seed();
    let mut game = Game::new(seed, Rules::canonical_v1());

    println!("Jackbot engine smoke checker");
    println!("Seed: {seed}");
    println!("Type a move number, `h` for help, or `q` to quit.\n");

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

        print_turn(&observation, &labels);
        println!("Legal moves:");
        for action in &legal_actions {
            println!("  {:>2}: {}", action.id, action_label(action, &labels));
        }

        let Some(input) = prompt("> ") else {
            break;
        };
        let input = input.trim();

        match input {
            "q" | "quit" | "exit" => break,
            "h" | "help" => {
                print_help();
                continue;
            }
            "l" | "legal" => {
                print_legal_actions(&legal_actions, &labels);
                continue;
            }
            "" => continue,
            _ => {}
        }

        let Ok(action_id) = input.parse::<usize>() else {
            println!("Not a move number. Type `h` for commands.\n");
            continue;
        };

        match game.step(action_id) {
            Ok(outcome) => {
                println!("{} selected action {action_id}.", player_label(player));
                print_outcome(&outcome);
                println!();
            }
            Err(error) => {
                println!("{error}\n");
            }
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  <number>  play that legal move");
    println!("  l         list legal moves again");
    println!("  h         show this help");
    println!("  q         quit");
    println!("This is an engine-generated game; use `jackbot-play` for checkpoint play.\n");
}

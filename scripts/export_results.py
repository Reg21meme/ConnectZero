"""
Run a full evaluation across checkpoint generations and print an Elo table.
Usage: python3 scripts/export_results.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from connectzero.model.network import ConnectZeroNet
from connectzero.model.checkpoint import load_checkpoint
from connectzero.eval.tournament import NetworkAgent, round_robin, play_match
from connectzero.eval.elo import compute_elo_ratings, print_elo_table
from connectzero.env.baselines import RandomAgent, HeuristicAgent, MinimaxAgent


CHECKPOINT_DIR = "runs/checkpoints"
NUM_SIMULATIONS = 50
NUM_GAMES = 20
SEED = 42


def load_network(path):
    net = ConnectZeroNet(num_res_blocks=4, channels=64)
    load_checkpoint(net, path)
    return net


def pick_checkpoints(checkpoint_dir, num_picks=6):
    """Pick evenly spaced checkpoints across all generations."""
    files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ])
    if not files:
        return []
    indices = [int(i * (len(files) - 1) / (num_picks - 1)) for i in range(num_picks)]
    return [os.path.join(checkpoint_dir, files[i]) for i in indices]


def main():
    print("=== ConnectZero Evaluation Report ===\n")

    # Pick evenly spaced checkpoints
    paths = pick_checkpoints(CHECKPOINT_DIR, num_picks=6)
    if not paths:
        print("No checkpoints found in", CHECKPOINT_DIR)
        return

    print("Checkpoints selected:")
    for p in paths:
        print(f"  {p}")

    # Build agents
    networks = [load_network(p) for p in paths]
    gen_names = [os.path.basename(p).replace(".pt", "") for p in paths]

    agents = [NetworkAgent(n, num_simulations=NUM_SIMULATIONS) for n in networks]
    agents += [RandomAgent(), HeuristicAgent(), MinimaxAgent(depth=2)]
    names = gen_names + ["Random", "Heuristic", "Minimax-d2"]

    # Round robin
    print(f"\nRunning round robin ({NUM_GAMES} games per matchup, seed={SEED})...")
    results = round_robin(agents, names, num_games=NUM_GAMES, seed=SEED)

    # Elo table
    ratings = compute_elo_ratings(results, names)
    print_elo_table(ratings)

    # Win rates for latest checkpoint vs baselines
    print("\n=== Latest Checkpoint vs Baselines ===")
    latest_agent = agents[len(networks) - 1]

    for baseline_name, baseline_agent in [
        ("Random", RandomAgent()),
        ("Heuristic", HeuristicAgent()),
        ("Minimax-d2", MinimaxAgent(depth=2)),
    ]:
        r = play_match(latest_agent, baseline_agent, num_games=100, seed=SEED)
        print(f"vs {baseline_name:<12} | W:{r['wins']:>3} L:{r['losses']:>3} D:{r['draws']:>3} | Win rate: {r['win_rate']:.1%}")


if __name__ == "__main__":
    main()
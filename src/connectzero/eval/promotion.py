from connectzero.eval.tournament import NetworkAgent, play_match


PROMOTION_THRESHOLD = 0.55
PROMOTION_GAMES = 200


def evaluate_candidate(champion_network, candidate_network, num_simulations=50, device="cpu"):
    """
    Play candidate vs champion over PROMOTION_GAMES games.
    Returns result dict and whether candidate should be promoted.
    """
    champion_agent = NetworkAgent(champion_network, num_simulations=num_simulations, device=device)
    candidate_agent = NetworkAgent(candidate_network, num_simulations=num_simulations, device=device)

    result = play_match(candidate_agent, champion_agent, num_games=PROMOTION_GAMES, seed=42)
    promoted = result["win_rate"] > PROMOTION_THRESHOLD

    print(f"\n=== Promotion Gate ===")
    print(f"Candidate win rate: {result['win_rate']:.1%} (threshold: {PROMOTION_THRESHOLD:.0%})")
    print(f"W:{result['wins']} L:{result['losses']} D:{result['draws']}")
    print(f"Decision: {'PROMOTED' if promoted else 'REJECTED'}")

    return result, promoted


class ChampionRegistry:
    """Tracks the current champion and promotion history."""

    def __init__(self, initial_network, initial_path):
        self.champion_network = initial_network
        self.champion_path = initial_path
        self.history = []

    def evaluate_and_promote(self, candidate_network, candidate_path, num_simulations=50, device="cpu"):
        result, promoted = evaluate_candidate(
            self.champion_network,
            candidate_network,
            num_simulations=num_simulations,
            device=device,
        )

        entry = {
            "candidate_path": candidate_path,
            "win_rate": result["win_rate"],
            "wins": result["wins"],
            "losses": result["losses"],
            "draws": result["draws"],
            "promoted": promoted,
            "champion_path": candidate_path if promoted else self.champion_path,
        }
        self.history.append(entry)

        if promoted:
            self.champion_network = candidate_network
            self.champion_path = candidate_path
            print(f"New champion: {candidate_path}")
        else:
            print(f"Champion retained: {self.champion_path}")

        return promoted
import math


INITIAL_ELO = 1000
K_FACTOR = 32


def expected_score(rating_a, rating_b):
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, score_a):
    """
    Update Elo ratings after a match.
    score_a: 1.0 = win, 0.5 = draw, 0.0 = loss
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_a = rating_a + K_FACTOR * (score_a - expected_a)
    new_b = rating_b + K_FACTOR * (score_b - expected_b)
    return new_a, new_b


def compute_elo_ratings(round_robin_results, names, initial_rating=INITIAL_ELO):
    """
    Compute Elo ratings from round-robin match results.

    round_robin_results: dict of {(name_a, name_b): {wins, losses, draws}}
    names: list of agent names
    """
    ratings = {name: float(initial_rating) for name in names}

    for (name_a, name_b), result in round_robin_results.items():
        wins = result["wins"]
        losses = result["losses"]
        draws = result["draws"]
        total = wins + losses + draws

        # Treat each game individually
        for _ in range(wins):
            ratings[name_a], ratings[name_b] = update_elo(
                ratings[name_a], ratings[name_b], 1.0
            )
        for _ in range(losses):
            ratings[name_a], ratings[name_b] = update_elo(
                ratings[name_a], ratings[name_b], 0.0
            )
        for _ in range(draws):
            ratings[name_a], ratings[name_b] = update_elo(
                ratings[name_a], ratings[name_b], 0.5
            )

    return ratings


def print_elo_table(ratings):
    """Print a sorted Elo leaderboard."""
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print("\n=== Elo Ratings ===")
    print(f"{'Agent':<20} {'Elo':>8}")
    print("-" * 30)
    for name, elo in sorted_ratings:
        print(f"{name:<20} {elo:>8.1f}")
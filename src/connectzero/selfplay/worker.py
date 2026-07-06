import numpy as np
from connectzero.env.connect4 import Connect4
from connectzero.model.network import encode_board
from connectzero.mcts.search import MCTS
from connectzero.selfplay.trajectory import get_temperature, select_move


def play_game(network, num_simulations=50, c_puct=1.5, device="cpu"):
    """Play one self-play game. Returns list of training examples."""
    mcts = MCTS(network=network, num_simulations=num_simulations, c_puct=c_puct, device=device)
    game = Connect4()
    examples = []
    move_number = 0

    while not game.done:
        # Run MCTS to get visit distribution
        visit_counts, _ = mcts.search(game)

        # Get temperature for this move
        temp = get_temperature(move_number)

        # Record training example — outcome filled in later
        examples.append({
            "board": game.board.copy(),
            "current_player": game.current_player,
            "mcts_policy": visit_counts.copy(),
            "outcome": None,
        })

        # Select and play move
        col = select_move(visit_counts, temp)
        game.step(col)
        move_number += 1

    # Fill in outcomes from winner's perspective
    for ex in examples:
        if game.winner is None:
            ex["outcome"] = 0.0
        elif ex["current_player"] == game.winner:
            ex["outcome"] = 1.0
        else:
            ex["outcome"] = -1.0

    return examples
import ray
import numpy as np
from connectzero.env.connect4 import Connect4
from connectzero.model.network import encode_board, ConnectZeroNet
from connectzero.mcts.search import MCTS
from connectzero.selfplay.trajectory import get_temperature, select_move


def play_game(network, num_simulations=50, c_puct=1.5, device="cpu"):
    """Play one self-play game. Returns list of training examples."""
    mcts = MCTS(network=network, num_simulations=num_simulations, c_puct=c_puct, device=device)
    game = Connect4()
    examples = []
    move_number = 0

    while not game.done:
        visit_counts, _ = mcts.search(game)
        temp = get_temperature(move_number)
        examples.append({
            "board": game.board.copy(),
            "current_player": game.current_player,
            "mcts_policy": visit_counts.copy(),
            "outcome": None,
        })
        col = select_move(visit_counts, temp)
        game.step(col)
        move_number += 1

    for ex in examples:
        if game.winner is None:
            ex["outcome"] = 0.0
        elif ex["current_player"] == game.winner:
            ex["outcome"] = 1.0
        else:
            ex["outcome"] = -1.0

    return examples


@ray.remote
class SelfPlayWorker:
    """Ray actor that generates self-play games in parallel."""

    def __init__(self, num_simulations=50, num_res_blocks=4, channels=64, device="cpu"):
        self.num_simulations = num_simulations
        self.device = device
        self.network = ConnectZeroNet(num_res_blocks=num_res_blocks, channels=channels)

    def update_weights(self, state_dict):
        """Load latest network weights from the learner."""
        import torch
        self.network.load_state_dict(state_dict)

    def play_game(self):
        """Generate one self-play game and return examples."""
        return play_game(self.network, num_simulations=self.num_simulations, device=self.device)

    def play_games(self, n):
        """Generate n self-play games and return all examples."""
        all_examples = []
        for _ in range(n):
            all_examples.extend(play_game(
                self.network, num_simulations=self.num_simulations, device=self.device
            ))
        return all_examples
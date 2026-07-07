import numpy as np
import random
from connectzero.env.connect4 import Connect4
from connectzero.env.baselines import RandomAgent, HeuristicAgent
from connectzero.mcts.search import MCTS


class NetworkAgent:
    """Wraps a trained network + MCTS into the same interface as baseline agents."""

    def __init__(self, network, num_simulations=50, device="cpu"):
        self.mcts = MCTS(network=network, num_simulations=num_simulations, device=device)
        self._game = None

    def select_move(self, board, legal_moves, player):
        _, action = self.mcts.search(self._game)
        return action

    def set_game(self, game):
        self._game = game


def play_match(agent1, agent2, num_games=100, seed=None):
    """
    Play num_games between agent1 (P1) and agent2 (P2).
    Alternates who goes first every game.
    Returns win/loss/draw counts from agent1's perspective.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        game = Connect4()

        if i % 2 == 0:
            agents = {1: agent1, 2: agent2}
            agent1_player = 1
        else:
            agents = {1: agent2, 2: agent1}
            agent1_player = 2

        while not game.done:
            current = game.current_player
            agent = agents[current]
            if hasattr(agent, "set_game"):
                agent.set_game(game)
            col = agent.select_move(game.board, game.legal_moves(), current)
            game.step(col)

        if game.winner is None:
            draws += 1
        elif game.winner == agent1_player:
            wins += 1
        else:
            losses += 1

    total = wins + losses + draws
    win_rate = wins / total
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def round_robin(agents, names, num_games=50, seed=42):
    """
    Play every agent against every other agent.
    Returns results dict: {(name_a, name_b): match_result}
    """
    results = {}
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue
            result = play_match(agents[i], agents[j], num_games=num_games, seed=seed)
            results[(names[i], names[j])] = result
            print(f"  {names[i]} vs {names[j]}: W{result['wins']} L{result['losses']} D{result['draws']} ({result['win_rate']:.1%})")
    return results


def evaluate_vs_random(network, num_games=100, num_simulations=50, device="cpu"):
    net_agent = NetworkAgent(network, num_simulations=num_simulations, device=device)
    rand_agent = RandomAgent()
    results = play_match(net_agent, rand_agent, num_games=num_games, seed=42)
    print(f"vs Random    | W:{results['wins']} L:{results['losses']} D:{results['draws']} | Win rate: {results['win_rate']:.1%}")
    return results


def evaluate_vs_heuristic(network, num_games=100, num_simulations=50, device="cpu"):
    net_agent = NetworkAgent(network, num_simulations=num_simulations, device=device)
    heur_agent = HeuristicAgent()
    results = play_match(net_agent, heur_agent, num_games=num_games, seed=42)
    print(f"vs Heuristic | W:{results['wins']} L:{results['losses']} D:{results['draws']} | Win rate: {results['win_rate']:.1%}")
    return results
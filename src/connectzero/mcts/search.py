import numpy as np
from connectzero.mcts.node import MCTSNode


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection.

    Each call to search() runs num_simulations iterations from the
    given game state and returns a move probability distribution
    (the policy target for training) and the selected move.
    """

    def __init__(self, network, num_simulations=50, c_puct=1.5, device="cpu"):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game):
        """
        Run MCTS from the current game state.
        """
        root = MCTSNode(game=game.copy())

        # Expand root immediately so it has children to select from
        policy, value = self.network.predict(
            game.board, game.current_player, game.legal_moves(), self.device
        )
        root.expand(policy)
        root.N = 1
        root.W = value

        for _ in range(self.num_simulations - 1):
            self._simulate(root)

        return self._policy_from_visits(root), self._select_action(root)

    def _simulate(self, root):
        """One full MCTS simulation: select -> evaluate -> expand -> backup."""

        # --- 1. Select ---
        node = root
        while not node.is_leaf() and not node.is_terminal():
            node = node.best_child(self.c_puct)

        # --- 2. Evaluate ---
        if node.is_terminal():
            # Value is exact: the player who just moved won (+1),
            # or it's a draw (0). game.winner tells us who won.
            if node.game.winner is None:
                value = 0.0
            else:
                # The node's current_player is the one who is about to move,
                # meaning the previous player just won.
                value = -1.0
        else:
            policy, value = self.network.predict(
                node.game.board,
                node.game.current_player,
                node.game.legal_moves(),
                self.device,
            )
            # --- 3. Expand ---
            node.expand(policy)

        # --- 4. Backup ---
        node.backup(value)

    def _policy_from_visits(self, root):
        """
        Convert root children visit counts to a probability distribution.
        Columns with no child node get visit count 0.
        """
        visits = np.zeros(7, dtype=np.float32)
        for col, child in root.children.items():
            visits[col] = child.N
        total = visits.sum()
        if total > 0:
            visits /= total
        return visits

    def _select_action(self, root):
        """Return the column with the highest visit count."""
        return max(root.children.keys(), key=lambda col: root.children[col].N)
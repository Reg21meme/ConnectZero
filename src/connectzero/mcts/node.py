import math
import numpy as np
from connectzero.env.connect4 import Connect4


class MCTSNode:
    """
    A single node in the MCTS search tree.

    Each node represents a board position reached by playing
    a specific move (edge) from the parent node.

    Attributes:
        game:      Connect4 instance at this node's position
        parent:    parent MCTSNode, or None if root
        move:      column played to reach this node, or None if root
        prior:     P(s,a) — network prior probability for this move
        children:  dict mapping column -> MCTSNode
        N:         visit count
        W:         total value backed up through this node
    """

    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game = game          # Connect4 state at this node
        self.parent = parent      # parent node (None for root)
        self.move = move          # move (column) that led here
        self.prior = prior        # network prior P(s,a)

        self.children = {}        # col -> MCTSNode
        self.N = 0                # visit count
        self.W = 0.0              # total backed-up value

    @property
    def Q(self):
        """Mean value estimate. 0 if never visited."""
        if self.N == 0:
            return 0.0
        return self.W / self.N

    def is_leaf(self):
        """True if this node has no expanded children yet."""
        return len(self.children) == 0

    def is_terminal(self):
        """True if the game at this node is over."""
        return self.game.done

    def puct_score(self, c_puct=1.5):
        """
        PUCT score used by the parent to select which child to visit.

        PUCT = Q + C * P * sqrt(parent_N) / (1 + N)

        Higher score = more likely to be selected.
        """
        parent_N = self.parent.N if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return -self.Q + exploration

    def best_child(self, c_puct=1.5):
        """Return the child with the highest PUCT score."""
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))

    def expand(self, policy):
        """
        Create child nodes for all legal moves.

        Args:
            policy: np.array of shape (7,) — network prior probabilities
                    for each column. Illegal moves should already be zeroed.
        """
        for col in self.game.legal_moves():
            game_copy = self.game.copy()
            game_copy.step(col)
            self.children[col] = MCTSNode(
                game=game_copy,
                parent=self,
                move=col,
                prior=float(policy[col]),
            )

    def backup(self, value):
        """
        Walk up the tree from this node to the root,
        incrementing N and adding value to W at each step.

        Value is negated at each level because the parent is the
        opposite player — a good position for the child is bad for the parent.
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            value = -value      # flip perspective for parent
            node = node.parent

    def __repr__(self):
        return (
            f"MCTSNode(move={self.move}, N={self.N}, "
            f"Q={self.Q:.3f}, prior={self.prior:.3f}, "
            f"children={list(self.children.keys())})"
        )
import random
import numpy as np
from connectzero.env.connect4 import ROWS, COLS, EMPTY


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_row(board, col):
    """Return the lowest empty row in col, or None if full."""
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == EMPTY:
            return row
    return None


def _would_win(board, col, player):
    """Return True if player placing a piece in col wins immediately."""
    row = _drop_row(board, col)
    if row is None:
        return False
    b = board.copy()
    b[row, col] = player
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and b[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a random legal move."""

    def select_move(self, board, legal_moves, player):
        return random.choice(legal_moves)


class HeuristicAgent:
    """
    Priority order:
      1. Win immediately if possible.
      2. Block opponent winning immediately.
      3. Prefer columns closest to center.
    """

    def select_move(self, board, legal_moves, player):
        opponent = 2 if player == 1 else 1

        # Priority 1: take the win
        for col in legal_moves:
            if _would_win(board, col, player):
                return col

        # Priority 2: block opponent
        for col in legal_moves:
            if _would_win(board, col, opponent):
                return col

        # Priority 3: prefer center
        return min(legal_moves, key=lambda c: abs(c - COLS // 2))
import random
import numpy as np
from connectzero.env.connect4 import ROWS, COLS, EMPTY, P1, P2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_row(board, col):
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == EMPTY:
            return row
    return None


def _would_win(board, col, player):
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


def _legal_moves(board):
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def _apply_move(board, col, player):
    b = board.copy()
    row = _drop_row(b, col)
    b[row, col] = player
    return b


def _check_win(board, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != player:
                continue
            for dr, dc in directions:
                count = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS and board[nr, nc] == player:
                    count += 1
                    nr += dr
                    nc += dc
                if count >= 4:
                    return True
    return False


def _heuristic_score(board, player):
    """Simple center-preference score for minimax leaf evaluation."""
    opponent = P2 if player == P1 else P1
    if _check_win(board, player):
        return 1000
    if _check_win(board, opponent):
        return -1000
    # Center preference
    center_col = COLS // 2
    center_count = np.sum(board[:, center_col] == player)
    return int(center_count)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent:
    def select_move(self, board, legal_moves, player):
        return random.choice(legal_moves)


class HeuristicAgent:
    def select_move(self, board, legal_moves, player):
        opponent = P2 if player == P1 else P1

        for col in legal_moves:
            if _would_win(board, col, player):
                return col

        for col in legal_moves:
            if _would_win(board, col, opponent):
                return col

        return min(legal_moves, key=lambda c: abs(c - COLS // 2))


class MinimaxAgent:
    """
    Minimax search to a fixed depth with alpha-beta pruning.
    depth=2 looks 2 moves ahead (one for each player).
    """

    def __init__(self, depth=2):
        self.depth = depth

    def select_move(self, board, legal_moves, player):
        best_col = legal_moves[0]
        best_score = float("-inf")

        for col in legal_moves:
            b = _apply_move(board, col, player)
            score = self._minimax(b, self.depth - 1, float("-inf"), float("inf"), False, player)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def _minimax(self, board, depth, alpha, beta, maximizing, original_player):
        opponent = P2 if original_player == P1 else P1
        current_player = original_player if maximizing else opponent

        if _check_win(board, original_player):
            return 1000 + depth
        if _check_win(board, opponent):
            return -1000 - depth

        moves = _legal_moves(board)
        if depth == 0 or not moves:
            return _heuristic_score(board, original_player)

        if maximizing:
            best = float("-inf")
            for col in moves:
                b = _apply_move(board, col, original_player)
                score = self._minimax(b, depth - 1, alpha, beta, False, original_player)
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = float("inf")
            for col in moves:
                b = _apply_move(board, col, opponent)
                score = self._minimax(b, depth - 1, alpha, beta, True, original_player)
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best
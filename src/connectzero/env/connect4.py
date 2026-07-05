import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = 2


class Connect4:
    """Connect Four rules engine.

    Board is a 6x7 numpy array.
    Row 0 is the top row, row 5 is the bottom row.
    Pieces fall due to gravity — a dropped piece lands in the lowest empty row.

    Players: 1 and 2.
    Rewards are from the perspective of the player who just moved:
        +1.0  win
         0.0  draw
        None  game continues
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to a fresh game. Returns the initial board."""
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = P1
        self.done = False
        self.winner = None
        return self.board.copy()

    def legal_moves(self):
        """Return list of columns that are not full (top row is empty)."""
        return [c for c in range(COLS) if self.board[0, c] == EMPTY]

    def step(self, col):
        """
        Drop a piece in col for the current player.

        Returns:
            board   (np.ndarray) copy of the board after the move
            reward  (float | None) +1.0 win, 0.0 draw, None if game continues
            done    (bool) whether the game has ended
            info    (dict) {"winner": 1 | 2 | None}
        """
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")
        if col not in self.legal_moves():
            raise ValueError(f"Column {col} is full or out of range.")

        row = self._drop_row(col)
        self.board[row, col] = self.current_player

        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1.0, True, {"winner": self.current_player}

        if not self.legal_moves():
            self.done = True
            return self.board.copy(), 0.0, True, {"winner": None}

        self.current_player = P2 if self.current_player == P1 else P1
        return self.board.copy(), None, False, {"winner": None}

    def _drop_row(self, col):
        """Return the lowest empty row index in col (gravity)."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                return row
        raise ValueError(f"Column {col} is full — should have been caught by legal_moves().")

    def _check_win(self, row, col):
        """Check if the piece at (row, col) creates 4 in a row for current_player."""
        player = self.board[row, col]
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def copy(self):
        """Return a deep copy of the game state (used by MCTS)."""
        g = Connect4.__new__(Connect4)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.done = self.done
        g.winner = self.winner
        return g

    def render(self):
        """Print the board to stdout."""
        symbols = {EMPTY: ".", P1: "X", P2: "O"}
        print("  " + " ".join(str(c) for c in range(COLS)))
        for row in range(ROWS):
            print("  " + " ".join(symbols[self.board[row, c]] for c in range(COLS)))
        print(f"  Player to move: {'X (1)' if self.current_player == P1 else 'O (2)'}")
        print()
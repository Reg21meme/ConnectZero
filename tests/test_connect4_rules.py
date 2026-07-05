import pytest
import numpy as np
from connectzero.env.connect4 import Connect4, ROWS, COLS, P1, P2, EMPTY
from connectzero.env.baselines import RandomAgent, HeuristicAgent


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

def test_legal_moves_full_board():
    """Fresh board has all 7 columns legal."""
    g = Connect4()
    assert g.legal_moves() == list(range(COLS))


def test_legal_moves_full_column():
    """A completely filled column is excluded from legal moves."""
    g = Connect4()
    for _ in range(ROWS):
        g.board[_, 0] = P1
    assert 0 not in g.legal_moves()


def test_legal_moves_after_fill():
    """Filling a column via step removes it from legal moves."""
    g = Connect4()
    for _ in range(ROWS):
        g.step(0)
    assert 0 not in g.legal_moves()


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

def test_gravity_first_piece():
    """First piece in a column lands at the bottom row."""
    g = Connect4()
    g.step(3)
    assert g.board[ROWS - 1, 3] == P1


def test_gravity_stacking():
    """Second piece lands directly above the first."""
    g = Connect4()
    g.step(3)  # P1 lands at row 5
    g.step(3)  # P2 lands at row 4
    assert g.board[ROWS - 1, 3] == P1
    assert g.board[ROWS - 2, 3] == P2


def test_cannot_drop_in_full_column():
    """Stepping into a full column raises ValueError."""
    g = Connect4()
    for _ in range(ROWS):
        g.step(0)
    with pytest.raises(ValueError):
        g.step(0)


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

def test_horizontal_win():
    """Four in a row horizontally triggers a win."""
    g = Connect4()
    # P1 plays cols 0,1,2,3 — P2 plays col 4 each time to alternate
    for col in range(3):
        g.step(col)   # P1
        g.step(4)     # P2
    _, reward, done, info = g.step(3)  # P1 wins
    assert done is True
    assert reward == 1.0
    assert info["winner"] == P1


def test_vertical_win():
    """Four in a column vertically triggers a win."""
    g = Connect4()
    for _ in range(3):
        g.step(0)   # P1
        g.step(1)   # P2
    _, reward, done, info = g.step(0)  # P1 wins
    assert done is True
    assert reward == 1.0
    assert info["winner"] == P1


def test_diagonal_win():
    """Four in a diagonal triggers a win."""
    g = Connect4()
    # Build a diagonal for P1 at (5,0),(4,1),(3,2),(2,3)
    moves = [
        (0, 1), (1, 1), (1, 2),  # setup
        (2, 1), (2, 2), (2, 3),  # setup
        (3, 1), (3, 2), (3, 3),  # setup
    ]
    for player_col, opp_col in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        pass  # replaced by direct board setup below

    # Set up board directly for clarity
    g = Connect4()
    g.board[5, 0] = P1
    g.board[4, 1] = P1
    g.board[3, 2] = P1
    # Fill below so piece lands correctly
    g.board[5, 3] = P2
    g.board[4, 3] = P2
    g.board[3, 3] = P2
    g.current_player = P1
    _, reward, done, info = g.step(3)  # P1 lands at row 2, col 3
    assert done is True
    assert reward == 1.0


def test_no_win_on_partial_row():
    """Three in a row is not a win."""
    g = Connect4()
    for col in range(2):
        g.step(col)   # P1
        g.step(4)     # P2
    g.step(2)          # P1 — three in a row, not four
    assert g.done is False


# ---------------------------------------------------------------------------
# Draw detection
# ---------------------------------------------------------------------------

def test_draw():
    """A full board with no winner returns reward 0.0 and done True."""
    g = Connect4()
    # Set up a board full except one cell, verified to have no 4-in-a-row
    g.board = np.array([
        [2, 1, 2, 1, 2, 1, 0],  # row 0 — last cell empty at col 6
        [1, 2, 1, 2, 1, 2, 1],  # row 1
        [2, 1, 2, 1, 2, 1, 2],  # row 2
        [2, 1, 2, 1, 2, 1, 2],  # row 3
        [1, 2, 1, 2, 1, 2, 1],  # row 4
        [1, 2, 1, 2, 1, 2, 1],  # row 5
    ], dtype=np.int8)
    g.current_player = P1
    _, reward, done, info = g.step(6)  # fills the last cell
    assert done is True
    assert reward == 0.0
    assert info["winner"] is None

# ---------------------------------------------------------------------------
# Reward symmetry
# ---------------------------------------------------------------------------

def test_reward_goes_to_winner_not_loser():
    """The +1 reward is returned to the player who just won, not the opponent."""
    g = Connect4()
    for col in range(3):
        g.step(col)   # P1
        g.step(4)     # P2
    _, reward, done, info = g.step(3)  # P1 wins
    assert reward == 1.0
    assert info["winner"] == P1


def test_game_over_raises():
    """Calling step after game over raises RuntimeError."""
    g = Connect4()
    for col in range(3):
        g.step(col)
        g.step(4)
    g.step(3)  # P1 wins
    with pytest.raises(RuntimeError):
        g.step(0)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_random_agent_returns_legal_move():
    """Random agent always returns a legal column."""
    from connectzero.env.baselines import RandomAgent
    agent = RandomAgent()
    g = Connect4()
    for _ in range(20):
        col = agent.select_move(g.board, g.legal_moves(), g.current_player)
        assert col in g.legal_moves()
        g.step(col)
        if g.done:
            break


def test_heuristic_blocks_win():
    """Heuristic agent blocks an opponent immediate win."""
    agent = HeuristicAgent()
    g = Connect4()
    # P1 has three in a row at cols 0,1,2 bottom row — P2 must block col 3
    g.board[5, 0] = P1
    g.board[5, 1] = P1
    g.board[5, 2] = P1
    g.current_player = P2
    col = agent.select_move(g.board, g.legal_moves(), P2)
    assert col == 3


def test_heuristic_takes_win():
    """Heuristic agent takes an immediate win over blocking."""
    agent = HeuristicAgent()
    g = Connect4()
    # P2 can win at col 3, but opponent also threatens — P2 should win
    g.board[5, 0] = P2
    g.board[5, 1] = P2
    g.board[5, 2] = P2
    g.board[5, 4] = P1
    g.board[5, 5] = P1
    g.board[5, 6] = P1
    g.current_player = P2
    col = agent.select_move(g.board, g.legal_moves(), P2)
    assert col == 3
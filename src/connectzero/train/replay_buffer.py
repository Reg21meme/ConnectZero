import random
import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add_game(self, examples):
        """Add all training examples from one self-play game."""
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size):
        """Sample a random batch. Returns arrays ready for PyTorch."""
        batch = random.sample(self.buffer, batch_size)

        boards = np.array([
            ex["board"] for ex in batch
        ], dtype=np.float32)

        players = np.array([
            ex["current_player"] for ex in batch
        ], dtype=np.int8)

        policies = np.array([
            ex["mcts_policy"] for ex in batch
        ], dtype=np.float32)

        outcomes = np.array([
            ex["outcome"] for ex in batch
        ], dtype=np.float32)

        return boards, players, policies, outcomes

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        """True if buffer has enough examples to sample a batch."""
        return len(self.buffer) >= batch_size
import ray
from connectzero.train.replay_buffer import ReplayBuffer


@ray.remote
class ReplayCoordinator:
    """Ray actor that holds the replay buffer and serves batches to the learner."""

    def __init__(self, max_size=50000):
        self.buffer = ReplayBuffer(max_size=max_size)

    def add_examples(self, examples):
        self.buffer.add_game(examples)

    def sample(self, batch_size):
        if not self.buffer.is_ready(batch_size):
            return None
        return self.buffer.sample(batch_size)

    def size(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return self.buffer.is_ready(batch_size)
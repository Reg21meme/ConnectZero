from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time


# --- Metrics ---
GAMES_PLAYED = Counter("connectzero_games_total", "Total self-play games generated")
EXAMPLES_GENERATED = Counter("connectzero_examples_total", "Total training examples generated")
BUFFER_SIZE = Gauge("connectzero_buffer_size", "Current replay buffer size")
LEARNER_LOSS = Gauge("connectzero_learner_loss", "Latest learner total loss")
POLICY_LOSS = Gauge("connectzero_policy_loss", "Latest learner policy loss")
VALUE_LOSS = Gauge("connectzero_value_loss", "Latest learner value loss")
GAMES_PER_SEC = Gauge("connectzero_games_per_sec", "Self-play games per second")
WORKER_COUNT = Gauge("connectzero_worker_count", "Number of active self-play workers")
CHAMPION_ELO = Gauge("connectzero_champion_elo", "Current champion Elo rating")


def start_metrics_server(port=8000):
    """Start Prometheus HTTP metrics endpoint."""
    start_http_server(port)
    print(f"Prometheus metrics available at http://localhost:{port}/metrics")


def record_games(n_games, n_examples, elapsed_sec):
    GAMES_PLAYED.inc(n_games)
    EXAMPLES_GENERATED.inc(n_examples)
    if elapsed_sec > 0:
        GAMES_PER_SEC.set(n_games / elapsed_sec)


def record_buffer_size(size):
    BUFFER_SIZE.set(size)


def record_learner_metrics(loss, policy_loss, value_loss):
    LEARNER_LOSS.set(loss)
    POLICY_LOSS.set(policy_loss)
    VALUE_LOSS.set(value_loss)


def record_worker_count(n):
    WORKER_COUNT.set(n)


def record_champion_elo(elo):
    CHAMPION_ELO.set(elo)
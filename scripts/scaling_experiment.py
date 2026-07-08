"""
Measure self-play games/sec with 1, 2, and 4 Ray workers.
Usage: python3 scripts/scaling_experiment.py
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ray
from connectzero.selfplay.worker import SelfPlayWorker

GAMES_PER_WORKER = 5
NUM_SIMULATIONS = 25


def measure_throughput(num_workers, games_per_worker=GAMES_PER_WORKER):
    workers = [
        SelfPlayWorker.remote(num_simulations=NUM_SIMULATIONS)
        for _ in range(num_workers)
    ]

    start = time.time()
    futures = [w.play_games.remote(games_per_worker) for w in workers]
    results = ray.get(futures)
    elapsed = time.time() - start

    total_games = num_workers * games_per_worker
    total_examples = sum(len(r) for r in results)
    games_per_sec = total_games / elapsed

    print(f"  Workers: {num_workers} | Games: {total_games} | "
          f"Examples: {total_examples} | "
          f"Time: {elapsed:.1f}s | "
          f"Games/sec: {games_per_sec:.2f}")

    return games_per_sec


def main():
    ray.init(ignore_reinit_error=True)
    print("=== ConnectZero Scaling Experiment ===\n")
    print(f"Games per worker: {GAMES_PER_WORKER}, Simulations: {NUM_SIMULATIONS}\n")

    results = {}
    for n in [1, 2, 4]:
        print(f"Running with {n} worker(s)...")
        gps = measure_throughput(n)
        results[n] = gps

    print("\n=== Summary ===")
    baseline = results[1]
    for n, gps in results.items():
        speedup = gps / baseline
        print(f"  {n} workers: {gps:.2f} games/sec (speedup: {speedup:.2f}x)")

    ray.shutdown()


if __name__ == "__main__":
    main()
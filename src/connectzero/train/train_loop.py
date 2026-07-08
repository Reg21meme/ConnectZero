import os
import time
import mlflow
from connectzero.model.network import ConnectZeroNet
from connectzero.model.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from connectzero.selfplay.worker import play_game
from connectzero.train.replay_buffer import ReplayBuffer
from connectzero.train.learner import Learner


def train(
    num_iterations=50,
    games_per_iteration=5,
    train_steps_per_iteration=10,
    batch_size=128,
    num_simulations=50,
    checkpoint_dir="runs/checkpoints",
    num_res_blocks=4,
    channels=64,
    device="cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    network = ConnectZeroNet(num_res_blocks=num_res_blocks, channels=channels)
    learner = Learner(network, device=device)
    buf = ReplayBuffer(max_size=50000)

    latest = get_latest_checkpoint(checkpoint_dir)
    if latest:
        meta = load_checkpoint(network, latest, device)
        start_iter = meta.get("iteration", 0) + 1
        print(f"Resuming from iteration {start_iter}")
    else:
        start_iter = 0
        print("Starting fresh training run")

    with mlflow.start_run():
        # Log config
        mlflow.log_params({
            "num_iterations": num_iterations,
            "games_per_iteration": games_per_iteration,
            "train_steps_per_iteration": train_steps_per_iteration,
            "batch_size": batch_size,
            "num_simulations": num_simulations,
            "num_res_blocks": num_res_blocks,
            "channels": channels,
            "device": device,
        })

        for iteration in range(start_iter, num_iterations):
            iter_start = time.time()

            # --- Self-play ---
            print(f"\n[Iter {iteration}] Generating {games_per_iteration} self-play games...")
            total_examples = 0
            for _ in range(games_per_iteration):
                examples = play_game(network, num_simulations=num_simulations, device=device)
                buf.add_game(examples)
                total_examples += len(examples)
            print(f"  Buffer size: {len(buf)} | New examples: {total_examples}")

            # --- Train ---
            if not buf.is_ready(batch_size):
                print(f"  Buffer not ready (need {batch_size} examples), skipping training.")
                continue

            print(f"  Training for {train_steps_per_iteration} steps...")
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            for _ in range(train_steps_per_iteration):
                boards, players, policies, outcomes = buf.sample(batch_size)
                metrics = learner.train_step(boards, players, policies, outcomes)
                total_loss += metrics["loss"]
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]

            avg_loss = total_loss / train_steps_per_iteration
            avg_policy_loss = total_policy_loss / train_steps_per_iteration
            avg_value_loss = total_value_loss / train_steps_per_iteration
            print(f"  Avg loss: {avg_loss:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "buffer_size": len(buf),
                "new_examples": total_examples,
            }, step=iteration)

            # --- Checkpoint ---
            ckpt_path = os.path.join(checkpoint_dir, f"gen_{iteration:04d}.pt")
            save_checkpoint(network, ckpt_path, metadata={
                "iteration": iteration,
                "avg_loss": avg_loss,
                "buffer_size": len(buf),
            })
            mlflow.log_artifact(ckpt_path)

            elapsed = time.time() - iter_start
            print(f"  Iteration {iteration} done in {elapsed:.1f}s")

        print("\nTraining complete.")

    return network


if __name__ == "__main__":
    train(
        num_iterations=50,
        games_per_iteration=5,
        train_steps_per_iteration=10,
        batch_size=128,
        num_simulations=50,
    )
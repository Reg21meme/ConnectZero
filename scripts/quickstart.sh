#!/bin/bash
set -e

echo "=== ConnectZero Quickstart ==="

echo "1. Installing dependencies..."
pip3 install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu --break-system-packages -q
pip3 install -e ".[dev]" --break-system-packages -q

echo "2. Running tests..."
pytest tests/ -q

echo "3. Running short training job (20 iterations)..."
python3 -c "
from connectzero.train.train_loop import train
train(
    num_iterations=20,
    games_per_iteration=5,
    train_steps_per_iteration=10,
    batch_size=32,
    num_simulations=25,
    checkpoint_dir='runs/quickstart',
)
"

echo "4. Starting Django dashboard at http://localhost:8001"
cd dashboard && python3 manage.py runserver 8001
import os
import torch
import json
from datetime import datetime


def save_checkpoint(network, path, metadata=None):
    """Save network weights and optional metadata to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": network.state_dict(),
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(network, path, device="cpu"):
    """Load weights into network from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    network.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded: {path}")
    return checkpoint.get("metadata", {})


def get_latest_checkpoint(checkpoint_dir):
    """Return path to the most recently saved checkpoint, or None."""
    if not os.path.exists(checkpoint_dir):
        return None
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]
    if not files:
        return None
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])
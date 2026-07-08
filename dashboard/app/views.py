import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings


def get_checkpoints():
    """List all checkpoint files with metadata."""
    checkpoint_dir = settings.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_dir):
        return []
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    checkpoints = []
    for i, f in enumerate(files):
        checkpoints.append({
            "generation": i,
            "filename": f,
            "path": str(checkpoint_dir / f),
        })
    return checkpoints


def dashboard(request):
    checkpoints = get_checkpoints()
    context = {
        "num_checkpoints": len(checkpoints),
        "latest": checkpoints[-1]["filename"] if checkpoints else "None",
    }
    return render(request, 'app/dashboard.html', context)


def api_checkpoints(request):
    return JsonResponse({"checkpoints": get_checkpoints()})


def api_elo(request):
    # Return hardcoded Elo data for now — Week 6 Session 4 will make this dynamic
    data = [
    {"generation": 0, "elo": 344.8},
    {"generation": 126, "elo": 953.9},
    {"generation": 252, "elo": 1087.9},
    {"generation": 378, "elo": 1247.9},
    {"generation": 504, "elo": 1008.5},
    {"generation": 631, "elo": 1523.6},
]
    return JsonResponse({"elo_history": data})
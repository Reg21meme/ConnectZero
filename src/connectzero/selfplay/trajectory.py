import numpy as np


def apply_temperature(visit_counts, temperature):
    """
    Convert raw visit counts into a move probability distribution
    using temperature scaling.
    """
    if temperature == 0 or temperature < 1e-8:
        # Greedy: put all probability on the most visited column
        probs = np.zeros_like(visit_counts, dtype=np.float32)
        probs[np.argmax(visit_counts)] = 1.0
        return probs

    # Raise visit counts to the power 1/temperature then normalise
    counts = np.array(visit_counts, dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    total = counts.sum()
    if total == 0:
        # Fallback: uniform over legal moves
        nonzero = counts > 0
        if nonzero.any():
            counts[nonzero] = 1.0
        else:
            counts[:] = 1.0
    return (counts / counts.sum()).astype(np.float32)


def get_temperature(move_number, temp_high=1.0, temp_low=0.0, threshold=20):
    """
    Return the temperature for a given move number.
    """
    if move_number < threshold:
        return temp_high
    return temp_low


def select_move(visit_counts, temperature):
    """
    Sample a move from the visit count distribution at the given temperature.
    """
    probs = apply_temperature(visit_counts, temperature)
    return int(np.random.choice(len(probs), p=probs))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from connectzero.env.connect4 import ROWS, COLS, P1, P2


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board, current_player):
    """
    Convert a 6x7 numpy board into a 3x6x7 float tensor.

    Channel 0: 1 where current player has a piece, 0 elsewhere
    Channel 1: 1 where opponent has a piece, 0 elsewhere
    Channel 2: all 1s if current_player is P1, all 0s if P2
                (lets the network know whose turn it is)
    """
    opponent = P2 if current_player == P1 else P1
    planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
    planes[0] = (board == current_player).astype(np.float32)
    planes[1] = (board == opponent).astype(np.float32)
    planes[2] = np.ones((ROWS, COLS), dtype=np.float32) if current_player == P1 else np.zeros((ROWS, COLS), dtype=np.float32)
    return torch.tensor(planes)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    Input and output have the same shape so they can be added directly.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # skip connection
        x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Policy/value network
# ---------------------------------------------------------------------------

class ConnectZeroNet(nn.Module):
    """
    Shared trunk + two heads.
    """

    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()

        # Entry conv: 3 input channels (our pieces, their pieces, turn) -> channels
        self.entry = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Trunk: stack of residual blocks
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        trunk_flat = channels * ROWS * COLS  # 64 * 6 * 7 = 2688

        # Policy head: trunk -> flat -> 7 logits (one per column)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, COLS),
        )

        # Value head: trunk -> flat -> scalar in (-1, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Args:
            x: float tensor of shape (batch, 3, 6, 7)
        Returns:
            policy_logits: (batch, 7) — raw logits, apply softmax + mask outside
            value:         (batch, 1) — scalar in (-1, 1)
        """
        x = self.entry(x)
        x = self.trunk(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def predict(self, board, current_player, legal_moves, device="cpu"):
        """
        Convenience method for single-board inference during MCTS.
        """
        self.eval()
        with torch.no_grad():
            x = encode_board(board, current_player).unsqueeze(0).to(device)
            policy_logits, value = self.forward(x)

            # Mask illegal moves
            mask = torch.full((1, COLS), float("-inf"))
            for col in legal_moves:
                mask[0, col] = 0.0
            policy_logits = policy_logits + mask

            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()

        return policy, value
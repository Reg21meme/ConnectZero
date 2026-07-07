import torch
import torch.optim as optim
import numpy as np
from connectzero.model.network import encode_board, ConnectZeroNet
from connectzero.train.losses import compute_loss
from connectzero.env.connect4 import P1


class Learner:

    def __init__(self, network, lr=1e-3, device="cpu"):
        self.network = network
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

    def train_step(self, boards, players, policy_targets, value_targets):
        """One gradient update on a batch of examples."""
        self.network.train()

        # Encode each board from the perspective of the player to move
        encoded = []
        for i in range(len(boards)):
            legal_moves = list(range(7))  # approximate — masking handled by loss
            enc = encode_board(boards[i], players[i])
            encoded.append(enc)
        x = torch.stack(encoded).to(self.device)

        policy_targets_t = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
        value_targets_t = torch.tensor(value_targets, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.network(x)
        loss, policy_loss, value_loss = compute_loss(
            policy_logits, value_pred, policy_targets_t, value_targets_t
        )
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
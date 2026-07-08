import ray
import torch
import torch.optim as optim
import numpy as np
from connectzero.model.network import encode_board, ConnectZeroNet
from connectzero.train.losses import compute_loss


class Learner:
    """Local learner for single-machine training."""

    def __init__(self, network, lr=1e-3, device="cpu"):
        self.network = network
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

    def train_step(self, boards, players, policy_targets, value_targets):
        self.network.train()
        encoded = []
        for i in range(len(boards)):
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


@ray.remote
class LearnerActor:
    """Ray actor version of the learner for distributed training."""

    def __init__(self, num_res_blocks=4, channels=64, lr=1e-3, device="cpu"):
        self.network = ConnectZeroNet(num_res_blocks=num_res_blocks, channels=channels)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.device = device

    def train_step(self, boards, players, policy_targets, value_targets):
        self.network.train()
        encoded = []
        for i in range(len(boards)):
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

    def get_weights(self):
        return self.network.state_dict()

    def get_network(self):
        return self.network
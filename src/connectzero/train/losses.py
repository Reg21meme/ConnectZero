import torch
import torch.nn.functional as F


def compute_loss(policy_logits, value_pred, policy_targets, value_targets):
    """
    policy_logits:  (batch, 7)  raw network output, no softmax yet
    value_pred:     (batch, 1)  network value estimate
    policy_targets: (batch, 7)  MCTS visit distributions
    value_targets:  (batch,)    game outcomes -1/0/+1
    """
    # Policy loss — cross entropy between network logits and MCTS policy
    policy_loss = F.cross_entropy(policy_logits, policy_targets)

    # Value loss — MSE between predicted value and actual outcome
    value_loss = F.mse_loss(value_pred.squeeze(1), value_targets)

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss
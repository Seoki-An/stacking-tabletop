import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def construct_mlp(channels: List[int], *modules: List[torch.nn.Module], act=None):
    assert len(channels) > 1
    layers = []
    for in_channel, out_channel in zip(channels[0:], channels[1:]):
        if layers and act is not None:
            layers.append(act)
        for module in modules:
            if module is nn.Linear:
                layers.append(module(in_channel, out_channel))
            elif module is nn.Conv1d or module is nn.Conv2d:
                layers.append(module(in_channel, out_channel, 1))
            elif module is nn.BatchNorm1d or module is nn.BatchNorm2d:
                layers.append(module(out_channel))
            elif module is nn.LayerNorm:
                layers.append(module(out_channel))

    return nn.Sequential(*layers)


def get_activation(type: str, **kwargs):
    try:
        return {
            "linear": None,
            "relu": nn.ReLU(),
            "leaky": nn.LeakyReLU(kwargs.get("negative_slope", 0.01)),
        }[type]
    except:
        raise ValueError(f"Unexpected activation: {type}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Focal Loss for Binary Classification
        Arguments:
            alpha (float): Class weight for positive samples (default: 0.25)
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        logits: (batch_size, 1) - raw model outputs
        targets: (batch_size, 1) - ground truth labels (0 or 1)
        """
        probs = F.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction is None:
            loss = focal_loss
        elif self.reduction == "mean":
            loss = focal_loss.mean()
        elif self.reduction == "sum":
            loss = focal_loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

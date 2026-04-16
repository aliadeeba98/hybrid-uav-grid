"""Actor network for behavior cloning."""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_uavs: int,
        action_dim: int = 4,
        hidden: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.num_uavs = num_uavs
        self.action_dim = action_dim
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_uavs * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        x = x.view(-1, self.num_uavs, self.action_dim)
        return torch.softmax(x, dim=2)

    def logits(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        return x.view(-1, self.num_uavs, self.action_dim)

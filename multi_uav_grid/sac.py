"""Discrete Soft Actor-Critic (joint action space per timestep)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multi_uav_grid.actions import joint_action_dim, joint_to_actions

if TYPE_CHECKING:
    from multi_uav_grid.config import RunConfig


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.as_tensor(self.s[idx], device=self.device),
            torch.as_tensor(self.a[idx], device=self.device, dtype=torch.long),
            torch.as_tensor(self.r[idx], device=self.device),
            torch.as_tensor(self.s2[idx], device=self.device),
            torch.as_tensor(self.d[idx], device=self.device),
        )


class SACAgent:
    """Discrete SAC with twin Q-networks and fixed entropy temperature."""

    def __init__(self, config: RunConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        sd = config.state_dim
        na = joint_action_dim(config.num_uavs, config.per_uav_action_dim)
        h = config.sac_hidden

        self.policy = PolicyNet(sd, na, h).to(device)
        self.q1 = QNetwork(sd, na, h).to(device)
        self.q2 = QNetwork(sd, na, h).to(device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad = False

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=config.sac_lr_policy)
        self.opt_q = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.sac_lr_q,
        )

        self.replay = ReplayBuffer(config.replay_capacity, sd, device)
        self.n_actions = na
        self.gamma = config.sac_gamma
        self.tau = config.sac_tau
        self.alpha = config.sac_alpha
        self.reward_scale = config.sac_reward_scale

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(p.data * self.tau)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(p.data * self.tau)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, *, deterministic: bool) -> int:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(s)
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def update(self, batch_size: int) -> dict[str, float] | None:
        if self.replay.size < batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(batch_size)
        r = r * self.reward_scale

        with torch.no_grad():
            logits_next = self.policy(s2)
            pi_next = F.softmax(logits_next, dim=-1)
            log_pi_next = F.log_softmax(logits_next, dim=-1)
            q1n = self.q1_t(s2)
            q2n = self.q2_t(s2)
            qn = torch.min(q1n, q2n)
            v_next = (pi_next * (qn - self.alpha * log_pi_next)).sum(dim=-1)
            target = r + (1.0 - d) * self.gamma * v_next

        q1_all = self.q1(s)
        q2_all = self.q2(s)
        q1_sa = q1_all.gather(1, a.unsqueeze(1)).squeeze(1)
        q2_sa = q2_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss_q = F.mse_loss(q1_sa, target) + F.mse_loss(q2_sa, target)

        self.opt_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.grad_clip_norm,
        )
        self.opt_q.step()

        logits = self.policy(s)
        pi = F.softmax(logits, dim=-1)
        log_pi = F.log_softmax(logits, dim=-1)
        q1_all = self.q1(s)
        q2_all = self.q2(s)
        q = torch.min(q1_all, q2_all).detach()
        loss_pi = (pi * (self.alpha * log_pi - q)).sum(dim=-1).mean()

        self.opt_pi.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
        self.opt_pi.step()

        self.soft_update_targets()

        return {"loss_q": float(loss_q.item()), "loss_pi": float(loss_pi.item())}

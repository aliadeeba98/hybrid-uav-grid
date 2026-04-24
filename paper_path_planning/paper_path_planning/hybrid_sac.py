"""
Algorithm 2: replay from PSO polylines + minimal continuous SAC (smoke / structure).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .pso import PSOState


@dataclass
class SimpleTransition:
    s: np.ndarray
    a: np.ndarray
    r: float
    s2: np.ndarray


class ReplayBufferCont:
    def __init__(self, state_dim: int, action_dim: int, cap: int) -> None:
        self.s = np.zeros((cap, state_dim), np.float32)
        self.a = np.zeros((cap, action_dim), np.float32)
        self.r = np.zeros((cap,), np.float32)
        self.s2 = np.zeros((cap, state_dim), np.float32)
        self.d = np.zeros((cap,), np.float32)
        self.cap = cap
        self.i = 0
        self.size = 0

    def add(self, tr: SimpleTransition) -> None:
        j = self.i
        self.s[j] = tr.s
        self.a[j] = tr.a
        self.r[j] = tr.r
        self.s2[j] = tr.s2
        self.d[j] = 0.0
        self.i = (self.i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(
        self, b: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, size=b)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx]


def transitions_from_pso_path(full_paths: np.ndarray, grid_size: int) -> list[SimpleTransition]:
    """Build toy (s, a, r, s') along each segment of each UAV's polyline."""
    n, m, _ = full_paths.shape
    trs: list[SimpleTransition] = []
    for u in range(n):
        pth = full_paths[u]
        for seg in range(m - 1):
            a0, a1 = pth[seg], pth[seg + 1]
            vec = a1 - a0
            step = vec / (np.linalg.norm(vec) + 1e-9)
            s = np.array([a0[0] / grid_size, a0[1] / grid_size, step[0], step[1]], dtype=np.float32)
            a = step.astype(np.float32) * 0.1
            s2 = np.array(
                [a1[0] / grid_size, a1[1] / grid_size, step[0], step[1]], dtype=np.float32
            )
            g = pth[-1] / float(grid_size)
            r = -float(np.linalg.norm(s2[:2] - g)) + 0.1
            r -= float(np.linalg.norm(a))
            trs.append(SimpleTransition(s, a, r, s2))
    return trs


def hybrid_pso_sac_smoke(
    pso: PSOState,
    full_paths: np.ndarray,
    grid_size: int,
    device: torch.device,
    state_dim: int = 4,
    action_dim: int = 2,
    prefill_steps: int = 32,
) -> dict[str, Any]:
    """
    Replay pre-fill, then soft Bellman + policy updates in the spirit of (38)–(40).
    """
    buf = ReplayBufferCont(state_dim, action_dim, 2048)
    trs = transitions_from_pso_path(full_paths, grid_size=grid_size)
    rng = np.random.default_rng(0)
    for tr in trs:
        buf.add(tr)

    class TwinCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def forward(
            self, s: torch.Tensor, a: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sa = torch.cat([s, a], dim=-1)
            return self.q1(sa), self.q2(sa)

    class StochasticActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim * 2)
            )
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward_actor(act: StochasticActor, s: torch.Tensor) -> torch.Tensor:
        o = act.trunk(s)
        a_dim = o.shape[1] // 2
        return o[:, :a_dim], o[:, a_dim:]

    def policy_and_logp(
        act: StochasticActor, s: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, _ = forward_actor(act, s)
        std = F.softplus(act.log_std) + 0.01
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        u = torch.tanh(z)
        a = 0.2 * u
        logp = dist.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        return a, logp

    actor = StochasticActor().to(device)
    critic = TwinCritic().to(device)
    critic_t = copy.deepcopy(critic).to(device)
    for p_ in critic_t.parameters():
        p_.requires_grad = False

    opt_a = optim.Adam(actor.parameters(), lr=1e-3)
    opt_q = optim.Adam(critic.parameters(), lr=1e-3)
    gamma, tau, alpha = 0.99, 0.05, 0.1

    losses: list[float] = []
    b = min(4, max(1, buf.size))
    n_steps = min(prefill_steps, 64)
    for _ in range(n_steps):
        if buf.size < b:
            break
        s_ns, a_ns, r_ns, s2_ns = buf.sample(b, rng)
        s = torch.as_tensor(s_ns, device=device, dtype=torch.float32)
        a = torch.as_tensor(a_ns, device=device, dtype=torch.float32)
        r = torch.as_tensor(r_ns, device=device, dtype=torch.float32).view(-1, 1)
        s2 = torch.as_tensor(s2_ns, device=device, dtype=torch.float32)
        a2, logp2 = policy_and_logp(actor, s2, alpha)
        with torch.no_grad():
            q1n, q2n = critic_t(s2, a2)
            qm = torch.min(q1n, q2n)
            y = r + gamma * (qm - alpha * logp2)
        q1, q2 = critic(s, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        a1, logp = policy_and_logp(actor, s, alpha)
        q1p, q2p = critic(s, a1)
        qp = torch.min(q1p, q2p)
        loss_pi = (alpha * logp - qp).mean()
        opt_a.zero_grad()
        loss_pi.backward()
        opt_a.step()

        with torch.no_grad():
            for p_, t_ in zip(critic.parameters(), critic_t.parameters()):
                t_.data.mul_(1.0 - tau).add_(p_.data * tau)
        losses.append(float((loss_q + loss_pi).item()))

    return {
        "pso_fitness": pso.gbest_fitness,
        "n_transitions": int(buf.size),
        "smoke_loss_last": losses[-1] if losses else 0.0,
    }

"""
Algorithms 1 and 2 from: Swarm Optimization guided Reinforcement learning
Approach for Multi-UAV Path Planning in Obstacle Rich Environments (A. Ali et al.).

- Algorithm 1: PSO for multi-UAV path planning (waypoint parameters, fitness (19)-(23))
- Algorithm 2: Hybrid PSO -> replay initialization -> online SAC (structure + minimal continuous SAC for tests)
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


# --- Fitness weights (19): F = w1*L + w2*C + w3*O + w4*S
@dataclass
class FitnessWeights:
    w1: float = 1.0
    w2: float = 10.0
    w3: float = 5.0
    w4: float = 1.0


@dataclass
class PSOConfig:
    num_uavs: int
    n_intermediate: int  # K in the paper: intermediate waypoints per UAV
    swarm_size: int
    max_iters: int
    inertia: float
    c1: float
    c2: float
    v_max: float
    grid_size: int
    d_safe: float
    te_samples: int  # Te (time samples for (21) and (22))
    fitness: FitnessWeights
    seed: int | None = 0
    conv_std_eps: float | None = 1e-3  # optional convergence: stop if std(fitness) < eps


def _clip_vec(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(v, lo, hi)


@dataclass
class PSOState:
    """Holds the optimized global-best waypoint tensor G with shape (N, K, 2)."""

    gbest: np.ndarray
    gbest_fitness: float
    history_best_fitness: list[float]

    @property
    def waypoints(self) -> np.ndarray:
        return self.gbest


def _cell_of(p: np.ndarray, grid_size: int) -> tuple[int, int]:
    x = int(np.floor(np.clip(p[0], 0, grid_size - 1e-6)))
    y = int(np.floor(np.clip(p[1], 0, grid_size - 1e-6)))
    return (x, y)


def _in_obstacles(p: np.ndarray, obstacles: set[tuple[int, int]], grid_size: int) -> bool:
    return _cell_of(p, grid_size) in obstacles


def _repair_out_of_obstacles(
    w: np.ndarray, obstacles: set[tuple[int, int]], grid_size: int, rng: np.random.Generator
) -> np.ndarray:
    """If any waypoint projects into a blocked cell, nudge to nearest free cell center."""
    out = w.copy()
    n, k, _ = out.shape
    free_cells: list[tuple[int, int]] = [
        (i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) not in obstacles
    ]
    if not free_cells:
        return out
    free_arr = np.array(free_cells, dtype=np.float64)

    for i in range(n):
        for j in range(k):
            if not _in_obstacles(out[i, j], obstacles, grid_size):
                continue
            c = out[i, j]
            d2 = ((free_arr[:, 0] + 0.5) - c[0]) ** 2 + ((free_arr[:, 1] + 0.5) - c[1]) ** 2
            nn = int(np.argmin(d2))
            out[i, j, 0] = free_arr[nn, 0] + 0.5
            out[i, j, 1] = free_arr[nn, 1] + 0.5
            if rng is not None and _in_obstacles(out[i, j], obstacles, grid_size):
                pick = free_arr[int(rng.integers(0, len(free_arr)))]
                out[i, j, 0] = float(pick[0]) + 0.5
                out[i, j, 1] = float(pick[1]) + 0.5
    return out


def _full_path(
    w_inter: np.ndarray, starts: np.ndarray, goals: np.ndarray
) -> np.ndarray:
    """W_inter: (N, K, 2); return (N, K+2, 2) with start and goal at ends."""
    n, k, _ = w_inter.shape
    s = starts.reshape(n, 1, 2)
    g = goals.reshape(n, 1, 2)
    return np.concatenate([s, w_inter, g], axis=1)


def _path_length_L(full: np.ndarray) -> float:
    """(20): sum of squared segment lengths in R^2."""
    dif = np.diff(full, axis=1)
    return float(np.sum(np.einsum("nkd,nkd->nk", dif, dif)))


def _sample_polyline_by_arclen(full: np.ndarray, te: int) -> np.ndarray:
    """(Te, N, 2) positions along each UAV's polyline, shared arc-length time across drones."""
    n, m, _ = full.shape
    if te <= 1:
        t = np.array([0.0])
    else:
        t = np.linspace(0.0, 1.0, te, endpoint=True)

    segs = np.diff(full, axis=1)
    seg_len = np.sqrt(np.einsum("nkd,nkd->nk", segs, segs))
    cum = np.zeros((n, m))
    for i in range(1, m):
        cum[:, i] = cum[:, i - 1] + seg_len[:, i - 1]
    tot = cum[:, -1].copy()
    tot[tot == 0.0] = 1.0
    p_out = np.zeros((te, n, 2), dtype=np.float64)
    for u in range(n):
        total = float(tot[u])
        for ti, s in enumerate(t):
            d = s * total
            idx = int(np.searchsorted(cum[u, :], d, side="right") - 1)
            idx = int(np.clip(idx, 0, m - 2))
            t_loc = 0.0
            a = full[u, idx]
            b = full[u, idx + 1]
            seg_leg = b - a
            L = float(np.linalg.norm(seg_leg))
            if L < 1e-12:
                p_out[ti, u] = a
            else:
                t_loc = (d - float(cum[u, idx])) / L
                p_out[ti, u] = a + t_loc * (seg_leg)
    return p_out


def _collision_C(samples: np.ndarray, d_safe: float) -> float:
    """(21) simplified: all pairs i != j, all time samples, indicator of distance < d_safe."""
    te, n, _ = samples.shape
    c = 0.0
    for t in range(te):
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(samples[t, i] - samples[t, j]))
                if d < d_safe:
                    c += 1.0
    return c


def _obstacle_O(samples: np.ndarray, obstacles: set[tuple[int, int]], grid_size: int) -> float:
    """(22) count violations over time and all UAVs."""
    te, n, _ = samples.shape
    o = 0.0
    for t in range(te):
        for i in range(n):
            if _in_obstacles(samples[t, i], obstacles, grid_size):
                o += 1.0
    return o


def _smoothness_S(full: np.ndarray) -> float:
    """
    (23): sum over k=1..K: ||(W_{k+1}-W_k) - (W_k - W_{k-1})|| with W_0= start, W_{K+1}= goal
    (full has shape (N, K+2, 2)); index k in 1..K+1-1? Paper says K in sum — use inner waypoints+goal chain.
    """
    n, m, _ = full.shape
    s = 0.0
    if m < 3:
        return 0.0
    for u in range(n):
        w = full[u]
        for k in range(1, m - 1):
            v0 = w[k] - w[k - 1]
            v1 = w[k + 1] - w[k]
            s += float(np.linalg.norm((v1 - v0)))
    return s


def fitness_f(
    w_flat: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    n_uavs: int,
    n_int: int,
    obstacles: set[tuple[int, int]],
    grid_size: int,
    fw: FitnessWeights,
    d_safe: float,
    te: int,
) -> float:
    w = w_flat.reshape(n_uavs, n_int, 2)
    full = _full_path(w, starts, goals)
    l = _path_length_L(full)
    smpl = _sample_polyline_by_arclen(full, te)
    c = _collision_C(smpl, d_safe)
    o = _obstacle_O(smpl, obstacles, grid_size)
    s = _smoothness_S(full)
    return fw.w1 * l + fw.w2 * c + fw.w3 * o + fw.w4 * s


# --- Algorithm 1
def pso_particle_swarm_path_planning(
    starts: np.ndarray,
    goals: np.ndarray,
    obstacles: set[tuple[int, int]],
    cfg: PSOConfig,
) -> PSOState:
    """
    Algorithm 1: PSO for Multi-UAV Path Planning
    (minimization, uniform init as in the paper's pseudocode)
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.num_uavs
    k = cfg.n_intermediate
    p = cfg.swarm_size
    dim = n * k * 2
    lb, ub = 0.0, float(cfg.grid_size - 1e-3)

    x = rng.uniform(lb, ub, size=(p, dim))
    v = rng.uniform(-cfg.v_max, cfg.v_max, size=(p, dim))

    def eval_one(z: np.ndarray) -> float:
        w = z.reshape(n, k, 2)
        w = _repair_out_of_obstacles(w, obstacles, cfg.grid_size, rng)
        z2 = w.reshape(-1)
        return fitness_f(
            z2, starts, goals, n, k, obstacles, cfg.grid_size, cfg.fitness, cfg.d_safe, cfg.te_samples
        )

    pfit = np.array([eval_one(x[i]) for i in range(p)])
    pbest = x.copy()
    pbest_f = pfit.copy()
    g_idx = int(np.argmin(pbest_f))
    gbest = pbest[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    hist: list[float] = [gbest_f]

    w, c1, c2 = cfg.inertia, cfg.c1, cfg.c2
    for t in range(1, cfg.max_iters + 1):
        for i in range(p):
            r1, r2 = rng.uniform(0.0, 1.0, size=dim), rng.uniform(0.0, 1.0, size=dim)
            v[i] = w * v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest - x[i])
            v[i] = _clip_vec(v[i], -cfg.v_max, cfg.v_max)
            x[i] = x[i] + v[i]
            x[i] = _clip_vec(x[i], lb, ub)
            w2 = x[i].reshape(n, k, 2)
            w2 = _repair_out_of_obstacles(w2, obstacles, cfg.grid_size, rng)
            x[i] = w2.reshape(-1)

            f_i = eval_one(x[i])
            if f_i < pbest_f[i]:
                pbest_f[i] = f_i
                pbest[i] = x[i].copy()

        g_idx = int(np.argmin(pbest_f))
        if float(pbest_f[g_idx]) < gbest_f:
            gbest_f = float(pbest_f[g_idx])
            gbest = pbest[g_idx].copy()
        hist.append(gbest_f)

        if cfg.conv_std_eps is not None and p > 1:
            if float(np.std(pbest_f)) < cfg.conv_std_eps:
                break

    w_final = gbest.reshape(n, k, 2)
    w_final = _repair_out_of_obstacles(w_final, obstacles, cfg.grid_size, rng)
    return PSOState(gbest=w_final, gbest_fitness=fitness_f(w_final.reshape(-1), starts, goals, n, k, obstacles, cfg.grid_size, cfg.fitness, cfg.d_safe, cfg.te_samples), history_best_fitness=hist)


# --- Algorithm 2: minimal replay + micro SAC
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
    """
    Interpolate PSO waypoints to simple (pos,vel)->(pos',vel') transitions with a toy reward.
    full_paths: (N, M, 2)
    """
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
    Algorithm 2, phases 2-3: replay pre-fill from PSO paths, then soft Bellman and policy updates
    matching (38)-(40) in spirit (twin Q, entropy term, target soft updates).
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
    for p in critic_t.parameters():
        p.requires_grad = False

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
            for p, t in zip(critic.parameters(), critic_t.parameters()):
                t.data.mul_(1.0 - tau).add_(p.data * tau)
        losses.append(float((loss_q + loss_pi).item()))

    return {
        "pso_fitness": pso.gbest_fitness,
        "n_transitions": int(buf.size),
        "smoke_loss_last": losses[-1] if losses else 0.0,
    }

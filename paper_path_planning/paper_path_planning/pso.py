"""
Algorithm 1: PSO for multi-UAV waypoint path planning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitness import (
    FitnessWeights,
    _repair_out_of_obstacles,
    fitness_f,
)


@dataclass
class PSOConfig:
    num_uavs: int
    n_intermediate: int
    swarm_size: int
    max_iters: int
    inertia: float
    c1: float
    c2: float
    v_max: float
    grid_size: int
    d_safe: float
    te_samples: int
    fitness: FitnessWeights
    seed: int | None = 0
    conv_std_eps: float | None = 1e-3


def _clip_vec(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(v, lo, hi)


@dataclass
class PSOState:
    """Optimized global-best waypoint tensor G with shape (N, K, 2)."""

    gbest: np.ndarray
    gbest_fitness: float
    history_best_fitness: list[float]

    @property
    def waypoints(self) -> np.ndarray:
        return self.gbest


def pso_particle_swarm_path_planning(
    starts: np.ndarray,
    goals: np.ndarray,
    obstacles: set[tuple[int, int]],
    cfg: PSOConfig,
) -> PSOState:
    """PSO (minimization), uniform init as in the paper's pseudocode."""
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
    for _t in range(1, cfg.max_iters + 1):
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
    return PSOState(
        gbest=w_final,
        gbest_fitness=fitness_f(
            w_final.reshape(-1),
            starts,
            goals,
            n,
            k,
            obstacles,
            cfg.grid_size,
            cfg.fitness,
            cfg.d_safe,
            cfg.te_samples,
        ),
        history_best_fitness=hist,
    )

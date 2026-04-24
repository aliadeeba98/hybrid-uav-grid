"""
Fitness terms (19)–(23): path length, inter-UAV collision, obstacle, smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- (19): F = w1*L + w2*C + w3*O + w4*S
@dataclass
class FitnessWeights:
    w1: float = 1.0
    w2: float = 10.0
    w3: float = 5.0
    w4: float = 1.0


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
            if _in_obstacles(out[i, j], obstacles, grid_size):
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
    """(23): second-difference smoothness on full chains including start/goal."""
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
    sm = _smoothness_S(full)
    return fw.w1 * l + fw.w2 * c + fw.w3 * o + fw.w4 * sm

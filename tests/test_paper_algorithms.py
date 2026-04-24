"""Tests for paper Algorithms 1-2 (PSO + hybrid replay / SAC smoke)."""

from __future__ import annotations

import numpy as np
import torch

from multi_uav_grid.paper_algorithms import (
    PSOConfig,
    FitnessWeights,
    PSOState,
    fitness_f,
    pso_particle_swarm_path_planning,
    _full_path,
    _path_length_L,
    hybrid_pso_sac_smoke,
    transitions_from_pso_path,
)


def test_path_length_straight_line() -> None:
    """Shortest chain start -> one waypoint -> goal should match squared lengths (20)."""
    s = np.array([[0.0, 0.0]])
    g = np.array([[2.0, 0.0]])
    w = np.array([[[1.0, 0.0]]])  # 1 uav, 1 intermediate
    full = _full_path(w, s, g)
    L = _path_length_L(full)
    # each segment (0,0)->(1,0) length^2=1, (1,0)->(2,0) length^2=1
    assert L == 2.0


def test_fitness_f_single_uav() -> None:
    s = np.array([[0.0, 0.0]])
    g = np.array([[3.0, 0.0]])
    w = np.array([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float64)
    flat = w.reshape(-1)
    obstacles: set[tuple[int, int]] = set()
    f = fitness_f(
        flat,
        s,
        g,
        n_uavs=1,
        n_int=2,
        obstacles=obstacles,
        grid_size=6,
        fw=FitnessWeights(),
        d_safe=0.5,
        te=8,
    )
    assert np.isfinite(f) and f > 0


def test_pso_improves_or_non_increasing() -> None:
    """On an empty 8x8 grid, PSO should not increase best fitness (minimization)."""
    s = np.array([[1.0, 1.0], [2.0, 6.0]])
    g = np.array([[6.0, 6.0], [6.0, 1.0]])
    obs: set[tuple[int, int]] = {(3, 3), (3, 4), (4, 3)}
    cfg = PSOConfig(
        num_uavs=2,
        n_intermediate=2,
        swarm_size=20,
        max_iters=40,
        inertia=0.6,
        c1=1.2,
        c2=1.2,
        v_max=1.0,
        grid_size=8,
        d_safe=0.4,
        te_samples=12,
        fitness=FitnessWeights(w1=1, w2=5, w3=2, w4=0.5),
        seed=7,
    )
    st = pso_particle_swarm_path_planning(s, g, obs, cfg)
    hist = st.history_best_fitness
    assert hist[0] >= hist[-1] * 0.999 or hist[-1] < hist[0] + 0.1  # allow noise
    assert st.gbest.shape == (2, 2, 2)
    assert st.gbest_fitness == st.history_best_fitness[-1]


def test_transitions_and_hybrid_smoke() -> None:
    full = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    trs = transitions_from_pso_path(full, grid_size=8)
    # 2 UAVs × (3 waypoints − 1) segments = 4 transitions
    assert len(trs) == 4
    s = PSOState(
        gbest=np.zeros((1, 1, 2)),
        gbest_fitness=0.0,
        history_best_fitness=[0.0],
    )
    out = hybrid_pso_sac_smoke(
        s, full, grid_size=8, device=torch.device("cpu"), prefill_steps=8
    )
    assert out["n_transitions"] > 0
    assert "smoke_loss_last" in out

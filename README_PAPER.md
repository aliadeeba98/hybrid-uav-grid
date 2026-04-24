# Paper implementation: PSO + hybrid PSO–SAC (multi-UAV path planning)

This document describes the implementation of **Algorithm 1** (Particle Swarm Optimization over waypoints) and the **hybrid PSO + SAC** smoke pipeline (**Algorithm 2** phases 2–3) from the IEEE-style manuscript *“Swarm Optimization guided Reinforcement learning Approach for Multi-UAV Path Planning in Obstacle Rich Environments.”*

Code: `multi_uav_grid/paper_algorithms.py`  
Tests: `tests/test_paper_algorithms.py`

## Run tests

From the `hybrid-uav-grid` directory:

```bash
cd hybrid-uav-grid
PYTHONPATH=. python3 -m pytest tests/test_paper_algorithms.py -v
```

## Automated test results

Last run in this environment:

| Test | Outcome |
|------|---------|
| `test_path_length_straight_line` | **PASSED** |
| `test_fitness_f_single_uav` | **PASSED** |
| `test_pso_improves_or_non_increasing` | **PASSED** |
| `test_transitions_and_hybrid_smoke` | **PASSED** |

**Summary:** 4 passed (runtime ~1.5 s, CPU)

## Representative numeric run (Algorithm 1 + hybrid smoke)

Same scenario as `test_pso_improves_or_non_increasing`: **2 UAVs**, **8×8** grid, **3** blocked cells `{(3,3), (3,4), (4,3)}`, `PSOConfig` with `swarm_size=20`, `max_iters=40`, `seed=7`, two intermediate waypoints per UAV.

| Metric | Value |
|--------|--------|
| Global-best fitness \(F\) at start (iter 0) | **76.43** |
| Global-best fitness at end | **36.29** |
| Change (minimization: lower is better) | **−40.14** |
| PSO iterations executed | **40** |

Optimized intermediate waypoints `gbest` (shape N × K × 2), grid coordinates:

```text
UAV 0:  [[2.41, 3.04], [4.28, 4.56]]
UAV 1:  [[4.26, 4.71], [5.66, 3.09]]
```

Full polylines (start → waypoints → goal):

```text
UAV 0:  (1,1) → … → (6,6)
UAV 1:  (2,6) → … → (6,1)
```

**Hybrid (Algorithm 2) smoke** (replay built from PSO polylines + short continuous twin-Q SAC updates):

| Field | Value |
|--------|--------|
| `pso_fitness` | 36.29 |
| `n_transitions` | 6 (2 UAVs × 3 polyline segments each) |
| `smoke_loss_last` | ~0.28 (Q + policy loss on last update; for sanity only) |

`smoke_loss_last` is **not** comparable to full paper training; it only checks that the replay buffer and gradient step run without error.

## Optional: reproduce the printed numbers

```bash
cd hybrid-uav-grid
PYTHONPATH=. python3 -c "
import numpy as np
import torch
from multi_uav_grid.paper_algorithms import (
    PSOConfig, FitnessWeights, pso_particle_swarm_path_planning,
    _full_path, hybrid_pso_sac_smoke, PSOState,
)
s = np.array([[1.0, 1.0], [2.0, 6.0]])
g = np.array([[6.0, 6.0], [6.0, 1.0]])
obs = {(3, 3), (3, 4), (4, 3)}
cfg = PSOConfig(
    num_uavs=2, n_intermediate=2, swarm_size=20, max_iters=40,
    inertia=0.6, c1=1.2, c2=1.2, v_max=1.0, grid_size=8,
    d_safe=0.4, te_samples=12,
    fitness=FitnessWeights(w1=1, w2=5, w3=2, w4=0.5), seed=7,
)
st = pso_particle_swarm_path_planning(s, g, obs, cfg)
full = _full_path(st.gbest, s, g)
print('F start:', st.history_best_fitness[0])
print('F end:', st.gbest_fitness)
h = hybrid_pso_sac_smoke(
    PSOState(st.gbest, st.gbest_fitness, st.history_best_fitness),
    full, grid_size=8, device=torch.device('cpu'), prefill_steps=32,
)
print(h)
"
```

Exact floats may differ slightly with another OS / CPU / library build; seeds and config were chosen for **reproducibility** on the machine used when this README was written.

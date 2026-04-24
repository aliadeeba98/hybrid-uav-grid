# paper-path-planning

Standalone package: **Algorithm 1** (PSO over waypoints) and a **hybrid PSO + SAC** smoke pipeline (**Algorithm 2** phases 2–3), from *“Swarm Optimization guided Reinforcement learning Approach for Multi-UAV Path Planning in Obstacle Rich Environments.”*

Not part of the `hybrid-uav-grid` project; install or develop from **this** directory.

## Layout

| Path | Content |
|------|---------|
| `paper_path_planning/fitness.py` | \(F = \omega_1 L + \omega_2 C + \omega_3 O + \omega_4 S\), path sampling, repair |
| `paper_path_planning/pso.py` | `PSOConfig`, `PSOState`, `pso_particle_swarm_path_planning` |
| `paper_path_planning/hybrid_sac.py` | Replay, `transitions_from_pso_path`, `hybrid_pso_sac_smoke` |
| `tests/` | Pytest suite |

## Install (editable)

```bash
cd paper_path_planning
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
cd paper_path_planning
python3 -m pytest tests/ -v
```

(Uses `pyproject.toml` so the package is on the path; or `PYTHONPATH=. python3 -m pytest tests/ -v`.)

## Test results (reference)

| Test | Outcome |
|------|---------|
| `test_path_length_straight_line` | **PASSED** |
| `test_fitness_f_single_uav` | **PASSED** |
| `test_pso_improves_or_non_increasing` | **PASSED** |
| `test_transitions_and_hybrid_smoke` | **PASSED** |

## Example (2 UAVs, 8×8, 3 blocked cells, seed 7)

| Metric | Value |
|--------|--------|
| Global-best fitness \(F\) at start | **~76.4** |
| Global-best fitness at end | **~36.3** |
| PSO iterations | **40** |

## Reproduce

```bash
cd paper_path_planning
PYTHONPATH=. python3 -c "
import numpy as np
import torch
from paper_path_planning import (
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
print(hybrid_pso_sac_smoke(
    PSOState(st.gbest, st.gbest_fitness, st.history_best_fitness),
    full, grid_size=8, device=torch.device('cpu'), prefill_steps=32,
))
"
```

After `pip install -e .`, you can run the same without `PYTHONPATH=.` from this directory (imports resolve as installed).

Exact floats can vary by platform.

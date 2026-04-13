# Multi-UAV grid hybrid planner

Multi-UAV path planning on a grid with random obstacles. A **hybrid controller** (BFS + reservations + goal-zone holding) runs every step; an **Actor** is trained with **behavior cloning** (cross-entropy on logits). Success means all UAVs end within Euclidean distance ≤ 1 of their goals with **no obstacle or inter-agent collision** in the episode.

The code is split into a small **package** (`multi_uav_grid/`) suitable for import in other apps, with **CLI**, **logging**, **typed config**, and **no side effects on import**.

## Layout

| Path | Role |
|------|------|
| `multi_uav_grid/config.py` | `RunConfig` dataclass |
| `multi_uav_grid/environment.py` | `GridEnv`, `StepResult` |
| `multi_uav_grid/planner.py` | `HybridPlanner` (BFS + greedy fallback) |
| `multi_uav_grid/policy.py` | `Actor` (PyTorch) |
| `multi_uav_grid/training.py` | `train()`, `evaluate()`, seed helper |
| `multi_uav_grid/__main__.py` | CLI (`python -m multi_uav_grid`) |

## Build

Prerequisites: **Python 3.9+**, **pip**; optionally **Docker** / **Docker Compose** and **GNU Make**.

### 1. Local install (editable package)

From the repo root, enter this project and install into your environment (virtualenv recommended):

```bash
cd hybrid-uav-grid
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Alternative: install dependencies only from `requirements.txt` (uses PyPI `torch`; not pinned to CPU wheels):

```bash
cd hybrid-uav-grid
pip install -r requirements.txt
pip install -e . --no-deps
```

Sanity check:

```bash
python -m multi_uav_grid --help
```

### 2. Docker image (CPU PyTorch)

The image **`Dockerfile.multi-uav-grid`** installs CPU `torch` from the PyTorch wheel index (smaller than the default CUDA bundle).

```bash
cd hybrid-uav-grid
docker build -f Dockerfile.multi-uav-grid -t multi-uav-grid:latest .
```

Run the default training + evaluation entrypoint:

```bash
docker run --rm multi-uav-grid:latest
```

Override CLI args (example):

```bash
docker run --rm multi-uav-grid:latest python -m multi_uav_grid --train-episodes 200 --test-episodes 50 --log-every 20
```

For **GPU**, use a separate Dockerfile that installs CUDA-enabled `torch`, then pass `--device cuda` when a GPU is available in the container.

### 3. Docker Compose

Build the service image and start the defined workload (default: 5000 train / 1000 test episodes, log every 100):

```bash
cd hybrid-uav-grid
docker compose build
docker compose up
```

One-shot build + run:

```bash
cd hybrid-uav-grid
docker compose up --build
```

### 4. Make

From `hybrid-uav-grid/`:

```bash
make help # lists targets (default when you run `make`)
make install-editable
make docker-build    # same as docker build -f Dockerfile.multi-uav-grid ...
make docker-up       # compose up --build
make run-smoke       # short local run
```

## Run (after local build)

```bash
cd hybrid-uav-grid
source .venv/bin/activate   # if you use a venv
python -m multi_uav_grid
# or (after pip install -e .)
multi-uav-grid
```

Useful flags: `--train-episodes`, `--test-episodes`, `--seed`, `--device cuda`, `--log-level DEBUG`, `--log-every N` (set `0` to log only summaries).

## Programmatic use

```python
from multi_uav_grid import RunConfig, GridEnv, HybridPlanner, Actor, train, evaluate
import numpy as np

config = RunConfig(seed=42)
rng_e = np.random.default_rng(0)
rng_p = np.random.default_rng(1)
env = GridEnv(config, rng=rng_e)
planner = HybridPlanner(config.grid_size, rng=rng_p)
actor = Actor(config.state_dim, config.num_uavs, config.action_dim, config.actor_hidden)
train(config, env, planner, actor)
evaluate(config, env, planner)
```

## Optimizations (algorithm)

### 1. BFS shortest-path planning

**What:** Replaced one-step greedy moves with BFS from each UAV’s position to **any cell in the goal zone** (distance ≤ 1 to that goal), respecting obstacles and cells reserved by other UAVs this timestep.

**Why:** Greedy steps get stuck; BFS finds a shortest feasible route when one exists.

### 2. Correct alignment of actions and grid moves in BFS

**What:** BFS expansions match `GridEnv.step`: `0` = left, `1` = right, `2` = down (y−1), `3` = up (y+1). First action is recovered via parent pointers.

**Why:** Wrong (dx, dy) ↔ action mapping breaks the planner.

### 3. Goal-zone “holding” behavior

**What:** Inside the goal zone, prefer moves that **stay** in the zone when possible.

**Why:** There is no “stay” action; otherwise agents can leave the success region before the episode ends.

### 4. Fallback when BFS is blocked

**What:** If no BFS path exists (e.g. reservations), fall back to distance-based greedy, then obstacle-only fallback.

**Why:** Defined behavior under multi-agent contention.

### 5. Imitation learning

**What:** Cross-entropy on logits vs hybrid actions; deeper MLP; gradient clipping; configurable LR.

**Why:** Proper discrete-action loss and stable optimization.

### 6. Goal-zone caching

**What:** `functools.lru_cache` on goal-zone cell sets keyed by `(grid_size, gx, gy)`.

**Why:** Avoids rebuilding the same sets every BFS call.

## Production-oriented changes

- **Package layout** and explicit **public API** in `multi_uav_grid/__init__.py`
- **`RunConfig`** instead of module-level constants
- **`logging`** instead of `print`
- **CLI** via `argparse` (`python -m multi_uav_grid`)
- **Structured step API** (`StepResult`) and validation on actions
- **Episode collision flag** aggregates collisions over all steps (not only the last)
- **Seeding**: `set_global_seeds` plus independent NumPy generators for env vs planner
- **`pyproject.toml`** / **`requirements.txt`** for packaging and deps
- **Device** selection for the Actor (`cpu` / `cuda`)

## Expected outcome

On default settings (20×20 grid, 40 obstacles, 3 UAVs, 100 steps), training and test success rates are **typically well above 92%**, driven by the hybrid planner; the Actor imitates it.

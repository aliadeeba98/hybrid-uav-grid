# Multi-UAV grid hybrid planner

Main script: [`multi_uav_grid_hybrid_planner.py`](multi_uav_grid_hybrid_planner.py) (formerly `hybrid_adeebs.py`).

Multi-UAV path planning on a 20×20 grid with random obstacles. The script runs a **hybrid controller** (heuristic planner) each environment step and trains an **Actor** network with **behavior cloning** (imitation) on the actions the hybrid produces. Training and test success are measured as: all UAVs end within Euclidean distance ≤ 1 of their goals, with **no obstacle or inter-agent collision** along the episode.

## Optimizations (what changed and why)

### 1. BFS shortest-path planning

**What:** Replaced one-step greedy moves (pick the neighbor closest to the goal) with breadth-first search from each UAV’s position to **any cell in the goal zone** (all cells with distance ≤ 1 to that UAV’s goal), respecting static obstacles and cells already reserved by other UAVs this timestep.

**Why:** Greedy steps get stuck in dead ends and waste steps; BFS finds a shortest feasible route when one exists, which sharply improves completion rate and stability across random maps.

### 2. Correct alignment of actions and grid moves in BFS

**What:** The expansion rules in BFS use the same semantics as `next_pos` / `GridEnv.step`: action `0` = left (x−1), `1` = right (x+1), `2` = down (y−1), `3` = up (y+1). The first step along a found path is recovered by walking parent pointers back to the start.

**Why:** If BFS uses the wrong (dx, dy) ↔ action mapping, the “optimal” first move does not match the environment, so the planner behaves incorrectly and success collapses.

### 3. Goal-zone “holding” behavior

**What:** When a UAV is already in the goal zone, it prefers moves that **keep** the next position inside the goal zone (with safe, non-reserved cells). Only if that is impossible does it fall back to other safe moves.

**Why:** There is no explicit “stay” action; without this rule, a UAV can reach the goal early and then **leave** the success region before the episode ends, failing the final success check even though it once arrived.

### 4. Fallback when BFS is blocked

**What:** If no BFS path exists to the goal zone (e.g. because another UAV reserved a critical cell this step), the planner falls back to the previous **distance-based greedy** rule (and then obstacle-only fallback).

**Why:** Keeps behavior defined under multi-agent contention; BFS alone can return no move when reservations block all shortest paths.

### 5. Imitation learning: cross-entropy, architecture, optimization

**What:**

- Loss is **cross-entropy** between raw logits (pre-softmax) and the hybrid’s discrete actions, instead of mean squared error between softmax outputs and one-hot targets.
- The Actor adds an extra **128-unit ReLU** layer before the output.
- **Gradient clipping** (norm1.0) is applied after `backward`.
- Adam learning rate was set to **1e-3** (from 3e-4).

**Why:** Cross-entropy is the standard objective for classifying discrete actions, so the network learns cleaner policies from the hybrid labels. A slightly deeper head and stable optimization help the clone track the improved hybrid signal.

### 6. Goal-zone caching

**What:** The set of grid cells in the goal zone for a given goal coordinate is cached in memory (`_GOAL_ZONE_CACHE`) so BFS does not rebuild that set every call.

**Why:** Small performance win; correctness is unchanged.

## How to run

```bash
python multi_uav_grid_hybrid_planner.py
```

Training runs for 5000 episodes; evaluation prints a cumulative training success rate per episode, then a **Final Test Success Rate** over 1000 episodes on the **fixed** map stored from training (`reset(fixed=True)`).

## Expected outcome

After these changes, **training and test success rates are typically well above 92%** on the default settings (40 random obstacle cells, 3 UAVs, 100 steps per episode), with the hybrid planner driving the measured success; the Actor is trained to imitate that planner.

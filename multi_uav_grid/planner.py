"""BFS-based hybrid multi-agent planner with goal-zone holding."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from multi_uav_grid.environment import GridEnv


def _in_goal_zone(p: tuple[int, int], goal: tuple[int, int]) -> bool:
    return float(np.linalg.norm(np.array(p) - np.array(goal))) <= 1.0


@lru_cache(maxsize=4096)
def _goal_zone_cells_frozen(grid_size: int, gx: int, gy: int) -> frozenset[tuple[int, int]]:
    g = np.array([gx, gy])
    cells: list[tuple[int, int]] = []
    for x in range(grid_size):
        for y in range(grid_size):
            if float(np.linalg.norm(np.array([x, y]) - g)) <= 1.0:
                cells.append((x, y))
    return frozenset(cells)


def next_pos(pos: tuple[int, int], action: int, grid_size: int) -> tuple[int, int]:
    x, y = pos
    high = grid_size - 1
    if action == 0:
        x -= 1
    elif action == 1:
        x += 1
    elif action == 2:
        y -= 1
    elif action == 3:
        y += 1
    else:
        raise ValueError(f"invalid action {action!r}")
    return (int(np.clip(x, 0, high)), int(np.clip(y, 0, high)))


def _is_safe(p: tuple[int, int], obstacles: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return (p not in obstacles) and (p not in occupied)


def _bfs_first_action(
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]],
    occupied_next: set[tuple[int, int]],
    grid_size: int,
) -> int | None:
    targets = _goal_zone_cells_frozen(grid_size, goal[0], goal[1])
    if start in targets:
        return None

    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    action_from_parent: dict[tuple[int, int], int | None] = {start: None}
    deltas = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    while q:
        cur = q.popleft()
        if cur in targets:
            first_a: int | None = None
            node = cur
            while parent[node] is not None:
                first_a = action_from_parent[node]
                node = parent[node]  # type: ignore[assignment]
            return first_a

        x, y = cur
        for dx, dy, a in deltas:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
                continue
            nxt = (nx, ny)
            if nxt in obstacles or nxt in occupied_next or nxt in parent:
                continue
            parent[nxt] = cur
            action_from_parent[nxt] = a
            q.append(nxt)

    return None


class HybridPlanner:
    """Prioritized planning: closest-to-goal UAVs choose first; BFS + greedy fallback."""

    def __init__(self, grid_size: int, rng: np.random.Generator | None = None) -> None:
        self._grid_size = grid_size
        self._rng = rng if rng is not None else np.random.default_rng()

    def _greedy_fallback(self, env: GridEnv, agent_index: int, occupied: set[tuple[int, int]]) -> int:
        x, y = env.pos[agent_index]
        gx, gy = env.goals[agent_index]
        actions = [0, 1, 2, 3]
        best: int | None = None
        best_score = 1e9

        for a in actions:
            p = next_pos((x, y), a, self._grid_size)
            if not _is_safe(p, env.obstacles, occupied):
                continue
            dist = float(np.linalg.norm(np.array(p) - np.array([gx, gy])))
            if dist < best_score:
                best_score = dist
                best = a

        if best is None:
            for a in actions:
                p = next_pos((x, y), a, self._grid_size)
                if p not in env.obstacles:
                    return a
            return 0
        return best

    def _best_action(self, env: GridEnv, agent_index: int, occupied: set[tuple[int, int]]) -> int:
        pos = env.pos[agent_index]
        goal = env.goals[agent_index]
        x, y = pos

        if _in_goal_zone(pos, goal):
            actions = [0, 1, 2, 3]
            self._rng.shuffle(actions)
            for a in actions:
                p = next_pos((x, y), a, self._grid_size)
                if _is_safe(p, env.obstacles, occupied) and _in_goal_zone(p, goal):
                    return a
            for a in actions:
                p = next_pos((x, y), a, self._grid_size)
                if _is_safe(p, env.obstacles, occupied):
                    return a
            return 0

        a = _bfs_first_action(pos, goal, env.obstacles, occupied, self._grid_size)
        if a is not None:
            p = next_pos((x, y), a, self._grid_size)
            if _is_safe(p, env.obstacles, occupied):
                return a

        return self._greedy_fallback(env, agent_index, occupied)

    def plan(self, env: GridEnv) -> list[int]:
        actions = [0] * env.num_uavs
        occupied: set[tuple[int, int]] = set()

        order = sorted(
            range(env.num_uavs),
            key=lambda i: float(np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i]))),
        )

        for i in order:
            a = self._best_action(env, i, occupied)
            actions[i] = a
            occupied.add(next_pos(env.pos[i], a, self._grid_size))

        return actions

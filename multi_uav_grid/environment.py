"""Grid world with multiple UAVs and obstacles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_uav_grid.config import RunConfig


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    collision: bool


class GridEnv:
    """Multi-UAV grid environment; coordinates are in [0, grid_size - 1]."""

    def __init__(self, config: RunConfig, rng: np.random.Generator | None = None) -> None:
        self._config = config
        self._rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.num_obstacles = config.num_obstacles
        self.grid_size = config.grid_size
        self.num_uavs = config.num_uavs
        self.max_steps = config.max_steps_per_episode
        self.fixed_map: set[tuple[int, int]] | None = None
        self.obstacles: set[tuple[int, int]] = set()
        self.starts: list[tuple[int, int]] = []
        self.goals: list[tuple[int, int]] = []
        self.pos: list[tuple[int, int]] = []
        self._high = config.grid_size - 1
        self.reset()

    def _rand_cell(self) -> tuple[int, int]:
        return (
            int(self._rng.integers(0, self.grid_size)),
            int(self._rng.integers(0, self.grid_size)),
        )

    def reset(self, *, fixed: bool = False) -> np.ndarray:
        if fixed and self.fixed_map is not None:
            self.obstacles = set(self.fixed_map)
        else:
            self.obstacles = set()
            while len(self.obstacles) < self.num_obstacles:
                self.obstacles.add(self._rand_cell())
            self.fixed_map = set(self.obstacles)

        self.starts = []
        self.goals = []
        for _ in range(self.num_uavs):
            while True:
                s = self._rand_cell()
                if s not in self.obstacles:
                    break
            while True:
                g = self._rand_cell()
                if g not in self.obstacles and g != s:
                    break
            self.starts.append(s)
            self.goals.append(g)

        self.pos = list(self.starts)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        g = float(self.grid_size)
        state: list[float] = []
        for i in range(self.num_uavs):
            x, y = self.pos[i]
            gx, gy = self.goals[i]
            state.extend([x / g, y / g, gx / g, gy / g, (gx - x) / g, (gy - y) / g])
        return np.asarray(state, dtype=np.float32)

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.num_uavs:
            raise ValueError(f"expected {self.num_uavs} actions, got {len(actions)}")

        new_pos: list[tuple[int, int]] = []
        collision = False

        for i, a in enumerate(actions):
            x, y = self.pos[i]
            if a == 0:
                x -= 1
            elif a == 1:
                x += 1
            elif a == 2:
                y -= 1
            elif a == 3:
                y += 1
            else:
                raise ValueError(f"invalid action {a!r}; expected 0–3")

            x = int(np.clip(x, 0, self._high))
            y = int(np.clip(y, 0, self._high))
            new_pos.append((x, y))

        for i, p in enumerate(new_pos):
            if p in self.obstacles:
                collision = True
            for j in range(len(new_pos)):
                if i != j and p == new_pos[j]:
                    collision = True

        old_sum_dist = sum(
            float(np.linalg.norm(np.array(self.pos[i]) - np.array(self.goals[i])))
            for i in range(self.num_uavs)
        )
        self.pos = new_pos
        new_sum_dist = sum(
            float(np.linalg.norm(np.array(self.pos[i]) - np.array(self.goals[i])))
            for i in range(self.num_uavs)
        )

        rewards: list[float] = []
        done = True
        cp = self._config.collision_penalty

        for i in range(self.num_uavs):
            dist = float(np.linalg.norm(np.array(self.pos[i]) - np.array(self.goals[i])))
            r = -0.01 * dist
            if dist <= 1.0:
                r += 200.0
            else:
                done = False
            if collision:
                r -= cp
            rewards.append(r)

        total = float(sum(rewards))
        if not collision and self._config.dense_reward_coef > 0.0:
            total += self._config.dense_reward_coef * max(0.0, old_sum_dist - new_sum_dist)

        obs = self.get_state()
        return StepResult(observation=obs, reward=total, done=done, collision=collision)

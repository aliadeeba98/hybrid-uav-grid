"""Typed configuration for environment and training runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Single place for hyperparameters and run controls."""

    grid_size: int = 20
    num_uavs: int = 3
    num_obstacles: int = 40
    max_steps_per_episode: int = 100
    train_episodes: int = 5000
    test_episodes: int = 1000
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 128)
    log_every_episodes: int = 1
    seed: int | None = None
    device: str = "cpu"

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def state_dim(self) -> int:
        return self.num_uavs * 6

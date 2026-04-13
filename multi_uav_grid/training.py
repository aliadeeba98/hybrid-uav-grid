"""Training and evaluation loops."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_uav_grid.config import RunConfig
from multi_uav_grid.environment import GridEnv
from multi_uav_grid.planner import HybridPlanner
from multi_uav_grid.policy import Actor


@dataclass(frozen=True)
class TrainSummary:
    episodes: int
    successes: int
    success_rate: float


@dataclass(frozen=True)
class EvalSummary:
    episodes: int
    successes: int
    success_rate: float


def _episode_success(env: GridEnv, collision_during_episode: bool) -> bool:
    if collision_during_episode:
        return False
    for i in range(env.num_uavs):
        if float(np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i]))) > 1.0:
            return False
    return True


def train(
    config: RunConfig,
    env: GridEnv,
    planner: HybridPlanner,
    actor: Actor,
    logger: logging.Logger | None = None,
) -> TrainSummary:
    log = logger or logging.getLogger(__name__)
    device = torch.device(config.device)
    actor.to(device)
    optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)

    successes = 0
    for ep in range(config.train_episodes):
        obs = env.reset()
        had_collision = False

        for _t in range(config.max_steps_per_episode):
            actions = planner.plan(env)
            step = env.step(actions)

            st = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor.logits(st)
            target = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config.action_dim),
                target.reshape(-1),
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip_norm)
            optimizer.step()

            obs = step.observation

            had_collision = had_collision or step.collision
            if step.done:
                break

        if _episode_success(env, had_collision):
            successes += 1

        if config.log_every_episodes > 0 and (ep + 1) % config.log_every_episodes == 0:
            rate = 100.0 * successes / (ep + 1)
            log.info("episode %s success_rate=%.2f%%", ep + 1, rate)

    rate = successes / config.train_episodes if config.train_episodes else 0.0
    return TrainSummary(episodes=config.train_episodes, successes=successes, success_rate=rate)


def evaluate(
    config: RunConfig,
    env: GridEnv,
    planner: HybridPlanner,
    logger: logging.Logger | None = None,
) -> EvalSummary:
    log = logger or logging.getLogger(__name__)
    successes = 0
    for _ in range(config.test_episodes):
        env.reset(fixed=True)
        had_collision = False

        for _t in range(config.max_steps_per_episode):
            actions = planner.plan(env)
            step = env.step(actions)
            had_collision = had_collision or step.collision
            if step.done:
                break

        if _episode_success(env, had_collision):
            successes += 1

    rate = successes / config.test_episodes if config.test_episodes else 0.0
    log.info("eval complete success_rate=%.2f%% (%s/%s)", rate * 100, successes, config.test_episodes)
    return EvalSummary(episodes=config.test_episodes, successes=successes, success_rate=rate)


def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

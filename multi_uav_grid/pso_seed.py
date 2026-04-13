"""Particle Swarm Optimization over RNG seeds for SAC network initialization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from multi_uav_grid.actions import joint_to_actions
from multi_uav_grid.sac import SACAgent

if TYPE_CHECKING:
    from multi_uav_grid.config import RunConfig
    from multi_uav_grid.environment import GridEnv


def _set_torch_np_seed(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def evaluate_init_seed(
    seed: int,
    config: RunConfig,
    device: torch.device,
    make_env: Callable[[], GridEnv],
    logger: logging.Logger | None = None,
) -> float:
    """Mean episode return over short rollouts with a freshly seeded SAC agent (stochastic policy)."""
    log = logger or logging.getLogger(__name__)
    _set_torch_np_seed(seed)
    agent = SACAgent(config, device)
    total_returns: list[float] = []
    for ep in range(config.pso_episodes_per_eval):
        env = make_env()
        obs = env.reset()
        ep_ret = 0.0
        for _ in range(config.max_steps_per_episode):
            ja = agent.select_action(obs, deterministic=False)
            actions = joint_to_actions(ja, config.num_uavs, config.per_uav_action_dim)
            step = env.step(actions)
            ep_ret += step.reward
            obs = step.observation
            if step.done:
                break
        total_returns.append(ep_ret)
    mean_ret = float(np.mean(total_returns))
    log.debug("pso seed %s mean_return=%.4f", seed, mean_ret)
    return mean_ret


def pso_find_best_seed(
    config: RunConfig,
    device: torch.device,
    make_env: Callable[[], GridEnv],
    logger: logging.Logger | None = None,
) -> int:
    """Run PSO on integer seeds; return best seed for full SAC training initialization."""
    log = logger or logging.getLogger(__name__)
    rng = np.random.default_rng(config.seed if config.seed is not None else None)
    n = config.pso_swarm_size
    dim = 1
    bounds = np.array([[config.pso_seed_low, config.pso_seed_high]], dtype=np.float64)

    positions = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, dim))
    velocities = rng.uniform(-(bounds[:, 1] - bounds[:, 0]) * 0.1, (bounds[:, 1] - bounds[:, 0]) * 0.1, size=(n, dim))

    fitness = np.array([evaluate_init_seed(int(p[0]) % (2**31), config, device, make_env, log) for p in positions])
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()
    g_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    w, c1, c2 = config.pso_inertia, config.pso_cognitive, config.pso_social

    for gen in range(config.pso_generations):
        r1, r2 = rng.random((n, dim)), rng.random((n, dim))
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        positions = np.clip(positions + velocities, bounds[:, 0], bounds[:, 1])
        fitness = np.array([evaluate_init_seed(int(p[0]) % (2**31), config, device, make_env, log) for p in positions])
        improved = fitness > pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit = np.maximum(pbest_fit, fitness)
        g_idx = int(np.argmax(pbest_fit))
        if float(pbest_fit[g_idx]) > gbest_fit:
            gbest_fit = float(pbest_fit[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()
        log.info("pso generation %s best_return=%.4f best_seed=%s", gen + 1, gbest_fit, int(gbest_pos[0]) % (2**31))

    best = int(gbest_pos[0]) % (2**31)
    log.info("pso finished best_seed=%s best_return=%.4f", best, gbest_fit)
    return best

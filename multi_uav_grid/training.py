"""Training and evaluation with PSO-seeded discrete SAC + expert pre-fill."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from multi_uav_grid.actions import actions_to_joint, joint_to_actions
from multi_uav_grid.config import RunConfig
from multi_uav_grid.environment import GridEnv
from multi_uav_grid.planner import HybridPlanner
from multi_uav_grid.pso_seed import pso_find_best_seed
from multi_uav_grid.sac import SACAgent


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


def _init_seed_for_sac(config: RunConfig, device: torch.device, logger: logging.Logger) -> int:
    if config.use_pso:

        def make_env() -> GridEnv:
            return GridEnv(config, rng=np.random.default_rng())

        return pso_find_best_seed(config, device, make_env, logger)
    if config.seed is not None:
        return int(config.seed) % (2**31)
    return int(np.random.default_rng().integers(1, 2**31))


def _effective_prefill(config: RunConfig) -> int:
    """Cap expert episodes so at least ~100 rollout episodes use the learned SAC policy."""
    if config.train_episodes <= 1:
        return 0
    max_prefill = max(0, config.train_episodes - 100)
    return min(config.expert_prefill_episodes, max_prefill)


def _exploration_epsilon(ep: int, config: RunConfig) -> float:
    if config.explore_decay_episodes <= 0:
        return config.exploration_eps_end
    t = min(1.0, ep / float(config.explore_decay_episodes))
    return config.exploration_eps_start + t * (config.exploration_eps_end - config.exploration_eps_start)


def _bc_policy_step(agent: SACAgent, obs: np.ndarray, joint_action: int, weight: float) -> None:
    if weight <= 0.0:
        return
    device = agent.device
    s = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits = agent.policy(s)
    loss = weight * F.cross_entropy(logits, torch.tensor([joint_action], device=device))
    agent.opt_pi.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.config.grad_clip_norm)
    agent.opt_pi.step()


def train(
    config: RunConfig,
    env: GridEnv,
    agent: SACAgent,
    planner_rng: np.random.Generator | None = None,
    logger: logging.Logger | None = None,
) -> TrainSummary:
    log = logger or logging.getLogger(__name__)
    successes = 0
    bs = config.sac_batch_size
    n_up = config.sac_updates_per_step
    device = agent.device
    planner = HybridPlanner(config.grid_size, rng=planner_rng or np.random.default_rng(0))
    explore_rng = np.random.default_rng(
        (config.seed if config.seed is not None else 0) + 7919
    )

    prefill_n = _effective_prefill(config)
    log.info("expert_prefill_episodes=%s (effective=%s)", config.expert_prefill_episodes, prefill_n)

    for ep in range(config.train_episodes):
        obs = env.reset()
        had_collision = False
        use_expert = ep < prefill_n
        # Pure expert during pre-fill so replay is high-quality (random joints here killed success)
        eps = 0.0 if use_expert else _exploration_epsilon(ep, config)

        for _t in range(config.max_steps_per_episode):
            if use_expert:
                act_list = planner.plan(env)
                ja = actions_to_joint(act_list, config.per_uav_action_dim)
            else:
                if explore_rng.random() < eps:
                    ja = int(explore_rng.integers(0, agent.n_actions))
                else:
                    ja = agent.select_action(obs, deterministic=False)

            obs_before = obs
            actions = joint_to_actions(ja, config.num_uavs, config.per_uav_action_dim)
            step = env.step(actions)
            agent.replay.push(obs_before, ja, step.reward, step.observation, step.done)
            obs = step.observation
            had_collision = had_collision or step.collision

            if agent.replay.size >= bs:
                for _ in range(n_up):
                    agent.update(bs)
                if use_expert:
                    _bc_policy_step(agent, obs_before, ja, config.expert_bc_weight)

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
    agent: SACAgent,
    logger: logging.Logger | None = None,
) -> EvalSummary:
    log = logger or logging.getLogger(__name__)
    successes = 0
    for _ in range(config.test_episodes):
        obs = env.reset(fixed=True)
        had_collision = False

        for _t in range(config.max_steps_per_episode):
            ja = agent.select_action(obs, deterministic=config.eval_deterministic)
            actions = joint_to_actions(ja, config.num_uavs, config.per_uav_action_dim)
            step = env.step(actions)
            obs = step.observation
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

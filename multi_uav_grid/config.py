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
    # Potential-based style shaping: bonus when total distance-to-goal decreases (helps SAC signal)
    dense_reward_coef: float = 0.25
    collision_penalty: float = 35.0
    grad_clip_norm: float = 1.0
    log_every_episodes: int = 1
    seed: int | None = None
    device: str = "cpu"

    # Discrete action per UAV (grid moves); joint space size = per_uav_action_dim ** num_uavs
    per_uav_action_dim: int = 4

    # SAC
    sac_hidden: tuple[int, ...] = (256, 128)
    sac_lr_policy: float = 3e-4
    sac_lr_q: float = 3e-4
    sac_gamma: float = 0.995
    sac_tau: float = 0.005
    sac_alpha: float = 0.15
    sac_reward_scale: float = 0.02
    sac_batch_size: int = 256
    replay_capacity: int = 200_000
    sac_updates_per_step: int = 4
    # Fill replay with HybridPlanner (BFS expert) before / while learning — large impact on success
    expert_prefill_episodes: int = 400
    # Extra policy CE toward expert action during prefill only
    expert_bc_weight: float = 0.5
    # ε-greedy random joint action; decays linearly over first `explore_decay_episodes`
    exploration_eps_start: float = 0.25
    exploration_eps_end: float = 0.02
    explore_decay_episodes: int = 2500

    # If False, eval samples from π (better match to training; greedy argmax can be brittle on joint actions)
    eval_deterministic: bool = False

    # PSO over RNG seeds before SAC (off by default — use --pso; keep swarm small)
    use_pso: bool = False
    pso_swarm_size: int = 6
    pso_generations: int = 1
    pso_episodes_per_eval: int = 1
    pso_seed_low: float = 1.0
    pso_seed_high: float = 1_000_000.0
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5

    @property
    def action_dim(self) -> int:
        return self.per_uav_action_dim

    @property
    def state_dim(self) -> int:
        return self.num_uavs * 6

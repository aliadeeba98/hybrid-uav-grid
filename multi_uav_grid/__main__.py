"""CLI: ``python -m multi_uav_grid``."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch

from multi_uav_grid.config import RunConfig
from multi_uav_grid.environment import GridEnv
from multi_uav_grid.sac import SACAgent
from multi_uav_grid.training import _init_seed_for_sac, evaluate, set_global_seeds, train


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-UAV grid — PSO-seeded discrete SAC")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--num-uavs", type=int, default=3)
    p.add_argument("--num-obstacles", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--test-episodes", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1, help="log every N training episodes (0=quiet)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--pso",
        action="store_true",
        help="Run PSO over init seeds before SAC (slow; can take minutes)",
    )
    p.add_argument(
        "--eval-greedy",
        action="store_true",
        help="Use argmax policy at test time (default: sample from π, often higher success)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("multi_uav_grid")

    config = RunConfig(
        grid_size=args.grid_size,
        num_uavs=args.num_uavs,
        num_obstacles=args.num_obstacles,
        max_steps_per_episode=args.max_steps,
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        grad_clip_norm=args.grad_clip,
        log_every_episodes=args.log_every,
        seed=args.seed,
        device=args.device,
        use_pso=args.pso,
        eval_deterministic=args.eval_greedy,
    )

    device = torch.device(config.device)
    set_global_seeds(config.seed)

    ss = np.random.SeedSequence(config.seed)
    child = list(ss.spawn(2))
    env_rng = np.random.default_rng(child[0])
    planner_rng = np.random.default_rng(child[1])
    env = GridEnv(config, rng=env_rng)

    init_seed = _init_seed_for_sac(config, device, log)
    log.info("SAC init seed=%s (pso=%s)", init_seed, config.use_pso)
    np.random.seed(init_seed % (2**32))
    torch.manual_seed(init_seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed % (2**32))

    agent = SACAgent(config, device)

    summary = train(config, env, agent, planner_rng=planner_rng, logger=log)
    log.info(
        "training finished success_rate=%.2f%% (%s/%s)",
        summary.success_rate * 100,
        summary.successes,
        summary.episodes,
    )

    ev = evaluate(config, env, agent, logger=log)
    log.info("final_test success_rate=%.2f%%", ev.success_rate * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())

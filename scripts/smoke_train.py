"""Minimal train+eval for CI/smoke (run from repo root: python scripts/smoke_train.py)."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from multi_uav_grid import RunConfig, GridEnv, SACAgent
from multi_uav_grid.training import evaluate, train
import numpy as np
import torch

c = RunConfig(
    train_episodes=5,
    test_episodes=3,
    expert_prefill_episodes=2,
    log_every_episodes=1,
    seed=0,
)
env = GridEnv(c, rng=np.random.default_rng(0))
planner_rng = np.random.default_rng(1)
agent = SACAgent(c, torch.device("cpu"))
train(c, env, agent, planner_rng=planner_rng)
evaluate(c, env, agent)
print("smoke_import_train_eval_ok")

"""Multi-UAV grid — PSO-seeded discrete SAC."""

from multi_uav_grid.config import RunConfig
from multi_uav_grid.environment import GridEnv, StepResult
from multi_uav_grid.sac import SACAgent
from multi_uav_grid.training import EvalSummary, TrainSummary, evaluate, train

__all__ = [
    "EvalSummary",
    "GridEnv",
    "RunConfig",
    "SACAgent",
    "StepResult",
    "TrainSummary",
    "evaluate",
    "train",
]

__version__ = "1.0.0"

"""Joint discrete action encoding (one categorical per team step)."""

from __future__ import annotations


def joint_action_dim(num_uavs: int, action_dim: int = 4) -> int:
    return int(action_dim**num_uavs)


def joint_to_actions(joint: int, num_uavs: int, action_dim: int = 4) -> list[int]:
    out: list[int] = []
    x = joint
    for _ in range(num_uavs):
        out.append(x % action_dim)
        x //= action_dim
    return out


def actions_to_joint(actions: list[int], action_dim: int = 4) -> int:
    j = 0
    for i, a in enumerate(actions):
        j += int(a) * (action_dim**i)
    return j

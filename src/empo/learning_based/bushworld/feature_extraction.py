"""
Feature extraction helpers for BushWorld neural network encoders.

These functions convert a raw BushWorld state ``(step_count, positions,
densities)`` into the flat input vector consumed by
:class:`~empo.learning_based.bushworld.state_encoder.BushWorldStateEncoder`.

The layout mirrors the multigrid design (a small multi-channel grid plus global
features), but is much simpler because BushWorld has no colours, directions or
interactive objects.
"""

from typing import Any, Tuple

import numpy as np

from .constants import (
    AGENT_FEATURE_SIZE,
    GOAL_COORD_DIM,
    NUM_GRID_CHANNELS,
)


def extract_state_grid(
    state: Any,
    grid_width: int,
    grid_height: int,
    num_robots: int,
    B: int,
) -> np.ndarray:
    """Return a ``(NUM_GRID_CHANNELS, H, W)`` float array for ``state``.

    Channels: 0 = bush density / ``B``, 1 = robot occupancy, 2 = human occupancy.
    """
    _, positions, densities = state
    grid = np.zeros((NUM_GRID_CHANNELS, grid_height, grid_width), dtype=np.float32)

    denom = float(max(B, 1))
    dens = np.asarray(densities, dtype=np.float32).reshape(grid_height, grid_width)
    grid[0] = dens / denom

    for i, (x, y) in enumerate(positions):
        if i < num_robots:
            grid[1, int(y), int(x)] = 1.0
        else:
            grid[2, int(y), int(x)] = 1.0
    return grid


def extract_global_world_features(state: Any, max_steps: int) -> np.ndarray:
    """Return the global (non-spatial) features: just the normalised step count."""
    step_count = state[0]
    denom = float(max(max_steps, 1))
    return np.array([step_count / denom], dtype=np.float32)


def extract_state_vector(
    state: Any,
    grid_width: int,
    grid_height: int,
    num_robots: int,
    B: int,
    max_steps: int,
) -> np.ndarray:
    """Return the flat input vector (grid channels flattened + global features)."""
    grid = extract_state_grid(state, grid_width, grid_height, num_robots, B)
    glob = extract_global_world_features(state, max_steps)
    return np.concatenate([grid.reshape(-1), glob], axis=0)


def extract_goal_coords(goal: Any, grid_width: int, grid_height: int) -> np.ndarray:
    """Return the normalised goal bounding box ``(x1, y1, x2, y2)``.

    Works for both cell goals (``target_pos``) and rectangle goals
    (``target_rect``); both expose ``target_rect``.
    """
    if hasattr(goal, "target_rect"):
        x1, y1, x2, y2 = goal.target_rect
    elif hasattr(goal, "target_pos"):
        x, y = goal.target_pos
        x1, y1, x2, y2 = x, y, x, y
    else:
        raise ValueError(f"Unsupported goal type for BushWorld encoder: {goal!r}")
    wn = float(max(grid_width - 1, 1))
    hn = float(max(grid_height - 1, 1))
    return np.array([x1 / wn, y1 / hn, x2 / wn, y2 / hn], dtype=np.float32)


def extract_agent_features(
    state: Any, agent_index: int, num_agents: int, grid_width: int, grid_height: int
) -> np.ndarray:
    """Return identity features for ``agent_index``: (idx, x, y), all normalised."""
    _, positions, _ = state
    x, y = positions[agent_index]
    wn = float(max(grid_width - 1, 1))
    hn = float(max(grid_height - 1, 1))
    idx_norm = agent_index / float(max(num_agents - 1, 1))
    return np.array([idx_norm, x / wn, y / hn], dtype=np.float32)


def goal_to_target(goal: Any) -> Tuple[int, int]:
    """Return a representative ``(x, y)`` target for a goal (rect centre or cell)."""
    if hasattr(goal, "target_pos"):
        return tuple(int(c) for c in goal.target_pos)
    if hasattr(goal, "target_rect"):
        x1, y1, x2, y2 = goal.target_rect
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    raise ValueError(f"Unsupported goal type for BushWorld encoder: {goal!r}")


__all__ = [
    "extract_state_grid",
    "extract_global_world_features",
    "extract_state_vector",
    "extract_goal_coords",
    "extract_agent_features",
    "goal_to_target",
    "AGENT_FEATURE_SIZE",
    "GOAL_COORD_DIM",
]

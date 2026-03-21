"""
MultiGrid-specific PPO-based Phase 2 implementations.

This module provides MultiGrid environment-specific subclasses for the
PPO Phase 2 trainer.  It does NOT modify any code in
``learning_based/multigrid/phase2/``.

Main components:
    - ``MultiGridWorldModelEnv`` — env wrapper with MultiGrid state encoding
    - ``create_multigrid_ppo_networks`` — factory for actor-critic +
      auxiliary networks with shared encoder
"""

from .env_wrapper import MultiGridWorldModelEnv
from .networks import create_multigrid_ppo_networks

__all__ = [
    "MultiGridWorldModelEnv",
    "create_multigrid_ppo_networks",
]

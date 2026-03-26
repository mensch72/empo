"""
MultiGrid Phase 1 PPO: Goal-conditioned human policy prior via PPO.

Provides:
- :class:`MultiGridPhase1PPOEnv` — MultiGrid-specific PPO env wrapper
- :func:`create_multigrid_phase1_ppo_networks` — Network factory
"""

from .env_wrapper import MultiGridPhase1PPOEnv
from .networks import create_multigrid_phase1_ppo_networks

__all__ = [
    "MultiGridPhase1PPOEnv",
    "create_multigrid_phase1_ppo_networks",
]

"""
BushWorld Phase 1 PPO: Goal-conditioned human policy prior via PPO.

This subpackage is the BushWorld analogue of
:mod:`empo.learning_based.multigrid.phase1_ppo`. It reuses the shared PufferLib
PPO infrastructure in :mod:`empo.learning_based.phase1_ppo` (no new training
algorithm) and only provides the BushWorld-specific observation encoding and a
network factory.

Provides:
- :class:`BushWorldPhase1PPOEnv` — BushWorld-specific PPO env wrapper.
- :func:`create_bushworld_phase1_ppo_networks` — network factory.
"""

from .env_wrapper import BushWorldPhase1PPOEnv
from .networks import create_bushworld_phase1_ppo_networks

__all__ = [
    "BushWorldPhase1PPOEnv",
    "create_bushworld_phase1_ppo_networks",
]

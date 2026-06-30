"""
BushWorld-specific PPO-based Phase 2 implementations.

This subpackage is the BushWorld analogue of
:mod:`empo.learning_based.multigrid.phase2_ppo`. It reuses the shared PufferLib
PPO Phase 2 infrastructure in :mod:`empo.learning_based.phase2_ppo` (no new
training algorithm) and only provides the BushWorld-specific observation
encoding and a network factory.

Main components:
    - :class:`BushWorldWorldModelEnv` — env wrapper with BushWorld state encoding.
    - :func:`create_bushworld_ppo_networks` — factory for actor-critic +
      auxiliary networks with a shared state encoder.
"""

from .env_wrapper import BushWorldWorldModelEnv
from .networks import create_bushworld_ppo_networks

__all__ = [
    "BushWorldWorldModelEnv",
    "create_bushworld_ppo_networks",
]

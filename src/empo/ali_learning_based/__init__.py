"""
Ali's learning-based EMPO module.

Provides a high-level pipeline for training the full EMPO system:
Phase 1: Human Q-learning with UCB exploration and PBRS
Phase 2: Robot policy learning to maximize human empowerment
"""

from .envs import get_env_path, load_env_from_yaml
from .pipeline import run_empo_learning, EMPOResult

__all__ = [
    'get_env_path',
    'load_env_from_yaml',
    'run_empo_learning',
    'EMPOResult',
]

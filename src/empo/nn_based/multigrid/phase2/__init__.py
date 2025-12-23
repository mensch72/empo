"""
Multigrid-specific Phase 2 implementations.

This module provides environment-specific implementations of Phase 2
neural networks for multigrid environments.
"""

from .robot_q_network import MultiGridRobotQNetwork

__all__ = [
    'MultiGridRobotQNetwork',
]

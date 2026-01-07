"""
Robot Policy for Phase 2 - Re-export.

The base RobotPolicy class has been moved to empo.robot_policy to be consistent
with the HumanPolicyPrior design pattern (top-level base class, implementations
in subpackages).

This module re-exports it for backward compatibility with existing imports.
"""

from empo.robot_policy import RobotPolicy

__all__ = ['RobotPolicy']

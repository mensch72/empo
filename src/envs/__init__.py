"""
Environment module for EMPO framework.

This module contains custom MultiGrid environments for multi-agent reinforcement learning.
"""

from .one_or_three_chambers import OneOrThreeChambersEnv, OneOrThreeChambersMapEnv, SmallOneOrTwoChambersMapEnv

__all__ = ['OneOrThreeChambersEnv', 'OneOrThreeChambersMapEnv', 'SmallOneOrTwoChambersMapEnv']

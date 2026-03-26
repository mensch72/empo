"""Two-level hierarchical model for MultiGrid environments.

Wires together a MacroGridEnv (M^0) and a MultiGridEnv (M^1) via a
MultiGridLevelMapper to form a two-level HierarchicalWorldModel.
"""

from typing import Any, Optional

from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.macro_grid_env import MacroGridEnv
from empo.hierarchical.multigrid_level_mapper import MultiGridLevelMapper


class TwoLevelMultigrid(HierarchicalWorldModel):
    """A two-level hierarchical model for MultiGrid environments.

    Given a MultiGridEnv as M^1 (the micro/fine level), constructs M^0
    (the macro/coarse level) whose "cells" are rectangular blocks of
    walkable M^1 cells, and a MultiGridLevelMapper connecting the two.

    Usage::

        micro_env = MultiGridEnv(config_file='my_grid.yaml')
        hierarchy = TwoLevelMultigrid(micro_env)
        macro_env = hierarchy.coarsest()   # MacroGridEnv
        micro_env = hierarchy.finest()     # MultiGridEnv

    Args:
        micro_env: The MultiGridEnv serving as M^1.
        seed: Random seed for cell partition tie-breaking.
    """

    def __init__(self, micro_env: Any, *, seed: Optional[int] = None):
        macro_env = MacroGridEnv(micro_env, seed=seed)
        mapper = MultiGridLevelMapper(macro_env, micro_env)
        super().__init__(levels=[macro_env, micro_env], mappers=[mapper])

    @property
    def macro_env(self) -> MacroGridEnv:
        """The macro-level environment (M^0)."""
        return self.levels[0]

    @property
    def micro_env(self) -> Any:
        """The micro-level environment (M^1)."""
        return self.levels[1]

    @property
    def mapper(self) -> MultiGridLevelMapper:
        """The level mapper connecting M^0 to M^1."""
        return self.mappers[0]

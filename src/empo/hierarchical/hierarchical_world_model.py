"""
Hierarchical World Model: a sequence of world models connected by level mappers.

A HierarchicalWorldModel consists of L world models M^0, ..., M^{L-1} from
coarsest (level 0) to finest (level L-1), connected by L-1 LevelMappers
F^0, ..., F^{L-2} where F^ℓ connects M^ℓ (coarser) to M^{ℓ+1} (finer).
"""

from typing import List

from empo.world_model import WorldModel
from empo.hierarchical.level_mapper import LevelMapper


class HierarchicalWorldModel:
    """A hierarchical model consisting of L world models connected by level mappers.

    Contains a sequence of WorldModels M^0, ..., M^{L-1} from coarsest to finest,
    and L-1 LevelMappers F^0, ..., F^{L-2} connecting adjacent levels.

    The LevelMapper F^ℓ connects M^ℓ (the coarser) to M^{ℓ+1} (the finer).

    Attributes:
        levels: List[WorldModel] — the world models [M^0, ..., M^{L-1}]
        mappers: List[LevelMapper] — the level mappers [F^0, ..., F^{L-2}]
    """

    def __init__(
        self, levels: List[WorldModel], mappers: List[LevelMapper]
    ):
        """Initialize the hierarchical world model.

        Args:
            levels: List of WorldModels [M^0, ..., M^{L-1}] from coarsest to finest.
                Must have at least 2 levels.
            mappers: List of LevelMappers [F^0, ..., F^{L-2}] connecting adjacent levels.
                Must have len(mappers) == len(levels) - 1.

        Raises:
            ValueError: If fewer than 2 levels are provided, or if the number of
                mappers doesn't match len(levels) - 1.
        """
        if len(levels) < 2:
            raise ValueError(
                f"HierarchicalWorldModel requires at least 2 levels, got {len(levels)}"
            )
        if len(mappers) != len(levels) - 1:
            raise ValueError(
                f"Expected {len(levels) - 1} mappers for {len(levels)} levels, "
                f"got {len(mappers)}"
            )
        self.levels = levels
        self.mappers = mappers

    @property
    def num_levels(self) -> int:
        """Return the number of levels in the hierarchy."""
        return len(self.levels)

    def coarsest(self) -> WorldModel:
        """Return the coarsest (macro) world model M^0."""
        return self.levels[0]

    def finest(self) -> WorldModel:
        """Return the finest (micro) world model M^{L-1}."""
        return self.levels[-1]

"""
Hierarchical World Model: a sequence of world models connected by level mappers.

A HierarchicalWorldModel consists of L world models M^0, ..., M^{L-1} from
coarsest (level 0) to finest (level L-1), connected by L-1 LevelMappers
F^0, ..., F^{L-2} where F^l connects M^l (coarser) to M^{l+1} (finer).
"""

from typing import List

from empo.world_model import WorldModel
from empo.hierarchical.level_mapper import LevelMapper


class HierarchicalWorldModel:
    """A hierarchical model consisting of L world models connected by level mappers.

    Contains a sequence of WorldModels M^0, ..., M^{L-1} from coarsest to finest,
    and L-1 LevelMappers F^0, ..., F^{L-2} connecting adjacent levels.

    The LevelMapper F^l connects M^l (the coarser) to M^{l+1} (the finer).

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
        # Type checking for levels
        assert isinstance(levels, list), f"levels must be a list, got {type(levels)}"
        for i, level in enumerate(levels):
            assert isinstance(level, WorldModel), (
                f"levels[{i}] must be a WorldModel instance, got {type(level)}"
            )
        
        # Type checking for mappers
        assert isinstance(mappers, list), f"mappers must be a list, got {type(mappers)}"
        for i, mapper in enumerate(mappers):
            assert isinstance(mapper, LevelMapper), (
                f"mappers[{i}] must be a LevelMapper instance, got {type(mapper)}"
            )
        
        if len(levels) < 2:
            raise ValueError(
                f"HierarchicalWorldModel requires at least 2 levels, got {len(levels)}"
            )
        if len(mappers) != len(levels) - 1:
            raise ValueError(
                f"Expected {len(levels) - 1} mappers for {len(levels)} levels, "
                f"got {len(mappers)}"
            )

        # Verify mapper connections match the provided levels
        for i, mapper in enumerate(mappers):
            assert mapper.coarse_model is levels[i], (
                f"mappers[{i}].coarse_model must be levels[{i}] (same object reference), "
                f"but got different object"
            )
            assert mapper.fine_model is levels[i + 1], (
                f"mappers[{i}].fine_model must be levels[{i + 1}] (same object reference), "
                f"but got different object"
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

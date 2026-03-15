"""
empo.hierarchical - Hierarchical Empowerment-Based AI Agents

This subpackage provides hierarchical reinforcement learning components
for multi-agent systems using empowerment-based objectives.

**Status:** Work in progress. This subpackage is under active development.

**Core abstractions:**
- HierarchicalWorldModel: A sequence of WorldModels connected by LevelMappers.
- LevelMapper: Abstract base class for mapping between adjacent levels.

**Planned features:**
- Hierarchical option framework for temporal abstraction
- Sub-goal discovery for complex environments
- Integration with LLM-based planning (see src.llm_hierarchical_modeler)

**Requirements:**
    Optional dependencies: ollama, mineland
    Install with: pip install -r setup/requirements-hierarchical.txt
    
    Note: MineLand requires additional setup from https://github.com/cocacola-lab/MineLand

**Related modules:**
- src.llm_hierarchical_modeler: LLM-based Minecraft world generation
- empo.backward_induction: Base planning algorithms
"""

from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.level_mapper import LevelMapper
from empo.hierarchical.cell_partition import CellPartition
from empo.hierarchical.macro_grid_env import MacroGridEnv, MACRO_PASS, macro_walk
from empo.hierarchical.multigrid_level_mapper import MultiGridLevelMapper
from empo.hierarchical.two_level_multigrid import TwoLevelMultigrid

__all__ = [
    "HierarchicalWorldModel",
    "LevelMapper",
    "CellPartition",
    "MacroGridEnv",
    "MACRO_PASS",
    "macro_walk",
    "MultiGridLevelMapper",
    "TwoLevelMultigrid",
]

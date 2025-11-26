"""
llm_hierarchical_modeler - LLM-based hierarchical world modeling

This package provides LLM-based world modeling components for hierarchical
reinforcement learning with empowerment-based objectives.

Requires optional dependencies: ollama, mineland
Install with: pip install -r requirements-hierarchical.txt
Note: MineLand requires additional setup from https://github.com/cocacola-lab/MineLand
"""

from .minecraft_world import (
    create_three_player_world_config,
    get_spawn_coordinates,
    get_player_spawn_info,
    create_mineland_environment,
    generate_world_description,
)

__all__ = [
    "create_three_player_world_config",
    "get_spawn_coordinates",
    "get_player_spawn_info",
    "create_mineland_environment",
    "generate_world_description",
]

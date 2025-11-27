"""
llm_hierarchical_modeler - LLM-based hierarchical world modeling

This package provides LLM-based world modeling components for hierarchical
reinforcement learning with empowerment-based objectives.

Requires optional dependencies: ollama, mineland
Install with: pip install -r requirements-hierarchical.txt
Note: MineLand requires additional setup from https://github.com/cocacola-lab/MineLand
"""

from .minecraft_world import (
    # Core functions
    get_spawn_points,
    generate_world_commands,
    generate_teleport_commands,
    setup_three_player_world,
    # Legacy/compatibility
    create_three_player_world_config,
    get_spawn_coordinates,
    get_player_spawn_info,
    generate_world_description,
)

__all__ = [
    # Core functions
    "get_spawn_points",
    "generate_world_commands",
    "generate_teleport_commands",
    "setup_three_player_world",
    # Legacy/compatibility
    "create_three_player_world_config",
    "get_spawn_coordinates",
    "get_player_spawn_info",
    "generate_world_description",
]

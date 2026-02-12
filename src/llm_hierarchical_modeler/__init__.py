"""
llm_hierarchical_modeler - LLM-based Hierarchical World Modeling for Minecraft.

This package provides LLM-based world modeling components for hierarchical
reinforcement learning experiments in Minecraft using the MineLand framework.

The package enables creating custom Minecraft worlds with specific terrain
features and resource distributions for multi-agent experiments involving
robot-human collaboration.

**Requirements:**
    - Optional dependencies: ollama, mineland
    - Install with: pip install -r setup/requirements/hierarchical.txt
    - MineLand requires additional setup from https://github.com/cocacola-lab/MineLand

World Layout:
    The three-player world consists of:
    - A 200x200 block valley with varied terrain
    - Central north-south river dividing east/west regions
    - Western forest with abundant wood resources
    - Eastern rocky area with abundant stone
    - Northern mountain with cave system and ores
    - Southern plains with farmland
    - Three spawn points: robot (center), human_a (west), human_b (east)

Functions:
    get_spawn_points: Get spawn coordinates for all three players.
    generate_world_commands: Get Minecraft commands to build the terrain.
    generate_teleport_commands: Get commands to teleport players to spawns.
    setup_three_player_world: Apply terrain to a running MineLand server.

Legacy/Compatibility:
    create_three_player_world_config: Get configuration dict for documentation.
    get_spawn_coordinates: Get spawn coordinates as tuples.
    get_player_spawn_info: Get detailed spawn info including resources.
    generate_world_description: Get human-readable world description.

Example usage:
    >>> import mineland
    >>> from src.llm_hierarchical_modeler import setup_three_player_world, get_spawn_points
    >>> 
    >>> # Create MineLand environment
    >>> env = mineland.make(task_id="playground", agents_count=3)
    >>> obs = env.reset()
    >>> 
    >>> # Apply custom world terrain
    >>> setup_three_player_world(env)
    >>> 
    >>> # Get spawn information
    >>> spawns = get_spawn_points()
    >>> print(f"Robot spawns at: ({spawns[0]['x']}, {spawns[0]['y']}, {spawns[0]['z']})")
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

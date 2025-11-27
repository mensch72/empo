"""
Minecraft world generation for multi-agent reinforcement learning.

This module provides functions to create Minecraft worlds with specific
terrain features and resource distributions for multi-agent experiments.

World Layout:
- A large valley (200x200 blocks) with a river dividing east/west
- Asymmetric resource distribution across regions
- Three spawn points for robot and two human players

Functions:
- generate_world_commands(): Returns Minecraft commands to build terrain
- get_spawn_points(): Returns spawn coordinates for all players
- setup_three_player_world(): Applies terrain to a running MineLand server
"""

from typing import Any


# =============================================================================
# World Configuration Constants (single source of truth)
# =============================================================================

# Valley dimensions
VALLEY_SIZE = 200  # Total size of the valley in blocks
RIVER_WIDTH = 10  # Width of the central river

# Y-levels
BASE_Y = 60  # Stone base level
WATER_LEVEL = 62  # Water surface
GRASS_LEVEL = 64  # Grass surface level
SPAWN_Y = 70  # Player spawn Y (above terrain)

# Spawn point coordinates
SPAWN_POINTS = [
    {
        "name": "robot",
        "x": 0,
        "y": SPAWN_Y,
        "z": 0,
        "role": "robot",
        "description": "Center of valley, near river",
    },
    {
        "name": "human_a",
        "x": -60,
        "y": SPAWN_Y,
        "z": 0,
        "role": "human",
        "description": "West side, forest area with abundant wood",
    },
    {
        "name": "human_b",
        "x": 60,
        "y": SPAWN_Y,
        "z": 0,
        "role": "human",
        "description": "East side, rocky area with abundant stone",
    },
]


# =============================================================================
# Core Functions
# =============================================================================


def get_spawn_points() -> list[dict[str, Any]]:
    """
    Get spawn point information for all three players.

    Returns:
        List of spawn point dictionaries with name, coordinates, role, description.

    Example:
        >>> spawns = get_spawn_points()
        >>> print(spawns[0])
        {'name': 'robot', 'x': 0, 'y': 70, 'z': 0, 'role': 'robot', ...}
    """
    return SPAWN_POINTS.copy()


def generate_world_commands() -> list[str]:
    """
    Generate Minecraft commands to build the three-player world terrain.

    Creates terrain with:
    - Valley floor (stone/dirt/grass layers)
    - Central river with sand bed (north-south)
    - Western forest with oak trees
    - Eastern rocky area with stone outcrops
    - Northern mountain with cave system and ores
    - Southern plains with farmland

    Returns:
        List of Minecraft command strings (without leading slash)

    Example:
        >>> commands = generate_world_commands()
        >>> print(len(commands))
        4358
        >>> print(commands[0])
        fill -100 55 -100 100 60 100 minecraft:stone
    """
    commands = []

    half_size = VALLEY_SIZE // 2
    half_river = RIVER_WIDTH // 2

    # =========================================================================
    # 1. Valley floor - stone base, dirt, grass top
    # =========================================================================
    commands.append(
        f"fill {-half_size} {BASE_Y-5} {-half_size} {half_size} {BASE_Y} {half_size} "
        "minecraft:stone"
    )
    commands.append(
        f"fill {-half_size} {BASE_Y+1} {-half_size} {half_size} {GRASS_LEVEL-1} {half_size} "
        "minecraft:dirt"
    )
    commands.append(
        f"fill {-half_size} {GRASS_LEVEL} {-half_size} {half_size} {GRASS_LEVEL} {half_size} "
        "minecraft:grass_block"
    )
    commands.append(
        f"fill {-half_size} {GRASS_LEVEL+1} {-half_size} {half_size} {GRASS_LEVEL+50} {half_size} "
        "minecraft:air"
    )

    # =========================================================================
    # 2. Central river (north-south along x=0)
    # =========================================================================
    commands.append(
        f"fill {-half_river} {WATER_LEVEL-2} {-half_size} {half_river} {GRASS_LEVEL} {half_size} "
        "minecraft:water"
    )
    commands.append(
        f"fill {-half_river} {WATER_LEVEL-3} {-half_size} {half_river} {WATER_LEVEL-3} {half_size} "
        "minecraft:sand"
    )

    # Underwater ores (iron scattered, diamonds rare)
    for z in range(-half_size + 10, half_size - 10, 20):
        commands.append(f"setblock 0 {WATER_LEVEL-2} {z} minecraft:iron_ore")
    commands.append(f"setblock -2 {WATER_LEVEL-3} 0 minecraft:diamond_ore")
    commands.append(f"setblock 2 {WATER_LEVEL-3} 30 minecraft:diamond_ore")

    # =========================================================================
    # 3. Western forest (x < -5) - abundant trees
    # =========================================================================
    for x in range(-half_size + 10, -half_river - 10, 15):
        for z in range(-half_size + 10, half_size - 10, 15):
            # Tree trunk
            for y in range(GRASS_LEVEL + 1, GRASS_LEVEL + 6):
                commands.append(f"setblock {x} {y} {z} minecraft:oak_log")
            # Leaves
            for dy in range(4, 7):
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        if abs(dx) + abs(dz) <= 3:
                            commands.append(
                                f"setblock {x+dx} {GRASS_LEVEL+dy} {z+dz} minecraft:oak_leaves"
                            )
    # Stone outcrop
    commands.append(
        f"fill {-half_size+5} {GRASS_LEVEL} {-20} {-half_size+10} {GRASS_LEVEL+3} {-15} "
        "minecraft:stone"
    )

    # =========================================================================
    # 4. Eastern rocky area (x > 5) - abundant stone
    # =========================================================================
    east_start = half_river + 10
    east_end = half_size - 10
    if east_start < east_end:
        for x in range(east_start, east_end, 12):
            for z in range(-half_size + 10, half_size - 10, 12):
                commands.append(
                    f"fill {x} {GRASS_LEVEL} {z} {x+5} {GRASS_LEVEL+4} {z+5} minecraft:stone"
                )
                commands.append(
                    f"setblock {x+2} {GRASS_LEVEL+1} {z+2} minecraft:coal_ore"
                )

    # Sparse trees in east
    sparse_start = half_river + 20
    sparse_end = half_size - 20
    if sparse_start < sparse_end:
        for x in range(sparse_start, sparse_end, 40):
            for z in range(-40, 41, 40):
                for y in range(GRASS_LEVEL + 1, GRASS_LEVEL + 4):
                    commands.append(f"setblock {x} {y} {z} minecraft:oak_log")

    # =========================================================================
    # 5. Northern mountain with cave (z > 70)
    # =========================================================================
    north_z = half_size - 30
    for level in range(0, 15):
        x_start = -40 + level * 2
        x_end = 40 - level * 2
        if x_start >= x_end:
            break
        commands.append(
            f"fill {x_start} {GRASS_LEVEL+level} {north_z-level} {x_end} {GRASS_LEVEL+level} "
            f"{half_size} minecraft:stone"
        )

    # Cave entrance and tunnel
    commands.append(
        f"fill -5 {GRASS_LEVEL+1} {north_z} 5 {GRASS_LEVEL+5} {north_z+10} minecraft:air"
    )
    commands.append(
        f"fill -3 {GRASS_LEVEL+1} {north_z+10} 3 {GRASS_LEVEL+4} {half_size-5} minecraft:air"
    )

    # Ores in cave
    commands.append(f"setblock 0 {GRASS_LEVEL+2} {north_z+15} minecraft:iron_ore")
    commands.append(f"setblock -2 {GRASS_LEVEL+1} {north_z+20} minecraft:gold_ore")
    commands.append(f"setblock 2 {GRASS_LEVEL+1} {north_z+25} minecraft:diamond_ore")
    commands.append(
        f"fill -10 {GRASS_LEVEL+3} {north_z+5} -8 {GRASS_LEVEL+5} {north_z+8} minecraft:coal_ore"
    )

    # =========================================================================
    # 6. Southern plains with farmland (z < -70)
    # =========================================================================
    for x in range(-30, 31, 20):
        commands.append(
            f"fill {x} {GRASS_LEVEL} {-half_size+10} {x+8} {GRASS_LEVEL} {-half_size+18} "
            "minecraft:farmland"
        )
        commands.append(
            f"fill {x} {GRASS_LEVEL+1} {-half_size+10} {x+8} {GRASS_LEVEL+1} {-half_size+18} "
            "minecraft:wheat"
        )

    # Flowers
    commands.append(f"setblock -20 {GRASS_LEVEL+1} {-half_size+25} minecraft:dandelion")
    commands.append(f"setblock -15 {GRASS_LEVEL+1} {-half_size+25} minecraft:poppy")
    commands.append(f"setblock 15 {GRASS_LEVEL+1} {-half_size+28} minecraft:dandelion")

    return commands


def generate_teleport_commands() -> list[str]:
    """
    Generate commands to teleport players to their spawn points.

    Returns:
        List of teleport command strings

    Example:
        >>> commands = generate_teleport_commands()
        >>> print(commands[0])
        tp robot 0 70 0
    """
    return [f"tp {sp['name']} {sp['x']} {sp['y']} {sp['z']}" for sp in SPAWN_POINTS]


def setup_three_player_world(env: Any) -> None:
    """
    Set up the three-player world terrain in a running MineLand environment.

    Executes world generation commands on the MineLand server to build
    the terrain after env.reset() has been called.

    Args:
        env: A MineLand environment object

    Raises:
        RuntimeError: If the environment doesn't have a server manager

    Example:
        >>> env = mineland.make(task_id="playground", agents_count=3)
        >>> obs = env.reset()
        >>> setup_three_player_world(env)
    """
    import time

    # Access server manager
    server_manager = None
    if hasattr(env, "env") and hasattr(env.env, "server_manager"):
        server_manager = env.env.server_manager
    elif hasattr(env, "server_manager"):
        server_manager = env.server_manager

    if server_manager is None:
        raise RuntimeError(
            "Cannot access MineLand server manager. "
            "Make sure you're using a local MineLand server."
        )

    print("Building custom world terrain...")
    for cmd in generate_world_commands():
        server_manager.execute(cmd)
        time.sleep(0.01)

    time.sleep(2)

    print("Teleporting players to spawn points...")
    for cmd in generate_teleport_commands():
        server_manager.execute(cmd)

    time.sleep(1)
    print("World setup complete!")


# =============================================================================
# Legacy/Compatibility Functions (for backward compatibility)
# =============================================================================


def create_three_player_world_config() -> dict[str, Any]:
    """
    Create a configuration dict describing the world (for documentation/metadata).

    Note: This is primarily for documentation. The actual world is built
    using generate_world_commands() which uses module constants directly.

    Returns:
        dict with spawn_points, world_settings, resource_distribution
    """
    return {
        "spawn_points": get_spawn_points(),
        "world_settings": {
            "name": "three_player_valley",
            "valley_size": VALLEY_SIZE,
            "river_width": RIVER_WIDTH,
        },
        "resource_distribution": {
            "west": {"wood": "abundant", "stone": "some"},
            "east": {"stone": "abundant", "wood": "some"},
            "center": {"iron": "rare", "diamonds": "rare"},
            "north": {"coal": "common", "iron": "moderate", "gold": "rare"},
            "south": {"food": "abundant"},
        },
    }


def get_spawn_coordinates() -> list[tuple[int, int, int]]:
    """Get spawn coordinates as tuples."""
    return [(sp["x"], sp["y"], sp["z"]) for sp in SPAWN_POINTS]


def get_player_spawn_info() -> list[dict[str, Any]]:
    """Get detailed spawn info including nearby resources."""
    resources = {
        "robot": {"iron": "rare", "diamonds": "rare"},
        "human_a": {"wood": "abundant", "stone": "some"},
        "human_b": {"stone": "abundant", "wood": "some"},
    }
    return [
        {
            "name": sp["name"],
            "role": sp["role"],
            "coordinates": (sp["x"], sp["y"], sp["z"]),
            "description": sp["description"],
            "nearby_resources": resources.get(sp["name"], {}),
        }
        for sp in SPAWN_POINTS
    ]


def generate_world_description() -> str:
    """Generate a human-readable description of the world."""
    lines = [
        "Three-Player Valley World",
        "==========================",
        "",
        "Terrain Features:",
        f"  - Valley size: {VALLEY_SIZE} blocks",
        f"  - Central river: {RIVER_WIDTH} blocks wide",
        "  - North: Mountain with cave system",
        "  - South: Flat plains with farmland",
        "  - West: Forest (abundant wood)",
        "  - East: Rocky terrain (abundant stone)",
        "",
        "Spawn Points:",
    ]
    for sp in SPAWN_POINTS:
        lines.append(
            f"  - {sp['name'].title()}: ({sp['x']}, {sp['y']}, {sp['z']}) - {sp['description']}"
        )
    return "\n".join(lines)

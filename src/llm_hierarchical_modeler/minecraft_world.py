"""
Minecraft world generation for multi-agent reinforcement learning.

This module provides functions to create Minecraft worlds with specific
terrain features and resource distributions for multi-agent experiments.

The main function `create_three_player_world()` generates a world with:
- A large valley with a river dividing it
- Asymmetric resource distribution across regions
- Three spawn points for a robot and two human players
"""

from typing import Any


# World region constants
VALLEY_SIZE = 200  # Total size of the valley in blocks
RIVER_WIDTH = 10  # Width of the central river
MOUNTAIN_HEIGHT = 80  # Height of the northern mountain
CAVE_DEPTH = 30  # Depth of the cave system

# Spawn point offsets from world center
SPAWN_OFFSET_WEST = -60  # X offset for Human A
SPAWN_OFFSET_EAST = 60  # X offset for Human B
SPAWN_Y = 64  # Y level for spawning (sea level)


def create_three_player_world_config() -> dict[str, Any]:
    """
    Create a MineLand world configuration for three players.

    This function generates the configuration needed to create a Minecraft
    world with the following features:

    Terrain:
    - Large valley with river dividing it (east-west orientation)
    - Northern mountain with cave system
    - Southern flat plains

    Resources:
    - West side: abundant wood (forest biome), some stone
    - East side: abundant stone (rocky terrain), some wood
    - Center (river): rare resources (iron, diamonds) underwater
    - North: mountain caves with ore deposits
    - South: flat plains (good for building)

    Spawn Points:
    - Robot: center (near river, coordinates ~0, 64, 0)
    - Human A: west side (forest area, coordinates ~-60, 64, 0)
    - Human B: east side (rocky area, coordinates ~60, 64, 0)

    Returns:
        dict: Configuration dictionary for MineLand world creation with:
            - world_settings: Terrain and biome configuration
            - spawn_points: List of spawn point coordinates for each player
            - resource_distribution: Configuration for resource placement
            - player_roles: Role assignments (robot, human_a, human_b)

    Example:
        >>> config = create_three_player_world_config()
        >>> print(config["spawn_points"])
        [{"name": "robot", "x": 0, "y": 64, "z": 0}, ...]
    """
    return {
        "world_settings": {
            "name": "three_player_valley",
            "world_type": "custom",
            "seed": 12345,  # Fixed seed for reproducibility
            "valley": {
                "size": VALLEY_SIZE,
                "orientation": "east_west",
            },
            "river": {
                "position": "center",
                "width": RIVER_WIDTH,
                "direction": "north_south",
            },
            "terrain": {
                "north": {
                    "type": "mountain",
                    "height": MOUNTAIN_HEIGHT,
                    "features": ["cave_system"],
                    "cave_depth": CAVE_DEPTH,
                },
                "south": {
                    "type": "plains",
                    "height": "flat",
                    "features": ["grass", "flowers"],
                },
                "west": {
                    "type": "forest",
                    "biome": "forest",
                    "tree_density": "high",
                },
                "east": {
                    "type": "rocky",
                    "biome": "extreme_hills",
                    "rock_density": "high",
                },
            },
        },
        "spawn_points": [
            {
                "name": "robot",
                "role": "robot",
                "x": 0,
                "y": SPAWN_Y,
                "z": 0,
                "description": "Center of valley, near river",
            },
            {
                "name": "human_a",
                "role": "human",
                "x": SPAWN_OFFSET_WEST,
                "y": SPAWN_Y,
                "z": 0,
                "description": "West side, forest area with abundant wood",
            },
            {
                "name": "human_b",
                "role": "human",
                "x": SPAWN_OFFSET_EAST,
                "y": SPAWN_Y,
                "z": 0,
                "description": "East side, rocky area with abundant stone",
            },
        ],
        "resource_distribution": {
            "west": {
                "wood": "abundant",
                "stone": "some",
                "description": "Forest biome with many trees",
            },
            "east": {
                "stone": "abundant",
                "wood": "some",
                "description": "Rocky terrain with exposed stone",
            },
            "center": {
                "iron": "rare",
                "diamonds": "rare",
                "location": "underwater",
                "description": "River with rare resources underwater",
            },
            "north": {
                "coal": "common",
                "iron": "moderate",
                "gold": "rare",
                "diamonds": "very_rare",
                "description": "Mountain caves with ore deposits",
            },
            "south": {
                "food": "abundant",
                "description": "Flat plains good for farming and building",
            },
        },
        "player_roles": {
            "robot": {
                "agent_index": 0,
                "type": "robot",
                "spawn_location": "center",
            },
            "human_a": {
                "agent_index": 1,
                "type": "human",
                "spawn_location": "west",
            },
            "human_b": {
                "agent_index": 2,
                "type": "human",
                "spawn_location": "east",
            },
        },
    }


def get_spawn_coordinates(config: dict[str, Any]) -> list[tuple[int, int, int]]:
    """
    Extract spawn point coordinates from a world configuration.

    Args:
        config: World configuration dictionary from create_three_player_world_config()

    Returns:
        List of (x, y, z) tuples for each spawn point in order:
        [robot, human_a, human_b]

    Example:
        >>> config = create_three_player_world_config()
        >>> coords = get_spawn_coordinates(config)
        >>> print(coords[0])  # Robot spawn
        (0, 64, 0)
    """
    spawn_points = config.get("spawn_points", [])
    return [(sp["x"], sp["y"], sp["z"]) for sp in spawn_points]


def get_player_spawn_info(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Get detailed spawn information for each player.

    Args:
        config: World configuration dictionary from create_three_player_world_config()

    Returns:
        List of dictionaries with spawn info for each player, containing:
        - name: Player identifier (robot, human_a, human_b)
        - role: Player role (robot or human)
        - coordinates: (x, y, z) tuple
        - description: Description of the spawn location
        - nearby_resources: Dict of resources available nearby

    Example:
        >>> config = create_three_player_world_config()
        >>> info = get_player_spawn_info(config)
        >>> print(info[1]["nearby_resources"]["wood"])
        'abundant'
    """
    spawn_points = config.get("spawn_points", [])
    resource_dist = config.get("resource_distribution", {})

    result = []
    for sp in spawn_points:
        name = sp["name"]

        # Determine nearby resources based on spawn location
        if name == "robot":
            # Robot is at center, has access to river resources
            nearby = resource_dist.get("center", {}).copy()
        elif name == "human_a":
            # Human A is in the west (forest)
            nearby = resource_dist.get("west", {}).copy()
        elif name == "human_b":
            # Human B is in the east (rocky)
            nearby = resource_dist.get("east", {}).copy()
        else:
            nearby = {}

        result.append(
            {
                "name": name,
                "role": sp["role"],
                "coordinates": (sp["x"], sp["y"], sp["z"]),
                "description": sp["description"],
                "nearby_resources": nearby,
            }
        )

    return result


def create_mineland_environment(config: dict[str, Any] | None = None) -> Any:
    """
    Create a MineLand environment with the specified world configuration.

    This function wraps mineland.make() with the appropriate settings
    for a three-player world. It requires MineLand to be installed.

    Args:
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Returns:
        MineLand environment object ready for use

    Raises:
        ImportError: If MineLand is not installed
        RuntimeError: If environment creation fails

    Example:
        >>> config = create_three_player_world_config()
        >>> env = create_mineland_environment(config)
        >>> obs = env.reset()
    """
    try:
        import mineland
    except ImportError as e:
        raise ImportError(
            "MineLand is required for environment creation. "
            "Install with: pip install -r requirements-hierarchical.txt"
        ) from e

    if config is None:
        config = create_three_player_world_config()

    spawn_points = config.get("spawn_points", [])
    agents_count = len(spawn_points)

    # Create MineLand environment
    # Note: MineLand uses its own world generation - this configuration
    # is primarily for documentation and spawn point management.
    # The actual world features depend on MineLand's task configuration.
    env = mineland.make(
        task_id="playground",
        agents_count=agents_count,
        headless=False,  # Required for RGB capture
        image_size=(180, 320),  # (height, width)
    )

    return env


def generate_world_description(config: dict[str, Any] | None = None) -> str:
    """
    Generate a human-readable description of the world configuration.

    Args:
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Returns:
        A formatted string describing the world layout and features.

    Example:
        >>> config = create_three_player_world_config()
        >>> print(generate_world_description(config))
        Three-Player Valley World
        ==========================
        ...
    """
    if config is None:
        config = create_three_player_world_config()

    world = config.get("world_settings", {})
    spawn_points = config.get("spawn_points", [])
    resources = config.get("resource_distribution", {})

    lines = [
        "Three-Player Valley World",
        "==========================",
        "",
        "Terrain Features:",
        f"  - Valley size: {world.get('valley', {}).get('size', 'N/A')} blocks",
        f"  - Central river: {world.get('river', {}).get('width', 'N/A')} blocks wide",
        "  - North: Mountain with cave system",
        "  - South: Flat plains",
        "  - West: Forest (abundant wood)",
        "  - East: Rocky terrain (abundant stone)",
        "",
        "Spawn Points:",
    ]

    for sp in spawn_points:
        lines.append(
            f"  - {sp['name'].title()}: ({sp['x']}, {sp['y']}, {sp['z']}) "
            f"- {sp['description']}"
        )

    lines.extend(
        [
            "",
            "Resource Distribution:",
        ]
    )

    for region, res in resources.items():
        desc = res.get("description", "")
        lines.append(f"  - {region.title()}: {desc}")

    return "\n".join(lines)

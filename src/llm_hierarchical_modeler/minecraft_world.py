"""
Minecraft world generation for multi-agent reinforcement learning.

This module provides functions to create Minecraft worlds with specific
terrain features and resource distributions for multi-agent experiments.

The main function `create_three_player_world_config()` generates a configuration
for a world with:
- A large valley with a river dividing it
- Asymmetric resource distribution across regions
- Three spawn points for a robot and two human players

The `generate_world_commands()` function returns Minecraft commands to build the
actual terrain, and `setup_three_player_world()` applies these commands to a
running MineLand environment.
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
SPAWN_Y = 70  # Y level for spawning (above terrain)


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
    for a three-player world and sets up the custom terrain using
    Minecraft commands.

    Args:
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Returns:
        MineLand environment object ready for use with custom world built

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

    # Create agent configs with custom names matching our player roles
    agents_config = []
    for sp in spawn_points:
        agents_config.append({"name": sp["name"]})

    # Create MineLand environment
    env = mineland.make(
        task_id="playground",
        agents_count=agents_count,
        agents_config=agents_config,
        headless=True,  # Use headless=True to avoid WebGL issues with multiple agents
        image_size=(180, 320),  # (height, width)
    )

    return env


def generate_world_commands(config: dict[str, Any] | None = None) -> list[str]:
    """
    Generate Minecraft commands to build the three-player world terrain.

    This function returns a list of Minecraft commands that, when executed
    on a MineLand server, will create the specified terrain features:
    - Valley base with grass floor
    - Central river running north-south
    - Western forest area with trees
    - Eastern rocky area with stone outcrops
    - Northern mountain with caves
    - Southern flat plains
    - Resources (ores) placed according to the distribution spec

    Args:
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Returns:
        List of Minecraft command strings (without leading slash)

    Example:
        >>> commands = generate_world_commands()
        >>> for cmd in commands[:5]:
        ...     print(cmd)
        fill -100 60 -100 100 63 100 minecraft:grass_block
        ...
    """
    if config is None:
        config = create_three_player_world_config()

    commands = []

    # Get configuration values
    valley_size = config["world_settings"]["valley"]["size"]
    river_width = config["world_settings"]["river"]["width"]
    half_size = valley_size // 2
    half_river = river_width // 2

    # Base Y levels
    base_y = 60  # Ground level
    water_level = 62  # Water surface
    grass_level = 64  # Grass surface

    # =========================================================================
    # 1. Create the valley floor - fill with stone and top with grass
    # =========================================================================
    # Stone base layer
    commands.append(
        f"fill {-half_size} {base_y-5} {-half_size} {half_size} {base_y} {half_size} "
        "minecraft:stone"
    )
    # Dirt layer
    commands.append(
        f"fill {-half_size} {base_y+1} {-half_size} {half_size} {grass_level-1} {half_size} "
        "minecraft:dirt"
    )
    # Grass top layer
    commands.append(
        f"fill {-half_size} {grass_level} {-half_size} {half_size} {grass_level} {half_size} "
        "minecraft:grass_block"
    )
    # Clear air above
    commands.append(
        f"fill {-half_size} {grass_level+1} {-half_size} {half_size} {grass_level+50} {half_size} "
        "minecraft:air"
    )

    # =========================================================================
    # 2. Create the central river (running north-south along x=0)
    # =========================================================================
    # Dig river channel
    commands.append(
        f"fill {-half_river} {water_level-2} {-half_size} {half_river} {grass_level} {half_size} "
        "minecraft:water"
    )
    # River bed with sand
    commands.append(
        f"fill {-half_river} {water_level-3} {-half_size} {half_river} {water_level-3} {half_size} "
        "minecraft:sand"
    )

    # =========================================================================
    # 3. Place rare underwater resources in the river
    # =========================================================================
    # Iron ore underwater (scattered)
    for z in range(-half_size + 10, half_size - 10, 20):
        commands.append(f"setblock 0 {water_level-2} {z} minecraft:iron_ore")
    # Diamond ore underwater (rare, only a few)
    commands.append(f"setblock -2 {water_level-3} 0 minecraft:diamond_ore")
    commands.append(f"setblock 2 {water_level-3} 30 minecraft:diamond_ore")

    # =========================================================================
    # 4. Western forest area - place trees and wood
    # =========================================================================
    # West side is from -half_size to -half_river (x < -5)
    # Place oak trees in a grid pattern
    for x in range(-half_size + 10, -half_river - 10, 15):
        for z in range(-half_size + 10, half_size - 10, 15):
            # Tree trunk (5 blocks high)
            for y in range(grass_level + 1, grass_level + 6):
                commands.append(f"setblock {x} {y} {z} minecraft:oak_log")
            # Leaves (simple ball shape)
            for dy in range(4, 7):
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        if abs(dx) + abs(dz) <= 3:  # Diamond-ish shape
                            commands.append(
                                f"setblock {x+dx} {grass_level+dy} {z+dz} "
                                "minecraft:oak_leaves"
                            )
    # Some stone outcrops in the west
    commands.append(
        f"fill {-half_size+5} {grass_level} {-20} {-half_size+10} {grass_level+3} {-15} "
        "minecraft:stone"
    )

    # =========================================================================
    # 5. Eastern rocky area - stone outcrops and exposed stone
    # =========================================================================
    # East side is from half_river to half_size (x > 5)
    # Replace grass with stone in patches
    # Validate ranges to avoid empty loops
    east_start = half_river + 10
    east_end = half_size - 10
    if east_start < east_end:
        for x in range(east_start, east_end, 12):
            for z in range(-half_size + 10, half_size - 10, 12):
                # Stone outcrop
                commands.append(
                    f"fill {x} {grass_level} {z} {x+5} {grass_level+4} {z+5} minecraft:stone"
                )
                # Coal ore in some outcrops
                commands.append(
                    f"setblock {x+2} {grass_level+1} {z+2} minecraft:coal_ore"
                )
    # A few trees in the east (sparse)
    sparse_tree_start = half_river + 20
    sparse_tree_end = half_size - 20
    if sparse_tree_start < sparse_tree_end:
        for x in range(sparse_tree_start, sparse_tree_end, 40):
            for z in range(-40, 41, 40):
                # Small tree trunk
                for y in range(grass_level + 1, grass_level + 4):
                    commands.append(f"setblock {x} {y} {z} minecraft:oak_log")

    # =========================================================================
    # 6. Northern mountain with cave system
    # =========================================================================
    north_z = half_size - 30  # Mountain starts at z = 70
    # Build mountain (stepped pyramid shape)
    for level in range(0, 15):
        z_start = north_z - level
        z_end = half_size
        x_start = -40 + level * 2
        x_end = 40 - level * 2
        # Skip if the range is invalid (x_start > x_end)
        if x_start >= x_end:
            break
        y = grass_level + level
        commands.append(
            f"fill {x_start} {y} {z_start} {x_end} {y} {z_end} minecraft:stone"
        )
    # Cave entrance
    commands.append(
        f"fill -5 {grass_level+1} {north_z} 5 {grass_level+5} {north_z+10} minecraft:air"
    )
    # Cave tunnel
    commands.append(
        f"fill -3 {grass_level+1} {north_z+10} 3 {grass_level+4} {half_size-5} minecraft:air"
    )
    # Ore deposits in mountain/cave
    commands.append(f"setblock 0 {grass_level+2} {north_z+15} minecraft:iron_ore")
    commands.append(f"setblock -2 {grass_level+1} {north_z+20} minecraft:gold_ore")
    commands.append(f"setblock 2 {grass_level+1} {north_z+25} minecraft:diamond_ore")
    # Coal seam
    commands.append(
        f"fill -10 {grass_level+3} {north_z+5} -8 {grass_level+5} {north_z+8} minecraft:coal_ore"
    )

    # =========================================================================
    # 7. Southern flat plains - farming area with wheat/flowers
    # =========================================================================
    # Farmland patches (in south region: z = -100 to z = -70)
    for x in range(-30, 31, 20):
        commands.append(
            f"fill {x} {grass_level} {-half_size+10} {x+8} {grass_level} {-half_size+18} "
            "minecraft:farmland"
        )
        # Wheat on farmland
        commands.append(
            f"fill {x} {grass_level+1} {-half_size+10} {x+8} {grass_level+1} {-half_size+18} "
            "minecraft:wheat"
        )
    # Flowers
    commands.append(f"setblock -20 {grass_level+1} {-half_size+25} minecraft:dandelion")
    commands.append(f"setblock -15 {grass_level+1} {-half_size+25} minecraft:poppy")
    commands.append(f"setblock 15 {grass_level+1} {-half_size+28} minecraft:dandelion")

    return commands


def generate_teleport_commands(config: dict[str, Any] | None = None) -> list[str]:
    """
    Generate commands to teleport players to their spawn points.

    Args:
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Returns:
        List of teleport command strings

    Example:
        >>> commands = generate_teleport_commands()
        >>> print(commands[0])
        tp robot 0 70 0
    """
    if config is None:
        config = create_three_player_world_config()

    commands = []
    for sp in config["spawn_points"]:
        commands.append(f"tp {sp['name']} {sp['x']} {sp['y']} {sp['z']}")

    return commands


def setup_three_player_world(env: Any, config: dict[str, Any] | None = None) -> None:
    """
    Set up the three-player world terrain in a running MineLand environment.

    This function executes the world generation commands on the MineLand
    server to build the terrain after the environment has been reset.

    IMPORTANT: Call this after env.reset() but before starting gameplay.

    Args:
        env: A MineLand environment object (from create_mineland_environment)
        config: Optional world configuration. If not provided, uses
                the default three-player world configuration.

    Raises:
        RuntimeError: If the environment doesn't have a server manager

    Example:
        >>> config = create_three_player_world_config()
        >>> env = create_mineland_environment(config)
        >>> obs = env.reset()
        >>> setup_three_player_world(env, config)
        >>> # Now the world is ready with custom terrain
    """
    import time

    if config is None:
        config = create_three_player_world_config()

    # Access the MineLand's internal server manager
    server_manager = None
    if hasattr(env, "env") and hasattr(env.env, "server_manager"):
        server_manager = env.env.server_manager
    elif hasattr(env, "server_manager"):
        server_manager = env.server_manager

    if server_manager is None:
        raise RuntimeError(
            "Cannot access MineLand server manager. "
            "Make sure you're using a local MineLand server (not connecting to remote)."
        )

    print("Building custom world terrain...")

    # Generate and execute world building commands
    world_commands = generate_world_commands(config)
    for cmd in world_commands:
        server_manager.execute(cmd)
        # Small delay to avoid overwhelming the server
        time.sleep(0.01)

    # Wait for world to be built
    time.sleep(2)

    # Teleport players to spawn points
    print("Teleporting players to spawn points...")
    teleport_commands = generate_teleport_commands(config)
    for cmd in teleport_commands:
        server_manager.execute(cmd)

    # Wait for teleport to complete
    time.sleep(1)

    print("World setup complete!")


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

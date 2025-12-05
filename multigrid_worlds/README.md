# Multigrid Worlds

This folder contains YAML configuration files for pre-defined multigrid world environments. Each YAML file specifies a complete environment configuration including the map layout, parameters, and metadata.

## Folder Structure

- **basic/** - Simple environments for testing and learning
- **puzzles/** - Multi-chamber and puzzle-like environments
- **obstacles/** - Environments with obstacles like magic walls, doors, and unsteady ground
- **switches_buttons/** - Environments featuring control buttons and switches
- **copilot_challenges/** - Simple 8x8 worlds (6x6 interior) for empowerment research

## YAML Configuration Format

Each YAML file contains:

```yaml
# Comment describing the world
metadata:
  name: "Human-readable name"
  description: "Description of the environment"
  author: "Creator name"
  version: "1.0.0"
  created: "ISO timestamp"
  category: "folder category"
  tags: ["tag1", "tag2"]

map: |
  We We We We We
  We .. .. .. We
  We .. Ar .. We
  We .. .. .. We
  We We We We We

max_steps: 100
orientations: ["e", "n", "s", "w"]
can_push_rocks: "e"
partial_obs: true
see_through_walls: false
```

### Map Format

Maps use two-character codes separated by spaces. The `|` syntax in YAML preserves newlines for readable multi-line maps:

- `We` - Wall
- `..` - Empty cell
- `Ar`, `Ag`, `Ab`, `Ap`, `Ay`, `Ae` - Agent (red, green, blue, purple, yellow, grey)
- `Bl` - Block (pushable)
- `Ro` - Rock (pushable by certain agents)
- `La` - Lava
- `Un` - Unsteady ground
- `Gr`, `Gg`, `Gb`, etc. - Goal (with color)
- `Kr`, `Kg`, etc. - Key (with color)
- `Br`, `Bg`, etc. - Ball (with color)
- `Xr`, `Xg`, etc. - Box (with color)
- `Lr`, `Cr`, `Or` - Door (locked, closed, open - with color)
- `Mn`, `Ms`, `Mw`, `Me` - Magic wall (north, south, west, east entry)
- `CB` - Control button
- `Sw` - Switch

### Orientation Codes

- `e` - East (right)
- `n` - North (up)
- `s` - South (down)
- `w` - West (left)

## Usage

Load an environment from a config file:

```python
from gym_multigrid.multigrid import MultiGridEnv

# Load environment from YAML config file
env = MultiGridEnv(config_file='multigrid_worlds/basic/empty_5x5.yaml')

# Parameters in __init__ override config file values
env = MultiGridEnv(config_file='multigrid_worlds/puzzles/one_or_three_chambers.yaml', max_steps=500)

# JSON files are also supported
env = MultiGridEnv(config_file='some_world.json')
```

## Available Worlds

### Basic
- `empty_5x5.yaml` - Simple 5x5 empty grid with one agent
- `single_agent_7x7.yaml` - 7x7 grid with unsteady ground
- `two_agents.yaml` - Two agents in a simple grid
- `all_agent_colors.yaml` - All six agent colors
- `all_object_types.yaml` - Reference grid with all object types

### Puzzles
- `one_or_three_chambers.yaml` - Full multi-chamber environment
- `small_one_or_three_chambers.yaml` - Smaller version for DAG computation
- `block_rock_test.yaml` - Block and rock pushing test

### Obstacles
- `obstacle_grid_9x9.yaml` - Grid with various obstacles
- `magic_wall_test.yaml` - Magic walls in all directions
- `door_test.yaml` - All door types
- `unsteady_ground_test.yaml` - Unsteady ground cells

### Switches & Buttons
- `control_button_prequel.yaml` - Control button demo (initial state)
- `control_button_ready.yaml` - Control button demo (ready state)

### Copilot Challenges
Simple worlds for empowerment research where the robot wants to help humans without knowing their goals:

**Single Human (8x8 / 6x6 interior):**
- `rock_gateway.yaml` - Rock blocking a passage
- `block_bridge.yaml` - Block that can be positioned to help
- `key_bearer.yaml` - Locked door with key
- `ball_mover.yaml` - Ball in a corridor
- `switch_operator.yaml` - Toggle switch
- `double_rock_maze.yaml` - Two rocks creating a maze

**Two Humans (7x7 / 5x5 interior):**
These focus on how the robot distributes empowerment between two humans:
- `split_access.yaml` - Robot controls access via rock position
- `key_choice.yaml` - Robot chooses which door to unlock
- `block_favor.yaml` - Block positioning favors one human
- `corridor_control.yaml` - Robot in corridor can let one human pass
- `dual_gateway.yaml` - Two rocks for fair/unfair distribution
- `asymmetric_start.yaml` - Humans start with different freedom levels

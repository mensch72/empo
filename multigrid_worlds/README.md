# Multigrid Worlds

This folder contains JSON configuration files for pre-defined multigrid world environments. Each JSON file specifies a complete environment configuration including the map layout, parameters, and metadata.

## Folder Structure

- **basic/** - Simple environments for testing and learning
- **puzzles/** - Multi-chamber and puzzle-like environments
- **obstacles/** - Environments with obstacles like magic walls, doors, and unsteady ground
- **switches_buttons/** - Environments featuring control buttons and switches

## JSON Configuration Format

Each JSON file contains:

```json
{
    "metadata": {
        "name": "Human-readable name",
        "description": "Description of the environment",
        "author": "Creator name",
        "version": "1.0.0",
        "created": "ISO timestamp",
        "category": "folder category",
        "tags": ["tag1", "tag2"]
    },
    "map": "ASCII map string with two-character codes",
    "max_steps": 100,
    "orientations": ["e", "n", "s", "w"],
    "can_push_rocks": "e",
    "partial_obs": true,
    "see_through_walls": false
}
```

### Map Format

Maps use two-character codes separated by spaces:
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

Load an environment from a JSON config file:

```python
from gym_multigrid.multigrid import MultiGridEnv

# Load environment from config file
env = MultiGridEnv(config_file='multigrid_worlds/basic/empty_5x5.json')

# Parameters in __init__ override config file values
env = MultiGridEnv(config_file='multigrid_worlds/puzzles/one_or_three_chambers.json', max_steps=500)
```

## Available Worlds

### Basic
- `empty_5x5.json` - Simple 5x5 empty grid with one agent
- `single_agent_7x7.json` - 7x7 grid with unsteady ground
- `two_agents.json` - Two agents in a simple grid
- `all_agent_colors.json` - All six agent colors
- `all_object_types.json` - Reference grid with all object types

### Puzzles
- `one_or_three_chambers.json` - Full multi-chamber environment
- `small_one_or_three_chambers.json` - Smaller version for DAG computation
- `block_rock_test.json` - Block and rock pushing test

### Obstacles
- `obstacle_grid_9x9.json` - Grid with various obstacles
- `magic_wall_test.json` - Magic walls in all directions
- `door_test.json` - All door types
- `unsteady_ground_test.json` - Unsteady ground cells

### Switches & Buttons
- `control_button_prequel.json` - Control button demo (initial state)
- `control_button_ready.json` - Control button demo (ready state)

# Encoder Architecture for Neural Policy Priors

This document explains the architecture of the neural network encoders used in the `NeuralHumanPolicyPrior` system, including the rationale for certain design decisions and the complementary roles of grid-based and list-based encodings.

## Overview

The neural policy prior system uses four main encoders to transform environment state into features suitable for Q-value prediction:

1. **StateEncoder** - Encodes the grid-based world state (objects, agent distributions by color)
2. **AgentEncoder** - Encodes individual agent features (query agent + per-color agent lists)
3. **GoalEncoder** - Encodes goal locations
4. **InteractiveObjectEncoder** - Encodes complex interactive objects (buttons, switches)

These features are combined by the `QNetwork` to predict Q-values for each action.

## Design Principle: No Normalization

**All values are passed as raw integers/floats, NOT normalized.** This is intentional because:

1. **Absolute distances matter**: The number of steps to reach a goal depends on absolute grid distance
2. **Scale information is useful**: The network can learn that larger grids require different strategies
3. **Simplicity**: No need for denormalization or scaling factors during inference

## Design Principle: Grid vs. List Encoding

The architecture uses two complementary encoding strategies:

- **Grid-based encoding (StateEncoder)**: Captures the *spatial distribution* of objects and agents. Good for objects with simple state (walls, lava, goals) and agent positions by color.

- **List-based encoding (AgentEncoder, InteractiveObjectEncoder)**: Captures *detailed features* of individual entities. Used for agents (with abilities, carried objects, status) and complex interactive objects (with multiple state variables).

This separation enables:
1. Policy transfer across different entity counts
2. Rich feature encoding without exponential channel growth
3. Efficient CNN processing for spatial information

## StateEncoder Channel Structure

The StateEncoder uses a CNN to process a multi-channel grid representation:

```
Total Channels = num_object_types + 3 + num_colors + 1
                 ---------------   -   -----------   -
                       |           |       |         |
                       |           |       |         +-- Query agent channel
                       |           |       +-- Per-color agent channels
                       |           +-- "Other objects" channels (3 types)
                       +-- Object type channels (29 total)
```

### Object Type Channels (29 channels)

| Channel Range | Object Type | Value Encoding |
|---------------|-------------|----------------|
| 0-13 | Base objects (wall, ball, box, goal, lava, block, rock, etc.) | 1.0 = present |
| 14-20 | Per-color doors (7 colors) | 0=none, 0.33=open, 0.67=closed, 1.0=locked |
| 21-27 | Per-color keys (7 colors) | 1.0 = present |
| 28 | Magic walls | 0=none, 0.2-0.8=active (by magic_side), 1.0=inactive |

### Additional Input Features

- **Remaining time**: Raw integer (max_steps - step_count), NOT normalized
- **Global world features** (4 values):
  - Stumble probability (for unsteady ground)
  - Magic wall entry probability
  - Magic wall solidify probability
  - Reserved for future use

## AgentEncoder Features

Each agent is encoded with 13 features (all raw, not normalized):

| Feature | Size | Description |
|---------|------|-------------|
| Position | 2 | Raw (x, y) coordinates |
| Direction | 4 | One-hot encoding (right, down, left, up) |
| Abilities | 2 | can_enter_magic_walls (0/1), can_push_rocks (0/1) |
| Carried object | 2 | (type_index, color_index), 0 if none |
| Status | 3 | paused (0/1), terminated (0/1), forced_action (-1 if none) |

### Encoding Structure

```
AgentEncoder input = [query_agent_features] + [per_color_agent_lists]

Query agent: 13 features
Per-color list: num_agents_per_color[color] × 13 features each
```

The query agent is always first, enabling policy transfer regardless of agent count.

## InteractiveObjectEncoder Features

Complex interactive objects are encoded in lists with configurable maximum counts. All values are raw (not normalized).

### KillButton (5 features each)
- position (2): raw x, y coordinates
- enabled (1): 0 or 1
- trigger_color (1): color index (integer)
- target_color (1): color index (integer)

### PauseSwitch (6 features each)
- position (2): raw x, y coordinates
- enabled (1): 0 or 1
- is_on (1): 0 or 1
- toggle_color (1): color index (integer)
- target_color (1): color index (integer)

### DisablingSwitch (6 features each)
- position (2): raw x, y coordinates
- enabled (1): 0 or 1
- is_on (1): 0 or 1
- toggle_color (1): color index (integer)
- target_type (1): object type index (integer)

### ControlButton (6 features each)
- position (2): raw x, y coordinates
- enabled (1): 0 or 1
- trigger_color (1): color index (integer)
- target_color (1): color index (integer)
- forced_action (1): action index (integer)

## GoalEncoder Features

Goals are encoded with raw coordinates:
- goal_coords (4): [x1, y1, x2, y2] for rectangular goals, or [x, y, x, y] for point goals

## Policy Transfer Capabilities

This architecture enables several transfer scenarios:

1. **More agents per color**: Extra agents visible in grid, up to max in list encoder
2. **More interactive objects**: Up to configurable maximum per type
3. **Different action spaces**: Action mapping with fallback for unknown actions
4. **New object types**: Encoded in "other objects" fallback channels
5. **Different agent abilities**: Abilities encoded per-agent, not assumed global

## Why This Redundancy?

The query agent appears in multiple places:
1. In the query agent grid channel (StateEncoder)
2. As the first element in AgentEncoder
3. In its color's per-color grid channel

This redundancy serves different purposes:

| Encoding | Purpose |
|----------|---------|
| Per-color grid | Agent distribution for spatial reasoning |
| Query agent grid | Mark which position needs a decision |
| Query agent features (first) | Full feature access regardless of config |

## Example: Full Feature Encoding

For an environment with:
- 10x10 grid
- 2 humans (yellow), 1 robot (grey)
- 2 control buttons
- Query for human at index 1

```python
# StateEncoder
grid_tensor: (29 + 3 + 2 + 1, 10, 10)  # 35 channels
remaining_time: (1,)  # e.g., 50 (raw integer)
global_world_features: (4,)

# AgentEncoder  
query_features: (13,)  # human 1 at position (3, 5), etc.
yellow_agents: (2 × 13,)  # both humans
grey_agents: (1 × 13,)  # robot

# InteractiveObjectEncoder
control_buttons: (2 × 6,)  # 2 buttons × 6 features
# ... other interactive objects ...

# GoalEncoder
goal_coords: (4,)  # e.g., (7, 8, 7, 8) for point goal
```

# Encoder Architecture for Neural Policy Priors

This document explains the architecture of the neural network encoders used in the `NeuralHumanPolicyPrior` system, including the rationale for certain design decisions and the complementary roles of grid-based and list-based encodings.

## Overview

The neural policy prior system uses four main encoders to transform environment state into features suitable for Q-value prediction:

1. **StateEncoder** - Encodes the grid-based world state (objects, agent distributions by color)
2. **AgentEncoder** - Encodes individual agent features (query agent + per-color agent lists)
3. **GoalEncoder** - Encodes goal locations
4. **InteractiveObjectEncoder** - Encodes complex interactive objects (buttons, switches)

These features are combined by the `QNetwork` to predict Q-values for each action.

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
| 0-9 | Base objects | 1.0 = present |
| 10-16 | Per-color doors | 0=none, 0.33=open, 0.67=closed, 1.0=locked |
| 17-23 | Per-color keys | 1.0 = present |
| 24 | Magic walls | 0=none, 0.2-0.8=active (by magic_side), 1.0=inactive |
| 25-28 | Buttons/switches | Grid presence only (details in InteractiveObjectEncoder) |

### Additional Input Features

- **Remaining time**: Raw integer (max_steps - step_count), NOT normalized
- **Global world features** (4 values):
  - Stumble probability (for unsteady ground)
  - Magic wall entry probability
  - Magic wall solidify probability
  - Reserved for future use

## AgentEncoder Features

Each agent is encoded with 13 features:

| Feature | Size | Description |
|---------|------|-------------|
| Position | 2 | Normalized (x/width, y/height) |
| Direction | 4 | One-hot encoding (right, down, left, up) |
| Abilities | 2 | can_enter_magic_walls, can_push_rocks |
| Carried object | 2 | (type_normalized, color_normalized), 0 if none |
| Status | 3 | paused, terminated, forced_next_action (-1 if none) |

### Encoding Structure

```
AgentEncoder input = [query_agent_features] + [per_color_agent_lists]

Query agent: 13 features
Per-color list: num_agents_per_color[color] × 13 features each
```

The query agent is always first, enabling policy transfer regardless of agent count.

## InteractiveObjectEncoder Features

Complex interactive objects are encoded in lists with configurable maximum counts:

### KillButton (5 features each)
- position (2): normalized x, y
- enabled (1): 0.0 or 1.0
- trigger_color (1): which color agent triggers it
- target_color (1): which color agents are killed

### PauseSwitch (6 features each)
- position (2): normalized x, y
- enabled (1): 0.0 or 1.0
- is_on (1): current on/off state
- toggle_color (1): which color agent can toggle
- target_color (1): which color agents are paused

### DisablingSwitch (6 features each)
- position (2): normalized x, y
- enabled (1): 0.0 or 1.0
- is_on (1): current on/off state
- toggle_color (1): which color agent can toggle
- target_type (1): which object type is controlled

### ControlButton (6 features each)
- position (2): normalized x, y
- enabled (1): 0.0 or 1.0
- trigger_color (1): which color agent triggers it
- target_color (1): which color agents are controlled
- forced_action (1): which action is forced

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
remaining_time: (1,)
global_world_features: (4,)

# AgentEncoder  
query_features: (13,)  # human 1
yellow_agents: (2 × 13,)  # both humans
grey_agents: (1 × 13,)  # robot

# InteractiveObjectEncoder
control_buttons: (2 × 6,)  # 2 buttons × 6 features
# ... other interactive objects ...

# GoalEncoder
goal_coords: (4,)
```

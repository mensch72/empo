# Encoder Architecture for Neural Policy Priors

This document explains the architecture of the neural network encoders used in the `NeuralHumanPolicyPrior` system, including the rationale for certain design decisions.

## Overview

The neural policy prior system uses a modular architecture with base classes and domain-specific implementations:

### Base Classes (nn_based/)

- **BaseStateEncoder** - Abstract base for encoding complete world state
- **BaseGoalEncoder** - Abstract base for encoding goals (separate from world state)
- **BaseQNetwork** - Base Q-network with Boltzmann policy
- **BasePolicyPriorNetwork** - Base for goal-marginalized policy computation
- **BaseNeuralHumanPolicyPrior** - Base class with save/load, action remapping
- **Trainer** - Generic training loop with TD loss computation

### Multigrid Implementation (nn_based/multigrid/)

- **MultiGridStateEncoder** - Unified encoder for complete multigrid world state:
  - Grid-based CNN for spatial information
  - List-based MLP for agent features
  - List-based MLP for interactive object features
  - Global world features
- **MultiGridGoalEncoder** - Encodes goal positions (not part of world state)
- **MultiGridQNetwork** - Combines state and goal encoders
- **MultiGridNeuralHumanPolicyPrior** - With multigrid-specific validation

## Design Principle: Unified State Encoding

The `MultiGridStateEncoder` encodes the **complete world state** as a single feature vector. This includes:

1. **Spatial grid information** (via CNN)
2. **Agent features** (via MLP)
3. **Interactive object features** (via MLP)
4. **Global world features**

Goals are encoded separately by `GoalEncoder` because they represent the agent's objective, not the world state itself.

## Design Principle: No Normalization

**All values are passed as raw integers/floats, NOT normalized.** This is intentional because:

1. **Absolute distances matter**: The number of steps to reach a goal depends on absolute grid distance
2. **Scale information is useful**: The network can learn that larger grids require different strategies
3. **Simplicity**: No need for denormalization or scaling factors during inference
4. **Policy transfer**: Using absolute coordinates enables loading policies trained on larger grids for use on smaller grids

## Cross-Grid Policy Loading

Policies trained on larger grids can be loaded and used on smaller grids. This enables:

- **Transfer learning**: Train once on a large grid, deploy on various smaller grids
- **Efficient training**: Train on diverse large environments, use in constrained spaces
- **Backward compatibility**: Upgrade to larger training environments without retraining for smaller deployments

When a policy trained on a larger grid is loaded for a smaller grid:
1. The encoder maintains the larger grid dimensions (from the trained policy)
2. The actual smaller world area is encoded normally
3. The area outside the smaller world is padded with grey walls (channel 0)
4. Agents, objects, and goals use absolute integer coordinates, so they remain valid

Loading a policy trained on a **smaller** grid for a **larger** grid is not supported, as coordinates would be out of bounds.

## Design Principle: Grid vs. List Encoding

Within the unified StateEncoder, two complementary strategies are used:

- **Grid-based encoding**: Captures the *spatial distribution* of objects and agents by color
- **List-based encoding**: Captures *detailed features* of individual entities

This separation enables:
1. Policy transfer across different entity counts
2. Rich feature encoding without exponential channel growth
3. Efficient CNN processing for spatial information

## MultiGridStateEncoder Structure

### Grid Channels

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
| 14-20 | Per-color doors (7 colors) | 0=none, 1=open, 2=closed, 3=locked |
| 21-27 | Per-color keys (7 colors) | 1.0 = present |
| 28 | Magic walls | 0=none, 1-4=active (by magic_side), 5=inactive |

### Global World Features (4 values)

- Remaining time: Raw integer (max_steps - step_count)
- Stumble probability (for unsteady ground)
- Magic wall entry probability
- Magic wall solidify probability

### Agent Features (13 per agent, all raw)

| Feature | Size | Description |
|---------|------|-------------|
| Position | 2 | Raw (x, y) coordinates |
| Direction | 4 | One-hot encoding (right, down, left, up) |
| Abilities | 2 | can_enter_magic_walls (0/1), can_push_rocks (0/1) |
| Carried object | 2 | (type_index, color_index), -1 if none |
| Status | 3 | paused (0/1), terminated (0/1), forced_action (-1 if none) |

Structure: `[query_agent_features] + [per_color_agent_lists]`

### Interactive Object Features

All values are raw integers (not normalized).

**KillButton** (5 features): position (2), enabled (1), trigger_color (1), target_color (1)

**PauseSwitch** (6 features): position (2), enabled (1), is_on (1), toggle_color (1), target_color (1)

**DisablingSwitch** (6 features): position (2), enabled (1), is_on (1), toggle_color (1), target_type (1)

**ControlButton** (7 features): position (2), enabled (1), trigger_color (1), target_color (1), forced_action (1), awaiting_action (1)

Note: The `awaiting_action` flag persists across time steps.

## MultiGridGoalEncoder

Goals are encoded with raw coordinates (2 values): x, y

## Policy Transfer Capabilities

This architecture enables:

1. **More agents per color**: Extra agents visible in grid, up to max in list encoder
2. **More interactive objects**: Up to configurable maximum per type
3. **Different action spaces**: Action mapping with fallback for unknown actions
4. **New object types**: Encoded in "other objects" fallback channels

## Known Limitations

### Extensibility
When adding new object types or agent features to the multigrid environment, update the encoders accordingly. See the maintenance note in `multigrid.py`.

### (unused) Box.contains
Boxes are encoded only by grid presence. Contents of boxes are not encoded. If needed, add a list-based encoder for Box objects.
But this project is not using boxes anyway.

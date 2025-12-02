# Encoder Architecture for Neural Policy Priors

This document explains the architecture of the neural network encoders used in the `NeuralHumanPolicyPrior` system, including the rationale for certain design decisions and the complementary roles of grid-based and list-based encodings.

## Overview

The neural policy prior system uses three main encoders to transform environment state into features suitable for Q-value prediction:

1. **StateEncoder** - Encodes the grid-based world state (objects, agent distributions by color)
2. **AgentEncoder** - Encodes individual agent features (query agent + per-color agent lists)
3. **GoalEncoder** - Encodes goal locations

These features are combined by the `QNetwork` to predict Q-values for each action.

## Design Principle: Grid vs. List Encoding

The architecture uses two complementary encoding strategies:

- **Grid-based encoding (StateEncoder)**: Captures the *distribution* of agents by color - where agents of each type are located on the grid. This is spatially rich but doesn't distinguish individual agents within a color.

- **List-based encoding (AgentEncoder)**: Captures *individual* agent features - the specific position and direction of each agent. This allows the network to reason about specific agents.

This separation enables policy transfer: the grid-based encoding naturally handles varying numbers of agents per color, while the list-based encoding provides fixed-size slots for individual agent information.

## StateEncoder Channel Structure

The StateEncoder uses a CNN to process a multi-channel grid representation:

```
Total Channels = num_object_types + 3 + num_colors + 1
                 ---------------   -   -----------   -
                       |           |       |         |
                       |           |       |         +-- Query agent channel
                       |           |       +-- Per-color agent channels
                       |           +-- "Other objects" channels (3 types)
                       +-- Explicit object type channels (walls, doors, lava, etc.)
```

### Channel Details

1. **Object Type Channels** (16 channels by default):
   - Wall, Door, Key, Ball, Box, Goal, Lava, Block, Rock, UnsteadyGround, Switch, etc.
   - Each cell marked with 1.0 if that object type is present

2. **"Other Objects" Channels** (3 channels):
   - **Other Overlappable Objects**: Objects not in the explicit list that agents can overlap with
   - **Other Non-Overlappable Immobile Objects**: Static blocking objects not in the explicit list
   - **Other Non-Overlappable Mobile Objects**: Movable blocking objects not in the explicit list
   
   **Justification**: These channels enable policy transfer to environments with novel object types. Objects the network wasn't trained on get encoded in these fallback channels based on their properties, allowing the learned policy to generalize.

3. **Per-Color Agent Channels** (num_colors channels):
   - One channel per agent color (e.g., "yellow" for humans, "grey" for robots)
   - All agents of a given color are marked in their color's channel
   
   **Justification**: Agents are distinguished by their role (color), not their index. This enables:
   - Policy transfer to environments with more agents of a color
   - Learning general behaviors that apply to any agent of a given type
   - Reasoning about the spatial distribution of agent types

4. **Query Agent Channel** (1 channel):
   - Marks the position of the specific agent being queried
   
   **Justification**: While agents are grouped by color in the grid, the network needs to know which specific agent is the "subject" of the query. This channel explicitly marks that agent's position.

## AgentEncoder Architecture

The AgentEncoder processes agent-specific features using a list-based approach:

```
Input Features (concatenated):
├── Query Agent Features (6D: pos + dir)  ← FIRST in the list
├── Color 0 Agents: [Agent 0 features, Agent 1 features, ...]
├── Color 1 Agents: [Agent 0 features, Agent 1 features, ...]
└── ...

Each agent contributes: position (2D) + direction (4D one-hot) = 6 features

Output: Combined feature vector (32D)
```

### Key Design Decisions

1. **Query Agent First**: The query agent's features are always the first element in the concatenation. This:
   - Makes the query agent easily identifiable regardless of its index
   - Enables policy transfer to environments with different agent configurations
   - Separates "who is being queried" from "what color is that agent"

2. **Per-Color Agent Lists**: For each color, the encoder has fixed slots for `num_agents_per_color[color]` agents. This:
   - Provides a consistent input size regardless of actual agent count
   - Allows reasoning about individual agents within each color group
   - Zero-pads missing agents (fewer than max in that color)

3. **No Agent Index Embedding**: Unlike the previous architecture, there's no learned embedding for agent indices. The query agent is identified by:
   - Being first in the AgentEncoder input
   - Having its position marked in the query agent grid channel
   - This removes the capacity limitation of index embeddings

### Example: 2 Humans + 1 Robot

```
num_agents_per_color = {'yellow': 2, 'grey': 1}
agent_colors = ['grey', 'yellow', 'yellow']  # robot at 0, humans at 1,2

Query for human 1:
├── Query features: [pos_1, dir_1]              # 6 features
├── Grey agents:    [pos_0, dir_0]              # 6 features (robot)
└── Yellow agents:  [pos_1, dir_1, pos_2, dir_2] # 12 features (both humans)

Total input: 24 features → MLP → 32D output
```

## Why This Redundancy?

The query agent appears in multiple places:
1. In the query agent grid channel (StateEncoder)
2. As the first element in AgentEncoder
3. In its color's per-color grid channel

This redundancy serves different purposes:

| Encoding | Information Provided | Transfer Benefit |
|----------|---------------------|------------------|
| Per-color grid | Agent distribution by type | Works with any agent count |
| Query agent grid | Which position is being queried | Spatial context for CNN |
| Query agent features (first) | Exact position/direction | Works with any configuration |
| Per-color agent lists | Individual agent states | Fixed slots, zero-padded |

## Policy Transfer Capabilities

This architecture enables several transfer scenarios:

1. **Same Grid, More Agents per Color**: Load a network trained with 2 humans into an environment with 3 humans. The extra human's features fit in the per-color list (if slots available) or are visible only in the grid.

2. **Same Grid, Different Actions**: Load a network trained with 4 actions into an environment with 8 actions. Action mapping handles compatible actions; new actions get zero probability.

3. **New Object Types**: Objects not in the training set are encoded in "other objects" channels based on their properties (overlappable, mobile, etc.).

4. **Different Grid Size**: Not supported - grid dimensions must match exactly because the CNN architecture is size-specific.

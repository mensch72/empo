# Encoder Architecture for Neural Policy Priors

This document explains the architecture of the neural network encoders used in the `NeuralHumanPolicyPrior` system, including the rationale for certain design decisions and seemingly redundant encodings.

## Overview

The neural policy prior system uses three main encoders to transform environment state into features suitable for Q-value prediction:

1. **StateEncoder** - Encodes the grid-based world state (objects, agents, walls)
2. **AgentEncoder** - Encodes the query agent's attributes (position, direction, index)
3. **GoalEncoder** - Encodes goal locations

These features are combined by the `QNetwork` to predict Q-values for each action.

## StateEncoder Channel Structure

The StateEncoder uses a CNN to process a multi-channel grid representation. The channels are organized as follows:

```
Total Channels = num_object_types + 3 + num_agents + 1 + 1
                 ---------------   -   ----------   -   -
                       |           |       |        |   |
                       |           |       |        |   +-- "Other humans" channel
                       |           |       |        +-- Query agent grid channel
                       |           |       +-- Per-agent position channels
                       |           +-- "Other objects" channels (3 types)
                       +-- Explicit object type channels (walls, doors, lava, etc.)
```

### Channel Details

1. **Object Type Channels** (8 channels by default):
   - Wall, Door, Key, Ball, Box, Goal, Lava, Block
   - Each cell marked with 1.0 if that object type is present

2. **"Other Objects" Channels** (3 channels):
   - **Other Overlappable Objects**: Objects not in the explicit list that agents can overlap with
   - **Other Non-Overlappable Immobile Objects**: Static blocking objects not in the explicit list
   - **Other Non-Overlappable Mobile Objects**: Movable blocking objects not in the explicit list
   
   **Justification**: These channels enable policy transfer to environments with novel object types. Objects the network wasn't trained on get encoded in these fallback channels based on their properties, allowing the learned policy to generalize.

3. **Per-Agent Position Channels** (num_agents channels):
   - One channel per agent index
   - Each agent's position marked with 1.0 on their dedicated channel
   
   **Justification**: Enables the network to distinguish between different agents and learn agent-specific behaviors or interactions.

4. **Query Agent Grid Channel** (1 channel):
   - Marks the position of the agent being queried (the one whose policy we're computing)
   
   **Justification**: This is redundant with the per-agent channel but serves an important purpose for policy transfer. When loading a trained network into an environment with more agents than originally trained, the per-agent embedding in AgentEncoder may not cover new agent indices. The grid channel ensures the query agent's position is always explicitly encoded regardless of agent count.

5. **"Other Humans" Channel** (1 channel):
   - Marks positions of all human agents except the query agent
   
   **Justification**: Provides anonymous information about other human agents without requiring the network to track specific identities. Useful for learning collision avoidance and coordination behaviors.

## AgentEncoder Architecture

The AgentEncoder processes agent-specific features:

```
Input Features:
├── Position (2D, normalized)
├── Direction (4D one-hot)
├── Agent Index Embedding (16D)
└── Query Agent Features (16D) ← NEW: Dedicated encoding pathway

Output: Combined feature vector (32D)
```

### Query Agent Encoder

The `query_agent_encoder` is a dedicated neural pathway that encodes the query agent's position and direction independently of the agent index:

```python
query_agent_encoder = Sequential(
    Linear(6, 16),  # pos(2) + dir(4) -> 16
    ReLU()
)
```

**Why This Redundancy is Important:**

1. **Policy Transfer Across Agent Counts**: The agent index embedding has fixed capacity (trained for N agents). When loading into an environment with N+K agents, agent indices > N cannot be properly embedded. The query agent encoder captures the agent's position and direction without relying on the index embedding.

2. **Index Embedding vs. Feature Encoding**: The index embedding learns to distinguish agents by their ID during training. The query agent encoder learns what being the "query agent" means regardless of which specific agent it is. This separation allows the network to learn both agent-specific patterns and general query-agent patterns.

3. **Graceful Degradation**: For agents beyond the trained capacity, the index embedding is clamped to the maximum valid index. The query agent encoder ensures the actual position/direction information is still properly encoded.

## Redundancy Summary

| Encoding | Purpose | Transfer Benefit |
|----------|---------|------------------|
| Per-agent grid channels | Distinguish agents by position | Works up to trained agent count |
| Query agent grid channel | Mark query agent position | Works with any agent count |
| Agent index embedding | Learn agent-specific behaviors | Works up to trained agent count |
| Query agent encoder | Encode query agent features | Works with any agent count |
| "Other objects" channels | Encode unknown object types | Enables new object type transfer |
| "Other humans" channel | Anonymous human positions | Reduces reliance on specific agent IDs |

## Policy Transfer Capabilities

This architecture enables several transfer scenarios:

1. **Same Grid, More Agents**: Load a network trained with 3 agents into an environment with 5 agents. The query agent encoder and grid channels ensure proper encoding.

2. **Same Grid, Different Actions**: Load a network trained with 4 actions into an environment with 8 actions. Action mapping handles compatible actions; new actions get zero probability.

3. **New Object Types**: Objects not in the training set are encoded in "other objects" channels based on their properties.

4. **Different Grid Size**: Not supported - grid dimensions must match exactly because the CNN architecture is size-specific.

## Example: How Query Agent Encoding Works

Consider an environment with 5 agents but a network trained with 3:

```
Training: agents [0, 1, 2] with embeddings E_0, E_1, E_2

Deployment: agents [0, 1, 2, 3, 4]
- Agent 0 query → E_0 + query_encoder(pos_0, dir_0)
- Agent 1 query → E_1 + query_encoder(pos_1, dir_1)
- Agent 2 query → E_2 + query_encoder(pos_2, dir_2)
- Agent 3 query → E_2 (clamped) + query_encoder(pos_3, dir_3) ← still captures actual position!
- Agent 4 query → E_2 (clamped) + query_encoder(pos_4, dir_4) ← still captures actual position!
```

Agents 3 and 4 share the embedding of agent 2, but their actual position and direction are properly encoded by the query agent encoder, and their grid position is marked in the query agent grid channel.

## Conclusion

The seemingly redundant encodings serve distinct purposes in enabling robust policy transfer. Each encoding pathway captures different aspects of agent identity and position, and their combination ensures the network can generalize beyond its training configuration while still leveraging learned patterns when applicable.

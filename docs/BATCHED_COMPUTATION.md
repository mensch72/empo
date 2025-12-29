# Batched Computation in Phase 2 Training

This document provides extensive documentation of the batching, vectorizing, tensorizing, stacking, and reshaping operations used in the Phase 2 EMPO trainer for efficient GPU/CPU computation.

## Table of Contents

1. [Overview](#overview)
2. [Split Tensorization Optimization](#split-tensorization-optimization)
3. [The Problem: Variable-Size Successor States](#the-problem-variable-size-successor-states)
4. [Key Data Structures](#key-data-structures)
5. [Batched Target Computation](#batched-target-computation) (Stages 1-4)
6. [State Encoding](#state-encoding)
7. [Scatter-Add Aggregation](#scatter-add-aggregation)
8. [V_h^e Batched Computation](#v_he-batched-computation) (Stage 5)
9. [Common Pitfalls](#common-pitfalls)
10. [Performance Comparison](#performance-comparison)

---

## Overview

Phase 2 training requires computing Q_r targets for all robot actions (by a "robot action" we always mean a vector assigning one action to each robot), which involves:
- For each transition in a batch
- For each possible robot actions (e.g., 16 robot action for 2 robots × 4 possible individual actions each)
- For each possible successor state (variable, typically 1-4 per action combination)
- Computing V_r(s') and aggregating by transition probability

**Naive approach:** Nested loops with individual forward passes → ~1000+ forward passes per batch  
**Batched approach:** Collect all states, one forward pass, scatter-add aggregation → 1-2 forward passes per batch

### Data Flow: Actor → Replay Buffer → Trainer

The computation is split between the **actor** (data collection) and **trainer** (learning), with a **replay buffer** in between:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ACTOR (data collection, possibly in parallel)                              │
│  ─────────────────────────────────────────────                              │
│  For each step:                                                             │
│   1. Execute action in environment → get next_state                         │
│   2. Pre-compute transition_probs_by_action for ALL robot actions           │
│      (queries world model once per action combination)                      │
│   3. Compute compact_features for state and next_state:                     │
│      - global_features (4 floats)                                           │
│      - agent_features (~26 floats for 2 agents)                             │
│      - interactive_features (~48-96 floats for buttons/switches)            │
│      - compressed_grid (H×W int32s = 49 values for 7×7 grid)               │
│   4. Store transition in replay buffer                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  REPLAY BUFFER                                                              │
│  ─────────────                                                              │
│  Stores Phase2Transition objects containing:                                │
│   - state, next_state (raw tuples for successor computation)                │
│   - robot_action, human_actions, goals                                      │
│   - transition_probs_by_action: Dict[action_idx → [(prob, successor), ...]] │
│   - compact_features = (global, agent, interactive, compressed_grid)        │
│   - Total storage per state: ~508 bytes (vs 7644 bytes for full tensor)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ sample(batch_size)
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINER (learning, Stages 1-5 below)                                       │
│  ───────────────────────────────────                                        │
│  For each training step:                                                    │
│   1. Sample batch of transitions from replay buffer                         │
│   2. Stage 1: Collect all successor states from cached transition_probs     │
│   3. Stage 2: Batch decompress grids (fully vectorized, no world_model!)    │
│   4. Stage 3: Single batched forward pass through networks                  │
│   5. Stage 4: Scatter-add aggregation for Q_r targets                       │
│   6. Stage 5: Compute V_h^e targets with policy weighting                   │
│   7. Compute losses and update network weights                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Actor pre-computes `transition_probs_by_action`:** This avoids redundant world model queries during training. Each transition's successor states for all robot actions are computed once at collection time.

- **Actor pre-computes `compact_features` including `compressed_grid`:** ALL information needed to reconstruct the full tensorized state is captured at collection time. This includes:
  - Expensive-to-compute features (global, agent, interactive)
  - Compressed grid that captures ALL grid objects (static + dynamic)

- **Trainer needs NO access to world_model:** The compressed grid contains everything needed to reconstruct the full grid tensor. This enables true off-policy learning where transitions may come from different episodes with different world layouts.

- **Fully vectorized batch decompression:** The `decompress_grid_batch_to_tensor` method unpacks an entire batch of compressed grids into full channel tensors using only PyTorch tensor operations - no Python loops over batch elements or grid cells.

---

## Compressed Grid Format

The grid tensor (39 channels × 7 × 7 = 1911 floats = 7644 bytes per state) is compressed into a single int32 per cell (49 int32s = 196 bytes for 7×7 grid).

### Bit Layout (int32 per cell)

| Bits | Field | Values |
|------|-------|--------|
| 0-4 | object_type | 0-29 = standard types, 30 = door, 31 = key |
| 5-7 | object_color | 0-6 = color index (for doors/keys) |
| 8-9 | object_state | 0-3 = door state (none/open/closed/locked) |
| 10-12 | agent_color | 0-6 = agent color, 7 = no agent |
| 13-15 | magic_wall_state | 0 = none, 1-5 = active with side, 6 = inactive |
| 16-17 | other_category | 0 = none, 1-3 = overlappable/immobile/mobile |

### Storage Savings

| Component | Size (7×7 grid) | Notes |
|-----------|-----------------|-------|
| **Full grid tensor** | 7644 bytes | 39 × 7 × 7 × 4 bytes |
| **Compressed grid** | 196 bytes | 7 × 7 × 4 bytes (int32) |
| **Global features** | 16 bytes | 4 floats |
| **Agent features** | 104 bytes | ~26 floats × 4 |
| **Interactive features** | 192 bytes | ~48 floats × 4 |
| **Total compact_features** | **508 bytes** | 15× smaller than full tensor |

---

## Split Tensorization Optimization

State tensorization in multigrid environments involves several steps with very different computational profiles:

### Cost Analysis

| Component | Computation | Size (floats) | Cost |
|-----------|-------------|---------------|------|
| **Grid tensor** | Position → channel mapping | 39×7×7 = 1911 | Cheap |
| **Global features** | Extract grid dimensions, carry status | 4 | Cheap |
| **Agent features** | Extract position, direction, carrying for each agent | 13 per agent | Medium |
| **Interactive features** | **Scan entire grid** for buttons, switches, doors | 24-48 per object | **Expensive** |

The **expensive** part is scanning the grid to find interactive objects (buttons, switches, doors) and extracting their properties. This requires iterating over all grid cells and checking object types.

The **cheap** part is the grid tensor itself - it's just a position-to-channel mapping that writes object type indices to a 3D tensor.

### The Problem

Without optimization, during training:
1. Sample batch of 32 transitions
2. Each transition has ~64 successor states (16 robot actions × 4 human actions)
3. Total: 32 × 64 = 2048 states need tensorization
4. Each tensorization scans the grid for interactive objects → 2048 expensive scans

### The Solution: Pre-compute Expensive Parts

```python
# ACTOR: At collection time (once per state)
state_encoder = ...

# Get compact features including compressed grid (expensive, but small to store)
global_feats, agent_feats, interactive_feats = state_encoder.tensorize_state_compact(
    state, goal, device
)
compressed_grid = state_encoder.compress_grid(world_model, state)
# Total: ~78 floats + 49 int32s = ~508 bytes (vs 7644 bytes for full tensor)

# Store in transition
transition.compact_features = (global_feats, agent_feats, interactive_feats, compressed_grid)
```

```python
# TRAINER: At training time (batch of states)
# Decompress grids directly - NO world_model needed!
compressed_grids = torch.stack([cf[3] for cf in compact_features_list])  # (B, H, W) int32
grid_tensors = decompress_grid_batch_to_tensor(compressed_grids, device)  # (B, C, H, W)

# Combine with pre-computed compact features  
full_state_tensor = state_encoder.tensorize_state_from_compact(
    state, goal, device, 
    global_feats, agent_feats, interactive_feats,
    grid_tensor=grid_tensors[i]  # provide pre-decompressed grid
)
```

### Storage Savings

Per state stored in replay buffer:
- **Before:** Full grid tensor = 39×7×7 = 1911 floats = **7644 bytes**
- **After:** Compact features + compressed grid = ~508 bytes

**Savings: 15× smaller storage, with NO dependency on world_model at training time**

### Implementation

See `MultiGridStateEncoder` for the split tensorization methods:
- `tensorize_state_compact()` - Returns (global, agent, interactive) tensors
- `compress_grid()` - Returns (H, W) int32 tensor with all grid info
- `decompress_grid_to_tensor()` - Unpacks single grid to (1, C, H, W) tensor
- `decompress_grid_batch_to_tensor()` - Fully vectorized batch decompression
- `tensorize_state_from_compact()` - Combines pre-computed features with grid

See `MultiGridPhase2Trainer` for usage:
- `collect_transition()` - Overrides parent to compute compact features + compressed grid
- `_batch_tensorize_from_compact()` - Decompresses grids in batch, no world_model access

---

## The Problem: Variable-Size Successor States

The fundamental challenge is that the number of successor states varies:

```
Batch of 32 transitions
├── Transition 0
│   ├── Action 0: [(0.8, s'_0), (0.2, s'_1)]     # 2 successors
│   ├── Action 1: [(1.0, s'_2)]                   # 1 successor
│   ├── ...
│   └── Action 15: [(0.5, s'_x), (0.3, s'_y), (0.2, s'_z)]  # 3 successors
├── Transition 1
│   ├── Action 0: [(1.0, s'_a)]                   # 1 successor
│   └── ...
└── Transition 31
    └── ...
```

**Total successors:** Variable, typically 32 × 16 × ~2 ≈ 1024 states

We cannot use a fixed-size tensor like `(batch, actions, max_successors)` without padding, which wastes computation. Instead, we **flatten** everything into variable-length lists and use **scatter-add** to aggregate.

---

## Key Data Structures

### Flattened Successor Lists

```python
# Collected during Stage 1 of _compute_model_based_targets_batched()
all_successor_states = []      # List of raw state tuples
all_successor_probs = []       # List of floats (transition probabilities)
all_successor_trans_idx = []   # List of ints (which transition: 0..n_transitions-1)
all_successor_action_idx = []  # List of ints (which action: 0..num_actions-1)
```

**Example** with 3 transitions, 4 actions, variable successors:

| Index | State | Prob | Trans | Action |
|-------|-------|------|-------|--------|
| 0 | s'_00a | 0.7 | 0 | 0 |
| 1 | s'_00b | 0.3 | 0 | 0 |
| 2 | s'_01 | 1.0 | 0 | 1 |
| 3 | s'_02a | 0.5 | 0 | 2 |
| 4 | s'_02b | 0.5 | 0 | 2 |
| 5 | s'_10 | 1.0 | 1 | 0 |
| 6 | s'_11a | 0.8 | 1 | 1 |
| 7 | s'_11b | 0.2 | 1 | 1 |
| ... | ... | ... | ... | ... |

All lists have the same length: `n_successors = len(all_successor_states)`

---

## Batched Target Computation

> **All stages below run in the TRAINER** after sampling a batch from the replay buffer.
> The `transition_probs_by_action` used in Stage 1 was pre-computed by the **actor** at data collection time.

### Stage 1: Collect All Successor States (Trainer)

The trainer iterates over the sampled batch and extracts all successor states from the **pre-computed** `transition_probs_by_action` dictionaries (computed by the actor, stored in replay buffer).

```python
def _compute_model_based_targets_batched(self, batch, effective_beta_r, ...):
    # batch = list of Phase2Transition sampled from replay buffer
    num_actions = self.networks.q_r.num_action_combinations  # e.g., 16
    n_transitions = len(batch)  # e.g., 32
    
    # Stage 1: Flatten all successors into lists
    # (using transition_probs_by_action that actor pre-computed)
    all_successor_states = []
    all_successor_probs = []
    all_successor_trans_idx = []
    all_successor_action_idx = []
    
    for trans_idx, t in enumerate(batch):
        cached_trans_probs = t.transition_probs_by_action  # Pre-computed by ACTOR
        
        for action_idx in range(num_actions):
            trans_probs = cached_trans_probs.get(action_idx, [])
            for prob, next_state in trans_probs:
                if prob > 0:
                    all_successor_states.append(next_state)
                    all_successor_probs.append(prob)
                    all_successor_trans_idx.append(trans_idx)
                    all_successor_action_idx.append(action_idx)
    
    n_successors = len(all_successor_states)  # e.g., 1024
```

**Output shapes:**
- `all_successor_states`: Python list of length `n_successors`
- `all_successor_probs`: Python list of length `n_successors`
- `all_successor_trans_idx`: Python list of length `n_successors`
- `all_successor_action_idx`: Python list of length `n_successors`

---

### Stage 2: Batch Tensorize All States (Trainer)

The trainer converts all raw successor states (collected in Stage 1) into tensor format for GPU computation. 

**Optimized path with pre-computed compact features + compressed grid:**

When the replay buffer contains pre-computed `compact_features` including compressed grid, the trainer can reconstruct full state tensors **without any access to world_model**. This is critical for off-policy learning where transitions may come from different episodes with different world layouts.

```python
# Stage 2: Fully vectorized batch tensorization from compressed data
# Decompress grids in batch - NO world_model needed!
s_prime_encoded = self._batch_tensorize_from_compact(
    all_successor_states,   # Not needed for grid (using compressed_grid instead)
    all_compact_features    # Pre-computed (global, agent, interactive, compressed_grid) tuples
)
```

The `_batch_tensorize_from_compact` method:

```python
def _batch_tensorize_from_compact(
    self, 
    states: List[Any], 
    compact_features: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Batch tensorize using pre-computed compact features and compressed grid.
    
    Key insight: The compressed_grid contains ALL grid information (static + dynamic),
    so we don't need world_model access. This enables true off-policy learning.
    
    Storage savings: ~15x smaller in replay buffer
      - Full grid tensor: 39×7×7 = 1911 floats = 7644 bytes per state
      - Compact features + compressed grid: ~508 bytes per state
    """
    state_encoder = self.networks.q_r.state_encoder
    
    # Fully vectorized batch decompression
    compressed_grids = torch.stack([cf[3] for cf in compact_features])  # (B, H, W) int32
    grid_tensors = state_encoder.decompress_grid_batch_to_tensor(
        compressed_grids, self.device
    )  # (B, C, H, W)
    
    global_tensors = torch.stack([cf[0] for cf in compact_features]).to(self.device)
    agent_tensors = torch.stack([cf[1] for cf in compact_features]).to(self.device)
    interactive_tensors = torch.stack([cf[2] for cf in compact_features]).to(self.device)
    
    return (grid_tensors, global_tensors, agent_tensors, interactive_tensors)
```

**Why compressed grid is necessary:**

Without the compressed grid, the trainer would need to call `tensorize_state_grid(state, world_model, device)` to reconstruct the grid tensor. But in off-policy learning, transitions in the replay buffer may come from episodes with different world layouts (different door/key colors, wall positions, etc.). The trainer cannot access the original world_model because:

1. It may have been from a different episode
2. We can't store thousands of world_models per transition
3. The environment may have been reset with a different world

The compressed grid solves this by capturing ALL grid information at collection time, making the trainer completely independent of world_model access.

**Fallback path (no compact features):**

If compact features are not available, use the full tensorization:

```python
def _batch_tensorize_states(self, states: List[Any]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    state_encoder = self.networks.q_r.state_encoder
    
    grid_list = []
    global_list = []
    agent_list = []
    interactive_list = []
    
    for state in states:
        # Full tensorization (expensive: scans grid, extracts attributes)
        grid, glob, agent, interactive = state_encoder.tensorize_state(state, None, self.device)
        
        grid_list.append(grid)
        global_list.append(glob)
        agent_list.append(agent)
        interactive_list.append(interactive)
    
    return (
        torch.cat(grid_list, dim=0),
        torch.cat(global_list, dim=0),
        torch.cat(agent_list, dim=0),
        torch.cat(interactive_list, dim=0)
    )
```

**Output shapes for 7×7 grid with 1024 successors:**
```
s_prime_encoded[0] (grid):        (1024, 39, 7, 7)
s_prime_encoded[1] (global):      (1024, 4)
s_prime_encoded[2] (agent):       (1024, 52)
s_prime_encoded[3] (interactive): (1024, 96)
```

---

### Stage 3: Single Forward Pass for V_r (Trainer)

The trainer performs a single batched forward pass through the value network for all successor states.

```python
# Stage 3: Compute V_r for ALL successor states in ONE forward pass
with torch.no_grad():
    if self.config.v_r_use_network:
        # Direct V_r network
        v_r_all = self.networks.v_r_target.forward(*s_prime_encoded)
        # v_r_all: (n_successors,)
    else:
        # Compute V_r = U_r + E[Q_r] analytically
        
        # U_r for all states
        u_r_all = self._compute_u_r_from_encoded_state_batched(s_prime_encoded, all_successor_states)
        # u_r_all: (n_successors,)
        
        # Q_r needs its own state encoder
        own_s_prime_encoded = self._batch_tensorize_states_with_encoder(
            all_successor_states, self.networks.q_r.own_state_encoder
        )
        # own_s_prime_encoded: tuple of (n_successors, ...) tensors
        
        # Q_r forward pass
        q_r_all_next = self.networks.q_r.forward(*s_prime_encoded, *own_s_prime_encoded)
        # q_r_all_next: (n_successors, num_actions) = (1024, 16)
        
        # Robot policy π_r(a|s)
        pi_r_all = self.networks.q_r.get_policy(q_r_all_next, beta_r=effective_beta_r)
        # pi_r_all: (n_successors, num_actions) = (1024, 16)
        
        # V_r = U_r + Σ_a π_r(a|s) Q_r(s,a)
        policy_value = (pi_r_all * q_r_all_next).sum(dim=-1)  # (n_successors,)
        v_r_all = u_r_all + policy_value  # (n_successors,)
```

**Critical shape handling:**

```python
# Ensure v_r_values is 1D
v_r_values = v_r_all.squeeze()  # Remove any trailing dims
if v_r_values.dim() == 0:
    v_r_values = v_r_values.unsqueeze(0)  # Handle single-element case
if v_r_values.dim() > 1:
    v_r_values = v_r_values.view(-1)  # Flatten to 1D
# Final: v_r_values: (n_successors,)
```

---

### Stage 4: Scatter-Add Aggregation (Trainer)

The trainer aggregates V_r values back into per-(transition, action) expected values using scatter-add.

This is the key innovation for handling variable successors without padding.

**Goal:** Compute `E[V_r(s')] = Σ P(s'|s,a) V_r(s')` for each (transition, action) pair.

```python
# Convert metadata to tensors
probs_tensor = torch.tensor(all_successor_probs, device=self.device, dtype=torch.float32)
# probs_tensor: (n_successors,)

trans_idx_tensor = torch.tensor(all_successor_trans_idx, device=self.device, dtype=torch.long)
# trans_idx_tensor: (n_successors,)

action_idx_tensor = torch.tensor(all_successor_action_idx, device=self.device, dtype=torch.long)
# action_idx_tensor: (n_successors,)

# Weight V_r by transition probability
weighted_v_r = probs_tensor * v_r_values
# weighted_v_r: (n_successors,)

# Compute flat index into (n_transitions × num_actions) array
flat_idx = trans_idx_tensor * num_actions + action_idx_tensor
# flat_idx: (n_successors,) with values in [0, n_transitions * num_actions)

# Initialize output tensor
expected_v_r_flat = torch.zeros(n_transitions * num_actions, device=self.device)
# expected_v_r_flat: (n_transitions * num_actions,) = (512,) for 32 trans × 16 actions

# Scatter-add: accumulate weighted_v_r into expected_v_r_flat at positions flat_idx
expected_v_r_flat.scatter_add_(0, flat_idx, weighted_v_r)
```

**How scatter_add_ works:**

```
For each i in range(n_successors):
    expected_v_r_flat[flat_idx[i]] += weighted_v_r[i]
```

This handles variable successors naturally - positions with multiple successors get multiple additions, positions with one successor get one addition.

```python
# Reshape to (n_transitions, num_actions)
q_r_targets = self.config.gamma_r * expected_v_r_flat.view(n_transitions, num_actions)
# q_r_targets: (32, 16)
```

---

## State Encoding

### MultiGridStateEncoder

The state encoder converts raw multigrid states to tensor representations:

```python
def tensorize_state(self, state, world_model, device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        state: (grid_encoding, agent_positions, agent_directions, metadata)
    
    Returns:
        grid_tensor:        (1, num_channels, H, W)
        global_features:    (1, num_global_features)
        agent_features:     (1, agent_feature_size)
        interactive_features: (1, interactive_feature_size)
    """
```

**Grid tensor channels (39 for 7×7 grid):**
- Object type one-hot (walls, doors, keys, etc.)
- Object state (open/closed doors, etc.)
- Agent presence per color

**Global features (4):**
- Step count normalized
- Episode progress
- Global state flags

**Agent features (52):**
- Per-agent: position (x, y), direction, carrying object

**Interactive features (96):**
- Button states, switch states, door states

---

### Batch Encoding with Caching

```python
def _batch_tensorize_states(self, states: List[Any]) -> Tuple[Tensor, ...]:
    """
    Encodes multiple states, concatenating along batch dimension.
    
    Input: List of n states
    Output: Tuple of tensors, each with shape (n, ...)
    """
    grid_list, global_list, agent_list, interactive_list = [], [], [], []
    
    for state in states:
        # tensorize_state returns (1, ...) tensors
        g, gl, a, i = self.state_encoder.tensorize_state(state, None, self.device)
        grid_list.append(g)
        global_list.append(gl)
        agent_list.append(a)
        interactive_list.append(i)
    
    return (
        torch.cat(grid_list, dim=0),       # (n, C, H, W)
        torch.cat(global_list, dim=0),     # (n, F_g)
        torch.cat(agent_list, dim=0),      # (n, F_a)
        torch.cat(interactive_list, dim=0) # (n, F_i)
    )
```

---

## V_h^e Batched Computation

V_h^e (human goal achievement value) requires additional complexity because it involves:
- Goals (variable per human)
- Agent identity encoding
- Robot policy weighting

### Stage 5: V_h^e Target Computation (Trainer)

The trainer computes V_h^e targets, which additionally require goal encoding and policy weighting.

```python
if v_h_e_goals:
    # Compute robot policies for original states (for weighting)
    original_states = [t.state for t in batch]
    s_encoded = self._batch_tensorize_states(original_states)
    # s_encoded: tuple of (n_transitions, ...) tensors
    
    own_s_encoded = self._batch_tensorize_states_with_encoder(
        original_states, self.networks.q_r.own_state_encoder
    )
    
    with torch.no_grad():
        q_r_orig = self.networks.q_r.forward(*s_encoded, *own_s_encoded)
        # q_r_orig: (n_transitions, num_actions)
        
        robot_policies = self.networks.q_r.get_policy(q_r_orig, beta_r=effective_beta_r)
        # robot_policies: (n_transitions, num_actions)
```

### Building V_h^e Successor Data

```python
# Group V_h^e entries by transition
trans_to_v_h_e = defaultdict(list)  # trans_idx -> [(entry_idx, h_idx, goal), ...]
for entry_idx, trans_idx in enumerate(v_h_e_indices):
    h_idx = v_h_e_human_indices[entry_idx]
    goal = v_h_e_goals[entry_idx]
    trans_to_v_h_e[trans_idx].append((entry_idx, h_idx, goal))

# Build expanded lists for all (successor, human, goal) combinations
v_h_e_succ_indices = []     # Index into all_successor_states
v_h_e_succ_humans = []      # Human index
v_h_e_succ_goals = []       # Goal object
v_h_e_succ_entry_idx = []   # Which v_h_e entry this contributes to
v_h_e_succ_action_idx = []  # Action (for policy weighting)
v_h_e_succ_probs = []       # Transition probability
v_h_e_succ_trans_idx = []   # Transition index

for succ_idx in range(n_successors):
    trans_idx = all_successor_trans_idx[succ_idx]
    action_idx = all_successor_action_idx[succ_idx]
    prob = all_successor_probs[succ_idx]
    
    # Expand for each V_h^e entry in this transition
    for entry_idx, h_idx, goal in trans_to_v_h_e.get(trans_idx, []):
        v_h_e_succ_indices.append(succ_idx)
        v_h_e_succ_humans.append(h_idx)
        v_h_e_succ_goals.append(goal)
        v_h_e_succ_entry_idx.append(entry_idx)
        v_h_e_succ_action_idx.append(action_idx)
        v_h_e_succ_probs.append(prob)
        v_h_e_succ_trans_idx.append(trans_idx)
```

### Batched V_h^e Forward Pass

```python
if v_h_e_succ_indices:
    n_v_h_e_successors = len(v_h_e_succ_indices)
    
    # Get successor states using indices into pre-encoded states
    succ_idx_tensor = torch.tensor(v_h_e_succ_indices, device=self.device, dtype=torch.long)
    v_h_e_s_prime = (
        s_prime_encoded[0][succ_idx_tensor],  # Index into batch dim
        s_prime_encoded[1][succ_idx_tensor],
        s_prime_encoded[2][succ_idx_tensor],
        s_prime_encoded[3][succ_idx_tensor],
    )
    # v_h_e_s_prime: tuple of (n_v_h_e_successors, ...) tensors
    
    # Check goal achievement (vectorized where possible)
    achieved_list = []
    for i, (next_state, h_idx, goal) in enumerate(zip(
        [all_successor_states[i] for i in v_h_e_succ_indices],
        v_h_e_succ_humans,
        v_h_e_succ_goals
    )):
        achieved = self.check_goal_achieved(next_state, h_idx, goal)
        achieved_list.append(1.0 if achieved else 0.0)
    achieved_tensor = torch.tensor(achieved_list, device=self.device)
    # achieved_tensor: (n_v_h_e_successors,)
    
    # Encode goals
    goal_features = self._batch_tensorize_goals(v_h_e_succ_goals)
    # goal_features: (n_v_h_e_successors, goal_feature_dim)
    
    # Encode agent identities
    v_h_e_idx, v_h_e_grid, v_h_e_feat = self._batch_tensorize_agent_identities(
        v_h_e_succ_humans,
        [all_successor_states[i] for i in v_h_e_succ_indices]
    )
    # v_h_e_idx: (n_v_h_e_successors,), v_h_e_grid: (n_v_h_e_successors, ...), etc.
    
    # Single batched forward pass
    with torch.no_grad():
        v_h_e_next_all = self.networks.v_h_e_target.forward(
            v_h_e_s_prime[0], v_h_e_s_prime[1],
            v_h_e_s_prime[2], v_h_e_s_prime[3],
            goal_features, v_h_e_idx, v_h_e_grid, v_h_e_feat
        ).squeeze()
        # v_h_e_next_all: (n_v_h_e_successors,)
```

### Aggregating V_h^e with Policy Weighting

```python
    # TD target: achieved + (1 - achieved) * γ * V_h^e(s')
    td_targets = achieved_tensor + (1.0 - achieved_tensor) * self.config.gamma_h * v_h_e_next_all
    # td_targets: (n_v_h_e_successors,)
    
    # Get indices as tensors
    probs_v_h_e = torch.tensor(v_h_e_succ_probs, device=self.device)
    action_idx_v_h_e = torch.tensor(v_h_e_succ_action_idx, device=self.device, dtype=torch.long)
    trans_idx_v_h_e = torch.tensor(v_h_e_succ_trans_idx, device=self.device, dtype=torch.long)
    entry_idx_v_h_e = torch.tensor(v_h_e_succ_entry_idx, device=self.device, dtype=torch.long)
    
    # Get policy weights: π_r(action | state) for each successor
    policy_weights = robot_policies[trans_idx_v_h_e, action_idx_v_h_e]
    # policy_weights: (n_v_h_e_successors,)
    
    # Weight TD targets by policy and transition probability
    weighted_td = policy_weights * probs_v_h_e * td_targets
    # weighted_td: (n_v_h_e_successors,)
    
    # Scatter-add to aggregate into final V_h^e targets
    v_h_e_targets_tensor = torch.zeros(len(v_h_e_goals), device=self.device)
    v_h_e_targets_tensor.scatter_add_(0, entry_idx_v_h_e, weighted_td)
    # v_h_e_targets_tensor: (n_v_h_e_entries,)
```

---

## Common Pitfalls

### 1. Broadcasting Shape Mismatch

**Bug:**
```python
# u_r_all: (n,)
# policy_value with keepdim=True: (n, 1)
v_r_all = u_r_all + (pi_r_all * q_r_all_next).sum(dim=-1, keepdim=True)
# Broadcasting: (n,) + (n, 1) → (n, n) ← WRONG!
```

**Fix:**
```python
policy_value = (pi_r_all * q_r_all_next).sum(dim=-1)  # (n,)
v_r_all = u_r_all + policy_value  # (n,) + (n,) → (n,) ✓
```

### 2. Forgetting to Handle dim=0 After Squeeze

**Bug:**
```python
v_r_values = v_r_all.squeeze()  # If v_r_all is (1,), squeeze gives scalar tensor
# Later operations fail on 0-dim tensor
```

**Fix:**
```python
v_r_values = v_r_all.squeeze()
if v_r_values.dim() == 0:
    v_r_values = v_r_values.unsqueeze(0)
```

### 3. Using Wrong Index for Scatter

**Bug:**
```python
# Forgetting to flatten the 2D index
expected_v_r[trans_idx_tensor, action_idx_tensor] += weighted_v_r  # Can't do this!
```

**Fix:**
```python
flat_idx = trans_idx_tensor * num_actions + action_idx_tensor
expected_v_r_flat.scatter_add_(0, flat_idx, weighted_v_r)
expected_v_r = expected_v_r_flat.view(n_transitions, num_actions)
```

### 4. Mixing Python Lists with Tensor Indexing

**Bug:**
```python
states = [t.state for t in batch]  # Python list
encoded = encoder.forward(states)  # Encoder expects tensor batches
```

**Fix:**
```python
states = [t.state for t in batch]
encoded = self._batch_tensorize_states(states)  # Properly handles list → tensor
```

---

## Performance Comparison

### Naive Implementation (SLOW)

```python
# ~1000+ forward passes for batch_size=32, num_actions=16
for trans_idx, t in enumerate(batch):
    for action_idx in range(num_actions):
        for prob, next_state in t.transition_probs[action_idx]:
            # Individual encode
            s_encoded = tensorize_state(next_state)  # Forward pass through encoder
            # Individual V_r forward pass
            v_r = v_r_network.forward(*s_encoded)  # Forward pass through V_r
            expected_v_r[trans_idx, action_idx] += prob * v_r
```

**Cost:** O(batch × actions × successors) forward passes

### Batched Implementation (FAST)

```python
# 1-2 forward passes total
all_states = collect_all_successors(batch)           # O(batch × actions × successors)
s_encoded = batch_tensorize_states(all_states)          # 1 encode pass
v_r_all = v_r_network.forward(*s_encoded)            # 1 forward pass
q_r_targets = scatter_add_aggregate(v_r_all, ...)    # O(n_successors) scatter ops
```

**Cost:** O(1) forward passes + O(n_successors) scatter operations

### Measured Speedup

| Configuration | Naive | Batched | Speedup |
|--------------|-------|---------|---------|
| 7×7 grid, 2 humans, 2 robots, batch=32 | ~120s/episode | ~3.5s/episode | **~34×** |

---

## Summary

The key techniques for efficient batched computation:

1. **Flatten variable-size data** into parallel lists with metadata indices
2. **Batch encode** all states in one pass using `torch.cat`
3. **Single forward pass** through neural networks
4. **Scatter-add aggregation** to handle variable successors without padding
5. **Careful shape management** to avoid broadcasting bugs

These techniques transform O(N×M×K) forward passes into O(1) forward passes, yielding 30-50× speedups for typical Phase 2 training configurations.

---

## Profiling

To identify bottlenecks in the batched computation, enable built-in profiling:

```python
config = Phase2Config(
    profile_batching=True,           # Enable timing instrumentation
    profile_batching_interval=100,   # Print stats every 100 training steps
)
```

This tracks time spent in each stage:
- `stage1_collect`: Collecting successor states from cached transition probabilities
- `stage2_encode`: Batch encoding raw states → tensors
- `stage3_forward`: Neural network forward passes
- `stage4_scatter`: Scatter-add aggregation for Q_r targets
- `stage5_v_h_e`: V_h^e target computation
- `tensor_conversion`: Python list → tensor conversion overhead

Example output:
```
=== Batching Profile (100 calls) ===
  stage3_forward      :   45.123 ms (52.3%)
  stage2_encode       :   25.456 ms (29.5%)
  stage5_v_h_e        :   10.234 ms (11.8%)
  stage1_collect      :    3.567 ms ( 4.1%)
  stage4_scatter      :    1.234 ms ( 1.4%)
  tensor_conversion   :    0.789 ms ( 0.9%)
  TOTAL               :   86.403 ms
```

You can also access profiling data programmatically:

```python
# After training
trainer.print_batching_profile_stats()  # Print summary
stats = trainer.get_batching_profile_stats()  # Get dict of avg ms per stage
trainer.reset_batching_profile_stats()  # Reset counters
```

For deeper profiling with `line_profiler`:

```bash
pip install line_profiler
# Add @profile decorator to methods of interest, then:
kernprof -l -v examples/phase2_robot_policy_demo.py --quick
```

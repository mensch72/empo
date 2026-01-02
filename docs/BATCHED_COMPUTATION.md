# Batched Computation in Phase 2 Training

> ⚠️ **IMPLEMENTATION STATUS NOTE** (January 2026)
> 
> This document was originally written as a **design specification** describing the ideal
> batched computation approach. Parts of it were never fully implemented:
> 
> **✅ IMPLEMENTED:**
> - Pre-computation of `transition_probs_by_action` by actor at collection time
> - Model-based V_h^e targets using expected value over transition probabilities
> - **Policy-weighted V_h^e targets over ALL robot actions** (weighted by π_r(a|s))
> - Model-based Q_r targets for ALL robot actions (full Bellman backup)
> - Batched forward passes: ONE call per network per batch
> - State deduplication for Q_r target computation
> - Scatter-add aggregation for U_r loss computation
> 
> **❌ NOT IMPLEMENTED (documentation describes design only):**
> - Compressed grid format and `compact_features` storage
> - `decompress_grid_batch_to_tensor` vectorized decompression
> - Split tensorization optimization (actor pre-computing compact features)
> - Scatter-add for Q_r/V_h^e aggregation (uses Python loops for these)
> - Profiling instrumentation (`profile_batching`, `profile_batching_interval`)
> 
> Sections that describe unimplemented features are marked with **[NOT IMPLEMENTED]**.

This document provides extensive documentation of the batching, vectorizing, tensorizing, stacking, and reshaping operations used in the Phase 2 EMPO trainer for efficient GPU/CPU computation.

## Table of Contents

1. [Overview](#overview)
2. [Split Tensorization Optimization](#split-tensorization-optimization) **[NOT IMPLEMENTED]**
3. [The Problem: Variable-Size Successor States](#the-problem-variable-size-successor-states)
4. [Key Data Structures](#key-data-structures)
5. [Batched Target Computation](#batched-target-computation) (Stages 1-4) — ✅ Implemented
6. [State Encoding](#state-encoding)
7. [Scatter-Add Aggregation](#scatter-add-aggregation) — Partial (U_r only)
8. [V_h^e Batched Computation](#v_he-batched-computation) (Stage 5) — ✅ Fully Implemented
9. [Common Pitfalls](#common-pitfalls)
10. [Performance Comparison](#performance-comparison)
11. [Alternative Modes: Disabling Encoders and Lookup Tables](#alternative-modes-disabling-encoders-and-lookup-tables)

---

## Overview

Phase 2 training requires computing Q_r targets for all robot actions (by a "robot action" we always mean a vector assigning one action to each robot), which involves:
- For each transition in a batch
- For each possible robot actions (e.g., 16 robot action for 2 robots × 4 possible individual actions each)
- For each possible successor state (variable, typically 1-4 per action combination)
- Computing V_r(s') and aggregating by transition probability

**Naive approach:** Nested loops with individual forward passes → ~1000+ forward passes per batch  
**Batched approach:** Collect all states, one forward pass, scatter-add aggregation → 1-2 forward passes per batch

### What's Actually Implemented (as of January 2026)

The current implementation achieves batching through:

1. **Actor pre-computes `transition_probs_by_action`** ✅ — Stored in replay buffer
2. **Model-based targets** ✅ — Uses expected values over successor states (not single-sample TD)
3. **Policy-weighted V_h^e** ✅ — V_h^e targets weighted by π_r(a|s) over ALL robot actions
4. **Batched forward passes** ✅ — ONE `forward_batch()` call per network
5. **State deduplication for Q_r** ✅ — Unique successor states collected, then aggregated
6. **Scatter-add for U_r** ✅ — U_r loss uses `scatter_add_` for efficient aggregation
7. **Python loop aggregation for Q_r/V_h^e** — Results aggregated via Python loops

**What's NOT implemented:**
- Compressed grid format (successor states stored as raw tuples)
- Scatter-add for Q_r/V_h^e aggregation (uses Python loops for these networks)
- Profiling instrumentation

### Data Flow: Actor → Replay Buffer → Trainer

The computation is split between the **actor** (data collection) and **trainer** (learning), with a **replay buffer** in between:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ACTOR (data collection, possibly in parallel)                              │
│  ─────────────────────────────────────────────                              │
│  For each step:                                                             │
│   1. Execute action in environment → get next_state                         │
│   2. Pre-compute transition_probs_by_action for ALL robot actions    ✅     │
│      (queries world model once per action combination)                      │
│   3. [NOT IMPLEMENTED] Compute compact_features for state:                  │
│      - global_features, agent_features, interactive_features                │
│      - compressed_grid (H×W int32s)                                         │
│   4. Store transition in replay buffer (next_state=None when model-based)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  REPLAY BUFFER                                                              │
│  ─────────────                                                              │
│  Stores Phase2Transition objects containing:                                │
│   - state (raw tuple)                                                       │
│   - next_state = None (when use_model_based_targets=True)            ✅     │
│   - robot_action, human_actions, goals, goal_weights                        │
│   - transition_probs_by_action: Dict[action_idx → [(prob, successor), ...]] │
│   - [NOT IMPLEMENTED] compact_features                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ sample(batch_size)
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINER (learning)                                                         │
│  ──────────────────                                                         │
│  For each training step:                                                    │
│   1. Sample batch of transitions from replay buffer                  ✅     │
│   2. Collect all successor states from transition_probs_by_action    ✅     │
│   3. [NOT IMPLEMENTED] Batch decompress grids (uses full tensorization)     │
│   4. Single batched forward pass through networks                    ✅     │
│   5. Scatter-add aggregation for U_r                                 ✅     │
│   6. Policy-weighted V_h^e targets over ALL robot actions            ✅     │
│   7. Compute losses and update network weights                       ✅     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Actor pre-computes `transition_probs_by_action`:** ✅ IMPLEMENTED. This avoids redundant world model queries during training. Each transition's successor states for all robot actions are computed once at collection time.

- **Actor pre-computes `compact_features` including `compressed_grid`:** ❌ NOT IMPLEMENTED. The documentation below describes an optimization that was never coded. Currently, states are stored as raw tuples and tensorized at training time using the world model.

- **Trainer needs NO access to world_model:** ❌ NOT TRUE. The current implementation still requires world_model access for tensorization during training. The compressed grid format that would enable world_model-free training is not implemented.

- **Fully vectorized batch decompression:** ❌ NOT IMPLEMENTED. The `decompress_grid_batch_to_tensor` method described below does not exist.

---

## Compressed Grid Format

> ⚠️ **[NOT IMPLEMENTED]** This entire section describes a design that was never coded.
> The compressed grid format, bit layout, and `decompress_grid_batch_to_tensor` method
> do not exist in the codebase. States are currently stored as raw tuples.

The grid tensor (39 channels × 7 × 7 = 1911 floats = 7644 bytes per state) is compressed into a single int32 per cell (49 int32s = 196 bytes for 7×7 grid).

### Bit Layout (int32 per cell)

See `constants.py` for the exact bit masks and shifts used:

| Bits | Field | Values |
|------|-------|--------|
| 0-4 | object_type | 0-28 = standard types (see `OBJECT_TYPE_TO_CHANNEL`), 30 = door, 31 = key |
| 5-7 | object_color | 0-6 = color index (for doors/keys) |
| 8-9 | object_state | 0-3 = door state (none/open/closed/locked) |
| 10-12 | agent_color | 0-6 = agent color, 7 = no agent |
| 13-15 | magic_wall_state | 0 = none, 1-4 = active with side, 5 = inactive |
| 16-17 | other_category | 0 = none, 1 = overlappable, 2 = immobile, 3 = mobile |

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

> ⚠️ **[NOT IMPLEMENTED]** This entire section describes a design that was never coded.
> The `tensorize_state_compact()`, `compress_grid()`, and `tensorize_state_from_compact()`
> methods do not exist. States are tensorized at training time using full tensorization.

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

See `MultiGridStateEncoder` in `src/empo/nn_based/multigrid/state_encoder.py` for the split tensorization methods:
- `tensorize_state_compact()` - **[NOT IMPLEMENTED]** Returns (global, agent, interactive) tensors without grid
- `compress_grid()` - **[NOT IMPLEMENTED]** Returns (H, W) int32 tensor with all grid info
- `decompress_grid_to_tensor()` - **[NOT IMPLEMENTED]** Unpacks single grid to (1, C, H, W) tensor
- `decompress_grid_batch_to_tensor()` - **[NOT IMPLEMENTED]** Fully vectorized batch decompression
- `tensorize_state_from_compact()` - **[NOT IMPLEMENTED]** Combines pre-computed features with grid tensor

See Phase 2 trainer for usage of compact features with compressed grids in replay buffer storage.

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

> **Note:** The scatter-add aggregation described in this document is **NOT IMPLEMENTED**.
> The current implementation uses Python loops for aggregation, which is slower but correct.

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

> **IMPLEMENTATION STATUS:**
> - Stage 1 (Collect Successor States): ✅ IMPLEMENTED
> - Stage 2 (Batch Tensorize): ✅ IMPLEMENTED (but not using compressed grids)
> - Stage 3 (Single Forward Pass): ✅ IMPLEMENTED
> - Stage 4 (Scatter-Add Aggregation): ❌ NOT IMPLEMENTED (uses Python loops)

> **All stages below run in the TRAINER** after sampling a batch from the replay buffer.
> The `transition_probs_by_action` used in Stage 1 was pre-computed by the **actor** at data collection time.

### Stage 1: Collect All Successor States (Trainer)

> ✅ **IMPLEMENTED** in `_compute_model_based_q_r_targets()` and `_compute_model_based_v_h_e_targets()`

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

> ✅ **PARTIALLY IMPLEMENTED** — Batched tensorization works, but NOT using compressed grids.
> The current implementation calls `forward_batch()` which internally tensorizes states.
> The compressed grid optimization described below is NOT IMPLEMENTED.

The trainer converts all raw successor states (collected in Stage 1) into tensor format for GPU computation. 

**Optimized path with pre-computed compact features + compressed grid:**

> ⚠️ **[NOT IMPLEMENTED]** The entire compressed grid path below does not exist.

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

> ✅ **THIS IS WHAT'S ACTUALLY IMPLEMENTED** — Full tensorization at training time.

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

> ✅ **IMPLEMENTED** in `_compute_model_based_q_r_targets()` — ONE batched call to
> `_compute_u_r_batch_target()` and ONE call to `q_r_target.forward_batch()` or
> `v_r_target.forward_batch()`.

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

> ⚠️ **[NOT IMPLEMENTED]** The scatter-add aggregation described below does not exist.
> The current implementation uses Python loops in `_compute_model_based_q_r_targets()`:
> ```python
> for batch_idx in range(batch_size):
>     for action_idx in range(num_actions):
>         for unique_idx, prob in successor_info[batch_idx][action_idx]:
>             expected_target += prob * q_targets_all[unique_idx].item()
> ```
> This is correct but slower than the scatter-add approach.

The trainer aggregates V_r values back into per-(transition, action) expected values using scatter-add.

---

## Scatter-Add Aggregation

> **PARTIAL IMPLEMENTATION:** Scatter-add IS used for U_r loss computation (aggregating X_h values
> by state). However, Q_r and V_h^e target aggregation uses Python loops.
> See `compute_losses()` in `trainer.py` for the U_r scatter-add implementation.

This is the key innovation for handling variable successors without padding.

**Goal:** Compute `E[V_r(s')] = Σ P(s'|s,a) V_r(s')` for each (transition, action) pair.

### U_r Scatter-Add (IMPLEMENTED)

The U_r loss computation uses scatter-add to aggregate X_h values by state:

```python
# Aggregate by state using scatter_add: sum X_h^{-xi} for each state
state_indices = []
for state_idx, n_humans in enumerate(u_r_humans_per_state):
    state_indices.extend([state_idx] * n_humans)
state_indices_t = torch.tensor(state_indices, device=self.device)

x_h_sums = torch.zeros(n_states, device=self.device)
x_h_sums.scatter_add_(0, state_indices_t, x_h_power)

# Average: y = E[X_h^{-xi}]
u_r_targets_tensor = x_h_sums / humans_per_state_t
```

### Q_r/V_h^e Aggregation (Python Loops)

Q_r and V_h^e target aggregation currently uses Python loops rather than scatter-add:

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

> **IMPLEMENTATION STATUS:** ✅ FULLY IMPLEMENTED
> - Batched forward pass: ✅ IMPLEMENTED (ONE call to `v_h_e_target.forward_batch()`)
> - Policy weighting over all actions: ✅ IMPLEMENTED (weighted by π_r(a|s))
> - Python loop aggregation: ✅ IMPLEMENTED (scatter-add not used)

V_h^e (human goal achievement value) requires additional complexity because it involves:
- Goals (variable per human)
- Agent identity encoding
- Robot policy weighting

### Stage 5: V_h^e Target Computation (Trainer)

> ✅ **FULLY IMPLEMENTED** in `_compute_model_based_v_h_e_targets()`
> 
> The current implementation:
> 1. Computes robot policy π_r(a|s) for each state in batch (ONE batched Q_r call)
> 2. For each (transition, human, goal), loops over ALL robot actions weighted by π_r(a|s)
> 3. Collects ALL successor states across ALL (transition, action, goal) combinations
> 4. Makes ONE batched `v_h_e_target.forward_batch()` call
> 5. Aggregates via Python loops with policy weighting
> 
> The V_h^e target is computed as:
> ```
> V_h^e(s, g) = Σ_a π_r(a|s) * Σ_{s'} P(s'|s,a) * [achieved(s',g) + (1-achieved) * γ_h * V_h^e(s',g)]
> ```

The trainer computes V_h^e targets, which additionally require goal encoding and policy weighting.

```python
# Phase 1: Get robot policy π_r(a|s) for each state in batch (ONE Q_r call)
states = [t.state for t in batch]
with torch.no_grad():
    q_r_batch = self.networks.q_r_target.forward_batch(states, self.env, self.device)
    robot_policies = self.networks.q_r_target.get_policy(q_r_batch, beta_r=effective_beta_r)
    # robot_policies: (batch_size, num_actions)

# Phase 2: Collect successor states weighted by π_r(a|s) * P(s'|s,a)
for data_idx, (trans_idx, human_idx, goal) in enumerate(v_h_e_data):
    policy = robot_policies[trans_idx]  # (num_actions,)
    
    # Loop over ALL robot actions, weighted by policy
    for action_idx in range(num_actions):
        action_prob = policy[action_idx].item()
        if action_prob < 1e-8:
            continue
            
        for state_prob, next_state in trans_probs_by_action[action_idx]:
            weight = action_prob * state_prob  # Combined weight
            # ... accumulate achieved or collect for batched V_h^e evaluation
```

### Building V_h^e Successor Data

The implementation collects successor states weighted by π_r(a|s) * P(s'|s,a):

```python
# Collect successor states for batched V_h^e evaluation
all_next_states = []
all_human_indices = []
all_goals = []
successor_mapping = []  # (v_h_e_data_idx, weight)
achieved_contributions = [0.0] * n_samples  # Direct contributions from achieved goals

for data_idx, (trans_idx, human_idx, goal) in enumerate(v_h_e_data):
    policy = robot_policies[trans_idx]
    
    for action_idx in range(num_actions):
        action_prob = policy[action_idx].item()
        if action_prob < 1e-8:
            continue
            
        for state_prob, next_state in trans_probs_by_action[action_idx]:
            weight = action_prob * state_prob
            achieved = check_goal_achieved(next_state, human_idx, goal)
            
            if achieved:
                achieved_contributions[data_idx] += weight  # weight * 1.0
            else:
                all_next_states.append(next_state)
                all_human_indices.append(human_idx)
                all_goals.append(goal)
                successor_mapping.append((data_idx, weight))
```

### Batched V_h^e Forward Pass

> ✅ **IMPLEMENTED** — ONE batched `v_h_e_target.forward_batch()` call for all successor states.

```python
if all_next_states:
    with torch.no_grad():
        v_h_e_all = self.networks.v_h_e_target.forward_batch(
            all_next_states, all_goals, all_human_indices,
            self.env, self.device
        )
        v_h_e_all = self.networks.v_h_e_target.apply_hard_clamp(v_h_e_all).squeeze()
```

### Aggregation with Policy Weighting

```python
# Initialize targets with achieved contributions
targets = torch.zeros(n_samples, device=self.device)
for data_idx, contrib in enumerate(achieved_contributions):
    targets[data_idx] += contrib

# Add V_h^e contributions from non-achieved successors
for succ_idx, (data_idx, weight) in enumerate(successor_mapping):
    targets[data_idx] += weight * gamma_h * v_h_e_all[succ_idx].item()
````
    
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

> ⚠️ **[NOT IMPLEMENTED]** The policy-weighted scatter-add aggregation below is not coded.
> The current implementation uses Python loops and does NOT weight by π_r(a|s).

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

> **Note:** The performance numbers below are theoretical based on the full design.
> The actual implementation is slower due to Python loop aggregation instead of scatter-add,
> but still much faster than fully naive per-state forward passes.

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

> ✅ **PARTIALLY IMPLEMENTED** — Batched forward passes work, but aggregation uses Python loops.

```python
# 1-2 forward passes total
all_states = collect_all_successors(batch)           # O(batch × actions × successors)
s_encoded = batch_tensorize_states(all_states)          # 1 encode pass
v_r_all = v_r_network.forward(*s_encoded)            # 1 forward pass
q_r_targets = scatter_add_aggregate(v_r_all, ...)    # O(n_successors) scatter ops
```

**Cost:** O(1) forward passes + O(n_successors) scatter operations

### Measured Speedup

> **Note:** These numbers are from the original design document and may not reflect
> actual performance of the current partial implementation.

| Configuration | Naive | Batched | Speedup |
|--------------|-------|---------|---------|
| 7×7 grid, 2 humans, 2 robots, batch=32 | ~120s/episode | ~3.5s/episode | **~34×** |

---

## Summary

The key techniques for efficient batched computation:

1. **Flatten variable-size data** into parallel lists with metadata indices ✅ IMPLEMENTED
2. **Batch encode** all states in one pass using `torch.cat` ✅ IMPLEMENTED (via `forward_batch`)
3. **Single forward pass** through neural networks ✅ IMPLEMENTED
4. **Scatter-add aggregation** to handle variable successors without padding ❌ NOT IMPLEMENTED
5. **Careful shape management** to avoid broadcasting bugs ✅ IMPLEMENTED

These techniques transform O(N×M×K) forward passes into O(1) forward passes, yielding 30-50× speedups for typical Phase 2 training configurations.

---

## Profiling

> ⚠️ **[NOT IMPLEMENTED]** The profiling instrumentation described below does not exist.
> The config options `profile_batching` and `profile_batching_interval` are not implemented.
> Use external profilers like `line_profiler` or `cProfile` instead.

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
---

## Alternative Modes: Disabling Encoders and Lookup Tables

The batched computation infrastructure supports two alternative modes that bypass or simplify the neural network encoding:

### Disabling Encoders (`use_encoders=False`)

For debugging or when working with very simple environments, you can disable the neural network encoding entirely:

```python
config = Phase2Config(
    use_encoders=False,  # Encoders return identity (flattened input)
)
```

**What this does:**
- State encoders (`MultiGridStateEncoder`, `AgentIdentityEncoder`) switch to identity mode
- `forward()` returns flattened concatenation of raw tensorized inputs instead of learned features
- Tensorization still happens (raw state → tensors) because MLP heads still need tensor inputs
- Output dimension changes to match raw input size (grid + global + agent + interactive features)

**Note:** If you want to skip tensorization entirely, use **lookup table networks** instead (see below). The `use_encoders=False` mode is specifically for debugging encoder networks while keeping the rest of the neural architecture.

**Use cases:**
- Debugging: isolate whether problems come from encoders vs. other components
- Baseline comparisons: pure tabular representation for small state spaces
- Async training pickle size: identity encoders are much smaller (~10MB vs ~130MB)

**Important:** When `use_encoders=False`, the encoder modules become `nn.Identity()` placeholders. This dramatically reduces pickle size for async training, which is important for Docker's shared memory limits.

### Lookup Table Networks (Tabular Mode)

For small state spaces or when interpretability is important, you can use dictionary-based lookup tables instead of neural networks:

```python
config = Phase2Config(
    use_lookup_tables=True,    # Enable lookup table mode
    use_lookup_q_r=True,       # Q_r as lookup table
    use_lookup_v_h_e=True,     # V_h^e as lookup table
    use_lookup_x_h=True,       # X_h as lookup table
    # Default values for unseen states
    lookup_default_q_r=-1.0,
    lookup_default_v_h_e=0.5,
    lookup_default_x_h=0.5,
)
```

**What this does:**
- Replaces neural network value functions with dictionary-based tables
- Each unique state gets its own entry (created lazily on first access)
- Values stored as `torch.nn.Parameter` for gradient tracking and optimizer compatibility
- **Completely bypasses tensorization** - states are hashed directly as Python objects
- No function approximation error—exact value storage

**When to use:**
- State space < 100K unique states
- Debugging and interpretability (inspect exact values)
- Baseline comparisons with neural approaches
- Environments where generalization to unseen states isn't needed

**Memory considerations:**
- Each Q_r entry: `num_action_combinations` floats (e.g., 16 for 2 robots × 4 actions)
- Each V_h^e entry: 1 float per (state, goal) pair
- Total size grows linearly with visited states

**API compatibility:**
- Lookup table networks have the same API as neural versions (`forward()`, `encode_and_forward()`)
- They accept optional `state_encoder` arguments for API compatibility but don't use them
- Can be mixed: some networks neural, others lookup tables

See `src/empo/nn_based/phase2/lookup/` for the lookup table implementations and `examples/lookup_table_phase2_demo.py` for usage examples.
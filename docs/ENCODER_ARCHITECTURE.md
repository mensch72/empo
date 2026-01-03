# Encoder Architecture for Neural Networks

This document explains the architecture of the neural network encoders used in the EMPO system, including Phase 1 (human policy priors) and Phase 2 (robot Q-functions).

## Overview

The neural systems use a modular architecture with base classes and domain-specific implementations.

---

## Phase 1: Human Policy Priors

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
  - **Compressed grid format** for efficient replay buffer storage (see below)
- **MultiGridGoalEncoder** - Encodes goal positions (not part of world state)
- **MultiGridQNetwork** - Combines state and goal encoders
- **MultiGridNeuralHumanPolicyPrior** - With multigrid-specific validation

**Terminology Note**: The state encoder performs **tensorization** (converting raw state tuples to tensors), not encoding in the neural network sense. The actual "encoding" (feature extraction via NN forward pass) happens in the Q-network's convolutional and MLP layers.

---

## Phase 2: Robot Q-Functions with Dual Encoder Architecture

Phase 2 trains five networks for the robot's empowerment-based decision making:
- **Q_r**: Robot action-value function
- **V_r**: Robot state-value function
- **V_h^e**: Human expected value function (for empowerment computation)
- **X_h**: Aggregate human goal-achievement ability
- **U_r**: Robot empowerment utility function

### Encoder Sharing Strategy

**Key Design Principle**: To avoid gradient conflicts and enable efficient learning, we use a **shared-plus-own encoder architecture**:

```
                    ┌─────────────────────────────────────────────────┐
                    │              SHARED ENCODERS                     │
                    │  (state_encoder, goal_encoder, agent_encoder)    │
                    │                                                  │
                    │     Trained ONLY by V_h^e loss                   │
                    │     (receives un-detached encoder outputs)       │
                    └─────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                          │
                    ▼                                          ▼
           DETACHED outputs to:                        UN-DETACHED to:
           - Q_r, V_r, X_h, U_r                        - V_h^e only
```

### Network-Specific Encoders

**Important**: All state encoders produce **agent-agnostic** representations. They encode the full world state without any query-agent-specific features. Agent identity is handled separately by `AgentIdentityEncoder` which encodes agent index, position on grid, and features.

**Q_r (Robot Q-function)**:
- Uses `shared_state_encoder` (detached) for general state features (agent-agnostic)
- Has `own_state_encoder` (trained with Q_r loss) for additional state features (also agent-agnostic)
- Concatenates both: `[shared_state_features, own_state_features]`
- Input dimension to MLP is doubled

**X_h (Aggregate Goal Ability)**:
- Uses `shared_agent_encoder` (detached) for general human identity features
- Has `own_agent_encoder` (trained with X_h loss) for ability-specific human features
- Concatenates: `[state_features, shared_agent_embedding, own_agent_embedding]`

**V_h^e, V_r, U_r**:
- Use only shared encoders
- V_h^e trains the shared encoders (no detach)
- V_r and U_r train only their MLP heads (shared encoders detached)

### Gradient Flow Diagram

```
                          TRAINING STEP
                               │
                               ▼
              ┌────────────────────────────────────┐
              │         Compute all losses         │
              │   (Q_r, V_r, V_h^e, X_h, U_r)      │
              └────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │       Single backward() call       │
              │       (all losses summed)          │
              └────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────────────┐
           │                   │                            │
           ▼                   ▼                            ▼
    ┌──────────────┐   ┌──────────────┐            ┌──────────────┐
    │   V_h^e      │   │  Q_r, V_r    │            │   X_h, U_r   │
    │  loss        │   │  losses      │            │   losses     │
    └──────────────┘   └──────────────┘            └──────────────┘
           │                   │                            │
           ▼                   ▼                            ▼
    Updates:             Updates:                    Updates:
    - shared encoders    - Q_r MLP head             - X_h/U_r MLP heads
    - V_h^e MLP head     - Q_r own_state_encoder    - X_h own_agent_encoder
                         - V_r MLP head
```

### Encoder Summary Table

| Network | Mode | Shared Encoders | Own Encoders | Gradient Flow |
|---------|------|-----------------|--------------|---------------|
| **V_h^e** | Neural | Creates & trains state, goal, agent | None (it IS the shared source) | Trains shared encoders + own MLP |
| **V_h^e** | Tabular | NullEncoders (output zeros) | None | Lookup table only |
| **Q_r** | Neural | state (detached) | own_state_encoder | Trains own encoder + MLP |
| **Q_r** | Tabular | None | None | Lookup table only |
| **X_h** | Neural | state (detached), agent (detached) | own_state_encoder, own_agent_encoder | Trains own encoders + MLP |
| **X_h** | Tabular | None | None | Lookup table only |
| **U_r** | Neural | state (detached) | own_state_encoder | Trains own encoder + MLP |
| **U_r** | Tabular | None | None | Lookup table only |
| **V_r** | Neural | state (detached) | own_state_encoder | Trains own encoder + MLP |
| **V_r** | Tabular | None | None | Lookup table only |
| **RND** | Neural | ALL (detached) × coefficients | None | Trains predictor MLP only |
| **RND** | Tabular | N/A (RND disabled in tabular mode) | N/A | N/A |

**Key points:**
- In **neural mode**, shared encoders provide frozen features (via `.detach()`), while own encoders learn network-specific features
- In **tabular mode**, lookup tables hash the state directly without any encoders
- **Only V_h^e** trains the shared encoders - all other networks use them in detached (frozen) mode
- **RND** uses concatenated features from ALL encoders (shared + all own encoders), each weighted by warmup coefficients
- This prevents gradient conflicts between networks with different objectives

### Why This Architecture?

1. **Gradient Conflict Avoidance**: Different networks have different objectives. Having V_h^e train shared encoders while others use detached outputs prevents conflicting gradient signals.

2. **Specialized Feature Learning**: Q_r needs state features optimized for robot action selection, while X_h needs agent features optimized for predicting human goal-achievement. Their own encoders can learn these specialized representations.

3. **Efficient Shared Learning**: V_h^e is trained on human Q-values from Phase 1, which provides a strong learning signal for general state/goal/agent representations that benefit all networks.

4. **Stability**: Detaching shared encoder outputs for most networks prevents loss scale differences from destabilizing the shared encoder training.

### RND (Random Network Distillation) Encoder Architecture

The RND module uses a special multi-encoder architecture to compute novelty scores for exploration:

**Input Features**: RND concatenates features from ALL state encoders:
1. Shared state encoder (from V_h^e)
2. X_h own state encoder
3. U_r own state encoder (if `u_r_use_network=True`)
4. Q_r own state encoder

```
          ┌─────────────────────────────────────────────────────────────────┐
          │                    RND Input Features                           │
          │ [shared_features | x_h_own | u_r_own (if enabled) | q_r_own]   │
          │                     × encoder_coefficients                      │
          └─────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │    Target    │   │  Predictor   │   │   Novelty    │
            │  (frozen)    │   │ (trainable)  │   │    Score     │
            └──────────────┘   └──────────────┘   └──────────────┘
```

**Warmup Coefficient Ramping**: Each encoder's features are multiplied by a coefficient that ramps from 0→1 during the warmup stage when that encoder is introduced:

| Warmup Stage | Encoder | Coefficient Ramp |
|--------------|---------|------------------|
| Stage 0 (V_h^e) | shared_state_encoder | 0→1 |
| Stage 1 (X_h) | x_h_own_state_encoder | 0→1 |
| Stage 2 (U_r) | u_r_own_state_encoder | 0→1 (if enabled) |
| Stage 3 (Q_r) | q_r_own_state_encoder | 0→1 |

**Why This Design?**:
1. **Smooth Transitions**: Coefficients ramp smoothly (0→1) as new encoders come online, avoiding abrupt changes in novelty distribution when training stages begin.
2. **Complete State Representation**: Using ALL encoder features gives RND a richer representation than any single encoder alone.
3. **No Gradient Conflicts**: All encoder features are detached (no gradients flow back to encoders from RND loss).
4. **Stability**: The coefficient ramping prevents the novelty metric from spiking when new encoders start training, which previously caused issues with exploration.

---

## Common Design Principles

### Design Principle: Unified State Encoding

The `MultiGridStateEncoder` encodes the **complete world state** as a single feature vector. This includes:

1. **Spatial grid information** (via CNN)
2. **Agent features** (via MLP)
3. **Interactive object features** (via MLP)
4. **Global world features**

Goals are encoded separately by `GoalEncoder` because they represent the agent's objective, not the world state itself.

**Split Tensorization**: For efficient training, tensorization is split between actor and trainer:
- **Actor** (at collection time): Computes expensive features (global, agent, interactive) + compressed grid
- **Trainer** (at training time): Decompresses grid in fully vectorized batch operations

See [BATCHED_COMPUTATION.md](BATCHED_COMPUTATION.md#split-tensorization-optimization) for the complete data flow.

### Design Principle: No Normalization

**All values are passed as raw integers/floats, NOT normalized.** This is intentional because:

1. **Absolute distances matter**: The number of steps to reach a goal depends on absolute grid distance
2. **Scale information is useful**: The network can learn that larger grids require different strategies
3. **Simplicity**: No need for denormalization or scaling factors during inference
4. **Policy transfer**: Using absolute coordinates enables loading policies trained on larger grids for use on smaller grids

### Cross-Grid Policy Loading

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

### Design Principle: Grid vs. List Encoding

Within the unified StateEncoder, two complementary strategies are used:

- **Grid-based encoding**: Captures the *spatial distribution* of objects and agents by color
- **List-based encoding**: Captures *detailed features* of individual entities

This separation enables:
1. Policy transfer across different entity counts
2. Rich feature encoding without exponential channel growth
3. Efficient CNN processing for spatial information

---

## MultiGridStateEncoder Structure

### Grid Channels

The state encoder is **agent-agnostic** - it encodes the full world state without any agent-specific perspective.

```
Total Channels = num_object_type_channels + 3 + num_colors
                 -----------------------   -   -----------
                       |                   |       |
                       |                   |       +-- Per-color agent channels (7 standard colors)
                       |                   +-- "Other objects" channels (3: overlappable, immobile, mobile)
                       +-- Object type channels (29 total, see OBJECT_TYPE_TO_CHANNEL)
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

### Compressed Grid Format (for Replay Buffer)

For off-policy learning, the full grid tensor (39 channels × H × W = 7644 bytes for 7×7) must be stored efficiently. The **compressed grid format** packs all grid information into a single `int32` per cell:

| Bits | Field | Description |
|------|-------|-------------|
| 0-4 | object_type | 0-28 standard types (see `OBJECT_TYPE_TO_CHANNEL`), 30=door, 31=key |
| 5-7 | object_color | 0-6 color index (for doors/keys) |
| 8-9 | object_state | 0-3 door state (none/open/closed/locked) |
| 10-12 | agent_color | 0-6 agent color, 7=no agent |
| 13-15 | magic_state | 0=none, 1-4=active+side, 5=inactive |
| 16-17 | other_category | 0=none, 1=overlappable, 2=immobile, 3=mobile |

**Storage**: 196 bytes (7×7×4) vs 7644 bytes = **39× smaller** per grid

**Key insight**: The compressed grid captures ALL static and dynamic grid information, so the trainer can reconstruct the full grid tensor **without access to world_model**. This is essential for off-policy learning where transitions may come from different episodes with different world layouts.

See [BATCHED_COMPUTATION.md](BATCHED_COMPUTATION.md#compressed-grid-format) for full implementation details.

### Agent Features (13 per agent, all raw)

Agent features are organized by color (agent-agnostic - no "query agent" concept in state encoding):

| Feature | Size | Description |
|---------|------|-------------|
| Position | 2 | Raw (x, y) coordinates |
| Direction | 4 | One-hot encoding (right, down, left, up) |
| Abilities | 2 | can_enter_magic_walls (0/1), can_push_rocks (0/1) |
| Carried object | 2 | (type_index, color_index), -1 if none |
| Status | 3 | paused (0/1), terminated (0/1), forced_action (-1 if none) |

Structure: Agents are grouped by color: `[blue_agents] + [red_agents] + ...`

Agent identity (for Q-functions that need to know which agent is querying) is handled separately by `AgentIdentityEncoder`.

### Interactive Object Features

All values are raw integers (not normalized).

**KillButton** (5 features): position (2), enabled (1), trigger_color (1), target_color (1)

**PauseSwitch** (6 features): position (2), enabled (1), is_on (1), toggle_color (1), target_color (1)

**DisablingSwitch** (6 features): position (2), enabled (1), is_on (1), toggle_color (1), target_type (1)

**ControlButton** (7 features): position (2), enabled (1), trigger_color (1), target_color (1), forced_action (1), awaiting_action (1)

Note: The `awaiting_action` flag persists across time steps.

## MultiGridGoalEncoder

Goals are encoded with raw coordinates (2 values): x, y

---

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

---

## Alternative Modes

### Identity Mode (`use_encoders=False`)

For debugging or when working with small state spaces, encoders can be switched to **identity mode**:

```python
config = Phase2Config(
    use_encoders=False,  # Encoders return identity (flattened input)
)
```

**What changes:**
- `MultiGridStateEncoder.forward()` returns flattened concatenation of inputs instead of CNN/MLP features
- `AgentIdentityEncoder.forward()` returns flattened+padded inputs instead of MLP features
- Tensorization (`tensorize_state()`) still happens because MLP heads need tensor inputs
- Output dimension automatically adjusts to match raw input size

**Note:** This mode still requires tensorization. To skip tensorization entirely (for small state spaces), use **lookup table networks** instead.

**Why use it:**
1. **Debugging**: Isolate whether problems come from encoder networks vs. other components
2. **Async training**: Identity encoders are much smaller when pickled (~10MB vs ~130MB), avoiding Docker shared memory limits (SIGBUS errors)
3. **Baseline**: Pure tabular representation for comparison

**Implementation detail**: When `use_encoders=False`, the encoder creates `nn.Identity()` placeholder modules instead of CNN/MLP layers. This is critical for pickle size in async training.

### Lookup Table Networks (Tabular Mode)

For small state spaces, all Phase 2 networks can use dictionary-based lookup tables instead of neural networks:

```python
config = Phase2Config(
    use_lookup_tables=True,
    use_lookup_q_r=True,       # Q_r(s, a) as lookup table
    use_lookup_v_h_e=True,     # V_h^e(s, g) as lookup table
    use_lookup_x_h=True,       # X_h(s) as lookup table (if x_h_use_network=True)
    use_lookup_u_r=True,       # U_r(s) as lookup table (if u_r_use_network=True)
    use_lookup_v_r=True,       # V_r(s) as lookup table (if v_r_use_network=True)
)
```

**How it works:**
- Each unique state gets a dictionary entry (created lazily on first access)
- States are hashed directly as Python objects - **no tensorization needed**
- Values stored as `torch.nn.Parameter` for gradient tracking
- Same API as neural networks (`forward()`, `encode_and_forward()`, `get_config()`)
- No function approximation error - exact value storage

**Trade-offs:**
- ✅ No approximation error, guaranteed convergence
- ✅ Interpretable - inspect exact values per state
- ❌ Memory grows linearly with visited states
- ❌ No generalization to unseen states

**When to use:**
- State space < 100K unique states
- Debugging and interpretability
- Baseline comparisons

See `src/empo/nn_based/phase2/lookup/` for implementations and [BATCHED_COMPUTATION.md](BATCHED_COMPUTATION.md#alternative-modes-disabling-encoders-and-lookup-tables) for more details.

# Exploration in Phase 2 Training

This document summarizes the exploration mechanisms in EMPO's Phase 2 training and discusses the ongoing challenge of achieving sufficient state space coverage.

## Overview

Phase 2 training uses **epsilon-greedy exploration** for both robot and human agents. During each action selection:
- With probability `epsilon`, the agent samples from an **exploration policy**
- With probability `1 - epsilon`, the agent follows its "actual" policy:
  - Humans are then modelled as following their goal-dependent policy prior that we consider to be part of the robot's belief system
  - Robots use (a frozen copy of) the learned policy.

Both `epsilon_r` (robot) and `epsilon_h` (human) decay over training according to configurable schedules so that finally the correct values will be learned (at least if `epsilon_r` and `epsilon_h` are decayed to zero).

## Action Selection Process (As Implemented)

This section describes exactly how human and robot actions are selected during `collect_transition()`.

### Order of Operations

Actions are sampled in a specific order to enable efficient computation:

1. **Sample human actions first** (`sample_human_actions`)
2. **Compute transition probabilities** for all robot actions (given those human actions)
3. **Sample robot action** (`sample_robot_action`) using those transition probabilities for curiosity

This order allows `transition_probabilities()` to be called only ONCE and reused for both curiosity-driven action selection and the replay buffer.

### Human Action Selection (`sample_human_actions`)

For each human agent `h` with assigned goal `g`:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUMAN ACTION SELECTION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Get base probabilities:                                     │
│     ├─ With probability epsilon_h (exploration):                │
│     │   ├─ If human_exploration_policy is None:                 │
│     │   │     → P(a) = 1/num_actions (uniform)                  │
│     │   └─ If human_exploration_policy is HumanPolicyPrior:     │
│     │         → P(a) = exploration_policy(state, h, goal)       │
│     │                                                           │
│     └─ With probability (1 - epsilon_h) (policy-based):         │
│           → P(a) = human_policy_prior(state, h, goal)           │
│                                                                 │
│  2. Apply curiosity bonus (if human_action_rnd enabled):        │
│     ├─ Get features: (state_feat, agent_feat) for human h       │
│     ├─ Compute novelty: novelty[a] for all actions              │
│     ├─ Clamp: novelty = max(0, novelty)                         │
│     ├─ Scale: scale[a] = exp(+bonus_coef_h * novelty[a])        │
│     ├─ Modify: P_eff(a) = P(a) * scale[a]                       │
│     └─ Renormalize: P_eff(a) = P_eff(a) / sum(P_eff)            │
│                                                                 │
│  3. Sample action from P_eff (or P if no curiosity)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key points:**
- **Curiosity applies to BOTH epsilon and policy branches** - this maximizes exploration benefit
- Human curiosity is **action-dependent**: novelty(state, human, action) - encourages trying actions that haven't been taken in this (state, human) context
- The multiplicative bonus `exp(+bonus * novelty)` increases probability of novel actions
- Human RND requires `use_rnd=True`, `use_human_action_rnd=True`, and `rnd_bonus_coef_h > 0`
- Features are computed in a **single batched call** for all humans, then indexed per-human

### Robot Action Selection (`sample_robot_action`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROBOT ACTION SELECTION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Get effective beta_r (may be ramped during warm-up)            │
│                                                                 │
│  With probability epsilon_r (exploration):                      │
│    1. Get base probabilities:                                   │
│       ├─ If robot_exploration_policy is None:                   │
│       │     → P(a) = 1/num_actions (uniform)                    │
│       ├─ If robot_exploration_policy is RobotPolicy:            │
│       │     → P(a) = policy(state) or sample directly           │
│       ├─ If robot_exploration_policy is callable:               │
│       │     → P(a) = policy(state, env)                         │
│       └─ If robot_exploration_policy is list/array:             │
│             → P(a) = fixed probabilities                        │
│                                                                 │
│    2. Apply curiosity bonus (if enabled & transition_probs):    │
│       ├─ Compute expected novelty for each action               │
│       ├─ Scale: scale[a] = exp(+bonus_coef_r * novelty[a])      │
│       ├─ Modify: P_eff(a) = P(a) * scale[a]                     │
│       └─ Renormalize                                            │
│                                                                 │
│    3. Sample from P_eff                                         │
│                                                                 │
│  With probability (1 - epsilon_r) (policy-based):               │
│    1. Compute Q-values: Q(s,a) = q_r_target.forward(state)      │
│                                                                 │
│    2. Apply curiosity bonus (if enabled & transition_probs):    │
│       ├─ Compute expected novelty for each action               │
│       ├─ Apply bonus (multiplicative on Q):                     │
│       │   Q_eff(a) = Q(a) * exp(-bonus_coef_r * novelty(a))     │
│       │   (Since Q < 0, smaller scale → Q closer to 0 → better) │
│       └─ Sample from Boltzmann policy over Q_eff                │
│                                                                 │
│    3. Otherwise: Sample from P(a) ∝ (-Q(a))^(-β_r)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key points:**
- **Curiosity applies to BOTH epsilon and policy branches** - this maximizes exploration benefit
- Robot curiosity is **state-dependent**: novelty(successor_state) - encourages visiting novel states
- Uses `q_r_target` (frozen copy) for stable action sampling
- In epsilon branch: multiplicative bonus `exp(+bonus * novelty)` on exploration probabilities
- In policy branch: multiplicative bonus `exp(-bonus * novelty)` on Q-values (preserves Q < 0)
- Robot RND requires `use_rnd=True` and `rnd_bonus_coef_r > 0`

### Comparison: Human vs Robot Curiosity

| Aspect | Human Curiosity | Robot Curiosity |
|--------|-----------------|-----------------|
| **What's novel** | (state, human, action) tuple | Successor state |
| **Network** | `human_rnd` (HumanActionRNDModule) | `rnd` (RNDModule) |
| **Config flag** | `use_human_action_rnd=True` | `use_rnd=True` |
| **Bonus coef** | `rnd_bonus_coef_h` | `rnd_bonus_coef_r` |
| **Applies to** | Both epsilon and policy branches | Both epsilon and policy branches |
| **Epsilon formula** | `P_eff(a) = P(a) * exp(+coef * novelty[a])` | `P_eff(a) = P(a) * exp(+coef * novelty[a])` |
| **Policy formula** | `P_eff(a) = P(a) * exp(+coef * novelty[a])` | `Q_eff(a) = Q(a) * exp(-coef * novelty)` |
| **Effect** | Higher novelty → higher P | Higher novelty → higher P |

## Configuration Parameters

In `Phase2Config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon_r_start` | 1.0 | Initial robot exploration rate |
| `epsilon_r_end` | 0.01 | Final robot exploration rate |
| `epsilon_r_decay_steps` | 10,000 | Steps over which epsilon_r decays |
| `epsilon_h_start` | 1.0 | Initial human exploration rate |
| `epsilon_h_end` | 0.01 | Final human exploration rate |
| `epsilon_h_decay_steps` | 10,000 | Steps over which epsilon_h decays |

The decay is linear: `epsilon = start - (start - end) * min(step / decay_steps, 1.0)`

Access current values via:
```python
config.get_epsilon_r(training_step)  # Current robot epsilon
config.get_epsilon_h(training_step)  # Current human epsilon
```

## Exploration Policies

### Default: Uniform Random

Without custom exploration policies, epsilon-exploration samples uniformly from all actions.

### Smart Exploration Policies

EMPO provides capability-aware exploration policies that bias toward useful actions.

*Remark: these policies are currently restricted to the SmallActions set for multigrid (still, left, right, forward) and should be extended for the full Actions set of 8 possible actions (including toggle, drop, etc.).*

#### `MultiGridMultiStepExplorationPolicy` (Recommended)

Located in `empo.learning_based.multigrid.phase2.robot_policy`.

This is a **non-Markovian** exploration policy that samples multi-step action sequences, enabling more directed spatial exploration than simple Markov policies. It can be used for both robots and humans.

```python
from empo.learning_based.multigrid.phase2 import MultiGridMultiStepExplorationPolicy

# For robots
robot_exploration = MultiGridMultiStepExplorationPolicy(
    agent_indices=[1],  # Robot at index 1
    sequence_probs={
        'still': 0.05,         # k times still
        'forward': 0.50,       # k times forward
        'left_forward': 0.18,  # turn left, then k times forward
        'right_forward': 0.18, # turn right, then k times forward
        'back_forward': 0.09,  # turn 180°, then k times forward
    },
    expected_k=2.0,  # Expected number of forward/still steps
)

# For humans (same class, different agent indices)
human_exploration = MultiGridMultiStepExplorationPolicy(
    agent_indices=[0],  # Human at index 0
    sequence_probs={'forward': 0.5, 'left_forward': 0.2, 'right_forward': 0.2, 'still': 0.1},
    expected_k=2.0,
)
```

**Sequence Types:**
- `'still'`: k times still (stay in place)
- `'forward'`: k times forward (move straight)
- `'left_forward'`: turn left, then k times forward
- `'right_forward'`: turn right, then k times forward
- `'back_forward'`: turn left twice (180°), then k times forward

**Key Features:**
- **Non-Markovian**: Samples multi-step sequences, enabling directed movement
- **Geometric k**: The number of steps k is drawn from a geometric distribution with configurable expected value
- **Per-sequence-type expected_k**: Can specify different expected_k for each sequence type:
  ```python
  exploration = MultiGridMultiStepExplorationPolicy(
      expected_k={
          'still': 1.0,         # Short waits
          'forward': 3.0,       # Longer straight runs
          'left_forward': 2.0,
          'right_forward': 2.0,
          'back_forward': 2.0,
      }
  )
  ```
- **Feasibility checking**: Only starts sequences if forward movement is possible in the target direction
- **Sequence cancellation**: Cancels ongoing sequence if forward becomes blocked (e.g., by another agent)
- **Dual interface**: Works as both `RobotPolicy` and `HumanPolicyPrior`
- **Capability-aware**: Uses `can_forward()` to account for agent capabilities

**Why Non-Markovian Exploration Helps:**

Standard Markovian exploration (sampling each action independently) tends to produce "random walk" behavior where agents frequently reverse direction. This results in √t coverage of the state space.

Multi-step sequences encourage more directed movement:
- Forward sequences encourage exploring in straight lines
- Turn+forward sequences explore new directions systematically
- The geometric distribution allows for variable-length runs

#### `MultiGridRobotExplorationPolicy` (Simple Markovian)

Located in `empo.learning_based.multigrid.phase2.robot_policy`.

A simpler Markovian exploration policy for robots.

```python
from empo.learning_based.multigrid.phase2 import MultiGridRobotExplorationPolicy

exploration = MultiGridRobotExplorationPolicy(
    action_probs=[0.1, 0.1, 0.2, 0.6],  # [still, left, right, forward]
    robot_agent_indices=[1]
)
```

Features:
- Biased toward forward movement (configurable probabilities), and has larger probability to turn right than left, in order to avoid random turning back and forth without movement.
- **Avoids attempting forward when blocked** (redistributes probability to other actions)
- Accounts for robot capabilities: can push rocks, can enter magic walls
- Uses `env.can_forward(state, agent_index)` internally

#### `MultiGridHumanExplorationPolicy` (Simple Markovian)

Located in `empo.human_policy_prior` (same file as `HeuristicPotentialPolicy`).

A simpler Markovian exploration policy for humans.

```python
from empo.human_policy_prior import MultiGridHumanExplorationPolicy

exploration = MultiGridHumanExplorationPolicy(
    world_model=env,  # Optional, can be set later via set_world_model()
    human_agent_indices=[0],
    action_probs=[0.1, 0.1, 0.2, 0.6]  # [still, left, right, forward]
)
```

Features:
- Inherits from `HumanPolicyPrior` (can be used anywhere a policy prior is expected)
- Biases toward forward movement (configurable probabilities)
- **Avoids attempting forward when blocked** for humans
- Accounts for human limitations: **cannot push rocks**, **cannot enter magic walls**
- Uses `env.can_forward(state, agent_index)` internally

### The `can_forward()` Method

All exploration policies rely on `MultiGridEnv.can_forward(state, agent_index)`:

```python
can_move = env.can_forward(state, agent_index=0)
```

This method checks whether an agent can move forward **in principle** (ignoring multi-agent conflicts):
- Bounds checking
- Empty cells → passable
- Overlappable objects (unsteady ground, buttons) → passable
- Blocks → passable (all agents can push)
- Rocks → only if `agent.can_push_rocks == True`
- Magic walls → only if `agent.can_enter_magic_walls == True` and wall is active

## Usage in Training

```python
from empo.learning_based.multigrid.phase2 import (
    MultiGridPhase2Trainer,
    MultiGridMultiStepExplorationPolicy,
)

# Configure exploration
config = Phase2Config(
    epsilon_r_start=1.0,
    epsilon_r_end=0.0,
    epsilon_r_decay_steps=50000,
    epsilon_h_start=1.0,
    epsilon_h_end=0.0,
    epsilon_h_decay_steps=50000,
)

# Create multi-step exploration policies (recommended)
robot_exploration = MultiGridMultiStepExplorationPolicy(
    agent_indices=[1],
    sequence_probs={'forward': 0.5, 'left_forward': 0.2, 'right_forward': 0.2, 'back_forward': 0.05, 'still': 0.05},
    expected_k={'forward': 3.0, 'left_forward': 2.0, 'right_forward': 2.0, 'back_forward': 2.0, 'still': 1.0},
)
human_exploration = MultiGridMultiStepExplorationPolicy(
    agent_indices=[0],
    sequence_probs={'forward': 0.5, 'left_forward': 0.2, 'right_forward': 0.2, 'back_forward': 0.05, 'still': 0.05},
    expected_k=2.0,
)

# Create trainer with exploration
trainer = MultiGridPhase2Trainer(
    env=env,
    config=config,
    robot_exploration_policy=robot_exploration,
    human_exploration_policy=human_exploration,
    # ... other parameters
)
```

## TensorBoard Monitoring

Exploration rates are logged under the `Exploration/` group:
- `Exploration/epsilon_r` - Robot exploration rate over time
- `Exploration/epsilon_h` - Human exploration rate over time
- `Exploration/unique_states_seen` - Count of unique states visited
- `Exploration/visit_count_distribution` - Histogram of visit counts per state

Both epsilon values appear side-by-side in TensorBoard for easy comparison.

### Interpreting the Visit Count Histogram

The `visit_count_distribution` histogram shows how many times each unique state has been visited. This is crucial for diagnosing exploration issues:

| Histogram Shape | Interpretation | Action |
|-----------------|----------------|--------|
| Many states at count 1-5, few at higher | Healthy exploration, hitting connectivity limits | Bonus is fine |
| Heavy tail: few states at 100+, most at 1 | Bottleneck/stuck in small region | **Increase curiosity bonus** |
| Flat/uniform counts | Good coverage but slow expansion | May need longer training |
| All states with similar high counts | Small reachable space, saturated | Expected for small envs |

**If exploration is stuck** (heavy-tailed distribution), try increasing the curiosity bonus coefficient:
- For count-based curiosity: `count_curiosity_bonus_coef_r` (try 2x current value)
- For RND: `rnd_bonus_coef_r` (try 2x current value)

### Theoretical Growth Rate of Unique States

For random walk / Markovian exploration on a graph, the number of unique states visited grows differently depending on structure:

- **1D lattice**: ~√t (recurrent, frequently revisits)
- **2D lattice**: ~t/log(t) (marginally recurrent)  
- **3D+ or transient**: ~t (rarely revisits)
- **Grid worlds with obstacles**: Typically √t due to bottlenecks and constrained movement

In constrained environments like MultiGrid, expect **√t growth** because:
- Movement is local (can only reach neighbors)
- Walls, obstacles, and bottlenecks constrain expansion
- The "frontier" of unexplored states grows as √t (diffusion)

## Known Challenge: State Space Coverage

**Even with aggressive exploration, achieving full state space coverage remains challenging.**

### Example: "Trivial" World Model

The `phase2_robot_policy_demo.py` uses a "trivial" world model which has only **256 possible states** (combinations of agent positions and orientations).

**Empirical observation**: Even with high exploration rates (`epsilon_r = epsilon_h = 1.0`), only about **75% of all 256 states** are visited after **250,000 training steps**.

### Why This Happens

Hypothesis: as long as the exploration strategies are essentially Markov processes that are independent of history, the resulting movement is a kind of Brownian motion whose "range" (std. dev. of positions) only grows roughly as sqrt(steps) even on an empty grid.

### Mitigation Strategies

Consider these approaches for better coverage:

1. **Curiosity-driven exploration**: Implement intrinsic rewards for visiting novel states. **Now implemented via RND (neural networks) and Count-Based Curiosity (tabular)** - see below.

2. **Prioritized experience replay**: Give higher priority to transitions from rare states (partial support via replay buffer). While this will not affect what states are visited, it can improve approximation of quantities in rarely visited regions.

3. **Ensemble environments**: Use `EnsembleWorldModelFactory` to train across multiple environment configurations simultaneously.

4. **Longer episodes**: Increase `max_steps` to allow deeper exploration within episodes.

## Curiosity-Driven Exploration

EMPO supports two approaches to curiosity-driven exploration:

| Approach | Best For | How It Works |
|----------|----------|--------------|
| **RND** (Random Network Distillation) | Neural network mode | Prediction error as novelty signal |
| **Count-Based Curiosity** | Tabular/lookup table mode | Visit counts → bonus = 1/√(n+1) |

Curiosity bonuses are applied to **both the epsilon exploration branch and the (1-epsilon) policy branch**. This maximizes the benefit of curiosity-driven exploration by biasing action selection toward novel states/actions regardless of which branch is taken.

### RND (Random Network Distillation)

EMPO supports **Random Network Distillation (RND)** for curiosity-driven exploration. RND provides an intrinsic motivation signal that biases the learned policy toward exploring states that have been seen less frequently during training.

### How RND Works

RND uses two networks:
- **Target network**: Fixed random weights (never trained)
- **Predictor network**: Trained to match target outputs

The **prediction error** serves as a novelty signal:
- **High error** → State is novel (rarely seen)
- **Low error** → State is familiar (frequently seen)

### Configuration

Enable RND in `Phase2Config`:

```python
config = Phase2Config(
    # Enable RND
    use_rnd=True,
    
    # RND network architecture
    rnd_feature_dim=64,        # Output dimension of RND networks
    rnd_hidden_dim=256,        # Hidden layer dimension
    
    # Robot curiosity (state-based novelty)
    rnd_bonus_coef_r=0.1,      # Robot curiosity bonus scale
    
    # Human curiosity (action-based novelty)
    use_human_action_rnd=True, # Enable per-action novelty for humans
    rnd_bonus_coef_h=0.1,      # Human curiosity bonus scale
    
    # Training parameters
    lr_rnd=1e-4,               # RND predictor learning rate
    rnd_weight_decay=1e-4,     # Weight decay for RND
    rnd_grad_clip=10.0,        # Gradient clipping
    
    # Normalization (recommended for stability)
    normalize_rnd=True,        # Normalize novelty by running mean/std
    rnd_normalization_decay=0.99,  # EMA decay for normalization stats
)
```

### How Curiosity Affects Action Selection

Curiosity bonuses are applied to **both** the epsilon exploration and policy-based branches to maximize exploration.

#### Robot Curiosity (State-Based Novelty)

Robot curiosity biases toward visiting novel successor states:

1. **During epsilon exploration**: Curiosity bonus is applied multiplicatively to the exploration policy probabilities:
   ```
   P_effective(a) = P_exploration(a) * exp(+rnd_bonus_coef_r * expected_novelty(s'))
   ```
   
   Higher novelty → larger scale factor → higher probability of selecting that action.
   This biases even random exploration toward novel states.

2. **During policy-based selection (1-epsilon)**: Curiosity bonus is applied multiplicatively to Q-values to preserve the power-law policy constraint (Q < 0):
   ```
   Q_effective(s, a) = Q_r(s, a) * exp(-rnd_bonus_coef_r * expected_novelty(s'))
   ```
   
   Since Q < 0 and exp(...) > 0, Q_effective remains negative.
   High novelty → smaller scale factor → Q_effective closer to 0 (better).
   This encourages exploration of novel states while preserving the power-law policy form.
   
   When `rnd_bonus_coef_r = 0`, we recover exactly the standard power-law policy `P(a) ∝ (-Q_r(s,a))^{-β_r}`.

The novelty for each action is computed using `transition_probabilities()` to determine expected next states, then computing RND prediction error for those states.

#### Human Curiosity (Action-Based Novelty)

Human curiosity (when `use_human_action_rnd=True`) biases toward novel (state, human, action) combinations:

1. **During epsilon exploration**: Curiosity bonus is applied to exploration policy probabilities:
   ```
   P_effective(a) = P_exploration(a) * exp(+rnd_bonus_coef_h * novelty(s, h, a))
   ```

2. **During policy-based selection (1-epsilon)**: Curiosity bonus is applied to prior probabilities:
   ```
   P_effective(a) = P_prior(a) * exp(+rnd_bonus_coef_h * novelty(s, h, a))
   ```

In both cases:
- Higher novelty → larger scale factor → higher effective probability
- This encourages humans to try actions they haven't taken in this (state, human) context
- The formula renormalizes so `sum(P_effective) = 1`

Human RND uses a separate `HumanActionRNDModule` that takes (state_features, agent_features) and outputs per-action novelty scores.

### TensorBoard Monitoring

When RND is enabled, additional metrics are logged:

**Robot RND (state-based):**
- `Loss/rnd` - RND predictor loss (grouped with other losses)
- `Exploration/rnd_raw_novelty_mean` - Raw novelty before normalization (watch this decrease!)
- `Exploration/rnd_raw_novelty_std` - Std of raw novelty in batch
- `Exploration/rnd_norm_running_mean` - Running mean used for normalization
- `Exploration/rnd_norm_running_std` - Running std used for normalization

**Human RND (action-based, when `use_human_action_rnd=True`):**
- `Loss/human_rnd` - Human RND predictor loss
- `Exploration/human_rnd_raw_novelty_mean` - Raw novelty across (state, human, action) tuples
- `Exploration/human_rnd_raw_novelty_std` - Std of raw novelty
- `Exploration/human_rnd_norm_running_mean` - Running mean for normalization
- `Exploration/human_rnd_norm_running_std` - Running std for normalization

**Interpreting the metrics:**
- `rnd_raw_novelty_mean` should **decrease** over training as the predictor learns to recognize states
- `Loss/rnd` should also decrease, tracking the predictor's learning progress
- `rnd_norm_running_*` are normalization parameters that adapt to keep normalized output around mean=0, std=1
- For human RND, similar trends should appear for `human_rnd_*` metrics

### RND Adaptive Learning Rate Diagnostics

When `rnd_use_adaptive_lr=True` is enabled (using RND as an uncertainty proxy for adaptive learning rates), additional detailed metrics are logged under `AdaptiveLR/`:

**Per-network metrics** (for each of q_r, v_h_e, x_h, etc.):

| Metric | Description |
|--------|-------------|
| `rnd_{net}_scale_mean` | Mean LR scale applied (after clamping) |
| `rnd_{net}_scale_std` | Std of LR scale before clamping |
| `rnd_{net}_scale_min` | Min LR scale in batch (before clamping) |
| `rnd_{net}_scale_max` | Max LR scale in batch (before clamping) |
| `rnd_{net}_mse_mean` | Mean raw RND MSE in batch |
| `rnd_{net}_mse_std` | Std of raw RND MSE in batch |
| `rnd_{net}_mse_min` | Min raw RND MSE in batch |
| `rnd_{net}_mse_max` | Max raw RND MSE in batch |
| `rnd_{net}_frac_clamped_low` | Fraction hitting min clamp |
| `rnd_{net}_frac_clamped_high` | Fraction hitting max clamp |
| `rnd_{net}_running_mean` | Running mean used for normalization |

**What to look for:**

| Observation | Diagnosis | Suggested Action |
|-------------|-----------|------------------|
| `mse_std ≈ 0` | States not differentiated by RND | Check state encoder output diversity; try larger RND hidden_dim |
| `mse_min ≈ mse_max` | All states have similar novelty | RND may be too simple or state features too low-dimensional |
| `scale_std ≈ 0` | Adaptive LR not providing differentiation | Expected if all states equally familiar/novel |
| `frac_clamped_low` high | Many states very familiar | Consider lowering `rnd_adaptive_lr_min` |
| `frac_clamped_high` high | Many states very novel | Consider raising `rnd_adaptive_lr_max` or increasing RND training |
| `scale_mean ≈ 1.0` always | Expected behavior | Scale is normalized by running mean, so mean ≈ 1 by design |

**Healthy signs:**
- `mse_std > 0` (some variance in novelty across batch)
- `mse_min << mse_max` (spread indicates state differentiation)
- Decreasing `mse_mean` over time (predictor learning)
- Low clamping fractions (values within reasonable range)

**Warning signs:**
- `mse_std` very small but `mse_mean` non-zero → RND can't distinguish states
- `mse_mean` not decreasing → RND predictor not learning (check `lr_rnd`)
- High `frac_clamped_*` → Need to adjust clamp bounds or scale parameter

### Example Usage

```python
from empo.learning_based.phase2 import Phase2Config
from empo.learning_based.multigrid.phase2 import train_multigrid_phase2

config = Phase2Config(
    use_rnd=True,
    rnd_bonus_coef_r=0.1,
    normalize_rnd=True,
    # ... other parameters
)

q_r, networks, history, trainer = train_multigrid_phase2(
    world_model=env,
    config=config,
    # ... other parameters
)
```

### Computational Overhead

RND adds overhead during:
- **Action selection**: Computing novelty for successor states requires calling `transition_probabilities()` and RND forward pass for each action
- **Training**: One additional backward pass per training step for RND loss

For small action spaces (typical in MultiGrid), this overhead is acceptable. For large action spaces, consider:
- Using curiosity only during epsilon exploration (not on Q-values)
- Sampling a subset of actions for novelty computation

### Design Notes

- RND is trained from the beginning (included in active networks from step 0)
- **Multi-encoder architecture**: Robot RND uses concatenated features from ALL state encoders:
  - Shared state encoder (from V_h^e)
  - X_h's own state encoder
  - U_r's own state encoder (if `u_r_use_network=True`)
  - Q_r's own state encoder
- **Human RND architecture**: Uses separate `HumanActionRNDModule` with:
  - State features from V_h^e's shared state encoder
  - Agent features from V_h^e's agent encoder
  - Outputs per-action novelty scores: (batch_size, num_actions)
- **Warmup coefficients**: Each encoder's features are multiplied by a coefficient that ramps 0→1 during the warmup stage when that encoder is introduced. This provides smooth transitions as new encoders come online.
- All encoder outputs are detached (RND trains only its predictor network, not the encoders)
- Normalization prevents bonus scale drift during training
- **Standard RND methodology**: Curiosity only affects the learned policy (via Q-value or prior probability bonuses), not epsilon exploration. This follows the original RND paper where intrinsic reward modifies the learned value function, while epsilon-greedy exploration remains separate.

See [docs/ENCODER_ARCHITECTURE.md](ENCODER_ARCHITECTURE.md) for details on the multi-encoder design.
See [docs/plans/curiosity.md](plans/curiosity.md) for detailed design rationale and alternative approaches considered.

## Count-Based Curiosity (Tabular Mode)

For **lookup table (tabular) mode**, EMPO provides a simpler count-based curiosity mechanism. This is more appropriate than RND when:
- States are exactly hashable
- The state space is enumerable
- No generalization across "similar" states is needed

### How Count-Based Curiosity Works

The system maintains a dictionary mapping states to visit counts:

```
bonus(s) = scale / √(visits[s] + 1)
```

- **Novel states** (0 visits): `bonus = scale / 1 = scale`
- **Visited states**: Bonus decreases with √visits
- **Frequently visited**: Bonus approaches 0

Alternatively, a **UCB-style** bonus is available:
```
bonus(s) = scale × √(log(total_visits) / (visits[s] + 1))
```

### Configuration

Enable count-based curiosity in `Phase2Config`:

```python
config = Phase2Config(
    # Enable lookup tables (required for count-based curiosity)
    use_lookup_tables=True,
    
    # Enable count-based curiosity
    use_count_based_curiosity=True,
    
    # Bonus parameters
    count_curiosity_scale=1.0,           # Bonus scale factor
    count_curiosity_use_ucb=False,       # Use UCB-style bonus instead of 1/√n
    
    # Curiosity bonus coefficients (how much bonus affects action selection)
    count_curiosity_bonus_coef_r=0.1,    # Robot curiosity bonus weight
    count_curiosity_bonus_coef_h=0.1,    # Human curiosity bonus weight
)
```

### TensorBoard Monitoring

When count-based curiosity is enabled, these metrics are logged:
- `Exploration/count_curiosity_unique_states` - Number of unique states visited
- `Exploration/count_curiosity_total_visits` - Total visit count
- `Exploration/count_curiosity_mean_visits` - Mean visits per unique state
- `Exploration/count_curiosity_max_visits` - Max visits to any single state
- `Exploration/count_curiosity_coverage_ratio` - States with >1 visit / total unique states

**Interpreting the metrics:**
- `unique_states` should grow over training (discovering new states)
- `mean_visits` indicates how uniformly exploration is distributed
- `coverage_ratio` shows what fraction of discovered states have been revisited

**What to look for:**

| Observation | Diagnosis | Suggested Action |
|-------------|-----------|------------------|
| `unique_states` plateaus early | Exploration stuck or state space exhausted | Increase `count_curiosity_scale` or check connectivity |
| `unique_states` grows as √t | Normal random walk behavior | Expected for constrained environments |
| `unique_states` grows linearly | Excellent exploration | Curiosity bonus is effective |
| `mean_visits` very high, `unique_states` low | Stuck revisiting same states | **Increase `count_curiosity_bonus_coef_r`** |
| `max_visits >> mean_visits` (heavy skew) | Bottleneck states | Some states unavoidably frequent (e.g., start state) |
| `coverage_ratio` near 1.0 | Good revisitation coverage | Healthy exploration pattern |

## General TensorBoard Diagnostics Guide

This section summarizes how to diagnose exploration issues using TensorBoard metrics.

### Quick Diagnostic Checklist

1. **Is exploration happening?**
   - Check `Exploration/epsilon_r` and `Exploration/epsilon_h` are non-zero
   - Check `Exploration/unique_states_seen` is growing

2. **Is exploration effective?**
   - Check `Exploration/visit_count_distribution` histogram for heavy tails (bad) vs spread (good)
   - For RND: Check `Exploration/rnd_raw_novelty_mean` is decreasing

3. **Is curiosity helping?**
   - For RND: Check `Loss/rnd` is decreasing (predictor learning)
   - For count-based: Check `unique_states` growth rate

4. **Is adaptive LR working?** (if enabled)
   - Check `AdaptiveLR/rnd_{net}_scale_std > 0` (variance exists)
   - Check `AdaptiveLR/rnd_{net}_mse_std > 0` (states differentiated)

### Common Problems and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `unique_states_seen` stuck | Exploration too weak or environment constrained | ↑ epsilon decay steps, ↑ curiosity bonus |
| `visit_count_distribution` heavily skewed | Random walk behavior, bottlenecks | ↑ curiosity bonus, use smart exploration policies |
| `rnd_raw_novelty_mean` not decreasing | RND predictor not learning | ↑ `lr_rnd`, check RND network size |
| `rnd_raw_novelty_std` ≈ 0 | State features not discriminative | Check encoder architecture, ↑ `hidden_dim` |
| All adaptive LR scales ≈ 1.0 | Expected (normalized), check std | Look at `scale_std`, not `scale_mean` |
| High `frac_clamped_*` | Clamp bounds too tight | Widen `rnd_adaptive_lr_min`/`max` |

### Metric Dependencies

Understanding which metrics affect which:

```
Environment Structure
        │
        ▼
┌───────────────────┐
│  State Features   │──────────────────────────────────────┐
│  (from encoders)  │                                      │
└───────────────────┘                                      │
        │                                                  │
        ▼                                                  ▼
┌───────────────────┐                           ┌───────────────────┐
│   RND Networks    │                           │  Value Networks   │
│  target/predictor │                           │   Q_r, V_h^e...   │
└───────────────────┘                           └───────────────────┘
        │                                                  │
        ▼                                                  │
┌───────────────────┐                                      │
│  RND MSE (novelty)│                                      │
│  mse_mean/std/... │                                      │
└───────────────────┘                                      │
        │                                                  │
        ├───────────────────────────────────────┐          │
        ▼                                       ▼          │
┌───────────────────┐               ┌───────────────────┐  │
│ Curiosity Bonus   │               │ Adaptive LR Scale │  │
│ (action selection)│               │ (gradient scaling)│──┘
└───────────────────┘               └───────────────────┘
        │                                       │
        ▼                                       ▼
┌───────────────────┐               ┌───────────────────┐
│  Exploration      │               │  Learning Speed   │
│  (unique_states)  │               │  (per-state)      │
└───────────────────┘               └───────────────────┘
```

If `mse_std ≈ 0`, both curiosity bonus AND adaptive LR will be ineffective because all states look the same to RND.

### Example Usage

```python
from empo.learning_based.phase2 import Phase2Config, CountBasedCuriosity

# Create config for tabular mode with curiosity
config = Phase2Config(
    use_lookup_tables=True,
    use_count_based_curiosity=True,
    count_curiosity_scale=1.0,
    count_curiosity_bonus_coef_r=0.1,
)

# The trainer will automatically create and use CountBasedCuriosity
# You can also create it directly:
curiosity = CountBasedCuriosity(scale=1.0, use_ucb=False)
curiosity.record_visit(state)
bonus = curiosity.get_bonus(state)
```

### Convenience Flag in Demos

The `phase2_robot_policy_demo.py` provides a `--curious` flag that automatically selects the appropriate curiosity mechanism:

```bash
# Neural network mode → uses RND
python phase2_robot_policy_demo.py --curious

# Tabular mode → uses count-based curiosity  
python phase2_robot_policy_demo.py --tabular --curious
```

Similarly, `lookup_table_phase2_demo.py` has a `--curiosity` flag:

```bash
python lookup_table_phase2_demo.py --curiosity
```

## API Reference

### Phase2Config Exploration Methods

```python
config.get_epsilon_r(training_step: int) -> float
config.get_epsilon_h(training_step: int) -> float
```

### MultiGridEnv

```python
env.can_forward(state, agent_index: int) -> bool
```

### CountBasedCuriosity Class

```python
from empo.learning_based.phase2 import CountBasedCuriosity

curiosity = CountBasedCuriosity(
    scale: float = 1.0,        # Bonus scale factor
    use_ucb: bool = False,     # Use UCB-style bonus
    min_bonus: float = 0.0     # Minimum bonus value
)

# Record state visits
curiosity.record_visit(state: Hashable)
curiosity.record_visits(states: List[Hashable])

# Get exploration bonuses
bonus = curiosity.get_bonus(state: Hashable) -> float
bonuses = curiosity.get_bonuses(states: List[Hashable]) -> List[float]

# Get statistics
stats = curiosity.get_statistics() -> Dict[str, float]
# Returns: unique_states, total_visits, mean_visits, max_visits, coverage_ratio

# Save/restore
state_dict = curiosity.state_dict()
curiosity.load_state_dict(state_dict)
curiosity.reset()
```

### Exploration Policy Classes

```python
# Robot exploration
MultiGridRobotExplorationPolicy(
    action_probs: List[float] = [0.1, 0.1, 0.2, 0.6],
    robot_agent_indices: List[int] = None
)
.reset(world_model)
.sample(state) -> Tuple[int, ...]

# Human exploration (inherits from HumanPolicyPrior)
MultiGridHumanExplorationPolicy(
    world_model: WorldModel = None,
    human_agent_indices: List[int] = None,
    action_probs: List[float] = [0.1, 0.1, 0.2, 0.6]
)
.set_world_model(world_model)
.sample(state, human_agent_index, goal) -> int
.__call__(state, human_agent_index, goal) -> np.ndarray  # Returns action distribution
```

## See Also

- [Phase 2 Warm-up Design](WARMUP_DESIGN.md) - How beta_r warm-up affects early exploration
- [API Reference](API.md) - Full API documentation
- [examples/phase2_robot_policy_demo.py](../examples/phase2_robot_policy_demo.py) - Working example with exploration

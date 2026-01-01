# Exploration in Phase 2 Training

This document summarizes the exploration mechanisms in EMPO's Phase 2 training and discusses the ongoing challenge of achieving sufficient state space coverage.

## Overview

Phase 2 training uses **epsilon-greedy exploration** for both robot and human agents. During each action selection:
- With probability `epsilon`, the agent samples from an **exploration policy**
- With probability `1 - epsilon`, the agent follows its "actual" policy:
  - Humans are then modelled as following their goal-dependent policy prior that we consider to be part of the robot's belief system
  - Robots use (a frozen copy of) the learned policy.

Both `epsilon_r` (robot) and `epsilon_h` (human) decay over training according to configurable schedules so that finally the correct values will be learned (at least if `epsilon_r` and `epsilon_h` are decayed to zero).

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

*Remark: both are currently restricted to the SmallActions set for multigrid (still, left, right, forward) and should be extended for the full Actions set of 8 possible actions (including toggle, drop, etc.).*

#### `MultiGridRobotExplorationPolicy`

Located in `empo.nn_based.multigrid.phase2.robot_policy`.

```python
from empo.nn_based.multigrid.phase2 import MultiGridRobotExplorationPolicy

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

#### `MultiGridHumanExplorationPolicy`

Located in `empo.human_policy_prior` (same file as `HeuristicPotentialPolicy`).

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

Both exploration policies rely on `MultiGridEnv.can_forward(state, agent_index)`:

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
from empo.nn_based.multigrid.phase2 import (
    MultiGridPhase2Trainer,
    MultiGridRobotExplorationPolicy,
)
from empo.human_policy_prior import MultiGridHumanExplorationPolicy

# Configure exploration
config = Phase2Config(
    epsilon_r_start=1.0,
    epsilon_r_end=0.0,
    epsilon_r_decay_steps=50000,
    epsilon_h_start=1.0,
    epsilon_h_end=0.0,
    epsilon_h_decay_steps=50000,
)

# Create smart exploration policies
robot_exploration = MultiGridRobotExplorationPolicy(
    action_probs=[0.1, 0.1, 0.2, 0.6],
    robot_agent_indices=[1]
)
human_exploration = MultiGridHumanExplorationPolicy(
    action_probs=[0.1, 0.1, 0.2, 0.6]
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

Both appear side-by-side in TensorBoard for easy comparison.

## Known Challenge: State Space Coverage

**Even with aggressive exploration, achieving full state space coverage remains challenging.**

### Example: "Trivial" World Model

The `phase2_robot_policy_demo.py` uses a "trivial" world model which has only **256 possible states** (combinations of agent positions and orientations).

**Empirical observation**: Even with high exploration rates (`epsilon_r = epsilon_h = 1.0`), only about **75% of all 256 states** are visited after **250,000 training steps**.

### Why This Happens

Hypothesis: as long as the exploration strategies are essentially Markov processes that are independent of history, the resulting movement is a kind of Brownian motion whose "range" (std. dev. of positions) only grows roughly as sqrt(steps) even on an empty grid.

### Mitigation Strategies

Consider these approaches for better coverage:

1. **Curiosity-driven exploration**: Implement intrinsic rewards for visiting novel states. **Now implemented via RND** - see below.

2. **Prioritized experience replay**: Give higher priority to transitions from rare states (partial support via replay buffer). While this will not affect what states are visited, it can improve approximation of quantities in rarely visited regions.

3. **Ensemble environments**: Use `EnsembleWorldModelFactory` to train across multiple environment configurations simultaneously.

4. **Longer episodes**: Increase `max_steps` to allow deeper exploration within episodes.

## Curiosity-Driven Exploration (RND)

EMPO supports **Random Network Distillation (RND)** for curiosity-driven exploration. RND provides an intrinsic motivation signal that biases action selection toward states that have been seen less frequently during training.

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
    
    # Curiosity bonus coefficients
    rnd_bonus_coef_r=0.1,      # Robot curiosity bonus scale
    rnd_bonus_coef_h=0.1,      # Human curiosity bonus scale (reserved)
    
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

When RND is enabled:

1. **During epsilon exploration**: Instead of uniform random actions, robot samples actions weighted by expected novelty of successor states.

2. **During policy-based selection**: Curiosity bonus is applied multiplicatively to Q-values to preserve the power-law policy constraint (Q < 0):
   ```
   Q_effective(s, a) = Q_r(s, a) * exp(-rnd_bonus_coef_r * expected_novelty(s'))
   ```
   
   Since Q < 0 and exp(...) > 0, Q_effective remains negative.
   High novelty → smaller scale factor → Q_effective closer to 0 (better).
   This encourages exploration of novel states while preserving the power-law policy form.

The novelty for each action is computed using `transition_probabilities()` to determine expected next states, then computing RND prediction error for those states.

### TensorBoard Monitoring

When RND is enabled, additional metrics are logged:
- `Loss/rnd` - RND predictor loss (grouped with other losses)
- `Exploration/rnd_raw_novelty_mean` - Raw novelty before normalization (watch this decrease!)
- `Exploration/rnd_raw_novelty_std` - Std of raw novelty in batch
- `Exploration/rnd_norm_running_mean` - Running mean used for normalization
- `Exploration/rnd_norm_running_std` - Running std used for normalization

**Interpreting the metrics:**
- `rnd_raw_novelty_mean` should **decrease** over training as the predictor learns to recognize states
- `Loss/rnd` should also decrease, tracking the predictor's learning progress
- `rnd_norm_running_*` are normalization parameters that adapt to keep normalized output around mean=0, std=1

### Example Usage

```python
from empo.nn_based.phase2 import Phase2Config
from empo.nn_based.multigrid.phase2 import train_multigrid_phase2

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
- The predictor uses the shared state encoder's output (detached from gradient computation)
- Normalization prevents bonus scale drift during training
- A small uniform component (10%) is added to curiosity-weighted exploration to ensure baseline coverage

See [docs/plans/curiosity.md](plans/curiosity.md) for detailed design rationale and alternative approaches considered.

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

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

1. **Curiosity-driven exploration**: Implement intrinsic rewards for visiting novel states (not yet implemented in EMPO).

2. **Prioritized experience replay**: Give higher priority to transitions from rare states (partial support via replay buffer). While this will not affect what states are visited, it can improve approximation of quantities in rarely visited regions.

3. **Ensemble environments**: Use `EnsembleWorldModelFactory` to train across multiple environment configurations simultaneously.

4. **Longer episodes**: Increase `max_steps` to allow deeper exploration within episodes.

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

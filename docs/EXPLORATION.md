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

Both approaches follow the **standard RND methodology**: curiosity affects only the **(1-epsilon) learned policy portion** via Q-value bonuses, not the epsilon exploration. The epsilon exploration always uses the configured exploration policy (or uniform random).

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

Following the **standard RND methodology** (Burda et al. 2018), curiosity affects only the **(1-epsilon) learned policy portion**, not the epsilon exploration:

1. **During epsilon exploration**: The configured `robot_exploration_policy` is used unchanged (or uniform random if none set). Curiosity does NOT affect this.

2. **During policy-based selection (1-epsilon)**: Curiosity bonus is applied multiplicatively to Q-values to preserve the power-law policy constraint (Q < 0):
   ```
   Q_effective(s, a) = Q_r(s, a) * exp(-rnd_bonus_coef_r * expected_novelty(s'))
   ```
   
   Since Q < 0 and exp(...) > 0, Q_effective remains negative.
   High novelty → smaller scale factor → Q_effective closer to 0 (better).
   This encourages exploration of novel states while preserving the power-law policy form.
   
   When `rnd_bonus_coef_r = 0`, we recover exactly the standard power-law policy `P(a) ∝ (-Q_r(s,a))^{-β_r}`.

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
- **Multi-encoder architecture**: RND uses concatenated features from ALL state encoders:
  - Shared state encoder (from V_h^e)
  - X_h's own state encoder
  - U_r's own state encoder (if `u_r_use_network=True`)
  - Q_r's own state encoder
- **Warmup coefficients**: Each encoder's features are multiplied by a coefficient that ramps 0→1 during the warmup stage when that encoder is introduced. This provides smooth transitions as new encoders come online.
- All encoder outputs are detached (RND trains only its predictor network, not the encoders)
- Normalization prevents bonus scale drift during training
- **Standard RND methodology**: Curiosity only affects the learned policy (via Q-value bonuses), not epsilon exploration. This follows the original RND paper where intrinsic reward modifies the learned value function, while epsilon-greedy exploration remains separate.

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

### Example Usage

```python
from empo.nn_based.phase2 import Phase2Config, CountBasedCuriosity

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
from empo.nn_based.phase2 import CountBasedCuriosity

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

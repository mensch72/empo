# EMPO API Reference

This document provides a comprehensive API reference for the `empo` package and related modules.

## Core Module: `empo`

The main package providing empowerment-based policy modeling for multi-agent systems.

### WorldModel (empo.world_model)

Abstract base class for gymnasium environments with explicit state management.

```python
from empo import WorldModel
```

#### Methods

| Method | Description |
|--------|-------------|
| `get_state()` | Get hashable representation of complete environment state |
| `set_state(state)` | Restore environment to a specific state |
| `transition_probabilities(state, actions)` | Compute exact transition probabilities |
| `initial_state()` | Get initial state without permanently resetting |
| `is_terminal(state=None)` | Check if a state is terminal |
| `get_dag(return_probabilities=False)` | Compute DAG of reachable states |
| `get_dag_parallel(return_probabilities=False, num_workers=None)` | Parallel DAG computation |
| `plot_dag(...)` | Visualize DAG with Graphviz |

#### Example

```python
from empo import WorldModel

class MyEnv(WorldModel):
    def get_state(self):
        return (self.position, self.direction)
    
    def set_state(self, state):
        self.position, self.direction = state
    
    def transition_probabilities(self, state, actions):
        # Return list of (probability, successor_state) tuples
        return [(1.0, new_state)]
```

---

### PossibleGoal (empo.possible_goal)

Abstract base class for defining possible goals in human behavior modeling.

```python
from empo import PossibleGoal
```

#### Methods (Abstract)

| Method | Description |
|--------|-------------|
| `is_achieved(state) -> int` | Returns 1 if goal achieved, 0 otherwise |
| `__hash__() -> int` | Hash for use as dictionary key |
| `__eq__(other) -> bool` | Equality comparison |

#### Example

```python
from empo import PossibleGoal

class ReachCell(PossibleGoal):
    def __init__(self, world_model, target_pos):
        super().__init__(world_model)
        self.target_pos = target_pos
    
    def is_achieved(self, state):
        agent_pos = (state[1][0][0], state[1][0][1])
        return 1 if agent_pos == self.target_pos else 0
    
    def __hash__(self):
        return hash(self.target_pos)
    
    def __eq__(self, other):
        return isinstance(other, ReachCell) and self.target_pos == other.target_pos
```

---

### PossibleGoalGenerator (empo.possible_goal)

Abstract base class for enumerating possible goals.

```python
from empo import PossibleGoalGenerator
```

#### Methods (Abstract)

| Method | Description |
|--------|-------------|
| `generate(state, human_agent_index)` | Yields (goal, weight) pairs |

#### Example

```python
from empo import PossibleGoalGenerator

class AllCellsGenerator(PossibleGoalGenerator):
    def generate(self, state, agent_idx):
        for x in range(self.world_model.width):
            for y in range(self.world_model.height):
                goal = ReachCell(self.world_model, (x, y))
                weight = 1.0 / (self.world_model.width * self.world_model.height)
                yield goal, weight
```

---

### HumanPolicyPrior (empo.human_policy_prior)

Abstract base class for human policy priors.

```python
from empo import HumanPolicyPrior
```

#### Methods

| Method | Description |
|--------|-------------|
| `__call__(state, agent_index, goal=None)` | Get action distribution (numpy array) |
| `sample(state, agent_index=None, goal=None)` | Sample action(s) |

---

### TabularHumanPolicyPrior (empo.human_policy_prior)

Concrete implementation using lookup tables.

```python
from empo import TabularHumanPolicyPrior
```

Created by `compute_human_policy_prior()` function.

---

### compute_human_policy_prior (empo.backward_induction)

Compute human policy prior via backward induction.

```python
from empo import compute_human_policy_prior
```

#### Signature

```python
def compute_human_policy_prior(
    world_model: WorldModel,
    human_agent_indices: List[int],
    possible_goal_generator: PossibleGoalGenerator,
    believed_others_policy: Optional[Callable] = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    parallel: bool = False,
    num_workers: Optional[int] = None,
    level_fct: Optional[Callable] = None,
    return_V_values: bool = False
) -> TabularHumanPolicyPrior
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `world_model` | WorldModel | Environment with state management |
| `human_agent_indices` | List[int] | Agent indices to compute policies for |
| `possible_goal_generator` | PossibleGoalGenerator | Generator for goals |
| `believed_others_policy` | Callable, optional | Belief about other agents |
| `beta` | float | Inverse temperature (higher = more deterministic) |
| `gamma` | float | Discount factor |
| `parallel` | bool | Use multiprocessing |
| `num_workers` | int, optional | Number of parallel workers |
| `level_fct` | Callable, optional | State level function for fast ordering |
| `return_V_values` | bool | Also return value function |

#### Example

```python
from empo import compute_human_policy_prior

policy_prior = compute_human_policy_prior(
    world_model=env,
    human_agent_indices=[0, 1],
    possible_goal_generator=goal_generator,
    beta=10.0,
    parallel=True
)

# Use the policy prior
action_dist = policy_prior(state, agent_idx=0, goal=my_goal)
sampled_action = policy_prior.sample(state, agent_idx=0, goal=my_goal)
```

---

## Environment Module: `src.envs`

Custom MultiGrid environments for human-robot collaboration.

### SmallOneOrTwoChambersMapEnv

Simplified environment for tractable computation.

```python
from src.envs import SmallOneOrTwoChambersMapEnv
```

#### Properties

| Property | Value |
|----------|-------|
| Grid size | 10×9 |
| Human agents | 2 (yellow) |
| Robot agents | 1 (grey) |
| Max steps | 8 |
| Action set | SmallActions (still, left, right, forward) |

#### Example

```python
from src.envs import SmallOneOrTwoChambersMapEnv

env = SmallOneOrTwoChambersMapEnv()
obs = env.reset()

# Get DAG for planning
states, state_to_idx, successors = env.get_dag()
print(f"State space size: {len(states)}")

# Run episode
done = False
while not done:
    actions = [env.action_space.sample() for _ in env.agents]
    obs, rewards, done, info = env.step(actions)
```

---

## Vendored MultiGrid Extensions

The vendored `gym_multigrid` in `vendor/multigrid/` extends the original with:

### New Object Types

| Type | Description |
|------|-------------|
| `Block` | Pushable by any agent |
| `Rock` | Pushable only by authorized agents (based on `can_push_rocks`) |
| `UnsteadyGround` | Agents may stumble (stochastic movement) |
| `MagicWall` | Probabilistic entry from one direction |

### Agent Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `can_push_rocks` | bool | Whether agent can push rocks |
| `can_enter_magic_walls` | bool | Whether agent can attempt magic wall entry |
| `on_unsteady_ground` | bool | (Derived) Currently on unsteady ground |

### Map Specification

Environments can be defined using ASCII map strings:

```python
MAP = """
WeWeWeWeWe
We..Ay..We
We..Ae..We
WeWeWeWeWe
"""

class MyEnv(MultiGridEnv):
    def __init__(self):
        super().__init__(
            map=MAP,
            max_steps=100,
            can_push_rocks='e'  # Grey agents can push rocks
        )
```

#### Cell Codes

| Code | Object |
|------|--------|
| `..` | Empty |
| `We` | Grey wall |
| `Ay` | Yellow agent |
| `Ae` | Grey agent |
| `Ro` | Rock |
| `Bl` | Block |
| `Un` | Unsteady ground |
| `Mn/Ms/Me/Mw` | Magic wall (north/south/east/west side) |

---

## See Also

- [GRIDWORLD_REFERENCE.md](../vendor/multigrid/GRIDWORLD_REFERENCE.md) - Complete MultiGrid reference
- [PROBABILISTIC_TRANSITIONS.md](../vendor/multigrid/PROBABILISTIC_TRANSITIONS.md) - Transition probability details
- [VENDOR.md](../VENDOR.md) - Vendor management documentation

# Backward Induction for EMPO

This document describes the backward induction algorithms used to compute exact human policy priors (Phase 1) and robot policies (Phase 2) in the EMPO framework.

## Overview

Backward induction computes all quantities by working backwards from terminal states through a Directed Acyclic Graph (DAG) of reachable states. This provides **exact solutions** (no approximation) but is only feasible for small state spaces due to computational complexity.

For larger state spaces, use the learning-based neural network approximations in `src/empo/nn_based/`.

Backward induction is possible because the current game time (step number) is part of the state and so the game tree is acyclic (one cannot return to the same state) and finite, hence it has terminal states, and because we do not assume the agents play best responses to each others' policies (which would introduce a different type of cyclic dependency).

## Implementation

### Core Module: `src/empo/backward_induction.py`

#### `compute_human_policy_prior()`

Computes goal-conditioned Boltzmann policies for all human agents.

```python
from empo.backward_induction import compute_human_policy_prior

policy_prior, Vh_values = compute_human_policy_prior(
    world_model=env,
    human_agent_indices=[0],
    possible_goal_generator=goal_gen,
    beta_h=10.0,
    gamma_h=0.99,
    return_Vh=True,
)

# Query policy for a specific state and goal
action_probs = policy_prior(state, agent_idx=0, goal=my_goal)
```

**Algorithm:**
1. Build state DAG via `world_model.get_dag()`
2. Compute dependency levels (topological sort)
3. Process states in reverse order (terminals first):
   - Terminal: $V_h = \mathbb{1}_g(s)$
   - Non-terminal: Compute Q-values, policy, then V-value

**Parallelization:** Supports parallel computation within dependency levels using multiprocessing with fork context.

**TODO:** check if $V_h$ equals $V_h^m$ from the theory paper, especially whether it is based on $\min_{a_r}$ rather than $E_{a_r}$, and whether it accounts for goal attainment correctly in the successor state rather than the source state.  

#### `compute_robot_policy()`

Computes the robot's power-law policy that maximizes human empowerment.

```python
from empo.backward_induction import compute_robot_policy

robot_policy, Vr_values, Vh_values = compute_robot_policy(
    world_model=env,
    human_agent_indices=[0],
    robot_agent_indices=[1],
    possible_goal_generator=goal_gen,
    human_policy_prior=policy_prior,
    beta_r=100.0,
    gamma_h=0.99,
    gamma_r=0.99,
    zeta=2.0,
    xi=1.0,
    eta=1.1,
    return_values=True,
)

# Sample robot action
robot_action_profile = robot_policy.sample(state)
```

**Algorithm:**
1. Build state DAG
2. Process states in reverse topological order:
   - Terminal: $V_r = $ `terminal_Vr` (must be negative)
   - Non-terminal:
     1. Compute $Q_r$ for all robot action profiles
     2. Compute $\pi_r$ via power-law softmax (log-space for numerical stability)
     3. Compute $V_h^e$ for all goals under $\pi_r$
     4. Compute $X_h$, $U_r$, $V_r$

**Numerical Stability:** The power-law formula $(-Q_r)^{-\beta_r}$ can overflow for large $\beta_r$. The implementation uses log-space computation:
$$\pi_r(a) = \exp(-\beta_r \cdot \log(-Q_r(a)) - \log Z)$$
where $Z$ is computed via `scipy.special.logsumexp`.

### Return Classes

#### `TabularHumanPolicyPrior`

Stores precomputed human policies indexed by (state, agent, goal).

- `__call__(state, agent_idx, goal)` → numpy array of action probabilities
- `profile_distribution(state)` → iterator of (prob, action_profile) pairs
- `profile_distribution_with_fixed_goal(state, agent_idx, goal)` → same but using goal-specific policy for one agent

#### `TabularRobotPolicy`

Stores precomputed robot policy indexed by state.

- `__call__(state)` → dict mapping action_profile → probability
- `sample(state)` → sampled action profile tuple
- `get_action(state, robot_idx)` → action for specific robot

## Comparing Phase 1 vs Phase 2 Vh Values

An important insight is that Phase 1 and Phase 2 compute different $V_h$ values:

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Robot behavior | Uniform random | Computed policy |
| Vh meaning | Goal achievement under random robot | Goal achievement under helpful robot |
| Typical values | Lower (no help) | Higher (robot helps) |

However, Phase 2 Vh can be *lower* for some specific goals because the robot optimizes *aggregate* empowerment, not individual goal achievement.

## Demo Script

### `examples/phase2_backward_induction.py`

Demonstrates backward induction on a small grid world:

```
Layout:
    We We We We We We
    We Ro Rk .. .. We
    We We Hu We We We
    We We We We We We

Ro = Robot (grey), Hu = Human (yellow), Rk = Rock
```

The robot can push the rock to clear paths for the human to reach goal cells.

**Usage:**
```bash
# Basic usage (5 steps)
python examples/phase2_backward_induction.py

# Longer horizon for harder goals
python examples/phase2_backward_induction.py --steps 12

# Emphasize goal (3,1) which requires robot to return
python examples/phase2_backward_induction.py --steps 12 --weight31 100.0

# All options
python examples/phase2_backward_induction.py \
    --steps 12 \
    --rollouts 20 \
    --beta_h 2.0 \
    --beta_r 100.0 \
    --gamma_h 0.99 \
    --gamma_r 0.99 \
    --zeta 2.0 \
    --xi 1.0 \
    --eta 1.1 \
    --weight31 100.0
```

**Output:**
1. Phase 1 human policy prior computation
2. Phase 2 robot policy computation  
3. Comparison of Vh values (Phase 1 vs Phase 2)
4. Rollout video saved to `outputs/phase2_backward_induction/rollouts.mp4`

**Goal Weights:**
- `--weight11`, `--weight21`, `--weight31`, `--weight41`, `--weight22`: Set importance of each goal cell
- Higher weight for a goal encourages the robot to enable that goal
- Example: `--weight31 100.0` makes reaching (3,1) very important, encouraging the robot to push the rock twice and return to its starting position

## Complexity

| Metric | Complexity |
|--------|------------|
| Time | $O(|S| \cdot |A|^n \cdot |G|)$ |
| Memory | $O(|S| \cdot n \cdot |G|)$ |

where $|S|$ = states, $|A|$ = actions per agent, $n$ = agents, $|G|$ = goals.

For practical use:
- Small grids (< 1000 states): Works well
- Medium grids (1000-10000 states): May take minutes
- Large grids (> 10000 states): Use neural network approximations instead

## See Also

- [API.md](API.md) - Full API reference
- [WARMUP_DESIGN.md](WARMUP_DESIGN.md) - Phase 2 neural network training
- `examples/human_policy_prior_example.py` - Phase 1 only example
- `examples/lookup_table_phase2_demo.py` - Alternative Phase 2 demo

# Phase 2 Training Deep Dive

This document details the training process for Phase 2 of the EMPO framework:
**how the robot policy is computed** to softly maximize aggregate human power
(equations 4–9 of the [EMPO paper](https://arxiv.org/html/2508.00159v2)).

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Networks and Their Roles](#2-networks-and-their-roles)
3. [Phase2Config — Configuration Reference](#3-phase2config--configuration-reference)
   - [Theory Parameters](#31-theory-parameters-not-hyperparameters)
   - [Warm-up Stage Durations](#32-warm-up-stage-durations)
   - [Learning Rate Schedule](#33-learning-rate-schedule)
   - [Exploration](#34-exploration)
   - [Replay Buffer & Training Ratio](#35-replay-buffer--training-ratio)
   - [Regularisation](#36-regularisation)
   - [Network Mode Flags](#37-network-mode-flags)
   - [Config Query Methods](#38-config-query-methods)
4. [Training Workflow](#4-training-workflow)
   - [Initialisation](#41-initialisation)
   - [Main Loop — `train()`](#42-main-loop--train)
   - [Actor Step — `_actor_step()`](#43-actor-step--_actor_step)
   - [Learner Step — `_learner_step()`](#44-learner-step--_learner_step)
   - [Training Step — `training_step()`](#45-training-step--training_step)
5. [Loss Computation — `compute_losses()`](#5-loss-computation--compute_losses)
   - [V_h^e Loss](#51-vhe-loss--human-goal-achievement)
   - [X_h Loss](#52-xh-loss--aggregate-human-goal-ability)
   - [U_r Loss](#53-ur-loss--intrinsic-reward)
   - [Q_r Loss](#54-qr-loss--robot-q-function)
   - [V_r Loss](#55-vr-loss--robot-value-function)
6. [Warm-up Stages](#6-warm-up-stages)
7. [Target Networks](#7-target-networks)
8. [Replay Buffer & Data Flow](#8-replay-buffer--data-flow)
9. [Curiosity-Driven Exploration](#9-curiosity-driven-exploration)
10. [Async Training Mode](#10-async-training-mode)
11. [GPU & Parallelization Notes](#11-gpu--parallelization-notes)
12. [Quick Start Code](#12-quick-start-code)

---

## 1. Conceptual Overview

Phase 2 is **not standard reinforcement learning**. The robot policy is the *solution to
a system of equations* (4–9), not a policy that maximises a scalar reward.

The key quantity is **aggregate human power** $U_r(s)$:

$$
U_r(s) = -\left(\mathbb{E}_h\left[X_h(s)^{-\xi}\right]\right)^\eta
$$

> The negative sign keeps $U_r < 0$, consistent with the range of $Q_r$. Raising to the power $\xi \geq 1$ makes the expectation sensitive to humans with *very low* ability — the larger $\xi$, the more the robot is penalised for leaving any human powerless (inter-human inequality aversion). The outer $\eta \geq 1$ then applies the same logic across states over time (intertemporal inequality aversion).

where $X_h(s)$ is human $h$'s **goal-achievement ability**:

$$
X_h(s) = \mathbb{E}_g\left[V_h^e(s,g)^\zeta\right]
$$

> Raising $V_h^e$ to the power $\zeta \geq 1$ introduces a *reliability preference*: a human who can achieve one goal with probability 1 scores higher than one who can achieve two goals with probability 0.5 each. The larger $\zeta$, the more the robot rewards consistently high ability over broad-but-shallow coverage.

and $V_h^e(s,g)$ is the probability that human $h$ can achieve goal $g$ from state $s$
**under the robot's current policy**.

The robot's Q-function bootstraps from $V_r = U_r + \mathbb{E}_{a \sim \pi_r}[Q_r]$,
creating a **mutual dependency** that warm-up breaks by staging network activation.

---

## 2. Networks and Their Roles

| Network | Symbol | Output | Range | Dependency |
|---------|--------|--------|-------|------------|
| `v_h_e` | $V_h^e(s, g)$ | Prob. of human $h$ achieving $g$ from $s$ | $[0, 1]$ | Needs robot policy $\pi_r$ |
| `x_h` | $X_h(s)$ | Aggregate ability of human $h$ | $(0, 1]$ | Needs $V_h^e$ |
| `u_r` | $U_r(s)$ | Robot intrinsic reward (power metric) | $(-\infty, 0)$ | Needs $X_h$ |
| `q_r` | $Q_r(s, a_r)$ | Robot Q-function | $(-\infty, 0)$ | Needs $V_r$ via bootstrapping |
| `v_r` *(optional)* | $V_r(s)$ | Robot value function | $(-\infty, 0)$ | Needs $U_r + Q_r$ |

All five live in `Phase2Networks` (a dataclass in `trainer.py`):

```python
@dataclass
class Phase2Networks:
    q_r:  BaseRobotQNetwork
    v_h_e: BaseHumanGoalAchievementNetwork
    x_h:  Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r:  Optional[BaseIntrinsicRewardNetwork]       = None
    v_r:  Optional[BaseRobotValueNetwork]            = None
    # Plus optional curiosity modules:
    # RND (Random Network Distillation) maintains a fixed random target network and a
    # trainable predictor; high prediction error signals unexplored states.
    rnd:  Optional[RNDModule]                        = None
    count_curiosity: Optional[CountBasedCuriosity]   = None
    # Target networks (frozen copies, created automatically in __init__)
    q_r_target:   Optional[...] = None
    v_h_e_target: Optional[...] = None
    ...
```

> **Note:** `x_h_use_network=False` (default `True`) and `u_r_use_network=False` (default `False`)
> are supported. When disabled, $X_h$ and $U_r$ are computed **analytically** from the
> previous network's outputs, reducing approximation error.

---

## 3. Phase2Config — Configuration Reference

`Phase2Config` is a plain Python `@dataclass` living in
`src/empo/learning_based/phase2/config.py`. It is instantiated once and passed to the
trainer; it is **never re-read from disk during training**.

### 3.1 Theory Parameters (NOT hyperparameters)

These are set by the *theory*, not tuned:

```python
gamma_r: float = 0.99    # Robot discount factor
gamma_h: float = 0.99    # Human discount factor (for V_h^e)
zeta:    float = 2.0     # ζ — risk/reliability preference (≥1)
xi:      float = 1.0     # ξ — inter-human inequality aversion (≥1)
eta:     float = 1.1     # η — intertemporal inequality aversion (≥1)
beta_r:  float = 10.0    # Nominal β_r — controls robot policy sharpness
```

`beta_r` is the *nominal* value reached after warm-up. The **effective** value at step $t$
is obtained via `config.get_effective_beta_r(t)`, which uses a sigmoidal ramp starting
from `_warmup_v_r_end` over `beta_r_rampup_steps`.

### 3.2 Warm-up Stage Durations

The five **duration** fields are specified in training steps (gradient updates):

```python
warmup_v_h_e_steps: int = 1e4   # Stage 1: V_h^e only
warmup_x_h_steps:  int = 1e4   # Stage 2: + X_h  (set to 0 if x_h_use_network=False)
warmup_u_r_steps:  int = 5e3   # Stage 3: + U_r  (set to 0 if u_r_use_network=False)
warmup_q_r_steps:  int = 1e4   # Stage 4: + Q_r
warmup_v_r_steps:  int = 5e3   # Stage 5: + V_r  (set to 0 if v_r_use_network=False)
beta_r_rampup_steps: int = 2e4 # Stage 6: β_r sigmoid ramp-up
```

`__post_init__` converts durations to **cumulative thresholds**:

```python
self._warmup_v_h_e_end = self.warmup_v_h_e_steps
self._warmup_x_h_end   = self._warmup_v_h_e_end + self.warmup_x_h_steps
self._warmup_u_r_end   = self._warmup_x_h_end   + self.warmup_u_r_steps
self._warmup_q_r_end   = self._warmup_u_r_end   + self.warmup_q_r_steps
self._warmup_v_r_end   = self._warmup_q_r_end   + self.warmup_v_r_steps
```

Stage transitions are **always checked against `training_step_count`**, never wall-clock time.

The full sequence looks like:

```
Step 0                    1e4       2e4      2.5e4     3.5e4    7.5e4  (example)
|-----V_h^e only---------|--+X_h---|--+U_r--|---+Q_r---|--β_r↑--|--Full training-->
Stage 0                   1        2        3          5        6
```

Use `config.format_stages_table()` at startup to print the exact schedule for your config.

### 3.3 Learning Rate Schedule

There are **three phases** within the LR schedule:

| Phase | Condition | LR |
|-------|-----------|----|
| Warm-up + ramp-up | `step < _warmup_v_r_end + beta_r_rampup_steps` | Constant base LR |
| Constant phase | `step < lr_constant_fraction * num_training_steps` | Constant base LR |
| Decay phase | After constant phase | `1/t` or `1/√t` decay |

```python
lr_constant_fraction:     float = 0.7   # Keep LR constant until 70 % of total steps
constant_lr_then_1_over_t: bool = True  # Use 1/t decay (Robbins-Monro) in late phase
use_sqrt_lr_decay:         bool = True  # Fallback: 1/√t if above is False
```

> The $1/t$ schedule satisfies the Robbins-Monro conditions ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$), providing theoretical guarantees of convergence for stochastic fixed-point iteration. The constant phase beforehand ensures networks are already in a reasonable region before the aggressive decay begins.

The method `config.get_learning_rate(network_name, step, update_count)` returns the
effective LR. The trainer calls it every step and writes it to each optimizer's
`param_groups[0]['lr']`.

Base LRs per network (trainable hyperparameters):

```python
lr_q_r:   float = 1e-4
lr_v_r:   float = 1e-4
lr_v_h_e: float = 1e-3
lr_x_h:   float = 1e-4
lr_u_r:   float = 1e-4
```

### 3.4 Exploration

Epsilon-greedy with **linear decay**:

```python
epsilon_r_start:      float = 1.0    # Robot ε at step 0
epsilon_r_end:        float = 0.01   # Robot ε at step epsilon_r_decay_steps
epsilon_r_decay_steps: int  = 10000

epsilon_h_start:      float = 1.0    # Human ε at step 0
epsilon_h_end:        float = 0.01
epsilon_h_decay_steps: int  = 10000
```

Query via `config.get_epsilon_r(step)` and `config.get_epsilon_h(step)`.

For deep exploration, optional curiosity modules are available (choose one):

```python
use_rnd:               bool = False  # RND for neural networks
use_count_based_curiosity: bool = False  # Count-based for lookup tables
```

### 3.5 Replay Buffer & Training Ratio

```python
buffer_size:   int   = 100000  # Max transitions stored
batch_size:    int   = 64      # Gradient update batch size
x_h_batch_size: Optional[int] = None  # Larger batch for X_h (None = batch_size)

training_steps_per_env_step: float = 1.0
# Examples:
#   4.0  → 4 gradient updates per env step
#   0.1  → 1 gradient update every 10 env steps
# Alternative notation (auto-converted in __post_init__):
env_steps_per_training_step: Optional[float] = None
```

The accumulator in `train()` handles fractional ratios:

```python
training_step_accumulator += config.training_steps_per_env_step
while training_step_accumulator >= 1.0:
    training_step_accumulator -= 1.0
    _learner_step(...)
```

### 3.6 Regularisation

Per-network weight decay, gradient clipping, and dropout:

```python
q_r_weight_decay:  float = 1e-4
q_r_grad_clip:     float = 10.0
q_r_dropout:       float = 0.5

# Automatic gradient clip scaling by current LR:
auto_grad_clip:              bool  = True
auto_grad_clip_reference_lr: float = 1e-4
# Effective clip = base_clip * (current_lr / reference_lr)
# As LR decays late in training, the clip tightens proportionally —
# preventing large gradients from overshooting the increasingly precise solution.
```

Target network update intervals (hard copy every N training steps):

```python
q_r_target_update_interval:   int = 100
v_h_e_target_update_interval: int = 100
x_h_target_update_interval:   int = 100
u_r_target_update_interval:   int = 100
v_r_target_update_interval:   int = 100
```

### 3.7 Network Mode Flags

```python
x_h_use_network: bool = True   # If False, X_h computed exactly from V_h^e samples
u_r_use_network: bool = False  # If False, U_r computed exactly from X_h values
v_r_use_network: bool = False  # If False, V_r = U_r + E_{π_r}[Q_r] (computed, not learned)
use_lookup_tables: bool = False # If True, use dict-based tabular networks (no NN)
use_model_based_targets: bool = True  # Use transition_probabilities() for expected targets
```

### 3.8 Config Query Methods

| Method | Returns |
|--------|---------|
| `get_epsilon_r(step)` | Robot ε at given training step |
| `get_epsilon_h(step)` | Human ε at given training step |
| `get_effective_beta_r(step)` | Sigmoid-ramped β_r (0 during warm-up) |
| `get_learning_rate(name, step, count)` | LR for a named network |
| `get_active_networks(step)` | `Set[str]` of networks training at this step |
| `get_warmup_stage(step)` | Integer 0–6 indicating current stage |
| `get_warmup_stage_name(step)` | Human-readable stage name |
| `is_in_warmup(step)` | True before all networks are active |
| `is_in_rampup(step)` | True during β_r sigmoid ramp |
| `is_in_decay_phase(step)` | True once LR decay has started |
| `get_effective_grad_clip(name, lr)` | Auto-scaled clip value |
| `format_stages_table()` | ASCII table of all stages |
| `save_yaml(path)` | Write full config to YAML |

---

## 4. Training Workflow

### 4.1 Initialisation

`BasePhase2Trainer.__init__()` performs:

1. Store env, networks, config, agent indices, human prior, goal sampler.
2. Create `SummaryWriter` if `tensorboard_dir` is provided (archives old events first).
3. Call `_init_target_networks()` — deep-copy each network, freeze parameters, set eval mode.
4. Call `_init_optimizers()` — create one `Adam` optimizer per network with its `weight_decay`.
5. Create `Phase2ReplayBuffer(capacity=config.buffer_size)`.
6. Zero counters: `total_env_steps=0`, `training_step_count=0`, `update_counts`.
7. Create `MemoryMonitor` for OOM protection.

```python
trainer = BasePhase2Trainer(
    env=world_model,
    networks=networks,          # Phase2Networks
    config=Phase2Config(...),
    human_agent_indices=[0],
    robot_agent_indices=[1],
    human_policy_prior=heuristic_policy,
    goal_sampler=tabular_goal_sampler,
    device='cuda',
    verbose=True,
    tensorboard_dir='runs/phase2',
)
```

### 4.2 Main Loop — `train()`

```
train(num_training_steps)
│
├── If async_training → _train_async(n)  [spawns actor processes]
│
└── Sync loop:
    │
    ├── _init_actor_state()   → reset env, sample initial goals
    ├── _init_learner_state() → record warmup stage, start timer
    │
    └── while training_step_count < num_training_steps:
        │
        ├── [ACTOR]  _actor_step(actor_state)
        │            → collect one transition
        │            → push to replay_buffer
        │            → total_env_steps += 1
        │
        ├── training_step_accumulator += training_steps_per_env_step
        │
        └── while accumulator >= 1.0:
                accumulator -= 1.0
                [LEARNER] _learner_step(learner_state, pbar)
                          → training_step()           [sample, losses, update]
                          → training_step_count += 1
                          → TensorBoard logging
                          → warmup stage transitions
                          → LR decay transitions
                          → checkpoint saving
```

The accumulator pattern means:
- `training_steps_per_env_step=4.0` → 4 gradient updates for every env step
- `training_steps_per_env_step=0.1` → 1 gradient update for every 10 env steps

### 4.3 Actor Step — `_actor_step()`

The actor's sole job is to **collect one transition** from the world model and push it to the
replay buffer. It runs on CPU and never touches network gradients — only forward passes
through target networks (which are frozen and in eval mode).

```python
def _actor_step(self, actor_state):
    # Mark as terminal so the learner knows not to bootstrap past this step.
    is_terminal = (actor_state.env_step_count + 1) >= config.steps_per_episode

    # collect_transition() bundles environment interaction + transition-prob pre-computation.
    transition, next_state = collect_transition(
        actor_state.state, actor_state.goals, actor_state.goal_weights,
        terminal=is_terminal
    )

    actor_state.state = next_state
    actor_state.env_step_count += 1

    # Immediately replace achieved goals so the next step uses a fresh, non-trivial goal.
    for h, g in actor_state.goals.items():
        if check_goal_achieved(next_state, h, g):
            actor_state.goals[h], actor_state.goal_weights[h] = goal_sampler.sample(...)

    # config.goal_resample_prob allows probabilistic goal replacement even when not achieved,
    # preventing the agent from over-fitting to a single goal throughout a long episode.
    # Reset episode counters when steps_per_episode is reached.
    return transition
```

Inside `collect_transition()` the order is critical — transition probabilities are queried
**once from the world model** and reused for action selection, curiosity bonuses, *and* the
learner's model-based Bellman targets, avoiding redundant calls to `transition_probabilities()`:

```
1. sample_human_actions(state, goals)
   # Draws human actions from the human policy prior, with ε_h exploration.
   # Human intent is fixed before the robot acts — matching the theory's turn order.

2. _precompute_transition_probs(state, H_actions)
   # Calls env.transition_probabilities() for ALL robot actions simultaneously.
   # This is the only environment call in the entire step; result is cached in the transition.

3. sample_robot_action(state, trans_probs)
   # Evaluates q_r_target over all actions, applies ε_r and optional curiosity bonus,
   # then samples. Using the target network keeps action selection stable.

4. step_environment(state, robot_action, H_actions)
   # Advances the environment to the next state for the actor's own trajectory.

5. Build Phase2Transition (stores trans_probs_by_action for the learner)
   # The pre-computed probs are stored so the learner can compute full Bellman
   # backups over all actions without re-querying the environment.
```

### 4.4 Learner Step — `_learner_step()`

The learner consumes data from the replay buffer and updates network parameters. It is
entirely decoupled from the environment — it only sees transitions that were already
collected and stored.

```python
def _learner_step(self, learner_state, pbar):
    # training_step() samples a batch, computes all losses, runs backprop, and
    # updates optimizer states. This is the sole place gradients flow.
    losses, grad_norms, pred_stats = training_step()
    training_step_count += 1

    # Progress bar shows the most informative signals at a glance:
    # v_h_e loss (is the human model converging?), Δx_h and Δq_r (are power/Q stable?),
    # and steps-per-second (throughput).

    # TensorBoard logs a full diagnostic snapshot every step:
    # losses, grad_norms, pred_stats (mean/std of each network's predictions),
    # param_norms, ε_r, ε_h, β_r (effective), active networks bitmask, per-network LRs, stage.
    # This lets you detect divergence or stagnation early without reading code.

    # Warmup stage transition check — the most consequential side-effect here:
    # When crossing into stage 5 (β_r ramp) or stage 6 (full training), the replay buffer
    # is cleared. Data collected under a different β_r would bias Q_r targets, so stale
    # transitions must be discarded even though it temporarily reduces batch diversity.

    # LR decay transition: logs a one-time message when the constant-LR phase ends.

    # Every 100 steps: flush TensorBoard writer and log a summary line to stdout.

    return losses
```

### 4.5 Training Step — `training_step()`

This is the inner update loop. Called once per gradient update:

```
training_step()
│
├── Sample batch (size=config.batch_size) from replay_buffer
│   [optionally also sample x_h_batch (size=x_h_batch_size) for X_h]
│
├── compute_losses(batch, x_h_batch)          → losses, pred_stats
│
├── Determine active_networks = config.get_active_networks(step)
│
├── Zero gradients for all active optimizers
│
├── Backward pass (each network independently, retain_graph except last)
│
├── _apply_adaptive_lr_scaling()              # 1/n for lookup tables
│
├── _apply_rnd_adaptive_lr_scaling()          # RND-based LR scaling (optional)
│
├── For each active network:
│   ├── update_counts[name] += 1
│   ├── new_lr = config.get_learning_rate(name, step, count)
│   ├── param_group['lr'] = new_lr
│   ├── clip_grad_norm_(net.parameters(), effective_clip)
│   └── optimizer.step()
│
├── update_target_networks()                  # Hard copy at their intervals
│
└── _add_new_lookup_params_to_optimizers()    # For growing lookup tables
```

---

## 5. Loss Computation — `compute_losses()`

All losses are MSE in value-space (or z-space if `use_z_space_transform=True`).
Networks that are **not active** in the current warm-up stage still compute a forward
pass (for prediction stats) but their loss is **excluded from the backward pass**.

> **Why MSE?** Every network approximates a conditional expectation (a probability, an average
> ability, an average Q-value). MSE is the standard loss for regression to a conditional mean,
> and it produces sub-gradient signals that are proportional to the prediction error — matching
> the Bellman-residual minimisation viewpoint.

> **Why independent backward passes?** Each network’s loss is back-propagated separately
> (with `retain_graph=True` for all but the last). This lets each optimizer apply its own
> gradient clip *before* any parameter update, preventing a poorly-scaled network from
> poisoning the shared computation graph and corrupting gradients of upstream networks.

### 5.1 V_h^e Loss — Human Goal Achievement

**Target (model-based, policy-weighted):**

$$
V_h^e(s, g)^{\text{target}}
= \sum_{a_r} \pi_r(a_r|s) \sum_{s'} P(s'|s, a_r)
\begin{cases}
1 & \text{if } g \text{ achieved at } s' \\
\gamma_h \cdot V_h^{e,\text{target}}(s', g) & \text{otherwise}
\end{cases}
$$

This is a standard discounted-reachability Bellman equation. It is **policy-weighted** — averaged over $\pi_r$ — because $V_h^e$ measures human ability *given that the robot follows its current policy*, not the best or worst case.

> **Why doesn't the self-reference cause an infinite loop?**
> The right-hand side uses `v_h_e_target` — the **frozen** copy of the network — not the live
> `v_h_e` that is being trained. The target network is held constant for
> `v_h_e_target_update_interval` steps, so the recursion terminates immediately: the bootstrap
> value $V_h^{e,\text{target}}(s', g)$ is just a vector lookup into a fixed function, not a
> further call to the equation being minimised.
>
> **Why is this approximation close to the exact tabular result?**
> The tabular (backward-induction) version solves the same Bellman equation exactly by sweeping
> from terminal states backward until convergence. Here, repeated application of the Bellman
> operator is replaced by repeated gradient steps against a slowly-moving target network.
> Because $\gamma_h < 1$, the Bellman operator is a **contraction** (Banach fixed-point theorem):
> every application shrinks the gap to the true fixed point by a factor of $\gamma_h$.
> After many training steps the neural estimate therefore converges to the same unique fixed
> point as backward induction — just stochastically rather than analytically.
> In practice, with `v_h_e_target_update_interval` large enough to keep the target stable but
> small enough to limit lag, the neural $V_h^e$ tracks the tabular solution closely; any
> residual difference is bounded by the approximation error of the network (its capacity to
> represent the true function) rather than by the bootstrap scheme itself.

**Implementation (batched in 4 phases):**

> **Why 4 phases instead of a simple loop?** A naïve implementation would call
> `v_h_e_target.forward()` once per (transition, action, successor-state) triple, i.e.,
> $O(\text{batch} \times |A_r| \times \text{branching})$ forward passes. The 4-phase approach
> collects *all* successor states first, then issues a **single batched forward pass** over
> them. This reduces GPU round-trips from thousands to one per training step, which is the
> dominant cost for neural implementations.

> **Why `q_r_target` for $\pi_r$ here?** The target policy is used (not the live $Q_r$)
> so that the $V_h^e$ targets do not co-evolve with $V_h^e$’s own gradient step within the
> same training step. Without this, the Bellman target would shift simultaneously with the
> prediction, making the fixed-point unstable.

```python
# Phase 1: get π_r(a|s) for all states in batch (one q_r_target forward pass)
q_r_batch = networks.q_r_target.forward_batch(states, ...)
robot_policies = networks.q_r_target.get_policy(q_r_batch, beta_r=effective_beta_r)
# Note: during warmup, effective_beta_r = 0 → uniform π_r.
# This is intentional: before Q_r is trained, any non-uniform policy would skew V_h^e
# targets toward actions that happen to look good for random reasons.

# Phase 2: collect ALL successor states (across all actions weighted by π_r)
for data_idx, (trans_idx, human_idx, goal) in enumerate(v_h_e_data):
    for action_idx, action_prob in enumerate(robot_policies[trans_idx]):
        for state_prob, next_state in trans_probs_by_action[action_idx]:
            weight = action_prob * state_prob
            if goal.is_achieved(next_state):
                achieved_contributions[data_idx] += weight
            else:
                all_next_states.append(next_state)
                successor_mapping.append((data_idx, weight))

# Phase 3: ONE batched V_h^e_target forward pass for ALL successor states
v_h_e_all = networks.v_h_e_target.forward_batch(all_next_states, all_goals, ...)

# Phase 4: aggregate
targets = achieved_contributions + scatter_add(gamma_h * v_h_e_all, data_indices, weights)

# Loss
loss_v_h_e = MSE(v_h_e_pred, targets)
```

### 5.2 X_h Loss — Aggregate Human Goal Ability

**Target:**

$$
X_h(s)^{\text{target}} = w_g \cdot V_h^e(s, g)^\zeta
$$

where $w_g$ is the goal sampling weight. Since goals are sampled non-uniformly, multiplying by $w_g$ corrects for sampling bias, making each sample an unbiased Monte Carlo estimate of $\mathbb{E}_g[V_h^e(s,g)^\zeta]$.

```python
# Forward pass
x_h_pred = networks.x_h.forward_batch(x_h_states, x_h_human_indices, ...)

# Target from V_h^e target network
v_h_e_for_x = networks.v_h_e_target.forward_batch(x_h_states, x_h_goals, ...)
target_x_h = x_h_weights * (v_h_e_for_x ** config.zeta)

loss_x_h = MSE(x_h_pred, target_x_h)
```

When `x_h_use_network=False`, this loss is skipped entirely and $X_h$ is computed
analytically in `_compute_u_r_batch_target()`.

> **Why a separate `x_h_batch_size`?** The $X_h$ target only requires a `v_h_e_target`
> forward pass — it does not depend on $Q_r$ or any other network. It is therefore cheap
> to evaluate with a larger batch, improving the Monte Carlo estimate of
> $\mathbb{E}_g[V_h^e(s,g)^\zeta]$ without increasing memory pressure on the networks that
> do depend on each other. A larger $X_h$ batch reduces the variance of the $U_r$ and
> $Q_r$ targets downstream.

### 5.3 U_r Loss — Intrinsic Reward

The network actually predicts an **intermediate variable** $y = \mathbb{E}_h[X_h^{-\xi}]$
(before the outer $-(\cdot)^\eta$ is applied). This avoids numerical issues near zero — directly predicting $U_r = -y^\eta$ would compound any approximation error in $y$ through the non-linear exponent, particularly problematic when $y$ is small and the gradient of $y^\eta$ is large.

**Target:**

$$
y^{\text{target}} = \frac{1}{|H|} \sum_h X_h(s)^{-\xi}
$$

```python
# Predict y for all states
y_pred, _ = networks.u_r.forward_batch(states, ...)

# Target from X_h target network (single batched call)
x_h_all = networks.x_h_target.forward_batch(flat_states, flat_humans, ...)
x_h_clamped = clamp(x_h_all, min=1e-3, max=1.0)
# Clamping before raising to a negative power is essential: X_h → 0 would cause
# x_h ** (-xi) → +∞, producing NaN gradients. The 1e-3 floor is a soft floor on
# human ability — it acknowledges that no state is truly hopeless for every goal.
x_h_power = x_h_clamped ** (-config.xi)
y_target = scatter_mean(x_h_power, state_indices)
# scatter_mean (not scatter_sum) keeps y scale-independent of the number of humans |H|,
# which matters if environments with different human counts share a config.

loss_u_r = MSE(y_pred, y_target)
# (U_r = -y^eta is only used for logging, not in the loss)
```

When `u_r_use_network=False`, $U_r$ is computed directly:
`_compute_u_r_batch_target()` → `_compute_u_r_from_v_h_e_samples()`.

### 5.4 Q_r Loss — Robot Q-Function

**Target (Equation 4):**

$$
Q_r(s, a_r)^{\text{target}} = \sum_{s'} P(s'|s, a_r) \cdot \gamma_r \cdot V_r(s')
$$

Critically, this is computed for **ALL** robot actions (full Bellman backup), not just
the action taken. Since `transition_probs_by_action` is pre-stored for every action,
there is no additional environment cost — using all actions instead of one reduces
estimation variance by a factor proportional to the number of actions:

```python
# Forward pass: Q-values for all actions
q_r_all = networks.q_r.forward_batch(states, ...)  # (B, num_actions)

# Model-based targets for ALL actions (deduplicates successor states)
target_q_r_all = _compute_model_based_q_r_targets(batch)  # (B, num_actions)

loss_q_r = MSE(q_r_all, target_q_r_all)  # over all (B * num_actions) entries
```

`_compute_model_based_q_r_targets()` deduplicates successor states for efficiency:

```python
# Collect unique next_states from all (batch, action) pairs
state_to_idx: Dict[hash, int] = {}
unique_states: List = []

for batch_idx, transition in enumerate(batch):
    for action_idx, trans_probs in transition.transition_probs_by_action.items():
        for prob, next_state in trans_probs:
            if hash(next_state) not in state_to_idx:
                unique_states.append(next_state)

# ONE batched U_r and V_r/Q_r forward pass over unique states
u_r_all = _compute_u_r_batch_target(unique_states)
v_r_all = networks.v_r_target.forward_batch(unique_states, ...)
# or if v_r_use_network=False:
#   v_r_all = U_r + E_{π_r}[Q_r] computed from q_r_target

q_targets_all = gamma_r * v_r_all  # V_r already incorporates U_r
# Because V_r = U_r + E[Q_r], writing Q_r^target = γ_r * V_r already has
# the power metric baked in. There is no separate U_r addend needed here,
# unlike a standard actor-critic where reward and bootstrapped value are separate.

# Aggregate back: probability-weighted sum over successor states
for batch_idx, action_idx, (unique_idx, prob):
    targets[batch_idx, action_idx] += prob * q_targets_all[unique_idx]
```

### 5.5 V_r Loss — Robot Value Function

Only computed when `v_r_use_network=True` (disabled by default). When disabled, $V_r$ is computed analytically as $U_r + \mathbb{E}_{\pi_r}[Q_r]$, which introduces no additional approximation error beyond what $U_r$ and $Q_r$ already carry. A separate network would compound those errors through an extra bootstrap layer.

**Target (Equation 9):**

$$
V_r(s)^{\text{target}} = U_r(s) + \mathbb{E}_{a \sim \pi_r}[Q_r(s, a)]
$$

```python
v_r_pred  = networks.v_r.forward_batch(states, ...)
u_r_for_v = _compute_u_r_batch_target(states)
# Uses the U_r *target* network (or analytical computation), not the live network.
# This keeps the V_r target anchored to a slowly-moving U_r estimate,
# consistent with how Q_r targets are computed, and avoids a double-moving-target problem.
q_r_for_v = networks.q_r_target.forward_batch(states, ...)
pi_r      = networks.q_r_target.get_policy(q_r_for_v, beta_r=effective_beta_r)
target_v_r = compute_v_r_from_components(u_r_for_v, q_r_for_v, pi_r)

loss_v_r = MSE(v_r_pred, target_v_r)
```

---

## 6. Warm-up Stages

Warm-up breaks the circular dependency between networks by activating them one by one.
During warm-up, `effective_beta_r = 0` (robot acts uniformly randomly).

```
Stage 0  V_h^e only          → learns goal-achievement probs under random robot
Stage 1  + X_h               → learns aggregate human ability from V_h^e
Stage 2  + U_r               → learns intrinsic reward from X_h  (if u_r_use_network)
Stage 3  + Q_r               → learns robot Q-function (robot still random)
Stage 4  + V_r               → learns robot value function  (if v_r_use_network)
Stage 5  β_r ramp-up         → β_r rises sigmoid from 0 → nominal; ALL networks active
Stage 6  Full training + LR↓ → LR decay begins; Robbins-Monro convergence
```

> **Replay buffer is cleared** at stage 5 start (β_r ramp begins) and at stage 6 start
> (β_r ramp ends). This prevents stale data collected under different β_r from polluting
> the Q_r loss.

`config.get_active_networks(step)` returns the exact set for each step:

```python
active = config.get_active_networks(training_step_count)
# → e.g., {'v_h_e', 'x_h', 'q_r'} at stage 3 with u_r_use_network=False
```

Transition steps can be queried before training:

```python
for step, name in config.get_stage_transition_steps():
    print(f"  Step {step:>8,}: {name}")
```

---

## 7. Target Networks

Target networks are **frozen copies** used for computing stable Bellman targets.
They are hard-copied (not soft-updated) at their own intervals. Hard copies produce targets that are piecewise-constant between updates — predictable and stable — whereas soft (EMA) updates would continuously shift targets, risking oscillation in the dependency chain $V_h^e \to X_h \to U_r \to Q_r$.

```python
# In update_target_networks() (called every training_step):
if step % config.q_r_target_update_interval == 0:
    networks.q_r_target.load_state_dict(networks.q_r.state_dict())
    networks.q_r_target.eval()
# Same for v_h_e, x_h, u_r, v_r
```

Target networks are **always in eval mode** (dropout disabled). Main networks switch
to `train()` mode at the start of `train()` and back to `eval()` afterwards.

---

## 8. Replay Buffer & Data Flow

`Phase2ReplayBuffer` is a standard circular replay buffer. Each stored `Phase2Transition` includes:

| Field | Type | Purpose |
|-------|------|---------|
| `state` | hashable | Current world state |
| `robot_action` | `Tuple[int, ...]` | Robot's action |
| `goals` | `Dict[int, PossibleGoal]` | Per-human goals at this step |
| `goal_weights` | `Dict[int, float]` | Sampling weights for goals |
| `human_actions` | `List[int]` | Human actions taken |
| `next_state` | `None` (model-based) | `None` when using transition_probs |
| `transition_probs_by_action` | `Dict[int, List[(prob, s')]]` | Pre-computed transitions for ALL robot actions |
| `terminal` | `bool` | Whether episode ends after this step |

When `use_model_based_targets=True` (default), `next_state` is **intentionally `None`**;
all successor states come from `transition_probs_by_action`.

---

## 9. Curiosity-Driven Exploration

Two mechanisms are available (only one should be enabled at a time):

**RND** (`use_rnd=True`) — for neural networks:
- A fixed random target network and a trainable predictor network.
- Novelty = prediction MSE. High MSE → unexplored state.
- Applied multiplicatively to Q-values (preserves Q < 0 required by power-law policy):
  $Q_{\text{eff}} = Q \cdot \exp(-\text{bonus\_coef} \cdot \text{novelty})$

**Count-based** (`use_count_based_curiosity=True`) — for lookup tables:
- Maintains state visit counts.
- Bonus: $\text{scale} / \sqrt{n + 1}$ (or UCB-style).
- Applied the same way to both action selection and ε-exploration.

Both curiosity modules have separate coefficients for robot and human exploration
(`rnd_bonus_coef_r`, `rnd_bonus_coef_h`).

---

## 10. Async Training Mode

When `config.async_training=True`, `train()` delegates to `_train_async()`:

```
Learner process (main):                Actor processes (spawned):
  - GPU training loop                    - CPU environment stepping
  - Owns replay buffer                   - Reads frozen policy (Q_r) from shared dict
  - Publishes policy every               - Pushes transitions to mp.Queue
    actor_sync_freq steps                - Throttled by max_env_steps_per_training_step
```

Communication:
- **Transitions**: `mp.Queue(maxsize=async_queue_size)` from actors to learner.
- **Policy weights**: `manager.dict()` with a lock, updated every `actor_sync_freq` steps.
- **Training step counter**: `mp.Value('i', ...)` shared for ε/β_r scheduling in actors.

> **Note:** Async mode does not work in Jupyter notebooks (no `__main__` spec).
> `_train_async()` raises `RuntimeError` if called from a notebook.

---

## 11. GPU & Parallelization Notes

### Device placement

All network parameters and optimizer states live on the device specified by the `device`
argument to `BasePhase2Trainer` (`'cuda'`, `'cpu'`, or a specific `'cuda:N'`). Batches are
moved to that device inside `training_step()` before any forward pass.

The **actor always runs on CPU** — environment states are Python objects (tuples), not
tensors, and `transition_probabilities()` is a Python call that cannot be GPU-accelerated.
Only the target-network forward passes during action selection are moved to GPU if needed;
in practice, the overhead rarely justifies it for small networks, so target inference during
the actor step defaults to CPU as well.

### Batched forward passes

All costly network queries in `compute_losses()` are **batched across the entire replay
sample**, not called per-transition. The implementation deduplicates successor states before
any forward pass (see §5.4) so the total number of network evaluations per training step is
bounded by `batch_size × avg_branching_factor`, not `batch_size × num_actions × branching_factor`.

### Async actor parallelism

When `config.async_training=True` (see §10), multiple actor processes run in parallel on CPU
while the learner trains on GPU. The number of actors is controlled by `num_actor_processes`.
Increasing it improves data throughput when `training_steps_per_env_step < 1`, but adds
latency to policy synchronisation. For `training_steps_per_env_step ≥ 1` (more gradient
updates than env steps), a single synchronous actor is usually sufficient and avoids the
overhead of inter-process communication.

### Memory considerations

- The replay buffer stores raw Python objects (hashable states, `PossibleGoal` instances),
  not tensors. Memory usage scales with `buffer_size × avg_state_size`, not with network
  width. For large grids, monitor via `MemoryMonitor` which is instantiated automatically.
- `x_h_batch_size` can be set larger than `batch_size` to improve the $X_h$ target estimate
  without increasing GPU memory pressure on other networks.

---

## 12. Quick Start Code

```python
from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.multigrid.phase2.trainer import train_multigrid_phase2

# 1. Define configuration (theory params fixed, training params tunable)
config = Phase2Config(
    # --- Theory parameters (do not tune) ---
    gamma_r=0.99, gamma_h=0.99,
    zeta=2.0, xi=1.0, eta=1.1,
    beta_r=1000.0,
    
    # --- Warm-up (training steps per stage) ---
    warmup_v_h_e_steps=5_000,
    warmup_x_h_steps=5_000,
    warmup_u_r_steps=0,          # U_r computed analytically
    warmup_q_r_steps=5_000,
    warmup_v_r_steps=0,          # V_r computed analytically
    beta_r_rampup_steps=10_000,
    
    # --- LR schedule ---
    lr_v_h_e=5e-4,
    lr_q_r=1e-4,
    lr_constant_fraction=0.7,
    constant_lr_then_1_over_t=True,
    
    # --- Exploration ---
    epsilon_r_start=1.0, epsilon_r_end=0.05, epsilon_r_decay_steps=20_000,
    epsilon_h_start=1.0, epsilon_h_end=0.05, epsilon_h_decay_steps=20_000,
    
    # --- Training ratio & buffer ---
    training_steps_per_env_step=2.0,
    buffer_size=5_000,
    batch_size=64,
    num_training_steps=50_000,
    
    # --- Network mode ---
    x_h_use_network=True,
    u_r_use_network=False,
    v_r_use_network=False,
    use_encoders=True,
    use_lookup_tables=False,
    use_model_based_targets=True,
)

# 2. Print the training schedule before starting
print(config.format_stages_table())

# 3. Train (high-level entry point)
robot_q_network, all_networks, history, trainer = train_multigrid_phase2(
    grid_map=GRID_MAP,
    max_steps=50,
    config=config,
    num_training_steps=config.num_training_steps,
    verbose=True,
    tensorboard_dir="runs/phase2_run",
)

# 4. Inspect effective β_r at the end
print(f"Final β_r: {config.get_effective_beta_r(trainer.training_step_count):.1f}")

# 5. Query training stages programmatically
for stage in config.get_stages_info():
    print(f"Stage {stage['stage_num']}: {stage['name']} "
          f"({stage['duration']:,} steps, ends at {stage['end_step']:,})")
```

**Expected output of `format_stages_table()`:**

```
Training Stages:
----------------------------------------------------------------------
Stage  Name                      Duration     End Step  Networks
----------------------------------------------------------------------
0      V_h^e only                   5,000        5,000  v_h_e
1      V_h^e + X_h                  5,000       10,000  v_h_e, x_h
3      + Q_r                        5,000       15,000  v_h_e, x_h, q_r
5      β_r ramp-up                 10,000       25,000  v_h_e, x_h, q_r
6      Full training               25,000       50,000  v_h_e, x_h, q_r
----------------------------------------------------------------------
Total warmup (before LR decay): 25,000 steps
Total training steps: 50,000
```

---

*See also:*
- [`docs/WARMUP_DESIGN.md`](WARMUP_DESIGN.md) — warm-up rationale
- [`docs/VALUE_TRANSFORMATIONS.md`](VALUE_TRANSFORMATIONS.md) — z-space transforms
- [`docs/ADAPTIVE_LEARNING.md`](ADAPTIVE_LEARNING.md) — adaptive LR for lookup tables
- [`docs/EXPLORATION.md`](EXPLORATION.md) — RND and count-based curiosity
- [`docs/PARALLELIZATION.md`](PARALLELIZATION.md) — async actor detail and cluster setup
- [`examples/phase2/phase2_robot_policy_demo.py`](../examples/phase2/phase2_robot_policy_demo.py) — runnable demo
- [`examples/phase2/hpo_phase2_example.py`](../examples/phase2/hpo_phase2_example.py) — Optuna HPO script

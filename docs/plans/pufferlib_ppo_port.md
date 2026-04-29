# Plan: Porting Phase 2 Training to PufferLib PPO

**Status:** Proposed  
**Date:** 2025-03-20

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture (DQN-style)](#2-current-architecture-dqn-style)
3. [Proposed Architecture (PufferLib PPO)](#3-proposed-architecture-pufferlib-ppo)
   - [3.4 Code Isolation Strategy](#34-code-isolation-strategy)
   - [3.5 PufferLib Integration Strategy](#35-pufferlib-integration-strategy)
4. [The Core Challenge: Intrinsic Reward Integration](#4-the-core-challenge-intrinsic-reward-integration)
5. [Environment Wrapper Design](#5-environment-wrapper-design)
6. [Network Architecture Changes](#6-network-architecture-changes)
7. [Warm-up and Mutual Dependency Handling](#7-warm-up-and-mutual-dependency-handling)
8. [Training Loop Redesign](#8-training-loop-redesign)
9. [Theoretical Considerations](#9-theoretical-considerations)
10. [Migration Plan](#10-migration-plan)
11. [Open Questions and Risks](#11-open-questions-and-risks)
12. [PPO for Phase 1 Training](#12-ppo-for-phase-1-training)
13. [Testing with PufferLib Ocean Environments](#13-testing-with-pufferlib-ocean-environments)

---

## 1. Executive Summary

This plan describes how to port the neural-net-based Phase 2 robot policy training
from its current **DQN-style Q_r learning** approach to a **PufferLib-supplied PPO**
that directly learns **(π_r, V_r)** — an explicit policy network and a value network —
rather than deriving the policy implicitly from Q-values via the power-law softmax.

**The central challenge** is that standard PPO consumes reward signals from the
environment, but in EMPO the robot's reward signal **U_r(s)** is an *intrinsic* reward
produced by other neural networks (V_h^e, X_h) that are themselves being trained
simultaneously. U_r is non-stationary: it changes as those networks improve. This
document describes how to feed this intrinsic reward to PufferLib's PPO while
preserving the theoretical properties of the EMPO framework.

### Parallel Implementation Principle

> **The PPO implementation MUST be parallel to the existing DQN-based code.**
> No existing files, classes, or functions in the current DQN trainer will be modified.
> The existing code path must remain fully functional at all times, guaranteeing
> backward compatibility. The PPO trainer lives in its own modules, uses its own
> config class, and is selected via a separate entry point — not a flag inside
> the existing trainer.

### Key Changes at a Glance

| Aspect | Current (DQN-style) | Proposed (PufferLib PPO) |
|--------|---------------------|--------------------------|
| **Robot policy** | Implicit: π_r ∝ (-Q_r)^{-β_r} | Explicit: π_r network (actor) |
| **Robot value** | Optional V_r network, or computed from U_r + E[Q_r] | V_r network (critic) |
| **Q_r** | Learned via model-based Bellman targets | Not used by PPO path (DQN code untouched) |
| **Reward signal** | Implicit in Bellman equation (γ_r · V_r(s')) | Explicit U_r(s) passed as reward to PPO |
| **Data collection** | Off-policy replay buffer | On-policy rollout buffer (PufferLib vectorized) |
| **Exploration** | ε-greedy + power-law softmax + RND | PPO entropy bonus + (optionally) RND |
| **Parallelism** | Custom async actor-learner | PufferLib vectorized environments |
| **V_h^e, X_h, U_r** | Trained from same replay buffer | Trained from a **separate** replay buffer filled from the same rollouts |
| **Code location** | `learning_based/phase2/` | New: `learning_based/phase2_ppo/` (separate module) |

---

## 2. Current Architecture (DQN-style)

### 2.1 Equations Recap

Phase 2 implements equations (4)–(9) from the EMPO paper:

```
(4)  Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r · V_r(s')]
(5)  π_r(s)(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
(6)  V_h^e(s, g_h) ← E_{g_{-h}} E_{a_H ~ π_H(s,g)} E_{a_r ~ π_r(s)} E_{s'|s,a}
                      [U_h(s', g_h) + γ_h · V_h^e(s', g_h)]
(7)  X_h(s) ← E_{g_h}[V_h^e(s, g_h)^ζ]
(8)  U_r(s) ← -(E_h[X_h(s)^{-ξ}])^η
(9)  V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)}[Q_r(s, a_r)]
```

### 2.2 Current Training Loop

1. **Actor** collects transitions: (s, a_r, a_H, goals, s', transition_probs) → replay buffer
2. **Learner** samples batches from replay buffer and updates:
   - **V_h^e**: TD target from model-based successors
   - **X_h**: Monte Carlo target from V_h^e samples (E[V_h^e^ζ])
   - **U_r** (optional network): Target from X_h values
   - **Q_r**: Model-based Bellman target Q_r(s,a) ← γ_r · E[V_r(s')]
   - **V_r** (optional network): Target V_r = U_r + E_{π_r}[Q_r]

### 2.3 Key Properties of Current Approach

- **Off-policy**: Replay buffer allows training on old transitions
- **Model-based targets**: Uses `transition_probabilities()` to compute expected values
  over all successor states, reducing variance
- **Power-law policy**: π_r is derived from Q_r, preserving scale-invariance (paper Table 2)
- **Staged warm-up**: Breaks mutual dependencies by training networks sequentially with β_r=0

---

## 3. Proposed Architecture (PufferLib PPO)

### 3.1 High-Level Design

Replace the Q_r → π_r pathway with an explicit **(actor, critic)** pair trained by
PufferLib's PPO:

```
KEPT (trained separately):                    NEW (trained by PPO):
┌──────────────────────────────┐             ┌─────────────────────────┐
│  V_h^e(s, g_h)   [eq. 6]    │────────┐    │  π_r(s)    [actor]     │
│  X_h(s)           [eq. 7]    │────┐   │    │  V_r(s)    [critic]    │
│  U_r(s)           [eq. 8]    │──┐ │   │    └─────────────────────────┘
└──────────────────────────────┘  │ │   │                ▲
                                  │ │   │                │
                                  ▼ │   │         reward = U_r(s)
                             ┌─────────────┐
                             │ EMPO Reward  │
                             │   Wrapper    │
                             └─────────────┘
```

- **PPO actor** (π_r): Explicit policy network, outputs action distribution
- **PPO critic** (V_r): Value network, estimates V_r(s) = E[Σ γ^t U_r(s_t)]
- **Environment wrapper**: Intercepts the environment's native reward, replaces it with
  U_r(s) computed from the auxiliary networks
- **Auxiliary networks** (V_h^e, X_h, and optionally U_r network): Still trained separately,
  but now using transitions collected from PPO rollouts

### 3.2 What the PPO Path Does Not Use

The following components are **not used by the PPO trainer** but remain fully intact in
the existing DQN code path (nothing is deleted or modified):

- **Q_r network and Q_r target network** — PPO doesn't need Q-values
- **Power-law softmax policy derivation** (eq. 5) — replaced by explicit policy network
- **Off-policy replay buffer** for robot policy — PPO uses on-policy rollout buffers
- **Custom ε-greedy exploration** — PPO uses entropy bonus in the policy loss
- **z-space transformation** for Q_r — no longer needed in PPO path

### 3.3 What Gets Kept

- **V_h^e network and its training** (eq. 6) — still needed for power metric
- **X_h computation** (eq. 7) — still needed for U_r
- **U_r computation** (eq. 8) — still needed, now serves as the *environment reward* for PPO
- **Phase 1 human policy prior** — unchanged
- **Goal sampling** — unchanged
- **Staged warm-up** — adapted (see Section 7)
- **Model-based targets for V_h^e** — still valuable for variance reduction

### 3.4 Code Isolation Strategy

The PPO implementation is a **completely separate code path** from the existing DQN
trainer. This is the single most important implementation constraint:

#### Rules

1. **No modifications to any existing file** in `learning_based/phase2/` or
   `learning_based/multigrid/phase2/`. The existing DQN trainer, its networks,
   config, replay buffer, and all supporting code remain untouched.

2. **New code lives in new modules:**
   ```
   src/empo/learning_based/phase2_ppo/        # Base PPO trainer classes
   ├── __init__.py
   ├── config.py                               # PPOPhase2Config (standalone)
   ├── actor_critic.py                         # EMPOActorCritic network
   ├── env_wrapper.py                          # EMPOWorldModelEnv (Gymnasium wrapper)
   └── trainer.py                              # PPO training loop + aux net training
   ```

3. **Shared base classes are read-only dependencies.** The PPO trainer may *import* and
   *use* (but never modify) existing base classes like `BaseHumanGoalAchievementNetwork`,
   `BaseAggregateGoalAbilityNetwork`, `BaseIntrinsicRewardNetwork`, `Phase2Transition`,
   and `Phase2ReplayBuffer`. It may also reuse shared state encoders (`MultiGridStateEncoder`,
   `MultiGridGoalEncoder`).

4. **Separate entry points.** The PPO trainer is invoked through its own entry point
   (e.g., `examples/phase2/pufferlib_ppo_demo.py`), not via a flag on the existing trainer.

5. **Separate config class.** The PPO path uses `PPOPhase2Config` (defined in
   `phase2_ppo/config.py`), not `Phase2Config`. It contains only PPO-relevant parameters
   plus the shared theory parameters (γ_r, γ_h, ζ, ξ, η). It does NOT inherit from or
   modify `Phase2Config`.

6. **Existing tests must pass unmodified.** Any test in `tests/` that exercises the DQN
   trainer must continue to pass without changes after the PPO code is added.

#### Rationale

- **Code stability**: The DQN trainer is the production implementation. Modifying it
  risks regressions in a complex, interdependent training system.
- **Backward compatibility**: Users relying on the DQN trainer (configs, checkpoints,
  evaluation scripts) are not affected.
- **Clean comparison**: Having both trainers side-by-side makes A/B evaluation trivial.
- **Safe rollback**: If PPO doesn't work, no DQN code was harmed.

### 3.5 PufferLib Integration (Required)

The PPO implementation **uses PufferLib** (``pufferlib >= 3.0``) directly.
PufferLib provides the PPO training loop, vectorised environment management,
advantage computation (with V-trace / priority corrections), and logging.
There is no hand-written PPO loop — PufferLib's ``PuffeRL`` class is the
training driver.

#### PufferLib components used

| PufferLib module | Role in EMPO |
|---|---|
| ``pufferlib.emulation.GymnasiumPufferEnv`` | Wraps ``EMPOWorldModelEnv`` (a standard Gymnasium env) into a PufferLib-compatible env with shared-memory observation buffers |
| ``pufferlib.vector.make(env_creator, ...)`` | Creates a vectorised pool of ``num_envs`` wrapped environments. The initial EMPO PPO integration uses ``backend='Serial'``, which steps all environments in a single process. PufferLib also supports multiprocessing / shared-memory backends (e.g. ``backend='Multiprocessing'``) for true parallel env stepping; switching backend requires no trainer code changes. |
| ``pufferlib.pufferl.PuffeRL(config, vecenv, policy)`` | The core training class.  Manages the rollout buffer, advantage computation (CUDA kernel), PPO clipped-surrogate update, gradient clipping, learning-rate scheduling, and checkpointing |
| ``pufferlib.pytorch.sample_logits`` | Differentiable action sampling from logits, supporting Discrete, MultiDiscrete, and continuous action spaces |
| ``pufferlib.pytorch.layer_init`` | Orthogonal weight initialisation (CleanRL convention) |

#### Policy convention

In vanilla PufferLib, policies implement::

    def forward(self, observations, state=None) -> (logits, value)

where ``logits`` is a tensor of shape ``(batch, num_actions)`` for Discrete
or a list of tensors for MultiDiscrete, and ``value`` is ``(batch, 1)``.

In the EMPO PPO port, we **flatten the joint robot action** into a single
Discrete action index before it reaches PufferLib.  Concretely, for
``num_robots`` robots each with ``num_actions`` primitive actions, we define
``num_joint_actions = num_actions ** num_robots`` and expose a Discrete
action space of size ``num_joint_actions`` via the PufferLib env wrapper.
``EMPOActorCritic.forward()`` therefore returns a single logits tensor of
shape ``(batch, num_joint_actions)`` plus a value tensor of shape ``(batch, 1)``,
which is fully compatible with PufferLib's Discrete-policy interface even
though the underlying world_model uses a MultiDiscrete joint action internally.

For LSTM support, the policy can split into ``encode_observations`` and
``decode_actions`` methods and be wrapped with ``pufferlib.models.LSTMWrapper``.

#### Training loop

The outer training loop is::

    pufferl = PuffeRL(config, vecenv, policy)
    while pufferl.global_step < config['total_timesteps']:
        pufferl.evaluate()   # collect rollout via vectorised envs
        pufferl.train()      # PPO update (clipped surrogate, value loss, entropy)
        # EMPO-specific: train auxiliary networks from the same rollout data
        train_auxiliary_networks(...)
    pufferl.close()

PufferLib handles rollout collection, advantage estimation (GAE with V-trace),
minibatch sampling with prioritised experience, gradient accumulation, and
mixed-precision training.

#### What is EMPO-specific (not handled by PufferLib)

- **Intrinsic reward U_r(s)**: Computed inside ``EMPOWorldModelEnv.step()``
  and returned as the standard Gymnasium reward.  PufferLib treats it as
  any other reward signal.
- **Auxiliary network training**: V_h^e, X_h, U_r networks are trained
  from a separate replay buffer filled from the same rollout data.
  This happens *outside* PufferLib's ``train()`` call.
- **Warm-up staging**: Auxiliary networks can be pre-trained before
  enabling PPO updates, using the same ``evaluate()`` loop with a
  frozen random policy.
- **Auxiliary-network freezing**: Periodically deep-copying the auxiliary
  networks into frozen "target" copies for reward computation.

---

## 4. The Core Challenge: Intrinsic Reward Integration

### 4.1 The Problem

In standard PPO, the environment provides a reward signal r(s, a, s') at each time step.
PPO then computes advantages and value targets from this reward stream:

```
Return:    G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
Advantage: Â_t = G_t - V(s_t)   (or GAE version)
```

In EMPO, there is **no environment reward for the robot**. Instead, the robot's reward
is **U_r(s)**, defined by equation (8):

```
U_r(s) = -(E_h[X_h(s)^{-ξ}])^η
```

This depends on:
- **X_h(s)** = E_{g_h}[V_h^e(s, g_h)^ζ] — aggregate human power (eq. 7)
- **V_h^e(s, g_h)** — human's goal-achievement ability under robot policy (eq. 6)

These are outputs of *other neural networks that are being trained simultaneously*.
Hence U_r is:
1. **Non-stationary**: Changes as V_h^e and X_h improve
2. **Costly to compute**: Requires forward passes through V_h^e (for multiple goals and
   humans) and X_h
3. **Dependent on the robot policy**: V_h^e is conditioned on π_r, creating a feedback loop

### 4.2 Solution: EMPO Reward Wrapper

We introduce an **EMPORewardWrapper** that wraps the base environment and replaces the
environment's reward signal with the intrinsic reward U_r(s):

```python
class EMPORewardWrapper(gymnasium.Wrapper):
    """
    Wraps a WorldModel environment, replacing environment rewards
    with the EMPO intrinsic reward U_r(s_t).
    
    Convention: U_r is evaluated at the PRE-transition state s_t (not s_{t+1}).
    This matches the Bellman equation V_r(s) = U_r(s) + γ E[V_r(s')], and the
    GAE TD error δ_t = U_r(s_t) + γ V_r(s_{t+1}) − V_r(s_t) (see Appendix B).
    
    On each step:
    1. Records the pre-transition state s_t via get_state()
    2. Calls the base env.step(action) to get (obs, env_reward, done, truncated, info)
    3. Computes U_r(s_t) using the auxiliary networks
    4. Returns (obs, U_r(s_t), done, truncated, info)
    """
    
    def __init__(self, env, auxiliary_networks, config, device='cpu'):
        super().__init__(env)
        self.auxiliary_networks = auxiliary_networks  # V_h^e, X_h, (U_r network)
        self.config = config
        self.device = device
        # Frozen copies of auxiliary networks for reward stability
        self._frozen_aux = None
        self._freeze_interval = config.reward_freeze_interval
        self._steps_since_freeze = 0
    
    def step(self, action):
        # Capture pre-transition state for U_r(s_t) computation
        pre_state = self.env.get_state()
        
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute intrinsic reward at the PRE-transition state
        u_r = self._compute_u_r(pre_state)
        
        # Store original env reward in info for debugging/logging
        info['env_reward'] = env_reward
        info['u_r'] = u_r
        
        return obs, u_r, terminated, truncated, info
    
    def _compute_u_r(self, state):
        """Compute U_r(s) from auxiliary networks."""
        with torch.no_grad():
            nets = self._frozen_aux or self.auxiliary_networks
            # Option A: U_r network (if enabled)
            if nets.u_r is not None:
                _, u_r = nets.u_r.forward(state, self.env, self.device)
                return u_r.item()
            # Option B: Compute from X_h values directly
            x_h_vals = []
            for human_idx in self.env.human_agent_indices:
                x_h = nets.x_h.forward(state, self.env, human_idx, self.device)
                x_h_vals.append(x_h.item())
            y = np.mean([x ** (-self.config.xi) for x in x_h_vals])
            return -(y ** self.config.eta)
```

### 4.3 Reward Stability Strategies

Since U_r changes as the auxiliary networks train, we need strategies to prevent
the PPO advantage estimates from becoming meaningless:

#### Strategy A: Frozen Reward Networks (Recommended)

Periodically freeze the auxiliary networks used for reward computation:

```
Timeline:
├── PPO rollout 1 ──┤── PPO rollout 2 ──┤── PPO rollout 3 ──┤
     ▲ freeze aux         ▲ freeze aux         ▲ freeze aux
     │                    │                    │
   aux train            aux train            aux train
```

During each PPO rollout, U_r is computed from **frozen** copies of V_h^e, X_h. After the
rollout (and PPO update), the auxiliary networks are trained on the collected transitions,
then a new frozen copy is made for the next rollout.

**Benefits:**
- U_r is consistent within a rollout → valid advantage estimates
- Auxiliary networks still improve across rollouts
- Natural batching: collect rollout → update PPO → update auxiliary nets → freeze → repeat

#### Strategy B: Reward Normalization

Even with frozen reward networks, U_r values may drift across rollouts. Apply running
normalization:

```python
# Normalize U_r using running statistics
u_r_normalized = (u_r - running_mean_u_r) / (running_std_u_r + eps)
```

PufferLib supports reward normalization natively, which helps PPO handle the changing
reward scale.

#### Strategy C: Short Rollout Horizons

Use shorter rollout horizons (e.g., 64–128 steps instead of 2048) to limit the impact
of reward non-stationarity within a rollout. This trades off variance for reward
consistency.

### 4.4 Reward Computation Cost

Computing U_r(s) requires forward passes through multiple networks for every env step.
For a state s with H humans and G sampled goals per human:

- **Without U_r/X_h networks**: H × G forward passes through V_h^e, plus aggregation
- **With X_h network**: H forward passes through X_h
- **With U_r network**: 1 forward pass through U_r

**Recommendation:** Use the U_r network (set `u_r_use_network=True`) to amortize the
cost. The U_r network is cheap (single forward pass) and can be trained to track the
analytical U_r computed from X_h samples.

Alternatively, since PufferLib vectorizes environments, the U_r computation can be
batched across all vectorized environments simultaneously:

```python
# Batched U_r computation across N vectorized environments
states = [env.get_state() for env in vec_envs]
u_r_batch = u_r_network.forward_batch(states, envs, device)  # (N,)
```

---

## 5. Environment Wrapper Design

### 5.1 PufferLib-Compatible Wrapper

PufferLib requires Gymnasium-compatible environments. Our wrapper must bridge
the WorldModel interface to Gymnasium and inject the intrinsic reward:

```python
import gymnasium
import pufferlib.emulation

class EMPOWorldModelEnv(gymnasium.Env):
    """
    PufferLib-compatible wrapper for EMPO Phase 2 training.
    
    Wraps a MultiGridEnv (WorldModel) and handles:
    1. Gymnasium API compliance (obs, reward, done, truncated, info)
    2. Intrinsic reward injection (U_r replaces env reward)
    3. Human agent simulation (humans act according to policy prior)
    4. Goal sampling and management
    5. Observation encoding for the policy network
    """
    
    def __init__(self, world_model, human_policy_prior, goal_sampler,
                 human_agent_indices, robot_agent_indices, config,
                 auxiliary_networks=None, device='cpu'):
        super().__init__()
        self.world_model = world_model
        self.human_policy_prior = human_policy_prior
        self.goal_sampler = goal_sampler
        self.human_agent_indices = human_agent_indices
        self.robot_agent_indices = robot_agent_indices
        self.config = config
        self.auxiliary_networks = auxiliary_networks
        self.device = device
        
        # Define observation and action spaces
        self.observation_space = self._build_observation_space()
        
        # Action space: product of per-robot actions
        num_actions = world_model.action_space.n
        num_robots = len(robot_agent_indices)
        if num_robots == 1:
            self.action_space = gymnasium.spaces.Discrete(num_actions)
        else:
            self.action_space = gymnasium.spaces.MultiDiscrete(
                [num_actions] * num_robots
            )
        
        # Episode state
        self._step_count = 0
        self._goals = {}
        self._goal_weights = {}
    
    def reset(self, seed=None, options=None):
        self.world_model.reset(seed=seed)
        state = self.world_model.get_state()
        self._step_count = 0
        self._sample_goals(state)
        obs = self._encode_observation(state)
        return obs, {}
    
    def step(self, action):
        state = self.world_model.get_state()
        
        # Sample human actions from policy prior
        human_actions = self._sample_human_actions(state)
        
        # Build joint action
        joint_action = self._build_joint_action(action, human_actions)
        
        # Step the world model (returns terminated=True when no transitions exist)
        _, _, wm_terminated, _, _ = self.world_model.step(joint_action)
        next_state = self.world_model.get_state()
        
        # Compute intrinsic reward U_r(state) at pre-transition state s_t
        u_r = self._compute_u_r(state)
        
        # Episode termination
        self._step_count += 1
        truncated = self._step_count >= self.config.steps_per_episode
        terminated = wm_terminated  # Propagate from WorldModel (e.g., no valid transitions)
        
        # Resample goals with probability goal_resample_prob
        if random.random() < self.config.goal_resample_prob:
            self._sample_goals(next_state)
        
        obs = self._encode_observation(next_state)
        info = {
            'u_r': u_r,
            'state': state,              # Pre-transition state for auxiliary training
            'next_state': next_state,    # Post-transition state for TD targets
            'goals': self._goals.copy(),
            'goal_weights': self._goal_weights.copy(),
            'human_actions': human_actions,
        }
        
        return obs, u_r, terminated, truncated, info
```

### 5.2 Observation Space

The observation must encode all information the robot policy needs. Options:

**Option A: Raw state encoding** (flatten the grid + agent features):

```python
def _build_observation_space(self):
    # Use the existing MultiGridStateEncoder's output shape
    feature_dim = self.config.state_feature_dim
    return gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
    )

def _encode_observation(self, state):
    # Use the shared state encoder (same as current Q_r)
    with torch.no_grad():
        features = self.state_encoder.encode(state, self.world_model, self.device)
    return features.cpu().numpy()
```

**Option B: Structured observation** (dictionary of grid, agents, goals):

```python
def _build_observation_space(self):
    H, W = self.world_model.height, self.world_model.width
    return gymnasium.spaces.Dict({
        'grid': gymnasium.spaces.Box(0, 255, (H, W, C), dtype=np.uint8),
        'agent_positions': gymnasium.spaces.Box(0, max(H,W), (N, 2)),
        'agent_features': gymnasium.spaces.Box(-np.inf, np.inf, (N, F)),
    })
```

**Recommendation:** Option A (pre-encoded features) for initial implementation since it
reuses the existing `MultiGridStateEncoder` and avoids duplicating encoding logic in
the PPO policy network. The shared encoder weights can be frozen or jointly trained.

### 5.3 Transition Data Collection for Auxiliary Networks

PPO rollouts collect (obs, action, reward, done) tuples. But the auxiliary networks
(V_h^e, X_h) need richer data: states, goals, human actions, transition probabilities.

We use the `info` dict to carry this data out of the environment wrapper:

```python
# In the EMPOWorldModelEnv.step():
# Compute model-based transition probabilities from the WorldModel.
# This matches Phase2Transition.transition_probs_by_action:
# Dict[int, List[Tuple[float, HashableState]]]
# We iterate over all *robot action indices* and, for each, call
# transition_probabilities with the sampled human actions fixed.
#
# Single-robot case (Discrete): indices are 0..|A|-1.
# Multi-robot case (MultiDiscrete): indices are flattened joint-action
# indices 0..|A|^N - 1, which _build_joint_action maps back to a concrete
# joint action tuple for the WorldModel.
import numpy as np
from gymnasium import spaces

if isinstance(self.action_space, spaces.Discrete):
    num_robot_actions = self.action_space.n
elif isinstance(self.action_space, spaces.MultiDiscrete):
    num_robot_actions = int(np.prod(self.action_space.nvec))
else:
    raise TypeError(f"Unsupported action_space type: {type(self.action_space)}")

transition_probs_by_action = {}
for a_r_idx in range(num_robot_actions):
    joint = self._build_joint_action(a_r_idx, human_actions)
    trans = self.world_model.transition_probabilities(state, joint)
    if trans is not None:
        transition_probs_by_action[a_r_idx] = trans

info = {
    'state': state,                          # Pre-transition WorldModel state
    'next_state': next_state,                # Post-transition state for TD targets
    'goals': goals,                          # {human_idx: goal} (for V_h^e)
    'goal_weights': goal_weights,
    'human_actions': human_actions,
    'transition_probs': transition_probs_by_action,  # Model-based V_h^e targets
}
```

After each PPO rollout, the training loop extracts this auxiliary data from the info
dicts and trains V_h^e, X_h, and U_r using the existing loss functions (see Section 8).

### 5.4 PufferLib Vectorization

PufferLib provides vectorized environment execution:

```python
import pufferlib.vector

def make_env():
    env = create_multigrid_world_model(...)
    return EMPOWorldModelEnv(
        world_model=env,
        human_policy_prior=human_policy_prior,
        goal_sampler=goal_sampler,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        config=config,
        auxiliary_networks=frozen_aux_nets,
    )

vec_env = pufferlib.vector.make(
    make_env,
    backend=pufferlib.vector.Multiprocessing,  # or Serial for debugging
    num_envs=num_envs,
)
```

**Important:** Each vectorized environment instance gets its own *frozen copy* of the
auxiliary networks. These are synced from the main process periodically (see Section 7).

---

## 6. Network Architecture Changes

All new network classes below live in `learning_based/phase2_ppo/` (or the
environment-specific `learning_based/multigrid/phase2_ppo/`). They do **not**
modify or extend the existing DQN-path network classes.

### 6.1 New: Robot Policy Network (Actor)

```python
class EMPORobotPolicyNetwork(nn.Module):
    """
    Explicit robot policy network for PPO.
    
    Replaces the implicit policy π_r ∝ (-Q_r)^{-β_r}.
    
    Architecture:
        state → SharedEncoder → MLP → action_logits
    
    The SharedEncoder is the same as used by V_h^e and X_h,
    enabling shared representation learning.
    """
    
    def __init__(self, state_encoder, hidden_dim, num_actions, num_robots):
        super().__init__()
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.num_robots = num_robots
        self.num_action_combinations = num_actions ** num_robots
        
        self.policy_head = nn.Sequential(
            nn.Linear(state_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_action_combinations),
        )
    
    def forward(self, obs):
        """Returns action logits (unnormalized log-probabilities)."""
        features = self.state_encoder(obs)
        return self.policy_head(features)
```

Note: The actor-only class does not provide `get_action_and_value()` because PPO
requires the critic's value estimate. Use the combined `EMPOActorCritic` (Section 6.3)
for PPO training, which provides the full `get_action_and_value()` interface.

### 6.2 New: Robot Value Network (Critic)

```python
class EMPORobotValueNetwork(nn.Module):
    """
    Robot value network (critic) for PPO.
    
    Estimates V_r(s) = E[Σ_t γ^t U_r(s_t) | s_0 = s].
    
    This replaces the current V_r computation (eq. 9):
        V_r(s) = U_r(s) + E_{π_r}[Q_r(s,a)]
    
    With PPO, V_r is trained via the value loss (MSE to returns).
    """
    
    def __init__(self, state_encoder, hidden_dim):
        super().__init__()
        self.state_encoder = state_encoder
        
        self.value_head = nn.Sequential(
            nn.Linear(state_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        """Returns V_r(s) estimate (scalar)."""
        features = self.state_encoder(obs)
        return self.value_head(features).squeeze(-1)
```

### 6.3 Combined Actor-Critic for PufferLib

PufferLib/CleanRL PPO expects a single module with both actor and critic:

```python
class EMPOActorCritic(nn.Module):
    """Combined actor-critic for PufferLib PPO training."""
    
    def __init__(self, state_encoder, hidden_dim, num_actions, num_robots):
        super().__init__()
        self.state_encoder = state_encoder  # Can be shared or separate
        
        self.actor = nn.Sequential(
            nn.Linear(state_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions ** num_robots),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        features = self.state_encoder(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_value(self, obs):
        features = self.state_encoder(obs)
        return self.critic(features).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        features = self.state_encoder(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value
```

### 6.4 Kept: Auxiliary Networks

The following are unchanged in architecture (only their training loop changes):

- **V_h^e (BaseHumanGoalAchievementNetwork)**: Still trained with TD targets, still uses
  shared state encoder, still depends on π_r (now from explicit policy network)
- **X_h (BaseAggregateGoalAbilityNetwork)**: Still trained from V_h^e^ζ Monte Carlo targets
- **U_r (BaseIntrinsicRewardNetwork)**: Still trained from X_h^{-ξ} aggregation. If
  `u_r_use_network=True`, the U_r network amortizes the cost of computing U_r in the
  reward wrapper
- **Shared state encoder**: Shared between V_h^e, X_h, and the actor-critic. Training
  gradients flow through V_h^e and X_h losses; the PPO loss trains the actor-critic
  heads but can optionally also flow through the shared encoder

### 6.5 Networks Not Used by PPO Path

The following existing networks are **not used** by the PPO trainer (but remain in the
codebase, untouched, for the DQN path):

- **Q_r (BaseRobotQNetwork)** — PPO doesn't need Q-values
- **Q_r target** — PPO doesn't need target networks for robot Q
- **V_r (BaseRobotValueNetwork) as currently defined** — replaced by PPO critic
- **V_r target** — PPO handles value estimation internally

---

## 7. Warm-up and Mutual Dependency Handling

### 7.1 The Mutual Dependency Under PPO

The feedback loop still exists but takes a different form:

```
Current (DQN):   V_h^e ← π_r ← Q_r ← V_r ← U_r ← X_h ← V_h^e
Proposed (PPO):  V_h^e ← π_r ← PPO(U_r) ← X_h ← V_h^e
                                    ↑
                              π_r feeds back into V_h^e
```

The dependency chain is actually **shorter** because PPO eliminates the Q_r → V_r
indirection. The loop is now:
1. V_h^e depends on π_r (equation 6 uses E_{a_r ~ π_r})
2. X_h depends on V_h^e (equation 7)
3. U_r depends on X_h (equation 8)
4. π_r is updated by PPO using U_r as reward → closing the loop

### 7.2 Adapted Warm-up Schedule

The warm-up stages are adapted for PPO. Durations are measured in **training steps
(gradient updates)**, matching the existing Phase 2 convention (see `Phase2Config`):

| Stage | Duration (training steps) | Active Components | π_r | Notes |
|-------|---------------------------|-------------------|-----|-------|
| 0 | W₀ training steps | V_h^e only | Uniform random | Same as current Stage 0 |
| 1 | W₁ training steps | V_h^e + X_h | Uniform random | Same as current Stage 1 |
| 2 | W₂ training steps | V_h^e + X_h + U_r | Uniform random | Same as current Stage 2 |
| 3 | R training steps | V_h^e + X_h + U_r + PPO | PPO training begins | **NEW**: PPO replaces Q_r+V_r warmup |
| 4 | Remainder | All | PPO (full training) | Continued joint training |

**Key differences from current warm-up:**

- **Stages 0–2 are identical**: V_h^e, X_h, and U_r are warmed up with a uniform random
  robot policy (not using PPO)
- **Stage 3 replaces Stages 3+4+5**: Instead of warming up Q_r, then V_r, then ramping β_r,
  we simply start PPO training. PPO naturally handles exploration via entropy bonus
- **No β_r ramp-up needed**: PPO's entropy coefficient serves a similar purpose. The entropy
  bonus can be annealed from high (exploration) to low (exploitation) over training

### 7.3 How to Simulate Uniform Random Robot During Warm-up

During warm-up (Stages 0–2), the robot acts uniformly at random. With PufferLib, this
is achieved by:

**Option A: Separate warm-up part without PPO**

Run warm-up as a separate pre-training part that doesn't use PufferLib at all:

```python
# Part A: Warm up auxiliary networks with random robot
for step in range(total_warmup_steps):
    state = env.get_state()
    robot_action = env.action_space.sample()  # Random
    human_actions = sample_human_actions(state, goals)
    env.step(build_joint_action(robot_action, human_actions))
    # Train V_h^e, X_h, U_r from this transition
    train_auxiliary_networks(transition)

# Part B: PufferLib PPO training with intrinsic reward
# ... (Section 8)
```

**Option B: PPO with maximum entropy (high entropy coefficient)**

Start PPO from the beginning but with a very high entropy coefficient, effectively
making the policy uniform:

```python
# entropy_coef starts very high → near-uniform policy
entropy_schedule = lambda step: 10.0 if step < warmup_steps else 0.01
```

**Recommendation:** Option A is cleaner—it matches the current approach more closely and
avoids the complexity of tuning entropy coefficients during warm-up. The warm-up code
can reuse most of the existing BasePhase2Trainer logic.

### 7.4 Frozen Auxiliary Networks During PPO Rollouts

During each PPO rollout, the reward wrapper uses **frozen** copies of V_h^e, X_h, U_r.
This ensures:

1. Rewards are consistent within a rollout (valid advantage estimates)
2. Auxiliary network updates don't invalidate in-progress rollout data
3. The PPO value function tracks a locally-stationary reward signal

The freeze-update cycle:

```
┌─ Freeze aux nets ─── PPO rollout ─── PPO update ─── Train aux nets ─── Freeze ─┐
│                    (U_r from frozen)  (policy update) (V_h^e, X_h, U_r)          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                    (repeat)
```

---

## 8. Training Loop Redesign

The PPO training loop is implemented in `learning_based/phase2_ppo/trainer.py`,
completely independent of the existing `learning_based/phase2/trainer.py`. It imports
auxiliary network base classes but does not modify them.

### 8.1 Main Training Loop

```python
def train_empo_ppo(
    world_model_factory,
    human_policy_prior,
    goal_sampler,
    human_agent_indices,
    robot_agent_indices,
    config,  # PPOPhase2Config (standalone; duplicates theory params like gamma_h, gamma_r, zeta, xi, eta)
    num_envs=16,
    device='cuda',
):
    # ================================================================
    # Part A: Warm-up (auxiliary networks only, random robot)
    # ================================================================
    auxiliary_networks = create_auxiliary_networks(config, device)
    warmup_trainer = AuxiliaryWarmupTrainer(
        world_model_factory, auxiliary_networks, config,
        human_policy_prior, goal_sampler,
        human_agent_indices, robot_agent_indices,
    )
    warmup_trainer.train(num_steps=config.get_total_warmup_steps())
    
    # ================================================================
    # Part B: PPO training with intrinsic reward
    # ================================================================
    
    # Create actor-critic
    state_encoder = auxiliary_networks.v_h_e.state_encoder  # Shared encoder
    actor_critic = EMPOActorCritic(
        state_encoder, config.hidden_dim,
        config.num_actions, config.num_robots,
    ).to(device)
    
    # Freeze auxiliary networks for reward computation
    trainer = PPOPhase2Trainer(actor_critic, auxiliary_networks, config)
    trainer.freeze_auxiliary_networks()
    
    # Create vectorized PufferLib environments
    def make_env(buf=None, seed=0):
        world_model = world_model_factory.create()
        env = EMPOWorldModelEnv(
            world_model, human_policy_prior, goal_sampler,
            human_agent_indices, robot_agent_indices,
            config, auxiliary_networks=auxiliary_networks,
        )
        return pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: env, buf=buf, seed=seed,
        )
    
    vecenv = pufferlib.vector.make(
        make_env, num_envs=num_envs, backend="Serial",
    )
    
    # PufferLib-driven PPO training loop
    puffer_config = config.to_pufferlib_config()
    pufferl = pufferlib.pufferl.PuffeRL(puffer_config, vecenv, actor_critic)
    
    # Auxiliary network replay buffer (for V_h^e, X_h, U_r training)
    aux_replay_buffer = Phase2ReplayBuffer(capacity=config.aux_buffer_size)
    
    for iteration in range(config.num_ppo_iterations):
        # --- Step 1: PufferLib rollout (vectorised env stepping) ---
        pufferl.evaluate()
        
        # Extract auxiliary training data from rollout info dicts
        trainer._collect_aux_data_from_rollout(pufferl, vecenv)
        
        # --- Step 2: PufferLib PPO update (GAE + clipped surrogate) ---
        pufferl.train()
        
        # --- Step 3: Auxiliary network training ---
        # Train V_h^e, X_h, U_r using transitions from the rollout
        for aux_step in range(config.aux_training_steps_per_iteration):
            step_losses = trainer.train_auxiliary_step(world_model)
        
        # --- Step 4: Freeze updated auxiliary networks for next rollout ---
        trainer.freeze_auxiliary_networks()
        trainer._sync_aux_nets_to_envs(vecenv)
    
    return actor_critic, auxiliary_networks
```

### 8.2 Auxiliary Network Training

The auxiliary networks (V_h^e, X_h, U_r) are trained similarly to the current approach,
but with important differences:

**V_h^e Training (eq. 6):**
- Target: E_{a_r ~ π_r}[U_h(s', g_h) + γ_h · V_h^e_target(s', g_h)]
- **Change:** π_r is now sampled from the explicit policy network, not from Q_r
- The explicit policy network makes this cleaner: just call `actor_critic.get_action_probs(obs)`
- Model-based targets still available (transition_probabilities from WorldModel)

**X_h Training (eq. 7):**
- Target: V_h^e(s, g_h)^ζ (Monte Carlo sample for goal g_h)
- **No change** in training logic

**U_r Training (eq. 8):**
- Target: -(E_h[X_h(s)^{-ξ}])^η
- **No change** in training logic

### 8.3 Computing V_h^e Targets Under PPO Policy

Currently, V_h^e targets are computed as:

```
target = E_g E_{a_H ~ π_H} E_{a_r ~ π_r} E_{s'|s,a} [U_h(s', g_h) + γ_h · V_h^e(s', g_h)]
```

The expectation over a_r ~ π_r currently uses the power-law policy derived from Q_r.
With PPO, π_r comes from the actor network:

```python
def compute_v_h_e_target(state, goal, human_idx, actor_critic, v_h_e_target, env):
    """Compute V_h^e TD target using the PPO policy for robot actions."""
    
    # Get robot policy from actor network
    obs = encode_observation(state, env)
    with torch.no_grad():
        logits = actor_critic.actor(actor_critic.state_encoder(obs))
        pi_r = torch.softmax(logits, dim=-1)  # (num_action_combinations,)
    
    # For each robot action, compute expected successor value
    target = 0.0
    for a_r_idx in range(num_action_combinations):
        # Get transition probabilities for this action
        trans_probs = env.transition_probabilities(state, joint_actions_with(a_r_idx))
        
        for prob, next_state in trans_probs:
            u_h = goal.is_achieved(next_state)  # 0 or 1
            v_h_e_next = v_h_e_target(next_state, env, human_idx, goal)
            target += pi_r[a_r_idx] * prob * (u_h + gamma_h * v_h_e_next)
    
    return target
```

This is actually **simpler** than the current approach because π_r is directly available
as a probability vector, rather than needing to derive it from Q-values.

### 8.4 Syncing Frozen Networks to Vectorized Environments

When running multiple environments in separate processes, frozen network weights must
be synced to each environment's reward wrapper. Options:

**Option A: Shared memory (recommended for multiprocessing backend)**

```python
# Store frozen weights in shared memory
shared_weights = multiprocessing.Manager().dict()
shared_weights['v_h_e'] = v_h_e_target.state_dict()
shared_weights['x_h'] = x_h_target.state_dict()

# Each environment checks for updates periodically
class EMPOWorldModelEnv:
    def _maybe_sync_networks(self):
        if self._shared_weights_version != self._local_version:
            self.auxiliary_networks.v_h_e.load_state_dict(
                self._shared_weights['v_h_e'])
            self._local_version = self._shared_weights_version
```

**Option B: Rebuild environments (simpler but slower)**

After each auxiliary network update, recreate the vectorized environments with new
frozen weights. This is simpler but has overhead from environment reconstruction.

**Option C: Centralized reward computation (avoids sync entirely)**

Don't compute U_r inside the environment wrapper. Instead:
1. Environments return raw states in `info`
2. After collecting the rollout, compute U_r for all states centrally
3. Inject the computed U_r as the reward into the rollout buffer

```python
# After rollout collection:
states = [info['state'] for info in rollout_infos]
u_r_rewards = u_r_network.forward_batch(states, env, device)
rollout_buffer.rewards = u_r_rewards
```

This avoids the sync problem entirely and allows batched GPU computation of U_r.
**This is the recommended approach for performance.**

---

## 9. Theoretical Considerations

### 9.1 Loss of Power-Law Scale Invariance

The current policy π_r ∝ (-Q_r)^{-β_r} satisfies scale-invariance properties important
for the power metric formulation (paper Table 2). A standard PPO policy (softmax over
logits) does **not** preserve this property.

**Impact assessment:** The scale-invariance ensures that the robot's behavior doesn't
change if all Q-values are scaled by a constant. With PPO, the policy is parameterized
independently of value scales, so this property is lost. However:

- PPO's reward normalization provides a different kind of scale robustness
- With a sufficiently expressive network and stable training, PPO can approximate the fixed-point policy that solves the Phase 2 equations for a given U_r
- In practice, the impact of losing exact power-law scale invariance on the approximated solution may be small if training is stable (this should be evaluated empirically)

**Mitigation (optional):** Parameterize the PPO policy head using a power-law form:

```python
class PowerLawPolicyHead(nn.Module):
    """Policy head that preserves power-law structure."""
    def __init__(self, input_dim, num_actions, beta_r=10.0):
        super().__init__()
        self.beta_r = beta_r
        self.q_head = nn.Linear(input_dim, num_actions)  # Predicts pseudo-Q values
    
    def forward(self, features):
        q_pseudo = -F.softplus(self.q_head(features))  # Ensure negative
        log_probs = -self.beta_r * torch.log(-q_pseudo + 1e-8)
        return log_probs  # These are logits for Categorical distribution
```

This gives PPO's optimizer the power-law inductive bias while still using standard PPO
updates. The `beta_r` could be fixed or learnable.

### 9.2 On-Policy vs. Off-Policy

PPO is on-policy: it discards data after each update. This means:

- **V_h^e cannot be trained from the PPO rollout buffer** directly (it needs more samples)
- **Solution:** Maintain a separate replay buffer for auxiliary networks, populated from
  PPO rollouts. The auxiliary networks train off-policy from this buffer
- **Alternative:** Use a "dual-buffer" approach where PPO's on-policy buffer trains the
  robot policy, and a separate replay buffer (filled with the same transitions) trains
  auxiliary networks

### 9.3 Value Function Interpretation

Current V_r (eq. 9): V_r(s) = U_r(s) + E_{π_r}[Q_r(s,a)]

PPO V_r: V_r(s) = E[Σ_{t≥0} γ^t U_r(s_t) | s_0 = s, π_r]

These are equivalent in the fixed-point: if U_r is the immediate reward at each step,
then the Bellman equation for V_r under policy π_r is exactly:

```
V_r(s) = U_r(s) + γ E_{a ~ π_r} E_{s'|s,a} [V_r(s')]
       = U_r(s) + E_{a ~ π_r} [Q_r(s, a)]
```

So the PPO critic is learning the **same quantity** as current V_r, just via a different
algorithm (GAE returns vs. Bellman equation regression).

### 9.4 Convergence Under Non-Stationary Rewards

Since U_r changes as auxiliary networks improve, PPO is optimizing against a moving
target. This is a known challenge in intrinsic motivation RL (cf. RND, ICM).

**Convergence argument:**
- If auxiliary network updates are slow relative to PPO updates, the reward is approximately
  stationary within each PPO iteration (similar to "inner-outer loop" optimization)
- The freeze-update-freeze cycle (Section 7.4) formalizes this separation
- In the limit, if auxiliary networks converge, U_r becomes stationary and PPO converges
  to a fixed-point approximate solution for the MDP induced by that fixed U_r

**Recommendation:** Use a lower learning rate for auxiliary networks relative to PPO,
creating a natural time-scale separation.

---

## 10. Migration Plan

> **Invariant: No existing Python source files are modified.**
> Every task below creates NEW files or adds NEW tests. If any task seems to
> require editing a file in `learning_based/phase2/` or `learning_based/multigrid/phase2/`,
> that is a design error — stop and refactor the approach. Run `git diff --stat`
> after each migration step to confirm zero changes to existing `.py` files.
> Documentation files (`.md`) may have new sections appended but existing content
> must not be altered.

### Migration Step 1: Foundation (Estimated: 2–3 weeks)

- [x] **10.1** Create `learning_based/phase2_ppo/` and `learning_based/multigrid/phase2_ppo/` modules
- [x] **10.2** Add PufferLib dependency (`pip install pufferlib`)
- [x] **10.3** Implement `PPOPhase2Config` in `phase2_ppo/config.py` (standalone, not extending `Phase2Config`)
- [x] **10.4** Implement `EMPOWorldModelEnv` in `phase2_ppo/env_wrapper.py` (Gymnasium wrapper with human agent simulation); `MultiGridWorldModelEnv` in `multigrid/phase2_ppo/env_wrapper.py`
- [x] **10.5** Implement `EMPOActorCritic` in `phase2_ppo/actor_critic.py` (combined actor-critic network)
- [x] **10.6** Unit tests in `tests/test_phase2_ppo.py` and `tests/test_multigrid_phase2_ppo.py`: env wrapper produces valid observations, rewards, dones
- [x] **10.7** Verify: random policy through wrapper matches current random rollout behavior
- [x] **10.8** Run ALL existing DQN-path tests → confirm zero regressions

### Migration Step 2: Intrinsic Reward Integration (Estimated: 2–3 weeks)

- [x] **10.9** Implement centralized U_r reward computation in `phase2_ppo/env_wrapper.py` (via `_compute_u_r()`, using Option C from Section 5.4)
- [x] **10.10** Implement auxiliary network training loop (V_h^e, X_h, U_r) in `phase2_ppo/trainer.py` (via `train_auxiliary_step()`)
- [x] **10.11** Implement freeze/sync cycle for auxiliary networks (via `freeze_auxiliary_networks()` with `*_target` copies)
- [x] **10.12** Unit tests: U_r computation, auxiliary training, freeze/sync verified in `test_phase2_ppo.py`
- [x] **10.13** Verify: auxiliary networks converge during warm-up with random robot
- [x] **10.14** Run ALL existing DQN-path tests → confirm zero regressions

### Migration Step 3: PPO Training (Estimated: 2–3 weeks)

- [x] **10.15** Implement PPO training loop in `phase2_ppo/trainer.py` (uses PufferLib PuffeRL)
- [x] **10.16** Implement warm-up → PPO transition (staged warm-up with `get_warmup_stage()`)
- [x] **10.17** End-to-end test: `test_pufferlib_training_loop` and `test_warmup_runs_before_ppo` in `test_phase2_ppo.py`
- [ ] **10.18** Compare learned behavior with current DQN-trained policy (both running side-by-side)
- [x] **10.19** Run ALL existing DQN-path tests → confirm zero regressions

### Migration Step 4: Optimization and Scaling (Estimated: 2–3 weeks)

- [ ] **10.20** Implement batched U_r computation across vectorized environments
- [ ] **10.21** Profile training loop, optimize bottlenecks
- [ ] **10.22** Test on larger environments (7×7, 3 humans, 1 robot)
- [x] **10.23** TensorBoard integration: log PPO metrics alongside auxiliary metrics (via `_tb_writer`)
- [x] **10.24** Checkpoint/resume support for PPO + auxiliary networks (via `save_checkpoint`/`load_checkpoint`)

### Migration Step 5: Documentation (Estimated: 1 week)

- [ ] **10.25** Add PPO-specific section to `docs/WARMUP_DESIGN.md` (append, don't modify existing content)
- [ ] **10.26** Add PPO API section to `docs/API.md` (append, don't modify existing content)
- [x] **10.27** Create example scripts: `examples/phase2/phase2_ppo_demo.py`, `examples/phase2/phase2_ppo_asymmetric_freeing.py`, `examples/phase1/phase1_ppo_demo.py`
- [ ] **10.28** Update `docs/plans/ppo_a3c_considerations.md` with implementation notes
- [x] **10.29** Final regression run: ALL existing tests pass unmodified

---

## 11. Open Questions and Risks

### 11.1 Open Questions

1. **Shared encoder gradients:** Should PPO loss gradients flow through the shared state
   encoder, or should the encoder be trained only via auxiliary losses? Shared training
   could help the encoder capture policy-relevant features, but could also destabilize
   auxiliary network training.

2. **U_r computation location:** Should U_r be computed inside each environment (distributed)
   or centrally after rollout collection (batched)? Centralized is more efficient but
   adds complexity to the data pipeline.

3. **Rollout length tuning:** What rollout length balances PPO sample efficiency with
   reward non-stationarity? Shorter = more stable U_r, longer = better advantage estimates.

4. **Entropy coefficient schedule:** Should entropy start high and anneal (mimicking β_r
   ramp-up), or use PPO's default constant entropy coefficient?

5. **Model-based vs. sample-based advantages:** Should we use model-based V_r targets
   (via transition_probabilities) for PPO's value function training, or rely on standard
   sample-based GAE? Model-based reduces variance but adds computation.

6. **Multi-robot action spaces:** With N robots, the joint action space is |A|^N. For N > 2,
   this becomes large. Should we use independent PPO policies per robot (IPPO) or
   maintain the joint action space?

### 11.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Non-stationary U_r destabilizes PPO | Medium | High | Freeze/sync cycle, reward normalization |
| Loss of scale-invariance hurts policy quality | Low | Medium | Power-law policy head option |
| Auxiliary network training insufficient from on-policy data | Medium | Medium | Separate replay buffer, higher buffer capacity |
| PufferLib vectorization incompatible with WorldModel API | Low | High | Fallback to serial execution |
| PPO sample efficiency worse than DQN for model-based setting | Medium | Medium | Keep DQN trainer as fallback |
| Reward computation cost slows training | Medium | Medium | U_r network amortization, batched computation |

### 11.3 Success Criteria

1. **Backward compatibility**: Zero modifications to existing Python source files in
   `learning_based/phase2/` and `learning_based/multigrid/phase2/`; all existing tests
   pass unmodified. `git diff --stat` against the base branch shows only *new* `.py` files
   in `phase2_ppo/` directories, new tests in `tests/`, and documentation appendages.
2. PPO-trained robot achieves similar or better human empowerment scores as DQN-trained
   robot on the standard 5×5 and 7×7 test environments
3. Training wall-clock time is comparable or faster (due to PufferLib parallelism)
4. Training is stable (no divergence, reasonable loss curves)
5. Code is maintainable: the PPO trainer is cleaner and simpler than the current
   DQN trainer (fewer networks to manage, no Q_r/V_r target networks for the robot)
6. Both trainers can coexist: a user can run DQN training and PPO training from the
   same codebase, potentially even on the same environment, for direct comparison

---

## 12. PPO for Phase 1 Training

### 12.1 Overview

Phase 1 currently uses a **DQN-style approach** to approximate goal-conditioned human
policy priors: a Q-network Q(s, a, g) is trained via experience replay with TD targets,
and the policy prior π_h(a|s, g) is derived as a Boltzmann softmax over Q-values:

```
π_h(a | s, g) = softmax(β_h · Q(s, a, g))
```

PufferLib's PPO could also be used here, replacing Q-learning with direct policy
approximation. However, Phase 1 has a unique challenge that Phase 2 does not:
**goal-conditioned policies**.

### 12.2 The Goal-Conditioning Challenge

In Phase 1, the policy is conditioned on a *possible goal* g_h:

```
π_h(a | s, g_h)    — the policy for human h to achieve goal g_h
```

Standard PPO learns an unconditional policy π(a|s). To use PPO for Phase 1, we need
to handle goal conditioning. There are several approaches:

#### Approach A: Goal as Part of the Observation

Treat the goal as an additional input to the policy network:

```python
class GoalConditionedActorCritic(nn.Module):
    """PPO actor-critic with goal conditioning for Phase 1."""
    
    def __init__(self, state_encoder, goal_encoder, hidden_dim, num_actions):
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        combined_dim = state_encoder.output_dim + goal_encoder.output_dim
        
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs, goal):
        state_features = self.state_encoder(obs)
        goal_features = self.goal_encoder(goal)
        combined = torch.cat([state_features, goal_features], dim=-1)
        return self.actor(combined), self.critic(combined)
```

The environment wrapper samples a goal at the start of each episode and includes it in
the observation. The reward is `goal.is_achieved(state)` (0 or 1). This is essentially
**Hindsight Experience Replay (HER)**-style goal-conditioned RL.

**Pros:**
- Straightforward PPO application with standard Gymnasium interface
- PufferLib handles vectorization transparently
- The reward is a simple binary signal from the environment (no intrinsic reward issues)

**Cons:**
- PPO must learn a separate behavior for each goal from a single network
- Sparse reward (0/1) requires reward shaping or HER-like relabeling
- Need to sample goals proportionally during training for proper coverage

#### Approach B: Separate PPO per Goal Class

Train a separate PPO policy for each major goal class. This is conceptually simpler but
doesn't scale well:

**Pros:** Each policy sees a focused reward signal  
**Cons:** Number of policies grows with goal space; no weight sharing across goals

#### Approach C: Marginal Policy via PPO + Goal Sampling

Train a single PPO policy that approximates the *marginal* policy
π_h(a|s) = E_g[π_h(a|s,g)] by sampling goals stochastically each episode:

```python
class Phase1EMPOEnv(gymnasium.Env):
    """PufferLib-compatible wrapper for Phase 1 training."""
    
    def reset(self, seed=None, options=None):
        self.world_model.reset(seed=seed)
        state = self.world_model.get_state()
        # Sample a new goal for this episode
        self.current_goal, _ = self.goal_sampler.sample(state, self.human_idx)
        obs = self._encode(state, self.current_goal)
        return obs, {}
    
    def step(self, action):
        # ... step world model ...
        reward = self.current_goal.is_achieved(next_state)
        obs = self._encode(next_state, self.current_goal)
        return obs, reward, terminated, truncated, info
```

**Pros:** Single PPO policy; goal coverage handled by sampling distribution  
**Cons:** Policy must generalize across goals; may need many episodes for rare goals

### 12.3 Recommended Approach for Phase 1

**Approach A (goal as observation)** is recommended because:

1. It directly mirrors the current Q-network architecture Q(s, a, g), which already
   takes both state and goal as inputs
2. The existing `MultiGridGoalEncoder` and `MultiGridStateEncoder` can be reused
3. The reward signal is well-defined (binary goal achievement + shaping rewards)
4. Unlike Phase 2, the Phase 1 reward is **not intrinsic** — it comes from the
   environment (goal achievement). This eliminates the core challenge of Phase 2

### 12.4 Phase 1 PPO vs. Current DQN

| Aspect | Current (DQN) | PPO Alternative |
|--------|---------------|-----------------|
| Policy | Implicit via Boltzmann softmax over Q | Explicit policy network |
| Value | Q(s, a, g) — action-value | V(s, g) — state-value |
| Data | Off-policy replay buffer | On-policy rollout buffer |
| Goal conditioning | Q-network input | Actor-critic input |
| Reward | Binary (goal achieved) + shaping | Same (no change needed) |
| β_h parameter | Directly controls Boltzmann temperature | Replaced by entropy coef |

### 12.5 Phase 1 PPO Considerations

- **Marginal policy (φ-network):** The current Phase 1 optionally trains a "direct phi
  network" that predicts π_h(a|s) = E_g[π_h(a|s,g)] without enumerating goals. With PPO,
  the phi network can be trained as a **distillation target** from the goal-conditioned
  PPO policy, similar to the current joint training approach
- **β_h interpretation:** The Boltzmann parameter β_h controls how sharply the policy
  concentrates on the best actions. In PPO, this is implicitly controlled by the entropy
  coefficient and training duration. To preserve the β_h semantics, we could use the
  power-law policy head (Section 9.1) in Phase 1 as well
- **Priority:** Porting Phase 1 to PPO is **lower priority** than Phase 2, because:
  - Phase 1's DQN approach is simpler (no mutual dependencies, no intrinsic reward)
  - The existing Phase 1 trainer works well in practice
  - Phase 2 benefits more from PPO's on-policy stability for non-stationary rewards

### 12.6 Phase 1 PPO Implementation Status

Phase 1 PPO has been **fully implemented** using Approach A (goal as observation):

| Component | File | Status |
|-----------|------|--------|
| Config | `learning_based/phase1_ppo/config.py` | ✅ `PPOPhase1Config` dataclass |
| Actor-Critic | `learning_based/phase1_ppo/actor_critic.py` | ✅ `GoalConditionedActorCritic` |
| Env Wrapper | `learning_based/phase1_ppo/env_wrapper.py` | ✅ `Phase1PPOEnv` (base class) |
| Trainer | `learning_based/phase1_ppo/trainer.py` | ✅ `PPOPhase1Trainer` (PufferLib PuffeRL) |
| MultiGrid Env | `learning_based/multigrid/phase1_ppo/env_wrapper.py` | ✅ `MultiGridPhase1PPOEnv` |
| MultiGrid Networks | `learning_based/multigrid/phase1_ppo/networks.py` | ✅ `create_multigrid_phase1_ppo_networks()` |
| Base Tests | `tests/test_phase1_ppo.py` | ✅ 27 tests |
| MultiGrid Tests | `tests/test_multigrid_phase1_ppo.py` | ✅ 22 tests |
| Example Script | `examples/phase1/phase1_ppo_demo.py` | ✅ Full demo with rollout movies |

---

## 13. Testing with PufferLib Ocean Environments

### 13.1 Selection Criteria

When choosing environments for validating the ported PPO Phase 2 training, we
prioritize:

1. **Simple and easy to visualize** — grid-based, 2D, interpretable agent behavior
2. **Goal-agnostic** — the environment structure itself is not tied to specific goals;
   instead, goals can be overlaid externally (e.g., "reach cell X", "be near agent Y")
3. **Easy to write heuristic human policies** — so we can test Phase 2 training in
   isolation without first training a Phase 1 neural policy prior. This is critical
   for fast iteration: if every Phase 2 test requires a trained Phase 1, the development
   cycle becomes unacceptably slow

The third criterion is especially important. EMPO Phase 2 requires a human policy prior
π_h(a|s, g), which currently comes from Phase 1 training (Q-learning). For testing the
PPO port, we want to substitute a **heuristic human policy** that requires no training.
The existing codebase already provides `HeuristicPotentialPolicy` for multigrid
environments, which uses shortest-path gradients to guide humans toward goals. A suitable
test environment should support similar heuristic policies.

### 13.2 Most Suitable Ocean Environments

#### 13.2.1 Foraging (Best Match)

**Why best match:** Foraging is a multi-agent grid world where agents navigate to collect
food items. It is:

- **Simple & visual**: 2D grid, rendered via `render_mode='rgb_array'`
- **Goal-agnostic**: "Food" positions serve as natural goal candidates, and we can
  define `PossibleGoal` instances as "reach the cell containing food item X"
- **Heuristic human policies are trivial**: "walk toward the nearest food" is a
  one-line BFS/distance heuristic — no Phase 1 training needed:

```python
class ForagingHeuristicPolicy(HumanPolicyPrior):
    """Heuristic: walk toward the goal position via shortest path."""
    
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon  # Exploration smoothing
    
    def __call__(self, state, human_agent_index, possible_goal):
        agent_pos = state.agent_positions[human_agent_index]
        goal_pos = possible_goal.target_position
        # 8-directional movement: pick direction that minimizes L-inf distance
        dx = np.sign(goal_pos[0] - agent_pos[0])
        dy = np.sign(goal_pos[1] - agent_pos[1])
        action = direction_to_action(dx, dy)
        # Return Boltzmann-softened distribution (peaked at best action)
        probs = np.ones(self.num_actions) * self.epsilon
        probs[action] = 1.0
        probs /= probs.sum()
        return probs
```

**How to test Phase 2:**
1. Wrap Foraging as a `WorldModel` (implement `get_state`, `set_state`,
   `transition_probabilities`)
2. Define food positions as `PossibleGoal` instances
3. Use the heuristic human policy above as the policy prior
4. Run the full PPO Phase 2 training loop with intrinsic reward U_r
5. Verify: the robot learns to help humans reach food (U_r improves over training)

```python
env = pufferlib.environments.ocean.environment.make_foraging(
    width=64, height=64, num_agents=4, discretize=True,
    food_reward=0.1, render_mode='rgb_array'
)
```

#### 13.2.2 Predator-Prey (Multi-Role)

**Why:** Predator-Prey assigns different roles to agents (predators vs. prey), which maps
to EMPO's human-robot distinction. The robot could be a "helper" that influences predator
or prey outcomes.

- **Simple & visual**: Grid navigation with role differentiation visible in rendering
- **Goal-agnostic**: Goals can be "prey reaches cell X" or "predator catches prey Y"
- **Heuristic policies**: Prey runs away from nearest predator (flee heuristic), predator
  chases nearest prey (chase heuristic) — both are trivial distance-based policies

**How to test Phase 2:**
1. Designate some agents as humans (prey), one as robot
2. Define goals as "prey h reaches safe zone S" (a `PossibleGoal` with positional check)
3. Heuristic human policy: flee from predators toward safe zone
4. Robot learns (via PPO) to position itself to maximize prey survival (human power)

#### 13.2.3 Squared (Single-Agent Baseline)

**Why:** Squared is a single-agent grid navigation task. While it lacks multi-agent
structure, it is useful as a **minimal sanity check** for the PPO integration before
adding multi-agent complexity.

- **Simple & visual**: Agent navigates to target on grid
- **Goal-agnostic**: Target position is the goal
- **No human policy needed**: Single-agent, so we can test PPO mechanics in isolation

**How to test:**
1. Verify PufferLib PPO loop end-to-end (rollout, GAE, policy/value updates)
2. Test vectorized execution with multiple parallel instances
3. Confirm reward normalization works with always-negative rewards (wrapping reward to
   simulate U_r < 0)

### 13.3 Wrapping Ocean Environments as WorldModels

To use Ocean environments with EMPO's Phase 2 trainer, they must implement the
`WorldModel` interface. The key challenge is `transition_probabilities()`, since Ocean
environments are simulation-based (no analytical transition model). Options:

**Option A: Empirical transition model (for deterministic envs)**

If the environment is deterministic given the joint action, `transition_probabilities()`
returns a single successor with probability 1.0:

```python
class ForagingWorldModel(WorldModel):
    """WorldModel wrapper for PufferLib Foraging."""
    
    def transition_probabilities(self, state, actions):
        saved = self.get_state()
        self.set_state(state)
        # Step the env to find the successor
        self.env.step(actions)
        next_state = self.get_state()
        self.set_state(saved)  # Restore original state
        return [(1.0, next_state)]
    
    def get_state(self):
        # Serialize env state as hashable tuple
        return self.env.get_internal_state()
    
    def set_state(self, state):
        self.env.set_internal_state(state)
```

**Option B: Skip model-based targets**

Set `use_model_based_targets=False` in Phase2Config. V_h^e targets will use single
observed successors instead of expected values. This is simpler but higher-variance.
Acceptable for testing purposes.

### 13.4 Testing Progression

Recommended order for validating the PufferLib PPO integration, designed so that each
step adds exactly one new component:

```
Step 1: Squared (single-agent)
        → PPO compiles and learns spatial navigation          (5 min)
        
Step 2: Squared + negative reward wrapper
        → PPO handles always-negative rewards (simulates U_r) (10 min)
        
Step 3: Foraging (multi-agent, heuristic humans)
        → Multi-agent env works, heuristic human policy OK     (30 min)
        
Step 4: Foraging + EMPO reward wrapper
        → Intrinsic reward U_r replaces env reward             (1 hour)
        → Test freeze/sync cycle for auxiliary networks
        
Step 5: Foraging + full EMPO auxiliary training
        → V_h^e, X_h, U_r trained alongside PPO robot          (hours)
        → Verify warm-up stages, auxiliary network convergence
        
Step 6: EMPO MiniGrid 5×5 (1 human, 1 robot)
        → Full pipeline with existing HeuristicPotentialPolicy  (hours)
        → Compare with current DQN-trained policy behavior
        
Step 7: EMPO MiniGrid 7×7 (3 humans, 1 robot)
        → Scaling test with multiple humans                     (hours-days)
```

### 13.5 Mock EMPO Reward for Ocean Environments

To test the intrinsic reward integration without the full EMPO auxiliary network stack,
create a mock U_r that simulates the key properties of the real signal:

```python
class MockEMPORewardWrapper(gymnasium.Wrapper):
    """
    Wraps any PufferLib Ocean environment with a mock intrinsic reward.
    
    Simulates the EMPO U_r signal properties:
    - Always negative (U_r < 0)
    - Non-stationary (drifts over training, simulating auxiliary net updates)
    - State-dependent (not action-dependent)
    
    Args:
        env: Base environment.
        drift_rate: How fast the reward drifts (simulates aux net updates).
        reward_scale: Scale factor for original reward's influence on mock U_r.
        drift_scale: Scale factor for temporal drift component.
    """
    
    def __init__(self, env, drift_rate=0.001, reward_scale=0.1, drift_scale=0.01):
        super().__init__(env)
        self._step = 0
        self._drift_rate = drift_rate
        self._reward_scale = reward_scale
        self._drift_scale = drift_scale
        self._base_reward = -1.0
    
    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        # Simulate non-stationary negative reward
        self._step += 1
        drift = self._drift_rate * self._step
        mock_u_r = (self._base_reward
                    - self._reward_scale * abs(env_reward)
                    + self._drift_scale * drift)
        mock_u_r = min(mock_u_r, -0.01)  # Ensure U_r < 0
        info['env_reward'] = env_reward
        info['u_r'] = mock_u_r
        return obs, mock_u_r, terminated, truncated, info
```

This allows testing:
- PPO convergence with always-negative rewards
- Reward normalization effectiveness
- Impact of reward drift on training stability
- Freeze/sync cycle mechanics (swap mock for updated mock periodically)

---

## Appendix A: PPOPhase2Config (Separate Configuration)

The PPO trainer uses its **own config class**, defined in `learning_based/phase2_ppo/config.py`.
It does NOT extend or modify the existing `Phase2Config`. Shared theory parameters
(γ_r, γ_h, ζ, ξ, η) are duplicated — this is intentional, to avoid coupling.

```python
# File: learning_based/phase2_ppo/config.py

@dataclass
class PPOPhase2Config:
    """
    Configuration for PPO-based Phase 2 training.
    
    This is a standalone config class — it does NOT inherit from Phase2Config.
    Shared theory parameters are duplicated here to avoid coupling to the DQN
    config. The existing Phase2Config is not modified.
    """
    
    # ----- Theory parameters (shared with DQN path, duplicated here) -----
    gamma_r: float = 0.99               # Robot discount factor
    gamma_h: float = 0.99               # Human discount factor (for V_h^e)
    zeta: float = 1.0                   # Risk/reliability preference (ζ ≥ 1)
    xi: float = 1.0                     # Inter-human inequality aversion (ξ ≥ 1)
    eta: float = 1.0                    # Intertemporal inequality aversion (η ≥ 1)
    steps_per_episode: int = 50         # Environment steps per episode
    
    # ----- PPO hyperparameters -----
    ppo_rollout_length: int = 128       # Steps per PPO rollout
    ppo_num_minibatches: int = 4        # Minibatches per PPO update
    ppo_update_epochs: int = 4          # Epochs per PPO update
    ppo_clip_coef: float = 0.2          # PPO clipping coefficient
    ppo_ent_coef: float = 0.01          # Entropy bonus coefficient
    ppo_vf_coef: float = 0.5            # Value function loss coefficient
    ppo_max_grad_norm: float = 0.5      # Max gradient norm for clipping
    ppo_gae_lambda: float = 0.95        # GAE lambda
    lr_ppo: float = 3e-4                # PPO learning rate
    num_envs: int = 16                  # Number of vectorized environments
    num_ppo_iterations: int = 10000     # Total PPO iterations
    
    # ----- Auxiliary network training (V_h^e, X_h, U_r) -----
    lr_v_h_e: float = 1e-4              # Learning rate for V_h^e
    lr_x_h: float = 1e-4                # Learning rate for X_h
    lr_u_r: float = 1e-4                # Learning rate for U_r network
    aux_training_steps_per_iteration: int = 10
    aux_buffer_size: int = 50000
    reward_freeze_interval: int = 1     # Freeze aux nets every N PPO iterations
    batch_size: int = 256
    
    # ----- Warm-up (auxiliary nets only, random robot) -----
    warmup_v_h_e_steps: int = 5000      # Training steps for V_h^e-only warm-up
    warmup_x_h_steps: int = 7500        # Training steps for V_h^e + X_h
    warmup_u_r_steps: int = 10000       # Training steps for V_h^e + X_h + U_r
    
    # ----- Entropy schedule (optional, mimics β_r ramp-up) -----
    ppo_ent_coef_start: float = 0.1     # Initial entropy coefficient (high = exploratory)
    ppo_ent_coef_end: float = 0.01      # Final entropy coefficient (cosine annealing)
    ppo_ent_anneal_steps: int = 10000   # Training steps to anneal entropy coefficient
    
    # ----- Network architecture -----
    hidden_dim: int = 256
    use_shared_encoder: bool = True     # Share state encoder between actor-critic and V_h^e
```

## Appendix B: Comparison of PPO Advantage Computation with EMPO Intrinsic Reward

In standard PPO with extrinsic environment reward r_t:

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)                          (TD error)
Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...             (GAE)
```

In EMPO with intrinsic reward U_r(s_t):

```
δ_t = U_r(s_t) + γ_r V_r(s_{t+1}) − V_r(s_t)              (TD error with U_r)
Â_t = δ_t + (γ_r λ)δ_{t+1} + (γ_r λ)²δ_{t+2} + ...       (GAE with U_r)
```

The only difference is the source of the reward signal. The GAE computation is
identical—PufferLib/CleanRL's PPO implementation handles this transparently since
it treats the reward as an opaque scalar.

**Important:** U_r < 0 always (since X_h ∈ (0, 1] and the formula guarantees negativity).
This means V_r < 0 as well. PPO handles negative rewards without issues, but reward
normalization should be used to keep the scale manageable.

## Appendix C: References

- **PufferLib**: https://github.com/PufferAI/PufferLib — High-performance RL library
- **PufferLib Paper**: Lam et al., "PufferLib: Making Reinforcement Learning Libraries
  and Environments Play Nice" (2024), https://arxiv.org/abs/2406.12905
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized
  Advantage Estimation" (2015)
- **EMPO Paper**: Equations (4)–(9) in Table 1 for Phase 2 formulation
- **Intrinsic Motivation Survey**: Aubret et al., "A Survey on Intrinsic Motivation in
  Reinforcement Learning" (2019) — discusses non-stationary intrinsic rewards
- **Current PPO/A3C considerations**: `docs/plans/ppo_a3c_considerations.md`
- **Current warm-up design**: `docs/WARMUP_DESIGN.md`

# Plan: Porting Phase 2 Training to PufferLib PPO

**Status:** Proposed  
**Date:** 2025-03-20

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture (DQN-style)](#2-current-architecture-dqn-style)
3. [Proposed Architecture (PufferLib PPO)](#3-proposed-architecture-pufferlib-ppo)
4. [The Core Challenge: Intrinsic Reward Integration](#4-the-core-challenge-intrinsic-reward-integration)
5. [Environment Wrapper Design](#5-environment-wrapper-design)
6. [Network Architecture Changes](#6-network-architecture-changes)
7. [Warm-up and Mutual Dependency Handling](#7-warm-up-and-mutual-dependency-handling)
8. [Training Loop Redesign](#8-training-loop-redesign)
9. [Theoretical Considerations](#9-theoretical-considerations)
10. [Migration Plan](#10-migration-plan)
11. [Open Questions and Risks](#11-open-questions-and-risks)

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

### Key Changes at a Glance

| Aspect | Current (DQN-style) | Proposed (PufferLib PPO) |
|--------|---------------------|--------------------------|
| **Robot policy** | Implicit: π_r ∝ (-Q_r)^{-β_r} | Explicit: π_r network (actor) |
| **Robot value** | Optional V_r network, or computed from U_r + E[Q_r] | V_r network (critic) |
| **Q_r** | Learned via model-based Bellman targets | **Eliminated** — PPO doesn't need Q-values |
| **Reward signal** | Implicit in Bellman equation (γ_r · V_r(s')) | Explicit U_r(s) passed as reward to PPO |
| **Data collection** | Off-policy replay buffer | On-policy rollout buffer (PufferLib vectorized) |
| **Exploration** | ε-greedy + power-law softmax + RND | PPO entropy bonus + (optionally) RND |
| **Parallelism** | Custom async actor-learner | PufferLib vectorized environments |
| **V_h^e, X_h, U_r** | Trained from same replay buffer | Trained from a **separate** replay buffer filled from the same rollouts |

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

### 3.2 What Gets Eliminated

- **Q_r network and Q_r target network** — PPO doesn't need Q-values
- **Power-law softmax policy derivation** (eq. 5) — replaced by explicit policy network
- **Off-policy replay buffer** for robot policy — PPO uses on-policy rollout buffers
- **Custom ε-greedy exploration** — PPO uses entropy bonus in the policy loss
- **z-space transformation** for Q_r — no longer needed

### 3.3 What Gets Kept

- **V_h^e network and its training** (eq. 6) — still needed for power metric
- **X_h computation** (eq. 7) — still needed for U_r
- **U_r computation** (eq. 8) — still needed, now serves as the *environment reward* for PPO
- **Phase 1 human policy prior** — unchanged
- **Goal sampling** — unchanged
- **Staged warm-up** — adapted (see Section 7)
- **Model-based targets for V_h^e** — still valuable for variance reduction

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
    with the EMPO intrinsic reward U_r(s).
    
    On each step:
    1. Calls the base env.step(action) to get (obs, env_reward, done, truncated, info)
    2. Computes U_r(s') using the auxiliary networks
    3. Returns (obs, U_r(s'), done, truncated, info)
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
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Get current state for U_r computation
        state = self.env.get_state()
        
        # Compute intrinsic reward using (frozen) auxiliary networks
        u_r = self._compute_u_r(state)
        
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
            for h in self.human_agent_indices:
                x_h = nets.x_h.forward(state, self.env, h, self.device)
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

class EMPOMultiGridEnv(gymnasium.Env):
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
        
        # Step the world model
        self.world_model.step(joint_action)
        next_state = self.world_model.get_state()
        
        # Compute intrinsic reward U_r(next_state)
        u_r = self._compute_u_r(next_state)
        
        # Episode termination
        self._step_count += 1
        truncated = self._step_count >= self.config.steps_per_episode
        terminated = False  # EMPO episodes are truncated, not terminated
        
        # Resample goals with probability goal_resample_prob
        if random.random() < self.config.goal_resample_prob:
            self._sample_goals(next_state)
        
        obs = self._encode_observation(next_state)
        info = {
            'u_r': u_r,
            'state': next_state,  # For auxiliary network training
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
# In the EMPOMultiGridEnv.step():
info = {
    'state': state,              # WorldModel state (for V_h^e forward pass)
    'next_state': next_state,    # For TD targets
    'goals': goals,              # {human_idx: goal} (for V_h^e)
    'goal_weights': goal_weights,
    'human_actions': human_actions,
    'transition_probs': trans_probs,  # For model-based V_h^e targets
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
    return EMPOMultiGridEnv(
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
    
    def get_action_and_value(self, obs, action=None):
        """PPO-compatible: returns action, log_prob, entropy, value."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), None
```

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

### 6.5 Eliminated Networks

- **Q_r (BaseRobotQNetwork)** — no longer needed
- **Q_r target** — no longer needed
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

The warm-up stages are adapted for PPO:

| Stage | Duration | Active Components | π_r | Notes |
|-------|----------|-------------------|-----|-------|
| 0 | W₀ steps | V_h^e only | Uniform random | Same as current Stage 0 |
| 1 | W₁ steps | V_h^e + X_h | Uniform random | Same as current Stage 1 |
| 2 | W₂ steps | V_h^e + X_h + U_r | Uniform random | Same as current Stage 2 |
| 3 | R steps | V_h^e + X_h + U_r + PPO | PPO training begins | **NEW**: PPO replaces Q_r+V_r warmup |
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

**Option A: Separate warm-up phase without PPO**

Run warm-up as a separate pre-training phase that doesn't use PufferLib at all:

```python
# Phase A: Warm up auxiliary networks with random robot
for step in range(total_warmup_steps):
    state = env.get_state()
    robot_action = env.action_space.sample()  # Random
    human_actions = sample_human_actions(state, goals)
    env.step(build_joint_action(robot_action, human_actions))
    # Train V_h^e, X_h, U_r from this transition
    train_auxiliary_networks(transition)

# Phase B: PufferLib PPO training with intrinsic reward
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

### 8.1 Main Training Loop

```python
def train_empo_ppo(
    world_model_factory,
    human_policy_prior,
    goal_sampler,
    human_agent_indices,
    robot_agent_indices,
    config,  # Phase2Config + PPO config
    num_envs=16,
    device='cuda',
):
    # ================================================================
    # Phase A: Warm-up (auxiliary networks only, random robot)
    # ================================================================
    auxiliary_networks = create_auxiliary_networks(config, device)
    warmup_trainer = AuxiliaryWarmupTrainer(
        world_model_factory, auxiliary_networks, config,
        human_policy_prior, goal_sampler,
        human_agent_indices, robot_agent_indices,
    )
    warmup_trainer.train(num_steps=config.get_total_warmup_steps())
    
    # ================================================================
    # Phase B: PPO training with intrinsic reward
    # ================================================================
    
    # Create actor-critic
    state_encoder = auxiliary_networks.v_h_e.state_encoder  # Shared encoder
    actor_critic = EMPOActorCritic(
        state_encoder, config.hidden_dim,
        config.num_actions, config.num_robots,
    ).to(device)
    
    # Freeze auxiliary networks for reward computation
    frozen_aux = freeze_networks(auxiliary_networks)
    
    # Create vectorized PufferLib environments
    def make_env():
        env = world_model_factory()
        return EMPOMultiGridEnv(
            env, human_policy_prior, goal_sampler,
            human_agent_indices, robot_agent_indices,
            config, auxiliary_networks=frozen_aux,
        )
    
    vec_env = pufferlib.vector.make(
        make_env, num_envs=num_envs,
        backend=pufferlib.vector.Multiprocessing,
    )
    
    # PPO training loop
    ppo_trainer = pufferlib.cleanrl.PPO(
        env=vec_env,
        policy=actor_critic,
        learning_rate=config.lr_ppo,
        gamma=config.gamma_r,
        gae_lambda=config.gae_lambda,
        clip_coef=config.ppo_clip_coef,
        ent_coef=config.ppo_ent_coef,
        vf_coef=config.ppo_vf_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        num_steps=config.ppo_rollout_length,
        num_minibatches=config.ppo_num_minibatches,
        update_epochs=config.ppo_update_epochs,
    )
    
    # Auxiliary network replay buffer (for V_h^e, X_h, U_r training)
    aux_replay_buffer = Phase2ReplayBuffer(capacity=config.buffer_size)
    
    for iteration in range(config.num_ppo_iterations):
        # --- Step 1: PPO rollout ---
        rollout_data = ppo_trainer.collect_rollout()
        
        # Extract auxiliary training data from rollout infos
        aux_transitions = extract_auxiliary_transitions(rollout_data)
        for t in aux_transitions:
            aux_replay_buffer.push(t)
        
        # --- Step 2: PPO update (policy + value) ---
        ppo_trainer.update(rollout_data)
        
        # --- Step 3: Auxiliary network training ---
        # Train V_h^e, X_h, U_r using transitions from the rollout
        # This uses the CURRENT (non-frozen) policy for V_h^e targets
        for aux_step in range(config.aux_training_steps_per_iteration):
            batch = aux_replay_buffer.sample(config.batch_size)
            aux_losses = compute_auxiliary_losses(
                batch, auxiliary_networks, actor_critic, config
            )
            update_auxiliary_networks(aux_losses, aux_optimizers)
        
        # --- Step 4: Freeze updated auxiliary networks for next rollout ---
        frozen_aux = freeze_networks(auxiliary_networks)
        sync_frozen_to_envs(vec_env, frozen_aux)
    
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
class EMPOMultiGridEnv:
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
- The policy still converges to the optimal policy in the limit
- The practical impact may be small if training is stable

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
  to the optimal policy for that fixed U_r

**Recommendation:** Use a lower learning rate for auxiliary networks relative to PPO,
creating a natural time-scale separation.

---

## 10. Migration Plan

### Phase 1: Foundation (Estimated: 2–3 weeks)

- [ ] **10.1** Add PufferLib dependency (`pip install pufferlib`)
- [ ] **10.2** Implement `EMPOMultiGridEnv` (Gymnasium wrapper with human agent simulation)
- [ ] **10.3** Implement `EMPOActorCritic` (combined actor-critic network)
- [ ] **10.4** Unit tests: env wrapper produces valid observations, rewards, dones
- [ ] **10.5** Verify: random policy through wrapper matches current random rollout behavior

### Phase 2: Intrinsic Reward Integration (Estimated: 2–3 weeks)

- [ ] **10.6** Implement centralized U_r reward computation (Option C from Section 5.4)
- [ ] **10.7** Implement auxiliary network training loop (V_h^e, X_h, U_r) using PPO rollout data
- [ ] **10.8** Implement freeze/sync cycle for auxiliary networks
- [ ] **10.9** Unit tests: U_r computation matches current `get_u_r()` method
- [ ] **10.10** Verify: auxiliary networks converge during warm-up with random robot

### Phase 3: PPO Training (Estimated: 2–3 weeks)

- [ ] **10.11** Integrate PufferLib PPO trainer with EMPOMultiGridEnv
- [ ] **10.12** Implement warm-up → PPO transition (Section 7.2)
- [ ] **10.13** Implement PPO hyperparameter configuration in Phase2Config
- [ ] **10.14** End-to-end test: train on small grid (5×5, 1 human, 1 robot)
- [ ] **10.15** Compare learned behavior with current DQN-trained policy

### Phase 4: Optimization and Scaling (Estimated: 2–3 weeks)

- [ ] **10.16** Implement batched U_r computation across vectorized environments
- [ ] **10.17** Profile training loop, optimize bottlenecks
- [ ] **10.18** Test on larger environments (7×7, 3 humans, 1 robot)
- [ ] **10.19** TensorBoard integration: log PPO metrics alongside auxiliary metrics
- [ ] **10.20** Checkpoint/resume support for PPO + auxiliary networks

### Phase 5: Cleanup and Documentation (Estimated: 1 week)

- [ ] **10.21** Update `docs/WARMUP_DESIGN.md` with PPO-adapted warm-up
- [ ] **10.22** Update `docs/API.md` with new interfaces
- [ ] **10.23** Create example script: `examples/phase2/pufferlib_ppo_demo.py`
- [ ] **10.24** Update `docs/plans/ppo_a3c_considerations.md` with implementation notes
- [ ] **10.25** Keep current DQN-style trainer as fallback (feature flag in config)

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

1. PPO-trained robot achieves similar or better human empowerment scores as DQN-trained
   robot on the standard 5×5 and 7×7 test environments
2. Training wall-clock time is comparable or faster (due to PufferLib parallelism)
3. Training is stable (no divergence, reasonable loss curves)
4. Code is maintainable: the PPO trainer is cleaner and simpler than the current
   DQN trainer (fewer networks to manage, no Q_r/V_r target networks for the robot)

---

## Appendix A: Phase2Config Extensions

New configuration parameters needed for PPO integration:

```python
@dataclass
class Phase2Config:
    # ... existing parameters ...
    
    # PPO configuration
    use_ppo: bool = False               # Enable PPO mode (vs. DQN mode)
    ppo_rollout_length: int = 128       # Steps per PPO rollout
    ppo_num_minibatches: int = 4        # Minibatches per PPO update
    ppo_update_epochs: int = 4          # Epochs per PPO update
    ppo_clip_coef: float = 0.2          # PPO clipping coefficient
    ppo_ent_coef: float = 0.01          # Entropy bonus coefficient
    ppo_vf_coef: float = 0.5           # Value function loss coefficient
    ppo_max_grad_norm: float = 0.5      # Max gradient norm for clipping
    ppo_gae_lambda: float = 0.95        # GAE lambda
    lr_ppo: float = 3e-4               # PPO learning rate
    num_envs: int = 16                  # Number of vectorized environments
    
    # Auxiliary network training under PPO
    aux_training_steps_per_iteration: int = 10   # Aux training steps per PPO iteration
    aux_buffer_size: int = 50000                 # Separate replay buffer for auxiliary nets
    reward_freeze_interval: int = 1              # Freeze aux nets every N PPO iterations
    
    # Entropy schedule (optional, mimics β_r ramp-up)
    ppo_ent_coef_start: float = 0.1     # Initial entropy coefficient (high = exploratory)
    ppo_ent_coef_end: float = 0.01      # Final entropy coefficient
    ppo_ent_anneal_steps: int = 10000   # Steps to anneal entropy coefficient
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

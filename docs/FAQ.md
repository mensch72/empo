# Phase 2 Training Design Choices

This document provides brief justifications for design choices in the Phase 2 training code that might appear nonstandard.

## Network Architecture

### Power-law softmax instead of Boltzmann softmax for robot policy (Eq. 5)
The power-law form `π_r(a_r) ∝ (-Q_r(s, a_r))^{-β_r}` satisfies scale-invariance properties that make the robot's decisions robust to arbitrary rescaling of the reward function.

### Predicting log(y-1) instead of y directly in U_r network
Since y = E[X_h^{-ξ}] can have heavy right tails when humans have low power, using log-space representation provides numerical stability while avoiding overflow.

### Using -softplus(x) to ensure Q_r < 0
Since V_r < 0 by construction (negative rewards), Q_r must also be negative, so we use -softplus(x) which maps all reals to (-∞, 0) smoothly.

### No immediate reward in Q_r target (Eq. 4)
The immediate reward U_r(s) accrues when arriving at state s, so it's included in V_r(s) rather than in Q_r(s, a_r) which only captures the value of taking action a_r.

## Training Procedure

### Multi-stage warm-up with sequential network activation
Breaking circular dependencies (V_h^e → X_h → U_r → Q_r → V_r → V_h^e) requires training networks sequentially so later networks can use stable earlier ones as targets.

### Setting beta_r = 0 during warm-up (uniform random robot policy)
During warm-up, V_h^e must learn independently of robot actions to provide stable targets for later networks, which requires a robot policy that doesn't depend on Q_r.

### Clearing replay buffer at beta_r ramp-up transitions
Data collected with beta_r=0 (uniform policy) becomes off-policy when beta_r increases, so we clear the buffer to ensure on-policy training after the transition.

### Sigmoid ramp-up of beta_r instead of linear
The sigmoid curve provides smooth acceleration and deceleration at the start and end of ramp-up, reducing training instability from abrupt policy changes.

### Using 1/sqrt(t) learning rate decay instead of 1/t
The 1/sqrt(t) schedule balances convergence guarantees for expectation learning with the non-stationary targets in Q-learning, providing a practical compromise.

### Separate (potentially larger) batch size for X_h network
Since X_h targets are computed by sampling goals (Monte Carlo approximation of E_g[V_h^e^ζ]), larger batches reduce variance in the expectation estimate.

### Optional human sampling for U_r network training (when u_r_use_network=True)
When using a U_r network with many humans, computing X_h^{-ξ} for all humans per training step is expensive, so we optionally sample a subset as an unbiased estimator of E_h[X_h^{-ξ}].

### Using MSE loss for y (not Huber loss) despite heavy tails
MSE converges to the arithmetic mean which is exactly E[X_h^{-ξ}] as required by Eq. 8, while Huber loss would introduce systematic bias toward the median.

## Network Computation Modes

### Computing U_r directly from X_h instead of using a network (default, few humans)
When there are few humans, computing U_r = -(E_h[X_h^{-ξ}])^η directly from X_h values for all humans is fast, eliminates approximation error, and avoids training an extra network.

### Computing V_r directly from U_r and Q_r instead of using a network (default)
Since V_r = U_r + π_r · Q_r is a deterministic weighted average, computing it directly from its components is exact and avoids training an unnecessary network.

## Target Networks and Stability

### What is the difference between "main" and "target" networks?

**Main networks** (e.g., `v_h_e`, `x_h`, `q_r`) are the networks being actively trained:
- Updated by gradient descent on every training step
- Used for computing predictions that are compared against targets (the left side of the loss equation)
- Used during policy execution and rollout collection

**Target networks** (e.g., `v_h_e_target`, `x_h_target`) are frozen copies of the main networks:
- NOT updated by gradient descent (requires_grad=False)
- Updated periodically by copying weights from the corresponding main network (every N steps)
- Used ONLY for computing training targets (the right side of the loss equation)
- Always in eval mode (disables dropout) for consistent target computation

**Example from V_h^e training:**
- Main network `v_h_e` predicts V_h^e(s, g_h) — this prediction is trained
- Target network `v_h_e_target` provides V_h^e(s', g_h) for the TD target — this is frozen
- Loss: (v_h_e(s, g_h) - [goal_achieved + γ * v_h_e_target(s', g_h)])²

This separation prevents the "moving target problem" where updating the network changes the targets it's trying to match, causing training instability. Similarly, in async training mode, actor processes use a frozen copy of the policy (periodically synced from the learner) to generate training data, ensuring data collection doesn't change mid-episode as the learner updates the policy.

### Using target networks for V_r, V_h^e, X_h, and U_r
Frozen target networks prevent the moving target problem where the TD target changes as the network being trained updates, improving training stability.

### Periodic full-copy target network updates
Target networks are updated by copying the full state dict every N steps, following the DQN approach rather than continuous blending.

### Using V_h^e target network (not main) in X_h target computation
Using the target network for V_h^e when computing X_h targets creates more stable gradient flow since X_h depends on V_h^e which is also being updated.

### Using X_h target network (not main) in U_r target computation
Using the target network for X_h when computing U_r targets prevents feedback loops where U_r changes affect X_h which affects U_r in the same update.

## Regularization

### High dropout rates (0.5 default) in hidden layers
Phase 2 training has high variance due to stochastic human actions and goal sampling, so aggressive dropout prevents overfitting to specific trajectories.

### Auto-scaled gradient clipping proportional to learning rate
Scaling clip values by lr/reference_lr keeps step sizes bounded regardless of learning rate magnitude, preventing both gradient explosion and over-restrictive clipping.

### Weight decay on all networks
L2 regularization prevents parameter norm growth during long training runs with decaying learning rates, improving generalization.

## Advanced Features

### Model-based targets using transition_probabilities()
Computing expected V(s') over all possible successors (like Expected SARSA) reduces variance and ensures actions with identical effects get identical Q-values.

### Caching transition probabilities at collection time
Pre-computing transition probabilities for all robot actions when collecting data avoids redundant expensive calls during training batch processing.

### Async actor-learner architecture option
Separating CPU-bound data collection (environment stepping, transition probabilities) from GPU-bound training (network updates) maximizes hardware utilization.

### Including step count in state encoding (default)
Adding remaining episode time as a feature allows value functions to be time-aware, which is important for finite-horizon planning.

### Batched computation with shared state encoder
Processing multiple queries (e.g., all robot actions, all humans) in parallel with a shared encoding amortizes the expensive state encoding cost.

## Policy and Exploration

### Epsilon-greedy exploration in addition to power-law stochasticity
Even with beta_r < ∞ providing some exploration, epsilon-greedy adds directed exploration to ensure sufficient coverage of the state-action space early in training.

### Linear epsilon decay to small but non-zero value
Maintaining a small constant exploration rate (e.g., 0.01) after initial decay prevents the policy from becoming fully deterministic and unable to recover from bad local optima.

### Goal resampling probability (default 0.01 per step)
Periodically resampling human goals during episodes ensures the robot learns to handle diverse goal distributions rather than overfitting to episode-initial goals.

## Design Philosophy

### Separate base classes for environment-agnostic code
Abstracting the core Phase 2 logic from environment-specific details allows reuse across different domains (MultiGrid, continuous control, etc.) without code duplication.

### Modular network architecture with explicit interfaces
Each network (Q_r, V_h^e, X_h, U_r, V_r) is a separate module with clear inputs/outputs, making it easier to debug, test, and swap implementations.

### Configuration dataclass with validation
Centralizing all hyperparameters in Phase2Config with post-init validation prevents invalid configurations and makes hyperparameter tuning more systematic.

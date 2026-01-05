# Phase 2 Warm-up Design

This document describes the staged warm-up approach used in Phase 2 training to break mutual network dependencies and ensure stable convergence.

## Motivation

Phase 2 training involves multiple interdependent networks:

- **V_h^e**: State- and goal-dependent individual human goal achievement ability (given robot policy)
- **X_h**: State-dependent individual human power, based on aggregating V_h^e across goals (given robot policy)
- **U_r**: Robot's state-dependent intrinsic reward, based on aggregating X_h across humans
- **Q_r**: Robot's action-value function
- **V_r**: Robot's state-value function (optional, normally directly computed from U_r and Q_r)

These networks have circular dependencies:
- V_h^e depends on the robot policy π_r (derived from Q_r)
- X_h depends on π_r
- U_r depends on X_h and V_h^e
- Q_r depends on U_r (via V_r)

Training all networks simultaneously from random initialization can lead to:
1. **Unstable gradients**: Networks chase moving targets
2. **Poor convergence**: Circular dependencies create feedback loops
3. **Gradient clipping issues**: Large gradients from random networks

## Solution: Staged Warm-up

We break the circular dependencies by training networks in stages, starting with the most foundational network (V_h^e) and progressively adding dependent networks.

### Training Stages

| Stage | Duration (default) | Cumulative End | Active Networks | β_r | Learning Rate |
|-------|-------------------|----------------|-----------------|-----|---------------|
| 0 | 1,000 steps | 1,000 | V_h^e | 0 | constant |
| 1 | 1,000 steps* | 2,000 | V_h^e, X_h | 0 | constant |
| 2 | 1,000 steps** | 3,000 | V_h^e, X_h, U_r | 0 | constant |
| 3 | 1,000 steps | 4,000 | V_h^e, X_h, U_r, Q_r | 0 | constant |
| 4 | 1,000 steps*** | 5,000 | V_h^e, X_h, U_r, Q_r, V_r | 0 | constant |
| 5 | 2,000 steps | 7,000 | All | 0 → β_r | constant |
| 6 | remainder | — | All | β_r | 1/√t decay |

*Stage 1 (X_h) is skipped when `x_h_use_network=False`, reducing total warmup by 1,000 steps.
**Stage 2 (U_r) is skipped when `u_r_use_network=False` (default), reducing total warmup by 1,000 steps.
***Stage 4 (V_r) is skipped when `v_r_use_network=False` (default), reducing total warmup by 1,000 steps.

### Stage Details

#### Stage 0: V_h^e Only (steps 0 - 1,000)

**Goal**: Learn the human value function under a uniform random robot policy.

Since β_r = 0, the robot takes actions uniformly at random. This provides a stable target for V_h^e to learn against. The human value function captures how well humans can achieve their goals when the robot is not strategically helping (or hindering).

**Why start here**: V_h^e is the foundation of the EMPO objective. All other networks ultimately depend on accurate V_h^e estimates.

#### Stage 1: + X_h (optional, skipped when `x_h_use_network=False`)

**Goal**: Learn the aggregate goal achievement ability under the random robot policy.

X_h predicts the aggregate ability of human h to achieve various goals, computed as E_{g_h}[V_h^e(s, g_h)^ζ] over possible goals. With V_h^e already converging, X_h can learn meaningful ability estimates.

**Why this order**: X_h is needed by U_r (or directly for computing U_r when no U_r network), so it must be trained first.

**Note**: When `x_h_use_network=False`, X_h is computed directly from V_h^e samples via Monte Carlo estimation: X_h = mean(V_h^e(s, g)^ζ) over sampled goals. This stage is skipped entirely (`warmup_x_h_steps` is set to 0).

#### Stage 2: + U_r (optional, skipped when `u_r_use_network=False`)

**Goal**: Learn the robot's expected future value (excluding current action).

U_r = E[V_r(s')] where the expectation is over next states. With X_h providing state expectations and V_h^e providing human values, U_r has stable inputs to learn from.

**Why this order**: U_r is needed by Q_r for computing action values.

**Note**: When `u_r_use_network=False` (the default), U_r is computed directly from X_h values without a separate network, and this stage is skipped entirely (`warmup_u_r_steps` is set to 0).

#### Stage 3: + Q_r

**Goal**: Learn the robot's action-value function.

Q_r(s, a) = γ_r × E[V_r(s')] (Equation 4). Note that U_r is NOT added here—it's already included in V_r's definition: V_r(s) = U_r(s) + E_{a~π_r}[Q_r(s,a)] (Equation 9). With U_r and V_r components available, Q_r can learn meaningful action values.

**Note**: Even though Q_r is now trained, β_r remains 0, so the policy is still uniform random. This prevents the policy from affecting V_h^e, X_h, and U_r targets while Q_r is still learning.

#### Stage 4: + V_r (optional, skipped when `v_r_use_network=False`)

**Goal**: Learn the robot's state-value function under the still-random policy.

V_r(s) = U_r(s) + E_{a~π_r}[Q_r(s,a)] (Equation 9). With Q_r now providing action values, V_r can learn the expected value under the current policy. Since β_r = 0 during this stage, V_r learns values under a uniform random robot policy.

**Why this order**: V_r depends on Q_r, so it must be trained after Q_r has had time to converge.

**Note**: When `v_r_use_network=False` (the default), V_r is computed directly from U_r and Q_r without a separate network: V_r(s) = U_r(s) + sum(π_r(a|s) * Q_r(s,a)). This stage is skipped entirely (`warmup_v_r_steps` is set to 0).

#### Stage 5: β_r Ramp-up

**Goal**: Gradually transition from random to optimal policy.

β_r controls the "sharpness" of the robot policy:
- β_r = 0: Uniform random (all actions equally likely)
- β_r → ∞: Greedy (always take best action)

We use a **sigmoidal ramp-up**:

```
β_r(t) = β_r_nominal × σ(6 × (t/T - 0.5))
```

where:
- t = steps after warmup ends
- T = beta_r_rampup_steps (default: 2,000)
- σ = sigmoid function

This provides:
- **Slow start**: Minimal disruption to learned representations
- **Fast middle**: Efficient transition
- **Slow end**: Fine-tuning near optimal policy

The sigmoid is normalized so β_r goes from ~0 at t=0 to ~β_r_nominal at t=T.

#### Stage 6: Full Training (steps 7,000+)

**Goal**: Fine-tune all networks with the optimal policy.

All networks continue training with:
- β_r at nominal value (default: 10.0)
- Learning rate decaying as 1/√t

**Replay buffer clearing**: The replay buffer is cleared at BOTH transitions around the ramp-up phase:
- **Start of ramp-up (stage 4→5)**: Removes all transitions collected during warmup when β_r = 0 (uniform random robot policy)
- **End of ramp-up (stage 5→6)**: Removes transitions collected during ramp-up when β_r was increasing, ensuring fine-tuning uses only data collected with full β_r

**Learning rate decay**: After the full warmup, learning rates decay as:

```
lr(t) = lr_base / √t
```

where t counts from when Stage 6 begins. This satisfies theoretical convergence requirements while maintaining reasonable learning speed.

## Target Networks

To provide stable training targets and avoid chasing moving targets, the following **target networks** are maintained as frozen copies:

| Target Network | Purpose |
|----------------|---------|
| `q_r_target` | Stable Q_r for policy computation in V_r and V_h^e targets |
| `v_r_target` | Stable V_r for Q_r target computation (when `v_r_use_network=True`) |
| `v_h_e_target` | Stable V_h^e for TD targets and X_h computation |
| `x_h_target` | Stable X_h for U_r target computation (when `x_h_use_network=True`) |
| `u_r_target` | Stable U_r for V_r computation (both network and direct modes) |

Target networks are updated periodically via hard copy. Each network has its own update interval (controlled by `q_r_target_update_interval`, `v_r_target_update_interval`, `v_h_e_target_update_interval`, `x_h_target_update_interval`, and `u_r_target_update_interval`).

### V_r Computation

When `v_r_use_network=False` (default), V_r is computed directly as:

```
V_r(s) = U_r(s) + π_r(s) · Q_r(s)
```

For computing Q_r targets, we need V_r(s'). To ensure stability, this uses:
- **`u_r_target`** (frozen) for U_r(s')
- **`q_r_target`** (frozen) for Q_r(s') and π_r(s')

This prevents the Q_r target from depending on rapidly-changing U_r or Q_r networks, providing more stable training especially during beta_r ramp-up.

## Configuration Parameters

All warmup parameters are in `Phase2Config`. Each `warmup_*_steps` parameter specifies the **duration** of that stage (not cumulative):

```python
@dataclass
class Phase2Config:
    # Duration of each warmup stage (absolute, not cumulative)
    warmup_v_h_e_steps: int = 1000   # Duration of V_h^e-only stage
    warmup_x_h_steps: int = 1000     # Duration of V_h^e + X_h stage (0 if x_h_use_network=False)
    warmup_u_r_steps: int = 1000     # Duration of V_h^e + X_h + U_r stage (0 if u_r_use_network=False)
    warmup_q_r_steps: int = 1000     # Duration of V_h^e + X_h + (U_r) + Q_r stage
    
    # Beta_r ramp-up after warmup
    beta_r_rampup_steps: int = 2000  # Steps to ramp β_r from 0 to nominal
    beta_r: float = 10.0             # Nominal β_r value
    
    # Learning rate schedule
    use_sqrt_lr_decay: bool = True   # Use 1/√t decay after constant phase
    lr_constant_fraction: float = 0.0  # Fraction of total steps to keep LR constant (0.8 recommended)
    constant_lr_then_1_over_t: bool = False  # Use 1/t instead of 1/√t for final convergence
    
    # Network computation modes
    x_h_use_network: bool = True     # If False, warmup_x_h_steps is set to 0
    u_r_use_network: bool = False    # If False, warmup_u_r_steps is set to 0
```

### Learning Rate Schedule

The learning rate follows a three-phase schedule:

1. **Warmup/Ramp-up (constant)**: During warmup stages and β_r ramp-up, LR is constant.
2. **Constant phase**: After ramp-up, LR stays constant until `lr_constant_fraction` of total steps.
3. **Decay phase**: LR decays as 1/√t (or 1/t if `constant_lr_then_1_over_t=True`).

**Recommended settings for stable training:**

```python
config = Phase2Config(
    lr_constant_fraction=0.8,        # Keep LR constant for first 80% of training
    constant_lr_then_1_over_t=True,  # Use 1/t decay for proper convergence
    lr_q_r=1e-3,                     # Higher base LR (will be constant longer)
)
```

With 500k steps:
- Steps 0–7000: Constant LR (warmup/rampup)
- Steps 7000–400000: Constant LR (main learning phase)
- Steps 400000–500000: 1/t decay (convergence to expected values)

The 1/t decay satisfies the Robbins-Monro conditions (∑lr = ∞, ∑lr² < ∞) required for converging to true expected values.

**Note:** When `x_h_use_network=False`, `warmup_x_h_steps` is automatically set to 0 in `__post_init__`, effectively skipping the X_h warmup stage. Similarly, when `u_r_use_network=False` (the default), `warmup_u_r_steps` is automatically set to 0, and when `v_r_use_network=False` (the default), `warmup_v_r_steps` is automatically set to 0.

Internally, cumulative thresholds are computed:
- `_warmup_v_h_e_end = warmup_v_h_e_steps` (1000)
- `_warmup_x_h_end = _warmup_v_h_e_end + warmup_x_h_steps` (2000 or 1000 if x_h skipped)
- `_warmup_u_r_end = _warmup_x_h_end + warmup_u_r_steps` (3000 or earlier if stages skipped)
- `_warmup_q_r_end = _warmup_u_r_end + warmup_q_r_steps` (4000 or earlier if stages skipped)
- `_warmup_v_r_end = _warmup_q_r_end + warmup_v_r_steps` (5000 or earlier if stages skipped)

### Disabling Warmup

To disable warmup entirely (not recommended), set all warmup durations to 0:

```python
config = Phase2Config(
    warmup_v_h_e_steps=0,
    warmup_x_h_steps=0,
    warmup_u_r_steps=0,
    warmup_q_r_steps=0,
    warmup_v_r_steps=0,
    beta_r_rampup_steps=0,
)
```

### Shorter Warmup for Simple Environments

For simple environments (e.g., small grids), shorter warmup may suffice:

```python
config = Phase2Config(
    warmup_v_h_e_steps=200,   # 200 steps V_h^e only
    warmup_x_h_steps=200,     # 200 steps + X_h
    warmup_u_r_steps=200,     # 200 steps + U_r (only if u_r_use_network=True)
    warmup_q_r_steps=200,     # 200 steps + Q_r
    beta_r_rampup_steps=400,  # 400 steps β_r ramp-up
)
# Total: 800-1000 steps warmup + 400 steps ramp-up
```

## Helper Methods

`Phase2Config` provides several methods for querying warmup state:

```python
config = Phase2Config()

# Check current phase
config.is_in_warmup(step)       # True if step < warmup_q_r_steps
config.is_in_rampup(step)       # True if in β_r ramp-up phase
config.is_fully_trained(step)   # True if past all warmup/rampup

# Get current values
config.get_warmup_stage(step)        # Numeric stage (0-5)
config.get_warmup_stage_name(step)   # Human-readable stage name
config.get_effective_beta_r(step)    # Current β_r value
config.get_active_networks(step)     # Set of active network names

# Get learning rate for a network
config.get_learning_rate('q_r', step, update_count)

# Get stage transition points
config.get_stage_transition_steps()  # List of (step, description) tuples
```

## TensorBoard Logging

During training, the following warmup-related metrics are logged:

- `Warmup/stage`: Current stage number (0-5)
- `Warmup/stage_transition`: 1.0 at stage transitions, 0.0 otherwise (creates vertical lines)
- `Warmup/effective_beta_r`: Current β_r value
- `Warmup/is_warmup`: 1.0 during warmup, 0.0 after
- `Warmup/active_networks_mask`: Bitmask of active networks
- `LearningRate/*`: Current learning rate for each network

## Console Output

When `verbose=True`, stage transitions are logged to console:

```
[Warmup] Starting in stage 0: Stage 1: V_h^e only (active networks: {'v_h_e'})

[Warmup] Stage transition at step 1000, episode 20:
  Stage 1: V_h^e only -> Stage 2: V_h^e + X_h
  Active networks: {'v_h_e', 'x_h'}
  Effective beta_r: 0.0000

[Warmup] Stage transition at step 6000, episode 120:
  β_r ramping -> Full training
  Active networks: {'v_h_e', 'x_h', 'u_r', 'q_r'}
  Effective beta_r: 10.0000
  [Training] Cleared replay buffer (50000 transitions) after β_r ramp-up
```

## Theoretical Justification

### Breaking Circular Dependencies

The key insight is that with β_r = 0, the robot policy is fixed (uniform random) and independent of Q_r. This breaks the circular dependency:

```
Without warmup (circular):
V_h^e ← π_r ← Q_r ← U_r ← X_h ← π_r ← Q_r ← ...

With warmup (acyclic during stages 0-3):
V_h^e ← π_r(uniform) [fixed]
X_h ← π_r(uniform) [fixed]
U_r ← X_h, V_h^e [already trained]
Q_r ← U_r [already trained]
```

### Sigmoidal β_r Ramp-up

The sigmoid ramp-up provides several benefits:

1. **Continuity**: Smooth transition avoids sudden policy changes
2. **Stability**: Slow start allows networks to adapt gradually
3. **Efficiency**: Fast middle section accelerates convergence
4. **Precision**: Slow end allows fine-tuning near optimal

### Learning Rate Decay

The 1/√t decay after full warmup satisfies the Robbins-Monro conditions for stochastic approximation:

- Σ lr(t) = ∞ (ensures convergence)
- Σ lr(t)² < ∞ (ensures bounded variance)

This is a compromise between:
- 1/t decay (optimal for expectations, but slow)
- Constant LR (fast but may not converge)

## Model-Based Targets

When `use_model_based_targets=True` (default), the trainer computes targets using the
expected value over all possible successor states, weighted by transition probabilities:

```python
# Instead of: target = reward + gamma * V(s'_observed)
# We compute: target = reward + gamma * E_{s'}[V(s')]
#                    = reward + gamma * sum_s' p(s'|s,a) * V(s')
```

**Benefits:**
- **Lower variance**: Reduces variance in target estimates
- **Action consistency**: Actions with identical transition distributions get identical Q-values
- **Better credit assignment**: All possible successor states contribute to the gradient

**Implementation details:**
- Transition probabilities are cached at collection time (stored in replay buffer)
- Caching reduces `transition_probabilities()` calls by ~300× during training
- Both Q_r and V_h^e use model-based targets when enabled

To disable model-based targets (not recommended):

```python
config = Phase2Config(use_model_based_targets=False)
```

## Understanding the Loss Values

**Important:** The MSE losses for Q_r, U_r, and X_h will NOT converge to zero, even with
perfect training. This is expected behavior due to irreducible variance.

### Why losses don't go to zero

These networks predict *expected values* of stochastic quantities:

- **Q_r(s, a_r)** predicts E[γ_r V_r(s')], but each sampled s' has different V_r(s')
- **U_r(s)** predicts the intrinsic reward, which varies with sampled goals
- **X_h(s)** predicts E[V_h^e(s, g_h)^ζ], aggregated over sampled goals

The MSE loss is bounded from below by the **irreducible variance** of the target:

```
MSE = E[(prediction - target)²]
    = E[(prediction - E[target])²] + E[(target - E[target])²]
    = (bias)² + (irreducible variance)
```

Even with zero bias (perfect prediction of the expected value), the variance term remains.

### What to look for instead

- **Loss plateau**: Losses stabilizing (not increasing) indicates convergence
- **Policy behavior**: The learned policy should produce reasonable robot behavior
- **V_h^e convergence**: V_h^e losses should be lower (bounded [0,1] with binary rewards)
- **TensorBoard metrics**: Monitor `Metrics/mean_episode_reward` and policy entropy

## Troubleshooting

### V_h^e not converging during Stage 0

- **Check gradient clipping**: May be too aggressive (try increasing from 1.0 to 100.0)
- **Check network size**: May be too small for the environment
- **Check learning rate**: May need adjustment

### Instability during β_r ramp-up

- **Extend ramp-up**: Increase `beta_r_rampup_steps`
- **Lower nominal β_r**: Reduce `beta_r` for a softer policy
- **Increase warmup**: Networks may not be fully trained before ramp-up

### Poor final performance

- **Check total training time**: May need more episodes after warmup
- **Verify environment rewards**: Ensure human goals are achievable
- **Inspect TensorBoard**: Look for signs of instability or non-convergence

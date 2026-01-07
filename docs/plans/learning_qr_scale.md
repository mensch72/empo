# Learning Q-values of Unknown Scale

## Problem Statement

In Phase 2 training, we observe that:
- **Predictions** (Q_r network output) stabilize around -1.4
- **Targets** (computed from Bellman equation) are around -200 to -400
- The network cannot bridge this gap because gradients are either too small (with low LR) or cause instability (with high LR)

The fundamental issue is that the network's output scale is initialized around O(1), but true Q-values may be orders of magnitude larger. Standard MSE loss creates gradients proportional to the error magnitude, which can be problematic when:
1. Early in training, errors are huge → unstable gradients
2. Late in training with decayed LR, the network can't catch up

## Chosen Solution: Theory-Grounded z-Space with Two-Phase Loss

### Key Insight

Neural networks naturally output values near 0 (due to weight initialization). Getting a network to output Q = -400 requires pushing weights to extreme values. But getting it to output z ≈ 0.4 (where z = f(Q) for our transformation) is trivial.

### Transformation

We use the theory-grounded power transformation:
```
z = f(Q) = (-Q)^{-1/(ηξ)}
Q = f^{-1}(z) = -z^{-ηξ}
```

This maps:
- Q ∈ (-∞, -1] → z ∈ (0, 1]
- Q = -1 → z = 1
- Q → -∞ → z → 0

The network predicts z ∈ (0, 1], which is a natural output range.

### Two-Phase Loss Strategy

**Phase A (constant LR, exploring the value space):**
```python
loss = (z_pred - z_target)²  where z_target = f(Q_target)
```
The z-space MSE gives balanced gradients across all scales, allowing the network to quickly find the right ballpark.

**Phase B (1/t decay, converging to expectations):**
```python
loss = (Q_pred - Q_target)²  where Q_pred = f^{-1}(z_pred)
```
Switch to Q-space MSE for proper Robbins-Monro convergence to arithmetic means.

**Rationale:**
- During Phase A, we want the network to output the correct *structure* of values (which states have higher/lower values). The z-space loss helps because it doesn't overweight large Q-values.
- During Phase B, we want convergence to the *exact* expected values. The Q-space loss with 1/t decay satisfies Robbins-Monro conditions for unbiased convergence.

The losses are mathematically equivalent in terms of gradients (chain rule), but the scale of the error term differs, which affects the effective learning rate.

### Configuration

```python
use_z_space_transform: bool = True  # Enable theory-grounded z-space
# Uses lr_constant_fraction and constant_lr_then_1_over_t to determine phases
```

## Standard Approaches from Literature

### 1. Pop-Art (Preserving Outputs Precisely, while Adaptively Rescaling Targets)

**Paper**: van Hasselt et al., "Learning values across many orders of magnitude" (DeepMind, 2016)

**Key idea**: Normalize targets to zero mean and unit variance, but adjust network weights to preserve the unnormalized output.

**Algorithm**:
```python
# Maintain running statistics
μ, σ = running_mean(targets), running_std(targets)

# Normalize targets for loss computation
normalized_target = (target - μ) / σ

# After each update to (μ, σ), adjust final layer weights to preserve outputs:
# If old stats were (μ_old, σ_old) and new are (μ, σ):
W_new = W_old * (σ_old / σ)
b_new = (σ_old * b_old + μ_old - μ) / σ
```

**Pros**:
- Network always sees normalized targets ∈ [-3, 3] approximately
- Output semantics preserved across normalization changes
- Proven effective in multi-task RL with varying reward scales

**Cons**:
- Requires modifying network weights outside of gradient descent
- Running statistics may lag behind non-stationary target distributions
- Adds implementation complexity

### 2. Value Function Rescaling (MuZero/R2D2 style)

**Papers**: 
- Pohlen et al., "Observe and Look Further" (2018)
- Schrittwieser et al., "Mastering Atari...with Planning" (MuZero, 2020)

**Key idea**: Apply an invertible transformation to squash large values.

**Transformation** (signed hyperbolic):
```python
def h(x):
    """Squash values to bounded range."""
    return sign(x) * (sqrt(|x| + 1) - 1) + ε * x

def h_inv(x):
    """Inverse transformation."""
    return sign(x) * ((sqrt(1 + 4*ε*(|x| + 1 + ε)) - 1) / (2*ε))² - 1)
```

With ε=0.001, this maps:
- x = -400 → h(x) ≈ -19.0
- x = -1 → h(x) ≈ -0.41
- x = 0 → h(x) = 0

**Training**:
```python
# Network predicts in transformed space
predicted_h = network(state)

# Transform targets
target_h = h(target)

# Loss in transformed space
loss = (predicted_h - target_h)²

# For policy/planning, invert predictions
q_value = h_inv(predicted_h)
```

**Pros**:
- Simple, no weight manipulation needed
- Bounded gradients for large targets
- Works with any optimizer

**Cons**:
- Introduces bias (network learns E[h(Q)], not h(E[Q]))
- Need to tune ε parameter
- Inverse transformation can amplify prediction errors

### 3. Adaptive Learning Rate per Output (Natural Gradient approximation)

**Key idea**: Scale learning rate inversely with target variance.

**Approaches**:
- **Adam** already adapts per-parameter, but not per-output
- **Per-head normalization**: Separate normalization for each output dimension
- **Uncertainty-weighted**: lr ∝ 1/prediction_variance (if using ensembles)

### 4. Target Normalization (Batch/Layer Normalization of Targets)

**Key idea**: Normalize targets within each batch.

```python
# Per-batch normalization
batch_mean = targets.mean()
batch_std = targets.std()
normalized_targets = (targets - batch_mean) / batch_std
```

**Pros**: Very simple
**Cons**: 
- High variance between batches
- Loses absolute scale information (network can't learn true magnitudes)
- Not suitable when we need actual Q-values for policy

### 5. Huber Loss / Smooth L1

**Key idea**: Use robust loss function instead of MSE.

```python
def huber_loss(pred, target, δ=1.0):
    error = pred - target
    return where(|error| < δ, 
                 0.5 * error²,
                 δ * (|error| - 0.5 * δ))
```

**Pros**: Bounded gradients for large errors
**Cons**: Doesn't solve the scale mismatch, just prevents explosion

### 6. Percentile Normalization

**Key idea**: Normalize by percentiles instead of mean/std (robust to outliers).

```python
p10, p90 = percentile(targets_history, [10, 90])
normalized = (target - p10) / (p90 - p10)
```

### 7. Symlog Transformation (Dreamer V3)

**Paper**: Hafner et al., "Mastering Diverse Domains through World Models" (2023)

```python
def symlog(x):
    return sign(x) * log(|x| + 1)

def symexp(x):
    return sign(x) * (exp(|x|) - 1)
```

Maps: -400 → -6.0, -1 → -0.69, 0 → 0

**Pros**: Simple, interpretable, proven in diverse domains
**Cons**: Similar bias issues as MuZero rescaling

## Comparison for Our Use Case

| Method | Handles Negatives | Preserves Convergence | Complexity | Proven in RL |
|--------|-------------------|----------------------|------------|--------------|
| Pop-Art | ✓ | ✓ (with care) | Medium | ✓ (Impala, R2D2) |
| Value Rescaling | ✓ | ✗ (biased) | Low | ✓ (MuZero) |
| Symlog | ✓ | ✗ (biased) | Low | ✓ (Dreamer V3) |
| Target Normalization | ✓ | ✗ | Low | Partially |
| Huber Loss | ✓ | ✓ | Low | ✓ (DQN) |

## Recommendation for EMPO Phase 2

### Primary Approach: Theory-Grounded Power Transformation

Since Q_r, U_r, V_r are all **guaranteed negative** by the EMPO theory, and their scale is determined by the power transformations in equations (4)-(9), we can use a principled inverse transformation.

**Key insight**: The transformation chain that creates U_r from X_h is:
```
X_h ∈ (0,1] → X_h^{-ξ} ∈ [1,∞) → E_h[...]^η ∈ [1,∞) → -(...) = U_r ∈ (-∞,-1]
```

We can approximately "undo" this by predicting:
```
z = (-Q_r)^{-1/(ηξ)}
```

**Transformation functions**:
```python
def to_z_space(q: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform Q < 0 to z-space: z = (-Q)^{-1/(ηξ)} ∈ (0, ∞)."""
    return torch.pow(-q.clamp(max=-eps), -1.0 / (eta * xi))

def from_z_space(z: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform z-space back to Q < 0: Q = -z^{-ηξ}."""
    return -torch.pow(z.clamp(min=eps), -eta * xi)
```

**Scale mapping** (with η=1.1, ξ=1.0):
| Q_r | z = (-Q_r)^{-1/(ηξ)} |
|-----|----------------------|
| -10000 | 0.00052 |
| -1000 | 0.0016 |
| -100 | 0.012 |
| -10 | 0.095 |
| -1 | 1.0 |

**Why this is ideal for EMPO**:

1. **Theory-grounded**: The transformation uses the same power parameters (η, ξ) from the EMPO equations. The z-space relates back to the X_h scale that humans operate on.

2. **Natural bounds**: For typical U_r ∈ (-∞, -1], we get z ∈ (0, 1]. The network predicts values in a natural range.

3. **Policy computation**: 
   ```python
   # π ∝ |Q_r|^{β_r} = (-Q_r)^{β_r} = z^{-ηξβ_r}
   # log π ∝ -ηξβ_r · log(z)
   # Softmax: softmax(-η*ξ*β_r * log(z_a)) for each action a
   ```

4. **Bounded gradients**: The power transformation compresses large values.

5. **Interpretable**: When z ≈ 1, Q_r ≈ -1 (neutral). When z → 0, Q_r → -∞ (very negative).

### Alternative: Simple Log-Space

For simpler implementation, predict z = log(-Q_r):

```python
def to_log_space(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Transform Q < 0 to log space: z = log(-Q)."""
    return torch.log(-q.clamp(max=-eps))

def from_log_space(z: Tensor) -> Tensor:
    """Transform log space back to Q < 0: Q = -exp(z)."""
    return -torch.exp(z)
```

| Q_r | z = log(-Q_r) |
|-----|---------------|
| -400 | 6.0 |
| -1 | 0 |
| -0.01 | -4.6 |

**Pros**: Simpler, no dependency on η/ξ
**Cons**: Not theory-grounded, z can be any real number

### Fallback: Pop-Art Normalization

Pop-Art remains a solid option if we need exact arithmetic means:
- More complex (requires weight adjustment after each batch)
- May be needed during 1/t decay phase for proper Robbins-Monro convergence

## Implementation Plan

### Phase 1: Add Theory-Grounded Power Transformation

1. **Helper functions** in `src/empo/learning_based/phase2/value_transforms.py`:
   ```python
   def to_z_space(q: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
       """Transform Q < 0 to z-space: z = (-Q)^{-1/(ηξ)} ∈ (0, ∞)."""
       return torch.pow(-q.clamp(max=-eps), -1.0 / (eta * xi))
   
   def from_z_space(z: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
       """Transform z-space back to Q < 0: Q = -z^{-ηξ}."""
       return -torch.pow(z.clamp(min=eps), -eta * xi)
   
   def z_space_loss(z_pred: Tensor, q_target: Tensor, 
                    eta: float, xi: float, eps: float = 1e-8) -> Tensor:
       """MSE loss in z-space."""
       z_target = to_z_space(q_target, eta, xi, eps)
       return F.mse_loss(z_pred, z_target)
   
   # Also include log-space as simpler alternative
   def to_log_space(q: Tensor, eps: float = 1e-8) -> Tensor:
       """Transform Q < 0 to log space: z = log(-Q)."""
       return torch.log(-q.clamp(max=-eps))
   
   def from_log_space(z: Tensor) -> Tensor:
       """Transform log space back to Q < 0: Q = -exp(z)."""
       return -torch.exp(z)
   ```

2. **Config options** in `Phase2Config`:
   ```python
   # Value transformation for Q_r, U_r, V_r (all guaranteed negative)
   # Options: 'none' (raw MSE), 'log' (log-space), 'power' (theory-grounded)
   q_r_transform: str = 'power'   # For Q_r network
   u_r_transform: str = 'power'   # For U_r network (if u_r_use_network=True)
   v_r_transform: str = 'power'   # For V_r network (if v_r_use_network=True)
   
   # During 1/t decay phase, optionally switch to MSE on raw values
   # for proper Robbins-Monro convergence to arithmetic mean
   transform_during_decay: bool = False  # If False, switch to 'none' during decay
   ```

3. **Network output interpretation**:
   - Raw network output is z > 0 (use softplus or ReLU+eps to ensure positivity)
   - For loss: compare z to transform(target)
   - For policy: 
     ```python
     # Power transform: π ∝ |Q|^β = z^{-ηξβ}
     # Log-softmax: -ηξβ * log(z) - logsumexp(...)
     log_policy = -eta * xi * beta_r * torch.log(z)
     policy = F.softmax(log_policy, dim=-1)
     ```
   - For logging: Q_r = from_z_space(z)

4. **Initialization**: 
   - Initialize final layer so z ≈ 1 initially (i.e., Q_r ≈ -1)
   - Use softplus(output) + eps to ensure z > 0

## Implementation Plan (Updated)

### Architecture

1. **Network output**: Network predicts raw value `x ∈ R`, transformed to `z = sigmoid(x) ∈ (0, 1)`
2. **Q-value recovery**: `Q = f^{-1}(z) = -z^{-ηξ}` (used for policy and logging)
3. **Loss function**: Depends on training phase (see below)

### Two-Phase Loss

```python
def compute_q_r_loss(z_pred, q_target, config, training_step):
    """
    Compute Q_r loss with phase-dependent transformation.
    
    Phase A (constant LR): MSE in z-space
    Phase B (1/t decay): MSE in Q-space
    """
    # Check if we're in the decay phase
    total_warmup = config.get_total_warmup_steps()
    decay_start = total_warmup + int(config.lr_constant_fraction * 
                                      (config.num_training_steps - total_warmup))
    in_decay_phase = training_step >= decay_start and config.constant_lr_then_1_over_t
    
    if in_decay_phase:
        # Phase B: Q-space MSE for Robbins-Monro convergence
        q_pred = from_z_space(z_pred, config.eta, config.xi)
        return F.mse_loss(q_pred, q_target)
    else:
        # Phase A: z-space MSE for balanced gradients
        z_target = to_z_space(q_target, config.eta, config.xi)
        return F.mse_loss(z_pred, z_target)
```

### Files to Modify

1. **New file**: `src/empo/learning_based/phase2/value_transforms.py`
   - `to_z_space(q, eta, xi)` - Q to z transformation
   - `from_z_space(z, eta, xi)` - z to Q transformation
   
2. **Modify**: `src/empo/learning_based/phase2/config.py`
   - Add `use_z_space_transform: bool = True`
   
3. **Modify**: `src/empo/learning_based/phase2/robot_q_network.py`
   - Change `ensure_negative()` to output z ∈ (0, 1) when z-space enabled
   - Add method to convert z to Q
   
4. **Modify**: `src/empo/learning_based/phase2/robot_value_network.py`
   - Same changes for V_r
   
5. **Modify**: `src/empo/learning_based/phase2/trainer.py`
   - Use phase-dependent loss computation
   - Log both z and Q values

6. **Modify lookup table networks**: Store z values, convert to Q for policy

### Policy Computation with z-Space

When networks output z, the policy computation simplifies:
```python
# π ∝ |Q_r|^{β_r} = (-Q_r)^{β_r} = (z^{-ηξ})^{β_r} = z^{-ηξβ_r}
# log π ∝ -ηξβ_r * log(z)

log_policy_unnorm = -config.eta * config.xi * effective_beta_r * torch.log(z)
policy = F.softmax(log_policy_unnorm, dim=-1)
```

This is numerically stable and avoids computing huge Q-values.

### Initialization

Initialize network so z ≈ 0.5 initially (Q ≈ -2^{ηξ} ≈ -2.14 for η=1.1, ξ=1):
- Final layer bias = 0 (sigmoid(0) = 0.5)
- Final layer weights near 0

## References

1. van Hasselt et al., "Learning values across many orders of magnitude" (2016)
2. Pohlen et al., "Observe and Look Further: Achieving Consistent Performance on Atari" (2018)
3. Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020)
4. Hafner et al., "Mastering Diverse Domains through World Models" (2023)
5. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning" (2018)

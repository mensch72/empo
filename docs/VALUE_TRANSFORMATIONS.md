# Value Transformations for Phase 2 Networks

## Problem Statement

In Phase 2 training, we observe that:
- **Predictions** (Q_r network output) stabilize around -1.4
- **Targets** (computed from Bellman equation) are around -200 to -400
- The network cannot bridge this gap because gradients are either too small (with low LR) or cause instability (with high LR)

The fundamental issue is that neural networks naturally output values near 0 (due to weight initialization). Getting a network to output Q = -400 requires pushing weights to extreme values. But getting it to output z ≈ 0.4 is trivial.

## Chosen Solution: Theory-Grounded z-Space with Two-Phase Loss

### Key Insight

We use invertible transformations that map the target domain to z ∈ (0, 1], which is a natural output range for neural networks.

### Transformations by Network Type

#### Q_r and V_r Networks (predict negative values)

Both Q_r and V_r are guaranteed negative by EMPO theory: Q_r, V_r ∈ (-∞, -1].

**Transformation**:
```
z = f(Q) = (-Q)^{-1/(ηξ)}
Q = f^{-1}(z) = -z^{-ηξ}
```

This maps:
- Q ∈ (-∞, -1] → z ∈ (0, 1]
- Q = -1 → z = 1
- Q → -∞ → z → 0

**Scale mapping** (with η=1.1, ξ=1.0):
| Q | z = (-Q)^{-1/(ηξ)} |
|---|---------------------|
| -10000 | 0.00052 |
| -1000 | 0.0016 |
| -100 | 0.012 |
| -10 | 0.095 |
| -1 | 1.0 |

**Implementation** (in `value_transforms.py`):
```python
def to_z_space(q: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform Q < 0 to z-space: z = (-Q)^{-1/(ηξ)} ∈ (0, 1]."""
    return torch.pow(-q.clamp(max=-eps), -1.0 / (eta * xi))

def from_z_space(z: Tensor, eta: float, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform z-space back to Q < 0: Q = -z^{-ηξ}."""
    return -torch.pow(z.clamp(min=eps), -eta * xi)
```

#### U_r Network (predicts intermediate value y, derives U_r)

The U_r network has a different structure. It predicts an intermediate value y, then derives U_r:

**Network predicts**: y = E_h[X_h^{-ξ}] where y ∈ [1, ∞)
**U_r is derived**: U_r = -y^η ∈ (-∞, -1]

**Transformation for y**:
```
z = y^{-1/ξ}
y = z^{-ξ}
```

This maps:
- y ∈ [1, ∞) → z ∈ (0, 1]
- y = 1 → z = 1
- y → ∞ → z → 0

**Scale mapping** (with ξ=1.0):
| y | z = y^{-1/ξ} | U_r = -y^η (η=1.1) |
|---|--------------|---------------------|
| 1 | 1.0 | -1.0 |
| 10 | 0.1 | -12.6 |
| 100 | 0.01 | -158.5 |
| 1000 | 0.001 | -1995.3 |

**Implementation** (in `value_transforms.py`):
```python
def y_to_z_space(y: Tensor, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform y ≥ 1 to z-space: z = y^{-1/ξ} ∈ (0, 1]."""
    return torch.pow(y.clamp(min=1.0), -1.0 / xi)

def z_to_y_space(z: Tensor, xi: float, eps: float = 1e-8) -> Tensor:
    """Transform z-space back to y ≥ 1: y = z^{-ξ}."""
    return torch.pow(z.clamp(min=eps, max=1.0), -xi)
```

**Why y instead of U_r?**
- The network naturally computes E_h[X_h^{-ξ}] = y first
- U_r = -y^η is a simple derivation
- Training on y (or z = y^{-1/ξ}) is more stable because the gradient flows directly to the expectation

### Two-Phase Loss Strategy

**Phase A (constant LR, exploring the value space):**
```python
loss = (z_pred - z_target)²  where z_target = f(target)
```
The z-space MSE gives balanced gradients across all scales, allowing the network to quickly find the right ballpark.

**Phase B (1/t decay, converging to expectations):**
```python
loss = (value_pred - value_target)²  # In original space (Q or y)
```
Switch to original-space MSE for proper Robbins-Monro convergence to arithmetic means.

**Phase detection**:
```python
def is_in_decay_phase(step):
    total_warmup = get_total_warmup_steps()
    decay_start = total_warmup + int(lr_constant_fraction * 
                                      (num_training_steps - total_warmup))
    return step >= decay_start and constant_lr_then_1_over_t
```

### Configuration

```python
use_z_space_transform: bool = True  # Enable theory-grounded z-space
# Uses lr_constant_fraction and constant_lr_then_1_over_t to determine phases
```

### Policy Computation with z-Space

When Q_r networks output z, the policy computation simplifies:
```python
# π ∝ |Q_r|^{β_r} = (-Q_r)^{β_r} = (z^{-ηξ})^{β_r} = z^{-ηξβ_r}
# log π ∝ -ηξβ_r * log(z)

log_policy_unnorm = -config.eta * config.xi * effective_beta_r * torch.log(z)
policy = F.softmax(log_policy_unnorm, dim=-1)
```

This is numerically stable and avoids computing huge Q-values.

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

**Pros**:
- Simple, no weight manipulation needed
- Bounded gradients for large targets
- Works with any optimizer

**Cons**:
- Introduces bias (network learns E[h(Q)], not h(E[Q]))
- Need to tune ε parameter
- Inverse transformation can amplify prediction errors

### 3. Symlog Transformation (Dreamer V3)

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

### 4. Other Approaches

- **Huber Loss**: Bounded gradients but doesn't solve scale mismatch
- **Target Normalization**: Simple but loses absolute scale information
- **Adaptive LR**: Adam adapts per-parameter but not per-output

## Comparison for Our Use Case

| Method | Handles Negatives | Preserves Convergence | Theory-Grounded | Complexity |
|--------|-------------------|----------------------|-----------------|------------|
| **EMPO Power Transform** | ✓ | ✓ (two-phase) | ✓ | Low |
| Pop-Art | ✓ | ✓ (with care) | ✗ | Medium |
| Value Rescaling | ✓ | ✗ (biased) | ✗ | Low |
| Symlog | ✓ | ✗ (biased) | ✗ | Low |

**Why EMPO Power Transform is ideal:**

1. **Theory-grounded**: The transformation uses the same power parameters (η, ξ) from the EMPO equations. The z-space relates back to the X_h scale that humans operate on.

2. **Natural bounds**: For Q_r, V_r ∈ (-∞, -1] and y ∈ [1, ∞), we get z ∈ (0, 1]. The network predicts values in a natural range.

3. **Unbiased convergence**: The two-phase strategy uses z-space for exploration, then switches to original-space MSE for proper Robbins-Monro convergence.

4. **Interpretable**: When z ≈ 1, the value is near its minimum magnitude. When z → 0, the value is very large in magnitude.

## Implementation Summary

### Files

1. **`src/empo/nn_based/phase2/value_transforms.py`** - All transformation functions:
   - `to_z_space(q, eta, xi)` - Q → z for Q_r, V_r
   - `from_z_space(z, eta, xi)` - z → Q for Q_r, V_r
   - `y_to_z_space(y, xi)` - y → z for U_r
   - `z_to_y_space(z, xi)` - z → y for U_r

2. **`src/empo/nn_based/phase2/config.py`**:
   - `use_z_space_transform: bool = False` - Enable z-space transformations
   - `is_in_decay_phase(step)` - Determine which loss phase

3. **`src/empo/nn_based/phase2/trainer.py`**:
   - Q_r loss: z-space MSE in Phase A, Q-space MSE in Phase B
   - V_r loss: z-space MSE in Phase A, V_r-space MSE in Phase B
   - U_r loss: z-space MSE in Phase A, y-space MSE in Phase B

### Network Architecture

- **Q_r, V_r**: Network predicts z ∈ (0, 1], converts to Q = -z^{-ηξ} for policy
- **U_r**: Network predicts y ≥ 1, converts to U_r = -y^η; training uses z = y^{-1/ξ}

## References

1. van Hasselt et al., "Learning values across many orders of magnitude" (2016)
2. Pohlen et al., "Observe and Look Further: Achieving Consistent Performance on Atari" (2018)
3. Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020)
4. Hafner et al., "Mastering Diverse Domains through World Models" (2023)

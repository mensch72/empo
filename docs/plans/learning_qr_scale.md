# Learning Q-values of Unknown Scale

## Problem Statement

In Phase 2 training, we observe that:
- **Predictions** (Q_r network output) stabilize around -1.4
- **Targets** (computed from Bellman equation) are around -200 to -400
- The network cannot bridge this gap because gradients are either too small (with low LR) or cause instability (with high LR)

The fundamental issue is that the network's output scale is initialized around O(1), but true Q-values may be orders of magnitude larger. Standard MSE loss creates gradients proportional to the error magnitude, which can be problematic when:
1. Early in training, errors are huge → unstable gradients
2. Late in training with decayed LR, the network can't catch up

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

### Primary Approach: Pop-Art Normalization

Pop-Art is the best fit because:
1. **Convergence matters**: We need networks to converge to *expected values* (Robbins-Monro), not transformed values
2. **Policy depends on Q-values**: The power-law policy π ∝ |Q|^β requires actual Q magnitudes
3. **Proven at scale**: Used successfully in Impala, R2D2, Agent57

### Fallback: Symlog during constant LR phase

During the constant LR phase (before Robbins-Monro convergence), we could use Symlog for stability, then switch to Pop-Art or MSE for the final 1/t decay phase.

## Implementation Plan

### Phase 1: Add Pop-Art Normalizer

1. **Create `PopArtNormalizer` class** in `src/empo/nn_based/phase2/normalization.py`:
   ```python
   class PopArtNormalizer:
       def __init__(self, beta=0.0001):
           self.mean = 0.0
           self.var = 1.0
           self.beta = beta  # EMA decay rate
       
       def update(self, targets: Tensor) -> None:
           """Update running statistics."""
           
       def normalize(self, x: Tensor) -> Tensor:
           """Normalize values."""
           
       def denormalize(self, x: Tensor) -> Tensor:
           """Denormalize values."""
           
       def update_network_weights(self, linear_layer: nn.Linear, 
                                   old_mean: float, old_std: float) -> None:
           """Adjust final layer weights to preserve outputs."""
   ```

2. **Config options** in `Phase2Config`:
   ```python
   # Pop-Art normalization for Q_r, U_r, V_r
   use_popart_normalization: bool = True
   popart_beta: float = 0.0001  # EMA decay for running stats
   
   # Per-network control
   popart_q_r: bool = True
   popart_u_r: bool = True  # Only if u_r_use_network=True
   popart_v_r: bool = True  # Only if v_r_use_network=True
   ```

3. **Integration points** in trainer:
   - After computing targets, before loss: `normalized_targets = normalizer.normalize(targets)`
   - After each batch: `normalizer.update(targets)` and `normalizer.update_network_weights(...)`
   - For policy sampling: use raw network output (already in correct scale due to weight adjustment)

### Phase 2: Add Symlog Option

1. **Symlog functions** in `normalization.py`:
   ```python
   def symlog(x: Tensor) -> Tensor:
       return torch.sign(x) * torch.log1p(torch.abs(x))
   
   def symexp(x: Tensor) -> Tensor:
       return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
   ```

2. **Config option**:
   ```python
   # Alternative: Symlog transformation (simpler but biased)
   use_symlog_targets: bool = False
   
   # Hybrid: Symlog during constant LR, Pop-Art during decay
   use_hybrid_normalization: bool = False
   ```

### Phase 3: Logging and Diagnostics

1. **TensorBoard metrics**:
   - `PopArt/mean_q_r`, `PopArt/std_q_r`: Running statistics
   - `PopArt/weight_adjustment_magnitude`: How much weights change
   - `Targets/raw_mean`, `Targets/normalized_mean`: Before/after normalization

2. **Sanity checks**:
   - Assert normalized targets are roughly in [-5, 5]
   - Warn if std is very small (< 0.01) or very large (> 100)
   - Log when weight adjustments are large

### Phase 4: Testing

1. **Unit tests**:
   - `test_popart_normalize_denormalize_inverse()`
   - `test_popart_weight_adjustment_preserves_output()`
   - `test_symlog_symexp_inverse()`

2. **Integration tests**:
   - Train on simple environment, verify Q-values converge to correct scale
   - Compare MSE vs Pop-Art vs Symlog on heavy-tailed target distribution

### Files to Modify

1. **New file**: `src/empo/nn_based/phase2/normalization.py`
2. **Modify**: `src/empo/nn_based/phase2/config.py` - Add config options
3. **Modify**: `src/empo/nn_based/phase2/trainer.py` - Integrate normalizer
4. **Modify**: `src/empo/nn_based/multigrid/phase2/trainer.py` - Same
5. **New file**: `tests/test_popart_normalization.py`
6. **Update**: `docs/WARMUP_DESIGN.md` - Document normalization options

### Open Questions

1. **Separate normalizers per network?** Q_r, U_r, V_r may have different scales. Probably yes.

2. **Share statistics across state-action pairs?** Pop-Art typically uses global statistics. For our case with (state, action) → Q, we use global stats across all Q(s,a).

3. **Interaction with 1/t decay?** During 1/t decay phase, Pop-Art weight adjustments may interfere with convergence. Options:
   - Freeze normalizer statistics during decay phase
   - Use very small beta during decay phase
   - Switch to pure MSE during decay phase

4. **Lookup table mode?** Pop-Art doesn't apply to lookup tables (no final linear layer). For lookup tables, consider direct target normalization or adaptive per-entry learning rates (already implemented).

5. **V_h^e normalization?** V_h^e is bounded in [0, 1] by construction, so probably doesn't need normalization. But X_h could benefit if it varies widely.

## References

1. van Hasselt et al., "Learning values across many orders of magnitude" (2016)
2. Pohlen et al., "Observe and Look Further: Achieving Consistent Performance on Atari" (2018)
3. Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2020)
4. Hafner et al., "Mastering Diverse Domains through World Models" (2023)
5. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning" (2018)

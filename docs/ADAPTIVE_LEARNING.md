# Adaptive Learning Rates in EMPO

This document describes EMPO's adaptive per-entry learning rate mechanism for lookup tables and discusses options for extending this concept to neural network mode.

## Motivation

In standard stochastic gradient descent, all parameters share the same learning rate. However, in reinforcement learning with function approximation, different state-action pairs may be visited with vastly different frequencies. Parameters corresponding to rarely-visited states receive fewer gradient updates and may not converge to accurate values.

**Key insight**: For lookup tables where each entry corresponds to a specific state (or state-goal pair), we can use **per-entry learning rates** that adapt based on how often each entry has been updated. This achieves exact arithmetic mean convergence for each entry.

## Lookup Table Mode: Per-Entry Adaptive Learning Rate

### Theory: Robbins-Monro Conditions

For stochastic approximation to converge to the true expectation, the learning rate schedule must satisfy the **Robbins-Monro conditions**:

1. $\sum_{n=1}^{\infty} \alpha_n = \infty$ (ensures we can reach any value)
2. $\sum_{n=1}^{\infty} \alpha_n^2 < \infty$ (ensures variance goes to zero)

The schedule $\alpha_n = 1/n$ satisfies both conditions exactly. With this schedule, after $n$ updates, the parameter value equals the **arithmetic mean** of all $n$ target values seen:

$$\theta_n = \theta_{n-1} + \frac{1}{n}(y_n - \theta_{n-1}) = \frac{1}{n}\sum_{i=1}^{n} y_i$$

### Implementation

EMPO implements per-entry adaptive learning rates for lookup tables via **gradient scaling**:

1. **Track update counts**: Each lookup table entry maintains a counter of how many gradient updates it has received.

2. **Scale gradients**: After `backward()` but before `optimizer.step()`, gradients are scaled by `1/update_count` for each entry.

3. **Use base learning rate of 1.0**: The optimizer's learning rate is set to 1.0, so the effective learning rate becomes exactly `1/n`.

This approach works through the optimizer rather than bypassing it, preserving compatibility with momentum-based optimizers like Adam (though the adaptive scaling may interact with Adam's own adaptive learning rates).

### Configuration

```python
from empo.learning_based.phase2.config import Phase2Config

config = Phase2Config(
    use_lookup_tables=True,
    
    # Enable per-entry adaptive learning rate
    lookup_use_adaptive_lr=True,
    
    # Minimum effective learning rate (prevents 1/∞ = 0)
    lookup_adaptive_lr_min=1e-6,
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookup_use_adaptive_lr` | `False` | Enable per-entry adaptive learning rate |
| `lookup_adaptive_lr_min` | `1e-6` | Minimum effective learning rate (floor) |

### API: Lookup Table Networks

All lookup table networks (`LookupTableRobotQNetwork`, `LookupTableHumanGoalAbilityNetwork`, etc.) provide:

| Method | Description |
|--------|-------------|
| `get_update_count(key)` | Get the number of updates for a specific entry |
| `increment_update_counts(keys)` | Increment counters for entries that received gradients |
| `scale_gradients_by_update_count(min_lr)` | Scale gradients by `1/update_count`, returns keys with gradients |
| `zero_grad()` | Zero all gradients in the table |

Update counts are persisted in `state_dict()` and restored by `load_state_dict()`.

### Example: Manual Usage

```python
import torch
from empo.learning_based.phase2.lookup import LookupTableRobotQNetwork

# Create network
q_r = LookupTableRobotQNetwork(num_actions=4, num_robots=1)

# Create optimizer with base lr=1.0 for adaptive mode
optimizer = torch.optim.SGD(q_r.parameters(), lr=1.0)

# Training loop
for batch in data_loader:
    optimizer.zero_grad()
    
    # Forward pass
    output = q_r.forward_batch(states, world_model, device='cpu')
    loss = compute_loss(output, targets)
    
    # Backward pass
    loss.backward()
    
    # Scale gradients by 1/update_count (key step!)
    keys_with_grads = q_r.scale_gradients_by_update_count(min_lr=1e-6)
    q_r.increment_update_counts(keys_with_grads)
    
    # Optimizer step (with base lr=1.0, effective lr = 1/n)
    optimizer.step()
```

### Automatic Integration

When using the standard `Phase2Trainer`, adaptive learning rates are applied automatically if `lookup_use_adaptive_lr=True`. The trainer:

1. Sets base learning rate to 1.0 for lookup table networks
2. Calls `_apply_adaptive_lr_scaling()` after backward passes
3. Increments update counts for entries that received gradients

---

## Neural Network Mode: Uncertainty-Weighted Learning

The per-entry adaptive learning rate concept can be generalized to neural networks by replacing the inverse update count (`1/n`) with an **uncertainty estimate**. Entries with high uncertainty should receive larger updates; entries with low uncertainty should receive smaller updates.

### Design Principle

For lookup tables:
- **Uncertainty metric**: `1/update_count` — fewer visits = more uncertainty
- **Effective learning rate**: `lr_effective = 1/n`

For neural networks:
- **Uncertainty metric**: Model's predictive uncertainty (various methods below)
- **Effective learning rate**: `lr_effective = base_lr * uncertainty_scale`

### Option 1: Ensemble-Based Uncertainty

Train an **ensemble of networks** (e.g., 3-5 copies with different initializations). Uncertainty is estimated from the variance across ensemble predictions.

#### How It Works

```
                    ┌──────────────┐
                    │   State s    │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Network 1  │ │  Network 2  │ │  Network 3  │
    │   Q₁(s,a)   │ │   Q₂(s,a)   │ │   Q₃(s,a)   │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Mean: μ = E[Qᵢ(s,a)]  │
              │  Var:  σ² = Var[Qᵢ]    │
              │  lr ∝ σ² (variance)    │
              └────────────────────────┘
```

#### Uncertainty Scaling

The key insight is that for the sample mean after $n$ observations, the **variance of the estimate** scales as $\sigma^2/n$. Since the optimal learning rate is $1/n$, we have:

$$\text{lr} = \frac{1}{n} \propto \text{Var}[\hat{\mu}]$$

This means the learning rate should be **proportional to variance**, not inversely proportional. High variance = high uncertainty = large learning rate (update aggressively). Low variance = low uncertainty = small learning rate (trust existing estimate).

```python
# Ensemble predictions
predictions = [net(state) for net in ensemble]
mean_pred = torch.stack(predictions).mean(dim=0)
var_pred = torch.stack(predictions).var(dim=0)  # Use variance, not std

# Uncertainty-weighted gradient scaling
# Higher variance = higher uncertainty = larger effective learning rate
# lr ∝ variance (since variance ∝ 1/n and lr = 1/n)
uncertainty_scale = var_pred / (var_pred.mean() + eps)  # Normalize
uncertainty_scale = uncertainty_scale.clamp(min=min_scale, max=max_scale)

# Apply to gradients (conceptually)
for param in network.parameters():
    if param.grad is not None:
        param.grad *= uncertainty_scale
```

#### Pros and Cons

| Pros | Cons |
|------|------|
| Well-understood theoretical basis | 3-5x computational cost |
| Captures epistemic uncertainty | Memory scales with ensemble size |
| Works for any network architecture | Training coordination complexity |
| Uncertainty estimates are calibrated | May need diversity-encouraging regularization |

#### Implementation Sketch

```python
class EnsembleQNetwork(nn.Module):
    def __init__(self, base_network_factory, ensemble_size=5):
        super().__init__()
        self.networks = nn.ModuleList([
            base_network_factory() for _ in range(ensemble_size)
        ])
    
    def forward(self, state):
        predictions = torch.stack([net(state) for net in self.networks])
        return predictions.mean(dim=0), predictions.var(dim=0)  # Return variance
    
    def get_uncertainty(self, state):
        _, var = self.forward(state)
        return var  # lr ∝ variance
```

### Option 2: RND-Based Uncertainty (Leveraging Existing Infrastructure)

EMPO already includes **Random Network Distillation (RND)** for curiosity-driven exploration. The same prediction error that drives exploration could potentially serve as an uncertainty signal for adaptive learning rates.

**Note:** This is a **speculative proposal**, not established practice. We are not aware of published work using RND prediction error specifically for adaptive learning rates. The canonical RND paper ([Burda et al., 2018](https://arxiv.org/abs/1810.12894)) uses RND for intrinsic reward / exploration bonuses, not learning rate adaptation.

#### How It Works

RND consists of:
- **Target network**: Fixed random network (never trained)
- **Predictor network**: Trained to match target's output

The **prediction error** `||predictor(s) - target(s)||²` is high for novel/uncertain states and low for familiar states.

```
                    ┌──────────────┐
                    │   State s    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐      ┌─────────────────┐
    │  Target Network │      │Predictor Network│
    │    (frozen)     │      │   (trainable)   │
    │   φ_target(s)   │      │   φ_pred(s)     │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             └──────────┬─────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │ Error = ||φ_t - φ_p||²  │
              │ (already squared!)      │
              │ High error = uncertain  │
              │ Low error = familiar    │
              │ lr ∝ error (directly)   │
              └─────────────────────────┘
```

#### Proposed Use of RND Error for Learning Rates

The RND prediction error is already a **squared quantity** (MSE or L2 norm squared). If we assume novelty correlates with value estimate uncertainty, then the learning rate should be proportional to RND error directly.

```python
# Get RND prediction error (already computed for curiosity)
# Note: This is ||φ_pred - φ_target||², already squared like variance
rnd_error = self.networks.rnd.compute_novelty(state_features)

# Convert to uncertainty scale
# lr ∝ rnd_error (assuming rnd_error correlates with value uncertainty)
uncertainty_scale = rnd_error / (rnd_error.mean() + eps)  # Normalize
uncertainty_scale = uncertainty_scale.clamp(min=min_scale, max=max_scale)

# Apply to gradient updates
# ... (gradient scaling as above)
```

#### Theoretical Caveats

Unlike ensemble variance or MC Dropout (which have Bayesian interpretations), using RND for learning rate adaptation is **heuristic**:

1. **RND measures novelty, not uncertainty**: A state can be novel (high RND error) but have low value uncertainty (e.g., an obviously bad state). Conversely, a familiar state may still have high value uncertainty due to stochastic outcomes.

2. **No convergence guarantees**: The Robbins-Monro conditions (sum of lr = ∞, sum of lr² < ∞) that guarantee convergence for 1/n schedules don't obviously hold for RND-based scaling.

3. **Empirical validation needed**: This approach would need experimental validation to determine if it improves convergence in practice.

#### Pros and Cons

| Pros | Cons |
|------|------|
| **No additional computation** — RND already exists | **Speculative** — no published validation |
| Single network (no ensemble overhead) | RND novelty ≠ value uncertainty |
| Consistent with exploration objective | No theoretical convergence guarantees |
| Already tracks running mean/std for normalization | Single scalar uncertainty (not per-output) |

#### Implementation Considerations

1. **Correlation assumption**: High RND error indicates the state is novel, which *might* mean the value estimates are uncertain. This correlation is plausible but not proven.

2. **Normalization**: Use the existing RND running mean/std normalization to get stable uncertainty scales.

3. **Per-network vs global**: RND provides a single uncertainty for the state, not per-network. Could either:
   - Apply same uncertainty scale to all network gradients
   - Train separate RND modules per network (expensive)

#### Implementation Status: ✓ IMPLEMENTED

RND-based adaptive learning rate is **implemented** in EMPO. Configuration:

```python
from empo.learning_based.phase2.config import Phase2Config

config = Phase2Config(
    # Enable RND (required for RND-based adaptive LR)
    use_rnd=True,
    
    # Enable RND-based adaptive learning rate
    rnd_use_adaptive_lr=True,
    
    # Scale multiplier for LR adjustment (default: 1.0)
    rnd_adaptive_lr_scale=1.0,
    
    # Minimum LR multiplier (prevents vanishing updates)
    rnd_adaptive_lr_min=0.1,
    
    # Maximum LR multiplier (prevents exploding updates)
    rnd_adaptive_lr_max=10.0,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rnd_use_adaptive_lr` | `False` | Enable RND-based adaptive learning rate |
| `rnd_adaptive_lr_scale` | `1.0` | Multiplier for LR scaling |
| `rnd_adaptive_lr_min` | `0.1` | Minimum LR scale (floor) |
| `rnd_adaptive_lr_max` | `10.0` | Maximum LR scale (ceiling) |

The implementation:
1. Uses the existing robot RND network (state-only input)
2. Computes raw MSE novelty for batch states
3. Normalizes by running mean: `lr_scale = (rnd_mse / running_mean) * scale`
4. Clamps to [min, max] range
5. Scales all neural network gradients by mean batch LR scale

**Usage with demo:**
```bash
# Tabular mode with 1/n adaptive LR
python phase2_robot_policy_demo.py --tabular --adaptive

# Neural mode with RND-based adaptive LR (requires --curious for RND)
python phase2_robot_policy_demo.py --curious --adaptive
```

### Option 3: Dropout-Based Uncertainty (MC Dropout)

Use **Monte Carlo Dropout** to estimate uncertainty by running multiple forward passes with dropout enabled at inference time.

```python
# Enable dropout at inference
network.train()  # Keep dropout active

# Multiple forward passes
predictions = []
for _ in range(n_samples):
    pred = network(state)
    predictions.append(pred)

predictions = torch.stack(predictions)
mean_pred = predictions.mean(dim=0)
var_pred = predictions.var(dim=0)  # Variance = uncertainty, lr ∝ variance
```

#### Theoretical Foundation

[Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142) showed that MC Dropout can be interpreted as approximate variational inference in a Bayesian neural network. Under this interpretation, the variance across dropout samples approximates the **predictive uncertainty** of the posterior:

$$\text{Var}[y^*] \approx \tau^{-1}I + \frac{1}{T}\sum_{t=1}^{T} f(x^*, W_t)^T f(x^*, W_t) - \mathbb{E}[y^*]^T\mathbb{E}[y^*]$$

where $\tau$ is a precision parameter related to dropout rate and weight decay, and $W_t$ are the weights after applying dropout mask $t$.

Key references:
- **Gal (2016)** - [Uncertainty in Deep Learning (PhD thesis)](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf) - Comprehensive treatment of MC Dropout calibration
- **Kendall & Gal (2017)** - [What Uncertainties Do We Need?](https://arxiv.org/abs/1703.04977) - Distinguishes aleatoric vs epistemic uncertainty
- **Lakshminarayanan et al. (2017)** - [Deep Ensembles](https://arxiv.org/abs/1612.01474) - Found ensembles generally better calibrated than MC Dropout

The Bayesian interpretation provides theoretical justification for using MC Dropout variance directly, despite the frequentist correlation between samples. However, empirical studies (Lakshminarayanan et al.) found that **deep ensembles tend to produce better-calibrated uncertainties** in practice.

#### Pros and Cons

| Pros | Cons |
|------|------|
| No additional networks | Requires dropout in architecture |
| Theoretically grounded (Bayesian approx.) | Multiple forward passes (slower) |
| Per-output uncertainty | Ensembles often better calibrated empirically |
| Established literature | Calibration depends on dropout rate, weight decay |

### Comparison of Approaches

| Approach | Compute Cost | Memory | Uncertainty Quality | Implementation Effort |
|----------|--------------|--------|--------------------|-----------------------|
| **Lookup table (1/n)** | O(1) | Per-entry counter | Exact | ✓ Implemented |
| **Ensemble** | 3-5x | 3-5x parameters | High | Medium |
| **RND-based** | ~0 extra | Already allocated | Medium | Low |
| **MC Dropout** | N forward passes | None | Medium | Low |

### Recommended Path Forward

For extending adaptive learning to neural networks, we recommend a **staged approach**:

#### Stage 1: RND-Based (Quick Win)

Since RND is already implemented, use RND prediction error as an uncertainty proxy:

```python
# In Phase2Config
use_uncertainty_weighted_lr: bool = False  # Enable uncertainty-weighted learning
uncertainty_source: str = 'rnd'  # 'rnd', 'ensemble', or 'dropout'
uncertainty_lr_scale_min: float = 0.1  # Minimum LR multiplier
uncertainty_lr_scale_max: float = 2.0  # Maximum LR multiplier
```

This provides immediate benefit with minimal implementation effort.

#### Stage 2: Ensemble (For Maximum Quality)

If uncertainty-weighted learning proves valuable, implement ensemble support for higher-quality uncertainty estimates. This could be optional and enabled via:

```python
ensemble_size: int = 1  # Number of network copies (1 = no ensemble)
```

### Future Work

1. **Theoretical analysis**: Derive optimal uncertainty-to-learning-rate mappings for the EMPO objective.

2. **Empirical comparison**: Compare convergence speed and final performance across uncertainty methods.

3. **Per-network uncertainty**: Investigate whether different networks (Q_r, V_h_e, X_h) benefit from different uncertainty sources.

4. **Active learning integration**: Use uncertainty to prioritize which transitions to sample from the replay buffer.

---

## See Also

- [EXPLORATION.md](EXPLORATION.md) - Curiosity-driven exploration (RND and count-based)
- [WARMUP_DESIGN.md](WARMUP_DESIGN.md) - Phase 2 warm-up and learning rate schedules
- [examples/phase2/lookup_table_phase2_demo.py](../examples/phase2/lookup_table_phase2_demo.py) - Lookup table training example

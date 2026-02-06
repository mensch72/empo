# Learning Rates and Time-Scale Separation in Phase 2

This document discusses the choice of learning rates for Phase 2 training, including the theory of time-scale separation and practical guidance for the EMPO framework.

## The Mutual Dependency Challenge

Phase 2 involves computing several interdependent quantities:

```
Q_r → π_r → V_h^e → X_h → U_r → V_r → Q_r
       ↘________________________↗
```

The key dependencies:
- **Q_r** (robot Q-values) determines **π_r** (robot policy) via power-law softmax
- **V_h^e** (human goal achievement) depends on **π_r** — how the robot acts affects human success
- **X_h** (human power) aggregates V_h^e across goals
- **U_r** (robot intrinsic reward) depends on X_h — the robot's reward is human empowerment
- **V_r** (robot state value) combines U_r and Q_r
- **Q_r** depends on V_r via the Bellman equation

This creates a **circular dependency**: Q_r needs U_r to compute correct targets, but U_r needs Q_r (via π_r) to compute correct values.

## Time-Scale Separation: The Classic Approach

### Actor-Critic Analogy

The standard solution in reinforcement learning is **two-timescale stochastic approximation**:

| Component | EMPO | Actor-Critic |
|-----------|------|--------------|
| **Evaluator** | V_h^e, X_h, U_r | Critic (V or Q) |
| **Optimizer** | Q_r, π_r | Actor (π) |

The classic recommendation: **evaluator fast, optimizer slow**.

### Why This Ordering?

The intuition:
1. The evaluator's job is to accurately assess the current policy
2. If the policy changes slowly, the evaluator can track it accurately
3. The optimizer then receives accurate feedback and can improve reliably

With the reverse ordering (optimizer fast):
1. The optimizer quickly converges to "optimal" for the current (wrong) evaluation
2. The evaluator can't keep up with the rapidly changing policy
3. Everyone ends up chasing moving targets

### Formal Justification

From two-timescale stochastic approximation theory (Borkar 2008):

> For coupled iterations with learning rates α_fast >> α_slow, the system converges to a **nested fixed point** where the fast variable is at equilibrium for the slow variable's current value.

In EMPO terms: if U_r pathway is fast and Q_r is slow, then at any moment U_r accurately reflects "human power under the current π_r", giving Q_r reliable targets.

## The Non-Contraction Problem

### Standard Bellman Is Contractive

The standard Bellman operator is a γ-contraction:
```
||T(Q_1) - T(Q_2)|| ≤ γ ||Q_1 - Q_2||
```

This guarantees convergence for γ < 1.

### Power-Law Policy Breaks Contraction

EMPO uses a power-law policy:
```
π_r(a|s) ∝ (-Q_r(s,a))^{-β_r}
```

The sensitivity of π_r to Q_r depends on **β_r**:
- **Small β_r** (soft policy): Small Q_r changes → small π_r changes
- **Large β_r** (sharp policy): Small Q_r changes → potentially large π_r changes

For large β_r (e.g., β_r = 10), the mapping Q_r → π_r → V_h^e → U_r → Q_r can have **Lipschitz constant > 1**, meaning the coupled system is **not a contraction**.

### Implications

| β_r Value | System Behavior |
|-----------|-----------------|
| Small (≈ 0) | Nearly contractive, likely converges with equal learning rates |
| Moderate (1-5) | Mildly non-contractive, benefits from time-scale separation |
| Large (≥ 10) | Strongly non-contractive, requires careful handling |

This is why **β_r ramp-up** during warmup is essential: starting with β_r = 0 ensures contractiveness, then gradually increasing β_r keeps the dynamics manageable.

## Current Approach: Staged Warmup

The EMPO implementation uses **staged warmup** as a discrete form of time-scale separation:

1. **Stage 0**: Train V_h^e only (with β_r = 0, uniform random robot)
2. **Stage 1**: Add X_h training
3. **Stage 2**: Add U_r training (if using U_r network)
4. **Stage 3**: Add Q_r training (β_r still 0)
5. **Stage 4**: Add V_r training (if using V_r network)
6. **Stage 5**: Ramp β_r from 0 to nominal value
7. **Stage 6**: Full training with learning rate decay

This approach:
- Completely separates timescales initially (U_r pathway converges before Q_r trains)
- Gradually couples the systems via β_r ramp-up
- Avoids the non-contraction issue by starting with β_r = 0

See [WARMUP_DESIGN.md](WARMUP_DESIGN.md) for detailed stage descriptions.

## Learning Rate Configuration

### Available Parameters

```python
@dataclass
class Phase2Config:
    lr_v_h_e: float = 1e-4   # V_h^e learning rate
    lr_x_h: float = 1e-4     # X_h learning rate  
    lr_u_r: float = 1e-4     # U_r learning rate
    lr_q_r: float = 1e-4     # Q_r learning rate
    lr_v_r: float = 1e-4     # V_r learning rate
```

### Time-Scale Separation via Learning Rates

To implement continuous time-scale separation (in addition to staged warmup):

```python
# Evaluator fast, optimizer slow (10:1 ratio)
config = Phase2Config(
    lr_v_h_e=1e-3,   # Fast
    lr_x_h=1e-3,     # Fast
    lr_u_r=1e-3,     # Fast
    lr_q_r=1e-4,     # Slow (10x smaller)
    lr_v_r=1e-4,     # Slow
)
```

### When to Use Different Learning Rates

| Situation | Recommendation |
|-----------|----------------|
| **Small β_r** (≤ 1) | Equal learning rates likely fine |
| **Large β_r** (≥ 5) | Consider U_r-fast, Q_r-slow |
| **Oscillating losses** | Slow down Q_r relative to U_r |
| **Slow convergence** | May need to speed up both (keeping ratio) |
| **V_h^e unstable** | Slow down Q_r to stabilize π_r |

### Alternative: Update Frequency Separation

Instead of different learning rates, use different update frequencies:

```python
# Multiple U_r pathway updates per Q_r update
for step in range(num_steps):
    for _ in range(5):  # 5 U_r updates
        update_v_h_e()
        update_x_h()
        update_u_r()
    update_q_r()  # 1 Q_r update
```

This achieves the same effect as a 5:1 learning rate ratio.

## When Equal Learning Rates May Work

Recent literature (particularly from the game theory and GAN optimization communities) suggests that equal learning rates can work in some settings:

### Conditions Favoring Equal Rates

1. **Small β_r**: System is approximately contractive
2. **After warmup**: Networks already near equilibrium
3. **With momentum/Adam**: Optimizer adapts effective learning rates
4. **With target networks**: Frozen targets provide stability

### EMPO Is Not Adversarial

Unlike GANs (zero-sum games with rotational dynamics), EMPO's networks are **cooperatively** trying to reach a consistent fixed point. This makes equal learning rates more viable than in adversarial settings.

### Empirical Testing Recommended

The theory provides guidance, but **empirical testing** for your specific environment is valuable:

```python
# Experiment 1: Equal rates (baseline)
config_equal = Phase2Config(lr_v_h_e=1e-4, lr_x_h=1e-4, lr_u_r=1e-4, lr_q_r=1e-4)

# Experiment 2: U_r pathway faster
config_fast_ur = Phase2Config(lr_v_h_e=1e-3, lr_x_h=1e-3, lr_u_r=1e-3, lr_q_r=1e-4)

# Experiment 3: Q_r faster (not recommended, but worth testing)
config_fast_qr = Phase2Config(lr_v_h_e=1e-4, lr_x_h=1e-4, lr_u_r=1e-4, lr_q_r=1e-3)
```

## Practical Recommendations

### Default Configuration

For most cases, start with the defaults and staged warmup:

```python
config = Phase2Config(
    # Equal learning rates (warmup handles separation)
    lr_v_h_e=1e-4,
    lr_x_h=1e-4,
    lr_u_r=1e-4,
    lr_q_r=1e-4,
    lr_v_r=1e-4,
    
    # Staged warmup (essential)
    warmup_v_h_e_steps=1000,
    warmup_x_h_steps=1000,
    warmup_q_r_steps=1000,
    
    # β_r ramp-up (essential for large β_r)
    beta_r=10.0,
    beta_r_rampup_steps=2000,
)
```

### If Training Is Unstable

1. **First**: Ensure warmup stages are long enough
2. **Second**: Ensure β_r ramp-up is gradual enough
3. **Third**: Try making Q_r learning rate smaller (e.g., 5-10x)
4. **Fourth**: Increase target network update intervals for Q_r

### If Training Is Too Slow

1. **First**: Check if learning rates are too small overall
2. **Second**: Try increasing U_r pathway learning rates (while keeping Q_r the same)
3. **Third**: Consider reducing warmup durations (if already stable)

### Monitoring for Time-Scale Issues

Watch for these signs in TensorBoard:

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Q_r loss oscillates | Q_r changing too fast | Reduce lr_q_r |
| U_r lags behind Q_r | U_r pathway too slow | Increase lr_u_r pathway |
| V_h^e unstable | π_r changing rapidly | Reduce lr_q_r |
| All losses plateau early | Learning rates too small | Increase all proportionally |

## Theoretical References

For deeper understanding:

1. **Borkar (2008)** *Stochastic Approximation: A Dynamical Systems Viewpoint* — Chapter 6 covers two-timescale theory

2. **Konda & Tsitsiklis (2000)** "Actor-Critic Algorithms" — Original actor-critic two-timescale analysis

3. **Heusel et al. (2017)** "GANs Trained by a Two Time-Scale Update Rule" — Application to GANs (different setting but relevant insights)

4. **Gidel et al. (2019)** "A Variational Inequality Perspective on GANs" — When equal rates can work

5. **Mescheder et al. (2018)** "Which Training Methods for GANs do actually Converge?" — Practical comparison of approaches

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Primary mechanism** | Staged warmup + β_r ramp-up |
| **Learning rates** | Start equal; adjust if unstable |
| **If using separation** | U_r pathway fast, Q_r slow |
| **Key insight** | Non-contraction from large β_r requires careful handling |
| **Empirical testing** | Always valuable for your specific environment |

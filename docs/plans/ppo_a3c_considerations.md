# Considerations for Using PPO or A3C Ideas in Phase 2

**Status:** Speculative / Discussion  
**Author:** GitHub Copilot  
**Date:** 2025-12-26

## 1. Overview

This document explores the potential advantages and disadvantages of incorporating ideas from **Proximal Policy Optimization (PPO)** or **Asynchronous Advantage Actor-Critic (A3C)** into Phase 2 of the EMPO framework. Currently, Phase 2 uses an approach more similar to **DQN** (Deep Q-Network): the robot policy π_r is directly derived from Q-value predictions via a power-law softmax (equation 5).

### Current Approach (DQN-like)

```
(4) Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]
(5) π_r(s)(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
```

The policy is implicitly defined by the Q-values—there's no separate policy network. This is conceptually similar to how DQN derives a greedy policy from Q-values, though here we use a power-law softmax instead of argmax.

---

## 2. PPO Approach: Potential Integration

### 2.1 What PPO Offers

PPO (Proximal Policy Optimization) maintains an **explicit policy network** separate from the value function. Key features:

1. **Clipped surrogate objective**: Limits policy updates to stay within a "trust region"
2. **Multiple epochs of updates**: Reuses collected data for several gradient steps
3. **Advantage estimation (GAE)**: Reduces variance while maintaining acceptable bias

### 2.2 Potential Benefits for EMPO Phase 2

| Benefit | Explanation |
|---------|-------------|
| **Direct policy optimization** | Instead of deriving π_r from Q_r, directly optimize π_r to maximize expected aggregate human power. This could yield more targeted policy improvements. |
| **Stability via clipping** | PPO's clipped objective prevents destructive large policy updates. Currently, sudden changes in Q_r can cause large swings in the implied policy. |
| **Sample efficiency** | Multiple epochs per batch could improve sample efficiency compared to the current single-update-per-transition approach. |
| **Natural entropy regularization** | PPO often adds an entropy bonus, which could replace or complement the power-law β_r for exploration. |
| **Gradient signal directly to policy** | Avoids the indirection of Q → policy, potentially leading to faster learning of nuanced behaviors. |

### 2.3 Potential Drawbacks for EMPO Phase 2

| Drawback | Explanation |
|----------|-------------|
| **Loss of scale-invariance** | The power-law form `(-Q)^{-β}` satisfies certain scale-invariance properties (see paper's Table 2). A standard policy network with softmax wouldn't preserve this without careful design. |
| **Additional network complexity** | Requires a separate policy network π_r alongside Q_r, V_r, V_h^e, X_h, U_r—adding to an already complex system. |
| **Hyperparameter sensitivity** | PPO introduces additional hyperparameters (clip ratio ε, GAE λ, number of epochs) that would need tuning alongside existing Phase 2 parameters. |
| **Mutual dependencies harder to manage** | V_h^e depends on π_r, which creates a circular dependency. With an explicit policy network, this feedback loop might become harder to stabilize. |
| **On-policy data requirements** | PPO is on-policy; replay buffer usage becomes problematic. The current replay buffer approach would need significant modification. |
| **Theoretical alignment** | The paper derives π_r from Q_r for principled reasons related to the power metric formulation. Switching to PPO might diverge from the theoretical framework. |

### 2.4 Implementation Sketch (if pursued)

If PPO ideas were integrated:

```python
# Explicit policy network
class RobotPolicyNetwork(nn.Module):
    def forward(self, state) -> Distribution:
        # Returns action distribution, not Q-values
        ...

# PPO-style update
def ppo_update(policy, old_policy, advantages, clip_ratio=0.2):
    ratio = policy.prob(action) / old_policy.prob(action)
    clipped = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    ...
```

The advantage function would need to be computed using the intrinsic reward U_r and value V_r.

---

## 3. A3C Approach: Potential Integration

### 3.1 What A3C Offers

A3C (Asynchronous Advantage Actor-Critic) features:

1. **Actor-Critic architecture**: Separate policy (actor) and value (critic) networks
2. **Asynchronous parallel training**: Multiple workers exploring simultaneously
3. **N-step returns**: Balances bias-variance trade-off in value estimation

### 3.2 Potential Benefits for EMPO Phase 2

| Benefit | Explanation |
|---------|-------------|
| **Parallelized data collection** | Multiple environment instances could speed up training, especially valuable for complex multi-agent simulations. |
| **Reduced correlation in updates** | Asynchronous updates from different workers naturally decorrelate data, potentially replacing/complementing the replay buffer. |
| **Actor-Critic synergy** | The critic (value network) is already present as V_r. Adding an explicit actor could improve learning dynamics. |
| **N-step returns** | Could improve V_h^e and V_r estimation by balancing immediate reward signal with long-term bootstrapping. |

### 3.3 Potential Drawbacks for EMPO Phase 2

| Drawback | Explanation |
|----------|-------------|
| **Complexity of asynchronous training** | Managing multiple workers with shared networks is significantly more complex than the current single-threaded approach. |
| **Gradient staleness** | Asynchronous updates can use stale gradients, potentially destabilizing training of the interdependent network system. |
| **Incompatible with replay buffer** | A3C is on-policy and doesn't use a replay buffer. The current Phase2ReplayBuffer architecture would need to be replaced. |
| **Hardware requirements** | A3C's benefits come from parallelism, requiring more computational resources. |
| **Same scale-invariance concerns** | Like PPO, would lose the power-law policy properties unless carefully designed. |
| **Coordination overhead** | In EMPO, networks are interdependent (Q_r ↔ π_r ↔ V_h^e ↔ X_h ↔ U_r ↔ V_r). Asynchronous updates could exacerbate instability in this chain. |

---

## 4. Hybrid Approaches

Rather than wholesale adoption of PPO or A3C, selective borrowing might be more practical:

### 4.1 GAE for Advantage Estimation

Use Generalized Advantage Estimation when computing TD targets for V_r and V_h^e:

```
Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

**Pro**: Reduces variance in value estimates without changing the policy derivation.  
**Con**: Requires trajectory-based computation, complicating the current transition-based approach.

### 4.2 Soft Clipping on Q-value Updates

Apply a trust-region constraint on how much Q_r can change per update:

```python
q_change = q_r_new - q_r_old
q_r_update = q_r_old + torch.clamp(q_change, -max_change, max_change)
```

**Pro**: Prevents large policy swings without adding a separate policy network.  
**Con**: Adds another hyperparameter; may slow convergence.

### 4.3 Parallel Environment Rollouts (without A3C)

Use multiple environment copies for faster data collection but update synchronously:

```python
# Vectorized environment
envs = VectorEnv([make_env() for _ in range(num_workers)])
transitions = envs.step(robot_actions)  # Parallel step
```

**Pro**: Gets parallelism benefits without asynchronous gradient complexity.  
**Con**: Requires environment vectorization support.

### 4.4 Entropy Regularization on Implied Policy

Add entropy bonus to the Q_r loss to encourage exploration:

```python
pi_r = networks.q_r.get_policy(q_values)  # Power-law softmax
entropy = -(pi_r * torch.log(pi_r + 1e-8)).sum()
loss = td_loss - entropy_coef * entropy
```

**Pro**: Encourages exploration without changing policy derivation.  
**Con**: Interacts with β_r in potentially complex ways.

---

## 5. Key Considerations

### 5.1 Theoretical vs. Practical

The current approach is **theoretically motivated**—the power-law policy derivation from Q_r satisfies scale-invariance properties important for the "empowerment" interpretation. PPO/A3C are **empirically motivated**—they work well in practice but may not preserve these theoretical guarantees.

**Question**: Is the scale-invariance property essential to EMPO's goals, or is it a nice-to-have that could be sacrificed for better learning dynamics?

### 5.2 System Complexity

Phase 2 already involves 5+ interacting networks:
- Q_r (robot Q-values)
- V_r (robot value) 
- V_h^e (human goal achievement under robot policy)
- X_h (aggregate goal ability)
- U_r (intrinsic robot reward)

Adding an explicit policy network increases this to 6+, with associated:
- Additional optimizer
- Additional hyperparameters
- Additional target network (possibly)
- Additional loss function

### 5.3 The Mutual Dependency Problem

The core challenge is that V_h^e depends on π_r, which depends on Q_r:

```
V_h^e(s, g_h) ← E_{...a_r ~ π_r(s)...} [U_h(s', g_h) + γ_h V_h^e(s', g_h)]
π_r(s) ∝ (-Q_r(s, ·))^{-β_r}
Q_r(s, a_r) ← E_{...} [γ_r V_r(s')]
V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)
```

This circular dependency is manageable with the current implicit policy because π_r is a deterministic function of Q_r. With an explicit policy network, the dependency becomes:

```
V_h^e depends on π_r
π_r is updated to maximize something involving V_r
V_r depends on U_r, Q_r, and π_r
U_r depends on X_h
X_h depends on V_h^e
```

This could make training less stable if the networks move in inconsistent directions.

---

## 6. Recommendations

### 6.1 Short-term: Keep Current Approach

The DQN-like approach is:
- Theoretically grounded
- Simpler to debug
- Sufficient for initial experimentation

### 6.2 Medium-term: Consider Hybrid Extensions

If training instability or sample inefficiency becomes a problem:

1. **Try GAE** for V_r/V_h^e targets (low-risk modification)
2. **Try entropy regularization** on the implied policy (preserves structure)
3. **Try soft clipping** on Q_r updates (trust-region spirit without policy network)

### 6.3 Long-term: Evaluate PPO-style if Needed

If the above don't suffice:

1. Design a policy network that preserves power-law scale-invariance
2. Carefully manage the mutual dependency through frozen target policies
3. Extensive hyperparameter search would be required

---

## 7. Open Questions

1. **Can scale-invariance be preserved with an explicit policy network?** Perhaps by parameterizing the policy as `π(a|s) ∝ f(s,a)^{-β}` where f is learned?

2. **Would importance sampling allow off-policy PPO?** This might enable replay buffer usage while getting PPO's benefits.

3. **Is the mutual dependency the main source of training instability?** If not, PPO/A3C might not help.

4. **What's the sample complexity of the current approach?** If it's already reasonable, the complexity of PPO/A3C may not be worthwhile.

5. **Could V-trace (from IMPALA) provide a middle ground?** Off-policy actor-critic with correction might combine benefits of both worlds.

---

## 8. References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **A3C**: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)
- **IMPALA/V-trace**: Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (2018)
- **EMPO Paper**: See equations (4)-(9) in Table 1 for the current Phase 2 formulation

# Curiosity-Driven Exploration for Phase 2

## Problem Statement

Current Phase 2 training uses epsilon-greedy exploration for both robot and human agents. Even with high exploration rates (`epsilon_r = epsilon_h = 1.0`), empirical testing shows only ~75% of states are visited after 250,000 training steps in trivial environments. This is because random exploration follows a Brownian motion pattern with range ∝ √steps, which is insufficient for thorough state space coverage.

Curiosity-driven exploration addresses this by biasing action selection toward novel states, enabling more systematic coverage of the state space.

---

## Approaches Considered

### Count-Based Curiosity (Tabular Only)

**Mechanism:** Maintain a dictionary `Dict[Hashable, int]` mapping states to visit counts. Compute bonus as:

```
bonus(s) = scale / sqrt(visits[s] + 1)
```

Or UCB-style:
```
bonus(s) = scale * sqrt(log(total_visits) / (visits[s] + 1))
```

**Pros:**
- Simple implementation
- No additional neural networks
- Works well with EMPO's hashable state representation

**Cons:**
- **Cannot generalize to unseen states**—inherently tabular
- Memory grows linearly with unique states visited
- Not suitable for neural network training where generalization is needed

**Verdict:** Suitable for lookup-table-based Phase 2 only. Not recommended for neural training.

### ICM (Intrinsic Curiosity Module)

**What problem ICM solves:** Define novelty in a **compressed feature space** that captures only action-relevant state aspects. This filters out:
- Uncontrollable stochastic elements (e.g., random TV static, wind)
- State details that don't affect transition outcomes
- High-dimensional observations where only a subset matters

**Mechanism:** Train two models:
1. **Inverse model:** Predicts action from (state, next_state) pair → Forces the embedding to capture action-relevant features
2. **Forward model:** Predicts next state *embedding* from (state embedding, action) → Prediction error = novelty in action-relevant space

The inverse model shapes the embedding: if a state feature doesn't help predict which action was taken, it gets filtered out.

**Potential fit for EMPO:**

Even though EMPO has exact `transition_probabilities()`, the raw state representation may be overly detailed. Consider:
- A 10×10 MultiGrid with multiple objects → large state tuple
- Many state aspects may not affect goal achievement or action outcomes
- Novelty in raw state space may not correlate with meaningful exploration

ICM could learn a compressed representation where:
- States that lead to similar transitions get similar embeddings
- Novelty is measured in this action-relevant space
- Exploration focuses on states that differ in *how they respond to actions*

**Trade-offs for EMPO:**

| Aspect | Pro | Con |
|--------|-----|-----|
| Compression | Filters irrelevant state details | Adds complexity vs RND |
| Forward model | Could leverage known transitions as supervision | Redundant with `transition_probabilities()` |
| Inverse model | Learns action-relevant features | Joint robot+human actions complicate prediction |
| Training | Self-supervised, no labels needed | More networks, more hyperparameters |

**Hybrid possibility:** Use ICM's inverse model to learn action-relevant embeddings, but skip the forward model (since we have exact dynamics). Measure novelty via RND on the inverse-model embeddings.

**Verdict:** More complex than RND, but potentially valuable if state spaces are high-dimensional and contain action-irrelevant details. Consider as a Phase 2 enhancement after RND baseline is established.

### NGU / Agent57

**Mechanism:** Combines episodic novelty (within-episode state similarity) with lifelong novelty (across all training). Uses sophisticated memory and embedding networks.

**Verdict:** Overkill for MultiGrid-scale environments. Not recommended.

### RND (Random Network Distillation) — **Recommended**

**Mechanism:** Use prediction error of a trainable network trying to match a fixed random network's output as a novelty signal.

**Pros:**
- Simple architecture (just two MLPs)
- No dynamics modeling required
- Reuses existing encoder infrastructure
- Prediction error naturally decreases for frequently-seen states
- Generalizes novelty across similar states

**Cons:**
- Requires tuning bonus coefficient
- May need running normalization for stability

**Verdict:** Best fit for EMPO's neural Phase 2 training.

---

## RND Theory

### Core Insight

Random Network Distillation (Burda et al., 2018) exploits a key observation: **a neural network can easily predict the output of a fixed random function for inputs it has seen many times, but struggles for novel inputs.**

### Architecture

```
                    ┌─────────────────────────┐
                    │    State Encoder        │
                    │  (shared, detached)     │
                    └───────────┬─────────────┘
                                │
                          state_features
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
    ┌─────────────────────┐             ┌─────────────────────┐
    │   Target Network    │             │  Predictor Network  │
    │   (FROZEN RANDOM)   │             │    (TRAINABLE)      │
    │                     │             │                     │
    │  f_target(φ(s))     │             │  f_pred(φ(s))       │
    └──────────┬──────────┘             └──────────┬──────────┘
               │                                   │
               │         MSE Loss                  │
               └────────────►◄─────────────────────┘
                              ║
                              ▼
                      novelty(s) = ||f_target(φ(s)) - f_pred(φ(s))||²
```

### Why It Works

1. **Target network outputs are deterministic but arbitrary:** For any state embedding, the target network produces a fixed random output. This output has no semantic meaning—it's just a consistent fingerprint.

2. **Predictor learns to memorize:** The predictor network is trained to match the target's output. For frequently-visited states, it learns to produce the correct fingerprint accurately.

3. **Prediction error indicates novelty:** 
   - **High error** → State rarely seen → Predictor hasn't learned this fingerprint → **Novel**
   - **Low error** → State frequently seen → Predictor has memorized this fingerprint → **Familiar**

4. **Generalization through embeddings:** Unlike count-based methods, RND operates on continuous embeddings. Similar states have similar embeddings, so novelty generalizes:
   - If state A is novel and state B is similar to A, then B is also likely novel
   - This enables meaningful exploration in large/continuous state spaces

### Mathematical Formulation

Let:
- $\phi(s)$ = state encoder output (e.g., 256-dim vector from `MultiGridStateEncoder`)
- $f_\theta(x)$ = target network with frozen random weights $\theta$
- $\hat{f}_\psi(x)$ = predictor network with trainable weights $\psi$

**Novelty score:**
$$
\text{novelty}(s) = \| f_\theta(\phi(s)) - \hat{f}_\psi(\phi(s)) \|^2
$$

**RND loss (to train predictor):**
$$
\mathcal{L}_\text{RND} = \mathbb{E}_{s \sim \mathcal{D}} \left[ \| f_\theta(\phi(s)) - \hat{f}_\psi(\phi(s)) \|^2 \right]
$$

**Exploration bonus for action selection:**
$$
Q_\text{explore}(s, a) = Q_r(s, a) + \beta_\text{curiosity} \cdot \text{novelty}(s')
$$

where $s'$ is the expected next state after taking action $a$.

### Running Normalization

RND prediction errors can vary widely in scale during training. To stabilize the bonus:

$$
\text{normalized\_novelty}(s) = \frac{\text{novelty}(s) - \mu_\text{running}}{\sigma_\text{running} + \epsilon}
$$

where $\mu_\text{running}$ and $\sigma_\text{running}$ are exponential moving averages of novelty scores.

### Why RND Over Other Methods

| Property | RND | ICM | Count-Based |
|----------|-----|-----|-------------|
| Generalizes to similar states | ✅ | ✅ | ❌ |
| No dynamics model needed | ✅ | ❌ | ✅ |
| Simple architecture | ✅ | ❌ | ✅ |
| Works with any encoder | ✅ | ⚠️ | N/A |
| Stable training | ✅ | ⚠️ | ✅ |

---

## Implementation Plan

### Phase 1: Configuration and Module

1. **Create `src/empo/nn_based/phase2/rnd.py`** (new file)
   
   Implement `RNDModule` class:
   - Frozen random target network (2-layer MLP)
   - Trainable predictor network (3-layer MLP)
   - `compute_novelty(state_features) -> Tensor` method
   - `compute_loss(state_features) -> Tensor` method
   - Optional: running mean/std tracking for normalization

2. **Extend `src/empo/nn_based/phase2/config.py`**
   
   Add configuration parameters:
   ```python
   # Curiosity-driven exploration (RND)
   use_rnd: bool = False
   rnd_feature_dim: int = 64          # Output dim of RND networks
   rnd_bonus_coef_r: float = 0.1      # Robot curiosity bonus scale
   rnd_bonus_coef_h: float = 0.1      # Human curiosity bonus scale
   lr_rnd: float = 1e-4               # RND predictor learning rate
   normalize_rnd: bool = True         # Use running normalization
   rnd_normalization_decay: float = 0.99  # EMA decay for normalization
   ```

### Phase 2: Network Integration

3. **Modify `src/empo/nn_based/phase2/networks.py`**
   
   Add optional RND module to `Phase2Networks` dataclass:
   ```python
   rnd: Optional[RNDModule] = None
   ```

4. **Modify `src/empo/nn_based/multigrid/phase2/networks.py`**
   
   Update `create_multigrid_phase2_networks()` to instantiate RND module when `config.use_rnd=True`, using shared encoder's `feature_dim` as input dimension.

### Phase 3: Training Integration

5. **Modify `src/empo/nn_based/phase2/trainer.py`**
   
   Key integration points:
   - `_init_optimizers()` (~L306): Add RND optimizer
   - `_compute_all_losses()` (~L918): Add RND loss computation
   - `_training_step()`: Include RND in backward pass loop

### Phase 4: Exploration Integration

6. **Modify action sampling in trainer**
   
   - `sample_robot_action()` (~L1500): When exploring, compute novelty bonus for successor states and bias selection toward high-novelty actions
   - `sample_human_action()` (~L1550): Same for human agent with separate bonus coefficient

### Phase 5: Documentation

7. **Update `docs/EXPLORATION.md`**
   
   Document:
   - RND curiosity exploration mechanism
   - Configuration options and recommended values
   - Interaction with epsilon-greedy (curiosity replaces/augments random exploration)
   - Guidance on tuning bonus coefficients

---

## Alternative: Random State Generation via World Model

### Leveraging EMPO's Unique World Model Interface

Unlike typical RL environments (which only support `reset()` to initial states), EMPO's `WorldModel` interface provides:

```python
def get_state(self) -> Hashable:     # Complete, hashable state
def set_state(self, state) -> None:  # Restore to ANY valid state
```

This capability enables a fundamentally different exploration strategy: **directly generating random valid states** rather than discovering them through trajectories.

### State Space Sweeping

**Core idea:** Instead of relying on policy-driven exploration to visit novel states, directly sample states from the valid state space and train on transitions starting from those states.

```
Traditional RL Exploration:
    s_0 → s_1 → s_2 → ... → s_T
    (Must traverse trajectory to reach s_T)

State Space Sweeping:
    Generate random valid state s_rand
    world_model.set_state(s_rand)
    Collect transition (s_rand, a, s')
    (Direct access to any state)
```

### Distinction from Length-1 Episode Ensemble

**Important subtlety:** Training on randomly generated states is NOT equivalent to training an ensemble of world models with episode length 1.

| Approach | What's Learned | Episode Structure |
|----------|----------------|-------------------|
| Length-1 episode ensemble | $V(s)$ for 1-step horizon | Terminal after 1 step |
| Random state generation | $V(s)$ for full horizon | Normal episode continues |

With random state generation:
- We sample a starting state $s$ uniformly (or from some coverage distribution)
- We then run a normal episode from $s$ with full horizon
- Value functions $Q_r(s,a)$, $V_h^e(s,g)$ etc. still estimate multi-step returns
- Only the **initial state distribution** changes, not the **return definition**

This is analogous to the difference between:
- **Curriculum learning** (changing where episodes start)
- **Reward shaping** (changing what episodes optimize)

### Implementation Approaches

#### Approach A: Random Valid State Sampling

For discrete/structured environments like MultiGrid, enumerate or sample valid states:

```python
def sample_random_valid_state(world_model: WorldModel) -> Hashable:
    """Generate a random valid state by randomizing world model components."""
    # Example for MultiGrid:
    # - Random agent positions (not overlapping, not in walls)
    # - Random object positions (keys, doors, etc.)
    # - Random door states (open/closed)
    # - Random agent inventories
    ...
    return world_model.get_state()
```

**Challenge:** Defining "valid" states. Not all combinations are reachable or meaningful.

#### Approach B: State Archive with Perturbation (Go-Explore style)

Maintain an archive of discovered states; periodically reset to archived states and perturb:

```python
state_archive: Set[Hashable] = set()

def explore_from_archive():
    # Select a state from archive (uniform or weighted by novelty)
    base_state = sample_from_archive(state_archive)
    world_model.set_state(base_state)
    
    # Take random actions to perturb
    for _ in range(perturbation_steps):
        action = sample_random_action()
        world_model.step(action)
    
    # Continue normal episode from perturbed state
    return collect_episode(world_model)
```

**Advantage:** Only explores states reachable from known states (avoids impossible configurations).

#### Approach C: Backward Reachability Sampling

Start from goal states and work backward:

```python
def sample_state_backward_from_goal(goal_state, steps_back):
    """Sample states that can reach the goal in ~steps_back actions."""
    world_model.set_state(goal_state)
    
    # Use inverse dynamics or rejection sampling
    for _ in range(steps_back):
        # Find predecessor states
        predecessor = find_predecessor_state(world_model)
        world_model.set_state(predecessor)
    
    return world_model.get_state()
```

**Note:** Requires invertible or enumerable transitions.

### Hybrid Strategy: Curiosity + State Sweeping

RND curiosity and state space sweeping are **complementary**:

| Aspect | RND Curiosity | State Sweeping |
|--------|---------------|----------------|
| Mechanism | Bias toward novel states | Direct access to states |
| State validity | Guaranteed (reached via transitions) | Must ensure validity |
| Coverage | Gradual, trajectory-dependent | Potentially uniform |
| Value learning | On-policy-ish | Off-policy (arbitrary starts) |

**Recommended hybrid:**
1. Use RND to guide exploration during normal episodes
2. Periodically inject archived states as episode starting points
3. Weight archived states by RND novelty for maximum coverage

### Theoretical Considerations

#### Sample Complexity

Random state generation can dramatically improve sample complexity for state-coverage-dependent objectives. If the goal is to learn accurate $V_h^e(s, g)$ for all $(s, g)$ pairs:

- **Trajectory-only exploration:** Coverage ∝ $\sqrt{T}$ (random walk)
- **With state sweeping:** Coverage ∝ $T$ (direct sampling)

#### Distributional Shift

Training on uniformly sampled states creates distributional shift from deployment:
- Training distribution: uniform over state space
- Deployment distribution: states reachable from initial states under learned policy

This may be acceptable for EMPO since we want accurate value estimates everywhere (for computing optimal policies), not just on-policy.

#### Relationship to Offline RL

Random state generation resembles offline RL's challenge: learning from a fixed dataset that may not match the deployment distribution. Techniques from offline RL (conservative Q-learning, pessimistic value estimation) may be relevant if state sweeping creates coverage gaps.

### Relevant Literature

#### Go-Explore (Ecoffet et al., 2019, 2021)

Explicitly archives interesting states and resets to them for continued exploration. Key innovations:
- Cell-based state abstraction for archive management
- "Robustification" phase to learn robust policies to archived states
- Achieved superhuman performance on Montezuma's Revenge

**Relevance to EMPO:** EMPO's `set_state()` provides the reset capability Go-Explore requires. The archive + reset pattern could be directly applied.

> Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J. (2021). *First return, then explore*. Nature, 590(7847), 580-586.

#### Backplay (Resnick et al., 2018)

Starts episodes from states encountered late in demonstrations, gradually moving starting points earlier. Creates natural curriculum from easy (near-goal) to hard (far-from-goal).

**Relevance to EMPO:** Could start Phase 2 training from states near human goal achievement, expanding coverage backward.

> Resnick, C., Eldridge, W., Ha, D., Brber, D., Eslami, S., Rezende, D., ... & Higgins, I. (2018). *Backplay: "Man muss immer umkehren"*. arXiv:1807.06919.

#### Maximum Entropy State Marginal Matching (Lee et al., 2019)

Learns policies that induce uniform state marginal distributions, maximizing state space coverage.

$$\max_\pi H(d^\pi(s)) \quad \text{where } d^\pi(s) = \mathbb{E}_\pi\left[\sum_t \gamma^t \mathbf{1}[s_t = s]\right]$$

**Relevance to EMPO:** Provides theoretical grounding for why uniform state coverage is desirable for learning accurate value functions.

> Lee, L., Eysenbach, B., Parisotto, E., Xing, E., Levine, S., & Salakhutdinov, R. (2019). *Efficient exploration via state marginal matching*. arXiv:1906.05274.

#### Skew-Fit (Pong et al., 2019)

Learns goal-conditioned policies by training on goals sampled from a skewed (diverse) distribution rather than uniform.

**Relevance to EMPO:** Phase 2 already conditions on goals. Skew-Fit's insight about goal distribution shaping could apply to $V_h^e$ training.

> Pong, V. H., Dalal, M., Lin, S., Nair, A., Bahl, S., & Levine, S. (2019). *Skew-fit: State-covering self-supervised reinforcement learning*. arXiv:1903.03698.

#### Planning to Explore (Sekar et al., 2020)

Uses world models to plan for information gain, exploring states that maximize expected model improvement.

**Relevance to EMPO:** Since EMPO has explicit world models, could plan exploration trajectories that maximize coverage of uncertain regions.

> Sekar, R., Rybkin, O., Daniilidis, K., Abbeel, P., Hafner, D., & Pathak, D. (2020). *Planning to explore via self-supervised world models*. ICML 2020.

### Recommendations for EMPO

1. **Phase 1 (immediate):** Implement RND curiosity for trajectory-based exploration
2. **Phase 2 (future):** Add Go-Explore-style state archiving with reset
3. **Phase 3 (experimental):** Investigate random valid state generation for MultiGrid

The state sweeping approach is particularly promising for EMPO because:
- World models already support `set_state()` (unlike most RL environments)
- MultiGrid has discrete, enumerable state spaces
- Accurate value functions everywhere (not just on-policy) matter for EMPO's policy computation

---

## Open Design Questions

### 1. Separate vs Shared RND Modules

**Option A: Single shared RND module**
- Robot and human share novelty landscape
- Simpler implementation
- Any agent visiting a state reduces novelty for both

**Option B: Separate RND modules per agent**
- Each agent has independent novelty tracking
- Allows agent-specific exploration strategies
- More parameters, more memory

**Recommendation:** Start with shared (Option A) for simplicity. Can extend to separate modules if exploration patterns need to differ.

### 2. When to Compute Novelty

**Option A: At actor (during action selection)**
- Enables curiosity-driven action selection
- Adds inference overhead during data collection
- Required for exploration benefit

**Option B: At learner (on batches)**
- Only affects replay prioritization
- Lower overhead
- Doesn't directly improve exploration

**Recommendation:** Compute at actor for full exploration benefit. Consider caching novelty scores in transitions for replay prioritization.

### 3. Integration with Warm-up Phase

During warm-up, `beta_r=0` (uniform random robot policy). Options:
- **Enable curiosity during warm-up:** May improve state coverage before meaningful learning
- **Disable curiosity during warm-up:** Simpler, focus curiosity on post-warm-up learning

**Recommendation:** Enable curiosity during warm-up—early state coverage benefits later training.

### 4. Count-Based for Tabular Mode

Should `lookup_table_phase2_demo.py` get count-based curiosity?
- Simpler than RND for tabular case
- States are already hashable
- Natural fit for non-neural training

**Recommendation:** Yes, implement count-based curiosity for tabular mode as a separate, simpler feature.

---

## References

- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). *Exploration by Random Network Distillation*. arXiv:1810.12894
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). *Curiosity-driven Exploration by Self-Supervised Prediction*. ICML 2017 (ICM paper)
- Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). *Unifying Count-Based Exploration and Intrinsic Motivation*. NeurIPS 2016

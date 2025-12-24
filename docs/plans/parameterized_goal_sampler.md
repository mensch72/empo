# Implementation Plan: Parameterized Goal Sampler with Bayesian Updating

**Status:** Planning  
**Date:** 2025-12-24

## 1. Overview

This document outlines a plan to extend the `PossibleGoalSampler` to maintain a **parameterized goal distribution** $P(g|\theta)$ that can be updated via **variational Bayesian inference** based on observed human actions. The sampler will also model temporal goal dynamics using a **Markov jump process** $P(g'|g)$.

### 1.1 Core Idea

Instead of sampling goals from a fixed distribution, the new `ParameterizedGoalSampler`:

1. **Maintains parameters** $\theta$ that define a goal distribution $P(g|\theta)$
2. **Predicts next goals** via $P(g'|\theta) = \sum_g P(g|\theta) P(g'|g)$ using a Markov transition model
3. **Predicts next human actions** by combining goal beliefs with a `HumanPolicyPrior`: $P(a'|\theta) = \sum_{g'} P(g'|\theta) P(a'|g')$
4. **Updates beliefs** after observing human actions using variational Bayesian updating

### 1.2 Use Case Context

This will be used in **small gridworld environments** with:
- Two human agents (yellow)
- One robot agent (grey)
- `HeuristicHumanPolicyPrior` for modeling human behavior
- A novel `HeuristicRobotPolicy` (to be developed)
- Eventually, integration with learned robot policies from PR #50

## 2. Mathematical Formulation

### 2.1 Goal Distribution Model

For multigrid environments, we model goal regions as **axis-aligned rectangles**. Each rectangle goal $g$ is defined by:
- $(x_1, y_1, x_2, y_2)$: bounding box coordinates (where $x_1 \leq x_2$, $y_1 \leq y_2$)

#### 2.1.1 Generalization of MultiGridGoalSampler

The existing `MultiGridGoalSampler` uses the distribution:

$$P(x_1, x_2, y_1, y_2) \propto (1 + x_2 - x_1) \cdot (1 + y_2 - y_1)$$

This samples rectangles with probability proportional to their area. We generalize this to a parameterized family that:
1. **Maintains the width weighting** $(1 + x_2 - x_1)$ which is important for proper goal weighting
2. **Allows modifying the center distribution** via a parameterized component $P_0$
3. **Maintains independence** between x and y coordinates

#### 2.1.2 Parameterized Distribution Family

The parameterized distribution factors as:

$$P(g|\theta) = P(x_1, x_2|\theta_x) \cdot P(y_1, y_2|\theta_y)$$

where each coordinate pair follows:

$$P(x_1, x_2|\theta_x) \propto (1 + x_2 - x_1) \cdot P_0\left(\frac{x_1 + x_2}{2} \bigg| \theta_x\right)$$

and analogously for $(y_1, y_2)$.

Here:
- The factor $(1 + x_2 - x_1)$ preserves the **width weighting** from the original sampler
- $P_0(c|\theta)$ is a **parameterized center distribution** on $\{x_{\min}, \ldots, x_{\max}\}$

#### 2.1.3 Center Distribution Family $P_0$

We need a family $P_0(c|\theta)$ on discrete values $\{c_{\min}, \ldots, c_{\max}\}$ that:
1. **Contains uniform** as a special case (to recover the original `MultiGridGoalSampler`)
2. **Allows concentration** around any point (to express beliefs about goal location)
3. **Has few parameters** for efficient Bayesian updating

**Recommended: Discretized Gaussian family**

$$P_0(c|\mu, \kappa) \propto \exp\left(-\kappa (c - \mu)^2\right)$$

where:
- $\mu \in [c_{\min}, c_{\max}]$ is the center mean (continuous parameter)
- $\kappa \geq 0$ is the concentration (inverse variance)

**Special cases:**
- $\kappa = 0$: Uniform distribution (recovers original `MultiGridGoalSampler`)
- $\kappa \to \infty$: Point mass at $\lfloor\mu\rfloor$ (rounding $\mu$ to nearest integer)

**Alternative: Categorical softmax family**

$$P_0(c|\mathbf{w}) = \frac{\exp(w_c)}{\sum_{c'} \exp(w_{c'})}$$

where $\mathbf{w} = (w_{c_{\min}}, \ldots, w_{c_{\max}})$ are log-weights.

- **Pro**: Most flexible, can represent any distribution
- **Con**: More parameters (one per grid position), harder to update efficiently

**Recommended parameterization**: Use discretized Gaussian with 2 parameters per axis:
- $\theta_x = (\mu_x, \kappa_x)$ for x-center distribution
- $\theta_y = (\mu_y, \kappa_y)$ for y-center distribution

This gives **4 total parameters** while containing the uniform distribution as $\kappa_x = \kappa_y = 0$.

#### 2.1.4 Full Joint Distribution

Combining the coordinate pairs:

$$P(x_1, x_2, y_1, y_2 | \mu_x, \kappa_x, \mu_y, \kappa_y) \propto (1 + x_2 - x_1) \cdot \exp\left(-\kappa_x \left(\frac{x_1+x_2}{2} - \mu_x\right)^2\right) \cdot (1 + y_2 - y_1) \cdot \exp\left(-\kappa_y \left(\frac{y_1+y_2}{2} - \mu_y\right)^2\right)$$

**Verification**: When $\kappa_x = \kappa_y = 0$, this reduces to:

$$P(x_1, x_2, y_1, y_2) \propto (1 + x_2 - x_1) \cdot (1 + y_2 - y_1)$$

which is exactly the existing `MultiGridGoalSampler` distribution. ✓

#### 2.1.5 Sampling Algorithm

To sample from $P(x_1, x_2|\theta_x)$:

1. **Enumerate** all valid $(x_1, x_2)$ pairs with $x_1 \leq x_2$ in $[x_{\min}, x_{\max}]$
2. **Compute weights** $w_{x_1,x_2} = (1 + x_2 - x_1) \cdot \exp\left(-\kappa_x \left(\frac{x_1+x_2}{2} - \mu_x\right)^2\right)$
3. **Normalize** and sample

For small grids (e.g., 10×10), this is efficient. For larger grids, approximate sampling may be needed.

### 2.2 Markov Jump Process for Goal Dynamics

The goal transition model $P(g'|g)$ describes how goals might change between time steps. We model this as independent transitions on the coordinate pairs:

$$P(g'|g) = P(x_1', x_2' | x_1, x_2) \cdot P(y_1', y_2' | y_1, y_2)$$

For each coordinate pair, we use a **coupled random walk** that:
1. Keeps the width distribution roughly proportional to $(1 + x_2 - x_1)$
2. Allows the center $(x_1 + x_2)/2$ to drift

**Transition model:**
$$P(x_1', x_2' | x_1, x_2) \propto \begin{cases}
p_{\text{stay}} & \text{if } (x_1', x_2') = (x_1, x_2) \\
(1 + x_2' - x_1') \cdot \exp\left(-\lambda \left|\frac{x_1'+x_2'}{2} - \frac{x_1+x_2}{2}\right|\right) & \text{otherwise}
\end{cases}$$

This preserves the width weighting while allowing the center to drift according to an exponential decay kernel.

**Fixed parameters** (not learned):
- $p_{\text{stay}}$: probability the goal doesn't change (typically high, e.g., 0.9)
- $\lambda$: rate parameter controlling how far the center can jump (higher = smaller jumps)

### 2.3 Action Prediction

Given the goal prediction $P(g'|\theta)$ and a `HumanPolicyPrior` that provides $P(a'|g')$:

$$P(a'|\theta) = \sum_{g'} P(g'|\theta) P(a'|g')$$

This is computed via Monte Carlo sampling when the goal space is large:

```python
def predict_action_distribution(self, state, human_agent_idx):
    action_probs = np.zeros(num_actions)
    for _ in range(n_samples):
        g_current = self.sample_goal(state, human_agent_idx)  # sample from P(g|θ)
        g_next = self.transition_goal(g_current)              # sample from P(g'|g)
        p_action_given_goal = self.human_policy_prior(state, human_agent_idx, g_next)
        action_probs += p_action_given_goal
    return action_probs / n_samples
```

### 2.4 Variational Bayesian Updating

After observing human action $a'$, we want to update $\theta$ so that $P(g'|\theta')$ approximates the posterior:

$$P(g'|\theta, a') = \frac{P(g'|\theta) P(a'|g')}{P(a'|\theta)}$$

where $P(a'|\theta) = \sum_{g'} P(g'|\theta) P(a'|g')$ is the marginal likelihood.

**Verification of soundness:**

The equation is indeed a valid application of Bayes' rule:
- Prior: $P(g'|\theta)$ — our belief about the next goal before observing action
- Likelihood: $P(a'|g')$ — probability of observed action given goal (from HumanPolicyPrior)
- Evidence: $P(a'|\theta)$ — marginal probability of action (normalizing constant)
- Posterior: $P(g'|\theta, a')$ — updated belief after observing action

**Variational approximation:**

Since $P(g'|\theta, a')$ may not have a closed form, we approximate it by finding $\theta'$ that minimizes the KL divergence:

$$\theta' = \arg\min_{\theta'} D_{KL}(P(g'|\theta, a') || P(g'|\theta'))$$

For the Gaussian parameterization, this can be done via **stochastic gradient descent**:

$$\theta' \leftarrow \theta + \alpha \nabla_\theta \mathbb{E}_{g' \sim P(g'|\theta)}[\log P(a'|g')]$$

This is equivalent to **maximum likelihood estimation** on the "pseudo-likelihood" $P(a'|g')$ weighted by the prior $P(g'|\theta)$.

**Practical implementation via importance sampling:**

```python
def update_parameters(self, state, human_agent_idx, observed_action):
    # Sample goals from current prior
    samples = [self.sample_goal(state, human_agent_idx) for _ in range(n_samples)]
    
    # Compute importance weights: P(a'|g') for each sampled goal g'
    weights = []
    for g in samples:
        g_next = self.transition_goal(g)
        p_action_dist = self.human_policy_prior(state, human_agent_idx, g_next)
        weights.append(p_action_dist[observed_action])
    
    # Normalize weights
    weights = np.array(weights)
    weights /= weights.sum()
    
    # Compute weighted statistics for parameter update
    # (reweighting samples to approximate posterior)
    ...
```

### 2.5 Alternative: Particle Filter Approach

For computational efficiency, we could use a **particle filter** instead of parametric updates:

1. Maintain $N$ particles (goal hypotheses) with weights
2. **Predict step**: Propagate particles through transition model $P(g'|g)$
3. **Update step**: Reweight particles by likelihood $P(a'|g')$
4. **Resample** when effective sample size is low

This approach:
- ✅ Handles multimodal posteriors naturally
- ✅ No gradient computation needed
- ❌ Requires more memory for many particles
- ❌ Suffers from particle degeneracy over long horizons

## 3. Integration with Existing Code

### 3.1 New Abstract Base Class

Extend `PossibleGoalSampler` to create `ParameterizedGoalSampler`:

```python
# src/empo/possible_goal.py

class ParameterizedGoalSampler(PossibleGoalSampler):
    """
    Abstract base class for goal samplers with learnable parameters
    that can be updated based on observed actions.
    """
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current goal distribution parameters θ."""
        pass
    
    @abstractmethod
    def set_parameters(self, theta: Dict[str, Any]) -> None:
        """Set goal distribution parameters θ."""
        pass
    
    @abstractmethod
    def predict_next_goal_distribution(
        self, 
        state, 
        human_agent_index: int
    ) -> 'ParameterizedGoalSampler':
        """
        Apply Markov transition model to predict P(g'|θ).
        Returns a sampler for the predicted distribution.
        """
        pass
    
    @abstractmethod
    def predict_action_distribution(
        self,
        state,
        human_agent_index: int,
        human_policy_prior: 'HumanPolicyPrior'
    ) -> np.ndarray:
        """
        Predict P(a'|θ) by marginalizing over predicted goals.
        """
        pass
    
    @abstractmethod
    def update_parameters(
        self,
        state,
        human_agent_index: int,
        observed_action: int,
        human_policy_prior: 'HumanPolicyPrior'
    ) -> None:
        """
        Update θ via variational Bayesian inference given observed action.
        """
        pass
```

### 3.2 Concrete Implementation for Multigrid

```python
# src/empo/nn_based/multigrid/parameterized_goal_sampler.py

class MultiGridParameterizedGoalSampler(ParameterizedGoalSampler):
    """
    Parameterized goal sampler for multigrid environments.
    
    Generalizes MultiGridGoalSampler with parameterized center distribution:
    P(x1,x2,y1,y2|θ) ∝ (1+x2-x1) * exp(-κx*(center_x - μx)²) 
                      * (1+y2-y1) * exp(-κy*(center_y - μy)²)
    
    When kappa_x = kappa_y = 0, this recovers the original MultiGridGoalSampler.
    """
    
    def __init__(
        self,
        env: 'WorldModel',
        # Initial center distribution parameters
        mu_x: float = None,     # x-center mean (None = grid center)
        kappa_x: float = 0.0,   # x-center concentration (0 = uniform)
        mu_y: float = None,     # y-center mean (None = grid center)
        kappa_y: float = 0.0,   # y-center concentration (0 = uniform)
        # Markov transition parameters (fixed)
        p_stay: float = 0.9,
        jump_rate: float = 0.5,
        # Learning parameters
        learning_rate: float = 0.1,
        n_samples: int = 100,
    ):
        super().__init__(env)
        # ... store parameters ...
```

### 3.3 Integration with World Model Step

The update should happen **before** each `step()` call:

```python
class EnhancedWorldModel(WorldModel):
    """World model with goal belief tracking."""
    
    def __init__(
        self,
        base_world_model: WorldModel,
        goal_sampler: ParameterizedGoalSampler,
        human_policy_prior: HumanPolicyPrior,
        human_agent_indices: List[int],
    ):
        self.goal_sampler = goal_sampler
        self.human_policy_prior = human_policy_prior
        self.human_agent_indices = human_agent_indices
    
    def step(self, actions):
        # Before step: Update goal beliefs based on observed human actions
        for i, human_idx in enumerate(self.human_agent_indices):
            observed_action = actions[human_idx]
            self.goal_sampler.update_parameters(
                state=self.get_state(),
                human_agent_index=human_idx,
                observed_action=observed_action,
                human_policy_prior=self.human_policy_prior
            )
        
        # Execute the step
        return super().step(actions)
```

Alternatively, this could be a **callback** or **observer pattern** rather than subclassing.

## 4. Heuristic Robot Policy

As mentioned in the context, we also need a `HeuristicRobotPolicy`. This is a separate but related component.

### 4.1 Relationship to Goal Sampler

The robot should use the `ParameterizedGoalSampler` to:
1. Understand current beliefs about human goals
2. Predict how goals might change
3. Choose actions that maximize expected human empowerment

### 4.2 Sketch of HeuristicRobotPolicy

```python
class HeuristicRobotPolicy:
    """
    Heuristic policy for robot agents based on goal beliefs.
    
    Uses the parameterized goal sampler to estimate human goals
    and chooses actions that don't obstruct human goal achievement.
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        robot_agent_indices: List[int],
        human_agent_indices: List[int],
        goal_sampler: ParameterizedGoalSampler,
        human_policy_prior: HumanPolicyPrior,
    ):
        pass
    
    def __call__(self, state) -> np.ndarray:
        """
        Returns action distribution for robot agent(s).
        
        Heuristic: Avoid blocking paths that humans are likely to take.
        """
        # Sample likely human goals
        likely_goals = [
            self.goal_sampler.sample(state, h)
            for h in self.human_agent_indices
            for _ in range(n_samples)
        ]
        
        # For each robot action, evaluate impact on human goal achievement
        # Choose actions that preserve or enhance human options
        ...
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Files to create:**
- `src/empo/possible_goal.py` — Add `ParameterizedGoalSampler` ABC
- `src/empo/nn_based/multigrid/parameterized_goal_sampler.py` — Concrete implementation

**Tasks:**
1. Define abstract interface for parameterized goal samplers
2. Implement Gaussian-parameterized sampler for multigrid rectangles
3. Implement Markov transition model with configurable parameters
4. Unit tests for sampling and transitions

### Phase 2: Bayesian Updating (Week 2-3)

**Tasks:**
1. Implement action prediction: $P(a'|\theta)$
2. Implement variational update via importance sampling
3. Add learning rate scheduling / adaptive updates
4. Unit tests for update convergence

### Phase 3: Integration (Week 3-4)

**Files to create/modify:**
- `src/empo/heuristic_robot_policy.py` — New heuristic robot policy
- Integration with world model step callbacks

**Tasks:**
1. Create `HeuristicRobotPolicy` class
2. Wire up goal sampler updates in step loop
3. Integration tests with small gridworld environments

### Phase 4: Validation (Week 4+)

**Tasks:**
1. Test in 2-human, 1-robot gridworld scenarios
2. Validate belief tracking accuracy
3. Benchmark computational overhead
4. Connect to PR #50 learned robot policies

## 6. Configuration

```python
@dataclass
class ParameterizedGoalSamplerConfig:
    """Configuration for parameterized goal sampler.
    
    The parameterized distribution is:
    P(x1,x2,y1,y2|θ) ∝ (1+x2-x1) * exp(-κx*(center_x - μx)²) 
                      * (1+y2-y1) * exp(-κy*(center_y - μy)²)
    
    When κx = κy = 0, this recovers the original MultiGridGoalSampler distribution
    P(goal) ∝ (1+x2-x1)*(1+y2-y1).
    """
    
    # Initial distribution parameters (4 parameters total)
    # mu_x, mu_y: center means (None = use grid center)
    # kappa_x, kappa_y: concentration parameters (0 = uniform center distribution)
    initial_params: Dict[str, Optional[float]] = field(default_factory=lambda: {
        'mu_x': None,      # x-center mean (None = grid center)
        'kappa_x': 0.0,    # x-center concentration (0 = uniform, recovers original sampler)
        'mu_y': None,      # y-center mean (None = grid center)  
        'kappa_y': 0.0,    # y-center concentration (0 = uniform, recovers original sampler)
    })
    
    # Markov transition model parameters (fixed, not learned)
    transition_params: Dict[str, float] = field(default_factory=lambda: {
        'p_stay': 0.9,          # Probability goal stays the same
        'jump_rate': 0.5,       # λ: exponential decay rate for center drift
    })
    
    # Bayesian updating parameters
    learning_rate: float = 0.1
    n_samples: int = 100
    min_kappa: float = 0.0     # Minimum concentration (0 = uniform allowed)
    max_kappa: float = 10.0    # Maximum concentration (prevents over-concentration)
    
    # Particle filter alternative (if used)
    use_particle_filter: bool = False
    n_particles: int = 1000
    resample_threshold: float = 0.5  # ESS ratio for resampling
```

## 7. API Design

### 7.1 Public Interface

```python
# Creating a parameterized goal sampler (recovers original MultiGridGoalSampler)
sampler = MultiGridParameterizedGoalSampler(
    env=world_model,
    kappa_x=0.0,  # uniform x-center distribution
    kappa_y=0.0,  # uniform y-center distribution
)

# Creating a sampler with concentrated center belief
sampler = MultiGridParameterizedGoalSampler(
    env=world_model,
    mu_x=5.0,     # x-center mean at x=5
    kappa_x=2.0,  # moderate concentration around x=5
    mu_y=3.0,     # y-center mean at y=3
    kappa_y=2.0,  # moderate concentration around y=3
)

# Sample a goal (inherits from PossibleGoalSampler)
goal, weight = sampler.sample(state, human_agent_index=0)

# Get current belief parameters
theta = sampler.get_parameters()
# Returns: {'mu_x': 5.0, 'kappa_x': 2.0, 'mu_y': 3.0, 'kappa_y': 2.0}

# Predict action distribution
p_action = sampler.predict_action_distribution(
    state, 
    human_agent_index=0,
    human_policy_prior=policy_prior
)

# Update after observing action
sampler.update_parameters(
    state,
    human_agent_index=0,
    observed_action=3,  # e.g., "forward"
    human_policy_prior=policy_prior
)
```

### 7.2 Integration with Existing Code

The `ParameterizedGoalSampler` extends `PossibleGoalSampler`, so it can be used anywhere a `PossibleGoalSampler` is expected:

```python
# Works with existing Monte Carlo integration
from empo.possible_goal import approx_integral_over_possible_goals

result = approx_integral_over_possible_goals(
    state,
    human_agent_index=0,
    sampler=parameterized_sampler,  # Uses parameterized distribution!
    func=my_function,
    sample_size=1000
)
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_parameterized_goal_sampler.py

def test_sample_produces_valid_goals():
    """Sampled goals should be valid rectangle regions."""
    pass

def test_transition_preserves_distribution_shape():
    """After many transitions, distribution should stabilize."""
    pass

def test_action_prediction_sums_to_one():
    """Predicted action distribution should be valid probability."""
    pass

def test_update_increases_likelihood_of_observed_action():
    """After update, P(observed_action|θ') > P(observed_action|θ)."""
    pass

def test_parameters_converge_with_consistent_actions():
    """If human always moves right, beliefs should concentrate on rightward goals."""
    pass
```

### 8.2 Integration Tests

```python
def test_integration_with_heuristic_policy():
    """Goal sampler + heuristic policy should improve human outcomes."""
    pass

def test_multi_human_tracking():
    """Separate beliefs maintained for each human."""
    pass
```

## 9. Open Questions

### 9.1 Goal Representation

**Q1:** How should we handle the case where the width distribution needs to be modified?

**Current approach:** We keep the width weighting $(1 + x_2 - x_1)$ fixed as in the original `MultiGridGoalSampler`, and only parameterize the center distribution. This preserves the important property that larger rectangles are sampled proportionally more often.

**Future extension:** If needed, we could add a parameterized width preference by using:
$$P(x_1, x_2|\theta) \propto (1 + x_2 - x_1)^{\alpha} \cdot P_0(\text{center}|\theta)$$
where $\alpha > 0$ controls width preference ($\alpha = 1$ is the current model).

### 9.2 Multi-Human Correlation

**Q2:** Should we model correlation between different humans' goals?

**Recommendation:** Start with independent beliefs per human. Correlation can be added later if needed.

### 9.3 Parameter Sharing

**Q3:** Should parameters be shared across humans (same prior) or independent?

**Recommendation:** Start with shared initial parameters but independent updates (each human's observed actions update only their own goal beliefs).

### 9.4 Handling Goal Achievement

**Q4:** What happens when a human achieves their goal (reaches the rectangle)?

**Options:**
- Reset parameters to prior
- Sample a new goal from the posterior
- Maintain current beliefs (goal persistence)

**Recommendation:** Sample new goal from current posterior — maintains continuity while allowing goal changes.

### 9.5 Numerical Stability

**Q5:** How to handle numerical issues?
- Very small likelihoods $P(a'|g') \approx 0$
- Sigma collapse when updates are too aggressive
- Importance weight degeneracy

**Mitigations:**
- Log-space computations where possible
- Minimum sigma bounds
- Regularization toward prior
- Effective sample size monitoring for importance sampling

## 10. Connection to PR #50 (Phase 2 Robot Policy Learning)

This work will integrate with the **Phase 2 learning** approach from PR #50, which implements neural network-based robot policy training using equations (4)-(9) from the EMPO paper. The Phase 2 approach learns robot Q-values, policies, and human goal achievement estimates through experience replay.

1. **HeuristicRobotPolicy** serves as a baseline/fallback before learned policies are available
2. Once learned robot policies are trained (per PR #50), they can use the same `ParameterizedGoalSampler` to inform their decisions with updated goal beliefs
3. The goal beliefs provide a **structured representation** that learned policies can condition on
4. After each step, both the goal sampler AND the learned policy can be updated:
   - Goal sampler: via Bayesian updating from observed human actions
   - Learned policy: via gradient updates as described in PR #50's implementation plan

## 11. References

- Existing `PossibleGoalSampler` interface: `src/empo/possible_goal.py`
- `MultiGridGoalSampler` implementation: `src/empo/multigrid.py`
- `HumanPolicyPrior` interface: `src/empo/human_policy_prior.py`
- `HeuristicPotentialPolicy` implementation: `src/empo/human_policy_prior.py`
- Phase 2 learning plan: See PR #50 branch `copilot/create-implementation-document` for `docs/plans/issue_49_phase2_learning.md`

## 12. Summary

This plan describes an extension to the goal sampling infrastructure that:

1. **Generalizes `MultiGridGoalSampler`** with a parameterized center distribution:
   $$P(x_1, x_2|\theta) \propto (1 + x_2 - x_1) \cdot P_0\left(\frac{x_1+x_2}{2} \bigg| \mu, \kappa\right)$$
   where $P_0$ is a discretized Gaussian with mean $\mu$ and concentration $\kappa$
2. **Recovers the original sampler** when $\kappa = 0$ (uniform center distribution)
3. **Uses 4 parameters** total: $(\mu_x, \kappa_x, \mu_y, \kappa_y)$
4. **Models dynamics** via a Markov jump process with fixed parameters
5. **Predicts actions** by marginalizing over predicted goals
6. **Updates beliefs** via variational Bayesian inference

The implementation builds on existing abstractions (`PossibleGoalSampler`, `MultiGridGoalSampler`, `HumanPolicyPrior`) and will integrate with both heuristic and learned robot policies for use in small gridworld experiments.

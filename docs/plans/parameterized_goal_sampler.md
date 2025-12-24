# Implementation Plan: Parameterized Goal Sampler with Bayesian Updating

**Status:** Planning  
**Author:** GitHub Copilot  
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

The parameterized distribution $P(g|\theta)$ factors into independent distributions for each dimension:

$$P(g|\theta) = P(w|\theta_w) \cdot P(h|\theta_h) \cdot P(x_c|\theta_x) \cdot P(y_c|\theta_y)$$

where:
- $w = x_2 - x_1 + 1$ is the width
- $h = y_2 - y_1 + 1$ is the height  
- $(x_c, y_c) = ((x_1 + x_2)/2, (y_1 + y_2)/2)$ is the center

**Parameterization options:**

| Distribution | Parameters | Description |
|--------------|------------|-------------|
| Truncated Gaussian | $(\mu, \sigma)$ | Mean and standard deviation |
| Categorical | $(p_1, ..., p_n)$ | Probability for each value |
| Beta (scaled) | $(\alpha, \beta)$ | Shape parameters for continuous [0,1] |

For a simple implementation with 8 total parameters, we could use:
- $\theta_w = (\mu_w, \sigma_w)$ for width distribution
- $\theta_h = (\mu_h, \sigma_h)$ for height distribution
- $\theta_x = (\mu_x, \sigma_x)$ for x-center distribution
- $\theta_y = (\mu_y, \sigma_y)$ for y-center distribution

### 2.2 Markov Jump Process for Goal Dynamics

The goal transition model $P(g'|g)$ describes how goals might change between time steps. For rectangle goals in multigrid, we model this as independent random walks:

$$P(g'|g) = P(w'|w) \cdot P(h'|h) \cdot P(x_c'|x_c) \cdot P(y_c'|y_c)$$

**Simple random walk model:**
$$P(d'|d) = \begin{cases}
p_{\text{stay}} & \text{if } d' = d \\
p_{\text{change}} \cdot \text{softmax}(|d' - d|^{-1}) & \text{otherwise}
\end{cases}$$

where $d$ is any of $(w, h, x_c, y_c)$.

**Fixed parameters** (not learned):
- $p_{\text{stay}}$: probability the goal dimension doesn't change
- $\lambda$: rate parameter for jump magnitude (exponential decay)

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
    
    Models goals as rectangular regions with Gaussian-parameterized
    distributions over width, height, and center position.
    """
    
    def __init__(
        self,
        env: 'WorldModel',
        # Initial parameters
        mu_w: float = 1.0,    sigma_w: float = 0.5,
        mu_h: float = 1.0,    sigma_h: float = 0.5,
        mu_x: float = None,   sigma_x: float = 2.0,  # None = grid center
        mu_y: float = None,   sigma_y: float = 2.0,
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
    """Configuration for parameterized goal sampler."""
    
    # Initial distribution parameters
    # Note: mu_x and mu_y can be None to indicate "use grid center"
    initial_params: Dict[str, Optional[float]] = field(default_factory=lambda: {
        'mu_w': 1.0, 'sigma_w': 0.5,
        'mu_h': 1.0, 'sigma_h': 0.5,
        'mu_x': None, 'sigma_x': 2.0,  # None = use grid center
        'mu_y': None, 'sigma_y': 2.0,
    })
    
    # Markov transition model parameters (fixed, not learned)
    transition_params: Dict[str, float] = field(default_factory=lambda: {
        'p_stay': 0.9,          # Probability goal dimension stays same
        'jump_rate': 0.5,       # Rate of exponential decay for jump size
    })
    
    # Bayesian updating parameters
    learning_rate: float = 0.1
    n_samples: int = 100
    min_sigma: float = 0.1     # Minimum allowed sigma (prevents collapse)
    max_sigma: float = 10.0    # Maximum allowed sigma
    
    # Particle filter alternative (if used)
    use_particle_filter: bool = False
    n_particles: int = 1000
    resample_threshold: float = 0.5  # ESS ratio for resampling
```

## 7. API Design

### 7.1 Public Interface

```python
# Creating a parameterized goal sampler
sampler = MultiGridParameterizedGoalSampler(
    env=world_model,
    config=ParameterizedGoalSamplerConfig(),
)

# Sample a goal (inherits from PossibleGoalSampler)
goal, weight = sampler.sample(state, human_agent_index=0)

# Get current belief parameters
theta = sampler.get_parameters()
# Returns: {'mu_w': 1.5, 'sigma_w': 0.3, 'mu_h': ..., ...}

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

**Q1:** Should goals be represented as:
- Exact rectangles $(x_1, y_1, x_2, y_2)$
- Center + size $(x_c, y_c, w, h)$
- Mixture of point goals with spatial correlation

**Recommendation:** Start with center + size (4 parameters per dimension × 2 = 8 parameters total), which aligns with the problem statement.

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

## 10. Connection to PR #50

This work will integrate with the Phase 2 learning from PR #50:

1. **HeuristicRobotPolicy** serves as a baseline/fallback
2. Once learned robot policies are available, they can use the same `ParameterizedGoalSampler` to inform their decisions
3. The goal beliefs provide a **structured representation** that learned policies can condition on
4. After each step, both the goal sampler AND the learned policy can be updated:
   - Goal sampler: via Bayesian updating from observed actions
   - Learned policy: via gradient updates as in PR #50

## 11. References

- Existing `PossibleGoalSampler` interface: `src/empo/possible_goal.py`
- `HumanPolicyPrior` interface: `src/empo/human_policy_prior.py`
- `HeuristicPotentialPolicy` implementation: `src/empo/human_policy_prior.py`
- Phase 2 learning plan: `docs/plans/issue_49_phase2_learning.md` (PR #50)

## 12. Summary

This plan describes an extension to the goal sampling infrastructure that:

1. **Parameterizes** the goal distribution with learnable parameters $\theta$
2. **Models dynamics** via a Markov jump process with fixed parameters
3. **Predicts actions** by marginalizing over predicted goals
4. **Updates beliefs** via variational Bayesian inference

The implementation builds on existing abstractions (`PossibleGoalSampler`, `HumanPolicyPrior`) and will integrate with both heuristic and learned robot policies for use in small gridworld experiments.

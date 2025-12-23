# Implementation Plan: Learning Approach for Robot Policy (Phase 2)

**Issue:** #49  
**Status:** Planning  
**Author:** GitHub Copilot  
**Date:** 2025-12-23

## 1. The Paper's Equations

This section reproduces and explains the equations from Table 1 of the EMPO paper.

### 1.1 Phase 1: Human Behavior Prior (Equations 1-3)

These equations compute what the robot believes about human behavior. They are already implemented.

```
(1) Q_h^m(s, g_h, a_h) ← E_{a_{-h} ~ μ_{-h}(s,g_h)} min_{a_r ∈ A_r(s)} E_{s'|s,a} [U_h(s', g_h) + γ_h V_h^m(s', g_h)]

(2) π_h(s, g_h) ← ν_h(s,g_h) π_h^0(s,g_h) + (1 - ν_h(s,g_h)) × softmax_{β_h(s,g_h)}(Q_h^m(s, g_h, ·))

(3) V_h^m(s, g_h) ← E_{a_h ~ π_h(s,g_h)} Q_h^m(s, g_h, a_h)
```

**Meaning:**
- `Q_h^m`: Human h's *model-based* Q-value for action `a_h` toward goal `g_h`. Uses `min_{a_r}` because we want to incentivize the robot (in our case: the robot fleet) to make binding commitments that restrict its actions space and thereby automatically increase this value automatically.
- `π_h`: Human policy - a mixture of habitual behavior `π_h^0` and boundedly-rational Boltzmann policy.
- `V_h^m`: Human's *model-based* value - expected Q under their own policy.

### 1.2 Phase 2: Robot Policy (Equations 4-9)

These equations compute the robot's (in our case: the robot fleet's) policy to softly maximize aggregate human power. **This is what we need to implement.**

```
(4) Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]

(5) π_r(s)(a_r) ∝ (-Q_r(s, a_r))^{-β_r}

(6) V_h^e(s, g_h) ← E_{g_{-h}} E_{a_H ~ π_H(s,g)} E_{a_r ~ π_r(s)} E_{s'|s,a} [U_h(s', g_h) + γ_h V_h^e(s', g_h)]

(7) X_h(s) ← Σ_{g_h ∈ G_h} V_h^e(s, g_h)^ζ

(8) U_r(s) ← -(Σ_h X_h(s)^{-ξ})^η

(9) V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)
```

### 1.3 Detailed Explanation of Phase 2 Equations

#### Equation (4): Robot State-Action Value `Q_r(s, a_r)`

```
Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]
```

**What it computes:** The expected discounted future value `V_r(s')` when the robot takes action `a_r` (in our case: when the robot fleet takes action combination `a_r`).

**Key observations:**
- The equation for `Q_r` does NOT explicitly include immediate reward `U_r(s)` - only the discounted continuation, because the reward from taking some action is assumed to accrue at the time of arrival at the successor state (and it also exclusively depends on that successor state), and so we chose to include in the value function of the successor state (`V_r(s')`, see below) instead of here. So it is included here, but indirectly via `V_r(s')`. 
- The expectation is over:
  - `g`: goal profile (one goal per human), sampled uniformly
  - `a_H ~ π_H(s,g)`: human actions from their goal-conditioned policies
  - `s'|s,a`: next state given current state and joint action
- Since `V_r < 0` (see eq. 9), we have `Q_r < 0`

#### Equation (5): Robot Policy `π_r(s)`

```
π_r(s)(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
```

**What it computes:** The robot's stochastic policy using a power-law softmax.

**Key observations:**
- This is NOT standard Boltzmann softmax `exp(β Q)`
- Since `Q_r < 0`, we have `-Q_r > 0`, so the expression is well-defined
- Higher `β_r` → more deterministic (concentrates on best action)
- The power-law form `(-Q)^{-β}` satisfies certain scale-invariance properties (see paper's Table 2).

#### Equation (6): Effective Human Goal Achievement `V_h^e(s, g_h)`

```
V_h^e(s, g_h) ← E_{g_{-h}} E_{a_H ~ π_H(s,g)} E_{a_r ~ π_r(s)} E_{s'|s,a} [U_h(s', g_h) + γ_h V_h^e(s', g_h)]
```

**What it computes:** The probability that human h achieves goal `g_h`, under the *actual* robot policy `π_r`.

**Key observations:**
- Different from `V_h^m` (eq. 3) which used `min_{a_r}` (worst-case robot assumption)
- `V_h^e` uses the learned robot policy `π_r(s)` instead (this causes a mutual dependency between `V_h^e` and `π_r`, so we cannot first learn one of them and then the other but have to learn them in common)
- `U_h(s', g_h) = 1_{s' ∈ g_h}` is the goal achievement indicator, and once it is 1 the episode ends for that human (the human gets no further reward in this episode).
- This implies that `V_h^e ∈ [0, 1]` since it's a (discounted) probability of achieving the goal
- The expectation over `g_{-h}` (other humans' goals) reflects uncertainty about what goals others are pursuing

#### Equation (7): Aggregate Goal Achievement Ability `X_h(s)`

```
X_h(s) ← Σ_{g_h ∈ G_h} V_h^e(s, g_h)^ζ
```

**What it computes:** A measure of human h's total "power" - their ability to achieve various goals.

**Key observations:**
- Sums over ALL possible goals `g_h ∈ G_h`
- `ζ > 1` introduces risk aversion/reliability preference:
  - Prefers certain achievement (one goal with V=1) over uncertain (two goals with V=0.5 each)
  - With `ζ = 2`: `1^2 = 1 > 2 × 0.5^2 = 0.5`
- Human "power in bits" would be `W_h(s) = log_2(X_h(s))`, but we don't actually use that quantity in any equations but work with `X_h` instead.
- `X_h > 0` always (at least one goal is achievable with some probability because the goals cover all possible trajectories and at least one trajectory will have positive probability)

#### Equation (8): Intrinsic Robot Reward `U_r(s)`

```
U_r(s) ← -(Σ_h X_h(s)^{-ξ})^η
```

**What it computes:** The robot's intrinsic reward based on aggregate human power.

**Key observations:**
- Always negative: `U_r(s) < 0`
- The `X_h^{-ξ}` term with `ξ > 0` means:
  - Humans with LOW power contribute MORE to the sum
  - This implements inter-human inequality aversion (protects the least powerful)
  - With `ξ = 1`: reducing one person's power from 1 bit to 0 bits cannot be compensated by any increase to one other human who has at least one bit already (but can be compensated by large increases in *several* other humans) 
- The outer `η > 1` power implements *additional inter*temporal inequality aversion (e.g., trajectories with constant `U_r=-2` are preferred to trajectories alternating between `U_r=-1` and `U_r=-3`)
- Robot wants to SOFTLY MAXIMIZE the expected long-term sum of `U_r` (make it less negative) → increase human power sustainably

#### Equation (9): Robot State Value `V_r(s)`

```
V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)
```

**What it computes:** The robot's value function.

**Key observations:**
- Standard Bellman equation: immediate reward + expected continuation
- Since `Q_r = γ_r E[V_r(s')]` (from eq. 4), this expands to:
  ```
  V_r(s) = U_r(s) + γ_r E_{a_r ~ π_r, a_H ~ π_H, s'|s,a}[V_r(s')]
  ```
- `V_r < 0` always (since `U_r < 0` and `Q_r < 0`)

### 1.4 Circular Dependencies

The Phase 2 equations have circular dependencies:

```
Q_r(s, a_r) depends on V_r(s')           [eq. 4]
π_r(s) depends on Q_r(s, ·)              [eq. 5]
V_h^e(s, g_h) depends on π_r(s)          [eq. 6]
X_h(s) depends on V_h^e(s, ·)            [eq. 7]
U_r(s) depends on X_h(s)                 [eq. 8]
V_r(s) depends on U_r(s) and Q_r(s, ·)   [eq. 9]
```

This creates a fixed-point problem that must be solved iteratively (backward induction in small enough, acyclic environments, or iterative learning in general).

## 2. Modifications for Our Implementation

Per issue #49, we make the following modifications:

### 2.1 Robot Fleet

The paper's single robot `r` represents our **robot fleet** - multiple robots coordinated by a single AI. The "robot action" `a_r` is a **tuple of actions**, one for each robot:

```
a_r = (a_{r_1}, a_{r_2}, ..., a_{r_k})
```

where `k` is the number of robots. The robot policy outputs a joint distribution over this action space.

### 2.2 Stochastic Approximation for Large Goal/Human Spaces

**Equation (7) modification:**
Instead of summing over all goals `Σ_{g_h ∈ G_h}`, we use expected value with sampling:
```
X_h(s) ← E_{g_h ~ possible_goal_sampler(h)} [V_h^e(s, g_h)^ζ]
```

This allows Monte Carlo approximation when `G_h` is large.

**Equation (8) modification:**
Instead of summing over all humans `Σ_h`, we use expected value with sampling:
```
U_r(s) ← -(E_{h ~ Unif(H)} [X_h(s)^{-ξ}])^η
```

This allows stochastic approximation when there are many humans.

## 3. Neural Network Architecture

### 3.1 Overview of Networks

We need to learn approximations of these quantities:

| Quantity | Equation | Network Class | Input | Output |
|----------|----------|---------------|-------|--------|
| `Q_r(s, a_r)` | (4) | `NeuralRobotStateActionValue` | state `s` | Q-values for all `a_r` combinations |
| `π_r(s)` | (5) | `NeuralRobotPolicy` | state `s` | action probabilities |
| `V_h^e(s, g_h)` | (6) | `NeuralHumanGoalAchievementAbility` | state `s`, human `h`, goal `g_h` | scalar in [0, 1] |
| `X_h(s)` | (7) | `NeuralAggregateGoalAchievementAbility` | state `s`, human `h` | scalar > 0 |
| `U_r(s)` | (8) | `NeuralIntrinsicRobotReward` | state `s` | scalar < 0 |
| `V_r(s)` | (9) | `NeuralRobotStateValue` | state `s` | scalar < 0 |

### 3.2 NeuralRobotStateActionValue (Q_r)

**Approximates equation (4):** `Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]`

```python
class NeuralRobotStateActionValue(nn.Module):
    """
    Approximates Q_r(s, a_r) for all robot action combinations.
    
    Output has shape (batch, num_actions^num_robots) where each output
    is the Q-value for a specific combination of robot actions.
    
    Since Q_r < 0, we use SoftClamp with feasible_range=(-∞, 0) or similar.
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        num_actions: int,
        num_robots: int,
        hidden_dim: int = 256,
        beta_r: float = 1.0,  # For computing π_r
    ):
        # Output dimension = num_actions^num_robots
        self.num_action_combinations = num_actions ** num_robots
        
    def forward(self, state_encoding) -> torch.Tensor:
        """Returns Q_r values, shape (batch, num_action_combinations)"""
        # ... MLP head ...
        # Ensure Q_r < 0 (e.g., using -softplus or SoftClamp)
        
    def get_policy(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute π_r using power-law softmax (equation 5):
        π_r(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
        """
        # q_values are negative, so -q_values are positive
        neg_q = -q_values  # Shape: (batch, num_action_combinations)
        # Power-law: (-Q)^{-β} = 1 / (-Q)^β
        unnormalized = neg_q ** (-self.beta_r)
        return unnormalized / unnormalized.sum(dim=-1, keepdim=True)
```

### 3.3 NeuralRobotPolicy (π_r)

**Approximates equation (5):** `π_r(s)(a_r) ∝ (-Q_r(s, a_r))^{-β_r}`

Two modes (configurable):

1. **Derived mode**: Compute directly from `Q_r` network using the power-law formula
2. **Learned mode**: Separate network trained to match the derived policy

```python
class NeuralRobotPolicy(nn.Module):
    """
    Robot policy - either derived from Q_r or learned separately.
    """
    
    def __init__(
        self,
        q_network: NeuralRobotStateActionValue,
        mode: str = 'derived',  # 'derived' or 'learned'
        # ... network params for learned mode ...
    ):
        self.mode = mode
        self.q_network = q_network
        
    def forward(self, state_encoding) -> torch.Tensor:
        """Returns action probabilities, shape (batch, num_action_combinations)"""
        if self.mode == 'derived':
            q_values = self.q_network(state_encoding)
            return self.q_network.get_policy(q_values)
        else:
            # Learned policy network
            return self.policy_head(state_encoding)
```

### 3.4 NeuralHumanGoalAchievementAbility (V_h^e)

**Approximates equation (6):** `V_h^e(s, g_h) ← E[U_h(s', g_h) + γ_h V_h^e(s', g_h)]`

```python
class NeuralHumanGoalAchievementAbility(nn.Module):
    """
    Approximates V_h^e(s, g_h) - the probability that human h achieves
    goal g_h under the current robot policy.
    
    This is similar to MultiGridQNetwork from Phase 1, but:
    - Uses the actual robot policy π_r (not min_{a_r})
    - Output is bounded to [0, 1] (it's a probability)
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        goal_encoder: GoalEncoder,
        hidden_dim: int = 256,
    ):
        self.soft_clamp = SoftClamp(a=0.0, b=1.0)
        
    def encode_and_forward(
        self,
        state,
        world_model,
        human_agent_idx: int,
        goal,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Returns V_h^e(s, g_h), a scalar in [0, 1]"""
        # Encode state and goal
        # ... 
        # Output through soft_clamp to bound to [0, 1]
        return self.soft_clamp(raw_output)
```

### 3.5 NeuralAggregateGoalAchievementAbility (X_h)

**Approximates equation (7):** `X_h(s) ← E_{g_h}[V_h^e(s, g_h)^ζ]`

```python
class NeuralAggregateGoalAchievementAbility(nn.Module):
    """
    Approximates X_h(s) - the aggregate goal achievement ability.
    
    X_h > 0 always. We use softplus or similar to ensure positivity.
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        num_humans: int,
        hidden_dim: int = 256,
        zeta: float = 2.0,  # ζ - risk/reliability preference
    ):
        self.zeta = zeta
        
    def encode_and_forward(
        self,
        state,
        world_model,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Returns X_h(s), a positive scalar"""
        # ...
        # Ensure X_h > 0 (e.g., softplus)
        return F.softplus(raw_output)
```

### 3.6 NeuralIntrinsicRobotReward (U_r)

**Approximates equation (8):** `U_r(s) ← -(Σ_h X_h(s)^{-ξ})^η`

```python
class NeuralIntrinsicRobotReward(nn.Module):
    """
    Approximates U_r(s) - the robot's intrinsic reward.
    
    U_r < 0 always. The network predicts an intermediate value y,
    then computes U_r = -y^η.
    
    The intermediate y = E_h X_h(s)^{-ξ} is in [1, ∞) since:
    - X_h is in (0,1] (expected value of nonnegative discounted probabilities, at least one of which >0)
    - Hence X_h^{-ξ} >= 1 for any ξ > 0
    - When X_h is small (human has little power), X_h^{-ξ} is large
    
    We use log-space for y to handle the heavy tail when humans have low power.
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        hidden_dim: int = 256,
        xi: float = 1.0,   # ξ - inter-human inequality aversion
        eta: float = 1.1,  # η - intertemporal inequality aversion
    ):
        self.xi = xi
        self.eta = eta
        
    def forward(self, state_encoding) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (y, U_r) where:
        - y = E_h[X_h^{-ξ}] (intermediate value, > 1)
        - U_r = -y^η (final reward, < 0)
        """
        # Predict log(y) for numerical stability
        log_y_minus_1 = self.head(state_encoding)
        y = 1 + torch.exp(log_y_minus_1)  # y > 1
        U_r = -(y ** self.eta)  # U_r < 0
        return y, U_r
```

### 3.7 NeuralRobotStateValue (V_r)

**Approximates equation (9):** `V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)`

```python
class NeuralRobotStateValue(nn.Module):
    """
    Approximates V_r(s) - the robot's state value.
    
    V_r < 0 always (since U_r < 0 and Q_r < 0).
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        hidden_dim: int = 256,
    ):
        pass
        
    def encode_and_forward(
        self,
        state,
        world_model,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Returns V_r(s), a negative scalar"""
        # ...
        # Ensure V_r < 0 (e.g., -softplus)
        return -F.softplus(-raw_output)
```

## 4. Training Algorithm

### 4.1 Data Generation

Training data consists of tuples `(s, a_r, g, a_H, s')` generated from rollouts:

```python
@dataclass
class Phase2Transition:
    state: Any                      # s
    robot_action: Tuple[int, ...]   # a_r (one action per robot)
    goals: Dict[int, PossibleGoal]  # g = {h: g_h} for each human
    human_actions: List[int]        # a_H (from π_H)
    next_state: Any                 # s'
```

**Rollout procedure:**
1. Sample goals `g = {g_h}` for all humans using `PossibleGoalSampler`
2. At each step:
   - Sample robot action `a_r ~ π_r(s)` (with optional ε-exploration)
   - Sample human actions `a_H ~ π_H(s, g)` from given `human_policy_prior` (some heuristic policy or the one generated by Phase 1 learning)
   - Execute joint action, observe `s'`
   - Store transition
3. Resample goals when achieved or periodically (every `N_g` steps)

### 4.2 Training Targets

Based on the paper's Section 3.1, the training targets are:

**For Q_r (equation 12 in paper):**
```
target_q_r(s, a_r) = γ_r V_r(s')
```
Note: No immediate reward! Q_r is purely discounted continuation.

**For V_r:**
```
target_v_r(s) = U_r(s) + E_{a_r ~ π_r}[Q_r(s, a_r)]
```

**For V_h^e (equation 13 in paper):**
```
target_v_h_e(s, g_h) = U_h(s', g_h) + γ_h V_h^e(s', g_h)
```
where `U_h(s', g_h) = 1` if goal achieved, 0 otherwise.

**For X_h (equation 14 in paper):**
```
target_x_h(s) = V_h^e(s, g_h)^ζ
```
Using sampled goal `g_h` (Monte Carlo approximation of expected value).

**For y in U_r:**
```
target_y(s) = X_h(s)^{-ξ}
```
Using sampled human `h` (Monte Carlo approximation of expected value).

### 4.3 Loss Functions

```python
def compute_losses(self, batch: List[Phase2Transition]) -> Dict[str, torch.Tensor]:
    losses = {}
    
    for transition in batch:
        s, a_r, g, a_H, s_prime = transition
        
        # V_h^e loss (for each human and their goal)
        for h, g_h in g.items():
            v_h_e_pred = self.v_h_e_network(s, h, g_h)
            goal_achieved = g_h.is_achieved(s_prime)
            v_h_e_next = self.v_h_e_target(s_prime, h, g_h)
            target = goal_achieved + self.gamma_h * (1 - goal_achieved) * v_h_e_next
            losses['v_h_e'] += (v_h_e_pred - target) ** 2
        
        # X_h loss (sample one human)
        h = random.choice(self.human_agent_indices)
        g_h = self.goal_sampler.sample(s, h)
        x_h_pred = self.x_h_network(s, h)
        v_h_e = self.v_h_e_network(s, h, g_h)
        target = v_h_e ** self.zeta
        losses['x_h'] += (x_h_pred - target) ** 2
        
        # U_r loss based on y:
        y_pred, _unused_u_r_pred = self.u_r_network(s)
        x_h = self.x_h_network(s, h)  # Sample human
        # Monte Carlo: target = X_h^{-ξ}
        target = x_h ** -self.xi
        losses['u_r'] += (y_pred - target) ** 2  # Remark: squared loss on y is the only loss function that will make y converge to the expected value of the targets as required! Using log(y) would instead make log(y) converge to the expected value of the logs of the targets, in other words, it would make y wrongly converge to the geometric mean of the targets!
        
        # Q_r loss
        q_r_pred = self.q_r_network(s)[a_r_index]
        v_r_next = self.v_r_target(s_prime)
        target = self.gamma_r * v_r_next
        losses['q_r'] += (q_r_pred - target) ** 2
        
        # V_r loss
        v_r_pred = self.v_r_network(s)
        u_r = self.u_r_network(s)
        q_r = self.q_r_network(s)
        pi_r = self.robot_policy(s)
        expected_q = (pi_r * q_r).sum()
        target = u_r + expected_q
        losses['v_r'] += (v_r_pred - target) ** 2
    
    return {k: v / len(batch) for k, v in losses.items()}
```

### 4.4 Target Networks

Following Double DQN, we use frozen target networks for:
- `V_r` (used in Q_r target)
- `V_h^e` (used in V_h^e target)
- Optionally: `X_h`, `U_r`

Target networks are updated every `V_r_target_update_freq` or `V_h_target_update_freq` steps (possibly different).

### 4.5 Loss Function Alternatives for Heavy-Tailed Targets

The `y` target in `U_r` (equation 8) can have a heavy tail since `y = E_h[X_h^{-ξ}]` and `X_h^{-ξ}` becomes very large when a human has low power (`X_h` close to 0). This raises the question of which loss function to use.

**Why MSE is the correct choice:**

We use MSE `(y_pred - target)^2` because it converges to the **arithmetic mean** (expected value) of the targets, which is exactly what equation (8) requires. Alternative loss functions have different convergence properties:

| Loss Function | Converges To | Heavy-Tail Behavior |
|---------------|--------------|---------------------|
| MSE `(y - target)²` | Arithmetic mean | High gradient from outliers |
| Log-MSE `(log y - log target)²` | Geometric mean | Robust but **wrong target** |
| MAE `|y - target|` | Median | Robust but **wrong target** |
| Huber loss | Between mean and median | Robust but **biased** |

**Why not Huber loss?**

Huber loss is often recommended for heavy-tailed data because it reduces the influence of outliers:
- For small errors (|error| < δ): quadratic like MSE
- For large errors (|error| ≥ δ): linear like MAE

However, Huber loss converges to a value **between the mean and median**, depending on δ. Since our goal is specifically to approximate the **expected value** (arithmetic mean), Huber loss would introduce systematic bias - underestimating `y` when the target distribution is right-skewed (which it is, due to the heavy right tail).

**Recommended approach:**

1. **Use MSE** for correctness (converges to expected value)
2. **Mitigate heavy-tail issues via:**
   - Gradient clipping to stabilize training
   - The network architecture already helps: predicting `log(y-1)` internally provides numerical stability while computing the loss on actual `y` values
   - Learning rate scheduling if training is unstable
   - Batch normalization / larger batch sizes to average out extreme samples

**Alternative if MSE proves unstable:**

If training instability persists, consider a **weighted MSE** that down-weights extreme targets:
```python
weight = 1.0 / (1.0 + alpha * (target - target.mean()).abs())
loss = weight * (y_pred - target) ** 2
```
This preserves convergence to the arithmetic mean while reducing gradient magnitude from outliers.

## 5. Configuration

```python
@dataclass
class Phase2Config:
    # Discount factors
    gamma_r: float = 0.99  # Robot discount
    gamma_h: float = 0.99  # Human discount (for V_h^e)
    
    # Power metric parameters
    zeta: float = 2.0   # ζ - risk/reliability preference (>=1, 1 meaning neutrality)
    xi: float = 1.0     # ξ - inter-human inequality aversion (>=1 to protect last bit of power of every human)
    eta: float = 1.1    # η - additional intertemporal inequality aversion factor (>=1, 1 meaning neutrality)
    
    # Robot policy
    beta_r: float = 10.0  # Robot power-law policy exponent (<infinity to prevent risks from overoptimization)
    
    # Additional exploration (in addition to power-law policy-induced randomization due to finite beta_r)
    epsilon_r_start: float = 1.0
    epsilon_r_end: float = 0.01
    epsilon_r_decay_steps: int = 10000
    
    # Learning rates (per network, these might have to be adjusted significantly to achieve time-scale separation if oscillations occur due to moving target effect)
    lr_q_r: float = 1e-3
    lr_v_r: float = 1e-3
    lr_v_h_e: float = 1e-3
    lr_x_h: float = 1e-3
    lr_u_r: float = 1e-3
    
    # Target network update
    target_update_freq: int = 100
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    
    # Training
    num_episodes: int = 10000
    steps_per_episode: int = 50
    updates_per_step: int = 1
    
    # Goal resampling
    goal_resample_prob: float = 0.01  # p_g from paper
```

## 6. Code Organization

```
src/empo/nn_based/
├── phase2/                           # Environment-agnostic Phase 2 base classes
│   ├── __init__.py
│   ├── config.py                     # Phase2Config
│   ├── robot_q_network.py            # BaseRobotStateActionValue
│   ├── robot_policy.py               # BaseRobotPolicy  
│   ├── human_goal_ability.py         # BaseHumanGoalAchievementAbility
│   ├── aggregate_ability.py          # BaseAggregateGoalAchievementAbility
│   ├── intrinsic_reward.py           # BaseIntrinsicRobotReward
│   ├── robot_value.py                # BaseRobotStateValue
│   ├── trainer.py                    # Phase2Trainer
│   └── replay_buffer.py              # Phase2ReplayBuffer
│
├── multigrid/
│   └── phase2/                       # Multigrid-specific Phase 2 implementations
│       ├── __init__.py
│       ├── robot_q_network.py        # MultiGridRobotStateActionValue
│       ├── robot_policy.py           # MultiGridRobotPolicy
│       ├── human_goal_ability.py     # MultiGridHumanGoalAchievementAbility
│       ├── aggregate_ability.py      # MultiGridAggregateGoalAchievementAbility
│       ├── intrinsic_reward.py       # MultiGridIntrinsicRobotReward
│       ├── robot_value.py            # MultiGridRobotStateValue
│       └── trainer.py                # train_multigrid_phase2()
```

## 7. Example Script

```python
# examples/phase2_multigrid_demo.py

from empo.nn_based.multigrid.phase2 import train_multigrid_phase2
from empo.human_policy_prior import HeuristicPotentialPolicy

# Load or create Phase 1 human policy
human_policy = HeuristicPotentialPolicy(env, human_indices, path_calc)

# Train Phase 2
robot_policy, networks = train_multigrid_phase2(
    world_model=env,
    human_agent_indices=human_indices,
    robot_agent_indices=robot_indices,
    human_policy_prior=human_policy,
    goal_sampler=goal_sampler,
    config=Phase2Config(
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,
        xi=1.0,
        eta=1.1,
        beta_r=1.0,
        num_episodes=10000,
    ),
)
```

## 8. Testing Strategy

1. **Unit tests**: Each network produces correct output shapes and bounds
2. **Integration tests**: Trainer can run one episode without errors
3. **Convergence tests**: Losses decrease over training
4. **Behavioral tests**: Trained robot policy improves human goal achievement vs random policy

## 9. Open Questions

1. **Scaling with robot count**: `num_actions^num_robots` grows exponentially. May need factorized representations for >3 robots.

2. **Heavy-tailed U_r**: The `X_h^{-ξ}` term can be very large when `X_h` is small. Log-space training may be needed.

3. **Convergence**: The circular dependencies mean this is not a contraction. May need careful learning rate scheduling.

## 10. References

- Issue #49: https://github.com/mensch72/empo/issues/49
- EMPO paper (attached to issue)
- Phase 1 implementation: `src/empo/nn_based/multigrid/`

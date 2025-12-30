# Planning Document: Dueling Architecture for Q_r Network

This document describes how to optionally replace the monolithic Q_r network with a dueling architecture (V_r + A_r) adapted to the EMPO framework.

## Background

### Standard Dueling DQN

In the standard dueling DQN architecture (Wang et al., 2016), the Q-function is decomposed as:

```
Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]
```

Where:
- **V(s)**: State value - expected return starting from state s
- **A(s, a)**: Advantage - how much better action a is than the average action
- The mean subtraction ensures identifiability (V is uniquely determined)

The key insight is that many states have similar values regardless of action taken, so separating V from A allows the network to learn state values more efficiently.

### EMPO's Q_r Equations

In EMPO (equations 4-9), the robot's value functions are:

```
(4) Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]
(5) π_r(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
(9) V_r(s) = U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)
```

Key properties:
- **Q_r < 0** always (since V_r < 0)
- **V_r < 0** always (since U_r < 0 from negative power metric)
- Q_r represents discounted future value only (no immediate reward term)
- V_r includes immediate intrinsic reward U_r plus expected future value

## Motivation for Dueling Architecture in EMPO

### Why This Might Help

1. **Shared state representation**: In many states, the robot's action has limited impact on human empowerment (e.g., when far from humans). Learning V_r separately allows the network to efficiently capture these common cases.

2. **Disentangled learning**: V_r depends on the overall "quality" of a state for human empowerment, while A_r captures action-specific deviations. This separation may improve training stability.

3. **Better generalization**: States with similar human configurations should have similar V_r values, even if optimal actions differ.

4. **Faster policy improvement**: The advantage function directly captures which actions are better, potentially accelerating policy learning.

### When This Might Not Help

1. **Small action spaces**: With few joint actions (e.g., A^K for K robots with A actions each), the advantage function has limited dimensionality.

2. **Action-dependent dynamics**: If different robot actions lead to dramatically different next states (unlike Atari where many actions have similar effects), the decomposition may be less beneficial.

3. **Already-separate V_r**: EMPO already has a separate V_r computation (equation 9). The dueling architecture would create a second V_r-like component within Q_r.

## Proposed Architecture

### Option A: Pure Dueling (Replace Q_r Head)

Replace the Q_r output head with separate V_r' and A_r streams:

```python
class DuelingMultiGridRobotQNetwork(BaseRobotQNetwork):
    def __init__(self, ...):
        # ... state encoder setup ...
        
        # Value stream (state-only, scalar output)
        self.v_stream = nn.Sequential(
            nn.Linear(combined_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream (state-to-actions)
        self.a_stream = nn.Sequential(
            nn.Linear(combined_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_combinations)
        )
    
    def forward(self, ...):
        # Encode state
        state_features = self.encode_state(...)
        
        # Compute V and A
        v = self.v_stream(state_features)  # (batch, 1)
        a = self.a_stream(state_features)  # (batch, num_actions)
        
        # Combine with mean-centering for identifiability
        # Q = V + (A - mean(A))
        q_raw = v + (a - a.mean(dim=-1, keepdim=True))
        
        # Ensure Q_r < 0 (EMPO-specific constraint)
        q_values = self.ensure_negative(q_raw)
        
        return q_values
```

**Key adaptations for EMPO**:
- Apply `ensure_negative()` after combining V and A
- The V stream learns the average negative Q-value across actions
- The A stream learns relative differences between actions

### Option B: Dueling with EMPO's V_r

Leverage EMPO's existing V_r computation (equation 9):

```
V_r(s) = U_r(s) + E_{a_r ~ π_r}[Q_r(s, a_r)]
```

This suggests a different decomposition:

```
Q_r(s, a_r) = E_{π_r}[Q_r(s, ·)] + A_r(s, a_r)
            = (V_r(s) - U_r(s)) + A_r(s, a_r)
```

Where `A_r(s, a_r)` represents how much action `a_r` deviates from the policy-weighted average.

**Implementation**:
```python
class EMPODuelingRobotQNetwork(BaseRobotQNetwork):
    def __init__(self, ...):
        # Advantage stream only
        self.a_stream = nn.Sequential(
            nn.Linear(combined_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_combinations)
        )
        
        # Reference to V_r and U_r networks/computations
        self.v_r_network = None  # Set during Phase2Networks construction
        self.u_r_computer = None  # Or U_r network
    
    def forward(self, state_features, v_r, u_r):
        """
        Args:
            state_features: Encoded state
            v_r: V_r(s) from equation 9 or V_r network
            u_r: U_r(s) from equation 8 or U_r network
        """
        # Expected Q under policy = V_r - U_r
        expected_q = v_r - u_r  # (batch, 1), negative
        
        # Advantage (mean-centered)
        a = self.a_stream(state_features)  # (batch, num_actions)
        a_centered = a - a.mean(dim=-1, keepdim=True)
        
        # Q = expected_Q + advantage
        q_values = expected_q + a_centered
        
        return q_values  # Already negative if expected_q is sufficiently negative
```

**Challenge**: This creates a circular dependency since V_r depends on Q_r (eq. 9). Solutions:
1. Use target network values for V_r/U_r
2. Use previous iteration's estimates
3. Train in alternating phases

### Option C: Independent Dueling Streams with Shared State Encoder

Keep the dueling architecture self-contained but share the state encoder with other Phase 2 networks:

```python
class IndependentDuelingRobotQNetwork(BaseRobotQNetwork):
    def __init__(self, state_encoder, ...):
        self.state_encoder = state_encoder  # Shared
        
        # Own feature processing
        self.q_feature_layer = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate streams from processed features
        self.v_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.a_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_action_combinations)
        )
```

## Recommended Approach

**Option A (Pure Dueling)** is recommended as the initial implementation because:

1. **Simplest integration**: Drop-in replacement for Q_r head, no changes to training loop
2. **No circular dependencies**: Self-contained computation
3. **Proven architecture**: Well-established in RL literature
4. **Compatible with existing warm-up**: No changes to staged training

## Configuration

Add configuration option to `Phase2Config`:

```python
@dataclass
class Phase2Config:
    # ... existing fields ...
    
    # Dueling architecture
    q_r_dueling: bool = False  # Enable dueling V_r + A_r decomposition
    q_r_dueling_aggregation: str = "mean"  # "mean" or "max" for advantage centering
```

## Implementation Checklist

### Phase 1: Base Infrastructure

1. [ ] Create `BaseDuelingRobotQNetwork` in `src/empo/nn_based/phase2/robot_q_network.py`
   - Add `v_stream` and `a_stream` abstract methods
   - Implement `combine_v_a()` method with mean-centering
   - Ensure `ensure_negative()` is applied correctly

2. [ ] Add `q_r_dueling` config option to `Phase2Config`

3. [ ] Update `BasePhase2Networks` to conditionally create dueling network

### Phase 2: MultiGrid Implementation

4. [ ] Create `MultiGridDuelingRobotQNetwork` in `src/empo/nn_based/multigrid/phase2/robot_q_network.py`
   - Inherit from both `BaseDuelingRobotQNetwork` and reuse MultiGrid encoding
   - Handle shared vs. own state encoders for both streams

5. [ ] Update `MultiGridPhase2Networks` factory to handle dueling option

### Phase 3: Training Integration

6. [ ] Verify loss computation works unchanged (since output is still Q-values)

7. [ ] Add optional separate learning rates for V and A streams
   - `q_r_v_stream_lr` and `q_r_a_stream_lr` config options (optional)

8. [ ] Update checkpointing to handle new architecture

### Phase 4: Testing and Validation

9. [ ] Add unit tests for dueling Q-value computation
   - Verify Q_r < 0 constraint is maintained
   - Verify identifiability (mean-centering works)

10. [ ] Add integration test comparing dueling vs. non-dueling

11. [ ] Benchmark training stability and convergence

### Phase 5: Documentation

12. [ ] Update `docs/API.md` with new config options

13. [ ] Add example demonstrating dueling architecture

14. [ ] Document when to use dueling vs. standard architecture

## Alternative Considerations

### Max vs. Mean Aggregation

Standard dueling DQN uses mean-centering:
```
Q = V + (A - mean(A))
```

An alternative uses max:
```
Q = V + (A - max(A))
```

Max aggregation ensures the greedy action has advantage = 0, which may be more interpretable. Both should be supported via config.

### Gradient Flow Concerns

With `ensure_negative()` applied after combining V + A:
- Gradients flow through both streams
- The softplus transformation may cause gradient scaling issues
- Consider applying separate constraints to V and A streams:
  - V stream: `ensure_negative()` (V should be average Q, which is negative)
  - A stream: unconstrained (can be positive or negative, averages to 0)

### Action-Conditional Advantage

For multi-robot scenarios with joint action space A^K, consider:
- **Factorized advantage**: `A(s, a_r) = sum_k A_k(s, a_{r_k})`
- Reduces parameters from O(A^K) to O(K*A)
- May lose expressiveness for action interactions

## Related Work

- Wang et al. (2016): "Dueling Network Architectures for Deep Reinforcement Learning"
- Schaul et al. (2016): Prioritized Experience Replay (compatible with dueling)
- Hessel et al. (2018): Rainbow DQN (combines dueling with other improvements)

## Questions for Further Investigation

1. How does dueling interact with EMPO's power-law policy (eq. 5)?
   - The policy uses `(-Q_r)^{-β_r}`, which transforms Q-values non-linearly
   - Does advantage centering interact poorly with this transformation?

2. Should advantage centering happen before or after `ensure_negative()`?
   - Before: Preserves standard dueling formulation
   - After: Ensures all Q-values are properly bounded

3. Can we share the V stream with the existing V_r network?
   - Would reduce parameters but create training dependencies
   - May require alternating optimization

4. Is there benefit to having both dueling Q_r AND a separate V_r network?
   - Redundancy may help with target stability
   - Or may cause inconsistencies between the two V estimates

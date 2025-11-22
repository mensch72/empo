# Multigrid State Transitions: When Are They Probabilistic?

## Overview

This document explains when state transitions in multigrid environments are probabilistic versus deterministic, and how the `transition_probabilities()` method computes exact probabilities.

## TL;DR

**Transitions are probabilistic ONLY when:**
- 2 or more agents are active (not terminated/paused, started=True)
- Those agents choose non-"still" actions
- The order in which agents act affects the outcome (e.g., competing for same cell)

**Otherwise, transitions are deterministic.**

## The Single Source of Non-Determinism

In the multigrid environment, there is **exactly one** source of randomness in state transitions:

```python
# From multigrid.py, line 1257:
order = np.random.permutation(len(actions))
```

This line creates a random permutation of agent indices to determine the order in which agents execute their actions within a single timestep. Each of the n! possible orderings has equal probability 1/n!.

All individual actions (left, right, forward, pickup, drop, toggle, done) are themselves completely deterministic - there is no randomness in their effects.

## When Transitions Are Deterministic

Transitions are **deterministic** (single outcome with probability 1.0) in these cases:

### 1. Zero or One Active Agent
When 0 or 1 agents are active, order doesn't matter:
```python
actions = [still, still, still]  # No agents act
# OR
actions = [forward, still, still]  # Only one agent acts
```
**Result:** Deterministic transition

### 2. All Agents Do Rotations
Rotation actions (left/right) are commutative - they never interfere:
```python
actions = [left, right, left]  # All rotations
```
**Result:** Deterministic transition (agents just change direction)

### 3. Agents Don't Interact
When agents are far apart and their actions don't affect each other:
```python
# Agent 0 at (1,1) moves forward to (2,1)
# Agent 1 at (5,5) moves forward to (6,5)
# These actions don't interfere
```
**Result:** Deterministic transition (order doesn't matter)

## When Transitions Are Probabilistic

Transitions can be **probabilistic** (multiple possible outcomes) when agent actions interact:

### 1. Competing for Same Cell
Two agents try to move into the same empty cell:
```python
# Agent 0 at (1,1) facing right
# Agent 1 at (3,1) facing left
# Both move forward to (2,1)
actions = [forward, forward, still]
```
**Result:** 2 possible outcomes
- Outcome A (prob 1/2): Agent 0 moves first, gets cell, Agent 1 blocked
- Outcome B (prob 1/2): Agent 1 moves first, gets cell, Agent 0 blocked

### 2. Competing for Same Object
Two agents try to pick up the same object:
```python
# Ball at (2,1)
# Agent 0 at (1,1) facing right
# Agent 1 at (3,1) facing left
actions = [pickup, pickup, still]
```
**Result:** 2 possible outcomes
- Outcome A (prob 1/2): Agent 0 picks up ball, Agent 1 gets nothing
- Outcome B (prob 1/2): Agent 1 picks up ball, Agent 0 gets nothing

### 3. Agent Interactions
Agents can transfer objects between each other:
```python
# Agent 0 carrying ball, next to Agent 1
actions = [drop, pickup, still]
```
**Result:** Order matters - different outcomes depending on who acts first

### 4. Sequential Dependencies
One agent's action changes the environment for another:
```python
# Agent 0 next to door, Agent 1 behind door
actions = [toggle, forward, still]  # Agent 0 opens door, Agent 1 moves through
```
**Result:** If Agent 1 acts first (tries to move before door opens), different outcome

## Computing Exact Probabilities

The `transition_probabilities(state, actions)` method computes **exact** probabilities through exhaustive enumeration, not sampling:

### Algorithm

1. **Identify active agents**: Which agents will actually act (not terminated/paused/still)

2. **Early exit for deterministic cases**:
   - ≤1 active agent → return single outcome with prob 1.0
   - All rotations → return single outcome with prob 1.0

3. **Generate permutations**: Create all k! orderings of k active agents

4. **Compute each outcome**: For each ordering, deterministically compute the successor state

5. **Aggregate probabilities**: Count how many orderings lead to each unique state
   ```python
   probability = count / k!
   ```

6. **Return results**: List of (probability, successor_state) pairs

### Example

With 3 agents all acting (forward, forward, forward):
- Total orderings: 3! = 6
- If 4 orderings lead to state A and 2 to state B:
  - State A: probability 4/6 = 0.667
  - State B: probability 2/6 = 0.333

## Optimization Strategy

The implementation is optimized for the common case where most orderings produce the same result:

### Optimizations Applied

1. **Early deterministic detection**: Check for ≤1 active agent or all-rotations
2. **Active-only permutations**: Only permute active agents (k! instead of n!)
3. **Sampling check**: Test a sample of orderings to detect likely deterministic cases
4. **Early exit**: If 25% of orderings all produce same state, likely all do
5. **Efficient state hashing**: Use hashable state tuples for fast duplicate detection

### Performance

- **Best case** (deterministic): O(1) - single state computation
- **Common case** (mostly same result): O(k) - early exit after sampling
- **Worst case** (truly probabilistic): O(k! × S) where S is state computation cost

For typical scenarios with k=3 agents, worst case is 6 state computations.

## Examples

### Example 1: Deterministic (Rotations)
```python
state = env.get_state()
actions = [Actions.left, Actions.right, Actions.left]
result = env.transition_probabilities(state, actions)
# Result: [(1.0, successor_state)]
```

### Example 2: Deterministic (One Agent)
```python
state = env.get_state()
actions = [Actions.forward, Actions.still, Actions.still]
result = env.transition_probabilities(state, actions)
# Result: [(1.0, successor_state)]
```

### Example 3: Probabilistic (Competing Agents)
```python
# Setup: Two agents facing same empty cell
state = env.get_state()
actions = [Actions.forward, Actions.forward, Actions.still]
result = env.transition_probabilities(state, actions)
# Result might be:
# [(0.5, state_A),  # Agent 0 moves first
#  (0.5, state_B)]  # Agent 1 moves first
```

## Testing

Comprehensive tests verify:
- ✓ Probabilities sum to exactly 1.0
- ✓ Each state appears exactly once
- ✓ Deterministic cases return single outcome
- ✓ State unchanged after calling method
- ✓ Probabilities are rational multiples of 1/k!
- ✓ All states are hashable and immutable

Run tests:
```bash
python tests/test_state_management.py
python tests/test_probability_computation.py
```

## Summary

**Key Takeaways:**
1. Only one source of randomness: agent execution order
2. Most transitions are deterministic in practice
3. Probabilities are computed exactly, not sampled
4. Implementation is optimized for common deterministic cases
5. Each unique successor state appears once with aggregated probability

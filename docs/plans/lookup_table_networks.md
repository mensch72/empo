# Lookup Table Networks: Design Plan for Tabular Policy Representations

**Status:** Planning  
**Author:** GitHub Copilot  
**Date:** 2025-12-30

## 0. mensch72's comments after reading this plan

We must make sure that we need to change *as little code outside the network classes as possible*, including the optimizers. So maybe a pragmatic approach is to recreate the optimizer periodically at a configurable interval, in particular at warmup stage boundaries.

Also, we must reuse the existing code for `use_encoders==False` that make the encoders act as identity functions, rather than adding additional code that basically does the same.

## 1. Overview

This document outlines a design for optionally converting individual networks and their private encoders into **lookup tables** (dictionaries) with one parameter per unique input. This approach enables tabular policy representations without requiring advance knowledge of the complete state/input space.

### 1.1 Motivation

**Why lookup tables?**

1. **No function approximation error** - Exact value storage for each observed state
2. **Guaranteed convergence** - Tabular methods have stronger theoretical guarantees
3. **Interpretability** - Direct inspection of learned values
4. **Debugging** - Easier to understand what the system has learned
5. **Small state spaces** - Optimal for environments with limited state/action combinations
6. **No neural network overhead** - Skip gradient computation, backpropagation for tabular components

**When NOT to use lookup tables:**

- Large or continuous state spaces (memory explosion)
- Generalization needed to unseen states
- When function approximation is beneficial

### 1.2 Scope

This design allows **selective** conversion of individual networks to lookup tables:

- **Phase 1 (Human Policy Prior):**
  - `BaseQNetwork` → Lookup table mapping `(state, goal)` to Q-values
  - `BasePolicyPriorNetwork` → Lookup table mapping `state` to marginal policy
  
- **Phase 2 (Robot Policy):**
  - `BaseRobotQNetwork` → Lookup table mapping `state` to robot Q-values
  - `BaseRobotValueNetwork` → Lookup table mapping `state` to V_r values
  - `BaseHumanGoalAbilityNetwork` (V_h^e) → Lookup table mapping `(state, goal)` to V_h^e
  - `BaseAggregateGoalAbilityNetwork` (X_h) → Lookup table mapping `(state, agent_idx)` to X_h
  - `BaseIntrinsicRewardNetwork` (U_r) → Lookup table mapping `state` to U_r

Each network can independently be neural or tabular, allowing hybrid architectures (e.g., tabular Q_r with neural V_h^e).

---

## 2. Design Principles

### 2.1 Zero Code Changes Outside Network Classes

**Critical requirement:** The trainer, replay buffer, and all other code should work identically whether a network is neural or tabular.

This means:
- Same API (`forward()`, `encode_and_forward()`, etc.)
- Same input/output shapes (batched tensors)
- Same device handling (CPU/GPU)
- Same optimizer compatibility (parameters still exist, just as lookup entries)

### 2.2 Dict-Based Implementation

Use Python `dict` (or `torch.nn.ParameterDict` for autodiff compatibility) keyed by **hashable inputs**:

```python
# Conceptual structure
class LookupTableQNetwork(BaseQNetwork):
    def __init__(self, num_actions, default_value=0.0, ...):
        super().__init__(num_actions, ...)
        # Store as ParameterDict for optimizer compatibility
        self.table = {}  # key: hash(input) -> value: Parameter(tensor)
        self.default_value = default_value
```

**Key design choice:** Use `dict` instead of `torch.nn.ParameterDict` for the main table, but create `torch.nn.Parameter` objects for each entry. This allows:
- Automatic gradient tracking
- Optimizer updates (Adam, SGD, etc. can update individual entries)
- No need to pre-allocate the full state space

### 2.3 Encoders Become Identity Functions

For lookup table networks:
- **State/goal encoders**: Return the raw input **as a tensor** (no encoding needed)
- **Tensorizers**: Convert hashable input → tensor representation (for batching)
- **Forward pass**: Uses tensor as lookup key after hashing

```python
class IdentityStateEncoder(BaseStateEncoder):
    def __init__(self):
        super().__init__(feature_dim=0)  # No features, direct lookup
    
    def tensorize_state(self, state, world_model, device='cpu'):
        # Convert state tuple to tensor for batching
        # This is NOT encoding, just preparing for dict lookup
        return self._state_to_tensor(state, device)
    
    def forward(self, state_tensor):
        # Identity: just return the input
        # (In practice, this isn't called—lookup happens directly in network)
        return state_tensor
```

---

## 3. Detailed Implementation Strategy

### 3.1 Hashable Keys

**Challenge:** PyTorch tensors are not hashable by default.

**Solution:** Convert inputs to hashable keys:

```python
def _tensorize_and_hash(state, goal=None):
    """Convert state/goal to hashable key."""
    # Assumes state is already hashable (tuple, frozenset, etc.)
    # For goals, use their __hash__ method
    if goal is None:
        return hash(state)
    else:
        return hash((state, goal))

def _key_to_tensor(key, device='cpu'):
    """Convert hash key back to tensor representation for batching."""
    # This is domain-specific
    # For MultiGrid: state is tuple of (grid, agents, objects, ...)
    # Convert to compact tensor representation
    pass
```

**Design decision:** Require states to be **pre-hashable** (already enforced by `WorldModel.get_state()` interface which returns `Hashable`). This means:
- `state` must be a tuple/frozenset/other hashable type
- `goal` must implement `__hash__` (already required by `PossibleGoal`)

### 3.2 Batch Processing Strategy

**Challenge:** Lookup tables are inherently single-input. How do we support batching?

**Solution 1: Vectorized Dictionary Lookup (Recommended)**

```python
def forward(self, state_tensors: torch.Tensor) -> torch.Tensor:
    """
    Process a batch of states.
    
    Args:
        state_tensors: (batch_size, *state_shape) - batched states
    
    Returns:
        values: (batch_size, output_dim) - batched values
    """
    batch_size = state_tensors.shape[0]
    output_dim = self.num_actions  # or other output dimension
    
    # Pre-allocate output tensor
    output = torch.full(
        (batch_size, output_dim),
        self.default_value,
        device=state_tensors.device,
        dtype=torch.float32
    )
    
    # Iterate through batch and lookup each state
    for i in range(batch_size):
        state_key = self._tensor_to_key(state_tensors[i])
        
        if state_key in self.table:
            # Use stored parameter (enables gradient tracking)
            output[i] = self.table[state_key]
        else:
            # Create new parameter with default value
            param = torch.nn.Parameter(
                torch.full((output_dim,), self.default_value, dtype=torch.float32)
            )
            self.table[state_key] = param
            output[i] = param
    
    return output
```

**Key insight:** Even though we loop over batch elements, each lookup:
1. Returns a `Parameter` (gradient-tracked)
2. Gets stacked into output tensor
3. Gradients flow back to individual parameters via autograd

**Performance:** This is efficient because:
- Dict lookup is O(1)
- Parameter creation is lazy (only when state is first seen)
- Gradient computation is automatic (no manual bookkeeping)

**Solution 2: Pre-gather Indices (Alternative)**

For very large batches, we could optimize further:

```python
def forward(self, state_tensors: torch.Tensor) -> torch.Tensor:
    # Step 1: Convert all states to keys (vectorized if possible)
    keys = [self._tensor_to_key(state_tensors[i]) for i in range(len(state_tensors))]
    
    # Step 2: Gather parameters for all keys
    params = []
    for key in keys:
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(...)
        params.append(self.table[key])
    
    # Step 3: Stack into single tensor
    return torch.stack(params, dim=0)
```

This avoids explicit indexing but has the same complexity.

### 3.3 Default Values for Unseen States

When a state is encountered for the first time:

```python
# Option 1: Zero initialization
default_value = 0.0

# Option 2: Optimistic initialization (encourages exploration)
default_value = 1.0  # For Q-values where higher is better

# Option 3: Domain-specific initialization
# For Q_r (always negative): default_value = -1.0
# For V_h^e (probabilities): default_value = 0.5
# For X_h (sum of probabilities): default_value = num_goals * 0.5
```

**Recommendation:** Make `default_value` a constructor parameter, with domain-specific defaults in each concrete implementation.

### 3.4 Handling Different Input Types

Different networks have different input signatures:

#### Phase 1 Q-Network: (state, goal) → Q-values

```python
class LookupTableQNetwork(BaseQNetwork):
    def encode_and_forward(self, state, world_model, query_agent_idx, goal, device='cpu'):
        # Create composite key
        key = hash((state, goal))
        
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(
                torch.full((self.num_actions,), self.default_q, dtype=torch.float32)
            )
        
        return self.table[key].unsqueeze(0).to(device)  # (1, num_actions)
```

#### Phase 2 Robot Q-Network: state → Q_r-values

```python
class LookupTableRobotQNetwork(BaseRobotQNetwork):
    def encode_and_forward(self, state, world_model, device='cpu'):
        key = hash(state)
        
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(
                torch.full((self.num_action_combinations,), self.default_q_r, dtype=torch.float32)
            )
        
        return self.table[key].unsqueeze(0).to(device)  # (1, num_action_combinations)
```

#### Phase 2 V_h^e Network: (state, goal) → probability

```python
class LookupTableHumanGoalAbilityNetwork(BaseHumanGoalAbilityNetwork):
    def encode_and_forward(self, state, goals, world_model, device='cpu'):
        # goals is a list of goals, one per human
        # This is more complex—need to handle multiple goals
        
        # Option A: Separate table entry per (state, goal_h) combination
        batch_size = len(goals)
        output = torch.zeros(batch_size, device=device)
        
        for i, goal_h in enumerate(goals):
            key = hash((state, goal_h))
            if key not in self.table:
                self.table[key] = torch.nn.Parameter(
                    torch.tensor([self.default_v_he], dtype=torch.float32)
                )
            output[i] = self.table[key]
        
        return output.unsqueeze(0)  # (1, num_humans)
```

### 3.5 Optimizer Compatibility

**Critical:** Lookup table parameters must be compatible with standard PyTorch optimizers.

```python
# In trainer setup:
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# For lookup table networks:
class LookupTableQNetwork(BaseQNetwork):
    def parameters(self):
        """Return iterator over all table entries (Parameters)."""
        return iter(self.table.values())
```

**Important:** New entries added during training must be registered:

```python
# When creating new entry:
new_param = torch.nn.Parameter(torch.full(...))
self.table[key] = new_param

# Optimizer will NOT automatically see this parameter!
# Solution: Re-create optimizer param groups when table grows

# Better solution: Use optimizer that can handle dynamic parameters
# OR: Add new params to existing optimizer
optimizer.add_param_group({'params': [new_param]})
```

**Design choice:** For simplicity, use **dictionary-based optimizer** that looks up parameters fresh each time:

```python
class LookupTableOptimizer:
    """Optimizer wrapper that dynamically handles new parameters."""
    
    def __init__(self, network, base_optimizer_class, **optimizer_kwargs):
        self.network = network
        self.base_optimizer_class = base_optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.state = {}  # optimizer state per parameter key
    
    def step(self):
        # Get current parameters (may include new ones)
        params = list(self.network.parameters())
        
        # For each parameter, update using stored state
        for param in params:
            param_id = id(param)
            if param_id not in self.state:
                self.state[param_id] = self._init_state(param)
            
            # Apply update (e.g., Adam step)
            self._apply_update(param, self.state[param_id])
```

**Alternative (simpler):** Just recreate optimizer periodically or use SGD (which has minimal state).

---

## 4. Architecture Details

### 4.1 Base Classes

Extend existing base classes with lookup table variants:

```python
# src/empo/nn_based/lookup/q_network.py

class LookupTableQNetwork(BaseQNetwork):
    """
    Lookup table implementation of Q-network.
    
    Stores Q-values in a dictionary keyed by (state, goal) hash.
    
    Args:
        num_actions: Number of possible actions.
        beta: Temperature for Boltzmann policy.
        default_q: Initial Q-value for unseen states.
        feasible_range: Optional Q-value bounds.
    """
    
    def __init__(
        self,
        num_actions: int,
        beta: float = 1.0,
        default_q: float = 0.0,
        feasible_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(num_actions, beta, feasible_range)
        self.default_q = default_q
        self.table: Dict[int, torch.nn.Parameter] = {}
        
        # Identity encoder (no encoding needed)
        self.state_encoder = None  # Not used
        self.goal_encoder = None   # Not used
    
    def forward(
        self,
        state_keys: torch.Tensor,  # (batch, ) tensor of hash keys
        goal_keys: torch.Tensor    # (batch, ) tensor of hash keys
    ) -> torch.Tensor:
        """
        Batch forward pass.
        
        Args:
            state_keys: (batch_size,) integer tensor of state hash keys
            goal_keys: (batch_size,) integer tensor of goal hash keys
        
        Returns:
            Q-values: (batch_size, num_actions)
        """
        batch_size = state_keys.shape[0]
        device = state_keys.device
        
        # Pre-allocate output
        output = torch.zeros(
            (batch_size, self.num_actions),
            device=device,
            dtype=torch.float32
        )
        
        # Lookup each (state, goal) pair
        for i in range(batch_size):
            key = self._combine_keys(
                state_keys[i].item(),
                goal_keys[i].item()
            )
            
            if key not in self.table:
                # Create new entry with default value
                self.table[key] = torch.nn.Parameter(
                    torch.full(
                        (self.num_actions,),
                        self.default_q,
                        dtype=torch.float32
                    )
                )
            
            output[i] = self.table[key].to(device)
        
        # Apply soft clamping if needed
        return self.apply_soft_clamp(output)
    
    def encode_and_forward(
        self,
        state: Hashable,
        world_model: Any,
        query_agent_idx: int,
        goal: PossibleGoal,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Single-state forward pass (used during inference)."""
        key = hash((state, goal))
        
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(
                torch.full((self.num_actions,), self.default_q, dtype=torch.float32)
            )
        
        return self.table[key].unsqueeze(0).to(device)
    
    def _combine_keys(self, state_hash: int, goal_hash: int) -> int:
        """Combine state and goal hashes into single key."""
        return hash((state_hash, goal_hash))
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'type': 'lookup_table',
            'num_actions': self.num_actions,
            'beta': self.beta,
            'default_q': self.default_q,
            'feasible_range': self.feasible_range,
            'table_size': len(self.table)
        }
    
    def parameters(self):
        """Return all table entries as parameters."""
        return iter(self.table.values())
```

### 4.2 Domain-Specific Implementations

Create lookup table versions for each domain:

```python
# src/empo/nn_based/multigrid/lookup/q_network.py

class MultiGridLookupTableQNetwork(LookupTableQNetwork):
    """
    Multigrid-specific lookup table Q-network.
    
    Inherits all functionality from base LookupTableQNetwork.
    Adds multigrid-specific configuration.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_actions: int,
        beta: float = 1.0,
        default_q: float = 0.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        **kwargs  # Accept but ignore neural network args for API compatibility
    ):
        super().__init__(num_actions, beta, default_q, feasible_range)
        self.grid_height = grid_height
        self.grid_width = grid_width
        # Ignore: num_agents_per_color, state_feature_dim, etc. (not needed)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'grid_height': self.grid_height,
            'grid_width': self.grid_width
        })
        return config
```

### 4.3 Phase 2 Lookup Tables

Similar structure for Phase 2 networks:

```python
# src/empo/nn_based/phase2/lookup/robot_q_network.py

class LookupTableRobotQNetwork(BaseRobotQNetwork):
    """Lookup table for Q_r(s, a_r)."""
    
    def __init__(
        self,
        num_actions: int,
        num_robots: int,
        beta_r: float = 10.0,
        default_q_r: float = -1.0,  # Q_r < 0
        feasible_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(num_actions, num_robots, beta_r, feasible_range)
        self.default_q_r = default_q_r
        self.table: Dict[int, torch.nn.Parameter] = {}
    
    def forward(self, state_keys: torch.Tensor) -> torch.Tensor:
        """
        Batch forward pass.
        
        Args:
            state_keys: (batch_size,) integer tensor of state hash keys
        
        Returns:
            Q_r values: (batch_size, num_action_combinations)
        """
        batch_size = state_keys.shape[0]
        device = state_keys.device
        
        output = torch.zeros(
            (batch_size, self.num_action_combinations),
            device=device,
            dtype=torch.float32
        )
        
        for i in range(batch_size):
            key = state_keys[i].item()
            
            if key not in self.table:
                self.table[key] = torch.nn.Parameter(
                    torch.full(
                        (self.num_action_combinations,),
                        self.default_q_r,
                        dtype=torch.float32
                    )
                )
            
            output[i] = self.table[key].to(device)
        
        # Ensure Q_r < 0 (same logic as neural version)
        return self.ensure_negative(output)
    
    def encode_and_forward(
        self,
        state: Hashable,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Single-state forward pass."""
        key = hash(state)
        
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(
                torch.full(
                    (self.num_action_combinations,),
                    self.default_q_r,
                    dtype=torch.float32
                )
            )
        
        return self.table[key].unsqueeze(0).to(device)
```

---

## 5. Batch Processing Details

### 5.1 Trainer Integration

**Key requirement:** No changes to trainer code.

Current trainer does:

```python
# Phase 1 training
q_values = q_network.forward(
    grid_tensor,
    global_features,
    agent_features,
    interactive_features,
    goal_coords
)
```

For lookup table version, we need to intercept these tensor inputs and convert to keys:

**Solution:** Override `forward()` to accept same signature but extract keys internally:

```python
def forward(
    self,
    grid_tensor: torch.Tensor,
    global_features: torch.Tensor,
    agent_features: torch.Tensor,
    interactive_features: torch.Tensor,
    goal_coords: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass with same signature as neural version.
    
    Internally converts tensors to hash keys for lookup.
    """
    # Convert tensors to states
    batch_size = grid_tensor.shape[0]
    
    # This requires a "reverse tensorization" function
    states = [
        self._tensors_to_state(
            grid_tensor[i],
            global_features[i],
            agent_features[i],
            interactive_features[i]
        )
        for i in range(batch_size)
    ]
    
    goals = [
        self._tensor_to_goal(goal_coords[i])
        for i in range(batch_size)
    ]
    
    # Now lookup
    output = torch.zeros((batch_size, self.num_actions))
    for i in range(batch_size):
        key = hash((states[i], goals[i]))
        if key not in self.table:
            self.table[key] = torch.nn.Parameter(...)
        output[i] = self.table[key]
    
    return output
```

**Challenge:** How to convert tensors back to hashable states?

**Solution 1: Store original states in replay buffer**

Modify replay buffer to keep raw states alongside tensorized versions:

```python
class Transition:
    state: Hashable  # Original state (NEW: keep this)
    state_tensors: Tuple[Tensor, ...]  # Tensorized (existing)
    goal: PossibleGoal  # Original goal (NEW: keep this)
    goal_tensor: Tensor  # Tensorized (existing)
```

Then trainer can pass original states to lookup networks.

**Solution 2: Inverse tensorization**

Implement `detensorize` methods that reverse the tensorization:

```python
def _tensors_to_state(self, grid_tensor, global_features, agent_features, interactive_features):
    """Convert tensors back to hashable state tuple."""
    # This is domain-specific and potentially lossy
    # For MultiGrid, we'd need to:
    # 1. Decode compressed grid
    # 2. Reconstruct agent positions from agent_features
    # 3. Reconstruct object states from interactive_features
    # 4. Return as hashable tuple
    pass
```

**Recommendation:** Use Solution 1 (store original states) because:
- Simpler and more reliable
- No information loss
- Minimal memory overhead (states are small compared to tensors)

### 5.2 Replay Buffer Modification

Add optional fields to Phase2Transition:

```python
@dataclass
class Phase2Transition:
    # Existing fields
    state: Hashable  # ALREADY EXISTS (for successor computation)
    next_state: Hashable  # ALREADY EXISTS
    robot_action: Tuple[int, ...]
    human_actions: List[Tuple[int, ...]]
    goals: List[PossibleGoal]
    transition_probs_by_action: Dict[int, List[Tuple[float, Hashable]]]
    compact_features: Tuple[Tensor, ...]  # For neural networks
    
    # NEW: Keep original goals for lookup networks
    # (actually, goals are already stored above!)
    
    # No changes needed! State and goals are already available
```

**Good news:** The replay buffer already stores raw `state`, `next_state`, and `goals`. No modifications needed!

### 5.3 Modified Trainer Forward Pass

```python
# In trainer._compute_q_r_targets():

# Current (neural network):
q_r_next = self.networks.q_r.forward(
    grid_tensor_batch,
    global_features_batch,
    agent_features_batch,
    interactive_features_batch
)

# Modified (check network type):
if isinstance(self.networks.q_r, LookupTableRobotQNetwork):
    # Pass original states instead of tensors
    q_r_next = self.networks.q_r.forward_from_states(
        states_batch,  # List of Hashable states
        device=self.device
    )
else:
    # Use tensors as before
    q_r_next = self.networks.q_r.forward(...)
```

**Better approach:** Make `forward()` signature flexible:

```python
class LookupTableRobotQNetwork(BaseRobotQNetwork):
    def forward(
        self,
        states_or_grid_tensor,  # Either List[Hashable] or Tensor
        global_features=None,
        agent_features=None,
        interactive_features=None
    ) -> torch.Tensor:
        # Detect input type
        if isinstance(states_or_grid_tensor, list):
            # Lookup table mode: states_or_grid_tensor is List[Hashable]
            return self._forward_from_states(states_or_grid_tensor)
        else:
            # Neural mode (for API compatibility): convert tensors to states
            # This requires inverse tensorization or error
            raise NotImplementedError(
                "Lookup table networks require raw states. "
                "Pass states as List[Hashable], not tensors."
            )
```

**Simplest approach:** Add a new method `forward_from_states()` and modify trainer to use it:

```python
class BaseRobotQNetwork:
    def forward_from_states(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states.
        
        Default implementation: tensorize then call forward().
        Lookup table networks override to use direct lookup.
        """
        # Default: use encode_and_forward for each state
        outputs = [
            self.encode_and_forward(state, None, device)
            for state in states
        ]
        return torch.cat(outputs, dim=0)

# Lookup table overrides:
class LookupTableRobotQNetwork(BaseRobotQNetwork):
    def forward_from_states(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        batch_size = len(states)
        output = torch.zeros(
            (batch_size, self.num_action_combinations),
            device=device
        )
        
        for i, state in enumerate(states):
            key = hash(state)
            if key not in self.table:
                self.table[key] = torch.nn.Parameter(...)
            output[i] = self.table[key]
        
        return output
```

Then trainer uses:

```python
# Always use forward_from_states (works for both neural and lookup):
q_r_next = self.networks.q_r.forward_from_states(
    next_states_list,
    device=self.device
)
```

---

## 6. Memory and Performance Considerations

### 6.1 Memory Usage

**Per-entry memory:**
- Neural network: Shared weights (e.g., 1M parameters) for all states
- Lookup table: `output_dim * sizeof(float32)` per unique state
  - Q-network: `num_actions * 4 bytes` (e.g., 7 actions = 28 bytes)
  - Robot Q-network: `num_action_combinations * 4 bytes` (e.g., 49 actions = 196 bytes)

**Total memory:**
- Small state space (< 10K states): Lookup table is **more efficient**
- Medium state space (10K - 1M states): Comparable
- Large state space (> 1M states): Neural network **far more efficient**

**Example (MultiGrid 7x7):**
- State space: ~10^6 states (rough estimate)
- Lookup table: 10^6 * 28 bytes = 28 MB (Q-network)
- Neural network: ~1M parameters * 4 bytes = 4 MB

**Conclusion:** Lookup tables are viable for small MultiGrid environments but may not scale to larger grids or complex environments.

### 6.2 Computational Performance

**Forward pass:**
- Neural network: O(1) per batch (amortized), constant time regardless of batch size
- Lookup table: O(batch_size) due to loop over batch elements

**Backward pass:**
- Neural network: O(weights) gradient computation, affects all parameters
- Lookup table: O(batch_size) gradient computation, updates only affected entries

**Expected performance:**
- Small batch (< 32): Lookup table **comparable or faster** (no conv/MLP overhead)
- Large batch (> 128): Neural network **much faster** (parallelism)

### 6.3 Optimization Strategies

To maintain batch processing efficiency with lookup tables:

**Strategy 1: Vectorized lookup (if all states are known)**

```python
# Pre-build index mapping (requires knowing all states in advance)
state_to_idx = {state: i for i, state in enumerate(all_states)}
lookup_matrix = torch.nn.Parameter(torch.zeros(len(all_states), output_dim))

def forward(self, states):
    indices = torch.tensor([state_to_idx[s] for s in states])
    return lookup_matrix[indices]  # Vectorized indexing
```

**Strategy 2: Lazy parameter stacking**

```python
# Cache a "parameter matrix" and rebuild when table grows
self._param_matrix = None
self._param_matrix_size = 0

def forward(self, states):
    # Rebuild matrix if table grew
    if len(self.table) > self._param_matrix_size:
        self._rebuild_param_matrix()
    
    # Use vectorized indexing
    indices = [self.state_to_idx[hash(s)] for s in states]
    return self._param_matrix[indices]
```

**Strategy 3: JIT compilation**

```python
@torch.jit.script
def lookup_batch(table: Dict[int, Tensor], keys: List[int]) -> Tensor:
    # TorchScript might optimize this loop
    pass
```

**Recommendation:** Start with simple loop-based implementation (Solution 1 from section 3.2). Optimize only if profiling shows it's a bottleneck.

---

## 7. Code Organization

### 7.1 Directory Structure

```
src/empo/nn_based/
├── lookup/                          # NEW: Lookup table implementations
│   ├── __init__.py
│   ├── q_network.py                 # LookupTableQNetwork
│   ├── policy_prior_network.py      # LookupTablePolicyPriorNetwork
│   └── README.md                    # Usage documentation
├── phase2/
│   └── lookup/                      # NEW: Phase 2 lookup tables
│       ├── __init__.py
│       ├── robot_q_network.py       # LookupTableRobotQNetwork
│       ├── robot_value_network.py   # LookupTableRobotValueNetwork
│       ├── human_goal_ability.py    # LookupTableHumanGoalAbilityNetwork
│       ├── aggregate_goal_ability.py
│       └── intrinsic_reward_network.py
└── multigrid/
    └── lookup/                      # NEW: MultiGrid lookup tables
        ├── __init__.py
        ├── q_network.py             # MultiGridLookupTableQNetwork
        └── phase2/
            ├── __init__.py
            └── robot_q_network.py   # MultiGridLookupTableRobotQNetwork
```

### 7.2 Factory Functions

Add factory functions to create networks based on configuration:

```python
# src/empo/nn_based/factory.py

def create_q_network(
    use_lookup_table: bool = False,
    **config
) -> BaseQNetwork:
    """
    Create Q-network (neural or lookup table).
    
    Args:
        use_lookup_table: If True, create LookupTableQNetwork.
        **config: Network configuration.
    
    Returns:
        BaseQNetwork instance.
    """
    if use_lookup_table:
        from .lookup.q_network import LookupTableQNetwork
        return LookupTableQNetwork(**config)
    else:
        # Existing neural network creation
        ...
```

### 7.3 Trainer Modifications

Minimal trainer changes to support both:

```python
# In Phase2Trainer._compute_q_r_targets():

# Check if using lookup table network
if hasattr(self.networks.q_r, 'table'):
    # Lookup table mode: pass raw states
    q_r_next = self.networks.q_r.forward_from_states(
        next_states_list,
        device=self.device
    )
else:
    # Neural mode: pass tensors
    q_r_next = self.networks.q_r.forward(
        grid_tensor_batch,
        global_features_batch,
        agent_features_batch,
        interactive_features_batch
    )
```

**Better alternative:** Add `forward_from_states()` to all networks (base class default implementation) and always use it:

```python
# Always use forward_from_states (works for both types):
q_r_next = self.networks.q_r.forward_from_states(
    next_states_list,
    device=self.device
)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_lookup_table_networks.py

def test_lookup_table_q_network_forward():
    """Test that lookup table Q-network produces correct outputs."""
    network = LookupTableQNetwork(num_actions=7, default_q=0.5)
    
    # Create test states
    state1 = (0, 1, 2)  # Hashable tuple
    state2 = (1, 2, 3)
    goal = PossibleGoal(...)
    
    # First forward pass (creates entries)
    q1 = network.encode_and_forward(state1, None, 0, goal, 'cpu')
    assert q1.shape == (1, 7)
    assert torch.allclose(q1, torch.full((1, 7), 0.5))
    
    # Modify entry
    key = hash((state1, goal))
    network.table[key].data.fill_(1.0)
    
    # Second forward pass (retrieves modified values)
    q1_new = network.encode_and_forward(state1, None, 0, goal, 'cpu')
    assert torch.allclose(q1_new, torch.full((1, 7), 1.0))
    
    # Different state gets default value
    q2 = network.encode_and_forward(state2, None, 0, goal, 'cpu')
    assert torch.allclose(q2, torch.full((1, 7), 0.5))

def test_lookup_table_batching():
    """Test batch processing."""
    network = LookupTableQNetwork(num_actions=7, default_q=0.0)
    
    states = [(0, 0), (0, 1), (1, 0), (0, 0)]  # Note: first and last are same
    goals = [goal1, goal2, goal3, goal1]
    
    output = network.forward_from_states_and_goals(states, goals, 'cpu')
    assert output.shape == (4, 7)
    
    # First and last should be identical (same key)
    assert torch.allclose(output[0], output[3])

def test_lookup_table_gradients():
    """Test that gradients flow correctly."""
    network = LookupTableQNetwork(num_actions=7, default_q=0.0)
    state = (0, 1, 2)
    goal = PossibleGoal(...)
    
    q_values = network.encode_and_forward(state, None, 0, goal, 'cpu')
    loss = q_values.mean()
    loss.backward()
    
    # Check that parameter has gradient
    key = hash((state, goal))
    assert network.table[key].grad is not None
    assert network.table[key].grad.shape == (7,)
```

### 8.2 Integration Tests

```python
# tests/test_lookup_table_integration.py

def test_phase2_training_with_lookup_tables():
    """Test Phase 2 training with lookup table networks."""
    
    # Create environment
    env = ...
    
    # Create networks (mix of neural and lookup)
    networks = Phase2Networks(
        q_r=LookupTableRobotQNetwork(...),  # Lookup
        v_r=MultiGridRobotValueNetwork(...),  # Neural
        v_he=LookupTableHumanGoalAbilityNetwork(...),  # Lookup
        x_h=MultiGridAggregateGoalAbilityNetwork(...),  # Neural
        u_r=MultiGridIntrinsicRewardNetwork(...)  # Neural
    )
    
    # Train
    trainer = Phase2Trainer(env, networks, ...)
    trainer.train(num_steps=100)
    
    # Verify that lookup tables grew
    assert len(networks.q_r.table) > 0
    assert len(networks.v_he.table) > 0

def test_lookup_table_saves_and_loads():
    """Test save/load functionality."""
    network1 = LookupTableQNetwork(num_actions=7)
    
    # Add some entries
    state = (0, 1)
    goal = PossibleGoal(...)
    network1.encode_and_forward(state, None, 0, goal, 'cpu')
    network1.table[hash((state, goal))].data.fill_(1.234)
    
    # Save
    config = network1.get_config()
    state_dict = network1.state_dict()
    
    # Load into new network
    network2 = LookupTableQNetwork(**config)
    network2.load_state_dict(state_dict)
    
    # Verify
    q = network2.encode_and_forward(state, None, 0, goal, 'cpu')
    assert torch.allclose(q, torch.full((1, 7), 1.234))
```

---

## 9. Example Usage

### 9.1 Phase 1 Example

```python
# examples/lookup_table_phase1_demo.py

from empo.nn_based.lookup.q_network import LookupTableQNetwork
from empo.nn_based.lookup.policy_prior_network import LookupTablePolicyPriorNetwork
from empo.nn_based.multigrid.lookup.neural_policy_prior import MultiGridLookupTablePolicyPrior

# Create lookup table networks
q_network = LookupTableQNetwork(
    num_actions=7,
    beta=2.0,
    default_q=0.5  # Optimistic initialization
)

policy_network = LookupTablePolicyPriorNetwork(
    num_actions=7,
    default_prob=1.0/7  # Uniform distribution
)

# Create policy prior (same interface as neural version)
policy_prior = MultiGridLookupTablePolicyPrior(
    q_network=q_network,
    policy_network=policy_network,
    world_model=env,
    human_agent_indices=[0, 1]
)

# Training loop (same as neural version)
trainer = Trainer(
    q_network=q_network,
    policy_network=policy_network,
    world_model=env,
    ...
)
trainer.train(num_episodes=1000)

# Inference (same as neural version)
action_probs = policy_prior(state, agent_idx=0, goal=some_goal)
```

### 9.2 Phase 2 Example

```python
# examples/lookup_table_phase2_demo.py

from empo.nn_based.phase2.lookup.robot_q_network import LookupTableRobotQNetwork
from empo.nn_based.phase2.config import Phase2Networks

# Create hybrid architecture (some lookup, some neural)
networks = Phase2Networks(
    q_r=LookupTableRobotQNetwork(
        num_actions=4,
        num_robots=2,
        beta_r=10.0,
        default_q_r=-1.0  # Q_r < 0
    ),
    v_r=MultiGridRobotValueNetwork(...),  # Neural (for comparison)
    v_he=LookupTableHumanGoalAbilityNetwork(
        default_v_he=0.5  # Probability
    ),
    x_h=MultiGridAggregateGoalAbilityNetwork(...),  # Neural
    u_r=MultiGridIntrinsicRewardNetwork(...)  # Neural
)

# Training (same interface)
trainer = Phase2Trainer(
    env=env,
    networks=networks,
    ...
)
trainer.train(num_steps=10000)

# Lookup table stats
print(f"Q_r table size: {len(networks.q_r.table)} states")
print(f"V_h^e table size: {len(networks.v_he.table)} (state, goal) pairs")
```

---

## 10. Advantages and Disadvantages

### 10.1 Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Exact values** | No function approximation error—each state stores exact learned value |
| **Convergence guarantees** | Tabular methods have proven convergence under standard assumptions |
| **Interpretability** | Can directly inspect what the agent has learned for each state |
| **No generalization bias** | States are treated independently (good for highly irregular value functions) |
| **Debugging** | Easy to see which states have been visited and their values |
| **Selective use** | Can mix lookup tables and neural networks in same system |

### 10.2 Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Memory usage** | Scales linearly with number of unique states visited |
| **No generalization** | Cannot infer values for unseen states (uses default) |
| **Slow for large batches** | Loop over batch elements vs. vectorized neural forward |
| **Inefficient for large state spaces** | Impractical for continuous or high-dimensional states |
| **State representation sensitivity** | Different state representations produce different hash keys |

---

## 11. Open Questions

1. **Optimizer state management:** How to handle optimizer state (momentum, second moments for Adam) when new parameters are added dynamically? 
   - **Option A:** Reset optimizer state for new entries
   - **Option B:** Initialize with global averages
   - **Option C:** Use optimizer without state (SGD)

2. **Hash collision handling:** Python's `hash()` can have collisions. Should we use more robust hashing (e.g., `hashlib.sha256`)?
   - **Recommendation:** Use `hash()` for speed; collisions are rare for typical state spaces

3. **State representation:** Should we enforce a canonical representation (e.g., sort agent positions) to avoid duplicate entries for equivalent states?
   - **Recommendation:** Let user control via `WorldModel.get_state()`—it should return canonical form

4. **Partial observability:** How to handle partially observable states where hash might not capture full state?
   - **Answer:** Not applicable—EMPO assumes full observability

5. **Save/load format:** Should we save the entire table to disk, or only states with non-default values?
   - **Recommendation:** Save all non-default entries for efficiency

6. **Default value adaptation:** Should default values adapt over time (e.g., mean of observed values)?
   - **Recommendation:** Keep fixed for simplicity; user can initialize strategically

---

## 12. Implementation Checklist

### Phase 1: Base Lookup Table Networks

- [ ] Implement `LookupTableQNetwork` (Phase 1)
- [ ] Implement `LookupTablePolicyPriorNetwork` (Phase 1)
- [ ] Add `forward_from_states()` to `BaseQNetwork`
- [ ] Add `forward_from_states()` to `BasePolicyPriorNetwork`
- [ ] Unit tests for lookup table networks

### Phase 2: Phase 2 Lookup Table Networks

- [x] Implement `LookupTableRobotQNetwork`
- [x] Implement `LookupTableRobotValueNetwork`
- [x] Implement `LookupTableHumanGoalAbilityNetwork`
- [x] Implement `LookupTableAggregateGoalAbilityNetwork`
- [x] Implement `LookupTableIntrinsicRewardNetwork`
- [x] Unit tests for Phase 2 lookup networks
- [x] Add `use_lookup_tables` config options to `Phase2Config`
- [x] Add helper methods: `should_use_lookup_table()`, `get_lookup_default()`, `should_recreate_optimizer()`

### Phase 3: Domain-Specific Implementations

- [ ] Implement `MultiGridLookupTableQNetwork`
- [ ] Implement `MultiGridLookupTablePolicyPrior`
- [ ] Implement `MultiGridLookupTableRobotQNetwork` (Phase 2)
- [ ] Integration tests with MultiGrid

### Phase 4: Trainer Integration

- [ ] Modify Phase 1 `Trainer` to support lookup tables
- [ ] Modify Phase 2 `Trainer` to support lookup tables
- [ ] Update replay buffer if needed (already has raw states)
- [ ] Integration tests for training

### Phase 5: Utilities and Documentation

- [ ] Factory functions for network creation
- [ ] Save/load functionality
- [ ] Example scripts (Phase 1 and Phase 2)
- [ ] Performance benchmarking
- [ ] API documentation

### Phase 6: Advanced Features (Optional)

- [ ] Vectorized batch lookup optimization
- [ ] JIT compilation for forward pass
- [ ] Adaptive default values
- [ ] Table pruning (remove rarely-accessed entries)
- [ ] Distributed lookup tables (sharding for very large tables)

---

## 13. Conclusion

This design enables **optional** conversion of EMPO networks to lookup tables with minimal code changes:

1. **Zero changes outside network classes** - Trainers, replay buffers, and examples work identically
2. **Dict-based implementation** - Efficient O(1) lookup with dynamic table growth
3. **Identity encoders** - No encoding overhead, direct state→value mapping
4. **Batch processing** - Loop-based batching maintains same API as neural networks
5. **Gradient compatibility** - Uses `torch.nn.Parameter` for automatic differentiation
6. **Selective application** - Mix lookup tables and neural networks as needed

**When to use:**
- Small state spaces (< 100K states)
- Debugging and interpretability
- Environments where exact values are critical
- Baseline comparisons with neural approaches

**When NOT to use:**
- Large or continuous state spaces
- When generalization is important
- Production systems prioritizing memory efficiency

The modular design allows experimentation with hybrid architectures (e.g., lookup table Q_r with neural V_h^e) to find the best trade-off between exactness and scalability.

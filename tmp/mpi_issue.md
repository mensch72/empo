## Summary

Extend the current local multiprocessing parallelization in `get_dag_parallel()` and `compute_human_policy_prior()` to support distributed execution across multiple cluster nodes using MPI, while keeping fast shared-memory parallelization within each node.

## Motivation

The current implementation uses:
- `get_dag_parallel()`: spawn-based `ProcessPoolExecutor` with pickle serialization
- `compute_human_policy_prior()`: fork-based `ProcessPoolExecutor` with shared memory (copy-on-write)

This works well on a single machine but cannot scale to HPC clusters where compute nodes communicate via MPI.

## Proposed Solution: Hybrid MPI + Local Multiprocessing

Use a two-level parallelization strategy:
- **Inter-node**: MPI for communication between nodes (slower, only when necessary)
- **Intra-node**: Existing `ProcessPoolExecutor` with fork/spawn (fast shared memory)

### Architecture

```
Node 0 (MPI rank 0)              Node 1 (MPI rank 1)
┌─────────────────────┐          ┌─────────────────────┐
│  Coordinator        │          │  Coordinator        │
│  ┌───┬───┬───┬───┐  │   MPI    │  ┌───┬───┬───┬───┐  │
│  │W0 │W1 │W2 │W3 │  │ <------> │  │W0 │W1 │W2 │W3 │  │
│  └───┴───┴───┴───┘  │          │  └───┴───┴───┴───┘  │
│  Local ProcessPool  │          │  Local ProcessPool  │
└─────────────────────┘          └─────────────────────┘
```

## Key Design Decisions

### 1. Simulation mode for testing

Support testing the full MPI code path on a single machine:
- Use environment variable or CLI flag to partition local CPUs into simulated nodes
- Example: 8-CPU machine simulating 2 nodes with 4 CPUs each
- Enables development and debugging without cluster access

### 2. Different data passing for DAG vs backward induction

**`get_dag_parallel()`** - must pass full serialized states:
- During DAG exploration, states are discovered dynamically
- A state discovered on Node 1 might be a successor of a state on Node 2
- State indices don't exist yet - we're still building the list
- Each node needs the full state tuple to check "have I seen this before?"
- After each wave: `Allgather()` newly discovered states (full tuples)
- After DAG complete: all nodes have identical `states` list and `state_to_idx` mapping

**`compute_human_policy_prior()`** - can use compact indices:
- DAG already computed, all nodes have identical state list
- States referenced by index (integer), not full tuple
- Goals referenced by ID (integer), not goal objects
- MPI messages contain: `{state_idx: {agent_idx: {goal_id: float_value}}}`
- Much more compact than serializing state tuples and goal objects

### 3. Goal ID system

- Do NOT pass `PossibleGoal` objects in MPI messages
- Each node has the same `PossibleGoalGenerator` and generates goals locally
- Only pass goal IDs (integers based on hash or sequential assignment)
- Requires adding `goal_id` property to `PossibleGoal` class

### 4. V_values synchronization

After each dependency level in backward induction:
- Each node computes V_values for its assigned states locally (using ProcessPoolExecutor)
- Use `MPI.Allgather()` to share computed V_values across nodes
- All nodes maintain a complete copy of V_values (simpler than partitioned approach)

### 5. Graceful fallback

If `MPI.COMM_WORLD.size == 1` (single node or no MPI), use pure local multiprocessing with zero MPI overhead.

## Implementation Plan

### Phase 1: Goal ID System
- [ ] Add `goal_id` property to `PossibleGoal` class (hash-based or sequential)
- [ ] Modify `PossibleGoalGenerator.generate()` to return `(goal, goal_id, weight)`
- [ ] Update workers to reference goals by ID in result messages

### Phase 2: Distributed Context
Create `src/empo/distributed.py`:

```python
from mpi4py import MPI
import multiprocessing as mp

class DistributedContext:
    """Manages hybrid MPI + local parallelization."""
    
    def __init__(self, simulate_nodes=None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_root = (self.rank == 0)
        
        # For simulation mode on single machine
        if simulate_nodes and self.size == 1:
            self._setup_simulated_ranks(simulate_nodes)
        
        # Local workers per node
        total_cpus = mp.cpu_count()
        self.local_workers = max(1, total_cpus // self.size)
    
    def scatter_work(self, items):
        """Distribute items across MPI ranks (round-robin)."""
        return items[self.rank::self.size]
    
    def allgather(self, local_data):
        """Gather data from all ranks to all ranks."""
        return self.comm.allgather(local_data)
    
    def broadcast(self, data, root=0):
        """Broadcast data from root to all ranks."""
        return self.comm.bcast(data, root=root)
    
    def barrier(self):
        """Synchronization barrier."""
        self.comm.Barrier()
```

### Phase 3: Refactor `get_dag_parallel()`

```python
def get_dag_parallel_distributed(self, ctx: DistributedContext):
    # Root initializes with root state
    if ctx.is_root:
        root_state = self.get_state()
        current_wave = [root_state]
        all_states = [root_state]
        state_to_idx = {root_state: 0}
    else:
        current_wave = all_states = state_to_idx = None
    
    # Broadcast initial state
    current_wave = ctx.broadcast(current_wave)
    
    while current_wave:
        # Each rank processes its share of states
        my_states = ctx.scatter_work(current_wave)
        
        # Local parallel processing (existing ProcessPoolExecutor code)
        my_new_states, my_edges = process_states_locally(my_states, ctx.local_workers)
        
        # Allgather: share discovered states (FULL TUPLES - unavoidable)
        all_new_states = ctx.allgather(my_new_states)
        all_edges = ctx.allgather(my_edges)
        
        # Merge and deduplicate (all ranks do this identically)
        current_wave = []
        for new_states in all_new_states:
            for state in new_states:
                if state not in state_to_idx:
                    state_to_idx[state] = len(all_states)
                    all_states.append(state)
                    current_wave.append(state)
        
        # Merge edges...
        
        ctx.barrier()
    
    # Now all ranks have identical states, state_to_idx, successors, transitions
    return states, state_to_idx, successors, transitions
```

### Phase 4: Refactor `compute_human_policy_prior()`

```python
def compute_human_policy_prior_distributed(..., ctx: DistributedContext):
    # DAG computation (all ranks get identical result)
    states, state_to_idx, successors, transitions = world_model.get_dag_distributed(ctx)
    
    # Compute dependency levels (all ranks compute identically)
    dependency_levels = compute_dependency_levels(successors)
    
    # Process levels
    for level in dependency_levels:
        # Each rank gets a slice of states (by INDEX now, not full state)
        my_state_indices = ctx.scatter_work(level)
        
        # Process locally with existing fork-based parallelization
        _init_shared_data(states, transitions, V_values, params)
        with ProcessPoolExecutor(max_workers=ctx.local_workers, mp_context=fork_ctx):
            # ... existing local parallel code ...
        
        # Pack results compactly: {state_idx: {agent_idx: {goal_id: value}}}
        packed_v = pack_v_results_by_id(my_v_results)
        
        # Allgather V_values across nodes (compact: indices and floats only)
        all_v_results = ctx.allgather(packed_v)
        
        # Merge into V_values (all ranks do this identically)
        for v_results in all_v_results:
            unpack_and_merge(V_values, v_results)
        
        ctx.barrier()
```

## Testing Strategy

1. **Unit tests**: Test goal ID generation, packing/unpacking
2. **Simulation mode**: `python train.py --mpi-simulate-nodes 2` on single machine
3. **Local MPI**: `mpirun -np 2 python train.py --mpi` (2 processes, same machine)
4. **Cluster**: `srun -N 4 apptainer exec empo.sif python train.py --mpi`

## Files to Modify

- [ ] `src/empo/possible_goal.py` - Add goal ID system
- [ ] `src/empo/distributed.py` - New file for MPI wrapper
- [ ] `src/empo/backward_induction.py` - Add distributed support
- [ ] `src/empo/world_model.py` - Add distributed support to `get_dag_parallel()`
- [ ] `train.py` - Add `--mpi` and `--mpi-simulate-nodes` CLI flags
- [ ] `scripts/run_cluster.sh` - Update for proper MPI + Apptainer invocation

## Dependencies

- `mpi4py>=3.1.0` (already in requirements-hierarchical.txt)

## Out of Scope (Future Work)

- Dynamic work stealing across nodes
- Partitioned V_values (each node holds only subset)
- Adaptive batch sizing based on network latency
- GPU acceleration within nodes

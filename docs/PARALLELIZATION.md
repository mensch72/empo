# Parallelization Strategies in EMPO

This document describes the different forms of pickling and data passing used in the parallelization strategies throughout the EMPO codebase.

## Overview

EMPO uses Python's `multiprocessing` module with `ProcessPoolExecutor` for parallel computation, in addition to the parallelization that pytorch does internally. There are currently three such non-pytorch parallelization sites:

1. **DAG Construction** (`src/empo/world_model.py`) - Parallelizes BFS exploration of the state space
2. **Backward Induction** (`src/empo/backward_induction.py`) - Parallelizes policy computation across states
3. **Async Phase 2 Training** (`src/empo/nn_based/phase2/trainer.py`) - Parallel actor-learner architecture

Each site uses different data passing strategies optimized for its specific workload.

---

## 1. DAG Construction (`get_dag_parallel`)

**Location:** `src/empo/world_model.py`

### Multiprocessing Context: `spawn`

The parallel DAG construction uses the `spawn` context:

```python
ctx = mp.get_context('spawn')
```

**Why `spawn`?**
- The `spawn` context creates fresh Python interpreter processes
- Each worker starts with a clean slate and must explicitly receive any data it needs
- Safer for environments that may not be fork-safe (e.g., those using threads or CUDA)
- Cross-platform compatible (works on Windows, macOS, and Linux)

### Data Passing Strategy: Standard Pickle with Initializer

```python
# Serialize the entire environment using standard pickle
env_pickle = pickle.dumps(self)

# Workers are initialized with the pickled environment
with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                         initializer=_init_dag_worker,
                         initargs=(env_pickle,)) as executor:
```

**How it works:**
1. The environment is serialized using standard `pickle.dumps()`
2. The pickled bytes are passed to `ProcessPoolExecutor` as an initializer argument
3. Each worker calls `_init_dag_worker()` which deserializes the environment
4. Workers store the deserialized environment in module-level globals

**Worker initialization:**
```python
def _init_dag_worker(env_pickle):
    global _worker_env, _worker_num_agents, _worker_num_actions
    _worker_env = pickle.loads(env_pickle)  # Each worker has its own copy
    _worker_num_agents = len(_worker_env.agents)
    _worker_num_actions = _worker_env.action_space.n
```

### Data Flow

```
Main Process                    Worker Processes
     │                               │
     ├─ pickle.dumps(env) ──────────►│ _init_dag_worker()
     │                               │   └─ pickle.loads(env_pickle)
     │                               │   └─ Store in _worker_env
     │                               │
     ├─ Submit state ──────────────► │ _process_state_actions(state)
     │                               │   └─ Use _worker_env.transition_probabilities()
     │ ◄── Return (state, succs) ────┤
     │                               │
```

**Characteristics:**
- **Serialization:** Standard `pickle` (sufficient for environment objects)
- **Copy semantics:** Each worker gets its own independent copy of the environment
- **State management:** Each worker can safely call `set_state()` without affecting others
- **Overhead:** Moderate initialization cost (one pickle/unpickle per worker, not per task)

---

## 2. Backward Induction (`compute_human_policy_prior`)

**Location:** `src/empo/backward_induction.py`

### Multiprocessing Context: `fork`

The parallel backward induction uses the `fork` context:

```python
ctx = mp.get_context('fork')
```

**Why `fork`?**
- The `fork` context creates child processes that inherit the parent's memory (copy-on-write)
- Large data structures (states, transitions, V-values) are shared without explicit copying
- Significantly faster for large state spaces
- Only works inside a Docker container (standard approach in this project) or on Unix-like systems (Linux, macOS - not Windows)

### Data Passing Strategy: Shared Memory via Module Globals

```python
# Module-level globals - inherited copy-on-write by forked processes
_shared_states = None
_shared_transitions = None
_shared_V_values = None
_shared_params = None
_shared_believed_others_policy_pickle = None

def _init_shared_data(states, transitions, V_values, params, believed_others_policy_pickle=None):
    """Initialize shared data for worker processes."""
    global _shared_states, _shared_transitions, _shared_V_values, _shared_params
    global _shared_believed_others_policy_pickle
    _shared_states = states
    _shared_transitions = transitions
    _shared_V_values = V_values
    _shared_params = params
    _shared_believed_others_policy_pickle = believed_others_policy_pickle
```

**How it works:**
1. Before forking workers, the main process sets module-level global variables
2. Workers inherit these globals via the fork (copy-on-write semantics)
3. Workers read the shared data without any explicit serialization/deserialization
4. Only state indices are passed as task arguments (minimal data transfer)

### Cloudpickle for Custom Functions

For custom `believed_others_policy` functions, standard pickle often fails because:
- Lambda functions cannot be pickled
- Closures capture variables that may not be picklable
- Nested functions have complex scope references

**Solution:** Use `cloudpickle` for function serialization:

```python
import cloudpickle

# In main process: serialize the custom function
if believed_others_policy is not None:
    believed_others_policy_pickle = cloudpickle.dumps(believed_others_policy)
else:
    believed_others_policy_pickle = None

# Pass to workers via shared globals
_init_shared_data(states, transitions, V_values, params, believed_others_policy_pickle)

# In worker: deserialize the function
def process_state_batch(state_indices):
    if _shared_believed_others_policy_pickle is not None:
        believed_others_policy = cloudpickle.loads(_shared_believed_others_policy_pickle)
    else:
        believed_others_policy = default_believed_others_policy_wrapper
```

**Why cloudpickle?**
- Can serialize lambdas, closures, and locally-defined functions
- Serializes the function's code and captured variables together
- Compatible with multiprocessing across process boundaries

### Data Flow

```
Main Process                         Worker Processes (forked)
     │                                      │
     ├─ Set module globals ─────────────────┼─── Inherit via fork (copy-on-write)
     │   _shared_states = states            │      _shared_states (shared read)
     │   _shared_transitions = transitions  │      _shared_transitions (shared read)
     │   _shared_V_values = V_values        │      _shared_V_values (shared read)
     │   _shared_params = params            │      _shared_params (shared read)
     │   _shared_believed_others_policy_    │      _shared_believed_others_policy_pickle
     │      pickle = cloudpickle.dumps(fn)  │
     │                                      │
     ├─ Submit [state_indices] ────────────►│ process_state_batch(state_indices)
     │   (only indices, not data!)          │   ├─ Access _shared_* globals
     │                                      │   ├─ cloudpickle.loads(policy_pickle)
     │                                      │   └─ Compute policies for batch
     │ ◄── Return (v_results, p_results) ───┤
     │                                      │
     ├─ Merge results into V_values ────────┤
     │                                      │
     │ (Repeat for each dependency level)   │
```

### Level-by-Level Execution

Because backward induction requires V-values from successor states, computation must proceed level-by-level (a level is a set of states neither of which is a possible ancestor state of another):

```python
for level_idx, level in enumerate(dependency_levels):
    if len(level) <= num_workers:
        # Few states - process sequentially (avoid fork overhead)
        for state_index in level:
            # ... compute policy ...
    else:
        # Many states - parallelize
        # Re-initialize shared data so workers see updated V_values
        _init_shared_data(states, transitions, V_values, params, 
                          believed_others_policy_pickle)
        
        # Create new executor to fork with current V_values
        with ProcessPoolExecutor(...) as executor:
            futures = [executor.submit(process_state_batch, batch) 
                      for batch in batches]
            # Collect and merge results
```

**Key insight:** A new `ProcessPoolExecutor` is created for each level to ensure workers fork with the latest `V_values` updates from the previous level.

---

## Comparison of Approaches

| Aspect | DAG Construction | Backward Induction | Async Phase 2 Training |
|--------|------------------|-------------------|------------------------|
| **Context** | `spawn` | `fork` | `spawn` |
| **Platform** | Cross-platform | Docker/Unix-only | Cross-platform |
| **Data Passing** | Pickle + initializer | Module globals (fork inheritance) | Queue + Manager dict |
| **Large Data** | Copied per worker | Shared (copy-on-write) | Each actor has local copy |
| **Custom Functions** | N/A | cloudpickle | N/A |
| **Initialization** | Once per worker | Once per level | Once per actor |
| **Task Arguments** | State tuples | State indices only | Transitions via queue |
| **Memory Usage** | Higher (copies) | Lower (shared) | Moderate (local envs) |
| **GPU Support** | N/A | N/A | Yes (spawn required) |

---

## Best Practices

### When to use `spawn` with pickle:
- Cross-platform compatibility needed
- Workers need independent copies of mutable state
- Data size is moderate
- Environment may use threads or CUDA

### When to use `fork` with shared globals:
- Docker/Unix-only deployment is acceptable
- Large read-only data structures
- Minimal per-task data transfer
- Need custom functions (use cloudpickle)

### Serializing Custom Functions:
1. Use `cloudpickle` instead of `pickle` for functions
2. Serialize once in the main process
3. Deserialize once per batch in workers (not per state)
4. Store serialized bytes in module globals for fork inheritance

---

## Implementation Details

### Module-Level Globals Pattern

```python
# Define at module level
_shared_data = None

def init_worker_data(data):
    """Called before forking or in worker initializer."""
    global _shared_data
    _shared_data = data

def worker_function(task):
    """Access shared data from globals."""
    data = _shared_data
    # ... use data ...
```

### Cloudpickle Serialization Pattern

```python
import cloudpickle

# Main process
custom_fn = lambda x: x * 2  # Standard pickle would fail
fn_bytes = cloudpickle.dumps(custom_fn)

# Worker process
restored_fn = cloudpickle.loads(fn_bytes)
result = restored_fn(21)  # Returns 42
```

---

## Troubleshooting

### "Can't pickle local object"
- **Cause:** Standard pickle cannot serialize lambdas or local functions
- **Fix:** Use `cloudpickle.dumps()` instead of `pickle.dumps()`

### "This platform does not support fork"
- **Cause:** Windows or certain macOS configurations
- **Fix:** Use `spawn` context instead, with explicit pickle serialization

### Workers not seeing updated data
- **Cause:** Fork happened before data was updated
- **Fix:** Create new ProcessPoolExecutor after updating shared globals

### High memory usage with many workers
- **Cause:** Each worker has copy of large data (spawn context)
- **Fix:** Use fork context with shared globals, or reduce worker count

---

## 3. Async Actor-Learner Training (Phase 2)

**Location:** `src/empo/nn_based/phase2/trainer.py`

### Overview

Phase 2 training supports an async actor-learner architecture where multiple actor processes
collect environment transitions in parallel while a learner process updates the neural networks.

```
Enable with: Phase2Config(async_training=True, num_actors=4)
```

### Multiprocessing Context: `spawn`

Async training uses the `spawn` context:

```python
ctx = mp.get_context('spawn')
```

**Why `spawn`?**
- CUDA tensors cannot be forked safely
- Neural networks with GPU state require fresh process initialization
- Cross-platform compatible (necessary for Colab, Windows)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Main Process                                │
│                                                                     │
│  ┌─────────────┐                              ┌─────────────────┐   │
│  │   Learner   │◄─────── Transitions ─────────│  Shared Queue   │   │
│  │   Process   │                              │   (mp.Queue)    │   │
│  │             │                              │                 │   │
│  │ - Update    │     ┌───────────────────────►│  Transitions    │   │
│  │   networks  │     │                        │  from actors    │   │
│  │ - Manage    │     │                        └─────────────────┘   │
│  │   warmup    │     │                                              │
│  │             │     │   ┌──────────────────────────────────────┐   │
│  └─────────────┘     │   │        Shared State (Manager)        │   │
│        │             │   │                                      │   │
│        │ Sync policy │   │  - policy_state_dict (for actors)    │   │
│        ▼             │   │  - shared_total_steps (Value)        │   │
│  ┌─────────────┐     │   │  - done_flag (Value)                 │   │
│  │   Manager   │─────┘   │                                      │   │
│  │   Dict      │         └──────────────────────────────────────┘   │
│  └─────────────┘                        ▲                           │
│                                         │                           │
└─────────────────────────────────────────┼───────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────┐
        │                                 │                         │
        ▼                                 ▼                         ▼
┌───────────────┐               ┌───────────────┐          ┌───────────────┐
│   Actor 0     │               │   Actor 1     │   ...    │   Actor N     │
│               │               │               │          │               │
│ - Local env   │               │ - Local env   │          │ - Local env   │
│ - Local nets  │               │ - Local nets  │          │ - Local nets  │
│ - Collect     │               │ - Collect     │          │ - Collect     │
│   transitions │               │   transitions │          │   transitions │
│ - Push to     │               │ - Push to     │          │ - Push to     │
│   queue       │               │   queue       │          │   queue       │
└───────────────┘               └───────────────┘          └───────────────┘
```

### Shared State for Warmup Coordination

A key challenge is coordinating warmup stages across processes. The solution uses
`multiprocessing.Value` for a shared step counter:

```python
# Shared counter for warmup/epsilon coordination
shared_total_steps = mp.Value('i', 0)  # Integer, lock=True by default

# Actors read to determine current warmup stage
with shared_total_steps.get_lock():
    current_steps = shared_total_steps.value

# Only learner increments the counter
with shared_total_steps.get_lock():
    shared_total_steps.value += batch_size
```

**Why shared steps matter:**
- `epsilon` (exploration) depends on total steps
- `beta_r` (policy softness) ramps up based on steps
- All actors must use consistent epsilon/beta_r values

### Policy Synchronization

Actors periodically sync their local networks with the learner's updated weights:

```python
# Every actor_sync_freq steps, actors check for new weights
if manager_dict.get('policy_state_dict') is not None:
    local_q_r.load_state_dict(manager_dict['policy_state_dict'])
```

The learner publishes updated weights at configurable intervals.

### Transition Queue

Actors push transitions to a shared queue; the learner consumes them:

```python
# Actor: push transition (as dict, not tensor)
transition_dict = {
    'state': state,  # Numpy arrays or lists
    'action': action,
    'reward': reward,
    'next_state': next_state,
    'done': done,
    'goal': goal,
    'transition_probs_by_action': cached_probs,  # For model-based targets
}
transition_queue.put(transition_dict)

# Learner: consume and add to replay buffer
while not transition_queue.empty():
    transition_dict = transition_queue.get_nowait()
    replay_buffer.add(**transition_dict)
```

### Configuration Options

```python
@dataclass
class Phase2Config:
    # Async training options
    async_training: bool = False        # Enable async mode
    num_actors: int = 4                 # Number of parallel actors
    actor_sync_freq: int = 100          # Steps between policy syncs
    async_min_buffer_size: int = 1000   # Min buffer before training starts
    async_queue_size: int = 10000       # Max transitions in queue
```

### Buffer Clearing in Async Mode

Buffer clearing at warmup stage transitions works the same way as synchronous training:
- Cleared at start of β_r ramp-up (stage 3→4)
- Cleared at end of β_r ramp-up (stage 4→5)

The learner tracks `last_stage` and clears when stage changes.

### Fallback Behavior

If async training fails (e.g., spawn not supported), it falls back to synchronous training:

```python
try:
    self._train_async(env, ...)
except Exception as e:
    logger.warning(f"Async training failed: {e}. Falling back to synchronous.")
    self._train_sync(env, ...)
```

---

## References

- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
- [cloudpickle on PyPI](https://pypi.org/project/cloudpickle/)
- [concurrent.futures documentation](https://docs.python.org/3/library/concurrent.futures.html)

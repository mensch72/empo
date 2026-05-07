# Implementation Plan: Speeding Up `transition_probabilities()`

**Status:** Phase 2B implemented (§3.12, §3.13); §3.15 reverted (correctness issue)  
**Date:** 2025-03-12

## 1. Overview

The `transition_probabilities()` method in `vendor/multigrid/gym_multigrid/multigrid.py` is the
performance-critical hot path for two major callers:

1. **Backward induction:** `get_dag()` and `get_dag_parallel()` call it **A^N times per
   state** (e.g., 36 times per state for a 2-agent, 6-action environment). Total calls across
   DAG construction can reach tens of millions.

2. **Phase 2 trainer:** `_precompute_transition_probs()` calls it **A_r times per env step**
   (once per robot action combination) during data collection. Additionally, `step()` is called
   afterward, which internally re-invokes the same machinery. With ~100K+ training steps, the
   cumulative cost is substantial.

Current profiling (from `examples/diagnostics/profile_transitions.py`):
- `get_state()`: ~0.02 ms
- `set_state()`: ~0.03 ms  
- `transition_probabilities()`: ~0.07 ms (includes internal set_state/get_state overhead)
- `get_dag()` for 5,000 states: ~5 s

This document proposes optimizations organized by expected impact and implementation difficulty,
covering both callers.

### 1.1 Architecture Reminder

**Core method:**
```
transition_probabilities(state, actions)          # PUBLIC wrapper: save/set/restore state
  └─ _transition_probabilities_impl(state, actions)  # CORE: optimizations + Cartesian product
       ├─ FAST PATH: ≤1 active agent, all-rotations, no conflicts → deterministic
       │    └─ _compute_successor_state()            # set_state + execute + get_state
       └─ SLOW PATH: conflict blocks / stochastic
            └─ for each outcome in Cartesian product:
                 └─ _compute_successor_state_with_unsteady()  # set_state + execute + get_state
```

Every call to `_compute_successor_state*()` does a full `set_state()` → execute actions →
`get_state()` round-trip. The public wrapper adds another `get_state()` (save) + `set_state()`
(query) + `set_state()` (restore) on top.

### 1.2 Phase 2 Trainer Call Pattern

**Data collection per env step** (`collect_transition()`, trainer.py lines 1466–1554):
```
collect_transition(state, goals, goal_weights, terminal)
  ├─ Step 1: sample_human_actions(state, goals)
  │    └─ Uses human_policy_prior (no env interaction)
  │
  ├─ Step 2: _precompute_transition_probs(state, human_actions)
  │    └─ for each robot_action_idx in 0..A_r-1:
  │         └─ transition_probabilities(state, actions)     ← A_r calls, same state
  │              ├─ get_state()     # save original        ← REDUNDANT (A_r times)
  │              ├─ set_state(state) # set query state     ← REDUNDANT (env already in state)
  │              ├─ _transition_probabilities_impl(...)     ← actual work
  │              └─ set_state(original_state)  # restore   ← needed, but repeated A_r times
  │
  ├─ Step 3: sample_robot_action(state, transition_probs_by_action)
  │    └─ Uses q_r_target network (no env interaction)
  │
  └─ Step 4: step_environment(state, robot_action, human_actions)
       └─ env.step(actions)              ← calls _transition_probabilities_impl AGAIN
       │    ├─ self.get_state()          ← REDUNDANT (env already in correct state)
       │    └─ _transition_probabilities_impl(state, actions, sample_one=True)
       │         └─ _compute_successor_state*()
       │              ├─ set_state(state)   ← REDUNDANT (env already in state from above)
       │              ├─ execute actions
       │              └─ get_state()        ← builds successor state
       └─ env.get_state()               ← REDUNDANT (successor state already returned above)
```

**Key observations:**
- The env is already in state `state` when `_precompute_transition_probs()` starts
- All A_r calls to `transition_probabilities()` query the SAME state
- Each call saves/restores, so there are A_r redundant save+restore round-trips
- After precompute, `step()` re-derives what was already computed (the transition for the
  chosen robot action is already in `transition_probs_by_action`)
- `step()` internally calls `get_state()` even though the env is already in the right state
- `step_environment()` calls `get_state()` to return the next state, but `_compute_successor_state*()`
  already built and returned it

## 2. Bottleneck Analysis

### 2.1 Redundant State Save/Restore in Public Wrapper

**Location:** Lines 5202–5222 (`transition_probabilities()`)

```python
original_state = self.get_state()    # ~0.02ms — save
self.set_state(state)                # ~0.03ms — set query state
try:
    return self._transition_probabilities_impl(...)
finally:
    self.set_state(original_state)   # ~0.03ms — restore
```

When called from `get_dag()`, the caller does **not** need the environment restored to its
original state—it will immediately call `set_state()` for the next state anyway. This
save/restore overhead adds ~0.08 ms per call (>100% of the actual computation).

**Impact:** High. ~0.08 ms × millions of calls = seconds of pure waste.

### 2.2 Per-Outcome `set_state()` + `get_state()` Round-Trip

**Location:** `_compute_successor_state()` (line 5763) and `_compute_successor_state_with_unsteady()` (line 5822)

Both methods begin with `self.set_state(state)` and end with `return self.get_state()`. For the
deterministic fast path this means one round-trip. For the Cartesian-product slow path, **every
outcome** in the product does a full round-trip—e.g., a 2×2 conflict produces 4 round-trips.

`set_state()` does substantial work (lines 4952–5138):
- Two-pass agent placement (remove all agents, then re-place)
- Restore terrain_grid
- Restore mobile objects (blocks/rocks) by clearing and re-placing
- Restore mutable objects (doors, boxes, magic walls, buttons)

`get_state()` (lines 4779–4910):
- Iterate all agents to extract position, direction, carrying state
- Iterate mobile object cache to extract positions
- Iterate mutable object cache to extract state
- Sort mobile objects for deterministic ordering

**Impact:** High. This is the dominant cost in the hot path.

### 2.3 Numpy Array Allocations in Inner Loops

Throughout the hot path, numpy arrays are created repeatedly:

| Location | Allocation | Frequency |
|----------|-----------|-----------|
| `set_state()` line 4992 | `np.array([pos_x, pos_y])` per agent | Every set_state call |
| `_move_agent_to_cell()` line 3644 | `np.array(target_pos)` | Every agent movement |
| `_push_objects()` line 3621 | `agent.pos = np.array(start_pos)` | Every push |
| `Agent.front_pos` line 1922 | `self.pos + self.dir_vec` (numpy add) | Every front_pos access |
| `DIR_TO_VEC` line 691–700 | Pre-allocated `np.array` tuples | Read-only, OK |
| `_can_push_objects()` line 3560 | `np.array(start_pos)` | Every push check |
| `_process_unsteady_forward_agents()` lines 4031, 4057 | `np.array(target_pos)` | Per unsteady agent |

For a 2-agent, 6-action environment with 5,000 states: 36 × 5,000 = 180,000 calls, each
creating 2+ numpy arrays = 360,000+ small allocations from `set_state()` alone.

**Impact:** Medium. Each allocation is small (~1 µs) but they accumulate.

### 2.4 Full Grid Scan for ControlButton in `_execute_single_agent_action()`

**Location:** Lines 3946–3956

```python
for j in range(self.grid.height):
    for ii in range(self.grid.width):
        cell = self.grid.get(ii, j)
        if (cell is not None and cell.type == 'controlbutton' and 
            cell._awaiting_action and cell.controlled_agent == agent_idx):
```

This runs once per agent action. For a 10×7 grid with 2 agents, that's 140 cell lookups per
`_compute_successor_state*()` call. Most environments have zero ControlButtons.

**Impact:** Medium. Constant factor, but adds up over millions of calls.

### 2.5 Redundant `hasattr()` Checks in Inner Loops

In `_identify_conflict_blocks()` (line 5571):
```python
elif hasattr(self.actions, 'pickup') and action == self.actions.pickup:
```

And `_is_still_action()` (line 3862):
```python
still_action = getattr(self.actions, 'still', None)
return still_action is not None and action == still_action
```

These checks happen per-agent per-call. The action set never changes after construction.

**Impact:** Low individually, but called millions of times.

### 2.6 Repeated `front_pos` Computation

`Agent.front_pos` (line 1922: `return self.pos + self.dir_vec`) creates a new numpy array every
time. The same agent's front_pos may be computed multiple times within a single
`transition_probabilities()` call:
- Once in the early exit checks (lines 5273, 5298)
- Once in `_categorize_agents()` (line 4190)
- Once in `_identify_conflict_blocks()` (line 5553)
- Once in `_execute_single_agent_action()` (line 3879)

**Impact:** Low-medium. 4× redundant numpy additions per agent.

### 2.7 Mobile Object Sorting in `get_state()`

**Location:** Line 4903

```python
mobile_objects.sort(key=lambda obj: (obj[0], obj[1], obj[2]))
```

This sort runs on every `get_state()` call, even when mobile objects haven't changed (common
case). The sorted order is needed for deterministic hashing.

**Impact:** Low for small object counts, but unnecessary in the common case.

### 2.8 [Phase 2] Redundant Save/Restore Across Batched Precompute Calls

**Location:** `_precompute_transition_probs()` (trainer.py lines 1556–1597)

This method calls `transition_probabilities(state, actions)` in a loop for each of A_r robot
action combinations. All A_r calls query the **same state**. But each call independently:
1. Saves the original env state via `get_state()` → ~0.02 ms
2. Sets the query state via `set_state(state)` → ~0.03 ms
3. Computes transition probabilities
4. Restores original state via `set_state(original_state)` → ~0.03 ms

Since the env is already in `state` before the loop starts (the trainer tracks
`actor_state.state` and the env is kept in sync), step (2) is a no-op that still incurs full
cost. And step (4) of call *i* is immediately undone by step (2) of call *i+1*. The net result
is A_r × ~0.08 ms of pure overhead that could be amortized to a single ~0.03 ms restore.

**Impact:** High for Phase 2. With A_r=6 robot actions, this is ~0.48 ms of wasted save/restore
per env step. Over 100K env steps: ~48 seconds of pure overhead.

### 2.9 [Phase 2] `step()` Duplicates Already-Computed Transition

**Location:** `step_environment()` (trainer.py lines 899–930) → `step()` (multigrid.py lines 4211–4247)

After `_precompute_transition_probs()` computes transition probabilities for ALL robot actions,
`step_environment()` calls `env.step(actions)` which internally:
1. Calls `self.get_state()` to capture the current state (~0.02 ms)
2. Calls `_transition_probabilities_impl(state, actions, sample_one=True)` to sample one outcome
3. This invokes `_compute_successor_state*()` which does `set_state()` + execute + `get_state()`

But the transition for the chosen (robot_action, human_actions) is **already in**
`transition_probs_by_action[selected_action_idx]`. For deterministic transitions (the common
case), the single successor state is already computed and stored. For probabilistic transitions,
the full distribution is stored—we just need to sample from it.

After step(), `step_environment()` additionally calls `env.get_state()` to return the next
state, but `_compute_successor_state*()` already computed this inside step().

**Impact:** High for Phase 2. An entire redundant `_transition_probabilities_impl()` call
(~0.07 ms) + an extra `get_state()` (~0.02 ms) per env step. Over 100K env steps: ~9 seconds.

### 2.10 [Phase 2] Environment Already in Correct State Before Calls

**Location:** Throughout `collect_transition()` (trainer.py lines 1466–1554)

The Phase 2 trainer maintains `actor_state.state` and keeps the environment in sync. At the
start of `collect_transition()`, the env is already in state `state`. Yet:

- `transition_probabilities()` does `set_state(state)` redundantly
- `step()` does `get_state()` redundantly (the env is in `state`; `get_state()` returns `state`)

These are instances of the general §2.1 problem but amplified because the Phase 2 trainer
makes the guarantee that the env is always in the correct state—a guarantee the general
`transition_probabilities()` wrapper cannot assume.

**Impact:** Medium. ~0.05 ms wasted per env step (1 redundant set_state + 1 redundant get_state).

## 3. Proposed Optimizations

### Optimizations for `get_dag()` (Backward Induction)

### 3.1 [HIGH IMPACT] Skip State Save/Restore via Caller Protocol

**Approach:** Add an internal variant `_transition_probabilities_no_restore()` that skips the
save/restore overhead. `get_dag()` and `get_dag_parallel()` would call this variant, since they
manage state externally.

**Changes:**
- Add `_transition_probabilities_no_restore(state, actions)` that calls `set_state(state)` once
  and then `_transition_probabilities_impl()`, without saving/restoring
- Modify `get_dag()` (world_model.py line 510) to call the no-restore variant
- Keep the public `transition_probabilities()` unchanged for external callers

**Savings:** ~0.05 ms per call (eliminate redundant `get_state()` for save + `set_state()` for
restore). With 180,000 calls, saves ~9 s.

**Risk:** Low. `get_dag()` already calls `set_state()` at the top of each iteration. The
no-restore variant is purely an internal optimization.

**Alternative:** Pass a `skip_restore=False` keyword argument to `transition_probabilities()`.
This is simpler but exposes internal optimization details in the public API.

### 3.2 [HIGH IMPACT] Deterministic Fast Path Without Full `set_state()`/`get_state()` 

**Approach:** For the most common case (deterministic transition: ≤1 active agent, no unsteady
ground, no magic walls, no conflicts), compute the successor state *directly from the input
state tuple* without going through set_state → execute → get_state.

**Observation:** Most transitions are deterministic (especially in typical EMPO environments
with 2–3 agents where at most one agent is non-still in any given action profile). For these,
the successor state can be computed purely from the state tuple:

1. **Still/terminated/paused agents:** State unchanged.
2. **Left/right rotations:** Only `dir` changes in the agent tuple: `(dir ± 1) % 4`.
3. **Forward into empty cell:** Update `pos_x, pos_y` based on direction, step_count += 1.
4. **Forward into wall/occupied:** No state change except step_count += 1.
5. **Forward into overlappable terrain:** Same as empty cell.

**Sketch:**
```python
def _fast_deterministic_successor(self, state, actions):
    """Compute successor directly from state tuple for deterministic cases."""
    step_count, agent_states, mobile_objects, mutable_objects = state
    new_step = step_count + 1
    new_agent_states = list(agent_states)
    
    for i, action in enumerate(actions):
        agent = agent_states[i]
        pos_x, pos_y, dir_, terminated, started, paused, *rest = agent
        
        if terminated or paused or not started or self._is_still_action(action):
            continue
        
        if action == self.actions.left:
            new_agent_states[i] = (pos_x, pos_y, (dir_ - 1) % 4, *agent[3:])
        elif action == self.actions.right:
            new_agent_states[i] = (pos_x, pos_y, (dir_ + 1) % 4, *agent[3:])
        elif action == self.actions.forward:
            dx, dy = _DIR_TO_DELTA[dir_]  # Pure-int lookup, no numpy
            new_x, new_y = pos_x + dx, pos_y + dy
            # Check grid for walkability (needs grid access but NOT set_state)
            cell = self.grid.get(new_x, new_y)
            if cell is None or cell.can_overlap():
                new_agent_states[i] = (new_x, new_y, dir_, *agent[3:])
            # ... handle push, goal, etc.
    
    return (new_step, tuple(new_agent_states), mobile_objects, mutable_objects)
```

**Savings:** Eliminates `set_state()` (~0.03 ms) + `get_state()` (~0.02 ms) = ~0.05 ms per
deterministic transition. For a 2-agent environment, ~90% of action profiles are deterministic,
so this saves 0.05 × 0.9 × 180,000 ≈ 8 s.

**Risk:** Medium. Must correctly handle ALL grid object interactions without set_state. The grid
is already in a valid state from a previous set_state, so cell lookups work. But:
- Forward movement must check `_initial_agent_positions` (chain conflict prevention)
- Carrying/pickup/drop logic modifies the grid and would need special handling
- Object pushing modifies multiple grid cells

**Recommended scope:** Start with the three most common deterministic cases only:
1. All agents still/terminated/paused → return `(step+1, agent_states, mobile, mutable)`
2. All actions are rotations → compute new directions directly  
3. Single active agent doing forward into empty/overlappable cell

These three cases likely cover >80% of calls.

### 3.3 [HIGH IMPACT] Avoid Redundant `set_state()` Within Outcome Enumeration

**Approach:** In the Cartesian-product slow path (lines 5452–5506), every outcome calls
`_compute_successor_state_with_unsteady()` which begins with `self.set_state(state)`. But the
state is the *same* for every outcome—only the ordering/outcomes differ.

**Option A — Reset only what changed:** After computing one outcome, only undo the changes from
the action execution (move agents back, restore grid cells) instead of doing a full
`set_state()`.

**Option B — Lightweight restore from diff:** Track which grid cells were modified during action
execution and restore only those cells. This is faster than full `set_state()` for small diffs.

**Option C — Pre-compute shared work:** Since `set_state()` and the initial_agent_positions
computation is identical for all outcomes, do it once before the loop and implement a lightweight
"reset to base" that skips agent placement/terrain that hasn't changed.

**Savings:** For k outcomes, saves (k-1) × 0.03 ms of `set_state()` overhead. A 2-block conflict
with 2 agents each has 4 outcomes → saves ~0.09 ms. With stochastic elements, savings can be
larger.

**Risk:** Medium-high. Correctly tracking and reverting grid mutations is tricky and
error-prone. Option C is the safest.

### 3.4 [MEDIUM IMPACT] Replace `numpy.array` With Plain Tuples for Positions

**Approach:** Agent positions are fundamentally 2-element integer pairs. Using numpy arrays for
them introduces allocation overhead and prevents Python-level optimizations. Replace
`np.array([x, y])` with `(x, y)` tuples throughout the hot path.

**Changes needed:**
- `Agent.pos`: Change from `np.array` to tuple `(x, y)`
- `Agent.front_pos`: Return `(pos[0] + dx, pos[1] + dy)` using integer arithmetic
- `Agent.dir_vec`: Return integer tuple `(dx, dy)` instead of `np.array`
- `_move_agent_to_cell()`: Remove `np.array(target_pos)` conversion
- `set_state()`: Set `agent.pos = (pos_x, pos_y)` instead of `np.array`
- `Grid.get()/set()`: Already work with `*pos` unpacking, no change needed
- `_can_push_objects()`, `_push_objects()`: Use tuple arithmetic instead of numpy

**New constant (replaces `DIR_TO_VEC`):**
```python
_DIR_TO_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))  # right, down, left, up
```

**Savings:** Eliminates ~360,000 `np.array` allocations per 5,000-state DAG traversal. Each
allocation takes ~1 µs → saves ~0.4 s. Also makes front_pos ~10× faster (tuple add vs numpy
add).

**Risk:** High. This is a pervasive change that affects rendering, observation generation, and
many other methods throughout the multigrid codebase. Requires careful migration:
1. Agent rendering code uses `agent.pos` as numpy for slicing/indexing
2. `encode_for_agents()` uses numpy operations on positions
3. External code may depend on numpy array interface

**Recommendation:** Introduce a parallel `pos_tuple` property or do the migration in stages,
starting with the hot path only. Alternatively, cache the tuple representation alongside the
numpy one.

### 3.5 [MEDIUM IMPACT] Cache ControlButton Locations

**Approach:** Replace the full grid scan in `_execute_single_agent_action()` (lines 3946–3956)
with a pre-built lookup table.

**Changes:**
- In `_build_object_cache()`, also build `self._control_buttons`: a list of
  `(x, y, cell)` for all ControlButton objects
- In `_execute_single_agent_action()`, iterate over `self._control_buttons` instead of
  scanning the full grid

```python
# After _build_object_cache:
self._control_buttons = [
    (x, y, cell) for (x, y), cell in self._mutable_objects
    if cell.type == 'controlbutton'
]

# In _execute_single_agent_action:
for x, y, cell in self._control_buttons:
    if cell._awaiting_action and cell.controlled_agent == agent_idx:
        if cell._just_activated:
            cell._just_activated = False
        else:
            cell.record_action(action)
```

**Savings:** Eliminates W×H grid scan per agent action. For 10×7 grid with 2 agents:
140 → ~2 lookups per call. Over 180,000 calls: saves ~0.01 ms × 180,000 ≈ 1.8 s (estimated).

**Risk:** Low. ControlButton positions are fixed (mutable_objects don't move). The cache is
already partially built; this just extends it. Environments without ControlButtons get
`self._control_buttons = []`, making the loop zero-cost.

### 3.6 [MEDIUM IMPACT] Cache Action Properties at Construction Time

**Approach:** Pre-compute action-set-dependent flags at environment construction instead of
checking `hasattr()` and `getattr()` in inner loops.

```python
# In __init__ or reset:
self._has_still = hasattr(self.actions, 'still')
self._still_action = getattr(self.actions, 'still', -1)  # -1 = no still
self._has_pickup = hasattr(self.actions, 'pickup')
self._has_drop = hasattr(self.actions, 'drop')
self._has_toggle = hasattr(self.actions, 'toggle')
self._has_build = 'build' in self.actions.available

# Replace _is_still_action:
def _is_still_action(self, action):
    return self._has_still and action == self._still_action
```

**Savings:** Eliminates `hasattr()`/`getattr()` overhead (~0.5 µs each). With 6+ checks per
call × 180,000 calls → saves ~0.5 s.

**Risk:** Very low. Action set is immutable after construction.

### 3.7 [MEDIUM IMPACT] Pre-Compute `front_pos` and Cache Per-Call

**Approach:** Compute each agent's front position once at the start of
`_transition_probabilities_impl()` and pass it around instead of recomputing via property access.

```python
# At top of _transition_probabilities_impl:
front_positions = {}
for i in range(num_agents):
    if not (self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started):
        fp = self.agents[i].front_pos
        front_positions[i] = (int(fp[0]), int(fp[1]))
```

Then pass `front_positions` to `_categorize_agents()`, `_identify_conflict_blocks()`, etc.

**Savings:** Eliminates 3–4 redundant numpy additions per active agent per call.

**Risk:** Low. Pure caching, no semantic change.

### 3.8 [LOW-MEDIUM IMPACT] Eliminate Mobile Object Sort in `get_state()`

**Approach:** Maintain a pre-sorted invariant for `_mobile_objects` so that `get_state()` doesn't
need to sort on every call.

**Changes:**
- Keep `_mobile_objects` sorted by `(obj.type, cur_pos_x, cur_pos_y)` at all times
- When `_push_objects()` changes a mobile object's position, update the sort order (or just
  mark the cache as dirty and re-sort lazily)
- In `get_state()`, skip the sort if the cache is clean

For the common case (no pushes), this eliminates the sort entirely. Even with pushes, a
"dirty flag + conditional sort" is cheaper than always sorting.

**Savings:** Small (~0.01 ms per get_state for typical counts), but it runs millions of times.

**Risk:** Low. Requires tracking a dirty flag in `_push_objects()` and `set_state()`.

### 3.9 [LOW IMPACT] Early Termination Check Before Action Validation

**Location:** Lines 5202–5210

Currently, action validation (loop over all actions) happens before the terminal state check.
Reversing the order saves the validation loop for terminal states:

```python
# Current:
step_count = state[0]
if step_count >= self.max_steps:
    return None
for action in actions:  # Validation
    ...

# Already optimal — terminal check is first. No change needed.
```

Actually, the terminal check IS already first (line 5203–5205). No change needed here.

### 3.10 [SPECULATIVE] Incremental State Encoding

**Approach:** Instead of rebuilding the entire state tuple in `get_state()`, maintain a running
state that is incrementally updated as actions execute.

This is the most ambitious optimization: rather than "set_state → execute → get_state", maintain
the state tuple as a mutable structure and update only the changed fields.

**Sketch:**
```python
class IncrementalState:
    """Mutable state wrapper that tracks changes for cheap get_state()."""
    __slots__ = ('step_count', 'agent_states', 'mobile_objects', 'mutable_objects', '_frozen')
    
    def freeze(self):
        """Return immutable (hashable) snapshot."""
        if self._frozen is None:
            self._frozen = (self.step_count, tuple(self.agent_states), ...)
        return self._frozen
    
    def update_agent_pos(self, idx, new_x, new_y):
        self.agent_states[idx] = (new_x, new_y, *self.agent_states[idx][2:])
        self._frozen = None  # Invalidate cache
```

**Savings:** Potentially eliminates ~50% of `get_state()` cost by avoiding tuple construction when
state hasn't changed. For successor computation, avoids full reconstruction.

**Risk:** High. Requires refactoring the entire action execution pipeline to use the incremental
state instead of directly modifying grid/agent objects. Large surface area for bugs.

### 3.11 [SPECULATIVE] Alternative State Encoding for Faster Hashing

**Observation:** The current state tuple contains nested tuples with heterogeneous types (int,
bool, None), which makes Python's tuple hashing do per-element type dispatch.

**Alternative:** Encode state as a flat `bytes` object:
```python
import struct

def get_state_bytes(self):
    """Encode state as flat bytes for fast hashing."""
    parts = [struct.pack('i', self.step_count)]
    for agent in self.agents:
        parts.append(struct.pack('iiiBBB', 
            agent.pos[0], agent.pos[1], agent.dir,
            agent.terminated, agent.started, agent.paused))
    return b''.join(parts)
```

**Benefits:**
- `hash(bytes_state)` is faster than `hash(nested_tuple)` for states with many agents
- More compact memory representation
- Faster equality comparison (single memcmp vs recursive tuple comparison)

**Drawbacks:**
- Loses readability for debugging
- Carrying objects and mutable objects need encoding/decoding schemes
- All callers that destructure the state tuple need updating

**Risk:** High. Pervasive API change. Best done as an opt-in parallel representation.

### Optimizations for Phase 2 Trainer (Data Production)

### 3.12 [HIGH IMPACT] Batch Precompute: Single Set/Restore for All Robot Actions

**Approach:** Replace A_r independent `transition_probabilities()` calls with a single batched
method that shares the save/set/restore overhead.

**Current flow** (`_precompute_transition_probs`, trainer.py lines 1556–1597):
```python
for action_idx in range(num_actions):    # A_r iterations
    ...
    trans_probs = self.env.transition_probabilities(state, actions)  # save + set + compute + restore
```

Each call does: save (~0.02 ms) + set (~0.03 ms) + compute + restore (~0.03 ms) = ~0.08 ms overhead.

**Proposed flow:**
```python
def _precompute_transition_probs(self, state, human_actions):
    result = {}
    # The env is already in `state` (trainer invariant).
    # Save once, compute A_r times, restore once.
    original_state = self.env.get_state()   # 1 save
    self.env.set_state(state)               # 1 set (no-op if env is already in state)
    try:
        for action_idx in range(num_actions):
            ...
            # Call internal method directly, bypassing save/restore wrapper
            trans_probs = self.env._transition_probabilities_impl(state, actions)
            result[action_idx] = trans_probs if trans_probs is not None else []
            # Restore query state for next iteration (undo side effects of _compute_successor_state*)
            self.env.set_state(state)
    finally:
        self.env.set_state(original_state)  # 1 restore
    return result
```

**Savings:** Reduces overhead from A_r × ~0.08 ms to 1 × ~0.05 ms + A_r × ~0.03 ms (one
restore per iteration, since `_transition_probabilities_impl` modifies env as a side effect).
For A_r=6: from ~0.48 ms to ~0.23 ms overhead per env step (2× improvement on this path).

**Risk:** Low-medium. Depends on `_transition_probabilities_impl()` being a stable internal API.
The trainer already tightly couples to the multigrid env; this is an extension of that pattern.

**Alternative (safer):** Add a `batch_transition_probabilities(state, list_of_actions)` method
to `MultiGridEnv` that implements the batched protocol internally:
```python
def batch_transition_probabilities(self, state, actions_list):
    """Compute transition probabilities for multiple action vectors from the same state."""
    original_state = self.get_state()
    self.set_state(state)
    try:
        results = []
        for actions in actions_list:
            result = self._transition_probabilities_impl(state, actions)
            results.append(result)
            self.set_state(state)  # restore between iterations
        return results
    finally:
        self.set_state(original_state)
```

This keeps the optimization encapsulated in the world model and avoids the trainer calling
internal methods.

### 3.13 [HIGH IMPACT] Skip `step()` by Sampling from Cached Transition Probs

**Approach:** When `use_model_based_targets=True` and `transition_probs_by_action` has already
been computed, skip calling `step_environment()` entirely. Instead, sample the next state
directly from the pre-computed transition probabilities and use `set_state()` to advance the
environment.

**Current flow:**
```python
# In collect_transition():
transition_probs_by_action = self._precompute_transition_probs(state, human_actions)
robot_action = self.sample_robot_action(state, transition_probs_by_action)
next_state = self.step_environment(state, robot_action, human_actions)  # calls step() + get_state()
```

**Proposed flow:**
```python
transition_probs_by_action = self._precompute_transition_probs(state, human_actions)
robot_action = self.sample_robot_action(state, transition_probs_by_action)

# Sample next_state from cached transition probs instead of calling step()
action_idx = self.networks.q_r.action_tuple_to_index(robot_action)
trans_probs = transition_probs_by_action[action_idx]

if not trans_probs:
    next_state = state  # Terminal state
else:
    probs = [p for p, _ in trans_probs]
    states = [s for _, s in trans_probs]
    chosen_idx = np.random.choice(len(trans_probs), p=probs)
    next_state = states[chosen_idx]

# Advance environment to the sampled state
self.env.set_state(next_state)
```

**Savings:** Eliminates an entire `step()` call (~0.09 ms: get_state + _transition_probabilities_impl
+ get_state) per env step. Over 100K env steps: ~9 seconds saved.

**Risk:** Medium. Must ensure that:
1. The sampled distribution matches what `step()` would produce (guaranteed by design, since
   `step()` delegates to `_transition_probabilities_impl(sample_one=True)`)
2. The environment is correctly advanced to `next_state` via `set_state()` (standard API)
3. Any side effects of `step()` that the trainer relies on are preserved—notably the observation
   generation. Currently `step_environment()` discards `obs, rewards, done, info` from `step()`,
   so no side effects are lost.
4. Visual feedback (`stumbled_cells`, `magic_wall_entered_cells`) is not needed during training

**Caveat:** When `use_model_based_targets=False`, transition_probs are NOT precomputed, so we
must fall back to `step_environment()`. The optimization only applies when model-based targets
are enabled.

### 3.14 [MEDIUM IMPACT] Skip Initial `set_state()` When Env is Already in Correct State

**Approach:** The Phase 2 trainer guarantees that the env is in state `state` at the start of
`collect_transition()`. Leverage this to skip the `set_state(state)` call in
`transition_probabilities()`.

**Option A — State comparison guard:**
```python
def transition_probabilities(self, state, actions, sample_one=False, skip_state_setup=False):
    ...
    if not skip_state_setup:
        original_state = self.get_state()
        self.set_state(state)
    ...
```

The trainer would call with `skip_state_setup=True` when it knows the env is in the right state.

**Option B — Cached state check:**
```python
# In transition_probabilities:
current = self.get_state()
if current == state:
    # Skip set_state, just compute directly
    return self._transition_probabilities_impl(state, actions, sample_one=sample_one)
```

However, this equality check on nested tuples may itself be costly (~0.01 ms for complex states),
partially negating the savings.

**Option C — Always skip in batched variant:**
Combined with §3.12, the batched variant already does a single `set_state(state)` and restores
between iterations. If the env is already in `state`, the first `set_state()` becomes a no-op
(but still incurs full cost). A `_state_equals_current` fast path could be added.

**Savings:** ~0.03 ms per call. For A_r=6: ~0.18 ms saved per env step.

**Risk:** Low for Option A (explicit caller contract). Medium for Option B (relies on state
equality being cheap).

### 3.15 [MEDIUM IMPACT] Avoid Redundant `get_state()` in `step_environment()`

**Approach:** `step_environment()` currently calls `self.env.step(actions)` (which internally
computes the successor state) and then `self.env.get_state()` to return it. But `step()`'s
internal `_compute_successor_state*()` already computed and returned the successor state—it's
just not exposed through `step()`'s return value.

**Option A — Expose successor state from step():**
Add a method to the world model that returns the successor state directly:
```python
def step_returning_state(self, actions):
    """Like step() but also returns the successor state tuple."""
    state = self.get_state()
    result = self._transition_probabilities_impl(state, actions, sample_one=True)
    if result is None:
        return self.get_state(), None, True
    return result[0][1], None, self.step_count >= self.max_steps
```

**Option B — Cache get_state() result after step():**
After `_compute_successor_state*()` returns, cache the state tuple so that the next
`get_state()` call returns it without recomputation:
```python
def step(self, actions):
    ...
    result = self._transition_probabilities_impl(state, actions, sample_one=True)
    if result is not None:
        self._cached_state = result[0][1]  # Cache the successor state
    ...

def get_state(self):
    if hasattr(self, '_cached_state') and self._cached_state is not None:
        cached = self._cached_state
        self._cached_state = None
        return cached
    # ...normal get_state logic
```

**Savings:** ~0.02 ms per env step (one `get_state()` eliminated). Minor individually but
free improvement.

**Risk:** Low for Option B. Option A changes the WorldModel API.

### 3.16 [MEDIUM IMPACT] Amortize Shared Work Across Robot Actions in Precompute

**Approach:** In `_precompute_transition_probs()`, the A_r calls to
`transition_probabilities()` share significant common work:
- Active agent identification (same for all calls if human actions are fixed and only robot
  action varies)
- Agent categorization (unsteady, magic wall checks)
- Forced action handling
- `_initial_agent_positions` computation (same for all calls)

If the robot is a single agent, only the robot's action changes between calls. All other agents'
status and front positions remain constant.

**Option A — Pre-compute shared state once:**
Factor out the common setup into a method that runs once, then iterate over robot actions:
```python
def _precompute_transition_probs_fast(self, state, human_actions):
    self.env.set_state(state)
    
    # Pre-compute shared data once
    num_agents = len(self.env.agents)
    front_positions = {i: tuple(self.env.agents[i].front_pos) for i in range(num_agents)
                       if not self.env.agents[i].terminated}
    
    for action_idx in range(num_actions):
        # Only the robot's action changes
        actions = self._build_action_vector(human_actions, robot_action)
        # Pass pre-computed data to avoid redundant work
        trans_probs = self.env._transition_probabilities_impl_fast(
            state, actions, front_positions=front_positions
        )
```

**Option B — Separate determinism check from computation:**
For many states, ALL robot actions lead to deterministic transitions (e.g., when agents are far
apart). A fast pre-check could detect this and use the cheaper deterministic path for all A_r
actions without repeating the check each time.

**Savings:** Reduces per-call overhead within `_transition_probabilities_impl()` by ~30%.
Most impactful when the number of robot actions is large.

**Risk:** Medium. Requires adding new internal methods that take pre-computed data.

### 3.17 [LOW-MEDIUM IMPACT] Reuse Transition Probs for Training Target Computation

**Approach:** The Phase 2 trainer stores `transition_probs_by_action` in the replay buffer with
each transition. During training, `_compute_model_based_v_h_e_targets()` and
`_compute_model_based_q_r_targets()` iterate over these stored transitions to compute targets.

Currently, successor states in these transition probs are raw state tuples that must be
tensorized during training. If the trainer pre-tensorized the compact features for successor
states at collection time (when the env is already set up), the training step would avoid
redundant tensorization.

**Current:**
```python
# In _compute_model_based_v_h_e_targets:
for prob, next_state in trans_probs:
    # next_state is a raw tuple → needs tensorization at training time
    all_next_states.append(next_state)
```

**Proposed:** At collection time, also compute compact features for each unique successor state:
```python
# In _precompute_transition_probs:
for next_state in unique_successor_states:
    features = self.state_encoder.tensorize_state_compact(next_state, self.env)
    successor_features[next_state] = features
```

**Savings:** Moves tensorization cost from the training step (where it's on the critical path
for gradient computation) to the data collection step (which can be amortized).

**Risk:** Medium. Increases replay buffer memory usage (storing tensors alongside state tuples).
May not be worthwhile if the tensorization cache hit rate is already high.

## 4. Behavioral Changes to Consider

### 4.1 Simplify Chain Conflict Prevention

**Current behavior:** `_initial_agent_positions` (a set) prevents agents from moving into cells
that were occupied by other agents at the start of the step. This requires recording all agent
positions at step start and checking the set for every forward movement.

**Simpler alternative:** Only prevent movement when the target cell currently contains another
agent (i.e., check `grid.get(*fwd_pos)` is an agent), rather than maintaining the initial
positions set. This changes behavior slightly—it allows agents to "chase" into vacated cells in
the same step—but is much cheaper (no set construction, no set lookups).

**Trade-off:** Slightly different game semantics. In the current model, if Agent A and Agent B
are adjacent, and A moves away while B tries to move into A's cell, B is blocked. In the
simpler model, B succeeds (if A moves first). This makes more transitions deterministic
(fewer conflicts), which actually speeds things up further.

**Risk:** Changes game semantics. Would need to update tests and verify that the changed
behavior is acceptable for EMPO's theoretical guarantees.

### 4.2 Skip ControlButton Processing When None Exist

**Current:** The full grid scan for ControlButtons runs unconditionally (lines 3946–3956).

**Change:** After `_build_object_cache()`, set a flag `self._has_control_buttons` and skip the
entire loop when False.

```python
if self._has_control_buttons:
    for x, y, cell in self._control_buttons:
        ...
```

This is a pure optimization with no behavioral change—environments without ControlButtons
(the common case) skip the check entirely.

## 5. Implementation Phases

### Phase 1: Low-Risk, High-Impact — `get_dag()` Basics (1–2 days)

1. **§3.1** — `_transition_probabilities_no_restore()` for `get_dag()`
2. **§3.6** — Cache action properties at construction time
3. **§3.5** — Cache ControlButton locations (or skip when none exist, §4.2)
4. **§3.7** — Pre-compute front_pos once per call

Expected speedup: **~2× overall** for `get_dag()`, dominated by eliminating the save/restore
overhead.

### Phase 2A: Medium-Risk, High-Impact — `get_dag()` Fast Paths (2–3 days)

5. **§3.2** — Deterministic fast path for rotations and single-agent forward
6. **§3.8** — Eliminate mobile object sort in get_state() (dirty flag)

Expected speedup: **~1.5× additional** (cumulative ~3×), dominated by avoiding set_state/get_state
for the majority of deterministic transitions.

### Phase 2B: Low-Medium Risk, High-Impact — Phase 2 Trainer (2–3 days) ✅ IMPLEMENTED

7. **§3.12** — Batch precompute: single set/restore for all robot actions ✅
8. **§3.13** — Skip step() by sampling from cached transition probs ✅
9. **§3.15** — Avoid redundant get_state() in step_environment() ❌ REVERTED (see below)

Measured speedup (A_r=4, CollectGame env):
- §3.12: 1.18x faster for precompute (~12.6s saved over 100K steps)
- §3.13: ~19.75x faster for step portion (~66.8s saved over 100K steps)
- §3.15: REVERTED — The _cached_state optimization returned stale state when env
  objects were directly modified between step() and get_state() (e.g., setting
  control_button.controlled_agent in tests/setup code). The 0.02ms/step savings
  does not justify the correctness risk from stale cache.

### Phase 3: Medium-Risk, Medium-Impact (3–5 days)

10. **§3.4** — Replace numpy arrays with tuples for positions in hot path
11. **§3.3** — Lightweight reset between Cartesian-product outcomes
12. **§3.14** — Skip initial set_state when env is already in correct state
13. **§3.16** — Amortize shared work across robot actions in precompute

Expected speedup: **~1.3× additional** (cumulative ~4× for get_dag, ~3× for Phase 2 actor).

### Phase 4: High-Risk, Speculative (1–2 weeks)

14. **§3.10** — Incremental state encoding
15. **§3.11** — Alternative state encoding (bytes)
16. **§3.17** — Pre-tensorize successor states at collection time
17. **§4.1** — Simplify chain conflict prevention (behavioral change)

Expected speedup: **~1.5× additional** (cumulative ~6×), but high implementation risk.

## 6. Testing Strategy

### 6.1 Correctness Invariant

The **critical invariant** that must hold after every optimization:

> `step()` and `transition_probabilities()` produce identical distributions over successor
> states for any given (state, actions) pair.

This is verified by `tests/test_statistical_correctness.py` which runs chi-squared tests
comparing `step()` sampling against `transition_probabilities()` enumeration.

### 6.2 Regression Tests

1. Run existing `tests/test_state_management.py` after each phase
2. Run `tests/test_statistical_correctness.py` to verify distribution correctness
3. Run `get_dag()` on reference environments and verify identical DAGs (same states, same
   transitions, same probabilities)

### 6.3 Performance Benchmarks

Extend `examples/diagnostics/profile_transitions.py` to benchmark:
- `transition_probabilities()` per-call time (current: ~0.07 ms)
- `get_dag()` total time for reference environments (current: ~5 s for 5,000 states)
- Breakdown by fast-path vs. slow-path percentage

**Benchmark environments:**
- `single_agent_7x7.yaml` (1 agent, small grid → fast-path dominated)
- `two_agents.yaml` (2 agents, medium grid → mixed)
- `all_agent_colors.yaml` (6 agents, small grid → slow-path dominated)
- Any environment with ControlButtons (to measure §3.5 impact)

### 6.4 Phase 2 Trainer Integration Tests

For Phase 2B optimizations (§3.12–§3.13):

1. **Distribution equivalence:** Run `collect_transition()` N times with the optimized path and
   verify the distribution of next_states matches the original `step_environment()` path
   (chi-squared test)
2. **Replay buffer contents:** Verify that transitions stored in the replay buffer have correct
   `transition_probs_by_action`, `state`, `next_state` fields after optimization
3. **End-to-end training:** Run a short training loop (100 steps) with both original and
   optimized paths and verify identical loss trajectories (with same random seed)
4. **Profiler sections:** Use the existing `TrainingProfiler` sections (`transition_probabilities`,
   `step_environment`, `actor_total`) to measure before/after timing
5. **Edge cases:** Test with:
   - `use_model_based_targets=True` AND `False` (§3.13 only applies when True)
   - Deterministic transitions (single agent) and probabilistic (multi-agent conflicts)
   - Terminal transitions (step_count >= max_steps)
   - Environments with stochastic elements (unsteady ground, magic walls)

### 6.5 Behavioral Change Validation

If §4.1 (simplified chain conflict) is implemented, create a dedicated test that:
1. Sets up a scenario where chain conflict prevention matters
2. Verifies the new behavior matches the new specification
3. Documents the behavioral difference

## 7. Open Questions

1. **Should the fast-path (§3.2) handle pickup/drop/toggle?** These are less common but not
   rare. Adding them to the fast path increases complexity but may be worthwhile for
   environments with inventory mechanics.

2. **Is the chain conflict prevention (§4.1) semantically important for EMPO theory?** If the
   theoretical results are robust to minor behavioral changes in the transition model, the
   simplification is worth pursuing.

3. **Should we consider Cython/C extension for the hot path?** The inner loop of
   `_compute_successor_state` is pure Python with no external dependencies. A Cython
   implementation could provide 10–50× speedup but adds build complexity.

4. **Can `get_dag()` batch multiple `transition_probabilities()` calls?** Instead of calling
   once per action profile, compute all A^N transitions for a state in a single method that
   shares the set_state overhead.

5. **Should §3.13 (skip step, sample from cached probs) be the default even when
   `use_model_based_targets=False`?** Currently the optimization only activates when model-based
   targets are enabled because that's when `_precompute_transition_probs` runs. We could make
   precomputation unconditional so that step() is always bypassed, but this adds A_r
   transition_probabilities calls per env step even when not needed for targets.

6. **How should §3.12 (batch precompute) interact with `_transition_probabilities_impl()`'s
   internal `set_state()` calls?** Each outcome computation in `_compute_successor_state*()`
   calls `set_state(state)` internally. The batched variant's inter-iteration `set_state(state)`
   restore may be redundant with the first `set_state(state)` inside the next call's
   `_compute_successor_state*()`. This could be eliminated if the internal method accepted a
   "state already set" flag.

7. **Should the batched API (§3.12) be on WorldModel or on the trainer?** Adding
   `batch_transition_probabilities()` to WorldModel makes it reusable by `get_dag()` too.
   But it couples the base class to a specific usage pattern. An alternative is a mixin or
   utility function.

## 8. References

- `vendor/multigrid/gym_multigrid/multigrid.py` — Main implementation (5,923 lines)
- `vendor/multigrid/PROBABILISTIC_TRANSITIONS.md` — Conflict block algorithm documentation
- `src/empo/world_model.py` — `WorldModel` base class and `get_dag()` implementation
- `src/empo/learning_based/phase2/trainer.py` — Phase 2 trainer (data production + training)
- `src/empo/learning_based/phase2/replay_buffer.py` — `Phase2Transition` and replay buffer
- `examples/diagnostics/profile_transitions.py` — Existing profiling script
- `tests/test_statistical_correctness.py` — Distribution correctness tests
- `tests/test_state_management.py` — State management tests
- `tests/test_phase2.py` — Phase 2 trainer tests

## 9. Summary

### For `get_dag()` (Backward Induction)

The biggest gains come from eliminating redundant state save/restore operations (§3.1) and
avoiding the full set_state/get_state round-trip for deterministic transitions (§3.2). Together,
these two optimizations should provide a ~3× speedup for `get_dag()` on typical environments.

Additional medium-impact optimizations (caching, numpy→tuple, lightweight resets) can push the
total speedup to ~4–6×. The speculative optimizations (incremental state, alternative encoding)
offer further gains but carry significant implementation risk.

### For Phase 2 Trainer (Data Production)

The biggest gains come from:
1. **§3.13 — Skip step()**: Eliminates an entire redundant transition computation per env step
   by sampling from the already-computed `transition_probs_by_action` (~0.09 ms saved/step).
2. **§3.12 — Batch precompute**: Amortizes save/restore overhead across A_r robot action calls
   (~0.25 ms saved/step for A_r=6).

Together, these reduce the actor step cost by ~2–3×. Combined with the shared optimizations
(§3.5, §3.6, §3.7) and the Phase 3 items (§3.14, §3.16), the cumulative improvement for Phase 2
data collection is estimated at ~3–4×.

### Recommended Approach

1. **Phase 1** (get_dag basics) and **Phase 2B** (Phase 2 trainer) can be implemented in
   parallel since they affect different callers.
2. Benchmark after each phase to guide prioritization of subsequent phases.
3. Phase 3 and 4 should be attempted only if measured profiles justify the risk.

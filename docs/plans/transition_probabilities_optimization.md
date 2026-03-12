# Implementation Plan: Speeding Up `transition_probabilities()`

**Status:** Planning  
**Date:** 2025-03-12

## 1. Overview

The `transition_probabilities()` method in `vendor/multigrid/gym_multigrid/multigrid.py` is the
performance-critical hot path for DAG construction in backward induction (`get_dag()` and
`get_dag_parallel()`). For an environment with `A` actions and `N` agents, `get_dag()` calls
`transition_probabilities()` **A^N times per state**—e.g., 36 times per state for a 2-agent
environment with 6 actions, or 46,656 times for 6 agents. Total calls across DAG construction
can reach tens of millions.

Current profiling (from `examples/diagnostics/profile_transitions.py`):
- `get_state()`: ~0.02 ms
- `set_state()`: ~0.03 ms  
- `transition_probabilities()`: ~0.07 ms (includes internal set_state/get_state overhead)
- `get_dag()` for 5,000 states: ~5 s

This document proposes optimizations organized by expected impact and implementation difficulty.

### 1.1 Architecture Reminder

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

## 3. Proposed Optimizations

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

### Phase 1: Low-Risk, High-Impact (1–2 days)

1. **§3.1** — `_transition_probabilities_no_restore()` for `get_dag()`
2. **§3.6** — Cache action properties at construction time
3. **§3.5** — Cache ControlButton locations (or skip when none exist, §4.2)
4. **§3.7** — Pre-compute front_pos once per call

Expected speedup: **~2× overall** for `get_dag()`, dominated by eliminating the save/restore
overhead.

### Phase 2: Medium-Risk, High-Impact (2–3 days)

5. **§3.2** — Deterministic fast path for rotations and single-agent forward
6. **§3.8** — Eliminate mobile object sort in get_state() (dirty flag)

Expected speedup: **~1.5× additional** (cumulative ~3×), dominated by avoiding set_state/get_state
for the majority of deterministic transitions.

### Phase 3: Medium-Risk, Medium-Impact (3–5 days)

7. **§3.4** — Replace numpy arrays with tuples for positions in hot path
8. **§3.3** — Lightweight reset between Cartesian-product outcomes

Expected speedup: **~1.3× additional** (cumulative ~4×).

### Phase 4: High-Risk, Speculative (1–2 weeks)

9. **§3.10** — Incremental state encoding
10. **§3.11** — Alternative state encoding (bytes)
11. **§4.1** — Simplify chain conflict prevention (behavioral change)

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

### 6.4 Behavioral Change Validation

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

## 8. References

- `vendor/multigrid/gym_multigrid/multigrid.py` — Main implementation (5,923 lines)
- `vendor/multigrid/PROBABILISTIC_TRANSITIONS.md` — Conflict block algorithm documentation
- `src/empo/world_model.py` — `WorldModel` base class and `get_dag()` implementation
- `examples/diagnostics/profile_transitions.py` — Existing profiling script
- `tests/test_statistical_correctness.py` — Distribution correctness tests
- `tests/test_state_management.py` — State management tests

## 9. Summary

The biggest gains come from eliminating redundant state save/restore operations (§3.1) and
avoiding the full set_state/get_state round-trip for deterministic transitions (§3.2). Together,
these two optimizations should provide a ~3× speedup for `get_dag()` on typical environments.

Additional medium-impact optimizations (caching, numpy→tuple, lightweight resets) can push the
total speedup to ~4–6×. The speculative optimizations (incremental state, alternative encoding)
offer further gains but carry significant implementation risk.

**Recommended approach:** Implement Phase 1 and 2 first, benchmark, then decide whether Phase 3–4
are worthwhile based on measured profiles.

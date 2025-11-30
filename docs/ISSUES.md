# Known Issues and Potential Improvements

This document lists known bugs, limitations, and potential improvements identified during documentation review. These can be imported into GitHub Issues.

---

## Bugs

### BUG-002: Custom believed_others_policy not supported in parallel mode
**Severity:** Low  
**Location:** `src/empo/backward_induction.py:292-293`  
**Description:**  
Custom `believed_others_policy` functions cannot be used with `parallel=True` due to pickling constraints.

**Current behavior:**  
```python
elif parallel:
    raise NotImplementedError("Custom believed_others_policy not supported in parallel mode yet")
```

**Suggested fix:**  
Document this limitation clearly or implement cloudpickle-based serialization.

---

### BUG-003: TabularHumanPolicyPrior marginal computation may fail
**Severity:** Low  
**Location:** `src/empo/human_policy_prior.py`  
**Description:**  
When computing marginal distribution without a goal, the initial `total = None` pattern could return `None` if no goals are generated (empty generator).

**Suggested fix:**  
Initialize with zeros or add explicit check for empty generator case.

---

## Improvements

### IMP-001: Add type hints throughout codebase
**Priority:** Medium  
**Scope:** All source files  
**Description:**  
While some modules have type hints, they are inconsistent. Adding comprehensive type hints would improve IDE support and catch bugs earlier.

**Affected files:**
- `src/empo/backward_induction.py` - partial type hints
- `src/empo/env_utils.py` - has type hints
- `vendor/multigrid/gym_multigrid/multigrid.py` - minimal type hints

---

### IMP-002: Add comprehensive test coverage for edge cases
**Priority:** Medium  
**Location:** `tests/`  
**Description:**  
Several edge cases are not covered by tests:
- Empty goal generators
- Single-agent environments
- Environments with max_steps=1
- Very large state spaces (memory limits)

---

### IMP-003: Improve error messages in transition_probabilities
**Priority:** Low  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py:2877-2886`  
**Description:**  
When `transition_probabilities` returns `None`, there's no indication of why (terminal state vs. invalid action). Consider returning more informative error information.

---

### IMP-004: Add caching for DAG computation
**Priority:** Low  
**Location:** `src/empo/world_model.py`  
**Description:**  
The `get_dag()` method recomputes the entire DAG on each call. For environments with unchanging structure, caching the result could improve performance significantly.

**Suggested approach:**
```python
def get_dag(self, use_cache=True, return_probabilities=False):
    if use_cache and hasattr(self, '_dag_cache'):
        return self._dag_cache
    # ... compute DAG ...
    if use_cache:
        self._dag_cache = result
    return result
```

---

### IMP-005: Support for gymnasium's new API (terminated/truncated)
**Priority:** Medium  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py`  
**Description:**  
The step() method still returns the old gym API with 4 values `(obs, reward, done, info)`. The new gymnasium API uses 5 values `(obs, reward, terminated, truncated, info)`.

**Current behavior:**
```python
return obs, rewards, done, {}
```

**Suggested fix:**
```python
terminated = done and self.step_count < self.max_steps
truncated = done and self.step_count >= self.max_steps
return obs, rewards, terminated, truncated, {}
```

---

### IMP-008: Add progress callback for long-running computations
**Priority:** Low  
**Location:** `src/empo/backward_induction.py`  
**Description:**  
For large state spaces, backward induction can take a long time. Adding an optional progress callback would improve user experience.

**Suggested API:**
```python
def compute_human_policy_prior(..., progress_callback=None):
    # ...
    if progress_callback:
        progress_callback(states_processed, total_states)
```

---

### IMP-009: UnsteadyGround stumble_probability not configurable via map
**Priority:** Low  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py:1391-1392`  
**Description:**  
When creating UnsteadyGround from map specification, the `stumble_probability` defaults to 0.5 and cannot be customized.

**Current behavior:**
```python
elif obj_type == 'unsteady':
    return UnsteadyGround(objects_set)
```

**Suggested fix:**  
Add extended map syntax like `U5` for 50% stumble probability, or allow post-creation modification.

---

### IMP-010: MagicWall parameters not configurable via map
**Priority:** Low  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py:1394-1396`  
**Description:**  
Similar to UnsteadyGround, MagicWall's `entry_probability` and `solidify_probability` default to fixed values when created from map specification.

---

### IMP-012: Document agent color semantics
**Priority:** Low  
**Location:** Documentation  
**Description:**  
The codebase uses colors to distinguish agent types (e.g., grey = robot, yellow = human), but this convention is only documented implicitly. Add explicit documentation.

---

## Performance Issues

### PERF-001: Object cache not updated during step()
**Severity:** Informational  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py`  
**Description:**  
The `_mobile_objects` and `_mutable_objects` caches are only built during reset(). If objects are added/removed during gameplay (e.g., boxes opened), the cache becomes stale. Current code handles this with fallback grid scans, but it's inefficient.

---

## Documentation Issues

### ~~DOC-001: README.md references outdated project structure~~ (FIXED)
**Status:** ✅ Resolved  
**Description:**  
README.md has been updated with a "Core Framework" section that documents all new `src/empo/` and `src/envs/` modules, the vendored MultiGrid modifications, and an updated project structure.

---

### ~~DOC-002: VENDOR.md doesn't list all modifications~~ (FIXED)
**Status:** ✅ Resolved  
**Description:**  
VENDOR.md now includes a comprehensive "EMPO-Specific Modifications" section documenting:
- WorldModel integration (get_state, set_state, transition_probabilities)
- New object types (Block, Rock, UnsteadyGround, MagicWall)
- New agent attributes (can_push_rocks, can_enter_magic_walls)
- Map-based environment specification
- Object caching and helper methods

---

### ~~DOC-003: Missing examples for common use cases~~ (FIXED)
**Status:** ✅ Resolved  
**Description:**  
The `examples/` directory now contains comprehensive examples covering all mentioned use cases:
- `human_policy_prior_example.py` - How to define custom goals and use backward induction for planning
- `dag_and_episode_example.py` - How to extend MultiGridEnv with custom environments  
- `magic_wall_demo.py` - How to use new object types (MagicWall)
- `blocks_rocks_animation.py` - Demonstrates Block/Rock objects
- `unsteady_ground_animation.py` - Demonstrates UnsteadyGround
- `state_management_demo.py` - How to use get_state/set_state
- And many more examples for various features

---

## Notes

This list was generated during documentation review on 2024-11-30. Some items may have been addressed in subsequent updates. Please verify each issue before creating GitHub issues.

To import these into GitHub, consider using the GitHub CLI:
```bash
gh issue create --title "BUG-001: Parallel backward induction requires 'fork' context" \
    --body "..." --label "bug"
```

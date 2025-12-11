# Known Issues and Potential Improvements

This document lists known bugs, limitations, and potential improvements identified during documentation review. These can be imported into GitHub Issues.

---

## Improvements

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
**Priority:** Low (unclear which behavior is right)  
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

## Redundant Code

### RED-002: Commented-out `Grid.decode()` method
**Priority:** Low  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py:1072-1091`  
**Description:**  
There is a commented-out static method `Grid.decode()` that was presumably used to decode grid arrays back into Grid objects. This dead code should either be removed or uncommented if needed.

**Current state:**
```python
# @staticmethod
# def decode(array):
#     """
#     Decode an array grid encoding back into a grid
#     """
#     ...
```

**Recommendation:** Remove if not needed, or uncomment and test if the functionality is required.

---

### RED-003: TODO comment for system-1 policy mixing
**Priority:** Informational  
**Location:** `src/empo/backward_induction.py:520`  
**Description:**  
There's a TODO comment indicating planned but unimplemented functionality:
```python
human_policy_priors = system2_policies # TODO: mix with system-1 policies!
```

This suggests the current implementation only uses "system-2" (deliberate/planning) policies, with planned support for "system-1" (intuitive/fast) policies not yet implemented.

**Recommendation:** Either implement the feature or document it as future work.

---

## Notes

This list was generated during documentation review on 2024-11-30. Some items may have been addressed in subsequent updates. Please verify each issue before creating GitHub issues.

To import these into GitHub, consider using the GitHub CLI:
```bash
gh issue create --title "BUG-001: Parallel backward induction requires 'fork' context" \
    --body "..." --label "bug"
```

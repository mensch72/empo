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

## Performance Issues

### PERF-001: Object cache not updated during step()
**Severity:** Informational  
**Location:** `vendor/multigrid/gym_multigrid/multigrid.py`  
**Description:**  
The `_mobile_objects` and `_mutable_objects` caches are only built during reset(). If objects are added/removed during gameplay (e.g., boxes opened), the cache becomes stale. Current code handles this with fallback grid scans, but it's inefficient.

---

## Redundant Code

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

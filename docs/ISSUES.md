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

## Phase 2 Training Pipeline Issues

The following issues were identified during an extensive review of the Phase 2 robot policy training pipeline (`examples/phase2_robot_policy_demo.py` and related modules) on 2024-12-27.

### ~~P2-BUG-001: Async training uses undefined `self.buffer` instead of `self.replay_buffer`~~ [FIXED on main]
**Priority:** High  
**Location:** `src/empo/nn_based/phase2/trainer.py:1401, 1407, 1415`  
**Description:**  
~~In the async training path (`_learner_loop`), the code references `self.buffer.size()` and `self.buffer`, but the attribute is defined as `self.replay_buffer`. This would cause an `AttributeError` if async training is enabled.~~

**Status:** **FIXED** in commit e73084b on main branch (2024-12-29)

**Fix applied:**
```python
# Changed from:
while self.buffer.size() < self.config.async_min_buffer_size:

# To:
while len(self.replay_buffer) < self.config.async_min_buffer_size:
```

All three occurrences (lines 1449, 1455, 1463 on main) have been corrected.

---

### ~~P2-BUG-002: Phase2ReplayBuffer lacks `size()` method referenced in async training~~ [FIXED on main]
**Priority:** High  
**Location:** `src/empo/nn_based/phase2/replay_buffer.py`  
**Description:**  
~~The `Phase2ReplayBuffer` class only has `__len__()` method, but the async training code in `trainer.py` calls `.size()` which doesn't exist on this class.~~

**Status:** **FIXED** in commit e73084b on main branch (2024-12-29)

**Fix applied:** Code now uses `len(self.replay_buffer)` consistently throughout. No `size()` method was added; instead, the calling code was changed to use the built-in `len()` function.

---

### ~~P2-BUG-003: Undefined `effective_beta_r` when V_r network mode is enabled during warmup~~ [FIXED]
**Priority:** Medium  
**Location:** `src/empo/nn_based/phase2/trainer.py:641-652`  
**Description:**  
~~The variable `effective_beta_r` is defined inside the `if q_r_active:` block (line 620) but is used in the subsequent `if self.config.v_r_use_network:` block (line 647). This would cause a `NameError` if V_r network mode is enabled (`v_r_use_network=True`) during a warmup stage where Q_r is not yet active.~~

**Status:** **FIXED** in commit 636677e (2024-12-29)

**Fix applied:**
```python
# Moved effective_beta_r definition outside both conditional blocks
# Before the batch loop (line ~575)
effective_beta_r = self.config.get_effective_beta_r(self.training_step_count)

for transition in batch:
    # ... rest of the code uses effective_beta_r as needed
```

---

### ~~P2-DISC-001: Discrepancy between compact_features tuple size in docstring vs usage~~ [FIXED]
**Priority:** Medium  
**Location:** `src/empo/nn_based/phase2/replay_buffer.py:75-76`  
**Description:**  
~~The `push()` method signature indicated `compact_features` as a 3-tuple, but it was used as a 4-tuple including `compressed_grid`.~~ Fixed to correctly use 4-tuple type hints matching `Phase2Transition`.

---

### ~~P2-DISC-002: U_r loss computed in `compute_losses()` but conditionally used~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:764-837`  
**Description:**  
~~U_r forward pass and loss computation were always performed even when not used for training.~~ Fixed to only compute U_r loss when `u_r_use_network=True`.

---

### ~~P2-DOC-001: WARMUP_DESIGN.md Stage 2 description mentions X_h dependency incorrectly~~ [FIXED]
**Priority:** Low  
**Location:** `docs/WARMUP_DESIGN.md:55-58`  
**Description:**  
~~The documentation stated "X_h predicts where the state will be after human and robot actions" which is incorrect.~~ Fixed to correctly describe X_h as predicting aggregate goal achievement ability.

---

### ~~P2-DOC-002: Config docstring mentions V_r warmup stage that doesn't exist~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:93`  
**Description:**  
~~The comment mentioned "warmup_v_r_steps would be set to 0 if v_r_use_network=False" but there is no `warmup_v_r_steps` parameter.~~ Fixed by removing the confusing comment.

---

### ~~P2-DOC-003: Stale example in WARMUP_DESIGN.md~~ [FIXED]
**Priority:** Low  
**Location:** `docs/WARMUP_DESIGN.md:211`  
**Description:**  
~~The "Shorter Warmup for Simple Environments" example had an extra triple-backtick creating empty formatting.~~ Fixed by removing the duplicate backticks.

---

### ~~P2-PERF-001: Model-based targets recompute transition probabilities for some cases~~ [INVALID]
**Status:** Invalid - transition probabilities ARE cached at collection time  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:379-397`  
**Description:**  
The fallback path in `_compute_model_based_targets_for_transition()` that recomputes transition probabilities is a safety net that should rarely be hit. Transition probabilities are pre-computed at collection time via `_precompute_transition_probs()` (lines 422-468 in base trainer) and stored in `Phase2Transition.transition_probs_by_action`. The trainer uses these cached values via `cached_trans_probs = t.transition_probs_by_action` (line 698).

The fallback only triggers when:
1. `use_model_based_targets=True` but Q_r was not active during collection (early warmup)
2. The environment doesn't support `transition_probabilities()`

This is correct design, not a performance issue.

---

### ~~P2-PERF-002: Goal encoder forward pass called inside loop without batching~~ [FIXED]
**Priority:** Medium  
**Location:** *(removed from `src/empo/nn_based/multigrid/phase2/trainer.py`)*  
**Description:**  
~~The old `tensorize_goals_batch()` method encoded goals one at a time in a loop with individual NN forward passes.~~ Fixed by removing the unused method. The codebase already has `_batch_tensorize_goals()` (line ~1130) which properly batches goal encoding with a single forward pass.

---

### ~~P2-ORPHAN-001: `forward_with_preencoded_state` method has inconsistent input dimensions~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/multigrid/phase2/aggregate_goal_ability.py`, `src/empo/nn_based/multigrid/phase2/human_goal_ability.py`  
**Description:**  
~~The `forward_with_preencoded_state()` methods only used the shared agent encoder, not the own_agent_encoder, creating a dimension mismatch. These methods were dead code (never called).~~ Fixed by removing the broken methods.

---

### ~~P2-ORPHAN-002: `tensorize_state` method returns raw state dict, unused~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:137-152`  
**Description:**  
~~The `tensorize_state()` method is required by the abstract base class but not actually used in the multigrid trainer.~~ Fixed by updating the docstring to clearly document this is a placeholder required by the base class, and that specialized batch methods are used instead.

---

### ~~P2-ORPHAN-003: Unused `get_config()` method in base network classes~~ [FIXED]
**Priority:** Informational  
**Location:** Multiple base network classes  
**Description:**  
~~The `get_config()` abstract method is defined in base classes but its implementations in multigrid networks return configurations that don't include all parameters needed for reconstruction. For example, `MultiGridRobotQNetwork.get_config()` includes `state_encoder_config` but not `own_state_encoder_config`. This could cause issues with model serialization/loading.~~

**Status:** **FIXED** (2024-12-29)

**Fix applied:**
Updated `get_config()` methods in all network classes to include complete configurations:
- `MultiGridRobotQNetwork`: Added `own_state_encoder_config`
- `MultiGridAggregateGoalAbilityNetwork`: Added `agent_encoder_config` and `own_agent_encoder_config`
- `MultiGridHumanGoalAchievementNetwork`: Added `agent_encoder_config`

---

### ~~P2-CONF-001: Legacy 1/t learning rate decay settings are confusing alongside new sqrt decay~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:107-111`  
**Description:**  
~~The legacy `lr_x_h_use_1_over_t` and `lr_u_r_use_1_over_t` flags coexisted with new `use_sqrt_lr_decay` without clear documentation.~~ Fixed by:
1. Adding DEPRECATED markers to the legacy flags
2. Adding a deprecation warning in `__post_init__()` when legacy flags are enabled
3. Documenting that legacy flags take precedence if enabled

---

### P2-CONF-002: `auto_grad_clip` scaling may produce very small/large clip values
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:279-317`  
**Description:**  
The `get_effective_grad_clip()` method scales gradient clipping by the learning rate ratio. During 1/√t decay, learning rates become very small, which would make the effective clip very small and potentially restrict gradients too aggressively.

For example, after 10,000 steps in stage 5: `lr ≈ base_lr / 100`, so `effective_clip ≈ base_clip / 100`.

---

### ~~P2-ARCH-001: Shared vs Own encoder gradient flow not enforced programmatically~~ [FIXED]
**Priority:** Medium  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py`  
**Description:**  
~~The detachment was done with inline `tuple(t.detach() for t in s_encoded)` which was error-prone.~~

Fixed by:
1. Adding `_detach_encoded_states()` helper method with clear documentation explaining which networks should use detached vs undetached outputs
2. Using the helper consistently for both `s_encoded_detached` and `x_h_s_all_encoded_detached`
3. Pre-computing detached versions of X_h state tensors instead of calling `.detach()` inline

---

### ~~P2-BUG-004: Target network update ignores per-network intervals~~ [FIXED]
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/trainer.py:794`, `src/empo/nn_based/phase2/config.py:119-122`  
**Description:**  
~~The config defines separate target update intervals for each network (`v_r_target_update_interval`, `v_h_e_target_update_interval`, `x_h_target_update_interval`, `u_r_target_update_interval`), but the training code only checks `v_r_target_update_interval` and updates ALL target networks at the same time.~~

**Status:** **FIXED** in commit 636677e (2024-12-29)

**Fix applied:**
Modified `update_target_networks()` to respect individual intervals for each network:
```python
def update_target_networks(self):
    # Update Q_r target if interval reached
    if self.training_step_count % self.config.q_r_target_update_interval == 0:
        self.networks.q_r_target.load_state_dict(self.networks.q_r.state_dict())
    # Similar checks for v_r, v_h_e, x_h, u_r
```

The method is now called every training step, and each target network checks its own interval internally.

---

### ~~P2-ARCH-002: Target network for Q_r not maintained~~ [FIXED]
**Priority:** Medium  
**Location:** `src/empo/nn_based/phase2/trainer.py:142-163`  
**Description:**  
~~The `_init_target_networks()` method creates target networks for V_r, V_h^e, X_h, and U_r, but not for Q_r. However, Q_r is used to compute the robot policy π_r for V_r computation and V_h^e model-based targets. Using the live Q_r network instead of a target network could introduce instability.~~

~~This is especially relevant during beta_r ramp-up when the policy changes rapidly.~~

**Status:** **FIXED** in commit 636677e (2024-12-29)

**Fix applied:**
1. Added `q_r_target` to `Phase2Networks` dataclass
2. Added `q_r_target_update_interval` parameter to config (default=100)
3. Initialize Q_r target network in `_init_target_networks()`
4. Update Q_r target network in `update_target_networks()` with its own interval
5. Use Q_r target network in loss computations for stable TD targets:
   - Q_r loss: Uses `q_r_target` for computing next-state Q-values and policy
   - V_r loss: Uses `q_r_target` for computing current-state policy

This provides stability during beta_r ramp-up when the policy changes rapidly.

---

### P2-EXPECT-001: Performance expectations for training convergence
**Priority:** Informational  
**Location:** N/A (architectural consideration)  
**Description:**  

**Expected Convergence Behavior:**

1. **V_h^e losses** should converge relatively quickly (within 2000-5000 steps) since they predict goal achievement probabilities which have bounded target variance.

2. **X_h and U_r losses** have irreducible variance due to Monte Carlo goal sampling. These losses will NOT converge to zero even with perfect training. A stable plateau indicates convergence.

3. **Q_r losses** similarly have irreducible variance from:
   - Stochastic transitions (when not using model-based targets)
   - The robot policy itself (power-law softmax adds stochasticity)

4. **Training time estimates** for a 7×7 grid with 2 humans, 1 robot:
   - Warmup: 3000-4000 steps (depends on config)
   - Beta ramp-up: 2000 steps
   - Full training: 10000+ steps for stable policy
   - Total: ~15000-20000 steps minimum

5. **Memory usage** scales with:
   - Replay buffer size × transition size (with compact features: ~1KB per transition)
   - Model parameters: ~4-8MB for typical hidden_dim=256

**Performance Bottlenecks:**
- Model-based target computation: O(|A|² × |S'|) per transition batch
- Transition probability caching helps but increases memory
- Goal sampling for X_h targets adds overhead

---

### P2-EXPECT-002: "Ensemble" clarification
**Priority:** Informational  
**Location:** `examples/phase2_robot_policy_demo.py`  
**Description:**  
"Ensemble" in the Phase 2 training context refers to training on an **ensemble of gridworlds** (multiple environment configurations), not:
- An ensemble of robot policies (which would require separate implementation)
- The ensemble of human agents in a single environment (which is standard multi-agent training)

The current implementation supports training on multiple gridworld configurations by resetting to different world layouts between episodes.

---

## Review Summary

### Phase 2 Training Code Review (2024-12-29)

A comprehensive review of the Phase 2 robot policy training pipeline was conducted on 2024-12-29. The review focused on `src/empo/nn_based/phase2/trainer.py`, related modules, and configuration files.

**New issues identified:**
- **P2-BUG-003**: Undefined `effective_beta_r` when V_r network mode is enabled during warmup (Medium priority)
- **P2-BUG-004**: Target network update ignores per-network intervals (Low priority)

**Previously documented issues verified:**
- **P2-BUG-001**: Async training uses `self.buffer` instead of `self.replay_buffer` (High priority)
- **P2-BUG-002**: Phase2ReplayBuffer lacks `size()` method (High priority)

**Areas reviewed:**
- Variable scoping and lifetime in loss computation
- Target network update logic
- Warmup stage transitions
- Async training implementation
- Replay buffer interface

**Testing:**
All documented bugs have been verified using direct code inspection and a verification script (`/tmp/test_phase2_issues.py`).

---

### Status Check of Issues from 2024-12-27 (Updated 2024-12-29 after main branch commits)

**FIXED on main branch (11 issues):**
- ✓ P2-DISC-001: compact_features tuple size corrected
- ✓ P2-DISC-002: U_r loss conditional on u_r_use_network AND u_r_active
- ✓ P2-DOC-001: X_h description in WARMUP_DESIGN.md corrected
- ✓ P2-DOC-002: warmup_v_r_steps reference removed from config
- ✓ P2-DOC-003: Empty code block in WARMUP_DESIGN.md removed
- ✓ P2-PERF-002: tensorize_goals_batch method removed
- ✓ P2-ORPHAN-001: forward_with_preencoded_state methods removed
- ✓ P2-CONF-001: Deprecation warnings added for legacy LR decay flags
- ✓ P2-ARCH-001: _detach_encoded_states helper method added
- ✓ **P2-BUG-001**: Fixed in commit e73084b - changed `self.buffer.size()` to `len(self.replay_buffer)`
- ✓ **P2-BUG-002**: Fixed in commit e73084b - code now uses `len()` consistently instead of `.size()` method

**STILL EXIST on main (0 issues):**
- All previously open issues have been fixed!

**FIXED in this PR (4 issues):**
- ✓ **P2-BUG-003**: `effective_beta_r` undefined in edge case - **FIXED** (commit 636677e)
  - Moved definition outside conditional blocks to ensure availability for both Q_r and V_r losses
  
- ✓ **P2-BUG-004**: Target network updates ignore per-network intervals - **FIXED** (commit 636677e)
  - Modified `update_target_networks()` to respect individual intervals for each network
  
- ✓ **P2-ORPHAN-003**: get_config() missing encoder config parameters - **FIXED**
  - Updated all network classes to include complete encoder configurations
  
- ✓ **P2-ARCH-002**: Q_r target network not maintained - **FIXED** (commit 636677e)
  - Added Q_r target network and corresponding parameter
  - Initialize and update Q_r target network with its own interval
  - Use Q_r target network in loss computations for stability

**BY DESIGN (1):**
- ⚠ **P2-CONF-002**: auto_grad_clip scaling may produce extreme values (Low priority, potential issue)

**INFORMATIONAL (remains as-is):**
- P2-ORPHAN-002: tensorize_state method exists but unused (required by base class)
- P2-PERF-001: Marked as INVALID (transition probabilities are cached correctly)
- P2-EXPECT-001: Performance expectations (informational)
- P2-EXPECT-002: "Ensemble" clarification (informational)

**Summary of main branch changes:**
- Commits e73084b, 3a39986, 90d49cc (Dec 29): Fixed async training bugs (P2-BUG-001, P2-BUG-002)
- Commit d206235 (Dec 28): Fixed separate target intervals (P2-BUG-004) - but was later reverted
- Commits 349a32d, 67b2312, d5a7f09, 958af68, 9ea5a22 (Dec 29): Refactoring that appears to have reverted some fixes
- **Commit 636677e (Dec 29)**: Fixed P2-BUG-003, P2-BUG-004, P2-ORPHAN-003, P2-ARCH-002

---

## Notes

This list was generated during documentation review on 2024-11-30. Some items may have been addressed in subsequent updates. Please verify each issue before creating GitHub issues.

Additional Phase 2 pipeline review conducted on 2024-12-27.

Comprehensive Phase 2 training code review conducted on 2024-12-29 (see Review Summary above).

To import these into GitHub, consider using the GitHub CLI:
```bash
gh issue create --title "BUG-001: Parallel backward induction requires 'fork' context" \
    --body "..." --label "bug"
```

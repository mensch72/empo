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

### P2-BUG-001: Async training uses undefined `self.buffer` instead of `self.replay_buffer`
**Priority:** High  
**Location:** `src/empo/nn_based/phase2/trainer.py:1367, 1393, 1414, 1419`  
**Description:**  
In the async training path (`_learner_loop`), the code references `self.buffer.size()` and `self.buffer`, but the attribute is defined as `self.replay_buffer`. This would cause an `AttributeError` if async training is enabled.

**Current behavior:**
```python
while self.buffer.size() < self.config.async_min_buffer_size:  # BUG: should be replay_buffer
```

**Suggested fix:**
```python
while self.replay_buffer.size() < self.config.async_min_buffer_size:
```

**Note:** This also affects lines 1393, 1414, and 1419 in the same file.

---

### P2-BUG-002: Phase2ReplayBuffer lacks `size()` method referenced in async training
**Priority:** High  
**Location:** `src/empo/nn_based/phase2/replay_buffer.py`  
**Description:**  
The `Phase2ReplayBuffer` class only has `__len__()` method, but the async training code in `trainer.py` calls `.size()` which doesn't exist on this class.

**Current behavior:**
`Phase2ReplayBuffer` implements `__len__()` but not `size()`.

**Suggested fix:** Add `size()` method or use `len(self.replay_buffer)` consistently.

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

### P2-ORPHAN-001: `forward_with_preencoded_state` method has inconsistent input dimensions
**Priority:** Low  
**Location:** `src/empo/nn_based/multigrid/phase2/aggregate_goal_ability.py:233-267`  
**Description:**  
The `forward_with_preencoded_state()` method in `MultiGridAggregateGoalAbilityNetwork` uses only the shared agent encoder, not the own_agent_encoder. This creates a dimension mismatch since `forward()` concatenates both encoders' outputs but this method only uses one.

The method would fail at runtime if called, since the combined dimension expected by `value_head` includes both encoder outputs.

---

### P2-ORPHAN-002: `tensorize_state` method returns raw state dict, unused
**Priority:** Low  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:137-141`  
**Description:**  
The `tensorize_state()` method in `MultiGridPhase2Trainer` returns `{'state': state}` which just wraps the raw state. This method is inherited from `BasePhase2Trainer` but isn't used in the multigrid trainer since direct tensor methods are used instead. Consider removing or marking as deprecated.

---

### P2-ORPHAN-003: Unused `get_config()` method in base network classes
**Priority:** Informational  
**Location:** Multiple base network classes  
**Description:**  
The `get_config()` abstract method is defined in base classes but its implementations in multigrid networks return configurations that don't include all parameters needed for reconstruction. For example, `MultiGridRobotQNetwork.get_config()` includes `state_encoder_config` but not `own_state_encoder_config`. This could cause issues with model serialization/loading.

---

### P2-CONF-001: Legacy 1/t learning rate decay settings are confusing alongside new sqrt decay
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:107-111`  
**Description:**  
The config has both:
- New `use_sqrt_lr_decay` system (1/√t decay after warmup)
- Legacy `lr_x_h_use_1_over_t` and `lr_u_r_use_1_over_t` flags

This creates potential confusion about which decay system is active. The legacy flags are checked first in `get_learning_rate()` and take precedence, which could lead to unexpected behavior if both are enabled.

---

### P2-CONF-002: `auto_grad_clip` scaling may produce very small/large clip values
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:279-317`  
**Description:**  
The `get_effective_grad_clip()` method scales gradient clipping by the learning rate ratio. During 1/√t decay, learning rates become very small, which would make the effective clip very small and potentially restrict gradients too aggressively.

For example, after 10,000 steps in stage 5: `lr ≈ base_lr / 100`, so `effective_clip ≈ base_clip / 100`.

---

### P2-ARCH-001: Shared vs Own encoder gradient flow not enforced programmatically
**Priority:** Medium  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:646`  
**Description:**  
The documentation states shared encoders are detached for all networks except V_h^e, but this detachment is done manually in `compute_losses()`:
```python
s_encoded_detached = tuple(t.detach() for t in s_encoded)
```

This manual approach is error-prone. If someone adds code that uses `s_encoded` instead of `s_encoded_detached` for Q_r/X_h/U_r, the shared encoder would receive gradients from those losses.

**Recommendation:** Consider a more robust approach like having separate `encode_for_training()` vs `encode_frozen()` methods.

---

### P2-ARCH-002: Target network for Q_r not maintained
**Priority:** Medium  
**Location:** `src/empo/nn_based/phase2/trainer.py:142-163`  
**Description:**  
The `_init_target_networks()` method creates target networks for V_r, V_h^e, X_h, and U_r, but not for Q_r. However, Q_r is used to compute the robot policy π_r for V_r computation and V_h^e model-based targets. Using the live Q_r network instead of a target network could introduce instability.

This is especially relevant during beta_r ramp-up when the policy changes rapidly.

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

## Notes

This list was generated during documentation review on 2024-11-30. Some items may have been addressed in subsequent updates. Please verify each issue before creating GitHub issues.

Additional Phase 2 pipeline review conducted on 2024-12-27.

To import these into GitHub, consider using the GitHub CLI:
```bash
gh issue create --title "BUG-001: Parallel backward induction requires 'fork' context" \
    --body "..." --label "bug"
```

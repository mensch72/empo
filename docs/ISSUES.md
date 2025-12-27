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

### P2-DISC-001: Discrepancy between compact_features tuple size in docstring vs usage
**Priority:** Medium  
**Location:** `src/empo/nn_based/phase2/replay_buffer.py:75-76`, `src/empo/nn_based/multigrid/phase2/trainer.py:262-263`  
**Description:**  
The `push()` method signature and docstring indicate `compact_features` is a 3-tuple `(global, agent, interactive)`, but in the trainer it's stored and used as a 4-tuple `(global, agent, interactive, compressed_grid)`. This inconsistency could cause confusion.

**In replay_buffer.py:**
```python
compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,  # 3 elements
```

**In trainer.py (actual usage):**
```python
compact_features = (global_feats, agent_feats, interactive_feats, compressed_grid)  # 4 elements
```

The `Phase2Transition` dataclass correctly documents this as a 4-tuple, but `push()` signature doesn't match.

---

### P2-DISC-002: U_r loss always computed even when `u_r_use_network=False`
**Priority:** Medium  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:766-826`  
**Description:**  
In `MultiGridPhase2Trainer.compute_losses()`, the U_r loss computation is always performed (lines 766-826), but the loss dict only conditionally includes it when `v_r_use_network=True` (line 919). When `u_r_use_network=False` (the default), this computation is wasted.

However, this is partially intentional since U_r values are still needed for Q_r targets. The issue is that the code computes a full loss but never uses it, which is inefficient.

---

### P2-DOC-001: WARMUP_DESIGN.md Stage 2 description mentions X_h dependency incorrectly
**Priority:** Low  
**Location:** `docs/WARMUP_DESIGN.md:57-59`  
**Description:**  
The documentation states "X_h predicts where the state will be after human and robot actions" which is incorrect. X_h predicts the aggregate goal achievement ability E[V_h^e^ζ] across goals for a human, not state transitions.

**Current text:**
```
X_h predicts where the state will be after human and robot actions.
```

**Suggested fix:**
```
X_h predicts the aggregate goal achievement ability of human h, computed as E_{g_h}[V_h^e(s, g_h)^ζ] over possible goals.
```

---

### P2-DOC-002: Config docstring mentions V_r warmup stage that doesn't exist
**Priority:** Low  
**Location:** `src/empo/nn_based/phase2/config.py:93`  
**Description:**  
The comment mentions "warmup_v_r_steps would be set to 0 if v_r_use_network=False" but there is no `warmup_v_r_steps` parameter, and V_r has no dedicated warmup stage. V_r is enabled after Q_r warmup (stage 4+) when `v_r_use_network=True`.

---

### P2-DOC-003: Stale example in WARMUP_DESIGN.md
**Priority:** Low  
**Location:** `docs/WARMUP_DESIGN.md:197-211`  
**Description:**  
The "Shorter Warmup for Simple Environments" example ends with triple backticks that look like leftover formatting. The last line is:
```python
# Total: 800-1000 steps warmup + 400 steps ramp-up
```
followed by another triple-backtick block marker that creates empty formatting.

---

### P2-PERF-001: Model-based targets recompute transition probabilities for some cases
**Priority:** Medium  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:379-397`  
**Description:**  
In `_compute_model_based_targets_for_transition()`, when `cached_trans_probs` is None, transition probabilities are recomputed on the fly. However, this fallback path recomputes for each call even when the same state/actions would be queried multiple times. The caching at collection time handles most cases, but during early warmup when transition probs aren't cached, this could be inefficient.

---

### P2-PERF-002: Goal encoder forward pass called inside loop without batching
**Priority:** Medium  
**Location:** `src/empo/nn_based/multigrid/phase2/trainer.py:486-498`  
**Description:**  
In `tensorize_goals_batch()`, goals are encoded one at a time in a loop:
```python
for goal in goals:
    goal_coords = goal_encoder.tensorize_goal(goal, self.device)
    goal_features = goal_encoder(goal_coords)  # NN forward pass per goal
    goal_features_list.append(goal_features)
```
This could be batched into a single forward pass for efficiency. While `_batch_tensorize_goals()` does batch the coordinate extraction, the same concern applies.

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

### P2-EXPECT-002: Ensemble training not explicitly demonstrated
**Priority:** Medium  
**Location:** `examples/phase2_robot_policy_demo.py` (as mentioned in problem statement)  
**Description:**  
The problem statement mentions "training on an ensemble" but the Phase 2 training pipeline doesn't have explicit ensemble support. Each training run produces a single policy. If ensemble training is intended:

1. **Multiple independent runs** would need to be orchestrated externally
2. **Ensemble inference** combining multiple Q_r predictions is not implemented
3. **Diversity mechanisms** (different seeds, architectures, or training subsets) are not built-in

If "ensemble" refers to the ensemble of humans in the multi-agent environment, that IS supported. But ensemble of robot policies would require additional implementation.

---

## Notes

This list was generated during documentation review on 2024-11-30. Some items may have been addressed in subsequent updates. Please verify each issue before creating GitHub issues.

Additional Phase 2 pipeline review conducted on 2024-12-27.

To import these into GitHub, consider using the GitHub CLI:
```bash
gh issue create --title "BUG-001: Parallel backward induction requires 'fork' context" \
    --body "..." --label "bug"
```

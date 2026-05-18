# Planning Document: Optional Multi-Step Targets and MCTS for Phase 2 DQN

**Status:** In Progress (trajectory targets, acting-time MCTS, search-policy distillation, replay-state search relabeling, and async replay freshness controls landed)  
**Date:** 2026-05-15

## Scope

This document describes a plan for extending the **DQN-style neural Phase 2 path** so that it can optionally:

1. replace the current **one-step bootstrap targets** for `V_h^e` and `Q_r` with **multi-step** or **whole-episode** targets, and
2. replace the current direct `π_r` derivation with an **MCTS-based policy improvement step** inspired by AlphaZero / AssistantZero.

The goal is to add these features as **optional execution modes** while keeping the current one-step target path and direct `Q_r → π_r` path as the default baseline.

## Current implementation status

The repository no longer matches the original "planning only" state of this document.
The following pieces have already landed in the current code:

- `Phase2Config` now exposes `v_h_e_target_mode`, `q_r_target_mode`, `v_h_e_n_step`, `q_r_n_step`, and `pi_r_mode`, with trajectory-aware helper methods in `src/empo/learning_based/phase2/config.py`.
- `Phase2ReplayBuffer` now stores `episode_id`, `env_step_index`, and optional MCTS root statistics (`search_policy`, `search_value`, `search_action_value`) and can recover ordered episode suffixes in `src/empo/learning_based/phase2/replay_buffer.py`.
- The trainer now supports sampled-trajectory `n_step` / `episode` targets for both `V_h^e` and `Q_r`, including trajectory metrics and taken-action-only `Q_r` supervision outside `one_step` mode.
- Sync and async data collection now preserve episode-linkage metadata and disable mid-rollout goal resampling when trajectory targets are enabled.
- An optional acting-time MCTS path now exists for robot action selection, using `Q_r` as the prior and the frozen Phase 2 value stack as the leaf evaluator, with integration tests in `tests/test_phase2_trainer_integration.py`.
- `Phase2Config` and the trainer now support an optional `Q_r` search-policy distillation loss that trains the direct `Q_r → π_r` policy approximation against stored MCTS root visit-count distributions.
- `Phase2Config` and the trainer now support optional replay-state MCTS relabeling that refreshes sampled replay entries with current-search root statistics before policy distillation.
- Async trajectory-target runs can now prefer replay entries inserted within a configurable learner-age window via `trajectory_replay_max_age_training_steps`, while falling back to full-buffer sampling when fresh data is scarce.

So this document should now be read as a **living status-and-next-steps plan**, not as a greenfield design.

## Current Baseline

The current DQN-style Phase 2 trainer already uses model-based one-step targets:

- `V_h^e` uses a **policy-weighted one-step expectation** over robot actions and successor states, then bootstraps from `v_h_e_target`.
- `Q_r` uses a **one-step expected backup for all robot actions**, then bootstraps from `V_r(s')`, where `V_r` is either predicted directly or computed from `U_r` and `q_r_target`.
- `π_r` is obtained directly from `Q_r` through the existing power-law policy transform.

This gives strong one-step supervision, but it still has the standard weaknesses of one-step bootstrapping:

- long-horizon credit is weak,
- targets move quickly because they depend on frozen copies of still-changing networks,
- `Q_r` can learn slowly when useful signal appears many env_steps later,
- `V_h^e` can remain noisy for sparse goal achievement.

## Design Goals

- Keep the existing one-step mode unchanged and default.
- Make target horizon configurable independently for `V_h^e` and `Q_r`.
- Preserve the EMPO interpretation of `V_h^e`, `Q_r`, `V_r`, and `π_r`; this is still solution approximation in a world model, not standard reward maximization.
- Support both sync and async training.
- Reuse the existing world-model interfaces (`get_state`, `set_state`, `transition_probabilities`) instead of introducing a separate simulator abstraction.
- Make the MCTS path optional and compatible with the current direct policy path.

## Non-Goals

- This plan does not replace the PPO path.
- This plan does not require a separate policy network in the first iteration.
- This plan does not attempt to redesign the full Phase 2 warm-up schedule.

## Part I: Optional Multi-Step / Whole-Episode Targets

### A. Unifying abstraction

Add a small target-horizon abstraction so the trainer can choose between:

- `one_step` - current behavior,
- `n_step` - bootstrap after `n` env_steps,
- `episode` - no bootstrap; use the remainder of the episode.

Recommended config shape:

```python
v_h_e_target_mode: str = "one_step"   # one_step | n_step | episode
q_r_target_mode: str = "one_step"     # one_step | n_step | episode
v_h_e_n_step: int = 5
q_r_n_step: int = 5
```

This configuration has now landed.

The `episode` mode should mean “use the available remainder of the episode from the sampled transition onward”, not “run an unbounded rollout outside the configured episode length”.

### B. Replay and episode storage changes

The current replay buffer stores isolated transitions. Multi-step and episode targets need **ordered episode context**.

Extend storage so every transition can be linked to its future continuation:

- `episode_id`
- `env_step_index`
- `next_transition_index` or equivalent per-episode arrays
- cached per-episode terminal index
- optional cached search statistics at the root:
  - `search_policy`
  - `search_value`
  - `search_action_value`

Recommended structure:

- keep the current flat replay buffer API for sampling, but
- add an episode store that lets the trainer recover the suffix
  `t, t+1, ..., min(t+n, T)`.

This is important for both sync and async training, because target construction must not depend on transitions still sitting only in actor-local state.

### C. `V_h^e` multi-step targets

`V_h^e(s, g_h)` is a discounted probability-like quantity for eventual goal achievement. The natural multi-step target is therefore:

- if goal `g_h` is first achieved at step `t+τ`, target is `γ_h^τ`,
- if it is not achieved within the horizon and the episode continues, bootstrap from `V_h^e(s_{t+n}, g_h)`,
- if the horizon reaches the episode end without achievement, target is `0`.

So:

- `one_step`: current target,
- `n_step`: truncated first-achievement target plus `γ_h^n V_h^e(s_{t+n}, g_h)`,
- `episode`: Monte Carlo target over the sampled episode suffix.

Because goals are hypothetical, the target should still be computed **per training tuple `(state, human_idx, goal)`** by checking the future stored states against that goal until achievement or episode end.

Recommended first implementation:

1. use the sampled episode suffix rather than branching over all robot actions at later steps,
2. keep the current exact one-step expectation mode available,
3. document clearly that `n_step` / `episode` should reduce the long-horizon bias of one-step targets, but will usually be higher-variance and more on-policy.

This first sampled-trajectory implementation has now landed.

### D. `Q_r` multi-step targets

The current one-step target is:

`Q_r(s_t, a_t) ≈ γ_r V_r(s_{t+1})`

Unrolling equation (9) gives the natural multi-step target:

`γ_r U_r(s_{t+1}) + γ_r^2 U_r(s_{t+2}) + ... + γ_r^n V_r(s_{t+n})`

with the final bootstrap removed in `episode` mode.

For `Q_r`, there is one important design decision:

#### Recommended choice: use trajectory-based targets for the taken root action

The current implementation updates **all robot actions** per sampled state because it has one-step transition probabilities for every root action. That is much harder to preserve for multi-step targets because later steps depend on future policy choices and future human behavior.

For the optional `n_step` / `episode` modes, the recommended first step is:

- keep the current **all-actions exact backup** in `one_step` mode,
- switch to **taken-action supervision** in `n_step` / `episode` mode,
- optionally recover some all-actions supervision later by running separate search/rollout expansions from each root action.

This makes the first implementation much simpler and much closer to standard DQN / AlphaZero-style training targets.

This split is now implemented in the trainer and reflected in logged supervision metrics.

### E. Off-policy handling

Multi-step and episode targets become more sensitive to policy drift. The simplest first version should therefore favor **fresh data** over strong off-policy correction:

- keep replay relatively recent,
- allow a smaller buffer or age-based sampling when multi-step mode is enabled,
- optionally clear or downweight stale data after major `beta_r` schedule transitions,
- postpone importance sampling unless it proves necessary.

If later search relabeling is added, stored states can be relabeled with current-network search statistics instead of relying only on historical behavior.

### F. Goal resampling during rollouts

The current Phase 2 trainer can resample goals during an episode:

- immediately after a sampled goal is achieved, and
- stochastically via `goal_resample_prob`.

That behavior is helpful for one-step training because it keeps data collection moving, but it is a poor fit for multi-step / episode targets. For longer-horizon targets, the rollout suffix should represent a **single fixed hypothetical goal assignment** from the root onward.

Recommended plan:

- when collecting data for `n_step` or `episode` target modes, **disable stochastic mid-rollout goal resampling**;
- when a goal is achieved, treat that as the endpoint of the target for that `(human, goal)` pair rather than immediately assigning a new goal inside the same target suffix;
- if continued environment interaction after goal achievement is still desired for throughput, start a **new rollout segment / new replay episode record** with freshly sampled goals instead of mutating the existing one in place;
- for MCTS expansions, keep the goal profile fixed throughout the simulated tree rooted at the current state.

In other words: for the longer-horizon path, goal resampling should happen at **root selection / episode-segment boundaries**, not inside the rollout used to define a target.

The replay-segment part of this plan has now landed in the shared actor logic.

### G. Sampling of initial states

Longer-horizon targets and MCTS also need an explicit story for where rollout roots come from.

Recommended plan:

- sample initial states only at **episode boundaries** via the existing reset / `world_model_factory` machinery or an explicit initial-state sampler layered on top of it;
- treat the initial-state distribution as a configurable part of the training setup, especially in ensemble mode where each episode may come from a different world model instance;
- distinguish clearly between:
  - **training initial-state sampling**: how a new real rollout starts, and
  - **search root selection**: which already-observed state MCTS is run from;
- do **not** sample fresh initial states inside MCTS itself; search must always start from the stored root state and expand forward from there;
- if later work introduces start-state diversification beyond plain `reset()`, prefer to expose it explicitly through `WorldModel.initial_state()` or the factory layer rather than hiding it inside the target-construction code.

This keeps the semantics clean: initial states determine which parts of the world model are explored by real rollouts, while MCTS refines action choice from a given root state without changing the state distribution by itself.

### H. Model-based variants to consider later

Once the simpler sampled-trajectory version exists, a more model-based extension can be added:

- for `V_h^e`: expected `n`-step backups that branch over robot actions at each search depth,
- for `Q_r`: limited-depth tree backups using `transition_probabilities`,
- mixed targets that follow sampled trajectories for most steps and use exact expectation at the bootstrap frontier.

This should be treated as a second phase because it is much more computationally expensive.

## Part II: Optional MCTS for `π_r`

### A. High-level idea

Add an optional policy-improvement operator:

- `pi_r_mode = "direct"`: current behavior, compute `π_r` directly from `Q_r`,
- `pi_r_mode = "mcts"`: run MCTS at decision time and derive `π_r` from root visit counts.

The direct path remains the default. The MCTS path is an optional search wrapper around the existing value machinery.

### B. What serves as prior and leaf evaluator

Recommended first design:

- **prior over root actions**: current `Q_r`-derived policy,
- **leaf value**: `V_r` if present, otherwise `U_r + E_{a~π_r}[Q_r]`, where both `Q_r` and the policy used in that expectation come from the frozen target networks,
- **tree policy**: PUCT-style selection,
- **root policy output**: normalized visit counts, optionally temperature-controlled.

This uses the current networks the same way AlphaZero uses policy/value heads, but without requiring a new policy head immediately.

### C. World-model specifics

MCTS in this codebase must respect that successor generation depends on:

- robot actions,
- human actions or human action distributions,
- stochastic environment transitions.

Recommended first rollout semantics:

1. treat the world model as a generative simulator,
2. at each node, sample human actions from the same goal-conditioned Phase 1 human policy prior that is currently used to generate human actions during Phase 2 training,
3. sample or enumerate successor states from `transition_probabilities`,
4. evaluate leaves with the current frozen Phase 2 value stack.

Later, a more exact expectimax-like version can be added with explicit chance nodes for:

- human actions,
- stochastic successor states.

### D. Relationship between search policy and theoretical `π_r`

Search should be treated as an **optional approximate policy-improvement operator** for the policy used during data collection and evaluation. The plan should avoid claiming that visit counts are the literal closed-form policy from the theory.

Recommended wording in the implementation and docs:

- the search policy is an approximation scheme for computing the Phase 2 robot policy,
- the direct power-law policy remains the baseline closed-form approximation,
- MCTS is an optional refinement layer built on top of the same value equations.

### E. Training targets from search

When `pi_r_mode="mcts"`, store root search statistics in replay:

- visit-count distribution over robot actions,
- chosen root action,
- optional root value estimate.

Recommended training use:

1. **behavior only, first iteration**
   - use MCTS only for acting,
   - keep value learning on TD / multi-step targets,
   - no extra loss beyond the existing value losses.

2. **search-distillation, second iteration**
   - add an optional policy head or policy distillation loss,
   - train the direct policy approximation to match search visit counts,
   - optionally relabel old states with fresh search.

The first iteration is likely enough to test whether search materially improves data quality.

### F. Search schedule and warm-up

To keep warm-up stable:

- keep `pi_r_mode="direct"` during early warm-up when `beta_r=0`,
- enable MCTS only after `Q_r` and `V_h^e` have reached the current stable stage,
- optionally ramp search effort:
  - small number of simulations first,
  - larger search budget later.

Recommended config shape:

```python
pi_r_mode: str = "direct"              # direct | mcts
mcts_num_simulations: int = 64
mcts_max_depth: int = 8
mcts_c_puct: float = 1.5
mcts_dirichlet_alpha: float = 0.3
mcts_root_noise_frac: float = 0.25
mcts_enable_after_training_step: int = 0
```

The currently landed implementation includes `pi_r_mode`, `mcts_num_simulations`,
`mcts_max_depth`, `mcts_c_puct`, `mcts_temperature`, optional root Dirichlet
noise, and delayed MCTS enablement via training-step gating.

## Recommended Implementation Order

### Phase 1: Target-horizon infrastructure

1. [x] Add config flags for `one_step` / `n_step` / `episode`.
2. [x] Extend replay storage with episode ordering metadata.
3. [x] Add episode-suffix retrieval utilities shared by sync and async paths.

### Phase 2: `V_h^e` trajectory targets

4. [x] Implement sampled-trajectory `n_step` and `episode` targets for `V_h^e`.
5. [x] Keep current model-based one-step target unchanged as the default.
6. [x] Add metrics comparing one-step vs multi-step target variance and achievement sparsity.

### Phase 3: `Q_r` trajectory targets

7. [x] Implement taken-action `n_step` and `episode` targets for `Q_r`.
8. [x] Keep current all-actions one-step update path for baseline runs.
9. [x] Add explicit reporting when training switches from all-actions supervision to taken-action-only supervision.

### Phase 4: MCTS acting path

10. [x] Introduce a `pi_r_mode` abstraction.
11. [x] Implement a search module that can run from a saved world-model state.
12. [x] Use current `Q_r` as prior and current frozen `V_r` stack as leaf evaluator.
13. [x] Store root visit counts and chosen action in replay.

### Phase 5: Search-aware training refinements

14. [x] Evaluate whether MCTS-only acting already improves `V_h^e` and `Q_r`.
15. [x] If useful, add optional replay-state search relabeling.
16. [x] Add an optional policy-distillation loss from stored MCTS root policies.

### Phase 6: Remaining search refinements

17. [x] Add optional root Dirichlet noise support for acting-time MCTS search diversity.
18. [x] Gate MCTS enablement by an explicit training-step threshold in addition to the current warm-up / `beta_r` checks.
19. [x] Add an optional async replay-age cutoff for trajectory modes to prefer fresher data while preserving full-buffer fallback behavior.

## Validation Plan

- Unit-test `V_h^e` target construction on short synthetic episodes:
  - achieved within horizon,
  - achieved after bootstrap frontier,
  - never achieved,
  - terminal before achievement.
- Unit-test `Q_r` target construction against hand-computed discounted `U_r` / `V_r` sums.
- Verify replay episode linkage in both sync and async training.
- Compare current one-step mode and new modes on the same small world model.
- Measure:
  - target variance,
  - convergence speed,
  - policy entropy,
  - search overhead,
  - wall-clock cost per training_step.

## Main Risks

1. **Cost explosion**: search plus multi-step target construction can become too expensive in async training.
2. **Policy-drift instability**: long-horizon targets may become stale faster than one-step targets.
3. **Mismatch with state-only policy definition**: search that conditions too strongly on sampled episode goals may drift away from the intended policy semantics.
4. **Loss of current all-actions supervision**: `Q_r` may need more samples once multi-step mode updates only the taken action.

## Recommended First Milestone

That original first milestone has now landed:

1. add episode-aware replay storage,
2. add optional sampled-trajectory `n_step` / `episode` targets for `V_h^e`,
3. add optional taken-action `n_step` / `episode` targets for `Q_r`,
4. add an acting-only MCTS mode that uses current `Q_r` as prior and current `V_r` stack for leaf evaluation.

## Recommended Next Milestone

The next highest-leverage milestone is now:

1. run controlled comparisons of `one_step`, `n_step`, and `episode` targets on the same small world models,
2. measure whether MCTS-only acting improves data quality enough to justify its cost,
3. decide whether replay relabeling is worth adding on top of the new distillation path,
4. only then add extra search features such as a dedicated search-trained policy head.

## Open milestone evaluation status (codified)

The Phase 2 open-comparison milestone is now codified via:

- runnable harness: `examples/phase2/phase2_multistep_mcts_open_eval.py`
- quick run command:

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
  python examples/phase2/phase2_multistep_mcts_open_eval.py --quick \
  --output-dir docs/evaluations/phase2_open_eval
```

- latest checked-in artifacts:
  - `docs/evaluations/phase2_open_eval/open_eval_summary_20260518T114931Z.json`
  - `docs/evaluations/phase2_open_eval/open_eval_summary_20260518T114931Z.csv`
  - `docs/evaluations/phase2_open_eval/README.md`

This run compares all `3 × 2` combinations of:
- target horizon mode: `one_step`, `n_step`, `episode`
- policy mode: `pi_r_mode=direct` vs `pi_r_mode=mcts`

and reports wall-clock throughput, search-cost proxies (searched transition rate and approximate simulations), and stability metrics (`q_r` / `v_h_e` tail mean/std/CV).

# Implementing a new WorldModel

This guide walks through **every step** needed to add a brand-new world model to
EMPO so that it works with the Phase 1 / Phase 2 backward induction and the
learning-based paths, including all helpers, loaders, rendering, tests, and
example scripts.

It uses [`src/empo/bushworld/`](../src/empo/bushworld) as the running, end-to-end
reference implementation. When in doubt, read the corresponding BushWorld file —
each step below cites it. See also [`docs/BUSHWORLD.md`](BUSHWORLD.md) for the
finished world's behavior.

> **Terminology.** EMPO is *model-based planning*, not RL. Policies are
> *computed/approximated* as solutions to equations; they are not "optimized".
> `beta_*`, `gamma_*`, `zeta`, `xi`, `eta` are **theory parameters**. Use
> `training_step` vs `env_step` (never bare "step"), and "Phase 1"/"Phase 2" only
> for the two computation phases.

## Overview / checklist

A complete new world consists of a first-party world model package
(`src/empo/<yourworld>/`) plus a learning package
(`src/empo/learning_based/<yourworld>/`):

1. **`env.py`** — the `WorldModel` subclass (state, dynamics, transitions).
2. **`goals.py`** — `PossibleGoal`s + a `PossibleGoalGenerator`/`Sampler`.
3. **`human_policy.py`** — a `HumanPolicyPrior` (heuristic or learned).
4. **`loader.py`** — a YAML/map loader.
5. **`rendering.py`** — frame rendering + movie generation.
6. **`empo/learning_based/<yourworld>/`** — encoders + Phase 2 networks + trainer
   for the learning-based Phase 2 path (see step 8), mirroring
   `empo/learning_based/multigrid/` and `empo/learning_based/bushworld/`.
7. **`__init__.py`** — package exports.
8. **An example world** under `<yourworld>_worlds/…`.
9. **Tests** under `tests/test_<yourworld>.py`.
10. **An example script** under `examples/<yourworld>/…`.
11. **Docs** under `docs/`.

## Step 1 — The `WorldModel` subclass (`env.py`)

Subclass `empo.world_model.WorldModel` (which extends `gym.Env`). The interface
that the EMPO machinery relies on:

| Requirement | Notes |
|---|---|
| `get_state() -> Hashable` | Complete, hashable state. Use tuples, never lists/arrays. |
| `set_state(state) -> None` | Restore to an *exact* state. Must round-trip with `get_state`. |
| `transition_probabilities(state, actions) -> list[(prob, next_state)]` | **Exact** distribution; probabilities must sum to 1. Return a small, factorized set of outcomes. |
| `initial_state() -> Hashable` | The starting state (without permanently resetting). |
| `is_terminal(state=None) -> bool` | Terminal predicate (e.g. `step_count >= max_steps`). |
| `action_space` | A `gym.spaces.Discrete(n)`. |
| `human_agent_indices` / `robot_agent_indices` | Lists of agent indices. Keep a stable ordering (e.g. robots first). |
| `reset`, `step` | Standard gym methods (`step` samples one transition). |

Key design rules (learned the hard way):

- **State must be hashable and minimal.** BushWorld uses
  `(step_count, positions_tuple, densities_tuple)`
  (see [`env.py`](../src/empo/bushworld/env.py) `get_state`/`set_state`).
- **Keep `transition_probabilities` exact and small.** Enumerate only genuine
  stochasticity and factorize independent coins. BushWorld only branches on the
  humans' move-success coins; everything else is deterministic
  (`transition_probabilities`).
- **Provide a `possible_goal_generator` property** so callers can obtain goals
  without knowing the concrete classes (see `env.py` property).
- **Expose a `_get_construction_kwargs()`** (or equivalent) if your env needs to
  be re-created/pickled for parallel DAG workers — states must be serializable.
- If you use **non-unit durations**, also implement `transition_durations` and
  `terminal_duration`; otherwise the default (1.0 per transition) applies and
  `gamma` discounting behaves as a per-step factor.

Validate immediately:

```python
s = env.initial_state(); env.set_state(s)
assert env.get_state() == s and hash(s) is not None
for actions in some_action_profiles:
    ts = env.transition_probabilities(s, actions)
    assert abs(sum(p for p, _ in ts) - 1.0) < 1e-9
```

## Step 2 — Goals (`goals.py`)

Goals are 0/1 rewards that are hashable and immutable:

- Subclass `empo.possible_goal.PossibleGoal`. In `__init__`, call
  `super().__init__(world_model, index=...)`, set your attributes, then
  `super()._freeze()` (goals are immutable afterward). Implement `is_achieved`,
  `__hash__`, `__eq__`, `__str__`. Goals **exclude the env on pickling**.
- Provide a `PossibleGoalGenerator` (`generate(state, human_agent_index)` yields
  `(goal, weight)`) and a matching `PossibleGoalSampler` (`sample(...)`). Mirror
  BushWorld's [`goals.py`](../src/empo/bushworld/goals.py), including
  `set_world_model` and `__getstate__/__setstate__` that drop the env reference
  (needed for parallel workers).

Cell and rectangle goals are a good default. Keep generator weights normalized.

## Step 3 — Human policy prior (`human_policy.py`)

Subclass `empo.human_policy_prior.HumanPolicyPrior`:

- Implement `__call__(state, human_agent_index, possible_goal=None) -> np.ndarray`
  returning a probability vector over actions. With a goal, return the
  goal-conditioned distribution; without one, return the marginal over the
  generator's goals.
- You get `profile_distribution` and `profile_distribution_with_fixed_goal` (the
  joint, independent human action distribution) **for free** from the base class.

BushWorld's [`human_policy.py`](../src/empo/bushworld/human_policy.py) is a
density-agnostic Manhattan shortest-path policy with `beta_h = ∞` by default.

## Step 4 — Loader (`loader.py`)

Provide a small map/YAML loader so worlds are data, not code. BushWorld's
[`loader.py`](../src/empo/bushworld/loader.py) parses a token grid
(`Ro`/`Hu`/integer-density/`.`) plus config keys (`B`, `max_steps`,
`fill_density`, `possible_goals`, `seed`) and returns a constructed env. Keep the
parse function pure (returns plain data) and a thin `load_*` wrapper that builds
the env, so the parser is easy to unit-test.

## Step 5 — Rendering and movies (`rendering.py`)

Render the current state to an RGB array and provide a movie writer. BushWorld's
[`rendering.py`](../src/empo/bushworld/rendering.py):

- Uses a non-interactive matplotlib backend (`Agg`) so it works headless.
- Supports an `annotation_text` side panel and `goal_overlays` (dashed outlines).
- `save_movie(frames, path)` writes `.gif` (via Pillow, no extra deps) or `.mp4`
  (via `imageio[ffmpeg]`). Prefer `.mp4` in examples where easy
  scrubbing/playback matters, and use `.gif` as the dependency-free default/
  fallback when avoiding ffmpeg dependencies.

Delegate from `env.render(...)` to `render_frame(...)` so callers use the gym API.

## Step 6 — Backward induction works *for free*

Once steps 1–3 are correct, the **exact** Phase 2 backward induction already
works — no world-specific code required:

```python
from empo.backward_induction.phase2 import compute_robot_policy

policy = compute_robot_policy(
    env, list(env.human_agent_indices), list(env.robot_agent_indices),
    env.possible_goal_generator, human_policy_prior,
    beta_r=5.0, gamma_h=0.95, gamma_r=0.95, zeta=1.0, xi=1.0, eta=1.0,
    level_fct=lambda s: s[0],  # a function mapping state -> DAG level (e.g. step)
)
```

`level_fct` must return a monotonically increasing integer along every
transition (BushWorld uses `step_count`), so the state DAG can be processed in
reverse-topological order. This is the fastest way to verify your dynamics: if
backward induction runs and returns a sensible policy, your `WorldModel` is sound.

## Step 7 — The Phase 2 equations (reference)

The learning-based path must match the **same** equations that backward induction
solves (see `empo/backward_induction/phase2.py`, `_rp_process_single_state`). For
unit durations they are:

- `Q_r(s, a_r) = E_{a_h ~ pi_h(·|s)} E_{s'} [ gamma_r · V_r(s') ]`
- `pi_r(a_r | s) ∝ (−Q_r(s, a_r))^{−beta_r}` (computed in log-space; `Q_r < 0`)
- `V_h^e(s, g) = E_{a_r ~ pi_r} E_{a_h ~ pi_h(·|s, g)} E_{s'}
   [ achieved(s', g) ? 1 : gamma_h · V_h^e(s', g) ]`
- `X_h(s) = Σ_g weight_g · V_h^e(s, g)^zeta`
- `y(s) = (1/n_humans) Σ_h X_h(s)^{−xi}`,  `U_r(s) = −y(s)^eta`
- `V_r(s) = dw · U_r(s) + Σ_{a_r} pi_r(a_r) · Q_r(s, a_r)`, where for unit
  durations `dw = 1` if `gamma_r = 1`, else `(1 − gamma_r)/(−ln gamma_r)`.

`terminal_Vr` is a small negative constant and `V_h^e = 0` at terminal states.

## Step 8 — Learning-based Phase 2 (`empo/learning_based/<yourworld>/`)

Wire your world into the **shared** production learning stack
(`empo/learning_based/phase2/`) — replay buffers, RND curiosity, warm-up
schedules (see [`docs/WARMUP_DESIGN.md`](WARMUP_DESIGN.md)), MCTS acting, and
async actor-learner training — instead of writing a bespoke learner. Mirror the
existing per-environment packages exactly:
`empo/learning_based/multigrid/` and `empo/learning_based/bushworld/`.

Create `src/empo/learning_based/<yourworld>/` with the same layout:

- `constants.py` — channel indices / feature sizes for your encoders.
- `feature_extraction.py` — pure functions turning a state (and a goal, an agent
  index) into fixed-size feature vectors.
- `state_encoder.py`, `goal_encoder.py`, `agent_encoder.py` — `nn.Module`
  encoders subclassing the base encoders (with the standard caching API:
  `clear_cache`, `get_cache_stats`, `reset_cache_stats`, `get_config`).
- `phase2/` subpackage with the **five** Phase 2 networks, each subclassing the
  corresponding base in `empo/learning_based/phase2/`:
  - `robot_q_network.py` → `Q_r` (one negative value per joint robot action),
  - `human_goal_ability.py` → `V_h^e(s, g)` in `[0, 1]`,
  - `aggregate_goal_ability.py` → `X_h(s)` in `(0, 1]`,
  - `intrinsic_reward_network.py` → `y(s)`, `U_r(s)`,
  - `robot_value_network.py` → `V_r(s)` (negative).
  Each network exposes `forward(...)`, `forward_batch(...)`, and `get_config()`.
- `phase2/robot_policy.py` — a `RobotPolicy` subclass that loads a trained `Q_r`
  (neural **or** lookup-table) from a checkpoint saved by `trainer.save_policy()`.
- `phase2/exploration_policies.py` — a uniform/Boltzmann exploration policy.
- `phase2/trainer.py` — a thin `BasePhase2Trainer` subclass plus
  `create_phase2_networks(env, config, num_robots, num_actions, ...)` (lookup and
  neural branches) and a `train_<yourworld>_phase2(...)` convenience function.

`Phase2Config` drives everything (theory parameters `beta_r`/`gamma_*`, warm-up
schedule, buffer/batch sizes, exploration `epsilon_r_*`, and `use_lookup_tables`
for the tabular/lookup mode that works with any hashable state). The base trainer
calls a **goal sampler** (`env.possible_goal_generator.get_sampler()`), not the
generator, and `save_policy()` writes a deployable policy that your
`RobotPolicy` subclass reloads.

> Two practical lessons carried over from MultiGrid/BushWorld:
> - **Encoders are cached** for gradient flow and speed — follow the base
>   encoder caching API.
> - **The power-law policy `pi_r ∝ (−Q_r)^{−beta_r}` is sensitive.** At large
>   `beta_r`, small `Q_r` errors move `pi_r` a lot, so the learned policy
>   **approximates** the backward-induction fixed point. Use backward induction
>   (step 7) when you need the exact policy.

Encoder design is documented in
[`docs/ENCODER_ARCHITECTURE.md`](ENCODER_ARCHITECTURE.md).

Whichever execution modes your world is meant for (sync/async, with/without
encoders, neural/lookup), **support them all** and keep them consistent.

## Step 9 — Package exports (`__init__.py`)

Re-export the public classes and functions (env, goals, human policy, loader,
rendering) so users can `from empo.<yourworld> import …`. Keep heavy optional
deps (e.g. torch) out of the world-model package's import path; the learning code
lives in `empo/learning_based/<yourworld>/` and is imported only when needed, so
importing the world-model package stays cheap.

## Step 10 — An example world

Add at least one small world file (e.g. `<yourworld>_worlds/example.yaml`) that is
easy to reason about and fast for backward induction. BushWorld ships
[`bushworld_worlds/two_humans_one_robot.yaml`](../bushworld_worlds/two_humans_one_robot.yaml).

## Step 11 — Tests (`tests/test_<yourworld>.py`)

Cover at least:

- `transition_probabilities` sums to 1, never places two players in one cell, and
  obeys your dynamics rules (movement, conflicts, any resource changes).
- `get_state`/`set_state` round-trip; states are hashable.
- Goals' `is_achieved`, hashing, and equality; generator weights normalized.
- The human policy moves sensibly toward goals; `profile_distribution` normalizes.
- Loader parsing.
- **Learner ↔ backward induction**: assert the tabular learner matches
  `compute_robot_policy` within a small tolerance (backward induction uses
  float16 caches internally, so allow ~1e-3).
- Checkpoint save/resume and policy save/load.

Tests rely on `PYTHONPATH`; run them with:

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
    python -m pytest tests/test_<yourworld>.py -v
```

## Step 12 — An example script (`examples/<yourworld>/…`)

Provide a script that **compares backward induction with the learned policy** and
exercises the lifecycle. BushWorld's
[`bushworld_compare.py`](../examples/bushworld/bushworld_compare.py) demonstrates:

- `argparse` to select the world, `method`, theory parameters, and rollout count.
- A `sys.path` bootstrap so it runs with or without `PYTHONPATH` set.
- **Checkpoint save + recovery** during training.
- **Final-policy persistence**: save, then reload from disk and verify.
- **Optional extra rollouts** from the *saved* final policy (`--extra-rollouts`).
- Rollout **movies** with a side annotation panel and goal overlays.
- A quantitative policy comparison (max `pi_r` difference, argmax agreement).

## Step 13 — Docs

Add a `docs/<YOURWORLD>.md` describing the world (as in
[`docs/BUSHWORLD.md`](BUSHWORLD.md)), and link it from any index you maintain.

## Common pitfalls

- **Unhashable state** — tuples only; no lists or numpy arrays inside the state.
- **`transition_probabilities` not summing to 1** — usually a missed conflict or
  off-grid case.
- **Goals must be hypothetical** — the robot reasons over *all* possible goals,
  not the human's "actual" goal.
- **Serializability** — parallel DAG workers pickle the env and goals; drop env
  references in `__getstate__` and provide construction kwargs.
- **Don't confuse `epsilon` (an exploration hyperparameter) with `beta` (a theory
  parameter).**
- **Keep terminology precise** (`training_step`/`env_step`, Phase 1/Phase 2,
  compute/approximate not learn/optimize).

# BushWorld

BushWorld is a small, efficient `WorldModel` for EMPO. It is intentionally simple
so that the **exact** Phase 2 backward induction runs quickly, while still
exhibiting the core empowerment dynamics: a robot that clears obstacles (bushes)
makes humans freer to reach a wide range of goals.

See also [`docs/IMPLEMENTING_A_NEW_WORLDMODEL.md`](IMPLEMENTING_A_NEW_WORLDMODEL.md),
which uses BushWorld as the running example for building a brand-new world model.

## The world

- **Players.** One or more **robot** bodies (steered collectively by the AI) and
  one or more **humans**. Agent indices are robots first, then humans
  (`env.robot_agent_indices`, `env.human_agent_indices`).
- **Grid.** A 2D rectangular grid of parameterizable `width × height`. The
  y-axis grows downward.
- **Bushes.** Every cell holds an integer **bush density** `0, 1, …, B`
  (initialized randomly, or from a map). `B` is the maximum density. All cells
  are in principle walkable.
- **Horizon.** `max_steps` (`T`): the episode terminates after `T` env steps.
- **State.** `(step_count, positions, densities)` where `positions` is a tuple of
  `(x, y)` per player (robots first) and `densities` is a flat row-major tuple of
  cell densities. The whole state is hashable (required by EMPO).
- **No overlap.** No two players may occupy the same cell.

## Actions

`Actions` is an `IntEnum`: `north=0, west=1, south=2, east=3, pass_=4`
(deltas `N=(0,-1)`, `W=(-1,0)`, `S=(0,1)`, `E=(1,0)`).

## Dynamics

Movement is resolved so that `transition_probabilities` stays **exact** and small:

- A player may only move into a cell that is **empty at the start of the step**
  (no trains of moving agents, no position swaps). Off-grid moves are blocked
  no-ops.
- **Conflicts.** If several players target the same empty cell, they are ordered
  by agent id (robots first, then humans) and only the **first in line** attempts
  the move. So every conflict has only two possible outcomes: the first-in-line
  succeeds, or it fails and no one moves in.
- **Robots** always succeed when moving into a (start-of-step empty) target cell.
  Moving in decrements the **target** cell's density by 1. A robot that **passes**
  decrements its **current** cell's density by 1. Densities never go below 0.
  Blocked/off-grid robot moves cause **no** density change (only an explicit
  `pass` clears the current cell).
- **Humans** move into a bush of density `d` with probability `1 − d/B`,
  otherwise they fall back to `pass` (stay put). Humans never change densities.
  `d = 0` ⇒ deterministic success; `d ≥ B` ⇒ deterministic failure.
- The **only** stochasticity is the humans' move-success coins, enumerated
  independently, so transitions remain a small, exact, factorized distribution.

## Goals

Goals mirror the multigrid design (0/1 reward, hashable):

- `ReachCellGoal(env, human_agent_index, (x, y))` — the human reaches a cell.
- `ReachRectangleGoal(env, human_agent_index, (x1, y1, x2, y2))` — the human
  reaches any cell in a rectangle.
- `BushWorldConfigGoalGenerator` / `BushWorldConfigGoalSampler` build these on the
  fly for any human index. By default the world uses **all cells** as possible
  goals (`all_cell_goal_coords`).

## Heuristic human policy

`ShortestPathHumanPolicyPrior` moves each human along a Manhattan shortest path
toward the goal, **regardless of bush densities** (as specified by the world).
With the default `beta_h = ∞` it puts uniform mass on all distance-minimizing
actions; finite `beta_h` gives a Boltzmann policy over negative distances. It
inherits `profile_distribution` / `profile_distribution_with_fixed_goal` from
`HumanPolicyPrior` for the joint (independent) human action distribution.

## Rendering and movies

- `env.render(annotation_text=..., goal_overlays=...)` returns an RGB array.
  Empty cells are black, fully dense cells brown, with interpolation in between;
  robots are grey squares, humans yellow circles, and goals are drawn as dashed
  blue outlines. An optional side panel shows annotation text.
- `empo.bushworld.rendering.save_movie(frames, path)` writes a `.gif` (no extra
  dependencies) or `.mp4` (needs `imageio[ffmpeg]`).

## YAML loader

`load_bushworld(path)` parses a map where `Ro` = robot, `Hu` = human, an integer
= a bush of that density, and `.` = empty (density 0). Cells occupied by a player
get `fill_density`. Recognized config keys: `map`, `B`, `max_steps`,
`fill_density`, `possible_goals`, `seed`.

Example world ([`bushworld_worlds/two_humans_one_robot.yaml`](../bushworld_worlds/two_humans_one_robot.yaml)):
two humans standing apart with one robot between them, closer to one of them,
`B = 1`, every cell density 1.

```python
from empo.bushworld import load_bushworld, ShortestPathHumanPolicyPrior
env = load_bushworld("bushworld_worlds/two_humans_one_robot.yaml")
hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
```

## Computing a robot policy

### Backward induction (exact)

```python
from empo.backward_induction.phase2 import compute_robot_policy

policy = compute_robot_policy(
    env, list(env.human_agent_indices), list(env.robot_agent_indices),
    env.possible_goal_generator, hpp,
    beta_r=5.0, gamma_h=0.95, gamma_r=0.95, zeta=1.0, xi=1.0, eta=1.0,
    level_fct=lambda s: s[0],  # DAG level = step_count
)
```

### Learning-based Phase 2

`empo.bushworld.learning` provides a compact, self-contained learning path that
matches the exact Phase 2 equations:

```python
from empo.bushworld.learning import Phase2Params, train_bushworld_phase2

params = Phase2Params(beta_r=5.0, gamma_h=0.95, gamma_r=0.95)

# Tabular fitted value iteration (converges to the backward-induction solution):
policy, history = train_bushworld_phase2(env, hpp, params, method="tabular")

# Neural fitted value iteration (DQN/AlphaZero-style approximation):
policy, history = train_bushworld_phase2(
    env, hpp, params, method="dqn",
    checkpoint_path="ckpt.pt", num_iterations=600,
)
```

- `method` ∈ {`"tabular"`/`"value_iteration"`, `"neural"`/`"dqn"`/`"alphazero"`}.
- Passing `checkpoint_path=` saves training state and (with `resume=True`)
  recovers from it.
- `save_policy(policy, path)` / `load_policy(path, env)` persist and reload the
  final policy.

> The tabular learner reproduces the backward-induction fixed point (it solves
> the same equations iteratively). The neural learner is an **approximation**;
> because the power-law policy `pi_r ∝ (−Q_r)^{−beta_r}` is very sensitive to small
> `Q_r` errors at large `beta_r`, exact policy agreement requires careful tuning
> and more training. Use the tabular path when you need the exact policy.

## Example script

[`examples/bushworld/bushworld_compare.py`](../examples/bushworld/bushworld_compare.py)
compares backward induction with a learned policy and demonstrates the full
lifecycle: checkpoint save/recovery, final-policy persistence and reload, optional
extra rollouts from the saved policy, and annotated rollout movies.

```bash
# Inside the dev container (PYTHONPATH already set):
python examples/bushworld/bushworld_compare.py --quick

# Outside the container:
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
    python examples/bushworld/bushworld_compare.py \
    --method tabular --rollouts 4

# Neural learner, then resume + extra rollouts from the saved policy:
python examples/bushworld/bushworld_compare.py --method dqn --neural-iterations 600
python examples/bushworld/bushworld_compare.py --method dqn --extra-rollouts 3
```

## Tests

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
    python -m pytest tests/test_bushworld.py -v
```

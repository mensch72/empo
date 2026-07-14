# Implementation Plan: Hindsight Goal Relabeling for V_h^e (Phase 2)

**Status:** Proposed
**Date:** 2026-07-14

## 1. Overview

Phase 2 learns the human goal-achievement value $V_h^e(s, g_h)$ — the (discounted)
probability that human $h$ eventually achieves goal $g_h$ under the current robot
policy $\pi_r$ and the frozen, goal-conditioned human policy prior
$\pi_h(g_h)$. Today, each collected transition trains $V_h^e$ **only** for the
goal profile that was actually assigned during that rollout
(`t.goals`); see the assembly loop in
[`trainer.py`](../../src/empo/learning_based/phase2/trainer.py) around the
`v_h_e_data` construction and the target computation in
`_compute_model_based_v_h_e_targets`.

This plan proposes **hindsight goal relabeling** (a goal-conditioned, HER-style
reuse of experience): use a transition collected under behavior goal $g_h$ to
*also* train $V_h^e(s, g_h')$ for other goals $g_h'$. The subtlety — and the
reason this is not a drop-in of vanilla HER — is that $V_h^e$ is an **on-policy
policy-evaluation** quantity (an expectation over $a_h \sim \pi_h(g_h)(s)$), not
an optimal-value quantity (a $\max$ over human actions). Q-learning-style HER
avoids importance corrections precisely because its bootstrap is a $\max$; here
the bootstrap is an on-policy expectation, so off-policy relabeling requires a
**per-decision importance-sampling (IS) correction** on the human action.

### 1.1 Goals of this work

- Increase sample efficiency of $V_h^e$ training by spreading each transition's
  signal across similar goals, **without introducing bias**.
- Keep the correction cheap and compatible with the existing model-based target
  machinery (reuse `transition_probs_by_action`).
- Restrict candidate goals with an $O(\text{path length})$ selector so the
  large goal space never has to be enumerated.

### 1.2 Non-goals

- No change to Phase 1 (`V_h^m`, human policy prior).
- No clipping-based variance control (clipping introduces bias; see §4). We use
  **self-normalized IS (SNIS)** instead.
- No relabeling for `n_step`/`episode` target modes in the first cut (those need
  a product of per-step ratios; see §7).

## 2. Why relabeling needs an importance correction

The model-based one-step target for entry $(s, h, g)$, holding the stored human
action profile $a_H$ fixed and summing explicitly over robot actions and
successors, is

$$
y(s,h,g) = \sum_{a_r} \pi_r(a_r\mid s)\sum_{s'} P\big(s'\mid s, a_H, a_r\big)\,
\Big[\,U_h(s',g) + \big(1-U_h(s',g)\big)\,\gamma_h\, V_h^e(s',g)\,\Big].
$$

The robot-action expectation ($\pi_r$) and successor expectation ($P$) are exact;
the **only** Monte-Carlo-sampled part is the human action $a_H$, a single sample.
That sample is on-policy for the behavior goal $g_h$, so the unweighted MSE is
unbiased there.

Factorize the joint human policy,
$\pi_H(s,g)(a_H)=\prod_j \pi_j(g_j)(s)(a_j)$. When we relabel **only human $h$'s**
goal $g_h \to g_h'$:

- The other humans' actions $a_{-h}$ were drawn from
  $\prod_{j\neq h}\pi_j(g_j)(s)$ with $g_{-h}$ from the sampler — a valid sample
  of the $\mathbb{E}_{g_{-h}}\mathbb{E}_{a_{-h}}$ marginal in eq. (6). **No
  reweighting for them.**
- Only $a_h$ came from the wrong distribution ($\pi_h(g_h)$ instead of
  $\pi_h(g_h')$).

So a **single, one-human categorical ratio** suffices:

$$
w(g_h') \;=\; \frac{\pi_h(g_h')(s)(a_h)}{\pi_h(g_h)(s)(a_h)},
$$

evaluable from the frozen Phase-1 human policy prior at the stored $(s, a_h)$ —
no extra world-model queries.

## 3. Loss function

Compute the ordinary model-based target `y` but with the relabeled goal $g_h'$
(use $g_h'$ in the achievement check and the bootstrap $V_h^e(s',g_h')$; reuse
the stored `transition_probs_by_action`, which already fixed $a_H$). Weight the
**squared error** by $w$:

$$
\mathcal{L} \;=\; w(g_h')\,\big(V_h^e(s,g_h') - y(s,h,g_h')\big)^2 .
$$

**Fixed point (unbiased).** Differentiating
$\mathbb{E}_{a_h\sim\pi_h(g_h)}\!\big[w\,(V-y)^2\big]$ gives

$$
V^\* = \frac{\mathbb{E}_{\pi_h(g_h)}[w\,y]}{\mathbb{E}_{\pi_h(g_h)}[w]}
= \frac{\mathbb{E}_{\pi_h(g_h')}[y]}{1} = \mathbb{E}_{\pi_h(g_h')}[y],
$$

using $\mathbb{E}_{\pi_h(g_h)}[w]=1$ — exactly the RHS of eq. (6) for $g_h'$.

**Weight the loss, not the target.** Regressing onto a scaled target
$(V - w\,y)^2$ also converges to the right value but lets $w\,y$ exceed $[0,1]$,
breaking the probability interpretation and the `SoftClamp`/`apply_hard_clamp`
on $V_h^e$, with higher variance. The loss-weighted form keeps $y\in[0,1]$ and
only scales the gradient.

**Self-normalization (no clipping).** To control the variance of $w$ without the
bias that clipping (e.g. V-trace $\bar\rho$) would introduce, normalize the
weights over the set of behavior samples that land on the same relabel entry
$(s,h,g')$:

$$
\hat{\mathcal{L}} = \frac{\sum_i w_i\,(V - y_i)^2}{\sum_i w_i}.
$$

SNIS is consistent (asymptotically unbiased) and removes reliance on
$\mathbb{E}[w]=1$ holding empirically in a finite minibatch. In a large state
space exact-$s$ recurrence is rare, so in practice this is a per-goal
minibatch normalization and is an approximation (see §7, open questions).

## 4. Goal-selection procedure (must be outcome-independent)

The relabel-goal selection must be **$\sigma(s)$-measurable**: a function of the
current state $s$ (and pre-outcome data such as the behavior goal profile), and
**independent of both $a_h$ and $s'$**. Formally, with inclusion variable
$Z(g')$ (or proposal $q(g'\mid s)$), require $Z \perp (a_h, s') \mid s$.

### 4.1 The forbidden shortcut

Selecting goals by "which goals were attained at the realized $s'$" is
selection-on-outcome (conditioning on success) and is biased in a way that
**self-normalization cannot fix** (SNIS only corrects the weight normalizer, not
an inclusion rule correlated with the integrand). Concretely, if we included
$g'$ only when $U_h(s',g')=1$, every target applied to $V_h^e(s,g')$ would be
$1$, and the network would converge to $V_h^e \equiv 1$ — we would only ever
train on successes and discard the failures that make $V_h^e$ a probability
in $[0,1)$.

### 4.2 Valid procedures

Selection may depend on: $s$, the behavior goal profile, sampler weights, and
current network estimates at $s$ (functions of $s$ and the **frozen target
nets**). It may **not** depend on: the realized $s'$, the sampled $a_h$, or
realized attainment.

- **Fixed goal set** $G(s)$ (deterministic function of $s$), or
- **Sample from a state-only proposal** $q(\cdot\mid s)$ (the current code is a
  special case: `t.goals` were sampled at collection time, before $s'$), or
- **Smart state-only proposal** (§5, §6) that predicts which goals are
  low-variance / high-information from $s$ alone.

The attainment signal enters **only through the target value** `y` (attained
successors contribute their $1$'s via $P(s'\mid s,a)$), never through selection.

## 5. Efficient selection via stored shortest paths

The goal space is potentially large, so scoring all goals is intractable. We add

```python
GoalSampler.get_similar_goals(state, goal, k) -> List[PossibleGoal]
```

that returns up to $k$ goals **without enumerating the goal space**, by reusing
the precomputed shortest paths.

- [`PathDistanceCalculator`](../../src/empo/learning_based/multigrid/path_distance.py)
  precomputes all-pairs shortest paths between passable cells (BFS on the static
  wall grid) and exposes `get_shortest_path(source, target)`.
- Goals map to target cells:
  [`ReachCellGoal`](../../src/empo/bushworld/goals.py) has `target_pos`,
  `ReachRectangleGoal` has a center `target_pos`.

So `get_similar_goals(s, g_h, k)` looks up the stored path from $h$'s cell in
$s$ to $g_h$'s target cell and maps each cell on it back to a goal —
$O(\text{path length})$, no goal-space iteration.

### 5.1 Why path membership is a good similarity proxy

What governs the SNIS weight variance is the divergence of $\pi_h(g')(s)$ from
$\pi_h(g_h)(s)$ over $h$'s **first action**. For a goal $g'$ on the shortest path
to $g_h$, the shortest path to $g'$ is a **prefix** of the path to $g_h$, so the
two goal-policies agree along the shared prefix — in particular the immediate
move. This is an exact, **topology-aware** overlap (it respects walls/doors,
which a target-cell *angle* heuristic silently ignores), and it yields
$w(g')\approx 1$ (low variance).

Selection is only the candidate filter; the weight $w(g')$ is still computed
**exactly** from $\pi_h$ at $s$ (just $k$ queries). Residual differences from
shortest-path ties or the Boltzmann spread ($\beta_h<\infty$) are absorbed by
the exact weight plus SNIS.

### 5.2 Which $k$ to pick along the path

Do **not** take the $k$ nearest (easiest) cells — that reintroduces an
easy-goal bias. Spread the $k$ roughly uniformly along the path to cover a range
of difficulties. The cells near $g_h$ are the hard, low-$V_h^e$ goals that
dominate $X_h$/$U_r$: with $X_h(s)=\mathbb{E}_g[V_h^e(s,g)^\zeta]$,

$$
\frac{\partial X_h}{\partial V_h^e(s,g)} \propto \zeta\,V_h^e(s,g)^{\zeta-1}
\xrightarrow[V_h^e\to 0]{}\infty \quad(\zeta<1),
$$

so $V_h^e$ accuracy matters **most** on hard goals, and
$U_r=-(\mathbb{E}_h[X_h^{-\xi}])^\eta$ further emphasizes low-$X_h$ humans.

### 5.3 Per-world implementation

`get_similar_goals` is world-specific and lives on the world's goal sampler:
- **Multigrid:** reuse `PathDistanceCalculator` (built once per world/episode).
- **Bushworld:** the human policy already "moves along the shortest path"
  ([`bushworld/human_policy.py`](../../src/empo/bushworld/human_policy.py)); the
  membership test is the cheap Manhattan-monotone frontier (cells in the
  axis-aligned bounding box between $h$ and the target that reduce Manhattan
  distance).
- Rectangle goals use the rectangle center (or nearest passable cell).

## 6. Optional: value-of-information refinement

Because the path restricts candidates to $O(\text{path length})$, we can then
afford richer scoring on just those survivors (still state-only, still
unbiased):

- exact $\chi^2$/TV divergence between $\pi_h(g')(s)$ and $\pi_h(g_h)(s)$
  (low-variance survivors), and/or
- an ensemble-disagreement / RND novelty score for the specific $(s,g')$ (the
  repo already has RND and RND-based adaptive LR).

Combine **multiplicatively**: $\text{score} \approx \text{overlap}(g') \times
\text{VoI}(g')$, take top-$k$, then SNIS. Keep a **coverage floor** (mix with the
base sampler, $q = (1-\epsilon)q_\text{score} + \epsilon\,q_\text{sampler}$) so
no goal is permanently starved — with bootstrapping, a collapsed proposal can
let un-selected goals go stale and oscillate. This refinement is deferred to a
second phase.

## 7. Scope and correctness constraints

- **One-step targets only (first cut).** The single ratio is valid because the
  target is a one-step backup with explicit robot/successor expectations. For
  `v_h_e_target_mode in {n_step, episode}`, the sampled suffix was generated
  under the behavior human policy, so an unbiased relabel needs a **product** of
  per-step ratios $\prod_k \pi_h(g')(s_k)(a_{h,k})/\pi_h(g_h)(s_k)(a_{h,k})$,
  with rapidly growing variance. Relabeling will therefore be **disabled** when
  trajectory targets are active, until a dedicated per-decision path is added.
- **Only human $h$'s own action is reweighted** (§2); other humans and the robot
  are unaffected.
- **SNIS normalization set.** Ideally per exact $(s,h,g')$; in practice per goal
  within the minibatch. Document this as an approximation and measure its effect.

## 8. Implementation touchpoints

1. **`PossibleGoalSampler.get_similar_goals(state, goal, k)`**
   ([`possible_goal.py`](../../src/empo/possible_goal.py)): abstract method with a
   safe default (return `[]` → relabeling becomes a no-op). Concrete
   implementations in the multigrid and bushworld samplers
   ([`bushworld/goals.py`](../../src/empo/bushworld/goals.py),
   multigrid `ConfigGoalSampler`), wired to a `PathDistanceCalculator`.
2. **`Phase2Config`** ([`config.py`](../../src/empo/learning_based/phase2/config.py)):
   `use_goal_relabeling: bool = False`, `relabel_goals_per_transition: int`
   (the $k$), plus a validation that force-disables it when
   `uses_trajectory_v_h_e_targets()` is true, and includes the flags in
   `save_yaml`.
3. **Trainer `v_h_e_data` assembly**
   ([`trainer.py`](../../src/empo/learning_based/phase2/trainer.py) near the
   `for h, g_h in t.goals.items()` loop): when enabled, append extra
   `(trans_idx, h, g')` entries from `get_similar_goals`, tagged as relabeled and
   carrying the IS weight `w(g')`.
4. **Weight computation:** query the frozen human policy prior at $(s, a_h)$ for
   $\pi_h(g')(s)(a_h)$ and $\pi_h(g_h)(s)(a_h)$; store `w`. Behavior entries get
   `w = 1`.
5. **`_compute_model_based_v_h_e_targets`:** unchanged except the achievement
   check and bootstrap use the (already per-entry) goal `g'`; it already reuses
   `transition_probs_by_action`, so no new world-model queries.
6. **Loss aggregation:** apply the SNIS-weighted MSE (§3), grouping by goal for
   normalization. Add diagnostics: relabel rate, mean/ESS of weights per goal,
   fraction of relabeled entries with attained successors.
7. **Modes:** support neural and lookup-table `V_h^e`, sync and async, with and
   without encoders. For lookup tables the weight simply scales the per-entry
   gradient (compatible with the `1/n` adaptive-LR path).

## 9. Testing

- **Unbiasedness (tabular).** On a small world where exact backward-induction
  $V_h^e$ is available, verify that enabling relabeling does not shift the
  converged lookup-table $V_h^e$ (within tolerance) versus no relabeling.
- **Selection is outcome-independent.** Unit test that `get_similar_goals`
  output depends only on $(s, g_h)$ and never on $a_h$/$s'$.
- **Bias guard.** A regression test that the (forbidden) "attained-goals"
  selection drives $V_h^e\to 1$, documenting why it is excluded.
- **Path selector.** Unit tests that returned goals lie on the stored shortest
  path and are spread (not the $k$ nearest); rectangle-center handling.
- **Efficiency.** Confirm no full goal-space enumeration
  (`get_similar_goals` cost is $O(\text{path length})$).
- **Convergence/efficiency.** On bushworld_compare, measure $V_h^e$ RMSE vs.
  backward-induction with/without relabeling at matched training steps.

## 10. Risks and open questions

- **SNIS grouping** across distinct states in a minibatch is an approximation;
  quantify its effect and consider per-state grouping when $s$ recurs.
- **Weight variance** on topology-similar-but-Boltzmann-divergent goals; monitor
  effective sample size and, if needed, tighten the similarity threshold rather
  than clip.
- **Coverage** when the (optional) VoI proposal is enabled — require the
  $\epsilon$ floor.
- **Trajectory-target integration** (per-decision products) is deferred; keep the
  hard disable until implemented.

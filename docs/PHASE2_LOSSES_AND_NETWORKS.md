# Phase 2 Neural Pathway: Networks and Loss Variants

Reference for the **DQN-style** Phase 2 trainer
([`BasePhase2Trainer`](../src/empo/learning_based/phase2/trainer.py)). This
document does **not** cover the PPO pathway.

All networks are trained by regressing their output onto exact,
model-based bootstrap targets (computed via `transition_probabilities()`),
not by policy-gradient optimization. Each network has a frozen **target copy**
used to form those targets. Losses are aggregated per training step and
back-propagated through the (optionally shared) encoder.

## Networks

Container: `Phase2Networks` in [trainer.py](../src/empo/learning_based/phase2/trainer.py).
`q_r` and `v_h_e` always exist; `x_h`, `u_r`, `v_r` are created only when their
`*_use_network` flag is set (otherwise the quantity is computed in closed form
from the others). RND is optional and only used for exploration.

| Network | Symbol | Predicts | Output domain | Loss space | Role |
|---|---|---|---|---|---|
| `q_r` | $Q_r(s,a_r)$ | robot Q-value (aggregate human power) | $(-\infty, -1]$ | Q or z (see below) | drives robot policy $\pi_r \propto (-Q_r)^{-\beta_r}$ |
| `v_h_e` | $V_h^e(s,g)$ | human goal-achievement value | $[0, 1]$ | raw MSE | building block of the power metric |
| `x_h` | $X_h(s,g)$ | aggregate human goal ability | $[0, 1]$ | raw MSE | per-human attainment feeding $U_r$ |
| `u_r` | $U_r(s)$ via $y$ | intrinsic reward (power) | $y \in [1,\infty)$, $U_r \le -1$ | z-space MSE on $y$ | robot's intrinsic reward |
| `v_r` | $V_r(s)$ | robot state value | $(-\infty, -1]$ | z-space MSE | bootstrap value for $Q_r$ targets |
| `rnd` | — | random-network distillation error | — | prediction MSE | curiosity/exploration bonus only |

Notes:
- **`v_h_e` / `x_h`** predict probability-like quantities in $[0,1]$, so they use
  a plain squared-error loss in raw space — no transform needed.
- **`u_r` / `v_r`** predict guaranteed-negative (or $\ge 1$) quantities, so they
  always train in **z-space** (see [VALUE_TRANSFORMATIONS.md](VALUE_TRANSFORMATIONS.md)):
  the network emits $z \in (0,1]$ and the target is transformed the same way.
- **`rnd`** is not part of the value equations; its loss only shapes the
  exploration bonus (`use_rnd`).

## `Q_r` loss variants

`Q_r` is the one network with multiple selectable loss formulations. They are
**not** mutually exclusive: the base MSE space, the optional advantage
weighting, and the optional MCTS distillation term compose.

### 1. Output representation — `use_z_space_transform`

Controls **what the network emits** (a pure, invertible reparameterization; does
not change the learned value):

$$z = (-Q_r)^{-1/(\eta\xi)} \in (0,1], \qquad Q_r = -z^{-\eta\xi}$$

Keeps the raw output in $(0,1]$ instead of forcing extreme weights to emit large
negative numbers. Recommended on.

### 2. Base MSE space — `use_z_based_loss`

Controls **where the squared error is measured** (only when
`use_z_space_transform=True`):

| Value | Loss | Effect |
|---|---|---|
| `False` (default) | $(\hat Q - Q_{\text{target}})^2$ in **Q-space** | large-magnitude errors get strong gradients → fast outlier correction |
| `True` (two-phase) | $(\hat z - z_{\text{target}})^2$ in **z-space** during the constant-LR phase, Q-space during LR decay | balanced gradients across scales, but **compresses** large negatives → they can be under-corrected and drift |

Empirically on bushworld, pure z-based loss inflated the value scale and
collapsed effective rank; Q-space loss is the safer default.

### 3. Advantage / sensitivity weighting — `q_r_advantage_weighted_loss`

Only active in the `one_step` (all-actions) target mode. Reweights each robot
action's per-action squared error by the local softmax sensitivity of the
**target** policy, renormalized per state:

$$w(a) = \pi_r(a)\,\bigl(1 - \pi_r(a)\bigr), \qquad
\mathcal{L} = \sum_a \frac{w(a)}{\sum_{a'} w(a')}\,\bigl(\text{err}_a\bigr)^2$$

This focuses capacity on **decision-relevant action gaps** rather than absolute
magnitudes, without compressing large values the way z-based loss does. Skipped
while $\beta_r = 0$ (warm-up), where the policy is uniform and carries no
decision signal. `q_r_advantage_weight_floor` adds a minimum weight so
near-deterministic actions still receive some gradient. Captures the
head-plasticity benefit of z-loss while preserving value scale and effective
rank.

### 4. MCTS policy-distillation term — `pi_r_mode="mcts"`

When acting-time MCTS is enabled, an auxiliary term distills the search-derived
root policy into $Q_r$, added to the base loss and scaled by
`mcts_policy_distillation_coef`.

## Target modes

- `q_r_target_mode="one_step"` (default): exact one-step Bellman backup over
  **all** robot actions (`Loss/q_r_all_actions`). Required for advantage
  weighting.
- `q_r_target_mode="n_step"` / `"episode"`: trajectory-aware suffix targets on
  the **taken** action only (`Loss/q_r_taken_action`); advantage weighting does
  not apply.

## Where to look in TensorBoard

- `Loss/q_r_all_actions`, `Loss/v_h_e`, `Loss/x_h`, `Loss/u_r`, `Loss/v_r`,
  `Loss/rnd` — per-network losses.
- `Predictions/q_r_mean` vs `Targets/q_r_mean` — value-fit fidelity (should track).
- `ZSpace/q_r_z_pred`, `ZSpace/q_r_z_target` — z-space views when the transform is on.
- `Plasticity/q_r/*` — see [PLASTICITY_DIAGNOSTICS.md](PLASTICITY_DIAGNOSTICS.md).

## Related docs

- [VALUE_TRANSFORMATIONS.md](VALUE_TRANSFORMATIONS.md) — z-space math and rationale.
- [WARMUP_DESIGN.md](WARMUP_DESIGN.md) — staged warm-up that breaks the mutual dependency.
- [PLASTICITY_DIAGNOSTICS.md](PLASTICITY_DIAGNOSTICS.md) — reading the plasticity metrics.

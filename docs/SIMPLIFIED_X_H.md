# Simplified X_h Computation

This document explains three related but distinct objects:

1. standard `X_h`, which aggregates goal-conditioned `V_h^e(s, g_h)`
2. exact simplified `X_h`, which removes explicit goals and uses a successor-state recursion
3. the current learning-based approximation, which estimates the simplified recursion from an inverse-dynamics ratio

The key point is that simplified `X_h` is not just “standard `X_h` without goals”. It changes the object being approximated.

The main implementation files are:

- `src/empo/backward_induction/phase2.py`
- `src/empo/learning_based/phase2/simplified_x_h.py`
- `src/empo/learning_based/phase2/trainer.py`
- `src/empo/learning_based/phase2_ppo/trainer.py`
- `src/empo/learning_based/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/trainer.py`
- `src/empo/learning_based/multigrid/phase2_ppo/networks.py`

## Standard `X_h`

In the standard Phase 2 formulation, `X_h` is built from the goal-conditioned attainment quantity `V_h^e(s, g_h)`:

$$
X_h(s) = \mathbb{E}_{g_h}\left[V_h^e(s, g_h)^\zeta\right].
$$

That is the ordinary goal-aggregation path. It depends on explicit hypothetical goals and Phase 1 style human-policy reasoning.

When `use_simplified_x_h=False`, this is still the object the learning-based code is approximating.

## Exact simplified `X_h`

Simplified mode replaces the goal dimension by a recursion over successor states. The exact object is

$$
X_h(s) = 1 + \gamma_h^\zeta \sum_{s'} q_h(s, s')^\zeta X_h(s').
$$

Here `h` is the focal human and

$$
q_h(s, s') = \max_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

So simplified `X_h` measures one-step successor-state control power and then bootstraps future control power from the successor state.

This is the object implemented exactly in the tabular backward-induction code in `src/empo/backward_induction/phase2.py`.

### Boundary condition

Terminal states use

$$
X_h(s_{\mathrm{terminal}}) = 1.
$$

That is why simplified-mode neural `X_h` networks use feasible range `(1, \infty)`.

### Bounded rationality

When `x_h_epsilon_h > 0`, the best action is mixed with the mean over actions:

$$
q_h(s, s') = (1 - \epsilon_h) \max_{a_h} P(s' \mid s, a_h, \pi_{-h})
+ \epsilon_h \operatorname{mean}_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

This is a theory-side bounded-rationality interpolation for simplified `X_h`. It is not an exploration schedule.

## Why inverse dynamics appears

The exact quantity `q_h(s, s')` is expensive in learning-based training because evaluating it directly requires reconstructing one-step transition mass for each candidate human action.

The agreed approximation strategy is to learn an inverse-dynamics model

$$
P_\theta(a_h \mid s, s')
$$

and use Bayes' rule to rank actions by how strongly they explain the observed successor state.

More precisely, the dependence on the non-focal agents' current policies is implicit throughout this section. The exact simplified quantity is based on

$$
P(s' \mid s, a_h, \pi_{-h}),
$$

and the inverse-dynamics approximation is trying to recover the action ordering induced by that one-step quantity.

Starting from

$$
P(a_h \mid s, s') = \frac{P(s' \mid s, a_h, \pi_{-h})\,\pi_h(a_h \mid s)}{P(s' \mid s, \pi_{-h})},
$$

we get

$$
P(s' \mid s, a_h, \pi_{-h}) = \frac{P(a_h \mid s, s')}{\pi_h(a_h \mid s)} P(s' \mid s, \pi_{-h}).
$$

For fixed `(s, s')`, the factor $P(s' \mid s, \pi_{-h})$ does not depend on $a_h$, so maximizing over $a_h$ is equivalent to maximizing the ratio

$$
\frac{P(a_h \mid s, s')}{\pi_h(a_h \mid s)}.
$$

That is the basis of the approximation used in the learning-based code.

Important limitation: this only preserves the action ranking up to a common multiplicative factor. The ratio is therefore not itself a probability mass and can exceed `1`.

## What the learning-based code now does

The shared helper `compute_simplified_x_h_td_targets(...)` in `src/empo/learning_based/phase2/simplified_x_h.py` supports two modes.

### Mode 1: inverse-dynamics ratio path

If an `inverse_dynamics_network` is provided, the helper computes logits for

$$
P_\theta(a_h \mid s, s')
$$

turns them into probabilities, divides by the focal-human policy prior `\pi_h(a_h \mid s)`, and forms the sampled controllability score

$$
\widehat{q}_h(s, s') =
\begin{cases}
\max_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)} & \text{if } \epsilon_h = 0, \\
(1-\epsilon_h) \max_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)}
+ \epsilon_h \operatorname{mean}_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)} & \text{otherwise.}
\end{cases}
$$

The actual TD target is then

$$
\widehat{X}_h^{\mathrm{target}}(s, s') = 1 + \gamma_h^\zeta \widehat{q}_h(s, s')^\zeta X_h^{\mathrm{target}}(s').
$$

Important detail: the implementation uses the ratio itself as the sampled score. It does not reconstruct the common factor $P(s' \mid s)$. So this is an approximation to the exact simplified recursion, not an exact algebraic rewrite.

Also, this is only “model-free” in the sense that it avoids exact transition-probability enumeration. The current implementation still passes `world_model` into the inverse-dynamics network so the network can tensorize or encode states.

### Mode 2: exact fallback path

If no inverse-dynamics network is provided, the same helper falls back to the older exact one-step reconstruction:

1. enumerate candidate focal-human actions
2. combine them with robot joint-action probabilities and non-focal human marginals
3. query `world_model.transition_probabilities(state, actions)`
4. extract the probability mass of the observed `next_state`
5. apply the same max or epsilon-mixed reduction over actions

So the helper now contains both:

- the preferred learning-based inverse-dynamics approximation
- the exact world-model fallback

But the fallback is exact only for the local one-step quantity `q_h(s, s')` on the sampled transition. The overall learning-based update is still a sampled one-step TD backup, not the full tabular fixed-point solution.

## DQN path

The DQN entry point is `BasePhase2Trainer._compute_simplified_x_h_td_target(...)` in `src/empo/learning_based/phase2/trainer.py`.

That method:

1. computes `X_h_target(s')` from `self.networks.x_h_target`
2. gets the current robot policy from `q_r_target`
3. builds the focal-human policy prior `\pi_h(a_h \mid s)` in array form
4. passes `self.networks.inverse_dynamics` into `compute_simplified_x_h_td_targets(...)`

So in simplified DQN mode, the `X_h` TD target now uses the inverse-dynamics ratio path when that network exists.

If no inverse-dynamics network is present, the same helper falls back to exact local `q_h(s, s')` reconstruction from `transition_probabilities(...)`.

In multigrid, the inverse-dynamics network is created in `src/empo/learning_based/multigrid/phase2/trainer.py` when simplified mode is active and `v_h_e` is neural.

## PPO path

The PPO entry point is `PPOPhase2Trainer._compute_simplified_x_h_td_target(...)` in `src/empo/learning_based/phase2_ppo/trainer.py`.

That method:

1. computes `X_h_target(s')` from `x_h_target` or `x_h`
2. computes the robot policy from the PPO actor-critic logits
3. constructs the focal-human policy prior array
4. passes a frozen `inverse_dynamics_target` if available, otherwise the online inverse-dynamics network

So PPO uses a target copy for the simplified `X_h` bootstrap path in the same spirit that it uses frozen copies for the other auxiliary reward computations.

In multigrid PPO, the inverse-dynamics network is created in `src/empo/learning_based/multigrid/phase2_ppo/networks.py` whenever `config.use_simplified_x_h` and `use_x_h` are both true.

## Inverse-dynamics training objective

The inverse-dynamics network itself is defined abstractly in `src/empo/learning_based/phase2/inverse_dynamics.py` and concretely for multigrid in `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`.

Its output semantics are straightforward:

$$
\\text{logits}_\theta(s, s', h) \longrightarrow P_\theta(a_h \mid s, s').
$$

The trainers also optimize an auxiliary cross-entropy loss on observed focal-human actions. That auxiliary loss is what teaches the network to approximate the posterior action distribution used in the simplified `X_h` target.

## Relationship to `U_r`

Once simplified `X_h` is available, the downstream structure stays the same:

$$
y(s) = \mathbb{E}_h[X_h(s, h)^{-\xi}],
$$

$$
U_r(s) = -y(s)^\eta.
$$

What changes is the meaning and scale of `X_h`:

- standard mode: goal-conditioned attainment aggregation
- simplified mode: successor-state control recursion, approximated from inverse dynamics in the learning-based path

Because terminal simplified `X_h` is `1`, the code consistently clamps simplified-mode values to be at least `1.0` before they are used downstream.

## Exact reference vs current approximation

It is important to keep these separate.

The exact semantic reference is:

- tabular backward induction in `src/empo/backward_induction/phase2.py`

Within the learning-based helper, the exact fallback branch in `src/empo/learning_based/phase2/simplified_x_h.py` computes the exact local one-step quantity `q_h(s, s')` for the sampled transition, but it does not replace the tabular fixed-point solution.

The current learning-based approximation is:

- train `P_\theta(a_h \mid s, s')`
- score each sampled transition by the Bayes-ratio surrogate
- bootstrap `X_h` from that sampled controllability score

So the code is no longer merely “adding inverse-dynamics diagnostics”. It is actually using the inverse-dynamics model inside the simplified `X_h` target construction.

## Practical reading order

If you want the shortest reliable path through the code, read in this order:

1. `src/empo/backward_induction/phase2.py` for the exact simplified object
2. `src/empo/learning_based/phase2/simplified_x_h.py` for the shared TD target helper
3. `src/empo/learning_based/phase2/inverse_dynamics.py` for the abstract inverse-dynamics interface
4. `src/empo/learning_based/phase2/trainer.py` for the DQN integration
5. `src/empo/learning_based/phase2_ppo/trainer.py` for the PPO integration
6. `src/empo/learning_based/multigrid/phase2/trainer.py` and `src/empo/learning_based/multigrid/phase2_ppo/networks.py` for the actual multigrid network construction
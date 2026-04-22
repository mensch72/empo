# Simplified X_h Computation

This document explains three related but distinct objects:

1. standard $X_h$, which aggregates goal-conditioned $V_h^e(s, g_h)$
2. exact simplified $X_h$, which removes explicit goals and uses a successor-state recursion
3. the current learning-based approximation, which uses a learned inverse-dynamics ratio together with an exact sampled next-state marginal factor

The key point is that simplified $X_h$ is not just “standard $X_h$ without goals”. It changes the object being approximated.

The main implementation files are:

- `src/empo/backward_induction/phase2.py`
- `src/empo/learning_based/phase2/simplified_x_h.py`
- `src/empo/learning_based/phase2/trainer.py`
- `src/empo/learning_based/phase2_ppo/trainer.py`
- `src/empo/learning_based/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/trainer.py`
- `src/empo/learning_based/multigrid/phase2_ppo/networks.py`

## Standard $X_h$

In the standard Phase 2 formulation, $X_h$ is built from the goal-conditioned attainment quantity $V_h^e(s, g_h)$:

$$
X_h(s) = \mathbb{E}_{g_h}\left[V_h^e(s, g_h)^\zeta\right].
$$

That is the ordinary goal-aggregation path. It depends on explicit hypothetical goals and Phase 1 style human-policy reasoning.

When `use_simplified_x_h=False`, this is still the object the learning-based code is approximating.

## Exact simplified $X_h$

Simplified mode replaces the goal dimension by a recursion over successor states. The exact object is

$$
X_h(s) = 1 + \gamma_h^\zeta \sum_{s'} q_h(s, s')^\zeta X_h(s').
$$

Here $h$ is the focal human and

$$
q_h(s, s') = \max_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

One useful way to interpret this object is as a special case of the general
goal-based construction. If the set of possible goals is identified with
finite state sequences and the focal human is perfectly rational with respect
to those sequence-goals, then the resulting power quantity collapses to this
successor-state recursion, up to the same normalization convention used in the
tabular derivation.

With bounded rationality enabled through `x_h_epsilon_h`, the corresponding
special case is not perfectly rational but epsilon-greedy at the human-action
selection step: the max over human actions is replaced by the same
$(1-\epsilon_h)$ / $\epsilon_h$ mixture written below.

So simplified $X_h$ measures one-step successor-state control power and then bootstraps future control power from the successor state.

This is the object implemented exactly in the tabular backward-induction code in `src/empo/backward_induction/phase2.py`.

### Boundary condition

Terminal states use

$$
X_h(s_{\mathrm{terminal}}) = 1.
$$

That is why simplified-mode neural $X_h$ networks use feasible range $(1, \infty)$.

### Bounded rationality

When `x_h_epsilon_h > 0`, the best action is mixed with the mean over actions:

$$
q_h(s, s') = (1 - \epsilon_h) \max_{a_h} P(s' \mid s, a_h, \pi_{-h})
+ \epsilon_h \operatorname{mean}_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

This is a theory-side bounded-rationality interpolation for simplified $X_h$. It is not an exploration schedule.

## Why inverse dynamics appears

The exact quantity $q_h(s, s')$ is expensive in learning-based training because evaluating it directly requires reconstructing one-step transition mass for each candidate human action.

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

Important limitation: this only preserves the action ranking up to a common multiplicative factor. The ratio is therefore not itself a probability mass and can exceed $1$.

## What the learning-based code now does

The shared helper `compute_simplified_x_h_td_targets(...)` in `src/empo/learning_based/phase2/simplified_x_h.py` supports two modes.

### Mode 1: hybrid inverse-dynamics path

If an `inverse_dynamics_network` is provided, the helper computes logits for

$$
P_\theta(a_h \mid s, s')
$$

turns them into probabilities, divides by the focal-human policy prior $\pi_h(a_h \mid s)$, and forms the surrogate controllability score

$$
\widetilde{c}_\theta(s, s') =
\begin{cases}
\max_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)} & \text{if } \epsilon_h = 0, \\
(1-\epsilon_h) \max_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)}
+ \epsilon_h \operatorname{mean}_{a_h} \dfrac{P_\theta(a_h \mid s, s')}{\pi_h(a_h \mid s)} & \text{otherwise.}
\end{cases}
$$

It also reconstructs the exact focal-human marginal next-state probability

$$
m_h(s' \mid s) = \sum_{a_h} \pi_h(a_h \mid s) P(s' \mid s, a_h, \pi_{-h})
$$

for the sampled successor state by enumerating focal-human actions, combining
them with the robot joint-action distribution and the non-focal human
marginals, and querying `world_model.transition_probabilities(...)`.

The actual implemented TD target is then

$$
\widetilde{X}_h^{\mathrm{target}}(s, s') = 1 + \gamma_h^\zeta m_h(s' \mid s)^{\zeta-1} \widetilde{c}_\theta(s, s')^\zeta X_h^{\mathrm{target}}(s').
$$

This notation matters. The implemented quantity $\widetilde{c}_\theta(s, s')$ is not an estimate of $q_h(s, s')$ itself.

This is the sampled estimator implied by the decomposition

$$
q_h(s, s') = m_h(s' \mid s) \, r_h(s, s')
$$

where $r_h$ is the action-ranking ratio quantity recovered from Bayes' rule.
When $\zeta = 1$, the marginal factor disappears; when $\zeta > 1$, the factor
$m_h(s' \mid s)^{\zeta-1}$ is essential.

So the current implementation no longer uses the earlier ratio-only heuristic. It is still not fully model-free, because it avoids exact reconstruction of $q_h(s, s')$ but still computes the sampled marginal factor from the world model.

Equivalently, written in the older denominator form, the same sampled correction is

$$
X_h^{\mathrm{sample}}(s, s') = 1 + \gamma_h^\zeta \frac{q_h(s, s')^\zeta X_h(s')}{P(s' \mid s, \pi)}.
$$

If one wanted to absorb that denominator into the sampled score before applying the power $\zeta$, the score would have to scale like

$$
\frac{q_h(s, s')}{P(s' \mid s, \pi)^{1/\zeta}},
$$

not like $q_h(s, s')$ itself.

Also, this is only “model-free” in the limited sense that the controllability term comes from a learned inverse-dynamics model rather than from exact $q_h$ reconstruction. The current implementation still passes `world_model` into the inverse-dynamics network for tensorization and uses `transition_probabilities(...)` to compute the exact sampled marginal factor.

### Mode 2: exact fallback path

If no inverse-dynamics network is provided, the same helper falls back to the older exact one-step reconstruction:

1. enumerate candidate focal-human actions
2. combine them with robot joint-action probabilities and non-focal human marginals
3. query `world_model.transition_probabilities(state, actions)`
4. extract the probability mass of the observed `next_state`
5. apply the same max or epsilon-mixed reduction over actions

So the helper now contains both:

- the preferred hybrid path: learned inverse-dynamics ratio plus exact sampled marginal factor
- the exact world-model fallback

But the fallback is exact only for the local one-step quantity $q_h(s, s')$ on the sampled transition. The overall learning-based update is still a sampled one-step TD backup, not the full tabular fixed-point solution.

## DQN path

The DQN entry point is `BasePhase2Trainer._compute_simplified_x_h_td_target(...)` in `src/empo/learning_based/phase2/trainer.py`.

That method:

1. computes $X_h^{\mathrm{target}}(s')$ from `self.networks.x_h_target`
2. gets the current robot policy from `q_r_target`
3. builds the focal-human policy prior $\pi_h(a_h \mid s)$ in array form
4. passes `self.networks.inverse_dynamics` into `compute_simplified_x_h_td_targets(...)`

So in simplified DQN mode, the $X_h$ TD target now uses the hybrid path when that network exists: learned inverse-dynamics ratio for the controllability score and exact sampled marginal reconstruction inside the shared helper.

If no inverse-dynamics network is present, the same helper falls back to exact local $q_h(s, s')$ reconstruction from `transition_probabilities(...)`.

In multigrid, the inverse-dynamics network is created in `src/empo/learning_based/multigrid/phase2/trainer.py` when simplified mode is active and `v_h_e` is neural.

## PPO path

The PPO entry point is `PPOPhase2Trainer._compute_simplified_x_h_td_target(...)` in `src/empo/learning_based/phase2_ppo/trainer.py`.

That method:

1. computes $X_h^{\mathrm{target}}(s')$ from `x_h_target` or `x_h`
2. computes the robot policy from the PPO actor-critic logits
3. constructs the focal-human policy prior array
4. passes a frozen `inverse_dynamics_target` if available, otherwise the online inverse-dynamics network

So PPO uses a target copy for the controllability term in the simplified $X_h$ bootstrap path, while the exact sampled marginal factor is still reconstructed inside the shared helper from `transition_probabilities(...)`.

In multigrid PPO, the inverse-dynamics network is created in `src/empo/learning_based/multigrid/phase2_ppo/networks.py` whenever `config.use_simplified_x_h` and `use_x_h` are both true.

## Inverse-dynamics training objective

The inverse-dynamics network itself is defined abstractly in `src/empo/learning_based/phase2/inverse_dynamics.py` and concretely for multigrid in `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`.

Its output semantics are straightforward:

$$
\\text{logits}_\theta(s, s', h) \longrightarrow P_\theta(a_h \mid s, s').
$$

The trainers also optimize an auxiliary cross-entropy loss on observed focal-human actions. That auxiliary loss is what teaches the network to approximate the posterior action distribution used in the simplified $X_h$ target.

## Relationship to `U_r`

Once simplified $X_h$ is available, the downstream structure stays the same:

$$
y(s) = \mathbb{E}_h[X_h(s, h)^{-\xi}],
$$

$$
U_r(s) = -y(s)^\eta.
$$

What changes is the meaning and scale of $X_h$:

- standard mode: goal-conditioned attainment aggregation
- simplified mode: successor-state control recursion, approximated from inverse dynamics in the learning-based path

Because terminal simplified $X_h$ is $1$, the code consistently clamps simplified-mode values to be at least `1.0` before they are used downstream.

## Exact reference vs current approximation

It is important to keep these separate.

The exact semantic reference is:

- tabular backward induction in `src/empo/backward_induction/phase2.py`

Within the learning-based helper, the exact fallback branch in `src/empo/learning_based/phase2/simplified_x_h.py` computes the exact local one-step quantity $q_h(s, s')$ for the sampled transition, but it does not replace the tabular fixed-point solution.

The current learning-based approximation is:

- train $P_\theta(a_h \mid s, s')$
- score each sampled transition by the Bayes-ratio surrogate
- reconstruct the sampled marginal factor $m_h(s' \mid s)$ exactly from the world model
- bootstrap $X_h$ from the corrected sampled target

So the code is no longer merely “adding inverse-dynamics diagnostics”. It is actually using the inverse-dynamics model inside the simplified $X_h$ target construction.

## Practical reading order

If you want the shortest reliable path through the code, read in this order:

1. `src/empo/backward_induction/phase2.py` for the exact simplified object
2. `src/empo/learning_based/phase2/simplified_x_h.py` for the shared TD target helper
3. `src/empo/learning_based/phase2/inverse_dynamics.py` for the abstract inverse-dynamics interface
4. `src/empo/learning_based/phase2/trainer.py` for the DQN integration
5. `src/empo/learning_based/phase2_ppo/trainer.py` for the PPO integration
6. `src/empo/learning_based/multigrid/phase2/trainer.py` and `src/empo/learning_based/multigrid/phase2_ppo/networks.py` for the actual multigrid network construction
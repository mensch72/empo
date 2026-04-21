# Simplified X_h Computation

This document explains the simplified, goal-agnostic `X_h` computation in two layers:

1. the theory: what quantity is being approximated and why it differs from standard `X_h`
2. the code: how the current tabular, DQN, and PPO implementations construct and train that quantity

The relevant implementation files are:

- `src/empo/backward_induction/phase2.py`
- `src/empo/learning_based/phase2/simplified_x_h.py`
- `src/empo/learning_based/phase2/trainer.py`
- `src/empo/learning_based/phase2_ppo/trainer.py`
- `src/empo/learning_based/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`

## Why simplified `X_h` exists

In the standard Phase 2 formulation, `X_h` is built from the goal-conditioned human attainment quantity `V_h^e(s, g_h)`. That path depends on Phase 1 style goal reasoning:

$$
X_h(s) = \mathbb{E}_{g_h}\left[V_h^e(s, g_h)^\zeta\right].
$$

This is the most direct implementation of the theory when we want `X_h` to summarize power by aggregating over hypothetical goals.

The simplified mode removes the explicit goal dimension and replaces it with a goal-agnostic recursion over successor states. This is useful when:

- we want a power-like quantity without enumerating or sampling goals at every update
- we want terminal states to have an explicit lower bound `X_h = 1`

In code, this mode is enabled with `use_simplified_x_h=True` in Phase 2 config objects.

## The theoretical object

The simplified `X_h` is the solution to the fixed-point equation

$$
X_h(s) = 1 + \gamma_h^\zeta \sum_{s'} q_h(s, s')^\zeta X_h(s').
$$

Here:

- `h` is the focal human agent
- `\gamma_h` is the human discount factor
- `\zeta` is the risk / power curvature parameter from the theory
- `q_h(s, s')` is the goal-agnostic probability mass that human `h` can force onto successor state `s'`, given the robot policy and the other humans' policies:
$$
q_h(s, s') = \max_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

This version of `X_h` is actually a special case of the general one: humans are perfectly rational, and each possible goal is a finite trajectory of states.

### Boundary condition

Terminal states are assigned

$$
X_h(s_{\mathrm{terminal}}) = 1.
$$

That makes `1` the natural lower bound in simplified mode. This is why simplified `X_h` networks use the feasible range `(1, \infty)` rather than `(0, 1)`.

### Bounded rationality

When `x_h_epsilon_h > 0`, the code does not use a pure best action. Instead it mixes the best human action with a uniform prior over human actions:

$$
q_h(s, s') = (1 - \epsilon_h) \max_{a_h} P(s' \mid s, a_h, \pi_{-h})
+ \epsilon_h \operatorname{mean}_{a_h} P(s' \mid s, a_h, \pi_{-h}).
$$

This should be read as bounded rationality in the simplified power recursion. It is distinct from exploration schedules such as `epsilon_h_start` / `epsilon_h_end`.

## What `q_h(s, s')` means operationally

The definition

$$
q_h(s, s') = \max_{a_h} P(s' \mid s, a_h, \pi_{-h})
$$

packs several policy assumptions into a single scalar:

1. Fix the current state `s`.
2. Fix a focal human `h`.
3. Let the robot act according to its current policy `\pi_r`.
4. Let all non-focal humans act according to their current policy priors.
5. For each action `a_h` available to human `h`, compute the total probability of reaching `s'` in one step.
6. Keep the largest of those probabilities.

So simplified `X_h` is not asking, "How much can human `h` achieve goal `g`?" It is asking, "How much successor-state control can human `h` exert from here, and how much future control does that successor state still contain?"

## Tabular reference implementation

The most faithful implementation of the theory is the backward-induction path in `src/empo/backward_induction/phase2.py`.

That code is useful as the semantic reference because it works directly with exact transition probabilities from the world model and solves the recursion over the reachable state graph. Conceptually it does the following:

1. Enumerate reachable states.
2. For each focal human and each state, compute `q_h(s, s')` from exact transition probabilities.
3. Solve the fixed-point recursion for `X_h` using the model-based structure of the environment.
4. Feed the resulting `X_h` values into the rest of Phase 2.

When reading the learning-based code, it helps to treat the tabular implementation as the ground-truth definition and the neural code as an approximation strategy for the same quantity.

## Learning-based implementation: shared helper

The shared helper lives in `src/empo/learning_based/phase2/simplified_x_h.py` and is used by both the DQN and PPO trainers.

Its public entrypoint is:

```python
compute_simplified_x_h_td_targets(...)
```

This helper does not solve the fixed-point exactly. Instead it builds one-step TD targets for sampled transitions `(s, s')` collected during training.

The target used in the helper is:

$$
\widehat{X}_h^{\mathrm{target}}(s, s') = 1 + \gamma_h^\zeta q_h(s, s')^\zeta X_h^{\mathrm{target}}(s').
$$

This is a stochastic one-step backup, not the full exact sum over all reachable successor states. In other words:

- the tabular code implements the fixed-point directly
- the learning-based code trains a network from sampled one-step TD targets derived from that fixed-point

### Inputs expected by the helper

The helper requires the trainers to prepare the pieces that are specific to each learning setup:

- `states`, `next_states`, `human_indices`: aligned samples
- `x_h_next_values`: bootstrap values `X_h_target(s')`
- `robot_policy_per_state`: the robot policy for each unique source state
- `action_index_to_tuple`: how a flat robot action index maps to per-robot actions
- `other_human_probs_fn`: a goal-agnostic marginal policy for other humans
- `world_model.transition_probabilities(state, actions)`: exact one-step dynamics

### What the helper computes internally

Inside `compute_simplified_x_h_td_targets`, the computation is:

1. Build the joint action distribution of the non-focal humans from independent marginals.
2. For each sampled `(state, focal_human)` pair, cache a transition-mass table for every possible human action `a_h`.
3. For each human action, sum over robot actions and other-human actions under their respective policies.
4. Query `world_model.transition_probabilities(state, joint_actions)`.
5. Extract the probability mass assigned to the observed `next_state` for each candidate `a_h`.
6. Take the max over `a_h`, or the `epsilon_h` mixture of max and mean.
7. Form the TD target `1 + gamma_h^zeta * q_h^zeta * X_h_target(next_state)`.

This is the core reason the helper is shared: the expensive part is not DQN- or PPO-specific. It is the reconstruction of `q_h(s, s')` from the world model.

## DQN path

The DQN trainer integration is in `src/empo/learning_based/phase2/trainer.py`.

The relevant method is:

```python
BasePhase2Trainer._compute_simplified_x_h_td_target(...)
```

That method prepares DQN-specific inputs for the shared helper:

1. It computes `X_h_target(s')` with `self.networks.x_h_target.forward_batch(...)`.
2. It clamps non-terminal simplified values to be at least `1.0`.
3. It obtains the robot policy from `q_r_target` using the current effective `beta_r`.
4. It wraps the human policy prior into the helper's expected `(state, agent_index) -> probability array` form.
5. It calls `compute_simplified_x_h_td_targets(...)`.

The X_h loss then uses MSE between:

- the online network prediction `x_h_pred = X_h_\theta(s, h)`
- the helper-generated TD target

### Where this is used in training

During auxiliary training, when `use_simplified_x_h=True`, the DQN trainer no longer builds `X_h` targets from `V_h^e(s, g_h)`. Instead it collects aligned `(state, next_state, human_idx, terminal)` samples from replay and uses the simplified target path.

That changes the meaning of the `X_h` network:

- standard mode: learn an aggregation of goal-conditioned human attainment values
- simplified mode: learn a goal-agnostic control recursion over successor states

## PPO path

The PPO integration is in `src/empo/learning_based/phase2_ppo/trainer.py`.

The analogous method is:

```python
PPOPhase2Trainer._compute_simplified_x_h_td_target(...)
```

Its structure mirrors the DQN path, but the robot policy is obtained from the actor-critic rather than from a `q_r_target` network:

1. Compute `X_h_target(s')` from `x_h_target` or `x_h`.
2. Convert each unique source state to PPO observations with the env wrapper.
3. Run the actor-critic to get action logits.
4. Softmax those logits to obtain `\pi_r(a_r \mid s)`.
5. Wrap the human prior into a goal-agnostic other-human policy function.
6. Call the shared helper.

During `train_auxiliary_step`, simplified mode changes the `X_h` branch exactly as in DQN: the target becomes the successor-state recursion rather than a `V_h^e` aggregate.

## Relationship to `U_r`

Once `X_h` is available, the rest of the power pipeline is unchanged in form.

The code computes the intermediate quantity

$$
y(s) = \mathbb{E}_h[X_h(s, h)^{-\xi}]
$$

and then the robot intrinsic reward is

$$
U_r(s) = -y(s)^\eta.
$$

Simplified mode therefore changes `U_r` indirectly by changing the semantics and scale of `X_h`:

- standard mode: `X_h` is goal-attainment based and typically lives in a bounded range
- simplified mode: `X_h` has lower bound `1` and no finite upper bound

That is why the code clamps simplified `X_h` at `min=1.0` before using it in the `U_r` computation.

## Feasible range and numerical conventions

In simplified mode, the code consistently treats:

- terminal `X_h` as exactly `1`
- valid non-terminal `X_h` as `\ge 1`
- the feasible range of learned `X_h` as `(1.0, inf)`

This differs from the standard mode, where `X_h` is treated as bounded and often clamped into `[10^{-3}, 1]` for stability.

That difference appears in both the network factories and the `U_r` target computation.

## Inverse-dynamics network classes

There are now inverse-dynamics modules in:

- `src/empo/learning_based/phase2/inverse_dynamics.py`
- `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py`

These define a learned model that takes `(state, next_state, focal_human)` and predicts a distribution over the focal human action.

The intended interpretation is:

$$
P_\theta(a_h \mid s, s').
$$

The trainer code also adds an auxiliary cross-entropy loss on observed human actions when simplified mode is enabled.

### Important current distinction

The exact simplified-`X_h` target construction still lives in the shared helper in `src/empo/learning_based/phase2/simplified_x_h.py`, and that helper is still written around exact world-model transition masses `q_h(s, s')`.

So there are two layers to keep separate:

1. the current authoritative target construction: exact world-model `q_h` from `transition_probabilities(...)`
2. the inverse-dynamics scaffolding: a learned approximation path being introduced around the same conceptual quantity

The config comments use the name `inverse_dynamics(s, s')` for the quantity that the theory previously called `q_h(s, s')`. In the code today, the cleanest reference implementation is still the exact helper and the tabular backward-induction path.

## End-to-end code flow summary

When `use_simplified_x_h=True`, the learning-based path is:

1. Collect a transition `(s, a, s')` in replay.
2. For each focal human in that transition, evaluate the online `X_h` network at `s`.
3. Evaluate the target `X_h` network at `s'`, unless `s'` is terminal, in which case use `1`.
4. Recover the robot policy `\pi_r` at `s`.
5. Recover the non-focal human action marginals at `s`.
6. Use the world model to compute the one-step state-control quantity `q_h(s, s')`.
7. Build the TD target `1 + \gamma_h^\zeta q_h(s, s')^\zeta X_h^{\mathrm{target}}(s')`.
8. Fit the `X_h` network by MSE to that target.
9. Use the resulting `X_h` values downstream in `U_r` and therefore in robot policy computation.

## Interpretation

The simplified recursion changes the semantics of `X_h` from

- "aggregate discounted goal achievement ability"

to

- "aggregate discounted successor-state control power"

under the current robot policy and the current other-human policies.

That makes simplified `X_h` especially natural when the goal set is large, expensive to sample, or not the right abstraction for the question being studied.

## Practical reading guide

If you want to understand the code in the shortest reliable order, read it in this sequence:

1. `src/empo/backward_induction/phase2.py` for the exact mathematical definition
2. `src/empo/learning_based/phase2/simplified_x_h.py` for the shared TD target construction
3. `src/empo/learning_based/phase2/trainer.py` for the DQN integration
4. `src/empo/learning_based/phase2_ppo/trainer.py` for the PPO integration
5. `src/empo/learning_based/phase2/inverse_dynamics.py` and `src/empo/learning_based/multigrid/phase2/inverse_dynamics.py` for the learned approximation layer being added around the same idea
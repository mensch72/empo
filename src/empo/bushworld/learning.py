"""
Learning-based Phase 2 robot-policy computation for BushWorld.

This module provides a *compact, self-contained* learning path for BushWorld that
mirrors the exact EMPO Phase 2 equations implemented by the tabular backward
induction in :mod:`empo.backward_induction.phase2` (see ``_rp_process_single_state``).
It is intended as a small, readable reference for how to wire a brand-new
``WorldModel`` into a learning-based Phase 2 computation without replicating the
large, MultiGrid-specific neural stack in :mod:`empo.learning_based`.

Two interchangeable "learning versions" are offered, selectable via ``method=``:

``"tabular"`` (a.k.a. ``"value_iteration"``)
    Fitted value iteration over the *exact* reachable-state DAG using Python
    dictionaries as the value tables. Because BushWorld exposes exact
    ``transition_probabilities``, this converges to the same fixed point as
    backward induction and is used to validate the equations.

``"neural"`` (a.k.a. ``"dqn"`` / ``"alphazero"``)
    Model-based fitted value iteration with small PyTorch networks
    (``BushWorldStateEncoder`` + value heads). Targets are bootstrapped from a
    slowly-updated *target* network using the model's exact one-step transitions
    (an AlphaZero-style, model-based target). ``"dqn"`` uses one-step targets;
    ``"alphazero"`` is an alias kept for discoverability and behaves identically
    here (the model-based one-step target already uses exact transition
    probabilities rather than sampled returns).

Both versions return a :class:`~empo.robot_policy.RobotPolicy` whose action
distribution is the EMPO power-law policy ``pi_r(a_r) ∝ (-Q_r(a_r))^{-beta_r}``,
directly comparable with the tabular policy produced by
``empo.backward_induction.phase2.compute_robot_policy``.

Terminology note (per repository conventions): policies here are *computed /
approximated* as solutions to the EMPO equations; they are not "optimized"
against an environment reward. ``beta_r``, ``gamma_h``, ``gamma_r``, ``zeta``,
``xi`` and ``eta`` are *theory parameters*, not training hyperparameters.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pickle
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from empo.backward_induction.phase2 import TabularRobotPolicy
from empo.robot_policy import RobotPolicy

State = Tuple[Any, ...]
RobotProfile = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# Theory parameters
# --------------------------------------------------------------------------- #
@dataclass
class Phase2Params:
    """EMPO Phase 2 *theory* parameters (not training hyperparameters).

    Mirrors the keyword parameters of
    :func:`empo.backward_induction.phase2.compute_robot_policy` so that a learned
    policy can be compared against backward induction under identical settings.

    Discounting uses unit transition durations (BushWorld's default), so a
    discrete discount ``gamma`` maps to a per-step factor of exactly ``gamma``.
    """

    beta_r: float = 5.0
    gamma_h: float = 1.0
    gamma_r: float = 1.0
    zeta: float = 1.0
    xi: float = 1.0
    eta: float = 1.0
    terminal_Vr: float = -1e-10

    def __post_init__(self) -> None:
        if not (0.0 < self.gamma_h <= 1.0):
            raise ValueError(f"gamma_h must be in (0, 1], got {self.gamma_h}")
        if not (0.0 < self.gamma_r <= 1.0):
            raise ValueError(f"gamma_r must be in (0, 1], got {self.gamma_r}")
        if self.terminal_Vr >= 0.0:
            raise ValueError("terminal_Vr must be < 0 (Q_r values are always negative)")

    @property
    def robot_duration_weight(self) -> float:
        """Reward weight ``(1 - gamma_r)/(-ln gamma_r)`` for unit durations.

        Equals ``1.0`` when ``gamma_r == 1`` (the undiscounted case).
        """
        if self.gamma_r >= 1.0:
            return 1.0
        rho_r = -math.log(self.gamma_r)
        return float(-math.expm1(-rho_r) / rho_r)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _robot_profiles(num_robots: int, num_actions: int) -> List[RobotProfile]:
    """Enumerate all joint robot action profiles."""
    return list(itertools.product(range(num_actions), repeat=num_robots))


def enumerate_reachable_states(
    env: Any,
    human_policy_prior: Any,
) -> List[State]:
    """Breadth-first enumeration of states reachable from ``env.initial_state()``.

    Considers *all* robot action profiles (the robot policy is what we are
    computing) and every human action profile with positive prior probability.
    Returns states in discovery order; terminal states are included but not
    expanded.
    """
    num_robots = env.num_robots
    num_actions = env.action_space.n
    robot_profiles = _robot_profiles(num_robots, num_actions)

    init = env.initial_state()
    seen = {init}
    order: List[State] = []
    frontier: List[State] = [init]
    while frontier:
        state = frontier.pop()
        order.append(state)
        if env.is_terminal(state):
            continue
        human_profiles = human_policy_prior.profile_distribution(state)
        for ra in robot_profiles:
            for _prob_h, human_profile in human_profiles:
                actions = list(ra) + [int(a) for a in human_profile]
                for _p, succ in env.transition_probabilities(state, actions):
                    if succ not in seen:
                        seen.add(succ)
                        frontier.append(succ)
    return order


def phase2_local_update(
    env: Any,
    human_policy_prior: Any,
    goal_generator: Any,
    state: State,
    params: Phase2Params,
    vr_fn: Callable[[State], float],
    vhe_fn: Callable[[State, int, Any], float],
    robot_profiles: Sequence[RobotProfile],
    human_agent_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, float, Dict[Tuple[int, Any], float], float]:
    """Apply one exact EMPO Phase 2 Bellman backup at ``state``.

    This is the local operator shared by the tabular and neural learners. It is a
    faithful re-implementation of ``_rp_process_single_state`` (gamma branch with
    unit durations) using the world model's exact one-step transitions.

    Args:
        vr_fn: Callback returning ``V_r(s')`` for a successor (callers handle the
            terminal case, typically returning ``params.terminal_Vr``).
        vhe_fn: Callback returning ``V_h^e(s', agent_index, goal)`` for a
            successor (terminal successors should return ``0.0``).

    Returns:
        ``(qr, pi_r, vr_target, vhe_targets, ur)`` where ``qr`` and ``pi_r`` are
        arrays aligned with ``robot_profiles``, ``vhe_targets`` maps
        ``(agent_index, goal)`` to the backed-up ``V_h^e`` value, and ``ur`` is
        the robot intrinsic reward ``U_r``.
    """
    gamma_r = params.gamma_r
    gamma_h = params.gamma_h
    n_profiles = len(robot_profiles)

    # --- Q_r(s, a_r) under the marginal human policy ----------------------- #
    human_marginal = human_policy_prior.profile_distribution(state)
    qr = np.zeros(n_profiles, dtype=np.float64)
    for i, ra in enumerate(robot_profiles):
        v = 0.0
        for prob_h, human_profile in human_marginal:
            actions = list(ra) + [int(a) for a in human_profile]
            for prob_t, succ in env.transition_probabilities(state, actions):
                v += prob_h * prob_t * gamma_r * vr_fn(succ)
        qr[i] = v

    # Power-law robot policy: pi_r(a) ∝ (-Q_r(a))^{-beta_r} (Q_r < 0). Computed
    # in log-space for numerical stability (matches backward induction).
    neg_qr = -qr
    neg_qr = np.where(neg_qr <= 0.0, 1e-300, neg_qr)
    log_powers = -params.beta_r * np.log(neg_qr)
    log_powers -= log_powers.max()
    pi_r = np.exp(log_powers)
    pi_r /= pi_r.sum()

    # --- V_h^e, X_h, U_r --------------------------------------------------- #
    vhe_targets: Dict[Tuple[int, Any], float] = {}
    powersum = 0.0
    for agent_index in human_agent_indices:
        xh = 0.0
        for goal, weight in goal_generator.generate(state, agent_index):
            goal_profiles = human_policy_prior.profile_distribution_with_fixed_goal(
                state, agent_index, goal
            )
            vh = 0.0
            for i, ra in enumerate(robot_profiles):
                if pi_r[i] == 0.0:
                    continue
                v_local = 0.0
                for prob_h, human_profile in goal_profiles:
                    actions = list(ra) + [int(a) for a in human_profile]
                    for prob_t, succ in env.transition_probabilities(state, actions):
                        if goal.is_achieved(succ):
                            sv = 1.0
                        else:
                            sv = gamma_h * vhe_fn(succ, agent_index, goal)
                        v_local += prob_h * prob_t * sv
                vh += pi_r[i] * v_local
            vhe_targets[(agent_index, goal)] = vh
            xh += weight * (vh ** params.zeta)
        powersum += xh ** (-params.xi)

    y = powersum / len(human_agent_indices)
    ur = -(y ** params.eta)
    vr_target = params.robot_duration_weight * ur + float(np.dot(pi_r, qr))
    return qr, pi_r, vr_target, vhe_targets, ur


# --------------------------------------------------------------------------- #
# Tabular fitted value iteration
# --------------------------------------------------------------------------- #
class _TabularValueTables:
    """Dictionary-backed value tables for the tabular learner (picklable)."""

    def __init__(self, terminal_Vr: float):
        self.terminal_Vr = float(terminal_Vr)
        self.vr: Dict[State, float] = {}
        self.vhe: Dict[State, Dict[Tuple[int, Any], float]] = {}

    def vr_fn(self, env: Any) -> Callable[[State], float]:
        def _fn(succ: State) -> float:
            if env.is_terminal(succ):
                return self.terminal_Vr
            return self.vr.get(succ, self.terminal_Vr)
        return _fn

    def vhe_fn(self, env: Any) -> Callable[[State, int, Any], float]:
        def _fn(succ: State, agent_index: int, goal: Any) -> float:
            if env.is_terminal(succ):
                return 0.0
            return self.vhe.get(succ, {}).get((agent_index, goal), 0.0)
        return _fn


def compute_tabular_phase2(
    env: Any,
    human_policy_prior: Any,
    params: Phase2Params,
    *,
    goal_generator: Any = None,
    max_iterations: int = 1000,
    tol: float = 1e-9,
    quiet: bool = True,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    initial_tables: Optional[_TabularValueTables] = None,
) -> Tuple["LearnedTabularRobotPolicy", _TabularValueTables, Dict[str, Any]]:
    """Compute a robot policy by fitted value iteration over the exact DAG.

    Returns ``(policy, tables, history)``. ``tables`` can be checkpointed and
    passed back via ``initial_tables`` to resume.
    """
    if goal_generator is None:
        goal_generator = env.possible_goal_generator
    human_agent_indices = list(env.human_agent_indices)
    robot_agent_indices = list(env.robot_agent_indices)
    robot_profiles = _robot_profiles(env.num_robots, env.action_space.n)

    states = enumerate_reachable_states(env, human_policy_prior)
    # Process non-terminal states; backups read successors, so iterate to a fixed
    # point (Gauss-Seidel sweeps converge to the backward-induction solution).
    nonterminal = [s for s in states if not env.is_terminal(s)]

    tables = initial_tables if initial_tables is not None else _TabularValueTables(params.terminal_Vr)
    vr_fn = tables.vr_fn(env)
    vhe_fn = tables.vhe_fn(env)
    pi_table: Dict[State, Dict[RobotProfile, float]] = {}

    history: Dict[str, Any] = {"deltas": [], "num_states": len(states), "iterations": 0}
    delta = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        for state in nonterminal:
            qr, pi_r, vr_target, vhe_targets, _ur = phase2_local_update(
                env, human_policy_prior, goal_generator, state, params,
                vr_fn, vhe_fn, robot_profiles, human_agent_indices,
            )
            old_vr = tables.vr.get(state, params.terminal_Vr)
            delta = max(delta, abs(vr_target - old_vr))
            tables.vr[state] = vr_target
            tables.vhe[state] = vhe_targets
            pi_table[state] = {rp: float(pi_r[i]) for i, rp in enumerate(robot_profiles)}
        history["deltas"].append(delta)
        if progress_callback is not None:
            progress_callback(iteration, delta)
        if not quiet:
            print(f"[tabular] iteration {iteration}: max |dV_r| = {delta:.3e}")
        if delta < tol:
            break
    history["iterations"] = iteration
    history["final_delta"] = delta

    policy = LearnedTabularRobotPolicy(env, robot_agent_indices, pi_table)
    return policy, tables, history


class LearnedTabularRobotPolicy(TabularRobotPolicy):
    """Tabular robot policy produced by :func:`compute_tabular_phase2`.

    Identical interface to ``empo.backward_induction.phase2.TabularRobotPolicy``;
    subclassed so the comparison example can treat both uniformly.
    """

    def save(self, path: str) -> None:
        """Persist the policy lookup table (without the world-model reference)."""
        payload = {
            "kind": "tabular",
            "robot_agent_indices": list(self.robot_agent_indices),
            "num_actions": int(self.num_actions),
            "values": self.values,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str, world_model: Any) -> "LearnedTabularRobotPolicy":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("kind") != "tabular":
            raise ValueError(f"{path} is not a tabular BushWorld policy checkpoint")
        policy = cls(world_model, payload["robot_agent_indices"], payload["values"])
        policy.num_actions = int(payload["num_actions"])
        return policy


# --------------------------------------------------------------------------- #
# Neural fitted value iteration
# --------------------------------------------------------------------------- #
def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The neural BushWorld learner requires PyTorch. Install it with "
            "`pip install torch`, or use method='tabular'."
        ) from exc
    import torch
    return torch


def state_to_features(env: Any, state: State) -> np.ndarray:
    """Encode a BushWorld state into a fixed-length float feature vector.

    Layout: ``[step/max_steps,  per-player (x/W, y/H)...,  per-cell density/B]``.
    This is the raw, world-specific encoding consumed by
    :class:`BushWorldStateEncoder`.
    """
    step_count, positions, densities = state
    width, height = env.width, env.height
    feats: List[float] = [float(step_count) / max(1, env.max_steps)]
    for (x, y) in positions:
        feats.append(float(x) / max(1, width - 1) if width > 1 else 0.0)
        feats.append(float(y) / max(1, height - 1) if height > 1 else 0.0)
    bnorm = float(max(1, env.B))
    feats.extend(float(d) / bnorm for d in densities)
    return np.asarray(feats, dtype=np.float32)


def feature_dim(env: Any) -> int:
    return 1 + 2 * env.num_players + env.width * env.height


class BushWorldStateEncoder:
    """Small MLP encoder mapping BushWorld state features to a latent vector.

    Wrapped lazily so importing this module does not require PyTorch. Call
    :meth:`module` to obtain the underlying ``torch.nn.Module``.
    """

    def __init__(self, env: Any, hidden_dim: int = 128, latent_dim: int = 128):
        self.env = env
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = feature_dim(env)

    def module(self):
        torch = _require_torch()
        import torch.nn as nn

        class _Encoder(nn.Module):
            def __init__(self, in_dim, hidden, latent):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, latent),
                    nn.ReLU(),
                )

            def forward(self, x):
                return self.net(x)

        return _Encoder(self.input_dim, self.hidden_dim, self.latent_dim)


def _build_network(
    env: Any,
    encoder_spec: BushWorldStateEncoder,
    n_robot_profiles: int,
    value_scale: float = 1.0,
):
    """Construct the combined encoder + value heads as a single nn.Module.

    ``value_scale`` rescales the (negative) Q_r and V_r outputs so the network
    fits O(1) pre-activations even when the EMPO values have large magnitude.
    This is essential for stable fitted value iteration.
    """
    torch = _require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    num_humans = env.num_humans
    num_cells = env.width * env.height
    latent = encoder_spec.latent_dim

    class _Phase2Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder_spec.module()
            # Q_r per robot action profile (forced negative via -softplus).
            self.q_head = nn.Linear(latent, n_robot_profiles)
            # V_r scalar (forced negative).
            self.vr_head = nn.Linear(latent, 1)
            # V_h^e per (human, cell goal), squashed to (0, 1) via sigmoid.
            self.vhe_head = nn.Linear(latent, num_humans * num_cells)
            self.num_humans = num_humans
            self.num_cells = num_cells
            self.register_buffer("value_scale", torch.tensor(float(value_scale)))

        def forward(self, x):
            z = self.encoder(x)
            qr = (-F.softplus(self.q_head(z)) - 1e-9) * self.value_scale
            vr = (-F.softplus(self.vr_head(z)).squeeze(-1) - 1e-9) * self.value_scale
            vhe = torch.sigmoid(self.vhe_head(z)).view(-1, self.num_humans, self.num_cells)
            return qr, vr, vhe

    return _Phase2Net()


def estimate_value_scale(
    env: Any,
    human_policy_prior: Any,
    goal_generator: Any,
    params: Phase2Params,
    robot_profiles: Sequence[RobotProfile],
    human_agent_indices: Sequence[int],
    nonterminal_states: Sequence[State],
    sweeps: Optional[int] = None,
) -> float:
    """Estimate a typical ``|V_r|`` magnitude via a few exact Gauss-Seidel sweeps.

    Used purely as a fixed normalization constant for the neural learner (a
    model-based reward/value-scale estimate, analogous to reward normalization).
    """
    tables = _TabularValueTables(params.terminal_Vr)
    vr_fn = tables.vr_fn(env)
    vhe_fn = tables.vhe_fn(env)
    if sweeps is None:
        sweeps = max(2, env.max_steps)
    max_abs = 1.0
    for _ in range(sweeps):
        for state in nonterminal_states:
            _qr, _pi, vr_t, vhe_t, _ur = phase2_local_update(
                env, human_policy_prior, goal_generator, state, params,
                vr_fn, vhe_fn, robot_profiles, human_agent_indices,
            )
            tables.vr[state] = vr_t
            tables.vhe[state] = vhe_t
            max_abs = max(max_abs, abs(vr_t))
    return float(1.1 * max_abs)


class LearnedNeuralRobotPolicy(RobotPolicy):
    """Neural robot policy: ``pi_r(a_r) ∝ (-Q_r(a_r))^{-beta_r}`` from a network."""

    def __init__(
        self,
        env: Any,
        net: Any,
        params: Phase2Params,
        robot_profiles: Sequence[RobotProfile],
        encoder_spec: BushWorldStateEncoder,
        device: str = "cpu",
    ):
        self.world_model = env
        self.env = env
        self.net = net
        self.params = params
        self.robot_profiles = list(robot_profiles)
        self.robot_agent_indices = list(env.robot_agent_indices)
        self.encoder_spec = encoder_spec
        self.device = device

    def q_values(self, state: State) -> np.ndarray:
        torch = _require_torch()
        self.net.eval()
        with torch.no_grad():
            feats = torch.from_numpy(state_to_features(self.env, state)).to(self.device).unsqueeze(0)
            qr, _vr, _vhe = self.net(feats)
        return qr.squeeze(0).cpu().numpy()

    def policy_distribution(self, state: State) -> Dict[RobotProfile, float]:
        qr = self.q_values(state)
        neg_qr = np.where(-qr <= 0.0, 1e-300, -qr)
        log_powers = -self.params.beta_r * np.log(neg_qr)
        log_powers -= log_powers.max()
        probs = np.exp(log_powers)
        probs /= probs.sum()
        return {rp: float(probs[i]) for i, rp in enumerate(self.robot_profiles)}

    def __call__(self, state: State) -> Dict[RobotProfile, float]:
        return self.policy_distribution(state)

    def sample(self, state: State) -> RobotProfile:
        dist = self.policy_distribution(state)
        profiles = list(dist.keys())
        probs = np.fromiter((dist[p] for p in profiles), dtype=np.float64, count=len(profiles))
        probs /= probs.sum()
        idx = int(np.random.choice(len(profiles), p=probs))
        return profiles[idx]

    def get_action(self, state: State, robot_agent_index: int) -> int:
        profile = self.sample(state)
        pos = self.robot_agent_indices.index(robot_agent_index)
        return profile[pos]

    def reset(self, world_model: Any) -> None:
        self.world_model = world_model
        self.env = world_model

    def save(self, path: str) -> None:
        torch = _require_torch()
        payload = {
            "kind": "neural",
            "state_dict": self.net.state_dict(),
            "params": asdict(self.params),
            "robot_profiles": self.robot_profiles,
            "hidden_dim": self.encoder_spec.hidden_dim,
            "latent_dim": self.encoder_spec.latent_dim,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, world_model: Any, device: str = "cpu") -> "LearnedNeuralRobotPolicy":
        torch = _require_torch()
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("kind") != "neural":
            raise ValueError(f"{path} is not a neural BushWorld policy checkpoint")
        params = Phase2Params(**payload["params"])
        encoder_spec = BushWorldStateEncoder(
            world_model, hidden_dim=payload["hidden_dim"], latent_dim=payload["latent_dim"]
        )
        robot_profiles = [tuple(rp) for rp in payload["robot_profiles"]]
        net = _build_network(world_model, encoder_spec, len(robot_profiles)).to(device)
        net.load_state_dict(payload["state_dict"])
        return cls(world_model, net, params, robot_profiles, encoder_spec, device=device)


def _goal_cell_index(env: Any, goal: Any) -> int:
    """Map a cell goal to its flat cell index (neural V_h^e head layout)."""
    target = getattr(goal, "target_pos", None)
    if target is None:
        raise ValueError(
            "The neural BushWorld learner currently supports cell goals only "
            "(goals exposing a `target_pos`). Use method='tabular' for rectangle "
            "goals, or extend the V_h^e head layout."
        )
    x, y = target
    return env.cell_index(int(x), int(y))


def compute_neural_phase2(
    env: Any,
    human_policy_prior: Any,
    params: Phase2Params,
    *,
    goal_generator: Any = None,
    num_iterations: int = 400,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    target_sync_every: int = 20,
    hidden_dim: int = 128,
    latent_dim: int = 128,
    device: str = "cpu",
    seed: Optional[int] = None,
    quiet: bool = True,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple["LearnedNeuralRobotPolicy", Dict[str, Any], Dict[str, Any]]:
    """Compute a robot policy via model-based fitted value iteration (neural).

    Targets are computed from a target network using the world model's exact
    one-step transitions and the EMPO Phase 2 equations (``phase2_local_update``).

    Returns ``(policy, checkpoint, history)``. ``checkpoint`` is a dict that can be
    persisted and passed back via ``checkpoint=`` to resume training.
    """
    torch = _require_torch()
    import torch.nn.functional as F

    if goal_generator is None:
        goal_generator = env.possible_goal_generator
    human_agent_indices = list(env.human_agent_indices)
    robot_profiles = _robot_profiles(env.num_robots, env.action_space.n)
    n_profiles = len(robot_profiles)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    states = enumerate_reachable_states(env, human_policy_prior)
    nonterminal = [s for s in states if not env.is_terminal(s)]
    if not nonterminal:
        raise ValueError("No non-terminal reachable states; nothing to compute.")

    num_cells = env.width * env.height

    # Fixed value-scale normalization (essential for stable fitted value iteration
    # given the potentially large magnitude of EMPO V_r / Q_r values).
    if checkpoint is not None and "value_scale" in checkpoint:
        value_scale = float(checkpoint["value_scale"])
    else:
        value_scale = estimate_value_scale(
            env, human_policy_prior, goal_generator, params,
            robot_profiles, human_agent_indices, nonterminal,
        )

    encoder_spec = BushWorldStateEncoder(env, hidden_dim=hidden_dim, latent_dim=latent_dim)
    net = _build_network(env, encoder_spec, n_profiles, value_scale=value_scale).to(device)
    target_net = _build_network(env, encoder_spec, n_profiles, value_scale=value_scale).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    start_iteration = 0
    history: Dict[str, Any] = {"losses": []}
    if checkpoint is not None:
        net.load_state_dict(checkpoint["state_dict"])
        target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint.get("iteration", 0))
        history = checkpoint.get("history", history)
    else:
        target_net.load_state_dict(net.state_dict())

    def make_target_callbacks():
        """Bootstrap callbacks reading the (frozen) target network."""
        cache: Dict[State, Tuple[np.ndarray, float, np.ndarray]] = {}

        def _predict(succ: State):
            cached = cache.get(succ)
            if cached is not None:
                return cached
            feats = torch.from_numpy(state_to_features(env, succ)).to(device).unsqueeze(0)
            with torch.no_grad():
                qr, vr, vhe = target_net(feats)
            out = (
                qr.squeeze(0).cpu().numpy(),
                float(vr.item()),
                vhe.squeeze(0).cpu().numpy(),  # shape (num_humans, num_cells)
            )
            cache[succ] = out
            return out

        def vr_fn(succ: State) -> float:
            if env.is_terminal(succ):
                return params.terminal_Vr
            return _predict(succ)[1]

        def vhe_fn(succ: State, agent_index: int, goal: Any) -> float:
            if env.is_terminal(succ):
                return 0.0
            h_pos = human_agent_indices.index(agent_index)
            cell = _goal_cell_index(env, goal)
            return float(_predict(succ)[2][h_pos, cell])

        return vr_fn, vhe_fn

    rng = np.random.default_rng(seed)
    last_loss = float("nan")
    for iteration in range(start_iteration, num_iterations):
        vr_fn, vhe_fn = make_target_callbacks()
        # Sample a minibatch of non-terminal states and compute exact targets.
        idx = rng.choice(len(nonterminal), size=min(batch_size, len(nonterminal)), replace=False)
        batch_states = [nonterminal[i] for i in idx]

        feat_batch = np.stack([state_to_features(env, s) for s in batch_states])
        qr_targets = np.zeros((len(batch_states), n_profiles), dtype=np.float32)
        vr_targets = np.zeros(len(batch_states), dtype=np.float32)
        vhe_targets = np.zeros((len(batch_states), env.num_humans, num_cells), dtype=np.float32)
        vhe_mask = np.zeros((len(batch_states), env.num_humans, num_cells), dtype=np.float32)

        for b, state in enumerate(batch_states):
            qr, _pi, vr_t, vhe_map, _ur = phase2_local_update(
                env, human_policy_prior, goal_generator, state, params,
                vr_fn, vhe_fn, robot_profiles, human_agent_indices,
            )
            qr_targets[b] = qr
            vr_targets[b] = vr_t
            for (agent_index, goal), value in vhe_map.items():
                h_pos = human_agent_indices.index(agent_index)
                cell = _goal_cell_index(env, goal)
                vhe_targets[b, h_pos, cell] = value
                vhe_mask[b, h_pos, cell] = 1.0

        net.train()
        feats_t = torch.from_numpy(feat_batch).to(device)
        qr_t = torch.from_numpy(qr_targets).to(device)
        vr_tt = torch.from_numpy(vr_targets).to(device)
        vhe_t = torch.from_numpy(vhe_targets).to(device)
        mask_t = torch.from_numpy(vhe_mask).to(device)

        qr_pred, vr_pred, vhe_pred = net(feats_t)
        # Normalize value losses by value_scale so Q_r/V_r are fit in O(1) space.
        inv_scale = 1.0 / value_scale
        loss_q = F.mse_loss(qr_pred * inv_scale, qr_t * inv_scale)
        loss_vr = F.mse_loss(vr_pred * inv_scale, vr_tt * inv_scale)
        denom = mask_t.sum().clamp_min(1.0)
        loss_vhe = ((vhe_pred - vhe_t) ** 2 * mask_t).sum() / denom
        loss = loss_q + loss_vr + loss_vhe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        history["losses"].append(last_loss)

        if (iteration + 1) % target_sync_every == 0:
            target_net.load_state_dict(net.state_dict())

        if progress_callback is not None:
            progress_callback(iteration + 1, last_loss)
        if not quiet and (iteration + 1) % max(1, num_iterations // 10) == 0:
            print(f"[neural] iteration {iteration + 1}/{num_iterations}: loss = {last_loss:.4e}")

    target_net.load_state_dict(net.state_dict())
    policy = LearnedNeuralRobotPolicy(
        env, net, params, robot_profiles, encoder_spec, device=device
    )
    out_checkpoint = {
        "state_dict": net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": num_iterations,
        "history": history,
        "params": asdict(params),
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "robot_profiles": robot_profiles,
        "value_scale": value_scale,
    }
    return policy, out_checkpoint, history


# --------------------------------------------------------------------------- #
# Unified dispatcher + checkpointing
# --------------------------------------------------------------------------- #
_METHOD_ALIASES = {
    "tabular": "tabular",
    "value_iteration": "tabular",
    "vi": "tabular",
    "neural": "neural",
    "dqn": "neural",
    "alphazero": "neural",
}


def train_bushworld_phase2(
    env: Any,
    human_policy_prior: Any,
    params: Optional[Phase2Params] = None,
    *,
    method: str = "tabular",
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    quiet: bool = True,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    **kwargs: Any,
) -> Tuple[RobotPolicy, Dict[str, Any]]:
    """Compute a BushWorld robot policy with the selected learning version.

    Args:
        method: One of ``"tabular"``/``"value_iteration"`` or
            ``"neural"``/``"dqn"``/``"alphazero"``.
        checkpoint_path: If given, training state is saved here and (when
            ``resume`` is True and the file exists) recovered from it.
        resume: Whether to resume from ``checkpoint_path`` if it exists.
        **kwargs: Forwarded to the underlying ``compute_*`` function.

    Returns:
        ``(policy, history)``.
    """
    params = params or Phase2Params()
    resolved = _METHOD_ALIASES.get(method.lower())
    if resolved is None:
        raise ValueError(
            f"Unknown method {method!r}; choose from {sorted(set(_METHOD_ALIASES))}."
        )

    if resolved == "tabular":
        initial_tables = None
        if checkpoint_path and resume and os.path.exists(checkpoint_path):
            if not quiet:
                print(f"[tabular] resuming from checkpoint {checkpoint_path}")
            with open(checkpoint_path, "rb") as fh:
                initial_tables = pickle.load(fh)
        policy, tables, history = compute_tabular_phase2(
            env, human_policy_prior, params,
            quiet=quiet, progress_callback=progress_callback,
            initial_tables=initial_tables, **kwargs,
        )
        if checkpoint_path:
            with open(checkpoint_path, "wb") as fh:
                pickle.dump(tables, fh)
        return policy, history

    # Neural
    checkpoint = None
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        if not quiet:
            print(f"[neural] resuming from checkpoint {checkpoint_path}")
        torch = _require_torch()
        checkpoint = torch.load(checkpoint_path, map_location=kwargs.get("device", "cpu"), weights_only=False)
    policy, out_checkpoint, history = compute_neural_phase2(
        env, human_policy_prior, params,
        quiet=quiet, progress_callback=progress_callback,
        checkpoint=checkpoint, **kwargs,
    )
    if checkpoint_path:
        torch = _require_torch()
        torch.save(out_checkpoint, checkpoint_path)
    return policy, history


def save_policy(policy: RobotPolicy, path: str) -> None:
    """Persist a learned BushWorld robot policy (tabular or neural)."""
    if isinstance(policy, LearnedNeuralRobotPolicy):
        policy.save(path)
    elif isinstance(policy, LearnedTabularRobotPolicy):
        policy.save(path)
    elif isinstance(policy, TabularRobotPolicy):
        # Backward-induction policy: store like a tabular learned policy.
        payload = {
            "kind": "tabular",
            "robot_agent_indices": list(policy.robot_agent_indices),
            "num_actions": int(policy.num_actions),
            "values": policy.values,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
    else:
        raise TypeError(f"Don't know how to save policy of type {type(policy)!r}")


def load_policy(path: str, world_model: Any, device: str = "cpu") -> RobotPolicy:
    """Load a learned BushWorld robot policy saved by :func:`save_policy`.

    Auto-detects the checkpoint kind (tabular pickle vs. neural torch archive).
    """
    # Try tabular (pickle) first; fall back to neural (torch archive).
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict) and payload.get("kind") == "tabular":
            policy = LearnedTabularRobotPolicy(
                world_model, payload["robot_agent_indices"], payload["values"]
            )
            policy.num_actions = int(payload["num_actions"])
            return policy
    except (pickle.UnpicklingError, KeyError, EOFError, ValueError):
        pass
    return LearnedNeuralRobotPolicy.load(path, world_model, device=device)

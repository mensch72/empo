#!/usr/bin/env python3
"""
Phase 2 PPO with Tabular U_r from Backward Induction.

Diagnostic script that isolates the PPO reward→policy pipeline from the
auxiliary-network reward computation.  It runs backward induction
(Phase 1 + Phase 2) to obtain exact tabular U_r(s) values, then uses
those as plug-in rewards for PPO training — bypassing V_h^e, X_h, and
U_r neural networks entirely.

If the robot learns with tabular U_r but not with neural U_r, the
problem is in auxiliary-network training.
If the robot still doesn't learn even with tabular U_r, the problem is
in the PPO integration (PufferLib, advantage computation, etc.).

Usage:
    # Inside Docker container (make shell):
    python examples/phase2/phase2_ppo_tabular_reward.py

    # Specific world:
    python examples/phase2/phase2_ppo_tabular_reward.py --world trivial.yaml

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_tabular_reward.py

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)
for _subdir in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    _path = os.path.join(_PROJECT_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gym_multigrid.multigrid import MultiGridEnv, SmallActions  # noqa: E402

from empo.backward_induction.phase1 import compute_human_policy_prior  # noqa: E402
from empo.backward_induction.phase2 import compute_robot_policy  # noqa: E402
from empo.human_policy_prior import HeuristicPotentialPolicy  # noqa: E402
from empo.learning_based.multigrid import PathDistanceCalculator  # noqa: E402
from empo.learning_based.phase2_ppo.config import PPOPhase2Config  # noqa: E402
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer  # noqa: E402
from empo.learning_based.multigrid.phase2_ppo import (  # noqa: E402
    MultiGridWorldModelEnv,
    create_multigrid_ppo_networks,
)

# ======================================================================
# Environment helpers
# ======================================================================

_WORLDS_DIR = os.path.join(_PROJECT_ROOT, "multigrid_worlds")


def _resolve_world_path(world: str) -> str:
    """Resolve a world specifier to an absolute YAML path."""
    if not world.endswith(".yaml") and not world.endswith(".yml"):
        world = world + ".yaml"
    if os.path.isabs(world):
        path = world
    else:
        path = os.path.join(_WORLDS_DIR, world)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"World file not found: {path}")
    return path


def _create_world_model(world_yaml: str, max_steps: int | None = None) -> MultiGridEnv:
    """Load a world from a YAML config file."""
    kwargs = {
        "config_file": world_yaml,
        "partial_obs": False,
        "actions_set": SmallActions,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return MultiGridEnv(**kwargs)


def _make_heuristic_human_policy(
    ref_env: MultiGridEnv,
    human_indices: list,
    beta: float = 1000.0,
) -> HeuristicPotentialPolicy:
    """Build a heuristic policy prior for the given environment."""
    path_calc = PathDistanceCalculator(
        grid_height=ref_env.height,
        grid_width=ref_env.width,
        world_model=ref_env,
    )
    return HeuristicPotentialPolicy(
        world_model=ref_env,
        human_agent_indices=human_indices,
        path_calculator=path_calc,
        beta=beta,
    )


def _wrap_human_policy(policy: HeuristicPotentialPolicy):
    """Wrap 3-arg HeuristicPotentialPolicy to 4-arg signature."""
    def _policy_prior(state, human_idx, goal, world_model):
        return policy(state, human_idx, goal)
    return _policy_prior


def _make_goal_sampler(ref_env: MultiGridEnv):
    """Create a goal sampler from the YAML-configured possible goals."""
    sampler = ref_env.possible_goal_sampler
    if sampler is None:
        raise RuntimeError(
            "The world model has no possible_goal_sampler. "
            "Ensure the YAML file includes a 'possible_goals' section."
        )
    def _sampler(state, human_idx):
        return sampler.sample(state, human_idx)
    return _sampler


# ======================================================================
# Tabular U_r env wrapper
# ======================================================================

class TabularRewardMultiGridEnv(MultiGridWorldModelEnv):
    """MultiGrid PPO env wrapper that uses tabular U_r from backward induction.

    Overrides ``_compute_u_r()`` to look up pre-computed values instead of
    using neural auxiliary networks.
    """

    def __init__(
        self,
        u_r_table: Dict[Any, float],
        default_u_r: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._u_r_table = u_r_table
        self._default_u_r = default_u_r

        # Override theoretical bound with empirical max |U_r| from table,
        # so rewards fill [-1, ≈0] instead of being crushed near zero.
        # Guard against all-zero (or numerically tiny) tables to avoid
        # division-by-zero when the base class normalises by _u_r_scale.
        if u_r_table:
            max_abs_u_r = max(abs(v) for v in u_r_table.values())
            if max_abs_u_r > 1e-8:
                self._u_r_scale = max_abs_u_r

    def _compute_u_r(self, state: Any) -> float:
        """Look up tabular U_r(state) from backward induction."""
        return self._u_r_table.get(state, self._default_u_r)


# ======================================================================
# Backward induction helpers
# ======================================================================

def compute_tabular_u_r(
    world_yaml: str,
    gamma_h: float,
    gamma_r: float,
    beta_h: float,
    beta_r: float,
    zeta: float,
    xi: float,
    eta: float,
    max_steps: int | None = None,
) -> tuple:
    """Run backward induction and return tabular U_r and V_r.

    Steps:
    1. Phase 1: compute tabular human policy prior
    2. Phase 2: compute robot policy with return_values=True
    3. Derive U_r(s) from V_h^e values for each state

    Returns ``(u_r_table, Vr_dict, robot_policy, human_policy_prior)``:
    - u_r_table: dict mapping state → float (U_r value, with X_h clamped)
    - Vr_dict: dict mapping state → float (V_r from backward induction, unclamped)
    - robot_policy: TabularRobotPolicy from backward induction
    - human_policy_prior: TabularHumanPolicyPrior from Phase 1
    """
    env = _create_world_model(world_yaml, max_steps=max_steps)
    env.reset()

    goal_generator = env.possible_goal_generator
    if goal_generator is None:
        raise RuntimeError("World has no possible_goal_generator")

    print(f"  Grid: {env.width}×{env.height}")
    print(f"  Goals: {len(goal_generator.goal_coords)} from config")
    print(f"  Humans: {env.human_agent_indices}, Robots: {env.robot_agent_indices}")

    # Phase 1 — tabular human policy prior
    print("  Running Phase 1 (human policy prior) …")
    human_policy_prior = compute_human_policy_prior(
        world_model=env,
        human_agent_indices=env.human_agent_indices,
        possible_goal_generator=goal_generator,
        believed_others_policy=None,
        beta_h=beta_h,
        gamma_h=gamma_h,
        level_fct=lambda state: state[0],
        use_disk_slicing=True,
    )

    # Phase 2 — tabular robot policy with V_r and V_h^e values
    print("  Running Phase 2 (robot policy + values) …")
    robot_policy, Vr_dict, Vh_dict = compute_robot_policy(
        world_model=env,
        human_agent_indices=env.human_agent_indices,
        robot_agent_indices=env.robot_agent_indices,
        possible_goal_generator=goal_generator,
        human_policy_prior=human_policy_prior,
        beta_r=beta_r,
        gamma_h=gamma_h,
        gamma_r=gamma_r,
        zeta=zeta,
        xi=xi,
        eta=eta,
        level_fct=lambda state: state[0],
        use_disk_slicing=True,
        return_values=True,
    )

    # Derive U_r(s) from Vh_dict for each state
    # U_r(s) = -(mean_h[X_h(s)^{-xi}])^eta
    # X_h(s) = mean_g[V_h^e(s, h, g)^zeta]
    human_indices = env.human_agent_indices
    u_r_table: Dict[Any, float] = {}
    _X_H_MIN = 1e-3

    for state, vh_by_agent in Vh_dict.items():
        x_h_vals = []
        for h_idx in human_indices:
            if h_idx not in vh_by_agent:
                continue
            goal_dict = vh_by_agent[h_idx]
            if not goal_dict:
                continue
            # X_h = mean over goals of V_h^e^zeta
            v_powers = [max(v, 0.0) ** zeta for v in goal_dict.values()]
            x_h = sum(v_powers) / len(v_powers) if v_powers else 0.0
            x_h = max(x_h, _X_H_MIN)
            x_h = min(x_h, 1.0)
            x_h_vals.append(x_h)

        if x_h_vals:
            y = float(np.mean([x ** (-xi) for x in x_h_vals]))
            u_r_table[state] = -(y ** eta)
        else:
            u_r_table[state] = 0.0

    print(f"  Tabular U_r computed for {len(u_r_table)} states")
    if u_r_table:
        vals = list(u_r_table.values())
        print(f"  U_r range: [{min(vals):.6f}, {max(vals):.6f}]")
        print(f"  U_r mean:  {np.mean(vals):.6f}")
        empirical_scale = max(abs(v) for v in vals)
        _X_H_MIN = 1e-3
        theoretical_scale = (_X_H_MIN ** (-xi)) ** eta
        print(f"  Empirical  max|U_r|:  {empirical_scale:.6f}  (used as scale)")
        print(f"  Theoretical bound:    {theoretical_scale:.6f}  (not used)")
        print(f"  U_r/scale range: [{min(vals)/empirical_scale:.6f}, {max(vals)/empirical_scale:.6f}]")

    # Also print V_r(root) for reference
    env.reset()
    root = env.get_state()
    if root in Vr_dict:
        print(f"  V_r(root): {Vr_dict[root]:.6f}")
    if root in u_r_table:
        print(f"  U_r(root): {u_r_table[root]:.6f}")

    # Print backward induction policy at root
    root_policy = robot_policy(root)
    if root_policy:
        action_names = ["still", "left", "right", "forward"]
        print("  π_r(root) from backward induction:")
        for profile, prob in sorted(root_policy.items()):
            action_str = action_names[profile[0]] if len(profile) == 1 else str(profile)
            print(f"    {action_str}: {prob:.4f}")

    env.close()
    return u_r_table, Vr_dict, robot_policy, human_policy_prior


def compute_clamped_target_vr(
    u_r_table: Dict[Any, float],
    u_r_scale: float,
    bi_robot_policy,
    human_policy_prior,
    env: MultiGridEnv,
    gamma_r: float,
) -> Dict[Any, float]:
    """Compute target V_r using clamped U_r and BI policy via backward DP.

    The BI V_r uses unclamped U_r internally (no X_h floor), while PPO
    trains on clamped U_r / u_r_scale.  This function computes the V_r
    that a perfect PPO critic should learn, by replaying the BI robot
    policy and Phase 1 human policy with the clamped reward signal.
    """
    from collections import defaultdict

    human_indices = list(env.human_agent_indices)
    robot_indices = list(env.robot_agent_indices)
    num_agents = len(env.agents)

    states_by_t: Dict[int, list] = defaultdict(list)
    for s in u_r_table:
        t = s[0] if isinstance(s, tuple) else 0
        states_by_t[t].append(s)

    max_t = max(states_by_t.keys())
    target_vr: Dict[Any, float] = {}

    # Terminal states: V_r = 0
    for s in states_by_t.get(max_t, []):
        target_vr[s] = 0.0

    # Backward pass
    for t in range(max_t - 1, -1, -1):
        for s in states_by_t.get(t, []):
            r = u_r_table.get(s, 0.0) / u_r_scale

            pi_r = bi_robot_policy(s)
            if not pi_r:
                target_vr[s] = r
                continue

            ev = 0.0
            for action_profile, robot_prob in pi_r.items():
                if robot_prob < 1e-10:
                    continue
                for h_prob, h_actions in human_policy_prior.profile_distribution(s):
                    if h_prob < 1e-10:
                        continue
                    actions = [0] * num_agents
                    for i, ri in enumerate(robot_indices):
                        actions[ri] = action_profile[i]
                    for i, hi in enumerate(human_indices):
                        actions[hi] = h_actions[i]
                    env.set_state(s)
                    transitions = env.transition_probabilities(s, actions)
                    if transitions:
                        for prob, next_s in transitions:
                            if prob > 0:
                                ev += robot_prob * h_prob * prob * target_vr.get(next_s, 0.0)

            target_vr[s] = r + gamma_r * ev

    return target_vr


# ======================================================================
# Rendering constants
# ======================================================================

RENDER_TILE_SIZE = 96
ANNOTATION_PANEL_WIDTH = 300
ANNOTATION_FONT_SIZE = 12
MOVIE_FPS = 2

SINGLE_ACTION_NAMES = ["still", "left", "right", "forward"]


def _get_joint_action_names(num_robots: int, num_actions: int = 4) -> List[str]:
    names = SINGLE_ACTION_NAMES[:num_actions]
    if num_robots == 1:
        return list(names)
    combos = list(itertools.product(names, repeat=num_robots))
    return [", ".join(c) for c in combos]


# ======================================================================
# Rollout with trained PPO policy
# ======================================================================


def run_ppo_rollout(
    env: MultiGridEnv,
    actor_critic,
    state_encoder,
    human_policy: HeuristicPotentialPolicy,
    goal_sampler,
    human_indices: List[int],
    robot_indices: List[int],
    device: str = "cpu",
) -> int:
    """Run a single rollout using the trained PPO actor-critic."""
    num_actions = env.action_space.n
    joint_action_names = _get_joint_action_names(len(robot_indices), num_actions)
    max_name_len = max(len(n) for n in joint_action_names)
    actor_critic.eval()

    env.reset()
    state = env.get_state()

    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler(state, h)
        human_goals[h] = goal

    def _state_to_obs(s):
        # Resolve encoder device robustly: try parameters, then buffers, then CPU.
        first_param = next(state_encoder.parameters(), None)
        if first_param is not None:
            encoder_device = first_param.device
        else:
            first_buffer = next(state_encoder.buffers(), None)
            if first_buffer is not None:
                encoder_device = first_buffer.device
            else:
                encoder_device = torch.device("cpu")
        with torch.no_grad():
            tensors = state_encoder.tensorize_state(s, env, device=encoder_device)
            features = state_encoder(*tensors)
        return features.squeeze(0)

    def _get_policy(obs_tensor):
        with torch.no_grad():
            logits, value = actor_critic(obs_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs, value.squeeze().item()

    def _select_action(probs):
        return torch.argmax(probs).item()

    def _annotation_text(s, selected_action=None):
        obs = _state_to_obs(s)
        probs, v_r = _get_policy(obs)
        lines = [f"V_r: {v_r:.4f}", ""]
        lines.append("π_r probs:")
        for i, p in enumerate(probs.cpu().tolist()):
            name = joint_action_names[i] if i < len(joint_action_names) else f"a{i}"
            marker = ">" if selected_action is not None and i == selected_action else " "
            lines.append(f"{marker}{name:>{max_name_len}}: {p:.3f}")
        return lines

    obs0 = _state_to_obs(state)
    probs0, _ = _get_policy(obs0)
    action0 = _select_action(probs0)
    env.render(
        mode="rgb_array",
        highlight=False,
        tile_size=RENDER_TILE_SIZE,
        annotation_text=_annotation_text(state, action0),
        annotation_panel_width=ANNOTATION_PANEL_WIDTH,
        annotation_font_size=ANNOTATION_FONT_SIZE,
        goal_overlays=human_goals,
    )

    steps_taken = 0
    for _step in range(env.max_steps):
        state = env.get_state()
        actions = [0] * len(env.agents)

        for h in human_indices:
            actions[h] = human_policy.sample(state, h, human_goals[h])

        obs = _state_to_obs(state)
        probs, _ = _get_policy(obs)
        robot_action = _select_action(probs)

        if len(robot_indices) == 1:
            actions[robot_indices[0]] = robot_action
        else:
            remaining = robot_action
            for r_idx in robot_indices:
                actions[r_idx] = remaining % num_actions
                remaining //= num_actions

        _, _, done, _ = env.step(actions)
        steps_taken += 1

        new_state = env.get_state()
        new_obs = _state_to_obs(new_state)
        new_probs, _ = _get_policy(new_obs)
        new_action = _select_action(new_probs)
        env.render(
            mode="rgb_array",
            highlight=False,
            tile_size=RENDER_TILE_SIZE,
            annotation_text=_annotation_text(new_state, new_action),
            annotation_panel_width=ANNOTATION_PANEL_WIDTH,
            annotation_font_size=ANNOTATION_FONT_SIZE,
            goal_overlays=human_goals,
        )

        if done:
            break

    return steps_taken


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 PPO with tabular U_r from backward induction (diagnostic)"
    )
    parser.add_argument("--world", type=str, default="trivial.yaml",
                        help='World YAML relative to multigrid_worlds/ (default: trivial.yaml)')
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of PPO iterations (default: 100)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps per episode (default: from world YAML)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of vectorised environments (default: 4)")
    parser.add_argument("--feature-dim", type=int, default=64,
                        help="State encoder feature dimension (default: 64)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Actor-critic hidden dimension (default: 64)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument("--rollouts", type=int, default=10,
                        help="Number of rollouts for the output movie (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output movie")
    parser.add_argument("--movie-fps", type=int, default=MOVIE_FPS,
                        help=f"Movie frames per second (default: {MOVIE_FPS})")
    # Theory parameters (must match between backward induction and PPO)
    parser.add_argument("--gamma-r", type=float, default=0.99)
    parser.add_argument("--gamma-h", type=float, default=0.99)
    parser.add_argument("--beta-h", type=float, default=1000.0,
                        help="Boltzmann temperature for Phase 1 human policy (default: 1000)")
    parser.add_argument("--beta-r", type=float, default=100.0,
                        help="Concentration for backward induction robot policy (default: 100)")
    parser.add_argument("--zeta", type=float, default=2.0)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.1)
    args = parser.parse_args()

    world_yaml = _resolve_world_path(args.world)

    # ==================================================================
    # 1. Run backward induction to get tabular U_r
    # ==================================================================
    print("=" * 60)
    print("Step 1: Computing tabular U_r via backward induction")
    print("=" * 60)
    u_r_table, Vr_dict, bi_robot_policy, human_policy_prior = compute_tabular_u_r(
        world_yaml=world_yaml,
        gamma_h=args.gamma_h,
        gamma_r=args.gamma_r,
        beta_h=args.beta_h,
        beta_r=args.beta_r,
        zeta=args.zeta,
        xi=args.xi,
        eta=args.eta,
        max_steps=args.steps,
    )

    # ==================================================================
    # 2. Create reference environment
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: Setting up PPO with tabular U_r rewards")
    print("=" * 60)
    ref_env = _create_world_model(world_yaml, max_steps=args.steps)
    ref_env.reset()

    num_actions = ref_env.action_space.n
    human_indices = ref_env.human_agent_indices
    robot_indices = ref_env.robot_agent_indices

    print(f"World: {world_yaml}")
    print(f"Grid : {ref_env.width}×{ref_env.height}")
    print(f"Actions: {num_actions}")
    print(f"Human agents: {human_indices}")
    print(f"Robot agents: {robot_indices}")
    print(f"Max steps: {ref_env.max_steps}")

    # ==================================================================
    # 3. PPO Configuration — NO warmup, NO auxiliary training
    # ==================================================================
    cfg = PPOPhase2Config(
        # Theory
        gamma_r=args.gamma_r,
        gamma_h=args.gamma_h,
        zeta=args.zeta,
        xi=args.xi,
        eta=args.eta,
        # PPO
        num_actions=num_actions,
        num_robots=len(robot_indices),
        hidden_dim=args.hidden_dim,
        ppo_rollout_length=64,
        ppo_num_minibatches=4,
        ppo_update_epochs=4,
        num_envs=args.num_envs,
        num_ppo_iterations=args.iters,
        lr_ppo=3e-4,
        # Auxiliary — disabled (we don't need aux networks for reward)
        aux_training_steps_per_iteration=0,
        aux_buffer_size=100,
        batch_size=64,
        reward_freeze_interval=999_999,
        # No warmup — tabular U_r is ready immediately
        warmup_v_h_e_steps=0,
        warmup_x_h_steps=0,
        warmup_u_r_steps=0,
        # Environment
        steps_per_episode=ref_env.max_steps,
        # Runtime
        device=args.device,
        seed=42,
    )

    # ==================================================================
    # 4. Create networks (actor-critic only, no aux needed for reward)
    # ==================================================================
    actor_critic, aux_nets, state_encoder = create_multigrid_ppo_networks(
        env=ref_env,
        config=cfg,
        feature_dim=args.feature_dim,
        use_x_h=False,
        use_u_r=False,
        device=args.device,
    )
    print(f"State encoder feature_dim: {state_encoder.feature_dim}")
    print(f"Actor-critic joint actions: {actor_critic.num_joint_actions}")

    # ==================================================================
    # 5. Build trainer
    # ==================================================================
    trainer = PPOPhase2Trainer(
        actor_critic=actor_critic,
        auxiliary_networks=aux_nets,
        config=cfg,
        device=args.device,
    )

    # ==================================================================
    # 6. Define env_creator with tabular U_r
    # ==================================================================
    shared_encoder = state_encoder

    # Compute default U_r for unseen states (use the worst/most negative)
    if u_r_table:
        default_u_r = min(u_r_table.values())
    else:
        default_u_r = 0.0
    print(f"Default U_r for unseen states: {default_u_r:.6f}")

    def env_creator():
        wm = _create_world_model(world_yaml, max_steps=args.steps)
        wm.reset()
        hp = _make_heuristic_human_policy(wm, wm.human_agent_indices)
        hp_fn = _wrap_human_policy(hp)
        gs_fn = _make_goal_sampler(wm)
        return TabularRewardMultiGridEnv(
            u_r_table=u_r_table,
            default_u_r=default_u_r,
            world_model=wm,
            human_policy_prior=hp_fn,
            goal_sampler=gs_fn,
            human_agent_indices=wm.human_agent_indices,
            robot_agent_indices=wm.robot_agent_indices,
            config=cfg,
            state_encoder=shared_encoder,
        )

    # ==================================================================
    # 7. Train
    # ==================================================================
    print(f"\nStarting PPO training for {args.iters} iterations with tabular U_r …")
    metrics = trainer.train(env_creator, num_iterations=args.iters)

    # ==================================================================
    # 8. Report results
    # ==================================================================
    if metrics:
        last = metrics[-1]
        print(f"\nTraining complete ({len(metrics)} iterations).")
        print(f"  Final policy_loss      : {last.get('policy_loss', 'N/A')}")
        print(f"  Final value_loss       : {last.get('value_loss', 'N/A')}")
        print(f"  Final entropy          : {last.get('entropy', 'N/A')}")
        print(f"  Final approx_kl        : {last.get('approx_kl', 'N/A')}")
        print(f"  Final clipfrac         : {last.get('clipfrac', 'N/A')}")
        print(f"  Final explained_var    : {last.get('explained_variance', 'N/A')}")
        print(f"  Global env steps       : {trainer.global_env_step}")

        # Print metrics trajectory at key iterations
        checkpoints = [0, len(metrics)//4, len(metrics)//2, 3*len(metrics)//4, len(metrics)-1]
        checkpoints = sorted(set(max(0, min(c, len(metrics)-1)) for c in checkpoints))
        print("\n  Metrics trajectory:")
        print(f"  {'iter':>6s}  {'ploss':>8s}  {'vloss':>8s}  {'entropy':>8s}  {'kl':>8s}  {'clip':>8s}  {'expl_var':>8s}")
        for c in checkpoints:
            m = metrics[c]
            print(f"  {m.get('iteration', c):6.0f}  {m.get('policy_loss', 0):8.5f}  "
                  f"{m.get('value_loss', 0):8.5f}  {m.get('entropy', 0):8.5f}  "
                  f"{m.get('approx_kl', 0):8.5f}  {m.get('clipfrac', 0):8.5f}  "
                  f"{m.get('explained_variance', 0):8.5f}")
    else:
        print("No metrics returned (training may have been too short).")

    # Diagnostics: PPO V_r(root) vs backward induction V_r(root)
    diag_env = _create_world_model(world_yaml, max_steps=args.steps)
    diag_env.reset()
    root_state = diag_env.get_state()

    actor_critic.eval()
    with torch.no_grad():
        # Resolve device safely: prefer parameters, then buffers, then CPU.
        enc_param = next(state_encoder.parameters(), None)
        if enc_param is not None:
            enc_device = enc_param.device
        else:
            enc_buffer = next(state_encoder.buffers(), None)
            enc_device = enc_buffer.device if enc_buffer is not None else torch.device("cpu")
        tensors = state_encoder.tensorize_state(root_state, diag_env, device=enc_device)
        features = state_encoder(*tensors)
        logits, value = actor_critic(features)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    # U_r scale factor — empirical max |U_r| from tabular data.
    u_r_scale = max(abs(v) for v in u_r_table.values()) if u_r_table else 1.0

    print("\n--- Diagnostics at root state ---")
    bi_vr_root = Vr_dict.get(root_state)
    print(f"  PPO V_r(root):           {value.item():.6f}")
    print(f"  BI  V_r(root) (raw):     {bi_vr_root}" if bi_vr_root is not None else "  BI  V_r(root) (raw):     N/A")
    print(f"  Tabular U_r(root):       {u_r_table.get(root_state, 'N/A')}")
    print(f"  U_r scale:               {u_r_scale:.6f}")
    print(f"  PPO π_r(root):    {dict(zip(SINGLE_ACTION_NAMES, [f'{p:.4f}' for p in probs.cpu().tolist()]))}")
    bi_root = bi_robot_policy(root_state)
    if bi_root:
        print(f"  BI  π_r(root):    {dict((SINGLE_ACTION_NAMES[p[0]], f'{v:.4f}') for p, v in bi_root.items())}")

    # ==================================================================
    # 8b. Compute clamped target V_r via backward DP
    # ==================================================================
    # BI V_r uses unclamped U_r internally (X_h not floored at 1e-3).
    # PPO trains on clamped U_r/u_r_scale, so the correct V_r target
    # is a fresh backward DP using the clamped u_r_table values.
    print("\nComputing clamped target V_r via backward DP …")
    cmp_env = _create_world_model(world_yaml, max_steps=args.steps)
    cmp_env.reset()
    clamped_vr = compute_clamped_target_vr(
        u_r_table=u_r_table,
        u_r_scale=u_r_scale,
        bi_robot_policy=bi_robot_policy,
        human_policy_prior=human_policy_prior,
        env=cmp_env,
        gamma_r=args.gamma_r,
    )
    clamped_vr_root = clamped_vr.get(root_state)
    print(f"  Clamped target V_r(root): {clamped_vr_root:.6f}" if clamped_vr_root is not None else "  Clamped target V_r(root): N/A")

    # ==================================================================
    # 8c. Compare V_r across ALL states (clamped target vs PPO)
    # ==================================================================
    print("\n" + "=" * 60)
    print("V_r comparison: clamped target (backward DP) vs PPO (all DAG states)")
    print("=" * 60)

    target_vals = []
    ppo_vals = []
    errors = []
    by_timestep: Dict[int, List[float]] = {}  # timestep → list of absolute errors

    actor_critic.eval()
    enc_device = next(state_encoder.parameters()).device

    for state, vr_target in clamped_vr.items():
        # Compute PPO V_r for this state
        with torch.no_grad():
            cmp_env.set_state(state)
            tensors = state_encoder.tensorize_state(state, cmp_env, device=enc_device)
            features = state_encoder(*tensors)
            _, v_ppo = actor_critic(features)
            v_ppo_f = v_ppo.squeeze().item()

        target_vals.append(vr_target)
        ppo_vals.append(v_ppo_f)
        err = v_ppo_f - vr_target
        errors.append(err)

        t = state[0] if isinstance(state, tuple) else 0
        by_timestep.setdefault(t, []).append(abs(err))

    target_arr = np.array(target_vals)
    ppo_arr = np.array(ppo_vals)
    err_arr = np.array(errors)

    print(f"  States compared:  {len(target_vals)}")
    print(f"  Target V_r range: [{target_arr.min():.6f}, {target_arr.max():.6f}]")
    print(f"  PPO V_r range:    [{ppo_arr.min():.6f}, {ppo_arr.max():.6f}]")
    print(f"  Mean error:       {err_arr.mean():.6f}")
    print(f"  Mean |error|:     {np.abs(err_arr).mean():.6f}")
    print(f"  RMSE:             {np.sqrt((err_arr ** 2).mean()):.6f}")
    print(f"  Max |error|:      {np.abs(err_arr).max():.6f}")

    vr_range = target_arr.max() - target_arr.min()
    if vr_range > 1e-10:
        print(f"  NRMSE (range):    {np.sqrt((err_arr ** 2).mean()) / vr_range:.4f}")

    if len(target_vals) > 1 and target_arr.std() > 1e-10 and ppo_arr.std() > 1e-10:
        corr = np.corrcoef(target_arr, ppo_arr)[0, 1]
        print(f"  Pearson corr:     {corr:.4f}")

    if by_timestep:
        print("\n  Per-timestep mean |error|:")
        for t in sorted(by_timestep):
            errs_t = by_timestep[t]
            print(f"    t={t}: mean|err|={np.mean(errs_t):.6f}  "
                  f"(n={len(errs_t)} states)")

    if len(errors) > 0:
        states_list = list(clamped_vr.keys())
        abs_errors = np.abs(err_arr)
        worst_idx = np.argsort(abs_errors)[-min(5, len(abs_errors)):]
        print("\n  Worst 5 states (largest |error|):")
        for idx in reversed(worst_idx):
            s = states_list[idx]
            print(f"    state t={s[0] if isinstance(s, tuple) else '?'}: "
                  f"target={target_vals[idx]:.6f}  PPO={ppo_vals[idx]:.6f}  "
                  f"err={errors[idx]:+.6f}")

    # ==================================================================
    # 8d. Policy comparison across all states
    # ==================================================================
    print("\n" + "=" * 60)
    print("Policy comparison: BI π_r vs PPO π_r (all non-terminal states)")
    print("=" * 60)

    n_agree = 0
    n_compared = 0
    kl_divs = []

    for state in clamped_vr:
        pi_r = bi_robot_policy(state)
        if not pi_r:  # terminal
            continue

        # BI policy as probability vector over single actions
        num_a = ref_env.action_space.n
        bi_probs_vec = np.zeros(num_a)
        for profile, p in pi_r.items():
            idx = profile[0] if len(profile) == 1 else sum(profile[i] * num_a**i for i in range(len(profile)))
            bi_probs_vec[idx] = p

        # PPO policy
        with torch.no_grad():
            cmp_env.set_state(state)
            tensors = state_encoder.tensorize_state(state, cmp_env, device=enc_device)
            features = state_encoder(*tensors)
            logits, _ = actor_critic(features)
            ppo_probs_vec = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if np.argmax(bi_probs_vec) == np.argmax(ppo_probs_vec):
            n_agree += 1
        n_compared += 1

        # KL(BI || PPO) with smoothing for numerical stability
        eps = 1e-8
        ppo_smooth = np.clip(ppo_probs_vec, eps, 1.0)
        bi_smooth = np.clip(bi_probs_vec, eps, 1.0)
        kl = float(np.sum(bi_smooth * np.log(bi_smooth / ppo_smooth)))
        kl_divs.append(kl)

    if n_compared > 0:
        print(f"  States compared:      {n_compared}")
        print(f"  Action agreement:     {n_agree}/{n_compared} ({100*n_agree/n_compared:.1f}%)")
        kl_arr = np.array(kl_divs)
        print(f"  Mean KL(BI||PPO):     {kl_arr.mean():.4f}")
        print(f"  Median KL(BI||PPO):   {np.median(kl_arr):.4f}")
        print(f"  Max KL(BI||PPO):      {kl_arr.max():.4f}")

    cmp_env.close()

    # ==================================================================
    # 9. Generate rollout movie
    # ==================================================================
    num_rollouts = args.rollouts
    if args.output_dir:
        output_dir = args.output_dir
    else:
        world_name = args.world
        if world_name.endswith('.yaml'):
            world_name = world_name[:-5]
        elif world_name.endswith('.yml'):
            world_name = world_name[:-4]
        output_dir = os.path.join(
            _PROJECT_ROOT, "outputs", "phase2_ppo_tabular_reward", world_name
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_rollouts} rollouts with learned policy …")

    rollout_env = _create_world_model(world_yaml, max_steps=args.steps)
    rollout_env.reset()
    rollout_human_policy = _make_heuristic_human_policy(
        rollout_env, rollout_env.human_agent_indices
    )
    rollout_goal_sampler = _make_goal_sampler(rollout_env)

    rollout_env.start_video_recording()

    for rollout_idx in range(num_rollouts):
        run_ppo_rollout(
            env=rollout_env,
            actor_critic=actor_critic,
            state_encoder=state_encoder,
            human_policy=rollout_human_policy,
            goal_sampler=rollout_goal_sampler,
            human_indices=list(rollout_env.human_agent_indices),
            robot_indices=list(rollout_env.robot_agent_indices),
            device=args.device,
        )
        if (rollout_idx + 1) % 5 == 0 or rollout_idx == num_rollouts - 1:
            print(
                f"  Completed {rollout_idx + 1}/{num_rollouts} rollouts "
                f"({len(rollout_env._video_frames)} total frames)"
            )

    movie_path = os.path.join(output_dir, "phase2_ppo_tabular_reward.mp4")
    if os.path.exists(movie_path):
        os.remove(movie_path)
    rollout_env.save_video(movie_path, fps=args.movie_fps)

    print(f"\n✓ Movie saved to: {os.path.abspath(movie_path)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 PPO Robot Policy Demo.

Demonstrates Phase 2 PPO training on a MultiGrid world specified via
the ``--world`` argument (default: ``trivial.yaml``).

Human behaviour:
    Humans use the existing ``HeuristicPotentialPolicy`` (deterministic
    goal-directed policy based on shortest-path potentials) instead of a
    uniform random prior.  This makes the demo more realistic: humans
    actively pursue goals, so V_h^e training has a meaningful signal.

Output:
    After training, generates a movie of rollouts with the learned policy,
    showing the robot's action probabilities and value estimates.

Usage:
    # Inside Docker container (make shell):
    python examples/phase2/phase2_ppo_robot_policy.py

    # Use a specific world (default: trivial.yaml):
    python examples/phase2/phase2_ppo_robot_policy.py --world jobst_challenges/asymmetric_freeing_simple

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_robot_policy.py

    # Quick smoke test (2 iterations, 2 rollouts):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_robot_policy.py --iters 2 --rollouts 2

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# Path setup (allows running the script directly without ``pip install -e``)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)
for _subdir in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    _path = os.path.join(_PROJECT_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# EMPO / MultiGrid imports
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
    """Resolve a world specifier to an absolute YAML path.

    ``world`` can be:
    - A bare filename like ``trivial.yaml``
    - A relative path like ``jobst_challenges/asymmetric_freeing_simple``
    - An absolute path

    A ``.yaml`` extension is appended automatically when missing.
    The file is looked up relative to the ``multigrid_worlds/`` directory.
    """
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
    """Wrap the 3-arg ``HeuristicPotentialPolicy.__call__`` to the 4-arg
    signature expected by ``EMPOWorldModelEnv``.

    Phase 2 PPO env wrapper calls:
        ``human_policy_prior(state, h_idx, goal, world_model)``
    but ``HeuristicPotentialPolicy.__call__`` takes:
        ``(state, human_agent_index, possible_goal)``
    """

    def _policy_prior(state, human_idx, goal, world_model):
        return policy(state, human_idx, goal)

    return _policy_prior


def _make_goal_sampler(ref_env: MultiGridEnv):
    """Create a goal sampler from the YAML-configured possible goals.

    Returns a callable ``(state, human_idx) → (goal, weight)`` that
    wraps the environment's built-in ``possible_goal_sampler``.
    """
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
# Rendering constants
# ======================================================================

RENDER_TILE_SIZE = 96
ANNOTATION_PANEL_WIDTH = 300
ANNOTATION_FONT_SIZE = 12
MOVIE_FPS = 2

# Action names for SmallActions
SINGLE_ACTION_NAMES = ["still", "left", "right", "forward"]


def _get_joint_action_names(num_robots: int, num_actions: int = 4) -> List[str]:
    """Generate joint action names for ``num_robots`` robots."""
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
    """Run a single rollout using the trained PPO actor-critic.

    Uses ``env``'s built-in video recording — frames are captured
    automatically via ``env.render(mode='rgb_array', ...)``.

    Returns the number of environment steps taken.
    """
    num_actions = env.action_space.n
    joint_action_names = _get_joint_action_names(len(robot_indices), num_actions)
    max_name_len = max(len(n) for n in joint_action_names)
    actor_critic.eval()

    env.reset()
    state = env.get_state()

    # Sample goals for each human
    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler(state, h)
        human_goals[h] = goal

    # ----- helpers -----

    def _state_to_obs(s):
        """Encode world-model state to a flat observation tensor."""
        # Resolve encoder device defensively: parameters → buffers → CPU
        params = list(state_encoder.parameters())
        if params:
            encoder_device = params[0].device
        else:
            buffers = list(state_encoder.buffers())
            if buffers:
                encoder_device = buffers[0].device
            else:
                encoder_device = torch.device("cpu")
        with torch.no_grad():
            tensors = state_encoder.tensorize_state(s, env, device=encoder_device)
            features = state_encoder(*tensors)
        return features.squeeze(0)  # (feature_dim,)

    def _get_policy(obs_tensor):
        """Return ``(probs, value)`` from the actor-critic."""
        with torch.no_grad():
            logits, value = actor_critic(obs_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs, value.squeeze().item()

    def _select_action(probs):
        """Greedy action selection for rollout visualisation."""
        return torch.argmax(probs).item()

    def _annotation_text(s, selected_action=None):
        """Build annotation lines for the current state."""
        obs = _state_to_obs(s)
        probs, v_r = _get_policy(obs)
        lines = [f"V_r: {v_r:.4f}", ""]
        lines.append("π_r probs:")
        for i, p in enumerate(probs.cpu().tolist()):
            name = joint_action_names[i] if i < len(joint_action_names) else f"a{i}"
            marker = (
                ">" if selected_action is not None and i == selected_action else " "
            )
            lines.append(f"{marker}{name:>{max_name_len}}: {p:.3f}")
        return lines

    # ----- initial frame -----
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

    # ----- step loop -----
    steps_taken = 0
    for _step in range(env.max_steps):
        state = env.get_state()
        actions = [0] * len(env.agents)

        # Humans use heuristic policy
        for h in human_indices:
            actions[h] = human_policy.sample(state, h, human_goals[h])

        # Robot uses trained PPO policy
        obs = _state_to_obs(state)
        probs, _ = _get_policy(obs)
        robot_action = _select_action(probs)

        # Decode joint-action index to per-robot actions
        if len(robot_indices) == 1:
            actions[robot_indices[0]] = robot_action
        else:
            remaining = robot_action
            for r_idx in robot_indices:
                actions[r_idx] = remaining % num_actions
                remaining //= num_actions

        _, _, done, _ = env.step(actions)
        steps_taken += 1

        # Render frame with annotation for the *new* state
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
        description="Phase 2 PPO demo (heuristic humans)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of PPO iterations (default: 100)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps per episode (default: from world YAML)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of vectorised environments (default: 4)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="State encoder feature dimension (default: 64)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Actor-critic hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Number of rollouts for the output movie (default: 10)",
    )
    parser.add_argument(
        "--world",
        type=str,
        default="trivial.yaml",
        help='Path to world YAML file relative to multigrid_worlds/, '
             'e.g., "jobst_challenges/asymmetric_freeing_simple" '
             '(default: trivial.yaml)',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output movie (default: outputs/<script_name>/<world_name>/)",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=MOVIE_FPS,
        help=f"Movie frames per second (default: {MOVIE_FPS})",
    )
    # Theory parameters (must match between backward induction and PPO)
    parser.add_argument("--beta-h", type=float, default=1000.0,
                        help="Boltzmann temperature for Phase 1 human policy (default: 1000)")
    parser.add_argument("--beta-r", type=float, default=1e6,
                        help="Concentration for backward induction robot policy "
                             "(default: 1e6 ≈ argmax, matching converged PPO)")
    parser.add_argument("--u-r-scale", type=float, default=None,
                        help="Empirical max|U_r| for reward scaling (default: auto-calibrate "
                             "after warmup; set to avoid PufferLib clamp crushing rewards)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Total warmup gradient steps (V_h_e gets first half, "
                             "X_h joins at half; default: auto-scale with --iters)")
    parser.add_argument("--aux-steps", type=int, default=10,
                        help="Auxiliary gradient steps per PPO iteration (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for auxiliary network training (default: 128)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Create a reference environment
    # ------------------------------------------------------------------
    world_yaml = _resolve_world_path(args.world)
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

    # ------------------------------------------------------------------
    # 2. Build heuristic human policy and goal sampler
    # ------------------------------------------------------------------

    print("Human policy: HeuristicPotentialPolicy (β=1000)")
    print(
        f"Goal sampler: YAML-configured ({len(ref_env.possible_goal_generator.goal_coords)} goals)"
    )

    # ------------------------------------------------------------------
    # 3. Configuration
    # ------------------------------------------------------------------
    # Warmup: default to 20% of total PPO aux steps but at least 500.
    # X_h needs V_h_e to be reasonable before it can learn, so V_h_e
    # gets the first half of warmup and X_h trains during the second half.
    if args.warmup is not None:
        total_warmup = args.warmup
    else:
        total_warmup = max(500, args.iters * args.aux_steps // 5)
    warmup_v_h_e = total_warmup // 2
    warmup_x_h = total_warmup
    print(f"Warmup: {total_warmup} total steps "
          f"(V_h_e: 0-{warmup_v_h_e}, X_h: {warmup_v_h_e}-{warmup_x_h})")

    cfg = PPOPhase2Config(
        # Theory
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,
        xi=1.0,
        eta=1.1,
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
        # Auxiliary
        aux_training_steps_per_iteration=args.aux_steps,
        aux_buffer_size=10_000,
        batch_size=args.batch_size,
        reward_freeze_interval=5,
        # Warm-up: V_h_e trains alone first, then X_h joins
        warmup_v_h_e_steps=warmup_v_h_e,
        warmup_x_h_steps=warmup_x_h,
        warmup_u_r_steps=warmup_x_h,  # No U_r network (use_u_r=False)
        # Environment
        steps_per_episode=ref_env.max_steps,
        # Reward scaling
        u_r_scale=args.u_r_scale,
        # Runtime
        device=args.device,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 4. Create networks
    # ------------------------------------------------------------------
    actor_critic, aux_nets, state_encoder = create_multigrid_ppo_networks(
        env=ref_env,
        config=cfg,
        feature_dim=args.feature_dim,
        use_x_h=True,
        use_u_r=False,  # Compute U_r from X_h formula, not a separate network
        device=args.device,
    )
    print(f"State encoder feature_dim: {state_encoder.feature_dim}")
    print(f"Actor-critic joint actions: {actor_critic.num_joint_actions}")

    # ------------------------------------------------------------------
    # 5. Build the trainer
    # ------------------------------------------------------------------
    trainer = PPOPhase2Trainer(
        actor_critic=actor_critic,
        auxiliary_networks=aux_nets,
        config=cfg,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 6. Define env_creator
    # ------------------------------------------------------------------
    shared_encoder = state_encoder

    # Each vectorised env slot creates its own world model and
    # HeuristicPotentialPolicy instance (to avoid shared mutable state).
    def env_creator():
        wm = _create_world_model(world_yaml, max_steps=args.steps)
        wm.reset()
        hp = _make_heuristic_human_policy(wm, wm.human_agent_indices)
        hp_fn = _wrap_human_policy(hp)
        gs_fn = _make_goal_sampler(wm)
        return MultiGridWorldModelEnv(
            world_model=wm,
            human_policy_prior=hp_fn,
            goal_sampler=gs_fn,
            human_agent_indices=wm.human_agent_indices,
            robot_agent_indices=wm.robot_agent_indices,
            config=cfg,
            state_encoder=shared_encoder,
            # auxiliary_networks injected by trainer.train()
        )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print(f"\nStarting PPO training for {args.iters} iterations …")
    metrics = trainer.train(env_creator, num_iterations=args.iters)

    # ------------------------------------------------------------------
    # 8. Report results
    # ------------------------------------------------------------------
    if metrics:
        last = metrics[-1]
        print(f"\nTraining complete ({len(metrics)} iterations).")
        print(f"  Final policy_loss : {last.get('policy_loss', 'N/A')}")
        print(f"  Final value_loss  : {last.get('value_loss', 'N/A')}")
        print(f"  Final v_h_e_loss  : {last.get('v_h_e_loss', 'N/A')}")
        print(f"  Final x_h_loss    : {last.get('x_h_loss', 'N/A')}")
        print(f"  Final u_r_loss    : {last.get('u_r_loss', 'N/A')}")
        print(f"  Global env steps  : {trainer.global_env_step}")
    else:
        print("No metrics returned (training may have been too short).")

    # ------------------------------------------------------------------
    # 8b. Comprehensive BI vs learned diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BI vs Learned Diagnostics: comparing across ALL states")
    print("=" * 70)

    # Run backward induction to get exact values
    # NOTE: BI computes V_h^e under the BI robot policy. With beta_r → ∞
    # (default 1e6), the BI policy is essentially argmax, matching what a
    # well-trained PPO policy converges to. This makes the comparison
    # meaningful for a converged PPO agent.
    print("\nRunning backward induction for exact V_h^e, X_h, U_r …")
    bi_env = _create_world_model(world_yaml, max_steps=args.steps)
    bi_env.reset()

    goal_generator = bi_env.possible_goal_generator
    if goal_generator is None:
        print("  WARNING: no possible_goal_generator — skipping BI comparison")
    else:
        bi_human_policy_prior = compute_human_policy_prior(
            world_model=bi_env,
            human_agent_indices=bi_env.human_agent_indices,
            possible_goal_generator=goal_generator,
            believed_others_policy=None,
            beta_h=args.beta_h,
            gamma_h=cfg.gamma_h,
            level_fct=lambda state: state[0],
            use_disk_slicing=True,
        )
        bi_robot_policy, bi_Vr_dict, bi_Vh_dict = compute_robot_policy(
            world_model=bi_env,
            human_agent_indices=bi_env.human_agent_indices,
            robot_agent_indices=bi_env.robot_agent_indices,
            possible_goal_generator=goal_generator,
            human_policy_prior=bi_human_policy_prior,
            beta_r=args.beta_r,
            gamma_h=cfg.gamma_h,
            gamma_r=cfg.gamma_r,
            zeta=cfg.zeta,
            xi=cfg.xi,
            eta=cfg.eta,
            level_fct=lambda state: state[0],
            use_disk_slicing=True,
            return_values=True,
        )

        # Derive BI X_h and U_r from Vh_dict
        _X_H_MIN = 1e-3
        bi_xh_dict = {}   # state → {h_idx: x_h}
        bi_ur_dict = {}    # state → u_r
        for state, vh_by_agent in bi_Vh_dict.items():
            xh_per_human = {}
            for h_idx in human_indices:
                if h_idx not in vh_by_agent:
                    continue
                goal_dict = vh_by_agent[h_idx]
                if not goal_dict:
                    continue
                v_powers = [max(v, 0.0) ** cfg.zeta for v in goal_dict.values()]
                xh = sum(v_powers) / len(v_powers) if v_powers else 0.0
                xh = max(xh, _X_H_MIN)
                xh = min(xh, 1.0)
                xh_per_human[h_idx] = xh
            bi_xh_dict[state] = xh_per_human
            if xh_per_human:
                y = float(np.mean([x ** (-cfg.xi)
                                   for x in xh_per_human.values()]))
                bi_ur_dict[state] = -(y ** cfg.eta)
            else:
                bi_ur_dict[state] = 0.0

        # ----- Compare V_h^e across all states, humans, goals -----
        diag_env = _create_world_model(world_yaml, max_steps=args.steps)
        diag_env.reset()
        nets = aux_nets
        v_h_e_net = nets.v_h_e
        x_h_net = nets.x_h
        device = args.device
        goals = (goal_generator.goals
                 if hasattr(goal_generator, 'goals') else [])

        print(f"\n  States in BI: {len(bi_Vh_dict)}")
        print(f"  Goals: {len(goals)}")
        print(f"  Humans: {human_indices}")

        # V_h^e comparison
        if v_h_e_net is not None and goals:
            v_h_e_net.eval()
            vhe_errors = []  # (state, h, g, bi_val, nn_val)
            n_states_done = 0
            for state, vh_by_agent in bi_Vh_dict.items():
                for h_idx in human_indices:
                    if h_idx not in vh_by_agent:
                        continue
                    goal_dict = vh_by_agent[h_idx]
                    for goal, bi_v in goal_dict.items():
                        with torch.no_grad():
                            nn_v = v_h_e_net(state, diag_env, h_idx, goal,
                                             device)
                            nn_v = float(
                                v_h_e_net.apply_hard_clamp(nn_v).item())
                        vhe_errors.append(
                            (state, h_idx, goal, float(bi_v), nn_v))
                n_states_done += 1

            if vhe_errors:
                bi_vals = np.array([e[3] for e in vhe_errors])
                nn_vals = np.array([e[4] for e in vhe_errors])
                abs_err = np.abs(bi_vals - nn_vals)
                print("\n  --- V_h^e comparison ---")
                print(f"  Samples: {len(vhe_errors)}")
                print(f"  BI  range: [{bi_vals.min():.6f}, "
                      f"{bi_vals.max():.6f}]  mean={bi_vals.mean():.6f}")
                print(f"  NN  range: [{nn_vals.min():.6f}, "
                      f"{nn_vals.max():.6f}]  mean={nn_vals.mean():.6f}")
                print(f"  MAE: {abs_err.mean():.6f}  "
                      f"max: {abs_err.max():.6f}")
                if bi_vals.std() > 1e-8:
                    corr = np.corrcoef(bi_vals, nn_vals)[0, 1]
                    print(f"  Correlation: {corr:.6f}")
                # Show worst cases
                worst_idx = np.argsort(-abs_err)[:5]
                print("  Worst errors:")
                for idx in worst_idx:
                    s, h, g, bv, nv = vhe_errors[idx]
                    gl = getattr(g, 'target_pos', str(g))
                    print(f"    h={h} g={gl} t={s[0]}: "
                          f"BI={bv:.4f} NN={nv:.4f} err={abs(bv-nv):.4f}")

        # X_h comparison
        if x_h_net is not None and bi_xh_dict:
            x_h_net.eval()
            xh_errors = []  # (state, h_idx, bi_xh, nn_xh)
            for state, xh_by_h in bi_xh_dict.items():
                for h_idx, bi_xh in xh_by_h.items():
                    with torch.no_grad():
                        nn_xh = x_h_net(state, diag_env, h_idx, device)
                        nn_xh = float(
                            x_h_net.apply_hard_clamp(nn_xh).item())
                    xh_errors.append((state, h_idx, bi_xh, nn_xh))

            if xh_errors:
                bi_xh_arr = np.array([e[2] for e in xh_errors])
                nn_xh_arr = np.array([e[3] for e in xh_errors])
                abs_err = np.abs(bi_xh_arr - nn_xh_arr)
                print("\n  --- X_h comparison ---")
                print(f"  Samples: {len(xh_errors)}")
                print(f"  BI  range: [{bi_xh_arr.min():.6f}, "
                      f"{bi_xh_arr.max():.6f}]  mean={bi_xh_arr.mean():.6f}")
                print(f"  NN  range: [{nn_xh_arr.min():.6f}, "
                      f"{nn_xh_arr.max():.6f}]  mean={nn_xh_arr.mean():.6f}")
                print(f"  MAE: {abs_err.mean():.6f}  "
                      f"max: {abs_err.max():.6f}")
                if bi_xh_arr.std() > 1e-8:
                    corr = np.corrcoef(bi_xh_arr, nn_xh_arr)[0, 1]
                    print(f"  Correlation: {corr:.6f}")
                worst_idx = np.argsort(-abs_err)[:5]
                print("  Worst errors:")
                for idx in worst_idx:
                    s, h, bx, nx = xh_errors[idx]
                    print(f"    h={h} t={s[0]}: "
                          f"BI={bx:.4f} NN={nx:.4f} err={abs(bx-nx):.4f}")

        # U_r comparison (derived from X_h)
        if x_h_net is not None and bi_ur_dict:
            ur_errors = []  # (state, bi_ur, nn_ur)
            for state, bi_ur in bi_ur_dict.items():
                x_vals = []
                for h_idx in human_indices:
                    with torch.no_grad():
                        xv = x_h_net(state, diag_env, h_idx, device)
                        xv = float(x_h_net.apply_hard_clamp(xv).item())
                    x_vals.append(max(min(xv, 1.0), _X_H_MIN))
                if x_vals:
                    y = float(np.mean([x ** (-cfg.xi) for x in x_vals]))
                    nn_ur = -(y ** cfg.eta)
                else:
                    nn_ur = 0.0
                ur_errors.append((state, bi_ur, nn_ur))

            if ur_errors:
                bi_ur_arr = np.array([e[1] for e in ur_errors])
                nn_ur_arr = np.array([e[2] for e in ur_errors])
                abs_err = np.abs(bi_ur_arr - nn_ur_arr)
                print("\n  --- U_r comparison (derived from X_h) ---")
                print(f"  Samples: {len(ur_errors)}")
                print(f"  BI  range: [{bi_ur_arr.min():.6f}, "
                      f"{bi_ur_arr.max():.6f}]  mean={bi_ur_arr.mean():.6f}")
                print(f"  NN  range: [{nn_ur_arr.min():.6f}, "
                      f"{nn_ur_arr.max():.6f}]  mean={nn_ur_arr.mean():.6f}")
                print(f"  MAE: {abs_err.mean():.6f}  "
                      f"max: {abs_err.max():.6f}")
                if bi_ur_arr.std() > 1e-8:
                    corr = np.corrcoef(bi_ur_arr, nn_ur_arr)[0, 1]
                    print(f"  Correlation: {corr:.6f}")

                # Show U_r scale diagnostic
                empirical_max_ur = max(abs(v) for v in bi_ur_arr)
                theoretical_scale = (_X_H_MIN ** (-cfg.xi)) ** cfg.eta
                print(f"\n  U_r scale diagnostic:")
                print(f"    Empirical max|U_r|:   {empirical_max_ur:.6f}")
                print(f"    Theoretical bound:    {theoretical_scale:.6f}")
                print(f"    Ratio (emp/theo):     "
                      f"{empirical_max_ur/theoretical_scale:.6f}")
                print(f"    Config u_r_scale:     {cfg.u_r_scale}")
                if cfg.u_r_scale is None:
                    print(f"    WARNING: u_r_scale=None → using theoretical "
                          f"bound {theoretical_scale:.1f}")
                    print(f"    This crushes rewards by factor "
                          f"{empirical_max_ur/theoretical_scale:.4f}")
                    print(f"    → Consider setting --u-r-scale or calling "
                          f"calibrate_reward_scale()")

                worst_idx = np.argsort(-abs_err)[:5]
                print("  Worst U_r errors:")
                for idx in worst_idx:
                    s, bu, nu = ur_errors[idx]
                    print(f"    t={s[0]}: "
                          f"BI={bu:.4f} NN={nu:.4f} err={abs(bu-nu):.4f}")

        # V_r comparison at root
        bi_env.reset()
        root_state = bi_env.get_state()
        actor_critic.eval()
        with torch.no_grad():
            first_param = next(state_encoder.parameters(), None)
            if first_param is not None:
                enc_device = first_param.device
            else:
                first_buf = next(state_encoder.buffers(), None)
                enc_device = (first_buf.device if first_buf is not None
                              else torch.device("cpu"))
            tensors = state_encoder.tensorize_state(
                root_state, diag_env, device=enc_device)
            features = state_encoder(*tensors)
            logits, value = actor_critic(features)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        bi_vr_root = bi_Vr_dict.get(root_state)
        print(f"\n  --- Root state summary ---")
        print(f"  PPO V_r(root):  {value.item():.6f}")
        print(f"  BI  V_r(root):  "
              f"{bi_vr_root:.6f}" if bi_vr_root is not None else "  N/A")
        print(f"  PPO π_r(root):  "
              f"{dict(zip(SINGLE_ACTION_NAMES, [f'{p:.4f}' for p in probs.cpu().tolist()]))}")
        bi_root_policy = bi_robot_policy(root_state)
        if bi_root_policy:
            action_names_map = ["still", "left", "right", "forward"]
            bi_probs = {}
            for profile, prob in bi_root_policy.items():
                a_name = (action_names_map[profile[0]]
                          if len(profile) == 1 else str(profile))
                bi_probs[a_name] = f"{prob:.4f}"
            print(f"  BI  π_r(root):  {bi_probs}")

        bi_env.close()
        diag_env.close()

    # ------------------------------------------------------------------
    # 9. Generate rollout movie
    # ------------------------------------------------------------------
    num_rollouts = args.rollouts
    # Derive world name for output directory: strip extension and replace
    # path separators so that e.g. "jobst_challenges/asymmetric_freeing_simple"
    # becomes "asymmetric_freeing_simple".
    world_basename = os.path.splitext(os.path.basename(args.world))[0]
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "outputs", script_name, world_basename
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_rollouts} rollouts with learned policy …")

    # Build a fresh env for rollouts (separate from training envs)
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

    movie_path = os.path.join(output_dir, f"{script_name}_{world_basename}.mp4")
    if os.path.exists(movie_path):
        os.remove(movie_path)
    rollout_env.save_video(movie_path, fps=args.movie_fps)

    print(f"\n✓ Movie saved to: {os.path.abspath(movie_path)}")


if __name__ == "__main__":
    main()

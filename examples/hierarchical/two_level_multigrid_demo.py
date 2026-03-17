#!/usr/bin/env python3
"""Two-level hierarchical MultiGrid demonstration.

Showcases the hierarchical planning pipeline implemented in ``empo.hierarchical``:

1. **TwoLevelMultigrid**: builds a macro-level abstraction (MacroGridEnv) from a
   MultiGridEnv by partitioning the walkable grid into rectangular macro-cells.
2. **MacroGoalGenerator / MacroHeuristicPolicy**: enumerates macro-level goals
   (reach-cell, proximity) and provides a Boltzmann-rational heuristic policy.
3. **compute_hierarchical_robot_policy**: solves Phase 2 backward induction at
   the macro level and returns a ``HierarchicalRobotPolicy`` that computes
   micro-level sub-problem policies on demand during rollouts.
4. **Rollout**: runs the hierarchical policy in the micro-level environment with
   optional video recording.

Environment
-----------
By default uses ``multigrid_worlds/hierarchical/two_room.yaml`` — a 7×4 grid
split into two tiny rooms by an internal wall with a single-cell passage:

::

    We We We We We We We
    We Ay .. We .. Ae We
    We We .. .. .. We We
    We We We We We We We

The human (yellow, agent 0) starts in the left room; the robot (grey, agent 1)
in the right room.  Rooms are kept very small (≤ 2 walkable cells each) so that
micro-level sub-problems are tractable for on-demand backward induction.

Usage
-----
::

    # Inside Docker (make shell):
    python examples/hierarchical/two_level_multigrid_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:multigrid_worlds \\
        python examples/hierarchical/two_level_multigrid_demo.py

    # Quick mode (fewer rollouts, no video):
    python examples/hierarchical/two_level_multigrid_demo.py --quick

    # Custom world:
    python examples/hierarchical/two_level_multigrid_demo.py \\
        --world copilot_challenges/rock_gateway --steps 20
"""

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'hierarchical_demo')

RENDER_TILE_SIZE = 64
ANNOTATION_PANEL_WIDTH = 300
ANNOTATION_FONT_SIZE = 11

# Full / quick defaults
FULL_NUM_ROLLOUTS = 5
FULL_MAX_STEPS = 10
QUICK_NUM_ROLLOUTS = 2
QUICK_MAX_STEPS = 8


# ---------------------------------------------------------------------------
# Helpers (specific types are imported in main(); use Any for annotations)
# ---------------------------------------------------------------------------

def macro_action_name(action: int) -> str:
    """Human-readable name for a macro action."""
    from empo.hierarchical.macro_grid_env import decode_macro_action
    kind, target = decode_macro_action(action)
    if kind == 'PASS':
        return 'PASS'
    return f'WALK({target})'


def print_macro_structure(hierarchy: Any) -> None:
    """Print a summary of the macro-level abstraction."""
    macro = hierarchy.macro_env
    partition = macro.partition

    print("Macro-level structure")
    print(f"  Cells: {partition.num_cells}")
    for i in range(partition.num_cells):
        positions = sorted(partition.cell_positions(i))
        print(f"    Cell {i}: {positions}")
    print("  Adjacency:")
    for cell_i, neighbours in sorted(partition.adjacency.items()):
        print(f"    {cell_i} → {sorted(neighbours)}")
    print(f"  Action space: {macro.action_space.n}  "
          f"(PASS + WALK to {macro.action_space.n - 1} cells)")
    print()


def print_macro_goals(
    gen: Any,
    macro_state: Any,
    human_idx: int,
) -> None:
    """Print the macro-level goals enumerated by the goal generator."""
    goals = list(gen.generate(macro_state, human_idx))
    print(f"Macro goals for human agent {human_idx}: {len(goals)}")
    for goal, weight in goals:
        print(f"  {goal}  (weight={weight})")
    print()


def print_macro_policy_summary(
    policy: Any,
    macro_state: Any,
    human_idx: int,
    gen: Any,
) -> None:
    """Print the heuristic macro-level human policy for the initial state."""
    # Marginal
    dist = policy(macro_state, human_idx)
    print("Macro heuristic human policy (initial state)")
    print("  Marginal distribution:")
    for a_idx, p in enumerate(dist):
        if p > 1e-6:
            print(f"    {macro_action_name(a_idx):>12s}: {p:.4f}")

    # First goal conditioned
    goals = list(gen.generate(macro_state, human_idx))
    if goals:
        first_goal, _ = goals[0]
        dist_g = policy(macro_state, human_idx, first_goal)
        print(f"  Conditioned on {first_goal}:")
        for a_idx, p in enumerate(dist_g):
            if p > 1e-6:
                print(f"    {macro_action_name(a_idx):>12s}: {p:.4f}")
    print()


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    micro_env: Any,
    hierarchy: Any,
    h_policy: Any,
    macro_gen: Any,
    *,
    max_steps: int = 30,
    record_video: bool = True,
    verbose: bool = True,
) -> int:
    """Run a single rollout using the hierarchical robot policy.

    The *robot* follows the hierarchical policy; the *human* takes
    uniformly random micro-level actions (this is a demo, not an
    exact simulation of rational human behaviour).

    Returns the number of steps taken.
    """
    micro_env.reset()
    h_policy.reset(hierarchy)

    human_indices = list(micro_env.human_agent_indices)
    robot_indices = list(micro_env.robot_agent_indices)

    if record_video:
        micro_env.start_video_recording()

    state = micro_env.get_state()

    def _annotation(state, robot_action=None):
        lines = []
        coarse = h_policy.current_coarse_action_profile
        if coarse is not None:
            acts = ', '.join(macro_action_name(a) for a in coarse)
            lines.append(f"Macro: {acts}")
        else:
            lines.append("Macro: (deciding)")
        level = "micro" if not h_policy.at_macro_level else "MACRO"
        lines.append(f"Level: {level}")
        if robot_action is not None:
            lines.append(f"Robot act: {robot_action}")
        return lines

    # Initial frame
    if record_video:
        annotation = _annotation(state)
        micro_env.render(
            mode='rgb_array', highlight=False,
            tile_size=RENDER_TILE_SIZE,
            annotation_text=annotation,
            annotation_panel_width=ANNOTATION_PANEL_WIDTH,
            annotation_font_size=ANNOTATION_FONT_SIZE,
        )

    steps = 0
    for step in range(max_steps):
        state = micro_env.get_state()

        # --- Robot action via hierarchical policy ---
        robot_action_profile = h_policy.sample(state)

        # --- Human action: uniform random at the micro level ---
        actions = [0] * len(micro_env.agents)
        for i, r_idx in enumerate(robot_indices):
            actions[r_idx] = robot_action_profile[i]

        for h_idx in human_indices:
            actions[h_idx] = np.random.randint(0, micro_env.action_space.n)

        # Step
        prev_state = state
        _, _, done, _ = micro_env.step(actions)
        steps += 1
        state = micro_env.get_state()

        # Notify policy of the transition
        h_policy.observe_transition(prev_state, tuple(actions), state)

        if verbose and step < 10:
            coarse = h_policy.current_coarse_action_profile
            coarse_str = (
                ', '.join(macro_action_name(a) for a in coarse)
                if coarse is not None else '(deciding)'
            )
            print(f"  step {step:2d}: robot={robot_action_profile}  "
                  f"macro={coarse_str}  at_macro={h_policy.at_macro_level}")

        if record_video:
            annotation = _annotation(state, robot_action_profile)
            micro_env.render(
                mode='rgb_array', highlight=False,
                tile_size=RENDER_TILE_SIZE,
                annotation_text=annotation,
                annotation_panel_width=ANNOTATION_PANEL_WIDTH,
                annotation_font_size=ANNOTATION_FONT_SIZE,
            )

        if done:
            break

    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-level hierarchical MultiGrid demo",
    )
    parser.add_argument(
        '--quick', '-q', action='store_true',
        help='Quick test mode (fewer rollouts, no video).',
    )
    parser.add_argument(
        '--world', type=str, default=None,
        help='Path to YAML world config (relative to multigrid_worlds/).',
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override max_steps per episode.',
    )
    parser.add_argument(
        '--rollouts', type=int, default=None,
        help='Number of rollouts to run.',
    )
    parser.add_argument(
        '--beta-r', type=float, default=5.0,
        help='Robot policy concentration parameter.',
    )
    parser.add_argument(
        '--beta-h', type=float, default=5.0,
        help='Human heuristic policy temperature.',
    )
    parser.add_argument(
        '--gamma-r', type=float, default=0.99,
        help='Robot discount factor.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed.',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for videos.',
    )
    parser.add_argument(
        '--no-video', action='store_true',
        help='Skip video recording.',
    )
    return parser.parse_args()


def main():
    # Gym-dependent imports (require gym compatibility patch applied first)
    from gym_multigrid.multigrid import MultiGridEnv
    from empo.hierarchical import (
        TwoLevelMultigrid,
        MacroGoalGenerator,
        MacroHeuristicPolicy,
        compute_hierarchical_robot_policy,
    )

    args = parse_args()

    # Resolve quick mode defaults
    quick = args.quick
    num_rollouts = args.rollouts or (QUICK_NUM_ROLLOUTS if quick else FULL_NUM_ROLLOUTS)
    max_steps = args.steps or (QUICK_MAX_STEPS if quick else FULL_MAX_STEPS)
    record_video = not args.no_video and not quick
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 70)
    print("Two-Level Hierarchical MultiGrid Demo")
    print("=" * 70)
    print()

    # ── 1. Create micro-level environment ──────────────────────────
    if args.world:
        world_path = args.world
        if not world_path.endswith('.yaml') and not world_path.endswith('.yml'):
            world_path += '.yaml'
        config_path = os.path.join(PROJECT_DIR, 'multigrid_worlds', world_path)
        print(f"Loading world: {world_path}")
    else:
        config_path = os.path.join(
            PROJECT_DIR, 'multigrid_worlds', 'hierarchical', 'two_room.yaml'
        )
        print("Loading default world: hierarchical/two_room.yaml")

    micro_env = MultiGridEnv(config_file=config_path)
    micro_env.reset()

    print(f"  Grid: {micro_env.width}×{micro_env.height}")
    print(f"  Agents: {len(micro_env.agents)}  "
          f"(humans={micro_env.human_agent_indices}, "
          f"robots={micro_env.robot_agent_indices})")
    for i, a in enumerate(micro_env.agents):
        role = "human" if i in micro_env.human_agent_indices else "robot"
        print(f"    Agent {i} ({role}): pos={tuple(a.pos)}, color={a.color}")
    print(f"  Actions: {micro_env.actions.__name__} ({micro_env.actions.available})")
    print(f"  Max steps: {max_steps}")
    print()

    # ── 2. Build two-level hierarchy ───────────────────────────────
    print("Building two-level hierarchy...")
    t0 = time.time()
    hierarchy = TwoLevelMultigrid(micro_env, seed=args.seed)
    print(f"  Done in {time.time() - t0:.3f}s")
    print()
    print_macro_structure(hierarchy)

    # ── 3. Macro-level goals and heuristic policy ──────────────────
    macro_env = hierarchy.macro_env
    macro_gen = MacroGoalGenerator(macro_env)
    macro_prior = MacroHeuristicPolicy(
        macro_env,
        human_agent_indices=list(macro_env.human_agent_indices),
        possible_goal_generator=macro_gen,
        beta=args.beta_h,
    )

    macro_state = macro_env.get_state()
    human_idx = list(macro_env.human_agent_indices)[0]
    print_macro_goals(macro_gen, macro_state, human_idx)
    print_macro_policy_summary(macro_prior, macro_state, human_idx, macro_gen)

    # ── 4. Compute hierarchical robot policy ───────────────────────
    print("Computing hierarchical robot policy (macro-level solve)...")
    t0 = time.time()
    h_policy = compute_hierarchical_robot_policy(
        hierarchy,
        human_agent_indices=list(macro_env.human_agent_indices),
        robot_agent_indices=list(macro_env.robot_agent_indices),
        possible_goal_generators=[macro_gen],
        human_policy_priors=[macro_prior],
        beta_r=args.beta_r,
        gamma_r=args.gamma_r,
        quiet=quick,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")
    print(f"  Macro V_r entries: {len(h_policy.macro_Vr)}")
    print()

    # ── 5. Run rollouts ────────────────────────────────────────────
    print(f"Running {num_rollouts} rollout(s), max_steps={max_steps}")
    if record_video:
        print(f"  Video recording: ON (output → {output_dir})")
    else:
        print("  Video recording: OFF")
    print()

    for r in range(num_rollouts):
        print(f"--- Rollout {r + 1}/{num_rollouts} ---")
        micro_env.reset()

        steps = run_rollout(
            micro_env, hierarchy, h_policy, macro_gen,
            max_steps=max_steps,
            record_video=record_video,
            verbose=True,
        )
        print(f"  → {steps} steps")

        if record_video:
            video_path = os.path.join(output_dir, f'rollout_{r}.mp4')
            if os.path.exists(video_path):
                os.remove(video_path)
            try:
                micro_env.save_video(video_path, fps=5)
                print(f"  → Video saved: {video_path}")
            except Exception as e:
                print(f"  → Video save failed: {e}")
        print()

    # ── Done ───────────────────────────────────────────────────────
    print("=" * 70)
    print("Demo completed!")
    if record_video:
        print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == '__main__':
    # Patch gym import for compatibility — done here rather than at module
    # level so that importing this file as a library doesn't mutate
    # sys.modules for other code.
    import gymnasium as gym
    sys.modules['gym'] = gym

    main()

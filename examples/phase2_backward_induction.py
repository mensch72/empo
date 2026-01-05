#!/usr/bin/env python3
"""
Phase 2 Robot Policy via Backward Induction Demo.

This script demonstrates computing the robot policy using exact backward induction
on the state DAG, rather than neural network training. This is computationally
exact but only feasible for small state spaces.

Uses the trivial world model from phase2_robot_policy_demo.py:
- 4x6 grid with 1 human, 1 robot
- Rock that can be pushed by the robot
- Human has goals to reach specific cells

The demo:
1. Computes human policy prior via backward induction
2. Computes robot policy via backward induction (Phase 2)
3. Records rollout movies with value annotations

Usage:
    python phase2_backward_induction.py           # Default 5 max steps
    python phase2_backward_induction.py --steps 3 # Shorter horizon
    python phase2_backward_induction.py --steps 7 # Longer horizon (slower)
    python phase2_backward_induction.py --parallel  # Use parallel computation
    python phase2_backward_induction.py --rollouts 20  # More rollouts

Theory Parameters (from EMPO paper):
    --beta_h    Inverse temperature for human policy (default: 10.0)
    --beta_r    Power-law concentration for robot policy (default: 100.0)
    --gamma_h   Discount factor for human values (default: 0.99)
    --gamma_r   Discount factor for robot values (default: 0.99)
    --zeta      Risk-aversion for goal achievement (default: 2.0)
    --xi        Inter-human inequality aversion (default: 1.0)
    --eta       Intertemporal inequality aversion (default: 1.1)

Output:
    - Movie of rollouts in outputs/phase2_backward_induction/rollouts.mp4
"""

import argparse
import itertools
import os
import random
import sys
import time
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions, SmallWorld,
    Rock
)
from empo.possible_goal import PossibleGoalGenerator, TabularGoalSampler
from empo.backward_induction import (
    compute_human_policy_prior, 
    compute_robot_policy,
    TabularHumanPolicyPrior,
    TabularRobotPolicy
)
from empo.multigrid import ReachCellGoal


# =============================================================================
# Configuration
# =============================================================================

# Rendering configuration for annotations
RENDER_TILE_SIZE = 96
ANNOTATION_PANEL_WIDTH = 380
ANNOTATION_FONT_SIZE = 12

# Action names for SmallActions (single agent)
SINGLE_ACTION_NAMES = ['still', 'left', 'right', 'forward']


# =============================================================================
# Environment Definition (Trivial World Model)
# =============================================================================

def create_trivial_env(max_steps: int = 5) -> MultiGridEnv:
    """
    Create the trivial environment from phase2_robot_policy_demo.py.
    
    Layout:
        We We We We We We
        We Ae Ro .. .. We
        We We Ay We We We
        We We We We We We
    
    Where:
        Ae = Grey agent (robot, agent 0)
        Ay = Yellow agent (human, agent 1)
        Ro = Rock (can be pushed by robot)
        We = Wall
        .. = Empty cell
    
    Note: Agent ordering is different here - robot is agent 0, human is agent 1.
    """
    GRID_MAP = """
    We We We We We We
    We Ae Ro .. .. We
    We We Ay We We We
    We We We We We We
    """
    
    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=max_steps,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )
    return env


def create_goal_sampler(env: MultiGridEnv, human_idx: int) -> TabularGoalSampler:
    """
    Create goal sampler for the trivial environment.
    
    Goals for the human (yellow agent) to reach specific cells:
    - (2,1): Easy once robot has pushed rock twice or moved back
    - (1,1): Medium difficulty 
    - (3,1): Hardest since robot must move back after pushing rock twice
    - (2,2): Already reached initially
    """
    goals = [
        ReachCellGoal(env, human_idx, (2, 1)),  # easy once robot pushes rock
        ReachCellGoal(env, human_idx, (1, 1)),  # medium difficulty
        ReachCellGoal(env, human_idx, (3, 1)),  # hardest
        ReachCellGoal(env, human_idx, (2, 2)),  # already reached
    ]
    probabilities = [0.4, 0.25, 0.25, 0.1]
    
    return TabularGoalSampler(goals, probabilities=probabilities)


# =============================================================================
# Goal Generator for Backward Induction
# =============================================================================

class TrivialGoalGenerator(PossibleGoalGenerator):
    """
    Goal generator for the trivial environment.
    
    Generates all possible cell goals for the human agent.
    
    Args:
        world_model: The environment.
        human_idx: Index of the human agent.
        goal_weights: Optional dict mapping cell tuples to weights.
                     Default weight is 1.0 for cells not in the dict.
    """
    
    def __init__(self, world_model: MultiGridEnv, human_idx: int, 
                 goal_weights: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__(world_model)
        self.human_idx = human_idx
        self.goal_weights = goal_weights or {}
        # All walkable cells the human could potentially reach
        self.goal_cells = [
            (1, 1),
            (2, 1), 
            (3, 1),
            (4, 1),
            (2, 2),  # Human's starting position
        ]
    
    def generate(self, state, human_agent_index: int):
        """Generate all possible goals with weights."""
        for cell in self.goal_cells:
            goal = ReachCellGoal(self.env, human_agent_index, cell)
            weight = self.goal_weights.get(cell, 1.0)
            yield (goal, weight)


# =============================================================================
# Rollout and Movie Generation
# =============================================================================

def get_joint_action_names(num_agents: int, num_single_actions: int = 4) -> list:
    """
    Generate joint action names for multiple agents.
    
    For 1 agent with 4 actions, generates names like:
    'still', 'left', 'right', 'forward'
    
    For 2 agents, generates names like:
    'still, still', 'still, left', etc.
    """
    single_names = SINGLE_ACTION_NAMES[:num_single_actions]
    if num_agents == 1:
        return single_names
    combinations = list(itertools.product(single_names, repeat=num_agents))
    return [', '.join(combo) for combo in combinations]


def run_policy_rollout(
    env: MultiGridEnv,
    robot_policy: TabularRobotPolicy,
    human_policy_prior: TabularHumanPolicyPrior,
    goal_sampler: TabularGoalSampler,
    human_idx: int,
    robot_idx: int,
    Vr_values: Optional[Dict] = None,
    Vh_values: Optional[Dict] = None,
    beta_r: float = 100.0,
    xi: float = 1.0,
    eta: float = 1.0,
    zeta: float = 2.0,
) -> int:
    """
    Run a single rollout with the computed robot policy.
    
    Uses env's internal video recording - frames are captured automatically.
    
    Returns:
        Number of steps taken.
    """
    # Generate action names
    joint_action_names = get_joint_action_names(1)  # Single robot
    
    env.reset()
    steps_taken = 0
    
    # Sample initial goal for human
    state = env.get_state()
    human_goal, _ = goal_sampler.sample(state, human_idx)
    human_goals = {human_idx: human_goal}
    
    def compute_annotation_text(state, selected_action=None):
        """Compute annotation text for the current state."""
        lines = []
        
        # Get robot policy distribution
        policy_dist = robot_policy(state)
        
        if policy_dist is None:
            lines.append("Terminal state")
            if Vr_values is not None and state in Vr_values:
                lines.append(f"V_r: {Vr_values[state]:.4f}")
            return lines
        
        # Compute Q_r values (from policy probabilities)
        # π_r(a) ∝ (-Q_r(a))^{-β_r}, so Q_r(a) ∝ -π_r(a)^{-1/β_r}
        # We can recover relative Q values but not absolute values
        # Instead, show V_r and U_r if available
        
        if Vr_values is not None and state in Vr_values:
            v_r = Vr_values[state]
            lines.append(f"V_r: {v_r:.4f}")
            
            # Compute U_r from X_h values if available
            if Vh_values is not None and state in Vh_values:
                x_h_vals = []
                vh_state = Vh_values.get(state, {})
                if human_idx in vh_state:
                    for goal in goal_sampler.goals:
                        if goal in vh_state[human_idx]:
                            vh = vh_state[human_idx][goal]
                            x_h_vals.append(vh ** zeta)
                if x_h_vals:
                    x_h = sum(x_h_vals) / len(x_h_vals)  # Average over goals
                    if x_h > 0:
                        y = x_h ** (-xi)
                        u_r = -(y ** eta)
                        lines.append(f"U_r: {u_r:.4f}")
                        lines.append(f"X_h: {x_h:.4f}")
        
        lines.append("")
        lines.append("π_r probs:")
        
        # Sort actions by probability for display
        sorted_actions = sorted(policy_dist.items(), key=lambda x: -x[1])
        
        for action_profile, prob in sorted_actions:
            action_idx = action_profile[0]  # Single robot
            action_name = joint_action_names[action_idx] if action_idx < len(joint_action_names) else f"a{action_idx}"
            marker = ">" if selected_action is not None and action_idx == selected_action else " "
            lines.append(f"{marker}{action_name:>7}: {prob:.3f}")
        
        return lines
    
    def get_robot_action(state):
        """Sample action from robot policy."""
        return robot_policy.sample(state)
    
    # Render initial frame
    robot_action = get_robot_action(state)
    selected_action = robot_action[0] if robot_action else None
    annotation = compute_annotation_text(state, selected_action)
    env.render(mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE,
               annotation_text=annotation, annotation_panel_width=ANNOTATION_PANEL_WIDTH,
               annotation_font_size=ANNOTATION_FONT_SIZE, goal_overlays=human_goals)
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Get actions
        actions = [0] * len(env.agents)
        
        # Human uses computed policy prior
        human_action_dist = human_policy_prior(state, human_idx, human_goal)
        if human_action_dist is not None:
            actions[human_idx] = np.random.choice(len(human_action_dist), p=human_action_dist)
        
        # Robot uses computed policy
        robot_action = robot_policy.sample(state)
        if robot_action is not None:
            actions[robot_idx] = robot_action[0]  # Single robot
        
        # Step environment
        _, _, done, _ = env.step(actions)
        steps_taken += 1
        
        # Render frame
        new_state = env.get_state()
        new_robot_action = get_robot_action(new_state)
        selected_action = new_robot_action[0] if new_robot_action else None
        annotation = compute_annotation_text(new_state, selected_action)
        env.render(mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE,
                   annotation_text=annotation, annotation_panel_width=ANNOTATION_PANEL_WIDTH,
                   annotation_font_size=ANNOTATION_FONT_SIZE, goal_overlays=human_goals)
        
        if done:
            break
    
    return steps_taken


# =============================================================================
# Main
# =============================================================================

def main(
    max_steps: int = 5,
    num_rollouts: int = 20,
    parallel: bool = True,
    num_workers: Optional[int] = 4,
    beta_h: float = 2.0,
    beta_r: float = 100.0,
    gamma_h: float = 0.99,
    gamma_r: float = 0.99,
    zeta: float = 2.0,
    xi: float = 1.0,
    eta: float = 1.1,
    terminal_Vr: float = -1e-10,
    goal_weights: Optional[Dict[Tuple[int, int], float]] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
    movie_fps: int = 3,
    save_video_path: Optional[str] = None,
):
    """Run Phase 2 backward induction demo."""
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("Phase 2 Robot Policy via Backward Induction")
    print("Computing exact robot policy using DAG backward induction")
    print("=" * 70)
    print()
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'phase2_backward_induction')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = create_trivial_env(max_steps=max_steps)
    env.reset()
    
    # Identify agents
    # In our trivial env, robot (grey/agent 0) comes before human (yellow/agent 1)
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
            print(f"  Human (agent {i}): pos={tuple(agent.pos)}")
        elif agent.color == 'grey':
            robot_idx = i
            print(f"  Robot (agent {i}): pos={tuple(agent.pos)}")
    
    if human_idx is None or robot_idx is None:
        raise ValueError("Could not identify human and robot agents")
    
    human_agent_indices = [human_idx]
    robot_agent_indices = [robot_idx]
    
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Max steps: {max_steps}")
    print()
    
    # Create goal generator for backward induction
    print("Creating goal generator...")
    goal_generator = TrivialGoalGenerator(env, human_idx, goal_weights=goal_weights)
    
    # Also create goal sampler for rollouts
    goal_sampler = create_goal_sampler(env, human_idx)
    
    print(f"  Goals: {len(goal_generator.goal_cells)} possible cell goals")
    if goal_weights:
        print(f"  Custom goal weights: {goal_weights}")
    print()
    
    # =========================================================================
    # Phase 1: Compute Human Policy Prior
    # =========================================================================
    print("=" * 70)
    print("Phase 1: Computing Human Policy Prior")
    print("=" * 70)
    print(f"  beta_h (inverse temperature): {beta_h}")
    print(f"  gamma_h (discount factor): {gamma_h}")
    if parallel:
        print(f"  Mode: parallel ({num_workers or 'auto'} workers)")
    else:
        print("  Mode: sequential")
    print()
    
    t0 = time.time()
    human_policy_prior, Vh_phase1 = compute_human_policy_prior(
        world_model=env,
        human_agent_indices=human_agent_indices,
        possible_goal_generator=goal_generator,
        believed_others_policy=None,  # Uniform prior for others
        beta_h=beta_h,
        gamma_h=gamma_h,
        parallel=parallel,
        num_workers=num_workers,
        return_Vh=True,
    )
    t1 = time.time()
    
    print(f"Human policy prior computed in {t1 - t0:.2f} seconds")
    print(f"  States in policy: {len(human_policy_prior.values)}")
    print(f"  States with Vh values: {len(Vh_phase1)}")
    print()
    
    # =========================================================================
    # Phase 2: Compute Robot Policy
    # =========================================================================
    print("=" * 70)
    print("Phase 2: Computing Robot Policy")
    print("=" * 70)
    print(f"  beta_r (power-law concentration): {beta_r}")
    print(f"  gamma_r (discount factor): {gamma_r}")
    print(f"  zeta (risk aversion): {zeta}")
    print(f"  xi (inter-human inequality aversion): {xi}")
    print(f"  eta (intertemporal inequality aversion): {eta}")
    if parallel:
        print(f"  Mode: parallel ({num_workers or 'auto'} workers)")
    else:
        print("  Mode: sequential")
    print()
    
    t0 = time.time()
    robot_policy, Vr_values, Vh_phase2 = compute_robot_policy(
        world_model=env,
        human_agent_indices=human_agent_indices,
        robot_agent_indices=robot_agent_indices,
        possible_goal_generator=goal_generator,
        human_policy_prior=human_policy_prior,
        beta_r=beta_r,
        gamma_h=gamma_h,
        gamma_r=gamma_r,
        zeta=zeta,
        xi=xi,
        eta=eta,
        terminal_Vr=terminal_Vr,
        parallel=parallel,
        num_workers=num_workers,
        return_values=True,
    )
    t1 = time.time()
    
    print(f"Robot policy computed in {t1 - t0:.2f} seconds")
    print(f"  States in policy: {len(robot_policy.values)}")
    print(f"  States with Vh values: {len(Vh_phase2)}")
    print()
    
    # Show some V_r values for initial state
    initial_state = env.get_state()
    if initial_state in Vr_values:
        print(f"  V_r(initial state): {Vr_values[initial_state]:.4f}")
    
    # Show robot policy for initial state
    policy_dist = robot_policy(initial_state)
    if policy_dist:
        print("  Robot policy at initial state:")
        for action_profile, prob in sorted(policy_dist.items(), key=lambda x: -x[1]):
            action_name = SINGLE_ACTION_NAMES[action_profile[0]] if action_profile[0] < 4 else f"a{action_profile[0]}"
            print(f"    {action_name}: {prob:.3f}")
    print()
    
    # =========================================================================
    # Compare Vh values from Phase 1 vs Phase 2
    # =========================================================================
    print("=" * 70)
    print("Comparing Vh values: Phase 1 vs Phase 2")
    print("=" * 70)
    print()
    print("Phase 1 computes Vh assuming UNIFORM RANDOM robot action.")
    print("Phase 2 computes Vh under the COMPUTED robot policy (which maximizes")
    print("aggregate human power, not individual goal achievement).")
    print()
    print("Differences arise because:")
    print("  - Phase 2 robot helps with some goals (increases Vh)")
    print("  - But optimizing aggregate power may hurt specific goals (decreases Vh)")
    print()
    
    # Collect all (state, agent, goal) triples that exist in both
    differences = []
    for state in Vh_phase1:
        if state not in Vh_phase2:
            continue
        for agent_idx in Vh_phase1[state]:
            if agent_idx not in Vh_phase2[state]:
                continue
            for goal in Vh_phase1[state][agent_idx]:
                if goal not in Vh_phase2[state][agent_idx]:
                    continue
                vh1 = Vh_phase1[state][agent_idx][goal]
                vh2 = Vh_phase2[state][agent_idx][goal]
                diff = vh2 - vh1
                differences.append((state, agent_idx, goal, vh1, vh2, diff))
    
    if differences:
        # Compute statistics
        diffs_only = [d[5] for d in differences]
        mean_diff = np.mean(diffs_only)
        max_diff = max(diffs_only)
        min_diff = min(diffs_only)
        num_improved = sum(1 for d in diffs_only if d > 1e-6)
        num_same = sum(1 for d in diffs_only if abs(d) <= 1e-6)
        num_worse = sum(1 for d in diffs_only if d < -1e-6)
        
        print(f"Statistics over {len(differences)} (state, agent, goal) entries:")
        print(f"  Mean difference (Phase2 - Phase1): {mean_diff:+.4f}")
        print(f"  Max difference:                    {max_diff:+.4f}")
        print(f"  Min difference:                    {min_diff:+.4f}")
        print(f"  Entries where Phase2 > Phase1:     {num_improved} ({100*num_improved/len(differences):.1f}%)")
        print(f"  Entries where Phase2 ≈ Phase1:     {num_same} ({100*num_same/len(differences):.1f}%)")
        print(f"  Entries where Phase2 < Phase1:     {num_worse} ({100*num_worse/len(differences):.1f}%)")
        print()
        
        # Show examples with largest improvements
        sorted_by_diff = sorted(differences, key=lambda x: -x[5])
        print("Top 5 entries with largest improvement (Phase2 - Phase1):")
        for i, (state, agent_idx, goal, vh1, vh2, diff) in enumerate(sorted_by_diff[:5]):
            print(f"  {i+1}. Goal={goal}, Vh1={vh1:.4f}, Vh2={vh2:.4f}, Δ={diff:+.4f}")
        
        # Show entries where Phase 2 is worse (if any)
        if num_worse > 0:
            print()
            print(f"Entries where Phase2 < Phase1 (robot policy hurts this goal):")
            sorted_worst = sorted(differences, key=lambda x: x[5])
            for i, (state, agent_idx, goal, vh1, vh2, diff) in enumerate(sorted_worst[:min(5, num_worse)]):
                print(f"  {i+1}. Goal={goal}, Vh1={vh1:.4f}, Vh2={vh2:.4f}, Δ={diff:+.4f}")
        
        # Show initial state comparison
        print()
        print("Vh comparison at initial state:")
        if initial_state in Vh_phase1 and initial_state in Vh_phase2:
            for agent_idx in Vh_phase1[initial_state]:
                if agent_idx in Vh_phase2[initial_state]:
                    print(f"  Agent {agent_idx}:")
                    for goal in Vh_phase1[initial_state][agent_idx]:
                        vh1 = Vh_phase1[initial_state][agent_idx].get(goal, 0)
                        vh2 = Vh_phase2[initial_state][agent_idx].get(goal, 0)
                        diff = vh2 - vh1
                        print(f"    {goal}: Phase1={vh1:.4f}, Phase2={vh2:.4f}, Δ={diff:+.4f}")
        else:
            print("  (initial state not in both Vh tables)")
    else:
        print("  No common (state, agent, goal) entries found for comparison.")
    print()
    
    # =========================================================================
    # Generate Rollout Movie
    # =========================================================================
    print("=" * 70)
    print(f"Generating {num_rollouts} rollouts")
    print("=" * 70)
    
    # Start video recording
    env.start_video_recording()
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        steps = run_policy_rollout(
            env=env,
            robot_policy=robot_policy,
            human_policy_prior=human_policy_prior,
            goal_sampler=goal_sampler,
            human_idx=human_idx,
            robot_idx=robot_idx,
            Vr_values=Vr_values,
            Vh_values=Vh_phase2,
            beta_r=beta_r,
            xi=xi,
            eta=eta,
            zeta=zeta,
        )
        if (rollout_idx + 1) % 5 == 0:
            print(f"  Completed {rollout_idx + 1}/{num_rollouts} rollouts")
    
    # Save movie
    if save_video_path:
        movie_path = save_video_path
    else:
        movie_path = os.path.join(output_dir, 'rollouts.mp4')
    
    if os.path.exists(movie_path):
        os.remove(movie_path)
    
    env.save_video(movie_path, fps=movie_fps)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"  Movie saved to: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 2 Robot Policy via Backward Induction Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment options
    parser.add_argument('--steps', '-s', type=int, default=5,
                        help='Maximum steps per episode (horizon length)')
    parser.add_argument('--rollouts', '-r', type=int, default=20,
                        help='Number of rollouts to generate')
    
    # Computation options
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Use parallel computation')
    parser.add_argument('--workers', '-w', type=int, default=12,
                        help='Number of parallel workers (default: auto)')
    
    # Theory parameters
    parser.add_argument('--beta_h', type=float, default=2.0,
                        help='Inverse temperature for human policy')
    parser.add_argument('--beta_r', type=float, default=100.0,
                        help='Power-law concentration for robot policy (lower values avoid numerical overflow)')
    parser.add_argument('--gamma_h', type=float, default=0.99,
                        help='Discount factor for human values')
    parser.add_argument('--gamma_r', type=float, default=0.99,
                        help='Discount factor for robot values')
    parser.add_argument('--zeta', type=float, default=2.0,
                        help='Risk-aversion parameter')
    parser.add_argument('--xi', type=float, default=1.0,
                        help='Inter-human inequality aversion')
    parser.add_argument('--eta', type=float, default=1.1,
                        help='Intertemporal inequality aversion')
    parser.add_argument('--terminal_Vr', type=float, default=-1e-10,
                        help='V_r value at terminal states (must be negative)')
    
    # Goal weight options
    parser.add_argument('--weight11', type=float, default=1.0,
                        help='Weight for goal cell (1,1)')
    parser.add_argument('--weight21', type=float, default=1.0,
                        help='Weight for goal cell (2,1) - easy once robot pushes rock')
    parser.add_argument('--weight31', type=float, default=1.0,
                        help='Weight for goal cell (3,1) - hardest, robot must return')
    parser.add_argument('--weight41', type=float, default=1.0,
                        help='Weight for goal cell (4,1)')
    parser.add_argument('--weight22', type=float, default=1.0,
                        help='Weight for goal cell (2,2) - human start position')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Path to save rollout video')
    parser.add_argument('--fps', type=int, default=3,
                        help='Frames per second for video')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Build goal_weights dict from command line args
    goal_weights = {
        (1, 1): args.weight11,
        (2, 1): args.weight21,
        (3, 1): args.weight31,
        (4, 1): args.weight41,
        (2, 2): args.weight22,
    }
    # Remove entries with default weight 1.0 to keep output clean
    goal_weights = {k: v for k, v in goal_weights.items() if v != 1.0}
    
    main(
        max_steps=args.steps,
        num_rollouts=args.rollouts,
        parallel=args.parallel,
        num_workers=args.workers,
        beta_h=args.beta_h,
        beta_r=args.beta_r,
        gamma_h=args.gamma_h,
        gamma_r=args.gamma_r,
        zeta=args.zeta,
        xi=args.xi,
        eta=args.eta,
        terminal_Vr=args.terminal_Vr,
        goal_weights=goal_weights if goal_weights else None,
        seed=args.seed,
        output_dir=args.output_dir,
        movie_fps=args.fps,
        save_video_path=args.save_video,
    )

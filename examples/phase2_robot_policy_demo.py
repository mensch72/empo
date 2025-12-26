#!/usr/bin/env python3
"""
Phase 2 Robot Policy Learning Demo.

This script demonstrates Phase 2 of the EMPO framework - learning a robot policy
that maximizes aggregate human power. Based on equations (4)-(9) from the paper.

Environment:
- 5x5 grid with 2 human agents (yellow) and 2 robot agents (grey)
- Humans follow HeuristicPotentialPolicy toward their goals
- Robots learn to help humans achieve their goals

The demo trains:
- Q_r: Robot state-action value (eq. 4)
- π_r: Robot policy using power-law softmax (eq. 5)
- V_h^e: Human goal achievement under robot policy (eq. 6)
- X_h: Aggregate goal achievement ability (eq. 7)
- U_r: Intrinsic robot reward (eq. 8)
- V_r: Robot state value (eq. 9)

Usage:
    python phase2_robot_policy_demo.py           # Full run (1000 episodes)
    python phase2_robot_policy_demo.py --quick   # Quick test (100 episodes)
    python phase2_robot_policy_demo.py --debug   # Enable verbose debug output

Output:
- TensorBoard logs in outputs/phase2_demo/
- Movie of 100 rollouts with the learned policy
"""

import sys
import os
import time
import random
import argparse

import numpy as np
import torch

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.multigrid import MultiGridGoalSampler, ReachCellGoal
from empo.possible_goal import DeterministicGoalSampler, TabularGoalSampler
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid import PathDistanceCalculator
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.multigrid.phase2 import train_multigrid_phase2


# ============================================================================
# Environment Definition
# ============================================================================

final_beta_r = 100.0  # Final beta_r for robot policy concentration

env_type = "trivial"

if env_type == "trivial":
    GRID_MAP = """
    We We We We We We
    We Ae Ro .. .. We
    We We Ay We We We
    We We We We We We
    """
    # note: human is agent index 1

    MAX_STEPS = 10
    NUM_ROLLOUTS = 10  # Rollouts for final movie
    MOVIE_FPS = 2

    goal_sampler_factory = lambda env: TabularGoalSampler([
        ReachCellGoal(env, 1, (2,1)),
        ReachCellGoal(env, 1, (2,2))
        ], probabilities=[0.9, 0.1])

else:
    GRID_MAP = """
    We We We We We We We
    We .. .. .. .. .. We
    We .. Ae .. Ay .. We
    We .. .. .. .. .. We
    We .. Ay .. Ae .. We
    We .. .. .. .. .. We
    We We We We We We We
    """

    MAX_STEPS = 20
    NUM_ROLLOUTS = 100  # Rollouts for final movie
    MOVIE_FPS = 3

    goal_sampler_factory = MultiGridGoalSampler
class Phase2DemoEnv(MultiGridEnv):
    """
    A simple grid environment for Phase 2 demo.
    """
    
    def __init__(self, max_steps: int = MAX_STEPS):
        super().__init__(
            map=GRID_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')


# ============================================================================
# Rollout and Movie Generation
# ============================================================================

# Action names for SmallActions
ACTION_NAMES = ['still', 'left', 'right', 'forward']


def run_policy_rollout(
    env: MultiGridEnv,
    robot_q_network,
    human_policy,
    goal_sampler,
    human_indices,
    robot_indices,
    device: str = 'cpu',
    networks=None,  # Phase2Networks for value annotations
    config=None,    # Phase2Config for beta_r
) -> int:
    """
    Run a single rollout with the learned robot policy.
    
    Uses env's internal video recording - frames are captured automatically.
    If networks is provided, adds value annotations to each frame.
    
    Returns:
        Number of steps taken.
    """
    env.reset()
    num_actions = env.action_space.n
    steps_taken = 0
    
    # Sample initial goals for humans
    state = env.get_state()
    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler.sample(state, h)
        human_goals[h] = goal
    
    def compute_annotation_text(state, selected_action=None):
        """Compute annotation text for the current state."""
        if networks is None:
            return None
        
        lines = []
        
        with torch.no_grad():
            # Get Q_r values
            q_values = robot_q_network.encode_and_forward(state, env, device)
            q_np = q_values.squeeze().cpu().numpy()
            
            # Compute policy probabilities
            beta_r = config.beta_r if config else 10.0
            pi_r = robot_q_network.get_policy(q_values, beta_r=beta_r)
            pi_np = pi_r.squeeze().cpu().numpy()
            
            # Compute U_r and V_r
            state_encoder = networks.u_r.state_encoder if hasattr(networks.u_r, 'state_encoder') else None
            u_r_val = 0.0
            v_r_val = 0.0
            if state_encoder is not None:
                # Use encode_state method (returns tuple of tensors with batch dim already)
                s_encoded = state_encoder.encode_state(state, env, device)
                # s_encoded already has batch dimension, no need to unsqueeze
                _, u_r = networks.u_r.forward(*s_encoded)
                u_r_val = u_r.item()
                v_r_val = u_r_val + (pi_r * q_values).sum().item()
            
            # Build annotation lines
            lines.append(f"U_r: {u_r_val:.4f}")
            lines.append(f"V_r: {v_r_val:.4f}")
            lines.append("")
            lines.append("Q_r values:")
            
            for action_idx in range(len(q_np)):
                action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else f"a{action_idx}"
                marker = ">" if selected_action is not None and action_idx == selected_action else " "
                lines.append(f"{marker}{action_name:>7}: {q_np[action_idx]:.3f}")
            
            lines.append("")
            lines.append("π_r probs:")
            
            for action_idx in range(len(pi_np)):
                action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else f"a{action_idx}"
                marker = ">" if selected_action is not None and action_idx == selected_action else " "
                lines.append(f"{marker}{action_name:>7}: {pi_np[action_idx]:.3f}")
        
        return lines
    
    def get_greedy_action(state):
        """Get the greedy action for annotation (what robot would do from this state)."""
        with torch.no_grad():
            q_values = robot_q_network.encode_and_forward(state, env, device)
            robot_action = robot_q_network.sample_action(q_values, epsilon=0.0)
            return robot_action[0] if len(robot_action) > 0 else None
    
    # Render initial frame with annotations (showing what action would be taken)
    annotation = compute_annotation_text(state, get_greedy_action(state))
    env.render(mode='rgb_array', highlight=False, annotation_text=annotation)
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Get actions
        actions = [0] * len(env.agents)
        
        # Humans use heuristic policy
        for h in human_indices:
            actions[h] = human_policy.sample(state, h, human_goals[h])
        
        # Robots use learned Q-network
        with torch.no_grad():
            q_values = robot_q_network.encode_and_forward(state, env, device)
            # Use greedy action (epsilon=0)
            robot_action = robot_q_network.sample_action(q_values, epsilon=0.0)
            
            # Assign actions to robots
            for i, r in enumerate(robot_indices):
                if i < len(robot_action):
                    actions[r] = robot_action[i]
        
        # Step environment
        _, _, done, _ = env.step(actions)
        steps_taken += 1
        
        # Render frame showing current state with what action WOULD be taken from it
        new_state = env.get_state()
        annotation = compute_annotation_text(new_state, get_greedy_action(new_state))
        env.render(mode='rgb_array', highlight=False, annotation_text=annotation)
        
        if done:
            break
    
    return steps_taken

# ============================================================================
# Main
# ============================================================================

def main(quick_mode: bool = False, debug: bool = False):
    """Run Phase 2 demo."""
    print("=" * 70)
    print("Phase 2 Robot Policy Learning Demo")
    print("Learning robot policy to maximize aggregate human power")
    print("Based on EMPO paper equations (4)-(9)")
    print("=" * 70)
    print()
    
    # Configuration
    if quick_mode:
        num_episodes = 100
        num_rollouts = 10
        # Use smaller batches and network for faster iteration in quick mode
        batch_size = 16
        x_h_batch_size = 32
        hidden_dim = 64  # Smaller network for faster forward passes
        goal_feature_dim = 32
        agent_embedding_dim = 8
        print("[QUICK MODE] Running with reduced episodes, rollouts, batch sizes, and network size")
    elif env_type == "trivial":
        num_episodes = 3000
        num_rollouts = NUM_ROLLOUTS
        batch_size = 16
        x_h_batch_size = 32
        # Very small networks for trivial task
        hidden_dim = 16
        goal_feature_dim = 8
        agent_embedding_dim = 4
        print("[TRIVIAL ENV] Using minimal network sizes for simple task")
    else:
        num_episodes = 1000
        num_rollouts = NUM_ROLLOUTS
        batch_size = 32
        x_h_batch_size = 64 #128
        hidden_dim = 128
        goal_feature_dim = 64
        agent_embedding_dim = 16
    
    device = 'cpu'
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'phase2_demo_trivial')
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    # Create environment
    print("Creating environment...")
    env = Phase2DemoEnv(max_steps=MAX_STEPS)
    env.reset()
    
    # Identify agents
    human_indices = []
    robot_indices = []
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_indices.append(i)
            print(f"  Human {i}: pos={tuple(agent.pos)}")
        elif agent.color == 'grey':
            robot_indices.append(i)
            print(f"  Robot {i}: pos={tuple(agent.pos)}")
    
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Humans: {len(human_indices)}")
    print(f"  Robots: {len(robot_indices)}")
    print()
    
    # Create path calculator for heuristic policy
    print("Creating path calculator and human policy...")
    path_calc = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env
    )
    
    # Create human policy using existing HeuristicPotentialPolicy
    human_policy = HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_indices,
        path_calculator=path_calc,
        beta=10.0  # Quite deterministic
    )
    
    # Create goal sampler using existing MultiGridGoalSampler
    goal_sampler = goal_sampler_factory(env)
    
    # Wrapper to adapt goal sampler to trainer's expected interface
    def goal_sampler_fn(state, human_idx):
        goal, _ = goal_sampler.sample(state, human_idx)
        return goal
    
    # Wrapper to adapt human policy to trainer's expected interface
    def human_policy_fn(state, human_idx, goal):
        return human_policy.sample(state, human_idx, goal)
    
    print()
    
    # Phase 2 configuration
    # Note: X_h now uses all human-goal pairs from each transition (not random sampling)
    # which provides more samples per update. We also use a larger batch specifically
    # for X_h since it has inherently higher variance (expectation over many goals).
    config = Phase2Config(
        gamma_r=0.95, # relatively impatient
        gamma_h=0.95, # relatively impatient
        zeta=2.0,      # Risk aversion
        xi=1.0,        # Inter-human inequality aversion
        eta=1.1,       # Intertemporal inequality aversion
        beta_r=final_beta_r,    # Robot policy concentration
        epsilon_r_start=1.0,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=num_episodes * 10,
        lr_q_r=1e-4,
        lr_v_r=1e-4,
        lr_v_h_e=1e-3,  # faster since V_h^e is critical and the basis for other things
        lr_x_h=1e-4,
        lr_u_r=1e-4,
        buffer_size=10000,
        batch_size=batch_size,
        x_h_batch_size=x_h_batch_size,  # Larger batch for X_h to reduce high variance
        num_episodes=num_episodes,
        steps_per_episode=env.max_steps,
        updates_per_step=1,
        goal_resample_prob=0.1,
        v_h_target_update_freq=100,  # Standard target network update frequency
    )
    
    # Train Phase 2
    print("Training Phase 2 robot policy...")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Steps per episode: {config.steps_per_episode}")
    print(f"  TensorBoard: {tensorboard_dir}")
    print()
    
    t0 = time.time()
    robot_q_network, networks, history = train_multigrid_phase2(
        world_model=env,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_policy_prior=human_policy_fn,
        goal_sampler=goal_sampler_fn,
        config=config,
        hidden_dim=hidden_dim,
        goal_feature_dim=goal_feature_dim,
        agent_embedding_dim=agent_embedding_dim,
        device=device,
        verbose=True,
        debug=debug,
        tensorboard_dir=tensorboard_dir
    )
    elapsed = time.time() - t0
    
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Show loss history
    if history and len(history) > 0:
        print("\nLoss history (last 5 episodes):")
        for i, losses in enumerate(history[-5:]):
            episode_num = len(history) - 5 + i
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if v > 0)
            print(f"  Episode {episode_num}: {loss_str}")
    
    # Generate rollout movie using env's built-in video recording
    print(f"\nGenerating {num_rollouts} rollouts with learned policy...")
    
    # Start video recording
    env.start_video_recording()
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        steps = run_policy_rollout(
            env=env,
            robot_q_network=robot_q_network,
            human_policy=human_policy,
            goal_sampler=goal_sampler,
            human_indices=human_indices,
            robot_indices=robot_indices,
            device=device,
            networks=networks,
            config=config
        )
        if (rollout_idx + 1) % 10 == 0:
            print(f"  Completed {rollout_idx + 1}/{num_rollouts} rollouts ({len(env._video_frames)} total frames)")
    
    # Save movie using env's save_video method (uses imageio[ffmpeg])
    movie_path = os.path.join(output_dir, 'phase2_robot_policy_demo.mp4')
    if os.path.exists(movie_path):
        os.remove(movie_path)
    env.save_video(movie_path, fps=MOVIE_FPS)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    print(f"  View with: tensorboard --logdir={tensorboard_dir}")
    print(f"  Movie: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2 Robot Policy Learning Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable verbose debug output')
    args = parser.parse_args()
    main(quick_mode=args.quick, debug=args.debug)

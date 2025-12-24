#!/usr/bin/env python3
"""
Phase 2 Robot Policy Learning Demo.

This script demonstrates Phase 2 of the EMPO framework - learning a robot policy
that maximizes aggregate human power. Based on equations (4)-(9) from the paper.

Environment:
- 5x5 grid with 2 human agents (yellow) and 1 robot agent (grey)
- Humans follow HeuristicPotentialPolicy toward their goals
- Robot learns to help humans achieve their goals

The demo trains:
- Q_r: Robot state-action value (eq. 4)
- π_r: Robot policy using power-law softmax (eq. 5)
- V_h^e: Human goal achievement under robot policy (eq. 6)
- X_h: Aggregate goal achievement ability (eq. 7)
- U_r: Intrinsic robot reward (eq. 8)
- V_r: Robot state value (eq. 9)
"""

import sys
import os
import time
import argparse

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.multigrid import MultiGridGoalSampler
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid import PathDistanceCalculator
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.multigrid.phase2 import train_multigrid_phase2


# ============================================================================
# Environment Definition
# ============================================================================

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


class Phase2DemoEnv(MultiGridEnv):
    """
    A simple 5x5 grid environment for Phase 2 demo.
    
    - 2 yellow agents (humans)
    - 1 grey agent (robot)
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
# Main
# ============================================================================

def main(quick_mode: bool = False):
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
        print("[QUICK MODE] Running with reduced episodes")
    else:
        num_episodes = 1000
    
    device = 'cpu'
    
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
        softness=10.0  # Moderately deterministic
    )
    
    # Create goal sampler using existing MultiGridGoalSampler
    goal_sampler = MultiGridGoalSampler(env)
    
    # Wrapper to adapt goal sampler to trainer's expected interface
    def goal_sampler_fn(state, human_idx):
        goal, _ = goal_sampler.sample(state, human_idx)
        return goal
    
    # Wrapper to adapt human policy to trainer's expected interface
    def human_policy_fn(state, human_idx, goal):
        return human_policy.sample(state, human_idx, goal)
    
    print()
    
    # Phase 2 configuration
    config = Phase2Config(
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,      # Risk preference
        xi=1.0,        # Inter-human inequality aversion
        eta=1.1,       # Intertemporal inequality aversion
        beta_r=5.0,    # Robot policy concentration
        epsilon_r_start=1.0,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=num_episodes * 10,
        lr_q_r=1e-3,
        lr_v_r=1e-3,
        lr_v_h_e=1e-3,
        lr_x_h=1e-3,
        lr_u_r=1e-3,
        buffer_size=10000,
        batch_size=32,
        num_episodes=num_episodes,
        steps_per_episode=env.max_steps,
        updates_per_step=1,
        goal_resample_prob=0.1,
    )
    
    # Train Phase 2
    print("Training Phase 2 robot policy...")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Steps per episode: {config.steps_per_episode}")
    print()
    
    t0 = time.time()
    robot_q_network, networks, history = train_multigrid_phase2(
        world_model=env,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_policy_prior=human_policy_fn,
        goal_sampler=goal_sampler_fn,
        config=config,
        hidden_dim=128,
        device=device,
        verbose=True
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
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2 Robot Policy Learning Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    args = parser.parse_args()
    main(quick_mode=args.quick)

#!/usr/bin/env python3
"""
Phase 2 Lookup Table Networks Example.

This example demonstrates using lookup table (tabular) networks for Phase 2
robot policy learning. Lookup tables store exact values for each state/goal
without function approximation, making them ideal for:
- Small state spaces (< 100K states)
- Debugging and interpretability  
- Baseline comparisons with neural approaches

Environment: Tiny 4x4 grid with 1 human, 1 robot
- Human tries to reach goals
- Robot learns to help by maximizing human's goal-achievement ability

This example shows:
1. How to configure Phase2Config for lookup tables
2. How to create lookup table networks using factory functions
3. How to run training with lookup tables (same API as neural networks)
4. How to inspect lookup table contents after training

Usage:
    python lookup_table_phase2_demo.py                 # Full training
    python lookup_table_phase2_demo.py --steps 100     # Quick test
    python lookup_table_phase2_demo.py --inspect       # Show learned values
"""

import sys
import os
import argparse
from typing import Dict, List, Tuple, Any

import torch

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Ball, Block
)
from empo.multigrid import MultiGridGoalSampler, ReachCellGoal
from empo.possible_goal import TabularGoalSampler
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.trainer import Phase2Networks
from empo.nn_based.phase2.network_factory import create_all_phase2_lookup_networks
from empo.nn_based.phase2.lookup import (
    is_lookup_table_network,
    get_all_lookup_tables,
    get_total_table_size,
)
import random as python_random


class SimpleRandomPolicy:
    """Simple uniform random policy for human agents."""
    
    def __init__(self, env, human_agent_indices):
        self.env = env
        self.human_agent_indices = human_agent_indices
        self.num_actions = env.action_space.n
    
    def sample(self, state, human_idx, goal):
        """Sample a random action."""
        return python_random.randint(0, self.num_actions - 1)


# =============================================================================
# Environment Configuration
# =============================================================================

GRID_MAP = """
We We We We
We Ae Ro We
We .. Ay We
We We We We
"""

MAX_STEPS = 15
NUM_GOALS = 3  # Number of goals to sample from


def create_environment():
    """Create the tiny test environment."""
    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=MAX_STEPS,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions,
    )
    return env


def create_goal_sampler(env):
    """Create goal sampler for the environment.
    
    Goals: human (agent 0) wants to reach specific cells.
    """
    # Define achievable goals
    goals = [
        ReachCellGoal(env, 0, (1, 1)),  # Human's start
        ReachCellGoal(env, 0, (2, 1)),  # Robot pushed out
        ReachCellGoal(env, 0, (1, 2)),  # Bottom-left
    ]
    
    return TabularGoalSampler(
        goals=goals,
        probabilities=[0.4, 0.3, 0.3],  # Weighted goal distribution
    )


# =============================================================================
# Training Configuration
# =============================================================================

def create_lookup_config() -> Phase2Config:
    """Create Phase2Config for lookup table training.
    
    Key settings:
    - use_lookup_tables=True: Enable lookup table mode
    - All per-network lookup flags set to True
    - Shorter warmup since lookup tables converge faster
    """
    return Phase2Config(
        # Enable lookup tables
        use_lookup_tables=True,
        use_lookup_q_r=True,
        use_lookup_v_h_e=True,
        use_lookup_x_h=True,
        use_lookup_u_r=True,
        use_lookup_v_r=True,
        
        # Use U_r and V_r networks (needed for lookup tables to be created)
        u_r_use_network=True,
        v_r_use_network=True,
        
        # Default values for new table entries
        lookup_default_q_r=-1.0,
        lookup_default_v_r=-1.0,
        lookup_default_v_h_e=0.5,
        lookup_default_x_h=0.5,
        lookup_default_y=2.0,  # For U_r (y > 1)
        
        # Optimizer recreation interval (for lookup table growth)
        lookup_optimizer_recreate_interval=100,
        
        # Theory parameters
        gamma_r=0.99,
        gamma_h=0.99,
        beta_r=10.0,  # Power-law policy concentration
        
        # Training parameters (shorter warmup for lookup tables)
        warmup_v_h_e_steps=50,   # V_h^e warmup
        warmup_x_h_steps=50,     # X_h warmup  
        warmup_u_r_steps=50,     # U_r warmup
        warmup_q_r_steps=50,     # Q_r warmup
        beta_r_rampup_steps=100, # β_r ramp-up period
        
        # Higher learning rates work well for lookup tables
        lr_q_r=0.01,
        lr_v_r=0.01,
        lr_v_h_e=0.01,
        lr_x_h=0.01,
        lr_u_r=0.01,
        
        # Replay buffer
        buffer_size=1000,
        batch_size=32,
        
        # Episode settings
        steps_per_episode=MAX_STEPS,
        goal_resample_prob=0.1,
        
        # Exploration
        epsilon_r_start=0.5,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=500,
    )


def create_lookup_networks(config: Phase2Config, env: MultiGridEnv) -> Phase2Networks:
    """Create lookup table networks for Phase 2.
    
    Uses the factory function to create all networks at once.
    """
    num_actions = env.action_space.n  # Typically 5 for SmallActions
    num_robots = 1
    
    q_r, v_h_e, x_h, u_r, v_r = create_all_phase2_lookup_networks(
        config=config,
        num_actions=num_actions,
        num_robots=num_robots,
    )
    
    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r,
    )


# =============================================================================
# Minimal Trainer for Demo
# =============================================================================

class SimpleLookupTableTrainer:
    """
    Simplified trainer to demonstrate lookup table network training.
    
    This is a minimal implementation for demo purposes. For real training,
    use the full MultiGridPhase2Trainer.
    """
    
    def __init__(
        self,
        env: MultiGridEnv,
        networks: Phase2Networks,
        config: Phase2Config,
        goal_sampler,
        human_policy,
        device: str = 'cpu',
    ):
        self.env = env
        self.networks = networks
        self.config = config
        self.goal_sampler = goal_sampler
        self.human_policy = human_policy
        self.device = device
        
        # Human and robot agent indices
        self.human_indices = [0]
        self.robot_indices = [1]
        
        # Training state
        self.step_count = 0
        
        # Initialize optimizers
        self.optimizers = self._create_optimizers()
    
    def _create_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for all networks.
        
        Note: For lookup tables that start empty, we create a placeholder parameter
        so the optimizer can be created. Real parameters will be added as states are visited.
        """
        optimizers = {}
        
        # Helper to get parameters or a placeholder
        def get_params(network):
            params = list(network.parameters())
            if len(params) == 0:
                # Lookup table is empty, create a dummy parameter
                # This will be ignored but allows optimizer creation
                return [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]
            return params
        
        optimizers['q_r'] = torch.optim.Adam(get_params(self.networks.q_r), lr=self.config.lr_q_r)
        optimizers['v_h_e'] = torch.optim.Adam(get_params(self.networks.v_h_e), lr=self.config.lr_v_h_e)
        optimizers['x_h'] = torch.optim.Adam(get_params(self.networks.x_h), lr=self.config.lr_x_h)
        
        if self.networks.u_r is not None:
            optimizers['u_r'] = torch.optim.Adam(get_params(self.networks.u_r), lr=self.config.lr_u_r)
        if self.networks.v_r is not None:
            optimizers['v_r'] = torch.optim.Adam(get_params(self.networks.v_r), lr=self.config.lr_v_r)
        return optimizers
    
    def collect_experience(self) -> Tuple[Any, Any, Dict[int, Any], Any]:
        """Collect one transition from environment.
        
        Returns:
            Tuple of (state, next_state, goals, robot_action)
        """
        state = self.env.get_state()
        
        # Sample goals for humans
        goals = {}
        for h in self.human_indices:
            goal, _ = self.goal_sampler.sample(state, h)
            goals[h] = goal
        
        # Sample robot action (with manual epsilon exploration since we're not using trainer)
        with torch.no_grad():
            q_values = self.networks.q_r.forward(state, None, self.device)
            # Apply epsilon-greedy exploration manually
            if torch.rand(1).item() < 0.3:  # epsilon = 0.3
                flat_idx = torch.randint(0, self.networks.q_r.num_action_combinations, (1,)).item()
                robot_action = self.networks.q_r.action_index_to_tuple(flat_idx)
            else:
                robot_action = self.networks.q_r.sample_action(q_values, beta_r=self.config.beta_r)
        
        # Sample human actions using policy prior
        human_actions = []
        for h in self.human_indices:
            action = self.human_policy.sample(state, h, goals[h])
            human_actions.append(action)
        
        # Execute actions
        joint_action = {
            self.robot_indices[0]: robot_action[0],
            self.human_indices[0]: human_actions[0],
        }
        next_state = self.env.step(joint_action)
        
        return state, next_state, goals, robot_action
    
    def training_step(self) -> Dict[str, float]:
        """Perform one training step.
        
        Returns:
            Dict of loss values.
        """
        # Collect experience
        state, next_state, goals, robot_action = self.collect_experience()
        
        losses = {}
        
        # Train V_h^e
        for h, goal in goals.items():
            v_h_e_pred = self.networks.v_h_e.forward(state, None, h, goal, self.device)
            
            # Check if goal achieved
            goal_achieved = float(goal.is_achieved(next_state))
            
            with torch.no_grad():
                v_h_e_next = self.networks.v_h_e.forward(next_state, None, h, goal, self.device)
            
            # TD target
            target = goal_achieved + (1 - goal_achieved) * self.config.gamma_h * v_h_e_next
            
            loss = (v_h_e_pred - target) ** 2
            
            self.optimizers['v_h_e'].zero_grad()
            loss.backward()
            self.optimizers['v_h_e'].step()
            
            losses['v_h_e'] = loss.item()
        
        # Train Q_r (simplified)
        q_r_pred = self.networks.q_r.forward(state, None, self.device)
        action_idx = self.networks.q_r.action_tuple_to_index(robot_action)
        q_r_action = q_r_pred.squeeze()[action_idx]
        
        with torch.no_grad():
            q_r_next = self.networks.q_r.forward(next_state, None, self.device)
            v_r_next = q_r_next.max()  # Simplified V_r computation
        
        target_q = self.config.gamma_r * v_r_next
        
        loss = (q_r_action - target_q) ** 2
        
        self.optimizers['q_r'].zero_grad()
        loss.backward()
        self.optimizers['q_r'].step()
        
        losses['q_r'] = loss.item()
        
        self.step_count += 1
        
        # Reset environment periodically
        if self.step_count % MAX_STEPS == 0:
            self.env.reset()
        
        return losses
    
    def train(self, num_steps: int, print_every: int = 100):
        """Run training for specified number of steps."""
        self.env.reset()
        
        print(f"Training lookup table networks for {num_steps} steps...")
        print(f"Config: use_lookup_tables={self.config.use_lookup_tables}")
        print()
        
        for step in range(num_steps):
            losses = self.training_step()
            
            if (step + 1) % print_every == 0:
                # Get table sizes
                total_entries = get_total_table_size(self.networks)
                
                print(f"Step {step + 1}/{num_steps}")
                print(f"  Losses: {', '.join(f'{k}={v:.4f}' for k, v in losses.items())}")
                print(f"  Total table entries: {total_entries}")
                
                # Show per-network sizes
                lookup_tables = get_all_lookup_tables(self.networks)
                for name, net in lookup_tables.items():
                    print(f"    {name}: {len(net.table)} entries")
                print()


def inspect_lookup_tables(networks: Phase2Networks):
    """Print contents of lookup tables for inspection."""
    print("\n" + "="*60)
    print("LOOKUP TABLE CONTENTS")
    print("="*60)
    
    lookup_tables = get_all_lookup_tables(networks)
    
    for name, net in lookup_tables.items():
        print(f"\n{name.upper()} ({len(net.table)} entries):")
        print("-" * 40)
        
        if len(net.table) == 0:
            print("  (empty)")
            continue
        
        # Show first few entries
        for i, (key, param) in enumerate(net.table.items()):
            if i >= 5:  # Only show first 5
                print(f"  ... and {len(net.table) - 5} more entries")
                break
            
            value = param.data.cpu().numpy()
            print(f"  Key {key}: {value}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Lookup Table Demo")
    parser.add_argument('--steps', type=int, default=500, help="Number of training steps")
    parser.add_argument('--inspect', action='store_true', help="Show learned values after training")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    # Create environment
    print("Creating environment...")
    env = create_environment()
    env.reset()
    
    print(f"  Grid size: {env.grid.width}x{env.grid.height}")
    print(f"  Agents: {len(env.agents)} (1 human, 1 robot)")
    print(f"  Actions per agent: {env.action_space.n}")
    
    # Create goal sampler
    goal_sampler = create_goal_sampler(env)
    print(f"  Goals: {len(goal_sampler.goals)}")
    
    # Create human policy prior (uniform random for simplicity)
    human_policy = SimpleRandomPolicy(env, human_agent_indices=[0])
    
    # Create config for lookup tables
    print("\nConfiguring lookup table networks...")
    config = create_lookup_config()
    
    # Create lookup table networks
    networks = create_lookup_networks(config, env)
    
    # Verify they are lookup tables
    print("Network types:")
    for name, net in get_all_lookup_tables(networks).items():
        print(f"  {name}: {type(net).__name__}")
    
    # Create trainer
    trainer = SimpleLookupTableTrainer(
        env=env,
        networks=networks,
        config=config,
        goal_sampler=goal_sampler,
        human_policy=human_policy,
    )
    
    # Train
    print()
    trainer.train(num_steps=args.steps, print_every=100)
    
    # Inspect tables
    if args.inspect:
        inspect_lookup_tables(networks)
    
    print("\nDone!")
    print(f"Final table sizes: {get_total_table_size(networks)} total entries")


if __name__ == "__main__":
    main()

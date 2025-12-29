#!/usr/bin/env python3
"""
Phase 2 Robot Policy Learning Demo.

This script demonstrates Phase 2 of the EMPO framework - learning a robot policy
that maximizes aggregate human power. Based on equations (4)-(9) from the paper.

Environment options (set env_type variable):
- "trivial": Small 4x6 grid with 1 human, 1 robot (quick testing)
- "ensemble": Random 7x7 grids with 2 humans, 2 robots, walls, and objects

The demo trains:
- Q_r: Robot state-action value (eq. 4)
- π_r: Robot policy using power-law softmax (eq. 5)
- V_h^e: Human goal achievement under robot policy (eq. 6)
- X_h: Aggregate goal achievement ability (eq. 7)
- U_r: Intrinsic robot reward (eq. 8)
- V_r: Robot state value (eq. 9)

Usage:
    # Basic usage
    python phase2_robot_policy_demo.py           # Full run (trivial env)
    python phase2_robot_policy_demo.py --quick   # Quick test (100 episodes)
    python phase2_robot_policy_demo.py --ensemble # Use random ensemble environment
    python phase2_robot_policy_demo.py --async   # Use async actor-learner training
    
    # Save/restore for continued training (networks/policy saved by default)
    python phase2_robot_policy_demo.py --save_networks model.pt  # Custom path for networks
    python phase2_robot_policy_demo.py --restore_networks model.pt  # Resume training
    
    # Use saved policy for rollouts only (skip training)
    python phase2_robot_policy_demo.py --use_policy policy.pt --rollouts 50  # Load & run
    
    # Advanced options
    python phase2_robot_policy_demo.py --debug   # Enable verbose debug output
    python phase2_robot_policy_demo.py --profile # Profile training with torch.profiler
    python phase2_robot_policy_demo.py --rollouts 100 --save_video my_rollouts.mp4

Output:
- TensorBoard logs in outputs/phase2_demo_<env_type>/
- All networks saved to outputs/phase2_demo_<env_type>/all_networks.pt
- Policy saved to outputs/phase2_demo_<env_type>/policy.pt  
- Movie of rollouts with the learned policy (loaded from disk to verify save/restore)
- Profiler trace (with --profile): outputs/phase2_demo_<env_type>/profiler_trace.json
"""

import sys
import os
import time
import random
import argparse

import numpy as np
import torch

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.multigrid import MultiGridGoalSampler, ReachCellGoal, ReachRectangleGoal
from empo.possible_goal import DeterministicGoalSampler, TabularGoalSampler, PossibleGoalSampler
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid import PathDistanceCalculator
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.world_model_factory import CachedWorldModelFactory, EnsembleWorldModelFactory
from empo.nn_based.multigrid.phase2 import train_multigrid_phase2
from empo.nn_based.multigrid.phase2.robot_policy import MultiGridRobotPolicy


# ============================================================================
# Environment Definition
# ============================================================================

final_beta_r = 100.0  # Final beta_r for robot policy concentration

# Default environment type (can be overridden via --ensemble flag)
# These will be set properly in main() based on command line args
env_type = "trivial"

# --- Configuration variables (set based on env_type) ---
GRID_MAP = None
MAX_STEPS = 10
NUM_ROLLOUTS = 10
MOVIE_FPS = 2
goal_sampler_factory = None

# Ensemble-specific config (only used when env_type == "ensemble")
ENSEMBLE_GRID_SIZE = 7
ENSEMBLE_NUM_HUMANS = 2
ENSEMBLE_NUM_ROBOTS = 2
WALL_PROBABILITY = 0.12
DOOR_PROBABILITY = 0.02
BLOCK_PROBABILITY = 0.02
ROCK_PROBABILITY = 0.02
UNSTEADY_GROUND_PROBABILITY = 0.08
DOOR_KEY_COLOR = 'r'


def configure_environment(use_ensemble: bool):
    """Configure global environment variables based on mode."""
    global env_type, GRID_MAP, MAX_STEPS, NUM_ROLLOUTS, MOVIE_FPS, goal_sampler_factory
    
    if use_ensemble:
        env_type = "ensemble"
        GRID_MAP = None  # Not used for ensemble
        MAX_STEPS = 30
        NUM_ROLLOUTS = 20
        MOVIE_FPS = 3
        goal_sampler_factory = None  # Will use SmallGoalSampler
    else:
        env_type = "trivial"
        GRID_MAP = """
        We We We We We We
        We Ae Ro .. .. We
        We We Ay We We We
        We We We We We We
        """
        MAX_STEPS = 10
        NUM_ROLLOUTS = 10
        MOVIE_FPS = 2
        goal_sampler_factory = lambda env: TabularGoalSampler([
            ReachCellGoal(env, 1, (2,1)),
            ReachCellGoal(env, 1, (2,2))
        ], probabilities=[0.9, 0.1])


# ============================================================================
# World Model Factory Functions (for async training)
# ============================================================================
# These are module-level functions/classes that can be pickled for multiprocessing.

def _create_trivial_env():
    """Create the trivial demo environment."""
    # Hardcoded grid map (same as configure_environment sets)
    grid_map = """
    We We We We We We
    We Ae Ro .. .. We
    We We Ay We We We
    We We We We We We
    """
    
    # Create environment using MultiGridEnv directly (not Phase2DemoEnv which uses globals)
    env = MultiGridEnv(
        map=grid_map,
        max_steps=10,  # Fixed for trivial
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )
    return env


class _EnsembleEnvCreator:
    """
    Picklable callable class for creating RandomMultigridEnv instances.
    
    Stores configuration as instance attributes, making it picklable for multiprocessing.
    """
    
    def __init__(
        self,
        grid_size: int,
        num_humans: int, 
        num_robots: int,
        max_steps: int,
        wall_prob: float,
        door_prob: float,
        block_prob: float,
        rock_prob: float,
        unsteady_prob: float,
        door_key_color: str
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.wall_prob = wall_prob
        self.door_prob = door_prob
        self.block_prob = block_prob
        self.rock_prob = rock_prob
        self.unsteady_prob = unsteady_prob
        self.door_key_color = door_key_color
    
    def __call__(self):
        """Create a new RandomMultigridEnv with stored configuration."""
        return RandomMultigridEnv(
            grid_size=self.grid_size,
            num_humans=self.num_humans,
            num_robots=self.num_robots,
            max_steps=self.max_steps,
            seed=None,  # New seed each time for variety
            wall_prob=self.wall_prob,
            door_prob=self.door_prob,
            block_prob=self.block_prob,
            rock_prob=self.rock_prob,
            unsteady_prob=self.unsteady_prob,
            door_key_color=self.door_key_color
        )


# ============================================================================
# Random Ensemble Environment Generator
# ============================================================================

class RandomMultigridEnv(MultiGridEnv):
    """
    A randomly generated multigrid environment with configurable agents and objects.
    
    The environment creates a grid with:
    - Outer walls on all edges
    - Random internal walls and obstacles
    - Random objects (keys, doors, blocks, rocks, unsteady ground)
    - Specified number of human (yellow) and robot (grey) agents
    
    Doors and keys are paired: when a door is placed, a matching key is also placed.
    """
    
    def __init__(
        self,
        grid_size: int = 7,
        num_humans: int = 2,
        num_robots: int = 2,
        max_steps: int = 30,
        seed: int = None,
        wall_prob: float = 0.12,
        door_prob: float = 0.02,
        block_prob: float = 0.02,
        rock_prob: float = 0.02,
        unsteady_prob: float = 0.08,
        door_key_color: str = 'r'
    ):
        """
        Initialize the random multigrid environment.
        
        Args:
            grid_size: Size of the grid (including outer walls).
            num_humans: Number of human agents (yellow).
            num_robots: Number of robot agents (grey).
            max_steps: Maximum steps per episode.
            seed: Random seed for reproducibility.
            wall_prob: Probability of internal walls.
            door_prob: Probability of placing doors (also places matching key).
            block_prob: Probability of placing blocks.
            rock_prob: Probability of placing rocks.
            unsteady_prob: Probability of placing unsteady ground.
            door_key_color: Color code for doors and keys.
        """
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        self.wall_prob = wall_prob
        self.door_prob = door_prob
        self.block_prob = block_prob
        self.rock_prob = rock_prob
        self.unsteady_prob = unsteady_prob
        self.door_key_color = door_key_color
        self._seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Build the map string
        map_str = self._generate_random_map()
        
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
    
    def _generate_random_map(self) -> str:
        """
        Generate a random map string for the environment.
        
        Doors and keys are paired: when a door is placed, a matching key
        is also placed in a random available cell.
        """
        # Track which cells are available for agents
        available_cells = []
        
        # Track cells where we'll place keys (for doors placed)
        pending_keys = []
        
        # Generate the grid
        grid = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                # Outer walls
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')  # Grey wall
                else:
                    # Inner cells - randomly place objects
                    r = random.random()
                    cumulative = 0
                    
                    cumulative += self.wall_prob
                    if r < cumulative:
                        row.append('We')  # Grey wall
                        continue
                    
                    cumulative += self.rock_prob
                    if r < cumulative:
                        row.append('Ro')  # Rock
                        available_cells.append((x, y))  # Can push rocks
                        continue
                    
                    cumulative += self.door_prob
                    if r < cumulative:
                        row.append(f'C{self.door_key_color}')  # Closed door
                        pending_keys.append(self.door_key_color)
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')  # Block (pushable)
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.unsteady_prob
                    if r < cumulative:
                        row.append('Un')  # Unsteady ground
                        available_cells.append((x, y))
                        continue
                    
                    # Empty cell
                    row.append('..')
                    available_cells.append((x, y))
            
            grid.append(row)
        
        # Place keys for each door in random empty cells
        empty_cells_for_keys = [(x, y) for (x, y) in available_cells 
                                if grid[y][x] == '..']
        random.shuffle(empty_cells_for_keys)
        for i, key_color in enumerate(pending_keys):
            if i < len(empty_cells_for_keys):
                kx, ky = empty_cells_for_keys[i]
                grid[ky][kx] = f'K{key_color}'  # Place key
        
        # Ensure we have enough cells for agents
        num_agents = self.num_humans + self.num_robots
        
        # Find all empty cells
        empty_cells = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if grid[y][x] == '..':
                    empty_cells.append((x, y))
        
        # If not enough empty cells, clear some
        while len(empty_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid[y][x] not in ['..', 'Ay', 'Ae'] and (x, y) not in empty_cells:
                        grid[y][x] = '..'
                        empty_cells.append((x, y))
                        if len(empty_cells) >= num_agents:
                            break
                if len(empty_cells) >= num_agents:
                    break
        
        # Place agents randomly
        random.shuffle(empty_cells)
        agent_positions = empty_cells[:num_agents]
        
        # Place human agents (yellow)
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid[y][x] = 'Ay'  # Yellow agent
        
        # Place robot agents (grey)
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid[y][x] = 'Ae'  # Grey agent
        
        # Build map string
        return '\n'.join(' '.join(row) for row in grid)


class SmallGoalSampler(PossibleGoalSampler):
    """
    Goal sampler that samples small rectangle goals (at most 3x3).
    
    Uses rejection sampling to ensure goals are small enough for efficient learning.
    """
    
    MAX_REJECTION_ATTEMPTS = 1000
    
    def __init__(self, world_model, seed: int = None):
        super().__init__(world_model)
        self._rng = np.random.default_rng(seed)
        self._update_valid_range()
    
    def _update_valid_range(self):
        """Update valid coordinate ranges for goal placement."""
        env = self.world_model
        self._x_range = (1, env.width - 2)
        self._y_range = (1, env.height - 2)
    
    def set_world_model(self, world_model):
        """Update world model and refresh valid ranges."""
        self.world_model = world_model
        self._update_valid_range()
    
    def sample_rectangle(self):
        """Sample a small rectangle (at most 3x3) via rejection sampling."""
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range
        
        for _ in range(self.MAX_REJECTION_ATTEMPTS):
            x1 = self._rng.integers(x_min, x_max + 1)
            y1 = self._rng.integers(y_min, y_max + 1)
            x2 = self._rng.integers(x_min, x_max + 1)
            y2 = self._rng.integers(y_min, y_max + 1)
            
            if x2 < x1 or y2 < y1:
                continue
            if x2 - x1 > 2 or y2 - y1 > 2:
                continue
            
            return (x1, y1, x2, y2)
        
        # Fallback: point goal
        x = self._rng.integers(x_min, x_max + 1)
        y = self._rng.integers(y_min, y_max + 1)
        return (x, y, x, y)
    
    def sample(self, state, human_agent_index: int):
        """Sample a small rectangle goal with uniform weight."""
        x1, y1, x2, y2 = self.sample_rectangle()
        goal = ReachRectangleGoal(self.world_model, human_agent_index, (x1, y1, x2, y2))
        return goal, 1.0
class Phase2DemoEnv(MultiGridEnv):
    """
    A simple grid environment for Phase 2 demo (trivial mode only).
    For ensemble mode, use RandomMultigridEnv instead.
    """
    
    def __init__(self, max_steps: int = 10):
        if GRID_MAP is None:
            raise ValueError("Phase2DemoEnv requires GRID_MAP to be set (use trivial mode)")
        super().__init__(
            map=GRID_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')


def create_ensemble_env(seed: int = None) -> RandomMultigridEnv:
    """Create a random ensemble environment with the configured parameters."""
    return RandomMultigridEnv(
        grid_size=ENSEMBLE_GRID_SIZE,
        num_humans=ENSEMBLE_NUM_HUMANS,
        num_robots=ENSEMBLE_NUM_ROBOTS,
        max_steps=MAX_STEPS,
        seed=seed,
        wall_prob=WALL_PROBABILITY,
        door_prob=DOOR_PROBABILITY,
        block_prob=BLOCK_PROBABILITY,
        rock_prob=ROCK_PROBABILITY,
        unsteady_prob=UNSTEADY_GROUND_PROBABILITY,
        door_key_color=DOOR_KEY_COLOR
    )


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
    # Set all networks to eval mode to disable dropout during inference
    # This ensures consistent value predictions for the same state
    robot_q_network.eval()
    if networks is not None:
        networks.u_r.eval()
        networks.v_r.eval()
        networks.v_h_e.eval()
        networks.x_h.eval()
    
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
            # U_r can be computed either from network or directly from X_h
            u_r_val = 0.0
            v_r_val = 0.0
            
            # Check if we have a U_r network with encoder, otherwise compute from X_h
            if hasattr(networks.u_r, 'state_encoder') and networks.u_r.state_encoder is not None:
                state_encoder = networks.u_r.state_encoder
                # Use encode_state method (returns tuple of tensors with batch dim already)
                s_encoded = state_encoder.tensorize_state(state, env, device)
                # s_encoded already has batch dimension, no need to unsqueeze
                _, u_r = networks.u_r.forward(*s_encoded)
                u_r_val = u_r.item()
            else:
                # Compute U_r directly from X_h values
                x_h_vals = []
                for h in human_indices:
                    x_h = networks.x_h.encode_and_forward(state, env, h, device)
                    x_h_clamped = torch.clamp(x_h.squeeze(), min=1e-3, max=1.0)
                    x_h_vals.append(x_h_clamped)
                if x_h_vals:
                    x_h_tensor = torch.stack(x_h_vals)
                    xi = config.xi if config else 1.0
                    eta = config.eta if config else 1.1
                    y = (x_h_tensor ** (-xi)).mean()
                    u_r_val = -(y ** eta).item()
            
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
            beta_r = config.beta_r if config else 10.0
            robot_action = robot_q_network.sample_action(q_values, epsilon=0.0, beta_r=beta_r)
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
            # Use greedy action (epsilon=0) with final beta_r for concentrated policy
            beta_r = config.beta_r if config else 10.0
            robot_action = robot_q_network.sample_action(q_values, epsilon=0.0, beta_r=beta_r)
            
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

def main(
    quick_mode: bool = False,
    debug: bool = False,
    profile: bool = False,
    use_ensemble: bool = False,
    use_async: bool = False,
    save_networks_path: str = None,
    save_policy_path: str = None,
    restore_networks_path: str = None,
    use_policy_path: str = None,
    num_rollouts_override: int = None,
    save_video_path: str = None,
):
    """Run Phase 2 demo."""
    # Configure environment based on command line option
    configure_environment(use_ensemble)
    
    print("=" * 70)
    print("Phase 2 Robot Policy Learning Demo")
    print("Learning robot policy to maximize aggregate human power")
    print("Based on EMPO paper equations (4)-(9)")
    print("=" * 70)
    print()
    
    # Configuration - start with environment-based defaults
    if env_type == "trivial":
        num_training_steps = 10000
        num_rollouts = NUM_ROLLOUTS
        batch_size = 16
        x_h_batch_size = 32
        # Very small networks for trivial task
        hidden_dim = 16
        goal_feature_dim = 8
        agent_embedding_dim = 4
        print("[TRIVIAL ENV] Using minimal network sizes for simple task")
    elif env_type == "ensemble":
        # Ensemble needs more training due to varied layouts
        num_training_steps = 50000
        num_rollouts = NUM_ROLLOUTS
        batch_size = 32  # Reduced batch size for faster training
        x_h_batch_size = 64
        # Moderate networks - don't need to be huge for 7x7 grids
        hidden_dim = 64
        goal_feature_dim = 32
        agent_embedding_dim = 8
        print(f"[ENSEMBLE ENV] Random {ENSEMBLE_GRID_SIZE}x{ENSEMBLE_GRID_SIZE} grids with {ENSEMBLE_NUM_HUMANS} humans, {ENSEMBLE_NUM_ROBOTS} robots")
    else:
        num_training_steps = 50000
        num_rollouts = NUM_ROLLOUTS
        batch_size = 32
        x_h_batch_size = 64
        hidden_dim = 128
        goal_feature_dim = 64
        agent_embedding_dim = 16
    
    # Override with quick_mode settings
    if quick_mode:
        num_training_steps = 1000
        num_rollouts = 10
        # Use smaller batches and network for faster iteration in quick mode
        batch_size = 16
        x_h_batch_size = 32
        hidden_dim = 64  # Smaller network for faster forward passes
        goal_feature_dim = 32
        agent_embedding_dim = 8
        # Shorter warmup stages to fit within quick mode training
        # Default warmup is 3000+ steps, which won't complete in quick mode
        warmup_v_h_e_steps = 100  # ~10 episodes
        warmup_x_h_steps = 100   # ~10 episodes  
        warmup_u_r_steps = 0     # Skipped (u_r_use_network=False by default)
        warmup_q_r_steps = 100   # ~10 episodes
        beta_r_rampup_steps = 200  # ~20 episodes
        # Total warmup: 500 steps, leaving 500 steps for actual training
        print("[QUICK MODE] Running with reduced episodes, rollouts, batch sizes, network size, and warmup stages")
    else:
        # Use default warmup stages (each stage ~1000 steps)
        warmup_v_h_e_steps = 1000
        warmup_x_h_steps = 1000
        warmup_u_r_steps = 1000  # Will be set to 0 if u_r_use_network=False
        warmup_q_r_steps = 1000
        beta_r_rampup_steps = 2000
    
    # Print async mode status
    if use_async:
        print("[ASYNC MODE] Using actor-learner architecture for parallel training")
    
    # Override with profile settings
    if profile:
        print("[PROFILE MODE] Will profile training with torch.profiler")
        # Reduce training steps for profiling to get meaningful trace without waiting too long
        if not quick_mode:
            num_training_steps = min(num_training_steps, 2500)  # 50 episodes * 50 steps/episode
            print(f"  Reduced to {num_training_steps} training steps for profiling")
    
    device = 'cpu'
    
    # Create output directory (includes env_type in name)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', f'phase2_demo_{env_type}')
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    # Default save paths for networks and policy
    default_networks_path = os.path.join(output_dir, 'all_networks.pt')
    default_policy_path = os.path.join(output_dir, 'policy.pt')
    
    # Create environment
    print("Creating environment...")
    if env_type == "ensemble":
        env = create_ensemble_env(seed=42)  # Use fixed seed for reproducibility
        goal_sampler = SmallGoalSampler(env, seed=123)
        
        # Create world model factory for async training (ensemble mode)
        if use_async:
            env_creator = _EnsembleEnvCreator(
                grid_size=ENSEMBLE_GRID_SIZE,
                num_humans=ENSEMBLE_NUM_HUMANS,
                num_robots=ENSEMBLE_NUM_ROBOTS,
                max_steps=MAX_STEPS,
                wall_prob=WALL_PROBABILITY,
                door_prob=DOOR_PROBABILITY,
                block_prob=BLOCK_PROBABILITY,
                rock_prob=ROCK_PROBABILITY,
                unsteady_prob=UNSTEADY_GROUND_PROBABILITY,
                door_key_color=DOOR_KEY_COLOR
            )
            # Ensemble mode: create new env each episode
            world_model_factory = EnsembleWorldModelFactory(env_creator, episodes_per_env=1)
        else:
            world_model_factory = None
    else:
        env = Phase2DemoEnv(max_steps=MAX_STEPS)
        goal_sampler = goal_sampler_factory(env)
        
        # Create world model factory for async training (trivial mode)
        if use_async:
            # Trivial mode: use cached (same env for all episodes)
            world_model_factory = CachedWorldModelFactory(_create_trivial_env)
        else:
            world_model_factory = None
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
    
    # goal_sampler was already created above (ensemble uses SmallGoalSampler, trivial uses factory)
    # Both goal_sampler and human_policy are passed directly to the trainer
    # (no wrappers needed - trainer expects .sample() methods on these objects)
    
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
        epsilon_r_decay_steps=num_training_steps // 5,  # Decay over 20% of training
        lr_q_r=1e-4,
        lr_v_r=1e-4,
        lr_v_h_e=1e-3,  # faster since V_h^e is critical and the basis for other things
        lr_x_h=1e-4,
        lr_u_r=1e-4,
        buffer_size=10000,
        batch_size=batch_size,
        x_h_batch_size=x_h_batch_size,  # Larger batch for X_h to reduce high variance
        num_training_steps=num_training_steps,
        steps_per_episode=env.max_steps,
        training_steps_per_env_step=1.0,
        goal_resample_prob=0.1,
        v_h_e_target_update_interval=100,  # Standard target network update frequency
        # Warmup stage durations (each is duration in steps, not cumulative)
        warmup_v_h_e_steps=warmup_v_h_e_steps,
        warmup_x_h_steps=warmup_x_h_steps,
        warmup_u_r_steps=warmup_u_r_steps,
        warmup_q_r_steps=warmup_q_r_steps,
        beta_r_rampup_steps=beta_r_rampup_steps,
        # Async training mode (actor-learner architecture)
        async_training=use_async,
        num_actors=2 if use_async else 1,  # 2 actors for CPU testing
        async_min_buffer_size=100 if quick_mode else 500,  # Smaller for quick mode
    )
    
    # If using policy directly (no training), skip to rollouts
    if use_policy_path:
        print("=" * 70)
        print(f"Loading policy from: {use_policy_path}")
        print("Skipping training, running rollouts only")
        print("=" * 70)
        
        policy = MultiGridRobotPolicy(path=use_policy_path, device=device)
        robot_q_network = policy.q_network
        networks = None  # No networks for annotations
        trainer = None
        history = []
        elapsed = 0.0
    else:
        # Train Phase 2
        print("Training Phase 2 robot policy...")
        print(f"  Training steps: {config.num_training_steps:,}")
        print(f"  Environment steps per episode: {config.steps_per_episode}")
        print(f"  TensorBoard: {tensorboard_dir}")
        if restore_networks_path:
            print(f"  Restoring from: {restore_networks_path}")
        print()
        
        t0 = time.time()
        
        if profile:
            # Profile training with torch.profiler + TensorBoard integration
            from torch.profiler import profile as torch_profile, ProfilerActivity, schedule, tensorboard_trace_handler
            
            profiler_dir = os.path.join(output_dir, 'profiler')
            os.makedirs(profiler_dir, exist_ok=True)
            profiler_trace_path = os.path.join(output_dir, 'profiler_trace.json')
            
            print(f"  Profiler TensorBoard output: {profiler_dir}")
            print(f"  View with: tensorboard --logdir={profiler_dir}")
            print()
            
            # Schedule: skip first 2 episodes (wait), warm up for 2, actively profile 6, repeat
            # This captures steady-state behavior, not just initialization overhead
            with torch_profile(
                activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device != 'cpu' else []),
                schedule=schedule(wait=2, warmup=2, active=6, repeat=2),
                on_trace_ready=tensorboard_trace_handler(profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                robot_q_network, networks, history, trainer = train_multigrid_phase2(
                    world_model=env,
                    human_agent_indices=human_indices,
                    robot_agent_indices=robot_indices,
                    human_policy_prior=human_policy,
                    goal_sampler=goal_sampler,
                    config=config,
                    hidden_dim=hidden_dim,
                    goal_feature_dim=goal_feature_dim,
                    agent_embedding_dim=agent_embedding_dim,
                    device=device,
                    verbose=True,
                    debug=debug,
                    tensorboard_dir=tensorboard_dir,
                    profiler=prof,
                    restore_networks_path=restore_networks_path,
                    world_model_factory=world_model_factory,
                )
            
            # Print profiler summary
            print("\n" + "=" * 70)
            print("PROFILER RESULTS")
            print("=" * 70)
            print("\nTop 30 operations by CPU time:")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
            
            print("\nTop 20 operations by self CPU time (excluding children):")
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
            
            # Also export Chrome trace as backup
            prof.export_chrome_trace(profiler_trace_path)
            print(f"\nProfiler outputs:")
            print(f"  TensorBoard: {profiler_dir}")
            print(f"    View: tensorboard --logdir={profiler_dir}")
            print(f"    Then open: http://localhost:6006/#pytorch_profiler")
            print(f"  Chrome trace: {profiler_trace_path}")
            print(f"    View in Chrome: chrome://tracing")
            print(f"    Or use: https://ui.perfetto.dev/")
        else:
            robot_q_network, networks, history, trainer = train_multigrid_phase2(
                world_model=env,
                human_agent_indices=human_indices,
                robot_agent_indices=robot_indices,
                human_policy_prior=human_policy,
                goal_sampler=goal_sampler,
                config=config,
                hidden_dim=hidden_dim,
                goal_feature_dim=goal_feature_dim,
                agent_embedding_dim=agent_embedding_dim,
                device=device,
                verbose=True,
                debug=debug,
                tensorboard_dir=tensorboard_dir,
                restore_networks_path=restore_networks_path,
                world_model_factory=world_model_factory,
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
    
    # Save networks/policy after training (default paths unless overridden)
    # Always save by default to enable testing save/restore functionality
    if trainer is not None:
        # Determine save paths (use defaults if not specified)
        networks_save_path = save_networks_path if save_networks_path else default_networks_path
        policy_save_path = save_policy_path if save_policy_path else default_policy_path
        
        print(f"\nSaving all networks to: {networks_save_path}")
        trainer.save_all_networks(networks_save_path)
        
        print(f"Saving policy to: {policy_save_path}")
        trainer.save_policy(policy_save_path)
    
    # Generate rollout movie using env's built-in video recording
    # Override num_rollouts if specified
    if num_rollouts_override is not None:
        num_rollouts = num_rollouts_override
    
    print(f"\nGenerating {num_rollouts} rollouts with learned policy...")
    
    # Always load policy from disk for rollouts to test save/restore functionality
    # This verifies that the saved policy works correctly
    if use_policy_path:
        rollout_policy_path = use_policy_path
    else:
        # Use the policy we just saved (or the default path)
        rollout_policy_path = save_policy_path if save_policy_path else default_policy_path
    
    print(f"Loading policy from disk for rollouts: {rollout_policy_path}")
    policy = MultiGridRobotPolicy(path=rollout_policy_path, device=device)
    
    # Start video recording
    env.start_video_recording()
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        policy.reset(env)  # Set world model for episode
        steps = run_policy_rollout(
            env=env,
            robot_q_network=policy.q_network,  # Use policy's Q network
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
    # Use custom path if specified
    if save_video_path:
        movie_path = save_video_path
    else:
        movie_path = os.path.join(output_dir, 'phase2_robot_policy_demo.mp4')
    if os.path.exists(movie_path):
        os.remove(movie_path)
    env.save_video(movie_path, fps=MOVIE_FPS)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    if trainer is not None:
        print(f"  TensorBoard logs: {tensorboard_dir}")
        print(f"  View with: tensorboard --logdir={tensorboard_dir}")
    print(f"  Movie: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2 Robot Policy Learning Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    parser.add_argument('--ensemble', '-e', action='store_true',
                        help='Use random ensemble environment (7x7 grids, 2 humans, 2 robots)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable verbose debug output')
    parser.add_argument('--profile', '-p', action='store_true',
                        help='Profile training with torch.profiler (outputs trace.json)')
    parser.add_argument('--async', '-a', dest='use_async', action='store_true',
                        help='Use async actor-learner training (tests async mode on CPU)')
    
    # Save/restore options
    parser.add_argument('--save_networks', type=str, default=None, metavar='PATH',
                        help='Save all trained networks to PATH after training')
    parser.add_argument('--save_policy', type=str, default=None, metavar='PATH',
                        help='Save trained policy (Q_r network only) to PATH after training')
    parser.add_argument('--restore_networks', type=str, default=None, metavar='PATH',
                        help='Restore networks from PATH before training (skips warmup/rampup)')
    parser.add_argument('--use_policy', type=str, default=None, metavar='PATH',
                        help='Skip training and only run rollouts using policy from PATH')
    
    # Rollout options
    parser.add_argument('--rollouts', type=int, default=None, metavar='N',
                        help='Number of rollouts to generate (overrides default)')
    parser.add_argument('--save_video', type=str, default=None, metavar='PATH',
                        help='Path to save rollout video (default: outputs/phase2_demo_<env>/...mp4)')
    
    args = parser.parse_args()
    main(
        quick_mode=args.quick,
        debug=args.debug,
        profile=args.profile,
        use_ensemble=args.ensemble,
        use_async=args.use_async,
        save_networks_path=args.save_networks,
        save_policy_path=args.save_policy,
        restore_networks_path=args.restore_networks,
        use_policy_path=args.use_policy,
        num_rollouts_override=args.rollouts,
        save_video_path=args.save_video,
    )

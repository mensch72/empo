#!/usr/bin/env python3
"""
Phase 2 Robot Policy via Backward Induction Demo.

This script demonstrates computing the robot policy using exact backward induction
on the state DAG, rather than neural network training. This is computationally
exact but only feasible for small state spaces.

Uses either:
- The trivial world model (default): 4x6 grid with 1 human, 1 robot, and a rock
- A custom world from multigrid_worlds/ via --world parameter

The demo:
1. Computes human policy prior via backward induction
2. Computes robot policy via backward induction (Phase 2)
3. Records rollout movies with value annotations

Usage:
    python phase2_backward_induction.py           # Default trivial world, 5 max steps
    python phase2_backward_induction.py --steps 3 # Shorter horizon
    python phase2_backward_induction.py --world basic/two_agents  # Use custom world
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
import inspect
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
    MultiGridEnv, World, SmallActions
)
from empo.possible_goal import PossibleGoalGenerator, TabularGoalSampler
from empo.backward_induction import (
    compute_human_policy_prior, 
    compute_robot_policy,
    TabularRobotPolicy
)
from empo import backward_induction as backward_induction_pkg
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_specific_helpers.multigrid import ReachCellGoal


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
    Human/robot indices are auto-detected by color (yellow=human, grey=robot).
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
    # Agent indices are auto-detected via properties (yellow=human, grey=robot)
    return env


def create_env_from_world(world_path: str, max_steps: int = 5) -> MultiGridEnv:
    """
    Create an environment from a YAML world file.
    
    Args:
        world_path: Path relative to multigrid_worlds/, e.g., "basic/two_agents"
                   (.yaml extension is optional)
        max_steps: Maximum steps per episode (overrides config file value)
    
    Returns:
        MultiGridEnv loaded from the config file.
    """
    # Find the multigrid_worlds directory
    script_dir = os.path.dirname(__file__)
    worlds_dir = os.path.join(script_dir, '..', 'multigrid_worlds')
    
    # Add .yaml extension if not present
    if not world_path.endswith('.yaml') and not world_path.endswith('.yml'):
        world_path = world_path + '.yaml'
    
    config_path = os.path.join(worlds_dir, world_path)
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"World file not found: {config_path}")
    
    env = MultiGridEnv(
        config_file=config_path,
        max_steps=max_steps,
        partial_obs=False,
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

class CellGoalSampler:
    """
    A goal sampler that creates ReachCellGoal on-the-fly for any human.
    
    Samples goals uniformly from a list of cell coordinates, creating the
    goal object on-the-fly with the correct human_agent_index.
    """
    
    def __init__(self, env: MultiGridEnv, goal_cells: List[Tuple[int, int]]):
        self.env = env
        self.goal_cells = goal_cells
    
    def sample(self, state, human_agent_index: int):
        """Sample a random cell goal for the specified human."""
        import random
        cell = random.choice(self.goal_cells)
        goal = ReachCellGoal(self.env, human_agent_index, cell)
        return goal, 1.0  # Uniform weight


class TrivialGoalGenerator(PossibleGoalGenerator):
    """
    Goal generator for the trivial environment.
    
    Generates all possible cell goals for any human agent.
    
    Args:
        world_model: The environment.
        goal_weights: Optional dict mapping cell tuples to weights.
                     Default weight is 1.0 for cells not in the dict.
    """
    
    def __init__(self, world_model: MultiGridEnv,
                 goal_weights: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__(world_model)
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
    
    def get_sampler(self) -> CellGoalSampler:
        """Create a sampler that samples from this generator's goal cells."""
        return CellGoalSampler(self.env, self.goal_cells)


class GenericGoalGenerator(PossibleGoalGenerator):
    """
    Goal generator that discovers walkable cells automatically.
    
    Generates ReachCellGoal for each empty/walkable cell in the grid.
    
    Args:
        world_model: The environment.
        goal_weights: Optional dict mapping cell tuples to weights.
                     Default weight is 1.0 for cells not in the dict.
    """
    
    def __init__(self, world_model: MultiGridEnv,
                 goal_weights: Optional[Dict[Tuple[int, int], float]] = None):
        super().__init__(world_model)
        self.goal_weights = goal_weights or {}
        self.goal_cells = self._discover_walkable_cells()
    
    def _discover_walkable_cells(self) -> List[Tuple[int, int]]:
        """Find all cells that are empty or contain only agents."""
        walkable = []
        grid = self.env.grid
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = grid.get(x, y)
                # Cell is walkable if empty or contains only an agent
                if cell is None:
                    walkable.append((x, y))
                elif hasattr(cell, 'can_overlap') and cell.can_overlap():
                    walkable.append((x, y))
        
        # Also include agent starting positions
        for agent in self.env.agents:
            pos = tuple(agent.pos)
            if pos not in walkable:
                walkable.append(pos)
        
        return sorted(walkable)
    
    def generate(self, state, human_agent_index: int):
        """Generate all possible goals with weights."""
        for cell in self.goal_cells:
            goal = ReachCellGoal(self.env, human_agent_index, cell)
            weight = self.goal_weights.get(cell, 1.0)
            yield (goal, weight)
    
    def get_sampler(self) -> CellGoalSampler:
        """Create a sampler that samples from this generator's goal cells."""
        return CellGoalSampler(self.env, self.goal_cells)


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
    goal_sampler,
    goal_generator,
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
    # Get agent indices from env
    human_agent_indices = env.human_agent_indices
    robot_agent_indices = env.robot_agent_indices
    
    # Generate action names
    joint_action_names = get_joint_action_names(1)  # Single robot
    
    env.reset()
    steps_taken = 0
    
    # Sample initial goal for each human
    state = env.get_state()
    human_goals = {}
    for h_idx in human_agent_indices:
        human_goal, _ = goal_sampler.sample(state, h_idx)
        human_goals[h_idx] = human_goal
    
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
                for h_idx in human_agent_indices:
                    if h_idx in vh_state:
                        for goal, _ in goal_generator.generate(state, h_idx):
                            if goal in vh_state[h_idx]:
                                vh = vh_state[h_idx][goal]
                                x_h_vals.append(vh ** zeta)
                if x_h_vals:
                    x_h = sum(x_h_vals) / len(x_h_vals)  # Average over goals and humans
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
            # Format action names for all robots
            action_names = []
            for action_idx in action_profile:
                name = SINGLE_ACTION_NAMES[action_idx] if action_idx < len(SINGLE_ACTION_NAMES) else f"a{action_idx}"
                action_names.append(name)
            action_str = ', '.join(action_names)
            marker = ">" if selected_action is not None and action_profile == selected_action else " "
            lines.append(f"{marker}{action_str}: {prob:.3f}")
        
        return lines
    
    def get_robot_action(state):
        """Sample action from robot policy."""
        return robot_policy.sample(state)
    
    # Render initial frame
    robot_action = get_robot_action(state)
    selected_action = robot_action if robot_action else None
    annotation = compute_annotation_text(state, selected_action)
    env.render(mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE,
               annotation_text=annotation, annotation_panel_width=ANNOTATION_PANEL_WIDTH,
               annotation_font_size=ANNOTATION_FONT_SIZE, goal_overlays=human_goals)
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Get actions
        actions = [0] * len(env.agents)
        
        # Each human uses computed policy prior with their sampled goal
        for h_idx in human_agent_indices:
            human_goal = human_goals[h_idx]
            human_action_dist = human_policy_prior(state, h_idx, human_goal)
            if human_action_dist is not None:
                actions[h_idx] = np.random.choice(len(human_action_dist), p=human_action_dist)
        
        # Each robot uses computed policy
        robot_action = robot_policy.sample(state)
        if robot_action is not None:
            for i, r_idx in enumerate(robot_agent_indices):
                actions[r_idx] = robot_action[i]
        
        # Step environment
        _, _, done, _ = env.step(actions)
        steps_taken += 1
        
        # Render frame
        new_state = env.get_state()
        new_robot_action = get_robot_action(new_state)
        selected_action = new_robot_action if new_robot_action else None
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

def get_all_functions_from_module(module, _seen=None):
    """
    Recursively get all functions from a module and its submodules.
    
    Returns a list of (function_object, qualified_name) tuples.
    Tracks already seen functions to avoid duplicates from re-exports.
    """
    if _seen is None:
        _seen = set()
    
    functions = []
    
    # Recursively process submodules FIRST so we get functions from their source modules
    if hasattr(module, '__path__'):
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
            submodule_name = f"{module.__name__}.{modname}"
            try:
                submodule = __import__(submodule_name, fromlist=[modname])
                functions.extend(get_all_functions_from_module(submodule, _seen))
            except ImportError:
                # Some submodules may not be importable (for example, they can depend on
                # optional packages that are not installed). In that case we intentionally
                # skip them and continue discovering functions from the remaining submodules.
                continue
    
    # Get functions defined directly in this module (not re-exports)
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            # Only include functions actually defined in this specific module
            if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                func_id = id(obj)
                if func_id not in _seen:
                    _seen.add(func_id)
                    functions.append((obj, f"{obj.__module__}.{obj.__name__}"))
        elif inspect.isclass(obj):
            # Get methods from classes defined in this module
            if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if not method_name.startswith('_') or method_name in ('__init__', '__call__'):
                        func_id = id(method)
                        if func_id not in _seen:
                            _seen.add(func_id)
                            functions.append((method, f"{obj.__module__}.{obj.__name__}.{method_name}"))
    
    return functions


def main(
    max_steps: int = 5,
    num_rollouts: int = 20,
    parallel: bool = True,
    num_workers: Optional[int] = 4,
    beta_h: float = 10.0,
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
    world_path: Optional[str] = None,
    human_idx: Optional[int] = None,
    robot_idx: Optional[int] = None,
    profile: bool = False,
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
    
    # Set up line profiler if requested
    profiler = None
    if profile:
        try:
            from line_profiler import LineProfiler
            profiler = LineProfiler()
            
            # Get all functions from the backward_induction package
            functions_to_profile = get_all_functions_from_module(backward_induction_pkg)
            
            print(f"Line profiling enabled. Profiling {len(functions_to_profile)} functions:")
            for func, name in functions_to_profile:
                profiler.add_function(func)
                print(f"  - {name}")
            
            # Also profile HumanPolicyPrior methods (called heavily in Phase 2)
            hpp_methods_to_profile = [
                (TabularHumanPolicyPrior.__call__, 'TabularHumanPolicyPrior.__call__'),
                (TabularHumanPolicyPrior.profile_distribution, 'TabularHumanPolicyPrior.profile_distribution'),
                (TabularHumanPolicyPrior.profile_distribution_with_fixed_goal, 'TabularHumanPolicyPrior.profile_distribution_with_fixed_goal'),
                (TabularHumanPolicyPrior._profile_distribution_numpy, 'TabularHumanPolicyPrior._profile_distribution_numpy'),
                (TabularHumanPolicyPrior._to_probability_array, 'TabularHumanPolicyPrior._to_probability_array'),
            ]
            print(f"  Plus {len(hpp_methods_to_profile)} HumanPolicyPrior methods:")
            for func, name in hpp_methods_to_profile:
                profiler.add_function(func)
                print(f"  - {name}")
            print()
        except ImportError:
            print("WARNING: line_profiler not installed. Install with: pip install line_profiler")
            print("Continuing without profiling...")
            print()
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'phase2_backward_induction')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    if world_path is not None:
        print(f"  Loading world from: {world_path}")
        env = create_env_from_world(world_path, max_steps=max_steps)
    else:
        env = create_trivial_env(max_steps=max_steps)
    env.reset()
    
    # Override agent indices from CLI if specified
    if human_idx is not None:
        env.human_agent_indices = [human_idx]
    if robot_idx is not None:
        env.robot_agent_indices = [robot_idx]
    
    # Print agent info
    for h_idx in env.human_agent_indices:
        print(f"  Human (agent {h_idx}): pos={tuple(env.agents[h_idx].pos)}, color={env.agents[h_idx].color}")
    for r_idx in env.robot_agent_indices:
        print(f"  Robot (agent {r_idx}): pos={tuple(env.agents[r_idx].pos)}, color={env.agents[r_idx].color}")
    
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Max steps: {max_steps}")
    print(f"  Humans: {len(env.human_agent_indices)}, Robots: {len(env.robot_agent_indices)}")
    print()
    
    # Create goal generator for backward induction
    print("Creating goal generator...")
    
    # First check if env has a goal generator from config (possible_goals in YAML)
    if hasattr(env, 'possible_goal_generator') and env.possible_goal_generator is not None:
        goal_generator = env.possible_goal_generator
        goal_sampler = env.possible_goal_sampler
        print(f"  Using goal generator from config file")
        print(f"  Goals: {len(goal_generator.goal_coords)} goals from config")
        print(f"  Goal coords: {goal_generator.goal_coords}")
    elif world_path is not None:
        # Use generic goal generator for custom worlds without possible_goals
        goal_generator = GenericGoalGenerator(env, goal_weights=goal_weights)
        # Create goal sampler for rollouts - creates goals on-the-fly with correct human_agent_index
        goal_sampler = goal_generator.get_sampler()
        print(f"  Goals: {len(goal_generator.goal_cells)} possible cell goals (auto-discovered)")
        print(f"  Goal cells: {goal_generator.goal_cells}")
        if goal_weights:
            print(f"  Custom goal weights: {goal_weights}")
    else:
        # Use trivial goal generator for the default world
        goal_generator = TrivialGoalGenerator(env, goal_weights=goal_weights)
        # Create goal sampler for rollouts - creates goals on-the-fly with correct human_agent_index
        goal_sampler = goal_generator.get_sampler()
        print(f"  Goals: {len(goal_generator.goal_cells)} possible cell goals")
        print(f"  Goal cells: {goal_generator.goal_cells}")
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
    if profiler:
        profiler.enable()
    human_policy_prior, Vh_phase1 = compute_human_policy_prior(
        world_model=env,
        human_agent_indices=env.human_agent_indices,
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
        terminal_Vr=terminal_Vr,
        parallel=parallel,
        num_workers=num_workers,
        return_values=True,
    )
    if profiler:
        profiler.disable()
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
            action_names = [SINGLE_ACTION_NAMES[a] if a < len(SINGLE_ACTION_NAMES) else f"a{a}" for a in action_profile]
            print(f"    {', '.join(action_names)}: {prob:.3f}")
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
            print(f"  {i+1}. State={state}")
            print(f"      Goal={goal}, Vh1={vh1:.4f}, Vh2={vh2:.4f}, Δ={diff:+.4f}")
        
        # Show entries where Phase 2 is worse (if any)
        if num_worse > 0:
            print()
            print(f"Entries where Phase2 < Phase1 (robot policy hurts this goal):")
            sorted_worst = sorted(differences, key=lambda x: x[5])
            for i, (state, agent_idx, goal, vh1, vh2, diff) in enumerate(sorted_worst[:min(5, num_worse)]):
                print(f"  {i+1}. State={state}")
                print(f"      Goal={goal}, Vh1={vh1:.4f}, Vh2={vh2:.4f}, Δ={diff:+.4f}")
        
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
            goal_generator=goal_generator,
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
    
    # Write profiling results if enabled
    if profiler:
        profile_path = os.path.join(output_dir, 'line_profile_results.txt')
        print("=" * 70)
        print("Writing line profiling results...")
        print("=" * 70)
        
        with open(profile_path, 'w') as f:
            profiler.print_stats(stream=f)
        
        print(f"  Profile saved to: {os.path.abspath(profile_path)}")
        print()
        
        # Also print summary to console
        print("Top functions by total time (see file for line-by-line details):")
        import io
        stats_buffer = io.StringIO()
        profiler.print_stats(stream=stats_buffer)
        stats_text = stats_buffer.getvalue()
        
        # Extract and show summary (first few function headers)
        lines = stats_text.split('\n')
        in_header = False
        func_count = 0
        for line in lines:
            if 'Total time:' in line:
                in_header = True
                func_count += 1
                if func_count <= 5:  # Show top 5 function summaries
                    print(f"  {line.strip()}")
            elif in_header and 'Function:' in line:
                if func_count <= 5:
                    print(f"  {line.strip()}")
                    print()
                in_header = False
        print()
    
    print("=" * 70)
    print("Demo completed!")
    print(f"  Movie saved to: {os.path.abspath(movie_path)}")
    if profiler:
        print(f"  Profile saved to: {os.path.abspath(profile_path)}")
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
    parser.add_argument('--world', type=str, default=None,
                        help='Path to world YAML file relative to multigrid_worlds/, '
                             'e.g., "basic/two_agents" (default: use trivial world)')
    parser.add_argument('--human-idx', type=int, default=None,
                        help='Index of human agent (default: auto-detect)')
    parser.add_argument('--robot-idx', type=int, default=None,
                        help='Index of robot agent (default: auto-detect)')
    
    # Computation options
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Use parallel computation')
    parser.add_argument('--workers', '-w', type=int, default=4,
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
    parser.add_argument('--profile', action='store_true',
                        help='Enable line profiling of backward_induction package '
                             '(requires line_profiler: pip install line_profiler)')
    
    args = parser.parse_args()
    
    # Build goal_weights dict from command line args (only for trivial world)
    goal_weights = None
    if args.world is None:
        goal_weights = {
            (1, 1): args.weight11,
            (2, 1): args.weight21,
            (3, 1): args.weight31,
            (4, 1): args.weight41,
            (2, 2): args.weight22,
        }
        # Remove entries with default weight 1.0 to keep output clean
        goal_weights = {k: v for k, v in goal_weights.items() if v != 1.0}
        if not goal_weights:
            goal_weights = None
    
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
        goal_weights=goal_weights,
        seed=args.seed,
        output_dir=args.output_dir,
        movie_fps=args.fps,
        save_video_path=args.save_video,
        world_path=args.world,
        human_idx=args.human_idx,
        robot_idx=args.robot_idx,
        profile=args.profile,
    )

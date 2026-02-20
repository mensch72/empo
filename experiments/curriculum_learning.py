
import os
import sys
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.generator import generate_rock_maze_env
from empo.world_specific_helpers.multigrid import ReachCellGoal
from empo.possible_goal import TabularGoalSampler
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.learning_based.multigrid import PathDistanceCalculator
from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.multigrid.phase2 import train_multigrid_phase2
from empo.learning_based.multigrid.phase2.robot_policy import (
    MultiGridMultiStepExplorationPolicy,
    MultiGridRobotPolicy
)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Curriculum parameters
    initial_distance: float = 0.0  # Start easy
    distance_step: float = 0.1  # How much to adjust distance
    p_min: float = 0.3  # Lower bound of target zone
    p_max: float = 0.7  # Upper bound of target zone
    ema_momentum: float = 0.9  # Exponential moving average for smoothing

    # Maze parameters
    maze_width: int = 11
    maze_height: int = 11
    num_trapped_agents: int = 1
    max_steps_per_episode: int = 50
    curriculum_maze_seed: int = 42  # Fixed seed for curriculum maze

    # Training parameters
    num_batches: int = 100
    episodes_per_batch: int = 32  # Episodes to evaluate success rate
    training_steps_per_batch: int = 20000  # Training steps per batch 

    # Random maze mixing
    random_maze_prob: float = 0.0  # Probability of using random maze

    # Evaluation
    eval_frequency: int = 50  # Evaluate every N batches
    num_eval_mazes: int = 5
    eval_distances: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])

    # Phase2 training config
    hidden_dim: int = 64
    goal_feature_dim: int = 32
    agent_embedding_dim: int = 8
    gamma_r: float = 0.99
    gamma_h: float = 0.99
    lr_q_r: float = 1e-4
    lr_v_h_e: float = 1e-3

    # Output
    output_dir: str = 'outputs/curriculum'
    save_frequency: int = 50  # Save checkpoint every N batches

    # Rollout parameters
    num_rollouts: int = 10
    rollout_distances: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    movie_fps: int = 3


class CurriculumTrainer:
    """Adaptive curriculum trainer for rock-pushing task."""

    def __init__(self, config: CurriculumConfig, device: str = 'cpu', verbose: bool = True, debug: bool = False):
        self.config = config
        self.device = device
        self.verbose = verbose
        self.debug = debug

        # Curriculum state
        self.robot_distance = config.initial_distance
        self.distance_history = []
        self.success_history = []
        self.smoothed_success_rate = None

        # Training state
        self.batch_idx = 0
        self.total_episodes = 0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.tensorboard_dir = os.path.join(config.output_dir, 'tensorboard')

        # Initialize trainer (will be created on first batch)
        self.trainer = None
        self.phase2_config = None
        self.robot_q_network = None
        self.networks = None

        # Evaluation mazes
        self.eval_maze_seeds = [100, 200, 300, 400, 500]

        if self.verbose:
            print("=" * 70)
            print("Curriculum Learning for Rock-Pushing Task")
            print("=" * 70)
            print(f"Initial distance: {self.robot_distance:.2f}")
            print(f"Target success rate: [{config.p_min:.2f}, {config.p_max:.2f}]")
            print(f"Distance step size: {config.distance_step:.2f}")
            print(f"Number of batches: {config.num_batches}")
            print(f"Episodes per batch: {config.episodes_per_batch}")
            print(f"Output: {config.output_dir}")
            print("=" * 70)
            print()
    
    def generate_maze(self, use_curriculum_seed: bool = True) -> MultiGridEnv:
        """Generate a rock maze environment."""
        seed = self.config.curriculum_maze_seed if use_curriculum_seed else None

        env = generate_rock_maze_env(
            width=self.config.maze_width,
            height=self.config.maze_height,
            num_trapped_agents=self.config.num_trapped_agents,
            robot_distance=self.robot_distance,
            max_steps=self.config.max_steps_per_episode,
            seed=seed
        )
        return env
    
    def evaluate_policy_on_env(self, env: MultiGridEnv, num_episodes: int = 10) -> float:
        """
        Evaluate current policy on environment.

        Returns:
            (success_rate, avg_steps)
        """
        if self.robot_q_network is None:
            if self.verbose:
                print("    Warning: No policy to evaluate (robot_q_network is None)")
            return 0.0, 0.0

        # Set network to eval mode
        self.robot_q_network.eval()

        # Get agent indices
        human_indices = [i for i, a in enumerate(env.agents) if a.color == 'yellow']
        robot_indices = [i for i, a in enumerate(env.agents) if a.color == 'grey']

        # Create goal sampler
        goal_cells = []
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    goal_cells.append((x, y))
        goals = [ReachCellGoal(env, 0, cell) for cell in goal_cells[:10]]
        goal_sampler = TabularGoalSampler(goals)

        # Create human policy
        path_calc = PathDistanceCalculator(
            grid_height=env.height,
            grid_width=env.width,
            world_model=env
        )
        human_policy = HeuristicPotentialPolicy(
            world_model=env,
            human_agent_indices=human_indices,
            path_calculator=path_calc,
            beta=1000.0
        )

        successes = []
        total_steps = 0

        for ep_idx in range(num_episodes):
            env.reset()
            # Store initial rock positions
            initial_rock_positions = []
            for x in range(env.width):
                for y in range(env.height):
                    cell = env.grid.get(x, y)
                    if cell is not None and hasattr(cell, 'type') and cell.type == 'rock':
                        initial_rock_positions.append((x, y))

            state = env.get_state()
            # Track robot initial position for debugging
            initial_robot_pos = None
            if self.debug and ep_idx == 0 and robot_indices:
                initial_robot_pos = tuple(env.agents[robot_indices[0]].pos)

            # Sample goals for humans
            human_goals = {}
            for h in human_indices:
                goal, _ = goal_sampler.sample(state, h)
                human_goals[h] = goal

            for step in range(env.max_steps):
                state = env.get_state()

                # Get actions
                actions = [0] * len(env.agents)

                # Humans use heuristic policy
                for h in human_indices:
                    actions[h] = human_policy.sample(state, h, human_goals[h])

                # Robots use learned policy (greedy)
                with torch.no_grad():
                    q_values = self.robot_q_network.forward(state, env, self.device)
                    # Use greedy action (very high beta for deterministic policy)
                    robot_action = self.robot_q_network.sample_action(q_values, beta_r=1000.0)

                    # Assign robot actions
                    for i, r in enumerate(robot_indices):
                        if i < len(robot_action):
                            actions[r] = robot_action[i]

                # Step environment
                _, _, done, _ = env.step(actions)

                if done:
                    break

            success = self.is_human_freed(env, initial_rock_positions, debug=self.debug and ep_idx == 0)
            successes.append(success)

        success_rate = sum(successes) / len(successes) if successes else 0.0
        return success_rate
    
    def is_human_freed(
        self,
        env: MultiGridEnv,
        initial_rock_positions: List[Tuple[int, int]],
        debug: bool = False
    ) -> bool:
        """
        Check if human was freed (rock moved from initial position).

        Args:
            env: Environment after episode
            initial_rock_positions: List of initial rock positions
            debug: Print debug information

        Returns:
            True if any rock moved from its initial position
        """
        if not initial_rock_positions:
            return False

        # Get current rock positions from grid
        current_rock_positions = []
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and hasattr(cell, 'type') and cell.type == 'rock':
                    current_rock_positions.append((x, y))

        # Check if any rock moved (initial position not in current positions)
        moved = any(pos not in current_rock_positions for pos in initial_rock_positions)

        if debug and moved:
            print(f"    [DEBUG] Rocks: {initial_rock_positions} → {current_rock_positions}")

        return moved
    
    def adjust_distance(self, success_rate: float) -> float:
        """
        Adjust robot_distance based on success rate.

        Returns:
            (new_distance, adjustment_reason)
        """
        # Update smoothed success rate
        if self.smoothed_success_rate is None:
            self.smoothed_success_rate = success_rate
        else:
            self.smoothed_success_rate = (
                self.config.ema_momentum * self.smoothed_success_rate +
                (1 - self.config.ema_momentum) * success_rate
            )

        # Adjust based on smoothed success rate
        if self.smoothed_success_rate < self.config.p_min:
            # Too hard, make easier (decrease distance)
            self.robot_distance = max(0.0, self.robot_distance - self.config.distance_step)
        elif self.smoothed_success_rate > self.config.p_max:
            # Too easy, make harder (increase distance)
            self.robot_distance = min(1.0, self.robot_distance + self.config.distance_step)

        # Record history
        self.distance_history.append(self.robot_distance)

        return self.robot_distance
    
    def train_batch(self, batch_idx: int):
        """Train for one batch of episodes."""
        # Decide whether to use curriculum maze or random maze
        use_curriculum = random.random() >= self.config.random_maze_prob

        # Generate environment
        env = self.generate_maze(use_curriculum_seed=use_curriculum)
        env.reset()

        # Get agent indices
        human_indices = [i for i, a in enumerate(env.agents) if a.color == 'yellow']
        robot_indices = [i for i, a in enumerate(env.agents) if a.color == 'grey']

        # Store initial rock positions
        initial_rock_positions = []
        if hasattr(env, 'maze_info') and 'rock_positions' in env.maze_info:
            initial_rock_positions = env.maze_info['rock_positions']

        # Create goal sampler (goals at all walkable cells)
        goal_cells = []
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    goal_cells.append((x, y))
        goals = [ReachCellGoal(env, 0, cell) for cell in goal_cells[:10]]  # Limit to 10 goals
        goal_sampler = TabularGoalSampler(goals)

        # Create path calculator and human policy
        path_calc = PathDistanceCalculator(
            grid_height=env.height,
            grid_width=env.width,
            world_model=env
        )
        human_policy = HeuristicPotentialPolicy(
            world_model=env,
            human_agent_indices=human_indices,
            path_calculator=path_calc,
            beta=1000.0
        )

        # Create Phase2 config if first batch
        if self.phase2_config is None:
            self.phase2_config = Phase2Config(
                gamma_r=self.config.gamma_r,
                gamma_h=self.config.gamma_h,
                lr_q_r=self.config.lr_q_r,
                lr_v_h_e=self.config.lr_v_h_e,
                buffer_size=10000,
                batch_size=self.config.episodes_per_batch,
                num_training_steps=self.config.training_steps_per_batch,
                steps_per_episode=self.config.max_steps_per_episode,
                training_steps_per_env_step=0.1,
            )
        else:
            # Update num_training_steps for this batch
            self.phase2_config.num_training_steps = self.config.training_steps_per_batch

        # Create exploration policies
        robot_exploration_policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=robot_indices,
            sequence_probs={'still': 0.05, 'forward': 0.50, 'left_forward': 0.18,
                          'right_forward': 0.18, 'back_forward': 0.09},
            expected_k={'still': 1.0, 'forward': 2.0, 'left_forward': 1.5,
                       'right_forward': 1.5, 'back_forward': 1.5},
        )

        human_exploration_policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=human_indices,
            sequence_probs={'still': 0.05, 'forward': 0.50, 'left_forward': 0.18,
                          'right_forward': 0.18, 'back_forward': 0.09},
            expected_k={'still': 1.0, 'forward': 2.0, 'left_forward': 1.5,
                       'right_forward': 1.5, 'back_forward': 1.5},
        )

        # Train on this batch
        # Each batch trains from scratch on the new environment
        restore_path = None
        if self.trainer is not None and batch_idx > 0:
            # Save previous networks to restore (warm start for next batch)
            temp_path = os.path.join(self.config.output_dir, 'temp_networks.pt')
            self.trainer.save_all_networks(temp_path)
            restore_path = temp_path

        # Train on current environment
        self.robot_q_network, self.networks, _, self.trainer = train_multigrid_phase2(
            world_model=env,
            human_agent_indices=human_indices,
            robot_agent_indices=robot_indices,
            human_policy_prior=human_policy,
            goal_sampler=goal_sampler,
            config=self.phase2_config,
            hidden_dim=self.config.hidden_dim,
            goal_feature_dim=self.config.goal_feature_dim,
            agent_embedding_dim=self.config.agent_embedding_dim,
            device=self.device,
            verbose=False,
            tensorboard_dir=self.tensorboard_dir,
            robot_exploration_policy=robot_exploration_policy,
            human_exploration_policy=human_exploration_policy,
            restore_networks_path=restore_path,  # Warm start from previous batch
        )

        # Evaluate success rate on the current environment
        success_rate = self.evaluate_policy_on_env(
            env,
            num_episodes=self.config.episodes_per_batch
        )

        self.success_history.append(success_rate)
        self.total_episodes += self.config.episodes_per_batch

        return success_rate, self.config.episodes_per_batch
    
    def train(self):
        """Main training loop."""
        pbar = tqdm(range(self.config.num_batches))
        for batch_idx in pbar:
            self.batch_idx = batch_idx

            # Train one batch
            try:
                success_rate, _ = self.train_batch(batch_idx)
            except Exception as e:
                tqdm.write(f"Error during training: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                break

            # Adjust distance based on success rate
            self.adjust_distance(success_rate)

                # Update description to show current batch being trained
            pbar.set_description_str(f"Batch {batch_idx+1}/{self.config.num_batches} | d={self.robot_distance:.2f}, succ={self.smoothed_success_rate:.0%}")
            pbar.refresh()

            # Log to tensorboard if available
            if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
                global_step = (batch_idx + 1) * self.config.training_steps_per_batch
                self.trainer.writer.add_scalar('curriculum/robot_distance', self.robot_distance, global_step)
                self.trainer.writer.add_scalar('curriculum/success_rate_raw', success_rate, global_step)
                self.trainer.writer.add_scalar('curriculum/success_rate_smoothed', self.smoothed_success_rate, global_step)

            # Evaluate periodically
            if (batch_idx + 1) % self.config.eval_frequency == 0:
                self.evaluate(batch_idx)

            # Save checkpoint periodically
            if (batch_idx + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(batch_idx)

        if self.verbose:
            print("\n" + "=" * 70)
            print("Curriculum Training Completed!")
            print(f"Final distance: {self.robot_distance:.2f}")
            print(f"Total episodes: {self.total_episodes}")
            print("\nCurriculum Progression:")
            print(f"  Initial distance: {self.config.initial_distance:.2f}")
            print(f"  Final distance: {self.robot_distance:.2f}")
            print(f"  Distance range: [{min(self.distance_history):.2f}, {max(self.distance_history):.2f}]")
            print(f"  Final success rate: {self.smoothed_success_rate:.3f}")
            print("=" * 70)

        # Save curriculum history
        history_path = os.path.join(self.config.output_dir, 'curriculum_history.txt')
        with open(history_path, 'w') as f:
            f.write("Batch,Distance,SuccessRate\n")
            for i, (dist, succ) in enumerate(zip(self.distance_history, self.success_history)):
                f.write(f"{i+1},{dist:.4f},{succ:.4f}\n")
        if self.verbose:
            print(f"\nCurriculum history saved to: {history_path}")

        # Generate rollouts after training
        self.generate_rollouts()

    def evaluate(self, batch_idx: int):
        """Evaluate on fixed test mazes at various distances."""
        if self.robot_q_network is None:
            return

        if self.verbose:
            tqdm.write(f"\n  Evaluation @ batch {batch_idx + 1}:")

        # Evaluate on fixed test mazes at various distances
        results = []
        for dist in self.config.eval_distances:
            success_rates = []

            # Test on multiple fixed mazes
            for seed in self.eval_maze_seeds[:self.config.num_eval_mazes]:
                env = generate_rock_maze_env(
                    width=self.config.maze_width,
                    height=self.config.maze_height,
                    num_trapped_agents=self.config.num_trapped_agents,
                    robot_distance=dist,
                    max_steps=self.config.max_steps_per_episode,
                    seed=seed
                )

                success_rate = self.evaluate_policy_on_env(env, num_episodes=5)
                success_rates.append(success_rate)

            mean_success = np.mean(success_rates)
            results.append(f"d={dist:.1f}: {mean_success:.0%}")

        if self.verbose:
            tqdm.write(f"    {' | '.join(results)}")

    def save_checkpoint(self, batch_idx: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f'checkpoint_batch_{batch_idx + 1}.pt'
        )

        if self.trainer is not None:
            self.trainer.save_all_networks(checkpoint_path)
            if self.verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

    def run_policy_rollout(
        self,
        env: MultiGridEnv,
        policy: MultiGridRobotPolicy,
        human_policy: HeuristicPotentialPolicy,
        goal_sampler: TabularGoalSampler,
        human_indices: List[int],
        robot_indices: List[int],
    ) -> Tuple[int, bool]:
        """
        Run a single rollout with the learned policy.

        Returns:
            (steps_taken, success)
        """
        env.reset()
        policy.reset(env)

        # Store initial rock positions by scanning grid
        initial_rock_positions = []
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and hasattr(cell, 'type') and cell.type == 'rock':
                    initial_rock_positions.append((x, y))

        steps_taken = 0
        state = env.get_state()

        # Sample initial goals for humans
        human_goals = {}
        for h in human_indices:
            goal, _ = goal_sampler.sample(state, h)
            human_goals[h] = goal

        # Render initial frame
        env.render(mode='rgb_array', highlight=False, tile_size=96, goal_overlays=human_goals)

        for step in range(env.max_steps):
            state = env.get_state()

            # Get actions
            actions = [0] * len(env.agents)

            # Humans use heuristic policy
            for h in human_indices:
                actions[h] = human_policy.sample(state, h, human_goals[h])

            # Robots use learned policy
            with torch.no_grad():
                q_values = policy.q_network.forward(state, env, self.device)
                robot_action = policy.q_network.sample_action(q_values, beta_r=1000.0)

                for i, r in enumerate(robot_indices):
                    if i < len(robot_action):
                        actions[r] = robot_action[i]

            # Step environment
            _, _, done, _ = env.step(actions)
            steps_taken += 1

            # Render frame
            env.render(mode='rgb_array', highlight=False, tile_size=96, goal_overlays=human_goals)

            if done:
                break

        # Check if human was freed
        success = self.is_human_freed(env, initial_rock_positions)

        return steps_taken, success

    def generate_rollouts(self):
        """Generate rollout videos at various distances."""
        if self.trainer is None:
            print("No trained policy available for rollouts")
            return

        if self.verbose:
            print("\n" + "=" * 70)
            print("Generating Rollouts")
            print("=" * 70)

        # Save policy
        policy_path = os.path.join(self.config.output_dir, 'policy.pt')
        self.trainer.save_policy(policy_path)
        if self.verbose:
            print(f"Policy saved to: {policy_path}")

        # Load policy from disk to test save/restore
        policy = MultiGridRobotPolicy(path=policy_path, device=self.device)

        # Generate rollouts at different distances
        for distance in tqdm(self.config.rollout_distances, desc="Generating rollouts",
                            disable=not self.verbose):

            # Create environment at this distance
            env = generate_rock_maze_env(
                width=self.config.maze_width,
                height=self.config.maze_height,
                num_trapped_agents=self.config.num_trapped_agents,
                robot_distance=distance,
                max_steps=self.config.max_steps_per_episode,
                seed=self.config.curriculum_maze_seed
            )

            # Get agent indices
            human_indices = [i for i, a in enumerate(env.agents) if a.color == 'yellow']
            robot_indices = [i for i, a in enumerate(env.agents) if a.color == 'grey']

            # Create goal sampler
            goal_cells = []
            for x in range(env.width):
                for y in range(env.height):
                    cell = env.grid.get(x, y)
                    if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                        goal_cells.append((x, y))
            goals = [ReachCellGoal(env, 0, cell) for cell in goal_cells[:10]]
            goal_sampler = TabularGoalSampler(goals)

            # Create human policy
            path_calc = PathDistanceCalculator(
                grid_height=env.height,
                grid_width=env.width,
                world_model=env
            )
            human_policy = HeuristicPotentialPolicy(
                world_model=env,
                human_agent_indices=human_indices,
                path_calculator=path_calc,
                beta=1000.0
            )

            # Start video recording
            env.start_video_recording()

            # Run rollouts
            total_steps = 0
            total_success = 0

            for rollout_idx in range(self.config.num_rollouts):
                steps, success = self.run_policy_rollout(
                    env=env,
                    policy=policy,
                    human_policy=human_policy,
                    goal_sampler=goal_sampler,
                    human_indices=human_indices,
                    robot_indices=robot_indices,
                )
                total_steps += steps
                if success:
                    total_success += 1

            # Save video
            video_path = os.path.join(
                self.config.output_dir,
                f'rollouts_d{distance:.1f}.mp4'
            )
            if os.path.exists(video_path):
                os.remove(video_path)
            env.save_video(video_path, fps=self.config.movie_fps)

            # Print stats
            avg_steps = total_steps / self.config.num_rollouts
            success_rate = total_success / self.config.num_rollouts
            if self.verbose:
                tqdm.write(f"  d={distance:.1f}: {success_rate:.0%} success, "
                          f"{avg_steps:.0f} avg steps → {os.path.basename(video_path)}")

        if self.verbose:
            print("\n" + "=" * 70)
            print("Rollouts completed!")
            print("=" * 70)

def main(
    quick_mode: bool = False,
    num_batches: int = None,
    random_maze_prob: float = 0.0,
    seed: int = 42,
    output_dir: str = None,
    num_rollouts: int = None,
    debug: bool = False,
):
    """Main entry point."""
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create config
    config = CurriculumConfig()

    # Apply overrides
    if quick_mode:
        config.num_batches = 10
        config.episodes_per_batch = 8
        config.training_steps_per_batch = 2000  # Increased - need more steps to learn
        config.eval_frequency = 3
        config.save_frequency = 5
        config.num_rollouts = 3
        print("[QUICK MODE] Reduced settings for testing")

    if num_batches is not None:
        config.num_batches = num_batches

    config.random_maze_prob = random_maze_prob

    if output_dir is not None:
        config.output_dir = output_dir

    if num_rollouts is not None:
        config.num_rollouts = num_rollouts

    # Create trainer
    device = 'cpu'
    trainer = CurriculumTrainer(config, device=device, verbose=True, debug=debug)

    # Train
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum Learning for Rock-Pushing')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick test mode with fewer batches')
    parser.add_argument('--batches', type=int, default=None,
                        help='Number of training batches')
    parser.add_argument('--mix-random', type=float, default=0.0, dest='random_maze_prob',
                        help='Probability of using random maze (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--rollouts', type=int, default=None,
                        help='Number of rollouts to generate after training')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug output (shows Q-values, positions, etc.)')

    args = parser.parse_args()

    main(
        quick_mode=args.quick,
        num_batches=args.batches,
        random_maze_prob=args.random_maze_prob,
        seed=args.seed,
        output_dir=args.output_dir,
        num_rollouts=args.rollouts,
        debug=args.debug,
    )

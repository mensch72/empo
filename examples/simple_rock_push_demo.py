#!/usr/bin/env python3
"""
Simple Rock Push Demo - Minimal Control Button Learning Example

This script demonstrates the multiplicative robot shaping and button-aware
exploration features in a minimal environment.

Scenario:
- A narrow 3x8 corridor with one control button and one rock
- Robot (grey) is pre-programmed to: walk to button, program "forward", return, wait
- Human (yellow) must learn to: step out of the way, wait until robot has programmed the button and returned, go to button, toggle button twice to make robot push rock, then reach rock's initial position

Map layout (after prequel):
  We We CB We We We We We
  We .. Ay Ae Ro .. .. We
  We We We We We We We We

Training features demonstrated:
- robot_shaping_exponent: Multiplicative potential Φ = Φ_human * (0.5 * |Φ_robot|^a)
- button_toggle_bias: State-dependent exploration biasing toggle near buttons
"""

import sys
import os
import random

from gym_multigrid.multigrid import MultiGridEnv, World, Actions
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.nn_based.multigrid import train_multigrid_neural_policy_prior

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

# Map: Button at (2,0), Human at (2,1), Robot at (3,1) facing east, Rock at (4,1)
INITIAL_MAP = """
We We CB We We We We We
We .. Ay Ae Ro .. .. We
We We We We We We We We
"""


class SimpleRockPushEnv(MultiGridEnv):
    """
    Minimal environment for demonstrating button-controlled rock pushing.
    
    Button at (2, 0) starts unprogrammed.
    Robot can push rocks. Human must use button to steer robot after robot programs it.
    
    Robot's fixed policy (executed via step_robot()):
    1. Wait until human moves out of the way (not at position 2,1)
    2. Turn north, move to button, toggle to program, do 'forward' action
    3. Return to starting position (3,1) facing east
    4. Wait for button presses (stay still)
    """
    
    def __init__(self, max_steps: int = 50):
        super().__init__(
            map=INITIAL_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=Actions
        )
        self.robot_phase = 'wait_for_human'  # Phases: wait_for_human, go_to_button, program, return, done
        self.robot_step_count = 0
    
    def _gen_grid(self, width, height):
        """Generate grid - button starts unprogrammed."""
        super()._gen_grid(width, height)
        
        # Find robot index and enable rock pushing
        for i, agent in enumerate(self.agents):
            if agent.color == 'grey':
                agent.can_push_rocks = True
                break
    
    def get_robot_action(self, human_pos):
        """
        Get the robot's next action based on its fixed policy.
        
        Robot policy:
        1. Wait until human is not blocking (not at 2,1)
        2. Turn left (to face north), move forward to (3,0), turn left (face west),
           move forward to (2,0), toggle button, do 'forward' to program it
        3. Turn around and return to (3,1) facing east
        4. Stay still forever (wait for button triggers)
        """
        robot = None
        robot_idx = None
        for i, agent in enumerate(self.agents):
            if agent.color == 'grey':
                robot = agent
                robot_idx = i
                break
        
        if robot is None:
            return Actions.still
        
        robot_pos = tuple(robot.pos)
        robot_dir = robot.dir  # 0=right, 1=down, 2=left, 3=up
        
        if self.robot_phase == 'wait_for_human':
            # Wait until human moves away from (2,1)
            if human_pos != (2, 1):
                self.robot_phase = 'go_to_button'
                self.robot_step_count = 0
            return Actions.still
        
        elif self.robot_phase == 'go_to_button':
            # Sequence: turn left (face north), forward, turn left (face west), forward
            # Robot starts at (3,1) facing east (dir=0)
            self.robot_step_count += 1
            if self.robot_step_count == 1:
                return Actions.left  # Now facing north (dir=3)
            elif self.robot_step_count == 2:
                return Actions.forward  # Move to (3,0)
            elif self.robot_step_count == 3:
                return Actions.left  # Now facing west (dir=2)
            elif self.robot_step_count == 4:
                return Actions.forward  # Move to (2,0)
            elif self.robot_step_count == 5:
                self.robot_phase = 'program'
                self.robot_step_count = 0
                return Actions.toggle  # Enter programming mode
        
        elif self.robot_phase == 'program':
            # Program button with 'forward' action
            self.robot_phase = 'return'
            self.robot_step_count = 0
            return Actions.forward  # Button is now programmed with 'forward'
        
        elif self.robot_phase == 'return':
            # Return to (3,1) facing east
            # From (2,0) facing west: turn around, forward, turn right, forward
            self.robot_step_count += 1
            if self.robot_step_count == 1:
                return Actions.left  # Face south
            elif self.robot_step_count == 2:
                return Actions.left  # Face east
            elif self.robot_step_count == 3:
                return Actions.forward  # Move to (3,0)
            elif self.robot_step_count == 4:
                return Actions.right  # Face south
            elif self.robot_step_count == 5:
                return Actions.forward  # Move to (3,1)
            elif self.robot_step_count == 6:
                return Actions.left  # Face east
            else:
                self.robot_phase = 'done'
                return Actions.still
        
        else:  # done
            return Actions.still


class HumanAtPositionGoal(PossibleGoal):
    """Goal: Human reaches a specific position."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = tuple(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if human is at target position."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            if len(agent_state) >= 2:
                pos_x, pos_y = int(agent_state[0]), int(agent_state[1])
                if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                    return 1
        return 0
    
    def __str__(self):
        return f"HumanAt({self.target_pos[0]},{self.target_pos[1]})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, HumanAtPositionGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                self.target_pos == other.target_pos)


class SingleGoalSampler(PossibleGoalSampler):
    """Sampler that always returns the same goal."""
    
    def __init__(self, world_model, human_idx: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_idx = human_idx
        self.target_pos = target_pos
    
    def sample(self, state, human_agent_index: int) -> tuple:
        """Always sample the same goal: human reaches initial rock position."""
        goal = HumanAtPositionGoal(self.world_model, self.human_idx, self.target_pos)
        return goal, 1.0


def create_movie(frames, output_path, fps=2):
    """Create a GIF from frames."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        pil_frames = []
        for frame in frames:
            if isinstance(frame, tuple):
                title, img = frame
            else:
                title, img = None, frame
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(img)
            if title:
                ax.set_title(title, fontsize=14)
            ax.axis('off')
            
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)
        
        if pil_frames:
            duration = int(1000 / fps)
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0
            )
            print(f"Saved movie to {output_path}")
            return True
    except ImportError as e:
        print(f"Could not create movie: {e}")
        return False
    return False


def train_human_policy():
    """Train the human to use button to steer robot and reach rock position."""
    print("=" * 60)
    print("Training Human Policy")
    print("=" * 60)
    
    env = SimpleRockPushEnv(max_steps=50)
    env.reset()
    
    # Find agent indices
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    print(f"Human agent index: {human_idx}")
    print(f"Robot agent index: {robot_idx}")
    print(f"Goal: Human reaches (4, 1) - initial rock position")
    print()
    
    # Target position: where rock starts (4, 1)
    target_pos = (4, 1)
    
    # Create goal sampler
    goal_sampler = SingleGoalSampler(env, human_idx, target_pos)
    
    # Train with multiplicative robot shaping and button-aware exploration
    print("Training with:")
    print("  - robot_shaping_exponent=0.5 (multiplicative potential)")
    print("  - button_toggle_bias=0.5 (biased exploration near buttons)")
    print()
    
    human_policy = train_multigrid_neural_policy_prior(
        env=env,
        goal_sampler=goal_sampler,
        human_agent_indices=[human_idx],
        steps_per_episode=30,
        num_episodes=1000,
        beta=100.0,
        gamma=0.95,
        epsilon=0.3,
        robot_shaping_exponent=0.5,
        button_toggle_bias=0.5,
        verbose=True
    )
    
    return human_policy, env, human_idx, robot_idx, target_pos


def demonstrate_learned_policy(human_policy, env, human_idx, robot_idx, target_pos):
    """Show the learned policy in action."""
    print()
    print("=" * 60)
    print("Demonstrating Learned Policy")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    env.reset()
    
    # Set goal
    goal = HumanAtPositionGoal(env, human_idx, target_pos)
    
    frames = []
    img = env.render(mode='rgb_array')
    frames.append((f"Step 0: Start | Goal: reach (4,1)", img))
    
    action_names = {
        Actions.still: 'still',
        Actions.left: 'left', 
        Actions.right: 'right',
        Actions.forward: 'forward',
        Actions.toggle: 'toggle'
    }
    
    max_steps = 30
    for step in range(max_steps):
        # Get action from learned policy for human
        state = env.get_state()
        human_action = human_policy.sample(state, human_idx, goal)
        
        # Get robot action from its fixed policy
        human_pos = tuple(env.agents[human_idx].pos)
        robot_action = env.get_robot_action(human_pos)
        
        actions = [Actions.still] * len(env.agents)
        actions[human_idx] = human_action
        actions[robot_idx] = robot_action
        
        obs, rewards, done, info = env.step(actions)
        
        human_pos = tuple(env.agents[human_idx].pos)
        robot_pos = tuple(env.agents[robot_idx].pos)
        
        img = env.render(mode='rgb_array')
        title = f"Step {step+1}: Human {action_names.get(human_action, '?')} | H:{human_pos} R:{robot_pos}"
        frames.append((title, img))
        
        print(f"Step {step+1}: Human={action_names.get(human_action, '?')}, "
              f"Human@{human_pos}, Robot@{robot_pos}")
        
        # Check if goal achieved
        state = env.get_state()
        if goal.is_achieved(state):
            print(f"\n*** Goal achieved at step {step+1}! ***")
            frames.append((f"Step {step+1}: GOAL ACHIEVED!", img))
            break
        
        if done:
            print("Episode ended")
            break
    else:
        print("\nMax steps reached without achieving goal")
    
    # Save movie
    movie_path = os.path.join(OUTPUT_DIR, 'simple_rock_push_demo.gif')
    create_movie(frames, movie_path, fps=2)
    
    return frames


def main():
    """Main entry point."""
    print()
    print("Simple Rock Push Demo")
    print("=" * 60)
    print("Environment: 3x8 corridor with 1 button and 1 rock")
    print("Human goal: Reach position (4,1) - initial rock position")
    print("Strategy: Use button to make robot push rock forward")
    print()
    
    # Train
    human_policy, env, human_idx, robot_idx, target_pos = train_human_policy()
    
    # Demonstrate
    demonstrate_learned_policy(human_policy, env, human_idx, robot_idx, target_pos)
    
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()

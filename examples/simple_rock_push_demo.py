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

import os

from gym_multigrid.multigrid import MultiGridEnv, World, Actions
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.learning_based.multigrid import train_multigrid_neural_policy_prior

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

# Map layout (3 rows x 8 cols):
# - Button at (2, 0) - in top wall, accessible from (2,1) facing north
# - Human at (2, 1)
# - Robot at (3, 1) facing east
# - Rock at (4, 1)
INITIAL_MAP = """
We We CB We We We We We
We .. Ay Ae Ro .. .. We
We We We We We We We We
"""


class SimpleRockPushEnv(MultiGridEnv):
    """
    Minimal environment for demonstrating button-controlled rock pushing.
    
    Button at (2, 0) starts unprogrammed (embedded in top wall).
    Robot can push rocks. Human must use button to steer robot after robot programs it.
    
    Robot's fixed policy (executed via step_robot()):
    1. Wait until human moves out of the way (not at position 2,1)
    2. Turn north, move to button, toggle to program, do 'forward' action
    3. Return to starting position (3,1) facing east
    4. Wait for button presses (stay still)
    """
    
    def __init__(self, max_steps: int = 1000):
        super().__init__(
            map=INITIAL_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=Actions,
            orientations=['e', 'e']  # Human faces east, Robot faces east
        )
        self.robot_phase = 'wait_for_human'  # Phases: wait_for_human, turn_west, move_to_button_cell, turn_north, program, turn_east, return_to_start, done
    
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
        
        Robot policy (direction-aware):
        1. Wait until human is not blocking (not at 2,1)
        2. Navigate to (2,1) facing north toward button at (2,0)
        3. Toggle button, do 'forward' to program it
        4. Return to (3,1) facing east
        5. Stay still forever (wait for button triggers)
        
        Map layout:
          We We CB We We We We We   <- Button at (2,0)
          We .. Ay Ae Ro .. .. We   <- Human(2,1), Robot(3,1), Rock(4,1)
          We We We We We We We We
        
        Directions: 0=east, 1=south, 2=west, 3=north
        """
        robot = None
        for i, agent in enumerate(self.agents):
            if agent.color == 'grey':
                robot = agent
                break
        
        if robot is None:
            return Actions.still
        
        robot_pos = tuple(robot.pos)
        robot_dir = robot.dir  # 0=east, 1=south, 2=west, 3=north
        
        # Helper to turn toward a target direction
        def turn_toward(current_dir, target_dir):
            """Return action to turn from current_dir toward target_dir, or None if aligned."""
            if current_dir == target_dir:
                return None
            # Calculate shortest turn
            diff = (target_dir - current_dir) % 4
            if diff == 1 or diff == -3:
                return Actions.right
            else:
                return Actions.left
        
        if self.robot_phase == 'wait_for_human':
            # Wait until human moves away from (2,1)
            if human_pos != (2, 1):
                self.robot_phase = 'turn_west'
            return Actions.still
        
        elif self.robot_phase == 'turn_west':
            # Turn to face west (dir=2) to move toward button position
            action = turn_toward(robot_dir, 2)  # 2 = west
            if action is None:
                self.robot_phase = 'move_to_button_cell'
            return action if action else Actions.still
        
        elif self.robot_phase == 'move_to_button_cell':
            # Move west from (3,1) to (2,1)
            if robot_pos == (2, 1):
                self.robot_phase = 'turn_north'
            else:
                return Actions.forward
            return Actions.still
        
        elif self.robot_phase == 'turn_north':
            # Turn to face north (dir=3) toward button at (2,0)
            action = turn_toward(robot_dir, 3)  # 3 = north
            if action is None:
                self.robot_phase = 'program'
                return Actions.toggle  # Enter programming mode
            return action
        
        elif self.robot_phase == 'program':
            # Program button with 'forward' action
            self.robot_phase = 'turn_east'
            return Actions.forward  # Button now programmed with 'forward'
        
        elif self.robot_phase == 'turn_east':
            # Turn to face east (dir=0) to return
            action = turn_toward(robot_dir, 0)  # 0 = east
            if action is None:
                self.robot_phase = 'return_to_start'
            return action if action else Actions.still
        
        elif self.robot_phase == 'return_to_start':
            # Move east from (2,1) to (3,1)
            if robot_pos == (3, 1):
                self.robot_phase = 'done'
            else:
                return Actions.forward
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


def demonstrate_handcrafted_policies(env, human_idx, robot_idx, target_pos):
    """
    Show the robot policy working with a handcrafted human policy.
    
    Human policy: Step to the side (east) to let robot pass, wait, then proceed to button.
    
    Initial state (with orientations=['e', 'e']):
    - Human at (2,1) facing east (dir=0)
    - Robot at (3,1) facing east (dir=0)
    """
    print()
    print("=" * 60)
    print("Demonstrating Handcrafted Policies")
    print("=" * 60)
    print("Human: Steps aside, waits for robot, then uses button")
    print("Robot: Goes to button, programs it, returns")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    env.reset()
    env.robot_phase = 'wait_for_human'
    
    # Start video recording
    env.start_video_recording()
    
    action_names = {
        Actions.still: 'still',
        Actions.left: 'left', 
        Actions.right: 'right',
        Actions.forward: 'forward',
        Actions.toggle: 'toggle'
    }
    
    # Helper to turn toward a target direction
    # Directions: 0=east, 1=south, 2=west, 3=north
    def turn_toward(current_dir, target_dir):
        """Return action to turn from current_dir toward target_dir, or None if aligned."""
        if current_dir == target_dir:
            return None
        diff = (target_dir - current_dir) % 4
        if diff == 1 or diff == -3:
            return Actions.right
        else:
            return Actions.left
    
    # Capture initial frame
    env.render(mode='rgb_array')
    
    # Human policy phases (direction-aware)
    # Human starts at (2,1) facing east (dir=0)
    # Robot is at (3,1) - human must step WEST to (1,1) to get out of robot's way
    human_phase = 'step_aside'
    
    max_steps = 50
    for step in range(max_steps):
        human_pos = tuple(env.agents[human_idx].pos)
        human_dir = env.agents[human_idx].dir
        robot_pos = tuple(env.agents[robot_idx].pos)
        robot_phase = env.robot_phase
        
        # Handcrafted human policy (direction-aware)
        if human_phase == 'step_aside':
            # Move west to (1,1) to let robot pass to button
            if human_pos == (1, 1):
                human_phase = 'wait_for_robot'
                human_action = Actions.still
            elif human_dir != 2:  # Not facing west
                human_action = turn_toward(human_dir, 2)  # Turn to face west
            else:
                human_action = Actions.forward  # Move west
        elif human_phase == 'wait_for_robot':
            # Wait until robot has programmed button and returned
            if robot_phase == 'done':
                human_phase = 'go_to_button'
            human_action = Actions.still
        elif human_phase == 'go_to_button':
            # Go east to button position (2,1)
            if human_pos == (2, 1):
                human_phase = 'face_button'
                human_action = Actions.still
            elif human_dir != 0:  # Not facing east
                human_action = turn_toward(human_dir, 0)  # Turn to face east
            else:
                human_action = Actions.forward  # Move east to (2,1)
        elif human_phase == 'face_button':
            # Turn north to face button at (2,0)
            if human_dir == 3:  # Facing north
                human_phase = 'toggle_button_1'
                human_action = Actions.toggle  # First toggle
            else:
                human_action = turn_toward(human_dir, 3)  # Turn to face north
        elif human_phase == 'toggle_button_1':
            human_phase = 'toggle_button_2'
            human_action = Actions.toggle  # Second toggle
        elif human_phase == 'toggle_button_2':
            human_phase = 'go_to_goal'
            human_action = Actions.still
        elif human_phase == 'go_to_goal':
            # Go east to goal at (4,1)
            if human_pos == target_pos:
                human_phase = 'done'
                human_action = Actions.still
            elif human_dir != 0:  # Not facing east
                human_action = turn_toward(human_dir, 0)  # Turn to face east
            else:
                human_action = Actions.forward  # Move east
        else:
            human_action = Actions.still
        
        # Get robot action from its fixed policy
        robot_action = env.get_robot_action(human_pos)
        
        actions = [Actions.still] * len(env.agents)
        actions[human_idx] = human_action
        actions[robot_idx] = robot_action
        
        obs, rewards, done, info = env.step(actions)
        
        human_pos = tuple(env.agents[human_idx].pos)
        robot_pos = tuple(env.agents[robot_idx].pos)
        
        # Capture frame
        env.render(mode='rgb_array')
        
        print(f"Step {step+1}: Human={action_names.get(human_action, '?')}@{human_pos}, "
              f"Robot={action_names.get(robot_action, '?')}@{robot_pos}, "
              f"RobotPhase={env.robot_phase}")
        
        # Check if goal achieved
        if human_pos == target_pos:
            print(f"\n*** Goal achieved at step {step+1}! ***")
            break
        
        if done:
            print("Episode ended")
            break
    
    # Save video
    movie_path = os.path.join(OUTPUT_DIR, 'simple_rock_push_handcrafted.mp4')
    env.save_video(movie_path, fps=2)
    print(f"\nSaved handcrafted demo video to: {movie_path}")


def train_human_policy():
    """Train the human to use button to steer robot and reach rock position."""
    print("=" * 60)
    print("Training Human Policy")
    print("=" * 60)
    
    env = SimpleRockPushEnv(max_steps=1000)
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
        steps_per_episode=100,
        num_episodes=10000,
        beta=100.0,
        gamma=0.95,
        epsilon=0.7,
        robot_shaping_exponent=0.5,
        button_toggle_bias=0.25,
        verbose=True
    )
    
    return human_policy, env, human_idx, robot_idx, target_pos


def demonstrate_learned_policy(human_policy, env, human_idx, robot_idx, target_pos):
    """Show the learned policy in action using video recording."""
    print()
    print("=" * 60)
    print("Demonstrating Learned Policy")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    env.reset()
    
    # Start video recording
    env.start_video_recording()
    
    # Set goal
    goal = HumanAtPositionGoal(env, human_idx, target_pos)
    
    action_names = {
        Actions.still: 'still',
        Actions.left: 'left', 
        Actions.right: 'right',
        Actions.forward: 'forward',
        Actions.toggle: 'toggle'
    }
    
    # Capture initial frame
    env.render(mode='rgb_array')
    
    max_steps = 1000
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
        
        # Capture frame
        env.render(mode='rgb_array')
        
        print(f"Step {step+1}: Human={action_names.get(human_action, '?')}, "
              f"Human@{human_pos}, Robot@{robot_pos}")
        
        # Check if goal achieved
        state = env.get_state()
        if goal.is_achieved(state):
            print(f"\n*** Goal achieved at step {step+1}! ***")
            break
        
        if done:
            print("Episode ended")
            break
    else:
        print("\nMax steps reached without achieving goal")
    
    # Save video using the new package method
    movie_path = os.path.join(OUTPUT_DIR, 'simple_rock_push_demo.mp4')
    env.save_video(movie_path, fps=2)


def main():
    """Main entry point."""
    print()
    print("Simple Rock Push Demo")
    print("=" * 60)
    print("Environment: 3x8 corridor with 1 button and 1 rock")
    print("Human goal: Reach position (4,1) - initial rock position")
    print("Strategy: Use button to make robot push rock forward")
    print()
    
    # First create environment and find agent indices
    env = SimpleRockPushEnv(max_steps=1000)
    env.reset()
    
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    target_pos = (4, 1)
    
    # Demonstrate handcrafted policies first (before training)
    demonstrate_handcrafted_policies(env, human_idx, robot_idx, target_pos)
    
    # Train
    human_policy, env, human_idx, robot_idx, target_pos = train_human_policy()
    
    # Demonstrate learned policy
    demonstrate_learned_policy(human_policy, env, human_idx, robot_idx, target_pos)
    
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()

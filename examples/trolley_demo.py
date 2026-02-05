"""
Demo: Trolley problem - robot crushes one human to free three others.

The grey robot pushes the rock onto a yellow human, killing them,
then moves out of the way so the three freed humans can escape to the chamber.
"""

import sys
sys.path.insert(0, 'vendor/multigrid')

from gym_multigrid.multigrid import MultiGridEnv

# Load the trolley world
env = MultiGridEnv(config_file='multigrid_worlds/berkin_challenges/trolley.yaml')
env.reset()

print("=== Trolley Problem Demo ===")
print(f"rocks_can_kill: {env.rocks_can_kill}")
print(f"Number of agents: {len(env.agents)}")
print()

# Action codes: 0=still, 1=left, 2=right, 3=forward
STILL, LEFT, RIGHT, FORWARD = 0, 1, 2, 3

# Direction codes: 0=east/right, 1=south/down, 2=west/left, 3=north/up
EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3

# Find all agents and their roles
robot_idx = None
trapped_indices = []  # Humans at y=1 (trapped at top)
victim_idx = None     # Human at (1,3)

for i, agent in enumerate(env.agents):
    print(f"Agent {i}: color={agent.color}, pos={tuple(agent.pos)}, can_push_rocks={agent.can_push_rocks}")
    if agent.can_push_rocks:
        robot_idx = i
    elif tuple(agent.pos) == (1, 3):
        victim_idx = i
    elif agent.pos[1] == 1:  # y=1, trapped at top
        trapped_indices.append(i)

print(f"\nRobot: agent {robot_idx}")
print(f"Victim: agent {victim_idx}")
print(f"Trapped humans: agents {trapped_indices}")

num_agents = len(env.agents)
frames = [env.render(mode='rgb_array')]

def step_and_record(actions):
    """Take a step and record frame."""
    env.step(actions)
    frames.append(env.render(mode='rgb_array'))

def make_actions(**agent_actions):
    """Create action array. Usage: make_actions(robot=FORWARD, human1=LEFT)"""
    actions = [STILL] * num_agents
    for name, action in agent_actions.items():
        if name == 'robot':
            actions[robot_idx] = action
        elif name.startswith('trapped'):
            idx = int(name.replace('trapped', ''))
            actions[trapped_indices[idx]] = action
    return actions

def turn_agent_to(agent_idx, target_dir):
    """Turn an agent to face target direction."""
    agent = env.agents[agent_idx]
    while agent.dir != target_dir:
        # Calculate shortest turn direction
        # diff = how many RIGHT turns needed
        diff = (target_dir - agent.dir) % 4
        if diff == 0:
            break  # Already facing target
        elif diff == 1 or diff == 2:
            action = RIGHT  # Turn right (clockwise)
        else:  # diff == 3
            action = LEFT   # Turn left (counter-clockwise, 1 turn instead of 3)
        actions = [STILL] * num_agents
        actions[agent_idx] = action
        step_and_record(actions)

# ============ PHASE 1: Robot crushes the victim ============
print("\n--- Phase 1: Robot pushes rock onto victim ---")

# Turn robot to face west (toward the rock)
turn_agent_to(robot_idx, WEST)
print(f"Robot facing: {env.agents[robot_idx].dir}")

# Push the rock (crushes victim)
print("Pushing rock...")
step_and_record(make_actions(robot=FORWARD))

print(f"Victim terminated: {env.agents[victim_idx].terminated}")

# Pause to show the result
step_and_record(make_actions())
step_and_record(make_actions())

# ============ PHASE 2: Robot moves out of the way ============
print("\n--- Phase 2: Robot moves to chamber ---")

# Robot is now at (2,3), needs to go down to chamber
# Turn to face south
turn_agent_to(robot_idx, SOUTH)

# Move down through corridor to chamber
for _ in range(3):  # Move to (2,4), (2,5), (2,6)
    step_and_record(make_actions(robot=FORWARD))
    print(f"Robot at: {tuple(env.agents[robot_idx].pos)}")

# Move robot to the side
turn_agent_to(robot_idx, EAST)
step_and_record(make_actions(robot=FORWARD))
print(f"Robot at: {tuple(env.agents[robot_idx].pos)}")

# Pause
step_and_record(make_actions())

# ============ PHASE 3: Trapped humans escape to chamber ============
print("\n--- Phase 3: Trapped humans escape ---")

# The trapped humans are at (1,1), (2,1), (3,1)
# They need to go through corridor at x=2 to reach the chamber

# Identify humans by position
left_human = None    # At (1,1)
middle_human = None  # At (2,1)
right_human = None   # At (3,1)

for i in trapped_indices:
    x = env.agents[i].pos[0]
    if x == 1:
        left_human = i
    elif x == 2:
        middle_human = i
    elif x == 3:
        right_human = i

print(f"Left human: agent {left_human}, Middle: agent {middle_human}, Right: agent {right_human}")

# Step 1: Middle human moves down first to make room
turn_agent_to(middle_human, SOUTH)
for _ in range(3):  # Move to (2,2), (2,3), (2,4)
    actions = [STILL] * num_agents
    actions[middle_human] = FORWARD
    step_and_record(actions)
print(f"Middle human at: {tuple(env.agents[middle_human].pos)}")

# Step 2: Left human moves to (2,1)
turn_agent_to(left_human, EAST)
actions = [STILL] * num_agents
actions[left_human] = FORWARD
step_and_record(actions)
print(f"Left human at: {tuple(env.agents[left_human].pos)}")

# Step 3: Left human continues down, making room for right human
turn_agent_to(left_human, SOUTH)
for _ in range(2):  # Move down a bit
    actions = [STILL] * num_agents
    actions[left_human] = FORWARD
    step_and_record(actions)
print(f"Left human at: {tuple(env.agents[left_human].pos)}")

# Step 4: Right human moves to (2,1)
turn_agent_to(right_human, WEST)
actions = [STILL] * num_agents
actions[right_human] = FORWARD
step_and_record(actions)
print(f"Right human at: {tuple(env.agents[right_human].pos)}")

# Step 5: Right human continues down
turn_agent_to(right_human, SOUTH)

# Step 6: All continue moving down to chamber
for _ in range(4):
    actions = [STILL] * num_agents
    for h in trapped_indices:
        if not env.agents[h].terminated:
            actions[h] = FORWARD
    step_and_record(actions)

# Final positions
print("\n--- Final State ---")
for i, agent in enumerate(env.agents):
    status = "TERMINATED" if agent.terminated else "alive"
    pos = tuple(agent.pos) if agent.pos is not None else "N/A"
    print(f"Agent {i} ({agent.color}): {pos} - {status}")

# Add pause frames at the end
for _ in range(3):
    step_and_record(make_actions())

# Save the video
print(f"\nSaving video with {len(frames)} frames...")
import imageio.v3 as iio

output_path = 'outputs/trolley_demo.mp4'
iio.imwrite(output_path, frames, fps=3)
print(f"Video saved to: {output_path}")

gif_path = 'outputs/trolley_demo.gif'
iio.imwrite(gif_path, frames, duration=333, loop=0)
print(f"GIF saved to: {gif_path}")

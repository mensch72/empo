# MultiGrid Gridworld Reference

This document provides a complete reference for the MultiGrid gridworld environment, describing all cell types, object types, actions, and their consequences.

## Overview

MultiGrid is a 2D grid-based multi-agent environment where agents navigate a grid, interact with objects, and complete tasks. The environment can be fully or partially observable.

## Grid Cells

Each cell in the grid can contain:
- **Empty space** (None) - Agents can move through
- **One object** - Wall, floor, door, key, ball, box, goal, lava, switch, object goal, block, or rock
- **One agent** - When an agent occupies a cell

## Object Types

### 1. Wall
- **Type**: `wall`
- **Color**: Grey (default) or other colors
- **Properties**:
  - Cannot be passed through by agents
  - Cannot be seen through (blocks vision)
  - Cannot be picked up or moved
  - Immovable and permanent

### 2. Floor
- **Type**: `floor`
- **Color**: Blue (default) or other colors
- **Properties**:
  - Can be overlapped (agents can walk on it)
  - Purely decorative, no gameplay effect
  - Used to mark specific areas or paths

### 3. Door
- **Type**: `door`
- **Color**: Red, green, blue, purple, yellow, or grey
- **States**: Open, closed, or locked
- **Properties**:
  - **Closed door**: Cannot be passed through or seen through
  - **Open door**: Can be passed through and seen through
  - **Locked door**: Cannot be opened without the matching colored key
- **Interaction**:
  - **Toggle action** on unlocked door: Opens/closes the door
  - **Toggle action** on locked door with matching key: Unlocks and opens the door (door becomes unlocked AND open)
  - **Toggle action** on locked door without key: No effect
- **Key matching**: A key can only unlock a door of the same color
- **Key reusability**: Keys are held by the agent and can be reused multiple times (not consumed)
- **Re-locking**: **Once unlocked, a door cannot be locked again**. It can only be opened/closed via toggle.

### 4. Key
- **Type**: `key`
- **Color**: Red, green, blue, purple, yellow, or grey
- **Properties**:
  - Can be picked up by agents
  - Used to unlock doors of matching color
  - Reusable (not consumed when unlocking)
  - Agent can carry one object at a time
- **Interaction**:
  - **Pickup action**: Agent picks up the key (if not already carrying something)
  - **Drop action**: Agent drops the key in front of them

### 5. Ball
- **Type**: `ball`
- **Color**: Red, green, blue, purple, yellow, or grey
- **Properties**:
  - Can be picked up by agents
  - Used in collection tasks (e.g., collect balls of matching color)
  - Cannot be moved by pushing/pulling, only by pickup and drop
  - Each ball has an associated reward value
- **Interaction**:
  - **Pickup action**: Agent picks up the ball
  - **Drop action**: Agent drops the ball in front of them

### 6. Box
- **Type**: `box`
- **Color**: Red, green, blue, purple, yellow, or grey
- **Properties**:
  - Can be picked up by agents (treated like any pickable object)
  - Can contain another object inside
  - **Cannot be pushed or pulled** - only moved via pickup and drop
  - **Cannot push multiple boxes** - boxes are picked up, not pushed
- **Interaction**:
  - **Pickup action**: Agent picks up the entire box
  - **Toggle action**: Opens the box, replacing it with its contents
  - **Drop action**: Agent drops the box in front of them
- **Movement**: Boxes are **not** pushable. They must be picked up and carried.

### 7. Goal
- **Type**: `goal`
- **Color**: Red, green, blue, purple, yellow, or grey
- **Properties**:
  - Can be overlapped (agents can walk through)
  - Triggers task completion when agent moves onto it
  - Each goal has an associated reward value
- **Interaction**:
  - **Forward action** onto goal: Agent receives reward, episode may end

### 8. Object Goal
- **Type**: `objgoal`
- **Color**: Red, green, blue, purple, yellow, or grey
- **Properties**:
  - Target location for delivering specific object types
  - Cannot be overlapped (blocks movement)
  - Rewards agent for delivering correct object type
- **Interaction**:
  - Used in tasks where agents must deliver specific objects to specific locations

### 9. Lava
- **Type**: `lava`
- **Color**: Orange/red
- **Properties**:
  - Can be overlapped (agents can walk into it)
  - Typically causes negative reward or episode termination
  - Represents hazardous terrain
- **Interaction**:
  - **Forward action** onto lava: Agent may receive penalty

### 10. Switch
- **Type**: `switch`
- **Color**: Red (default)
- **Properties**:
  - Can be overlapped (agents can walk through)
  - Triggers special events when activated
- **Interaction**:
  - **Forward action** onto switch: Triggers switch activation
  - Effects depend on environment implementation

### 11. Block
- **Type**: `block`
- **Color**: Brown (light brown)
- **Appearance**: Light brown square
- **Properties**:
  - Cannot be passed through by agents
  - Cannot be picked up or carried
  - **Can be pushed** by any agent moving forward into it
  - Multiple consecutive blocks can be pushed as a group
- **Push Mechanics**:
  - Agent must be facing the block
  - Agent uses **forward action** to push
  - Block(s) move one cell in the direction the agent is facing
  - Push succeeds only if the cell behind the block (or consecutive flight of blocks) is empty
  - If cell behind is blocked, push fails and agent doesn't move
  - All blocks in a consecutive line are pushed together
- **Interaction**:
  - **Forward action** when facing block: Attempts to push the block(s)
  - Cannot be picked up with pickup action
  - Cannot be toggled

### 12. Rock
- **Type**: `rock`
- **Color**: Grey (medium grey)
- **Appearance**: Irregularly shaped rock (medium grey with darker texture)
- **Properties**:
  - Cannot be passed through by agents
  - Cannot be picked up or carried
  - **Can be pushed** only by specific agents (based on agent type/role)
  - Multiple consecutive rocks/blocks can be pushed as a group
- **Agent Restrictions**:
  - Each rock has a `pushable_by` attribute that specifies which agents can push it
  - Can be set to:
    - `None`: Pushable by all agents (behaves like block)
    - Single agent index (e.g., `0`): Only that specific agent can push
    - List of agent indices (e.g., `[0, 2]`): Only those agents can push
  - If agent cannot push a rock, forward action is blocked
  - **Default behavior (when using `map` parameter)**: Only grey agents (representing robots) can push rocks. This can be changed via the `can_push_rocks` parameter.
- **Push Mechanics**:
  - Same as blocks: agent must face the rock, uses forward action
  - Rock(s) move one cell in the direction the agent is facing
  - Push succeeds only if the cell behind is empty
  - Can push consecutive rocks and blocks together (if agent has permission)
- **Interaction**:
  - **Forward action** when facing rock: Attempts to push if agent has permission
  - Cannot be picked up with pickup action
  - Cannot be toggled

### 13. Unsteady Ground
- **Type**: `unsteadyground`
- **Color**: Brown (default) or other colors
- **Appearance**: Floor-like tile with diagonal lines indicating instability
- **Properties**:
  - Can be overlapped (agents can walk on it)
  - **Stochastic movement**: Agents attempting forward action on unsteady ground may stumble
  - Each unsteady ground cell has a `stumble_probability` parameter (default 0.5)
  - Behaves like empty floor for all other actions (left, right, pickup, etc.)
- **Stumbling Mechanic**:
  - When an agent on unsteady ground attempts **forward action**:
    - With probability `stumble_probability`: Agent stumbles and the forward action is replaced by **left+forward** or **right+forward** (chosen randomly)
    - With probability `1 - stumble_probability`: Agent moves straight forward (no stumble)
  - The turn (left/right) and forward movement happen **in the same time step**
  - Stumbling does **not** consume extra actions or time steps
- **Special Processing Order**:
  - Agents on unsteady ground attempting forward are processed **after** all other agents
  - Other agents (including those on unsteady ground doing non-forward actions) are processed first in random order
  - This ensures unsteady-forward agents' stumbling is resolved after normal agent movements
- **Conflict Resolution**:
  - If an unsteady-forward agent's target cell is occupied by another agent, the forward movement is blocked (but the turn still occurs)
  - If multiple unsteady-forward agents compete for the same target cell, **none of them move forward** (but they still turn)
  - This differs from normal conflict resolution, which picks a winner randomly
- **Transition Probabilities**:
  - In `transition_probabilities()` method, each unsteady-forward agent creates a stochastic block with 3 outcomes:
    1. Forward (straight)
    2. Left+forward
    3. Right+forward
  - All outcomes have equal probability (1/3 each if no other factors)
  - These blocks are combined with regular conflict blocks via Cartesian product
- **Use Cases**:
  - Simulating difficult terrain (ice, mud, loose gravel)
  - Adding stochasticity to agent movement
  - Creating navigation challenges
  - Testing robustness of multi-agent policies

### 14. Magic Wall
- **Type**: `magicwall`
- **Color**: Grey (default) or other colors
- **Appearance**: Wall with a dashed blue line near the magic side, parallel to it; magenta flash on successful entry
- **Properties**:
  - Cannot be passed through by agents under normal circumstances
  - Cannot be seen through (blocks vision)
  - Can be entered by specific agents with a certain probability from one specific direction (or all directions)
  - Once entered, agents can step off as if it was an empty cell (can overlap)
  - Cannot be picked up or moved
  - May turn into a normal wall after a failed entry attempt
- **Attributes**:
  - `magic_side`: Direction from which the wall can be entered (0=right, 1=down, 2=left, 3=up, 4=all)
  - `entry_probability`: Probability (0.0 to 1.0) that an authorized agent successfully enters from the magic side
  - `solidify_probability`: Probability (0.0 to 1.0) that a failed entry attempt turns the magic wall into a normal wall (default 0.0)
- **Agent Requirements**:
  - Agent must have `can_enter_magic_walls` attribute set to `True` to attempt entry
  - Agent must approach from the magic side (opposite direction to their facing direction), or magic_side=4 allows any direction
  - Entry attempt is probabilistic based on `entry_probability`
- **Entry Mechanics**:
  - Agent uses **forward action** when facing the magic wall from the magic side
  - Entry succeeds with probability `entry_probability`
  - If entry succeeds, agent moves into the magic wall cell
  - If entry fails, agent remains in place
  - If entry fails AND random < `solidify_probability`, the magic wall becomes a normal wall
  - Entry attempts are processed **last** in each step (after normal agents and unsteady ground agents)
  - **No conflicts possible**: At most one agent can be next to the magic side of each magic wall
- **Rendering**:
  - Base wall rendered in configured color (default grey)
  - Dashed blue line drawn parallel to and near the magic side(s)
  - Line position indicates which side can be entered from:
    - magic_side=0 (right): Line on right edge
    - magic_side=1 (down): Line on bottom edge
    - magic_side=2 (left): Line on left edge
    - magic_side=3 (up): Line on top edge
    - magic_side=4 (all): Lines on all four edges
  - Magenta flash and brighter wall color when entry succeeds
- **Use Cases**:
  - Adding controlled stochasticity to navigation
  - Creating special passages accessible only to certain agents
  - Simulating doors with uncertain access that may lock permanently
  - Testing agent behavior with probabilistic barriers

### 15. KillButton
- **Type**: `killbutton`
- **Map Code**: `Kb` or `Ki`
- **Color**: Red
- **Appearance**: Red floor tile with X pattern (grey when disabled)
- **Properties**:
  - Non-overlappable (agents cannot walk on it, must face and toggle)
  - When toggled by `trigger_color` agent (default: yellow), permanently terminates all `target_color` agents (default: grey)
  - Terminated agents can only use "still" action
  - Can be enabled/disabled via DisablingSwitch
- **Attributes**:
  - `trigger_color`: Color of agents that trigger the kill effect (default: 'yellow')
  - `target_color`: Color of agents that will be terminated (default: 'grey')
  - `enabled`: Whether the button is active (default: True)
- **Use Cases**:
  - Emergency stop mechanisms for robots
  - Human-robot interaction scenarios where humans can shut down robots
  - Creating hazardous zones for specific agent types

### 16. PauseSwitch
- **Type**: `pauseswitch`
- **Map Code**: `Ps` or `Pa`
- **Color**: Orange (on) / Blue (off) / Grey (disabled)
- **Appearance**: Colored square with pause bars (||) when on
- **Properties**:
  - Cannot be overlapped (blocks movement)
  - Toggle action by `toggle_color` agent (default: yellow) switches on/off
  - When ON, all `target_color` agents (default: grey) are paused (can only use "still" action)
  - Can be enabled/disabled via DisablingSwitch
- **Attributes**:
  - `toggle_color`: Color of agents that can toggle the switch (default: 'yellow')
  - `target_color`: Color of agents that will be paused (default: 'grey')
  - `is_on`: Whether the switch is currently on (default: False)
  - `enabled`: Whether the switch is active (default: True)
- **Use Cases**:
  - Temporary pause controls for robots
  - Human-robot coordination scenarios
  - Turn-based gameplay mechanics

### 17. DisablingSwitch
- **Type**: `disablingswitch`
- **Map Code**: `Dk`/`dK` (for killbuttons), `Dp`/`dP` (for pauseswitches), `dC`/`DC` (for controlbuttons)
- **Color**: Purple
- **Appearance**: Purple square with circle-slash (disabled symbol)
- **Properties**:
  - Cannot be overlapped (blocks movement)
  - Toggle action by `toggle_color` agent (default: grey) enables/disables all objects of `target_type`
  - Affects all objects of the target type in the grid
- **Attributes**:
  - `toggle_color`: Color of agents that can toggle the switch (default: 'grey')
  - `target_type`: Type of objects to enable/disable ('killbutton', 'pauseswitch', or 'controlbutton')
- **Use Cases**:
  - Robots disabling human control mechanisms
  - Strategic control over interactive elements
  - Multi-stage puzzle environments

### 18. ControlButton
- **Type**: `controlbutton`
- **Map Code**: `CB`
- **Color**: Green (bright when programmed, dark when unprogrammed, grey when disabled)
- **Appearance**: Green square with arrow (programmed) or empty circle (unprogrammed)
- **Properties**:
  - Cannot be overlapped (blocks movement)
  - Two-phase interaction: programming and triggering
  - **Programming**: `controlled_color` agent (default: grey) toggles, then performs any action - that action is memorized
  - **Triggering**: `trigger_color` agent (default: yellow) toggles - forces the controlled agent to perform the memorized action on the next step
  - Can be enabled/disabled via DisablingSwitch
- **Attributes**:
  - `trigger_color`: Color of agents that can trigger programmed actions (default: 'yellow')
  - `controlled_color`: Color of agents that can program the button (default: 'grey')
  - `enabled`: Whether the button is active (default: True)
  - `controlled_agent`: Index of the agent that programmed this button (None initially)
  - `triggered_action`: The action that was programmed (None initially)
- **Programming Workflow**:
  1. Robot (grey) faces button and uses toggle action → enters programming mode
  2. Robot performs any action (left, right, forward, toggle, etc.) → action is recorded
  3. Button is now programmed with that action
- **Triggering Workflow**:
  1. Human (yellow) faces programmed button and uses toggle action
  2. Robot's `forced_next_action` is set to the programmed action
  3. On the next step(), robot's chosen action is replaced by the programmed action
- **Use Cases**:
  - Human-robot control interfaces
  - Programmable robot behaviors
  - Remote control mechanisms where humans guide robot movement
  - Teaching scenarios where robots pre-program actions for humans to trigger

### 19. Agent
- **Type**: `agent`
- **Color**: Red, green, blue, purple, yellow, grey (assigned by index)
- **Properties**:
  - Occupies one cell at a time
  - Has a direction (0=right, 1=down, 2=left, 3=up)
  - Can carry at most one object
  - Has a partially observable view (default 7×7 grid)
  - Can be in states: active, terminated, paused, started
- **Attributes**:
  - `pos`: Current position (x, y)
  - `dir`: Current direction (0-3)
  - `carrying`: Object being carried (or None)
  - `terminated`: Whether agent has finished
  - `paused`: Whether agent is temporarily inactive
  - `started`: Whether agent has begun acting
  - `view_size`: Size of observable area (default 7)
  - `can_enter_magic_walls`: Whether agent can attempt to enter magic walls (default False)
  - `can_push_rocks`: Whether agent can push rocks (default False)
  - `forced_next_action`: If set, overrides the agent's next action (used by ControlButton)

## Agent Types

**Single Agent Type**: All agents in MultiGrid are of the same type. There are **no distinct agent types** such as robots vs. humans or different agent classes.

All agents:
- Have the same capabilities
- Use the same action space
- Are distinguished only by color/index
- Can be assigned different roles through game rules (e.g., teams in soccer)

## Actions

### Standard Actions (Actions class)

1. **Still** (0)
   - Agent takes no action this turn
   - Agent position and state remain unchanged
   - No interaction with environment

2. **Left** (1)
   - Agent rotates 90° counterclockwise
   - Direction changes: 0→3, 1→0, 2→1, 3→2
   - Position unchanged
   - Always succeeds

3. **Right** (2)
   - Agent rotates 90° clockwise
   - Direction changes: 0→1, 1→2, 2→3, 3→0
   - Position unchanged
   - Always succeeds

4. **Forward** (3)
   - Agent attempts to move forward one cell in current direction
   - **Success conditions**:
     - Target cell is empty, OR
     - Target cell contains an object that can be overlapped (floor, goal, lava, switch, open door)
   - **Failure conditions**:
     - Target cell contains wall
     - Target cell contains closed door
     - Target cell contains non-overlappable object
     - Target cell contains another agent
   - When successful: Agent moves to new position
   - When failed: Agent remains in place

5. **Pickup** (4)
   - Agent attempts to pick up object in front
   - **Requirements**:
     - Agent is not already carrying something
     - Object in front can be picked up (key, ball, box)
   - **Effect**: Object removed from grid, held by agent
   - **Failure**: If requirements not met, no effect

6. **Drop** (5)
   - Agent attempts to drop carried object
   - **Requirements**:
     - Agent is carrying something
     - Cell in front is empty or can be overlapped
   - **Effect**: Object placed in cell in front of agent
   - **Failure**: If cell is blocked, no effect

7. **Toggle** (6)
   - Agent activates/toggles object in front
   - **Effects by object type**:
     - **Door (unlocked)**: Opens if closed, closes if open
     - **Door (locked)**: Unlocks and opens if agent carries matching colored key
     - **Box**: Opens box, replaces it with contents
     - **Other objects**: No effect
   - Always succeeds (may have no visible effect)

8. **Done** (7)
   - Signals task completion (optional)
   - Typically not used in standard environments
   - Effect depends on environment implementation

### Small Actions (SmallActions class)

Simplified action set:
- Still (0)
- Left (1)
- Right (2)
- Forward (3)

Used in simpler environments with no object interaction.

### Mine Actions (MineActions class)

Extended action set for building:
- Still (0)
- Left (1)
- Right (2)
- Forward (3)
- Build (4) - Environment-specific construction action

## Action Consequences and Stochasticity

### Deterministic Actions

Most individual actions are **deterministic**:
- Rotation (left/right) always succeeds
- Pickup/drop/toggle have predictable effects
- Forward movement is deterministic given current state (except on unsteady ground)

### Sources of Non-Determinism

There are **two sources** of stochasticity in the environment:

#### 1. Agent Execution Order (Multi-Agent Conflicts)
- When multiple agents act simultaneously, they execute in random order
- Order determined by: `order = np.random.permutation(len(actions))`
- Each permutation has equal probability: 1/n! where n is the number of active agents
- **Exception**: Agents on unsteady ground attempting forward are processed after all other agents

#### 2. Unsteady Ground Stumbling (Movement Stochasticity)
- When an agent on unsteady ground attempts forward action
- Agent may stumble with probability defined by the cell's `stumble_probability` parameter
- If stumbling occurs, the forward action is replaced by left+forward or right+forward (50-50 chance)
- This introduces action-level stochasticity independent of agent ordering

### When Order Matters (Probabilistic Outcomes from Conflicts)

Transitions become probabilistic when 2+ agents:
1. **Compete for same cell**: Two agents move forward to same empty cell
   - One succeeds (gets to move), one fails (stays in place)
   - Winner determined by execution order
   
2. **Compete for same object**: Two agents try to pick up same object
   - One succeeds (picks up object), one fails (picks up nothing)
   
3. **Sequential dependencies**: One agent's action affects another's outcome
   - Example: Agent A opens door, Agent B moves through
   - If B acts first, door is still closed

### When Stumbling Matters (Probabilistic Outcomes from Unsteady Ground)

Movement becomes probabilistic when:
1. **Agent on unsteady ground moves forward**:
   - May move straight (probability 1 - stumble_probability)
   - May turn left and move (probability stumble_probability / 2)
   - May turn right and move (probability stumble_probability / 2)

2. **Multiple unsteady-forward agents compete for same cell**:
   - Each agent's stumbling is resolved independently
   - If agents target the same cell (after stumbling), none move forward
   - This creates complex probability distributions combining stumbling and conflicts

### When Order Doesn't Matter (Deterministic Outcomes)

Transitions remain deterministic when:
- Only 0 or 1 agent acts (no conflicts)
- All agents only rotate (rotations never interfere)
- Agents act on independent, non-overlapping resources
- No agents are on unsteady ground attempting forward action

## Object Movement

### Pickable Objects (Keys, Balls, Boxes)

**Movement method**: Pickup and drop only
- Agent uses **pickup** action to take object
- Agent carries object while moving
- Agent uses **drop** action to place object

**NOT supported for these objects**:
- ❌ Pushing keys, balls, or boxes
- ❌ Pulling objects
- ❌ Moving objects without picking them up

### Boxes Specifically

- Boxes are **not pushable** like in Sokoban
- Boxes must be **picked up** to be moved
- Agent can carry one box at a time
- Box can contain another object (revealed when toggled)
- **Box pushing is NOT supported** in this environment

### Pushable Objects (Blocks and Rocks)

**Movement method**: Pushing only
- Agent uses **forward** action while facing the object
- Agent pushes object(s) one cell in the direction they're facing
- Objects move immediately when pushed
- Agent moves into the space the object vacated

**Push mechanics**:
- ✅ Blocks can be pushed by any agent
- ✅ Rocks can be pushed only by authorized agents (based on `pushable_by` attribute)
- ✅ Multiple consecutive blocks/rocks can be pushed as a group
- ✅ Push succeeds only if cell behind the object(s) is empty
- ❌ Cannot push if cell behind is blocked by wall, door, agent, or other objects
- ❌ Blocks and rocks cannot be picked up or carried
- ❌ Cannot pull blocks or rocks (push only)

## Doors and Keys

### Door Mechanics

1. **Unlocked Doors**:
   - Can be opened/closed with toggle action
   - Any agent can toggle
   - No key required

2. **Locked Doors**:
   - Require key of matching color
   - Agent must carry matching key
   - Toggle action unlocks and opens door
   - Key is NOT consumed (can be reused)

3. **Key-Door Matching**:
   - Red key → Red door
   - Blue key → Blue door
   - Green key → Green door
   - Purple key → Purple door
   - Yellow key → Yellow door
   - Keys cannot open doors of different colors

### Key Properties

- **Reusable**: Keys are not consumed when unlocking
- **Universal matching**: Any key of the correct color works
- **One at a time**: Agent can only carry one object (key or other)
- **Persistent**: Keys remain in agent's inventory until dropped

## Environment Observations

### Encoding

Each grid cell encodes as a 6-tuple:
1. Object type (wall, door, agent, etc.)
2. Object color
3. Type of object agent is carrying (if cell contains agent)
4. Color of object agent is carrying (if cell contains agent)
5. Agent direction (if cell contains agent)
6. Whether this agent is the observing agent (1) or another agent (0)

### Visibility

- Agents have limited field of view (default 7×7)
- Walls and closed doors block vision
- Agents can see through open doors and overlappable objects
- Environment can be configured as fully or partially observable

## Special Behaviors

### Agent Interaction

- Agents **cannot** move into cells occupied by other agents
- Agents **can** pass objects between each other via drop/pickup
- Agents **compete** for resources when trying to interact simultaneously

### Carrying Objects

- Agent can carry **exactly one object** at a time
- Must drop current object before picking up another
- Carried object moves with agent
- Dropped object placed in front of agent

### Episode Termination

Episodes end when:
- Agent reaches goal (if goal-based task)
- Maximum steps reached (`step_count >= max_steps`)
- Environment-specific termination conditions met

## Summary

**Key Points**:
- **19 object types**: wall, floor, door, key, ball, box, goal, objgoal, lava, switch, block, rock, unsteady ground, magic wall, killbutton, pauseswitch, disablingswitch, controlbutton, and agent
- **8 standard actions**: still, left, right, forward, pickup, drop, toggle, done
- **Single agent type**: No distinction between robot/human or different agent classes (though rocks can have agent-specific push permissions and agents can have magic wall entry capability)
- **Boxes are NOT pushable**: Must be picked up and carried
- **Blocks ARE pushable**: Can be pushed by any agent using forward action
- **Rocks ARE pushable with restrictions**: Can only be pushed by specific agents based on `can_push_rocks` attribute
- **Unsteady ground introduces stochasticity**: Agents may stumble when moving forward on unsteady ground
- **Magic walls introduce stochasticity**: Agents with `can_enter_magic_walls=True` can attempt entry with configurable probability from one specific direction
- **Keys are reusable**: Not consumed when unlocking doors
- **Color matching required**: Keys must match door color
- **Agent control mechanisms**:
  - **KillButton**: Permanently terminates target agents when trigger agent toggles it
  - **PauseSwitch**: Temporarily pauses target agents when toggled on
  - **DisablingSwitch**: Enables/disables KillButtons, PauseSwitches, or ControlButtons
  - **ControlButton**: Allows programming actions that can be triggered later (human-robot control)
- **Stochasticity sources**: 
  1. Agent execution order (random permutation for normal agents)
  2. Unsteady ground stumbling (configurable probability per cell)
  3. Magic wall entry (configurable probability per wall)
- **No agent subtypes**: All agents have same capabilities, distinguished by color/index only

This gridworld focuses on **multi-agent coordination** and **object manipulation**, including Sokoban-style pushing mechanics for blocks and rocks, stochastic movement on unsteady ground and magic wall entry, and **human-robot interaction** via control buttons and switches.

## Developer Notes

### Adding New Stochastic Object Types

**CRITICAL**: When adding any new object type or stochastic behavior that affects agent actions, you MUST maintain consistency between `step()` and `transition_probabilities()` methods. Both must produce the same probability distribution over successor states.

**Required changes** for adding a new stochastic element:

1. **Update `_categorize_agents()` helper**: Add logic to identify agents affected by the new stochastic element
2. **Update `step()` method**: Process the new agent category appropriately, sampling stochastic outcomes
3. **Update `transition_probabilities()` method**: Add uncertainty blocks for the new stochastic element to the Cartesian product
4. **Update `_compute_successor_state_with_unsteady()` method**: Handle deterministic execution based on resolved outcomes

**Uncertainty Block Pattern**:

All sources of randomness must be represented as **uncertainty blocks** that are combined via Cartesian product. Each block represents one stochastic element (e.g., one agent on unsteady ground, one agent attempting magic wall entry).

Each uncertainty block contains a list of **(probability, outcome)** pairs:
- The probabilities must sum to 1.0 for each block
- Each outcome is a string identifier (e.g., 'forward', 'succeed', 'fail')
- The Cartesian product of all blocks generates all possible outcome combinations
- Each combination's probability = product of individual outcome probabilities

**Example pattern** (as used for unsteady ground and magic walls):

```python
# In _categorize_agents(): Identify affected agents
if condition_for_stochastic_element:
    stochastic_agents.append(i)

# In step(): Sample random outcome based on probabilities
rand_val = random()
if rand_val < probability_a:
    outcome = 'option_a'
else:
    outcome = 'option_b'
    
# In transition_probabilities(): Create uncertainty block with (probability, outcome) pairs
outcomes = [
    (probability_a, 'option_a'),
    (1.0 - probability_a, 'option_b')
]
all_blocks.append(('element_type', agent_idx, outcomes))

# When computing outcome probabilities (automatic from structure):
for outcome_idx, (prob, outcome_name) in enumerate(block_outcomes):
    outcome_probability *= prob  # Multiply probabilities in Cartesian product

# In _compute_successor_state_with_unsteady(): Execute based on outcome
if outcome == 'option_a':
    # execute option A
else:
    # execute option B
```

**Existing examples**:
1. **Conflict blocks**: Multiple agents compete for same resource
   - Structure: ('conflict', [agent_indices])
   - Probabilities: uniform (1 / num_agents per agent)
   - Note: conflict blocks use a different structure for efficiency

2. **Unsteady ground blocks**: Agent stumbles when moving forward
   - Structure: ('unsteady', agent_idx, [(prob, outcome), ...])
   - Example outcomes with stumble_probability=0.5:
     - (0.5, 'forward') - doesn't stumble
     - (0.25, 'left-forward') - stumbles left
     - (0.25, 'right-forward') - stumbles right

3. **Magic wall blocks**: Agent attempts probabilistic entry
   - Structure: ('magicwall', agent_idx, [(prob, outcome), ...])
   - Example outcomes with entry_probability=0.7:
     - (0.7, 'succeed') - entry succeeds
     - (0.3, 'fail') - entry fails

**All future additions** of randomness sources MUST follow this uncertainty block pattern with **(probability, outcome)** pairs to maintain consistency between step() and transition_probabilities().

Failure to maintain this consistency will break correctness of probability computations and planning algorithms.

## Additional Clarifications

### Can doors be locked again after being unlocked?

**No.** Once a door is unlocked (by using a matching key with the toggle action), it permanently becomes an unlocked door. The `is_locked` flag is set to `False` and cannot be set back to `True` through any standard mechanism. After unlocking, the door can only be opened and closed via the toggle action, but never re-locked.

### Can there be more than 6 different door/ball/box/goal types?

**No, maximum 6 colors.** The environment is limited to **6 colors** defined in `World.COLOR_TO_IDX`:
- Red (0)
- Green (1)
- Blue (2)
- Purple (3)
- Yellow (4)
- Grey (5)

This means:
- Maximum 6 different colored doors
- Maximum 6 different colored keys
- Maximum 6 different colored balls
- Maximum 6 different colored boxes
- Maximum 6 different colored goals

Since agents are also assigned colors by index, and there are 6 colors, you can have at most 6 agents (though technically agent index can be 0 which might allow special behavior in some environments).

### Can agents pass by keys, balls, boxes without picking them up?

**No.** Keys, balls, and boxes **block movement**. These objects have `can_overlap() = False` and `can_pickup() = True`, which means:

- **Agents cannot walk through cells** containing keys, balls, or boxes using the forward action
- **The cell is blocked** until the object is removed
- **To interact**: 
  1. Use **pickup** action to take the object (if not already carrying something), OR
  2. **Walk around** the cell containing the object

**Objects that CAN be passed through** (have `can_overlap() = True`):
- Floor tiles
- Goals
- Open doors
- Lava
- Switches

So agents automatically walk over these objects when using the forward action, without needing to pick them up.

### Can agents be made to observe the state fully?

**Yes.** The environment supports both partial and full observability via the `partial_obs` parameter:

- **Partial observation** (`partial_obs=True`, default):
  - Each agent has a limited field of view (default 7×7 grid)
  - Observation is agent-centric and rotated based on agent direction
  - Agents cannot see through walls or closed doors
  
- **Full observation** (`partial_obs=False`):
  - Agents observe the entire grid state
  - Observation includes all objects and all agents' positions
  - No visibility restrictions

Set when creating the environment:
```python
env = MultiGridEnv(partial_obs=False, ...)  # Full observability
```

### What makes an agent pause or unpause?

**Agent pause/unpause is NOT automatically controlled** by the base environment. The `paused` and `started` flags exist as attributes on each agent, but the base `MultiGridEnv` class does **not** provide methods to pause or unpause agents.

These flags are:
- **Checked** in the step function: `if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started`
- **Initialized**: `paused=False`, `started=True` by default
- **Not modified** by any base environment code

**To use these flags**, you must:
1. Create a custom environment that extends `MultiGridEnv`
2. Implement your own logic to set `agent.paused = True/False` or `agent.started = True/False`
3. Common use cases might include:
   - Turn-based gameplay (pause agents waiting for their turn)
   - Agents that enter/leave the game dynamically
   - Penalty states where agents are temporarily frozen

### What determines what a switch does and what could it do?

**Switches are environment-specific.** The `Switch` object is defined in the base code, but its behavior is **completely customizable** through the `_handle_switch()` method:

```python
def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
    pass  # Base implementation does nothing
```

**To implement switch behavior**, create a custom environment:

```python
class MyEnv(MultiGridEnv):
    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        # Custom logic, e.g.:
        # - Open/close doors
        # - Spawn objects
        # - Give rewards
        # - Change environment state
        # - Toggle lights/visibility
        # - Activate mechanisms
        pass
```

**Switch is triggered** when an agent moves forward onto a cell containing a switch (forward action into switch cell).

### Does the code provide means for adding further object types and interaction logic?

**Yes, the code is designed for extension.** You can add new object types and behaviors by:

#### 1. **Adding New Object Types**

```python
# Define new object class
class MyNewObject(WorldObj):
    def __init__(self, world, color='blue'):
        super().__init__(world, 'mynewobject', color)
    
    def can_overlap(self):
        return True  # or False
    
    def can_pickup(self):
        return True  # or False
    
    def toggle(self, env, pos):
        # Define toggle behavior
        return True
    
    def render(self, img):
        # Define how to render this object
        pass
```

#### 2. **Extending World Object Registry**

Add your object type to the `OBJECT_TO_IDX` dictionary (requires modifying base code):

```python
OBJECT_TO_IDX = {
    # ... existing types ...
    'mynewobject': 13,
}
```

#### 3. **Customizing Interaction Logic**

Override handler methods in your custom environment:

```python
class MyEnv(MultiGridEnv):
    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        # Custom pickup logic
        pass
    
    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        # Custom drop logic
        pass
    
    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        # Custom movement consequences
        pass
    
    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        # Custom switch behavior
        pass
    
    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        # Custom build action (if using MineActions)
        pass
```

#### 4. **Creating Custom Actions**

Define a new action set:

```python
class MyActions:
    available = ['still', 'left', 'right', 'forward', 'custom1', 'custom2']
    still = 0
    left = 1
    right = 2
    forward = 3
    custom1 = 4
    custom2 = 5
```

Then handle these actions in your environment's step function (would require overriding `step()`).

**Examples in the codebase:**
- `CollectGameEnv`: Customizes `_handle_pickup()` and `_reward()` for ball collection
- `SoccerGameEnv`: Customizes pickup/drop for ball passing mechanics
- `MineActions`: Adds a "build" action for construction

### What are tasks and how are they set?

**Tasks are defined by the environment implementation**, not by a formal task system. A "task" in MultiGrid is simply the objective that the environment designer creates through:

#### 1. **Grid Generation** (`_gen_grid` method)
```python
def _gen_grid(self, width, height):
    # Place walls, objects, goals, agents
    # This defines the initial state and objective
    self.grid = Grid(width, height)
    # ... place objects ...
    # ... place agents ...
```

#### 2. **Reward Structure** (`_reward` method)
```python
def _reward(self, i, rewards, reward=1):
    # Define which agents get rewards for which actions
    rewards[i] += reward
```

#### 3. **Termination Conditions** (`step` method)
```python
def step(self, actions):
    # ... execute actions ...
    if <task_completed>:
        done = True
    if self.step_count >= self.max_steps:
        done = True
    return obs, rewards, done, info
```

**Common task patterns:**

- **Collection task**: Agents collect objects of matching color
  - Reward when correct object picked up
  - Penalty when wrong object picked up
  
- **Goal-reaching task**: Agents navigate to goal locations
  - Reward when agent moves onto goal cell
  - Episode terminates on goal reach

- **Delivery task**: Agents deliver objects to specific locations
  - Reward when object dropped at correct location
  
- **Cooperative task**: Multiple agents must coordinate
  - Shared rewards for team success
  - Individual penalties for failures

- **Competitive task**: Agents compete for resources
  - Zero-sum rewards
  - First agent to complete wins

**Task parameters** are typically passed to the environment constructor:
```python
env = CollectGameEnv(
    num_balls=[5],           # Task: collect 5 balls
    agents_index=[1, 2, 3],  # 3 agents participate
    balls_reward=[1],        # Reward value
    zero_sum=True            # Competitive (zero-sum)
)
```

**There is no formal task specification language** - tasks are implemented programmatically by creating custom environment classes that inherit from `MultiGridEnv` and override the relevant methods.

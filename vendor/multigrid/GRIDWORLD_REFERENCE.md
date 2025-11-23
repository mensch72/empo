# MultiGrid Gridworld Reference

This document provides a complete reference for the MultiGrid gridworld environment, describing all cell types, object types, actions, and their consequences.

## Overview

MultiGrid is a 2D grid-based multi-agent environment where agents navigate a grid, interact with objects, and complete tasks. The environment can be fully or partially observable.

## Grid Cells

Each cell in the grid can contain:
- **Empty space** (None) - Agents can move through
- **One object** - Wall, floor, door, key, ball, box, goal, lava, switch, or object goal
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
  - **Toggle action** on locked door with matching key: Unlocks and opens the door
  - **Toggle action** on locked door without key: No effect
- **Key matching**: A key can only unlock a door of the same color
- **Key reusability**: Keys are held by the agent and can be reused multiple times (not consumed)

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

### 11. Agent
- **Type**: `agent`
- **Color**: Red, green, blue, purple, yellow (assigned by index)
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
- Forward movement is deterministic given current state

### Source of Non-Determinism

**Agent execution order** is the ONLY source of stochasticity:
- When multiple agents act simultaneously, they execute in random order
- Order determined by: `order = np.random.permutation(len(actions))`
- Each permutation has equal probability: 1/n!

### When Order Matters (Probabilistic Outcomes)

Transitions become probabilistic when 2+ agents:
1. **Compete for same cell**: Two agents move forward to same empty cell
   - One succeeds (gets to move), one fails (stays in place)
   - Winner determined by execution order
   
2. **Compete for same object**: Two agents try to pick up same object
   - One succeeds (picks up object), one fails (picks up nothing)
   
3. **Sequential dependencies**: One agent's action affects another's outcome
   - Example: Agent A opens door, Agent B moves through
   - If B acts first, door is still closed

### When Order Doesn't Matter (Deterministic Outcomes)

Transitions remain deterministic when:
- Only 0 or 1 agent acts
- All agents only rotate (rotations never interfere)
- Agents act on independent, non-overlapping resources

## Object Movement

### Pickable Objects (Keys, Balls, Boxes)

**Movement method**: Pickup and drop only
- Agent uses **pickup** action to take object
- Agent carries object while moving
- Agent uses **drop** action to place object

**NOT supported**:
- ❌ Pushing objects
- ❌ Pulling objects
- ❌ Pushing multiple objects in a row
- ❌ Moving objects without picking them up

### Boxes Specifically

- Boxes are **not pushable** like in Sokoban
- Boxes must be **picked up** to be moved
- Agent can carry one box at a time
- Box can contain another object (revealed when toggled)
- **No box pushing mechanics** exist in this environment

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
- **11 object types**: wall, floor, door, key, ball, box, goal, objgoal, lava, switch, agent
- **8 standard actions**: still, left, right, forward, pickup, drop, toggle, done
- **Single agent type**: No distinction between robot/human or different agent classes
- **Boxes are NOT pushable**: Must be picked up and carried
- **Keys are reusable**: Not consumed when unlocking doors
- **Color matching required**: Keys must match door color
- **Stochasticity source**: Agent execution order (random permutation)
- **No agent subtypes**: All agents have same capabilities, distinguished by color/index only

This gridworld focuses on **multi-agent coordination** and **object manipulation** rather than physics-based mechanics like box pushing.

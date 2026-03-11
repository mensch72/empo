#!/usr/bin/env python3
"""
Test switches and buttons for Issue #25.

Tests for:
- KillButton: non-overlappable switch that kills agents when toggled
- PauseSwitch: non-overlappable switch that pauses agents when toggled
- DisablingSwitch: non-overlappable switch that disables KillButtons or PauseSwitches
"""

import sys
import numpy as np

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import (
    MultiGridEnv, Agent, World, Actions, Grid, Wall,
    KillButton, PauseSwitch, DisablingSwitch
)


class KillButtonEnv(MultiGridEnv):
    """Simple environment for testing KillButton."""
    
    def __init__(self):
        # Create a yellow agent (human) and grey agent (robot)
        agents = [
            Agent(World, World.COLOR_TO_IDX['yellow']),  # Yellow agent
            Agent(World, World.COLOR_TO_IDX['grey']),    # Grey agent
        ]
        
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around perimeter
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))
        
        # Place yellow agent at (1, 1)
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0  # facing right
        self.grid.set(1, 1, self.agents[0])
        
        # Place grey agent at (3, 3)
        self.agents[1].pos = np.array([3, 3])
        self.agents[1].dir = 0  # facing right
        self.grid.set(3, 3, self.agents[1])
        
        # Place KillButton at (2, 1) - right in front of yellow agent
        self.kill_button = KillButton(World, trigger_color='yellow', target_color='grey')
        self.grid.set(2, 1, self.kill_button)


class PauseSwitchEnv(MultiGridEnv):
    """Simple environment for testing PauseSwitch."""
    
    def __init__(self):
        # Create a yellow agent (human) and grey agent (robot)
        agents = [
            Agent(World, World.COLOR_TO_IDX['yellow']),  # Yellow agent
            Agent(World, World.COLOR_TO_IDX['grey']),    # Grey agent
        ]
        
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around perimeter
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))
        
        # Place yellow agent at (1, 2)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # facing right
        self.grid.set(1, 2, self.agents[0])
        
        # Place grey agent at (3, 3)
        self.agents[1].pos = np.array([3, 3])
        self.agents[1].dir = 3  # facing up
        self.grid.set(3, 3, self.agents[1])
        
        # Place PauseSwitch at (2, 2) - right in front of yellow agent
        self.pause_switch = PauseSwitch(World, toggle_color='yellow', target_color='grey')
        self.grid.set(2, 2, self.pause_switch)


class DisablingSwitchEnv(MultiGridEnv):
    """Simple environment for testing DisablingSwitch."""
    
    def __init__(self):
        # Create a yellow agent (human) and grey agent (robot)
        agents = [
            Agent(World, World.COLOR_TO_IDX['yellow']),  # Yellow agent
            Agent(World, World.COLOR_TO_IDX['grey']),    # Grey agent
        ]
        
        super().__init__(
            width=6,
            height=5,
            max_steps=100,
            agents=agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around perimeter
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))
        
        # Place yellow agent at (2, 2) facing down towards KillButton at (2, 3)
        self.agents[0].pos = np.array([2, 2])
        self.agents[0].dir = 1  # facing down
        self.grid.set(2, 2, self.agents[0])
        
        # Place grey agent at (3, 2)
        self.agents[1].pos = np.array([3, 2])
        self.agents[1].dir = 0  # facing right
        self.grid.set(3, 2, self.agents[1])
        
        # Place KillButton at (2, 3)
        self.kill_button = KillButton(World, trigger_color='yellow', target_color='grey')
        self.grid.set(2, 3, self.kill_button)
        
        # Place DisablingSwitch for killbuttons at (4, 2) - in front of grey agent
        self.disabling_switch = DisablingSwitch(World, toggle_color='grey', target_type='killbutton')
        self.grid.set(4, 2, self.disabling_switch)


def test_kill_button_kills_target_agents():
    """Test that KillButton kills target agents when toggled by trigger agent."""
    env = KillButtonEnv()
    env.reset()
    
    # Verify initial state
    assert not env.agents[0].terminated, "Yellow agent should not be terminated initially"
    assert not env.agents[1].terminated, "Grey agent should not be terminated initially"
    
    # Yellow agent toggles kill button (facing it)
    actions = [Actions.toggle, Actions.still]  # Yellow toggles, grey stays
    env.step(actions)
    
    # Verify grey agent is now terminated (killed)
    assert not env.agents[0].terminated, "Yellow agent should not be terminated"
    assert env.agents[1].terminated, "Grey agent should be terminated after yellow toggled kill button"


def test_kill_button_disabled():
    """Test that disabled KillButton does not kill agents."""
    env = KillButtonEnv()
    env.reset()
    
    # Disable the kill button
    env.kill_button.enabled = False
    
    # Yellow agent toggles kill button (facing it)
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Verify grey agent is NOT terminated
    assert not env.agents[1].terminated, "Grey agent should not be terminated when kill button is disabled"


def test_kill_button_wrong_trigger_color():
    """Test that KillButton only responds to correct trigger color."""
    env = KillButtonEnv()
    env.reset()
    
    # Set trigger color to something else
    env.kill_button.trigger_color = 'blue'
    
    # Yellow agent toggles kill button (facing it)
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Grey agent should NOT be terminated because yellow is not the trigger color
    assert not env.agents[1].terminated, "Grey agent should not be terminated when wrong trigger color toggles button"


def test_pause_switch_pauses_target_agents():
    """Test that PauseSwitch pauses target agents when toggled on."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Verify initial state
    assert not env.agents[0].paused, "Yellow agent should not be paused initially"
    assert not env.agents[1].paused, "Grey agent should not be paused initially"
    assert not env.pause_switch.is_on, "Switch should be off initially"
    
    # Yellow agent toggles the switch
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Verify switch is now on and grey agent is paused
    assert env.pause_switch.is_on, "Switch should be on after toggle"
    assert not env.agents[0].paused, "Yellow agent should not be paused"
    assert env.agents[1].paused, "Grey agent should be paused after switch toggled on"


def test_pause_switch_toggle_off():
    """Test that toggling PauseSwitch off unpauses agents."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Toggle on
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    assert env.agents[1].paused, "Grey agent should be paused"
    
    # Toggle off
    env.step(actions)
    assert not env.pause_switch.is_on, "Switch should be off after second toggle"
    assert not env.agents[1].paused, "Grey agent should not be paused after switch toggled off"


def test_pause_switch_disabled():
    """Test that disabled PauseSwitch cannot be toggled."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Disable the switch
    env.pause_switch.enabled = False
    
    # Yellow agent tries to toggle the switch
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Switch should still be off
    assert not env.pause_switch.is_on, "Disabled switch should not toggle"
    assert not env.agents[1].paused, "Grey agent should not be paused"


def test_pause_switch_wrong_toggle_color():
    """Test that PauseSwitch only responds to correct toggle color."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Set toggle color to something else
    env.pause_switch.toggle_color = 'blue'
    
    # Yellow agent tries to toggle the switch
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Switch should still be off
    assert not env.pause_switch.is_on, "Switch should not toggle when wrong color tries"


def test_disabling_switch_disables_kill_buttons():
    """Test that DisablingSwitch disables all KillButtons."""
    env = DisablingSwitchEnv()
    env.reset()
    
    # Verify kill button is enabled initially
    assert env.kill_button.enabled, "KillButton should be enabled initially"
    
    # Grey agent toggles the disabling switch
    actions = [Actions.still, Actions.toggle]
    env.step(actions)
    
    # Kill button should now be disabled
    assert not env.kill_button.enabled, "KillButton should be disabled after disabling switch toggled"


def test_disabling_switch_enables_kill_buttons():
    """Test that DisablingSwitch can re-enable KillButtons."""
    env = DisablingSwitchEnv()
    env.reset()
    
    # First disable
    actions = [Actions.still, Actions.toggle]
    env.step(actions)
    assert not env.kill_button.enabled, "KillButton should be disabled"
    
    # Then re-enable
    env.step(actions)
    assert env.kill_button.enabled, "KillButton should be enabled after second toggle"


def test_disabling_switch_wrong_toggle_color():
    """Test that DisablingSwitch only responds to correct toggle color."""
    env = DisablingSwitchEnv()
    env.reset()
    
    # Set toggle color to something else
    env.disabling_switch.toggle_color = 'blue'
    
    # Grey agent tries to toggle
    actions = [Actions.still, Actions.toggle]
    env.step(actions)
    
    # Kill button should still be enabled
    assert env.kill_button.enabled, "KillButton should still be enabled when wrong color tries to toggle"


def test_paused_agent_actions_ignored():
    """Test that paused agents can only use 'still' action effectively."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Record grey agent's position
    grey_initial_pos = tuple(env.agents[1].pos)
    
    # Toggle pause switch on
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    assert env.agents[1].paused, "Grey agent should be paused"
    
    # Grey agent tries to move forward
    actions = [Actions.still, Actions.forward]
    env.step(actions)
    
    # Grey agent should not have moved
    grey_pos_after = tuple(env.agents[1].pos)
    assert grey_pos_after == grey_initial_pos, "Paused agent should not move"


def test_kill_button_state_in_get_set_state():
    """Test that KillButton state is preserved in get_state/set_state."""
    env = KillButtonEnv()
    env.reset()
    
    # Disable kill button
    env.kill_button.enabled = False
    
    # Get state
    state = env.get_state()
    
    # Re-enable and verify it changed
    env.kill_button.enabled = True
    assert env.kill_button.enabled
    
    # Restore state
    env.set_state(state)
    
    # Verify kill button is disabled again
    assert not env.kill_button.enabled, "KillButton enabled state should be restored"


def test_pause_switch_state_in_get_set_state():
    """Test that PauseSwitch state is preserved in get_state/set_state."""
    env = PauseSwitchEnv()
    env.reset()
    
    # Toggle switch on
    env.pause_switch.is_on = True
    env.pause_switch.enabled = False
    
    # Get state
    state = env.get_state()
    
    # Change state
    env.pause_switch.is_on = False
    env.pause_switch.enabled = True
    
    # Restore state
    env.set_state(state)
    
    # Verify state is restored
    assert env.pause_switch.is_on, "PauseSwitch is_on state should be restored"
    assert not env.pause_switch.enabled, "PauseSwitch enabled state should be restored"


def test_terminated_agent_actions_ignored():
    """Test that terminated (killed) agents can only use 'still' action effectively."""
    env = KillButtonEnv()
    env.reset()
    
    # Record grey agent's position and direction
    grey_initial_pos = tuple(env.agents[1].pos)
    grey_initial_dir = env.agents[1].dir
    
    # Yellow toggles kill button, killing grey
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    assert env.agents[1].terminated, "Grey agent should be terminated"
    
    # Grey agent tries to move
    actions = [Actions.still, Actions.forward]
    env.step(actions)
    
    # Grey agent should not have moved
    grey_pos_after = tuple(env.agents[1].pos)
    assert grey_pos_after == grey_initial_pos, "Terminated agent should not move"
    
    # Grey agent tries to turn
    actions = [Actions.still, Actions.left]
    env.step(actions)
    
    # Grey agent should not have turned
    assert env.agents[1].dir == grey_initial_dir, "Terminated agent should not turn"


# ============================================================================
# ControlButton Tests
# ============================================================================

from gym_multigrid.multigrid import ControlButton


class ControlButtonEnv(MultiGridEnv):
    """Simple environment for testing ControlButton."""
    
    def __init__(self):
        # Create a yellow agent (human) and grey agent (robot)
        agents = [
            Agent(World, World.COLOR_TO_IDX['yellow']),  # Yellow agent (trigger)
            Agent(World, World.COLOR_TO_IDX['grey']),    # Grey agent (controlled)
        ]
        
        super().__init__(
            width=6,
            height=5,
            max_steps=100,
            agents=agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around perimeter
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))
        
        # Place yellow agent at (1, 2) facing right
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # facing right/east
        self.grid.set(1, 2, self.agents[0])
        
        # Place grey agent at (3, 2) facing right
        self.agents[1].pos = np.array([3, 2])
        self.agents[1].dir = 0  # facing right/east
        self.grid.set(3, 2, self.agents[1])
        
        # Place ControlButton at (2, 2) - between agents
        self.control_button = ControlButton(World, trigger_color='yellow', controlled_color='grey')
        self.grid.set(2, 2, self.control_button)


def test_control_button_creation():
    """Test that ControlButton can be created with correct attributes."""
    cb = ControlButton(World, trigger_color='yellow', controlled_color='grey')
    
    assert cb.type == 'controlbutton'
    assert cb.trigger_color == 'yellow'
    assert cb.controlled_color == 'grey'
    assert cb.enabled == True
    assert cb.controlled_agent is None
    assert cb.triggered_action is None
    assert cb.can_overlap() == False


def test_control_button_programming_phase():
    """Test that robot can program a control button."""
    env = ControlButtonEnv()
    env.reset()
    
    # Grey agent (robot) is at (3, 2) facing east
    # ControlButton is at (2, 2)
    # Grey needs to turn to face the button (west)
    
    # Turn grey agent to face west (button direction)
    actions = [Actions.still, Actions.left]  # human still, robot turns left
    env.step(actions)
    actions = [Actions.still, Actions.left]  # human still, robot turns left again
    env.step(actions)
    
    # Now grey faces west, front_pos should be (2, 2)
    assert tuple(env.agents[1].front_pos) == (2, 2), "Robot should face the button"
    
    # Robot toggles to enter programming mode
    actions = [Actions.still, Actions.toggle]
    env.step(actions)
    
    # Button should now have controlled_agent set
    assert env.control_button.controlled_agent == 1, "Button should have controlled_agent set to robot index"
    assert env.control_button._awaiting_action == True, "Button should be awaiting action"
    
    # Robot performs an action to program
    actions = [Actions.still, Actions.forward]
    env.step(actions)
    
    # Button should now have triggered_action set
    assert env.control_button.triggered_action == Actions.forward, "Button should have 'forward' programmed"
    assert env.control_button._awaiting_action == False, "Button should no longer be awaiting action"


def test_control_button_triggering_phase():
    """Test that human can trigger a control button to force robot action."""
    env = ControlButtonEnv()
    env.reset()
    
    # Pre-program the button
    env.control_button.controlled_agent = 1  # Grey agent
    env.control_button.triggered_action = Actions.left  # Programmed to turn left
    
    # Yellow agent (human) is at (1, 2) facing east
    # ControlButton is at (2, 2)
    # Yellow should face the button already
    assert tuple(env.agents[0].front_pos) == (2, 2), "Human should face the button"
    
    # Record robot's initial direction
    robot_initial_dir = env.agents[1].dir
    
    # Human toggles the button
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Robot should have forced_next_action set
    assert env.agents[1].forced_next_action == Actions.left, "Robot should have forced_next_action set"
    
    # On next step, robot's action is forced to 'left' regardless of what it chooses
    actions = [Actions.still, Actions.forward]  # Robot "tries" to go forward
    env.step(actions)
    
    # Robot should have turned left (not moved forward)
    expected_dir = (robot_initial_dir - 1) % 4  # Left turn decreases direction
    assert env.agents[1].dir == expected_dir, f"Robot should have turned left: expected dir {expected_dir}, got {env.agents[1].dir}"
    
    # forced_next_action should be cleared
    assert env.agents[1].forced_next_action is None, "forced_next_action should be cleared after use"


def test_control_button_disabled():
    """Test that disabled ControlButton doesn't work."""
    env = ControlButtonEnv()
    env.reset()
    
    # Disable the button
    env.control_button.enabled = False
    
    # Pre-program it (shouldn't matter since disabled)
    env.control_button.controlled_agent = 1
    env.control_button.triggered_action = Actions.left
    
    # Human toggles the button
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Robot should NOT have forced_next_action set (button is disabled)
    assert env.agents[1].forced_next_action is None, "Disabled button should not set forced_next_action"


def test_control_button_encode():
    """Test that ControlButton encode() includes all attributes."""
    cb = ControlButton(World, trigger_color='yellow', controlled_color='grey')
    cb.controlled_agent = 1
    cb.triggered_action = Actions.forward
    cb.enabled = False
    
    # encode() requires the world argument to access OBJECT_TO_IDX and COLOR_TO_IDX
    encoding = cb.encode(World)
    
    # encoding should include: type, color, trigger_color, controlled_color, enabled, controlled_agent, triggered_action
    assert len(encoding) >= 5, "Encoding should include multiple attributes"
    assert encoding[0] == World.OBJECT_TO_IDX['controlbutton'], "First element should be object type"


def test_control_button_state_in_get_set_state():
    """Test that ControlButton mutable state is preserved in get_state/set_state."""
    env = ControlButtonEnv()
    env.reset()
    
    # Set mutable state
    env.control_button.controlled_agent = 1
    env.control_button.triggered_action = Actions.right
    env.control_button.enabled = False
    
    # Get state
    state = env.get_state()
    
    # Change state
    env.control_button.controlled_agent = None
    env.control_button.triggered_action = None
    env.control_button.enabled = True
    
    # Restore state
    env.set_state(state)
    
    # Verify mutable state is restored
    assert env.control_button.controlled_agent == 1, "controlled_agent should be restored"
    assert env.control_button.triggered_action == Actions.right, "triggered_action should be restored"
    assert env.control_button.enabled == False, "enabled state should be restored"


def test_control_button_forced_action_overrides_choice():
    """Test that forced_next_action truly overrides the agent's chosen action."""
    env = ControlButtonEnv()
    env.reset()
    
    # Pre-program and trigger
    env.control_button.controlled_agent = 1
    env.control_button.triggered_action = Actions.right
    
    # Human triggers
    actions = [Actions.toggle, Actions.still]
    env.step(actions)
    
    # Robot has forced_next_action
    assert env.agents[1].forced_next_action == Actions.right
    
    # Record position and direction
    robot_pos = tuple(env.agents[1].pos)
    robot_dir = env.agents[1].dir
    
    # Robot "chooses" forward but should turn right instead
    actions = [Actions.still, Actions.forward]
    env.step(actions)
    
    # Robot should have turned right, not moved forward
    expected_dir = (robot_dir + 1) % 4  # Right turn increases direction
    assert env.agents[1].dir == expected_dir, "Robot should have turned right"
    assert tuple(env.agents[1].pos) == robot_pos, "Robot should not have moved"


# ============================================================================
# ControlButton transition_probabilities() consistency tests
# ============================================================================


def _setup_cb_env_facing_button():
    """Helper: create ControlButtonEnv with grey robot facing the button.
    
    Layout (6x5):
        Walls around perimeter.
        Yellow (human) at (1,2) facing east.
        Grey (robot) at (3,2) facing *west* (front_pos = (2,2) = button).
        ControlButton at (2,2).
    """
    env = ControlButtonEnv()
    env.reset()
    # Turn robot to face west (towards button)
    env.step([Actions.still, Actions.left])
    env.step([Actions.still, Actions.left])
    assert tuple(env.agents[1].front_pos) == (2, 2), "Robot should face the button"
    return env


def test_control_button_transition_probs_programming():
    """transition_probabilities() must produce same state as step() for programming toggle."""
    env = _setup_cb_env_facing_button()
    
    state_before = env.get_state()
    actions = [Actions.still, Actions.toggle]  # Robot toggles → enters programming mode
    
    # Get transition_probabilities result
    tp_result = env.transition_probabilities(state_before, actions)
    assert len(tp_result) == 1, "Programming toggle is deterministic"
    tp_prob, tp_state = tp_result[0]
    assert tp_prob == 1.0
    
    # Execute step() to get the actual state
    env.step(actions)
    step_state = env.get_state()
    
    assert tp_state == step_state, (
        f"transition_probabilities and step() diverge after programming toggle.\n"
        f"  tp_state:   {tp_state}\n"
        f"  step_state: {step_state}"
    )
    
    # Verify the button is awaiting action
    # ControlButton state: ('controlbutton', x, y, enabled, controlled_agent, triggered_action, _awaiting_action)
    cb_state = [m for m in step_state[3] if m[0] == 'controlbutton'][0]
    assert cb_state[4] == 1, "controlled_agent should be 1 (robot)"
    assert cb_state[6] == True, "_awaiting_action should be True"


def test_control_button_transition_probs_recording():
    """transition_probabilities() must produce same state as step() for action recording."""
    env = _setup_cb_env_facing_button()
    
    # Robot toggles → enters programming mode
    env.step([Actions.still, Actions.toggle])
    assert env.control_button._awaiting_action == True
    
    state_before = env.get_state()
    actions = [Actions.still, Actions.forward]  # Robot moves forward → action gets recorded
    
    # Get transition_probabilities result
    tp_result = env.transition_probabilities(state_before, actions)
    assert len(tp_result) == 1
    tp_prob, tp_state = tp_result[0]
    assert tp_prob == 1.0
    
    # Execute step()
    env.step(actions)
    step_state = env.get_state()
    
    assert tp_state == step_state, (
        f"transition_probabilities and step() diverge after action recording.\n"
        f"  tp_state:   {tp_state}\n"
        f"  step_state: {step_state}"
    )
    
    # Verify button recorded the action
    cb_state = [m for m in step_state[3] if m[0] == 'controlbutton'][0]
    assert cb_state[5] == Actions.forward, "triggered_action should be forward"
    assert cb_state[6] == False, "_awaiting_action should be False after recording"


def test_control_button_transition_probs_trigger():
    """transition_probabilities() must produce same state as step() for triggering."""
    env = _setup_cb_env_facing_button()
    
    # Pre-program the button directly
    env.control_button.controlled_agent = 1
    env.control_button.triggered_action = Actions.left
    
    # Human is at (1,2) facing east → front_pos is (2,2) = button
    assert tuple(env.agents[0].front_pos) == (2, 2), "Human should face the button"
    
    state_before = env.get_state()
    actions = [Actions.toggle, Actions.still]  # Human triggers the button
    
    tp_result = env.transition_probabilities(state_before, actions)
    assert len(tp_result) == 1
    tp_prob, tp_state = tp_result[0]
    assert tp_prob == 1.0
    
    env.step(actions)
    step_state = env.get_state()
    
    assert tp_state == step_state, (
        f"transition_probabilities and step() diverge after trigger.\n"
        f"  tp_state:   {tp_state}\n"
        f"  step_state: {step_state}"
    )
    
    # Verify robot has forced_next_action set in the state
    robot_state = step_state[1][1]  # agent index 1
    assert robot_state[8] == Actions.left, "Robot should have forced_next_action=left"


def test_control_button_transition_probs_forced_override():
    """transition_probabilities() must correctly apply forced_next_action override."""
    env = _setup_cb_env_facing_button()
    
    # Pre-program and trigger the button
    env.control_button.controlled_agent = 1
    env.control_button.triggered_action = Actions.right
    env.step([Actions.toggle, Actions.still])  # Human triggers → robot gets forced_next_action
    
    assert env.agents[1].forced_next_action == Actions.right
    
    robot_dir_before = env.agents[1].dir
    robot_pos_before = tuple(env.agents[1].pos)
    state_before = env.get_state()
    
    # Robot "tries" forward but should be forced to turn right
    actions = [Actions.still, Actions.forward]
    
    tp_result = env.transition_probabilities(state_before, actions)
    assert len(tp_result) == 1
    tp_prob, tp_state = tp_result[0]
    assert tp_prob == 1.0
    
    env.step(actions)
    step_state = env.get_state()
    
    assert tp_state == step_state, (
        f"transition_probabilities and step() diverge after forced override.\n"
        f"  tp_state:   {tp_state}\n"
        f"  step_state: {step_state}"
    )
    
    # Verify robot turned right (forced) not moved forward (chosen)
    expected_dir = (robot_dir_before + 1) % 4
    robot_state = step_state[1][1]
    assert robot_state[2] == expected_dir, "Robot should have turned right (forced)"
    assert (robot_state[0], robot_state[1]) == robot_pos_before, "Robot should not have moved"
    # forced_next_action should be cleared in successor state
    assert robot_state[8] is None, "forced_next_action should be cleared after use"


def test_control_button_full_lifecycle_via_transition_probs():
    """Full lifecycle: program → record → trigger → forced override, all via transition_probabilities()."""
    env = _setup_cb_env_facing_button()
    
    # STEP 1: Robot toggles to enter programming mode
    state = env.get_state()
    actions_1 = [Actions.still, Actions.toggle]
    tp1 = env.transition_probabilities(state, actions_1)
    assert len(tp1) == 1
    state_after_program = tp1[0][1]
    
    # Verify programming mode in state
    cb = [m for m in state_after_program[3] if m[0] == 'controlbutton'][0]
    assert cb[4] == 1, "controlled_agent should be robot (1)"
    assert cb[6] == True, "_awaiting_action should be True"
    
    # STEP 2: Robot turns left → recorded as programmed action
    actions_2 = [Actions.still, Actions.left]
    tp2 = env.transition_probabilities(state_after_program, actions_2)
    assert len(tp2) == 1
    state_after_record = tp2[0][1]
    
    # Verify action was recorded
    cb = [m for m in state_after_record[3] if m[0] == 'controlbutton'][0]
    assert cb[5] == Actions.left, "triggered_action should be left"
    assert cb[6] == False, "_awaiting_action should be False after recording"
    
    # STEP 3: Human triggers the button
    # Human is at (1,2) facing east, button at (2,2) — needs to be facing button
    actions_3 = [Actions.toggle, Actions.still]
    tp3 = env.transition_probabilities(state_after_record, actions_3)
    assert len(tp3) == 1
    state_after_trigger = tp3[0][1]
    
    # Verify forced_next_action set on robot
    robot_state = state_after_trigger[1][1]
    assert robot_state[8] == Actions.left, "Robot should have forced_next_action=left"
    
    # STEP 4: Robot's chosen action (forward) is overridden by forced (left)
    robot_dir_before = state_after_trigger[1][1][2]
    actions_4 = [Actions.still, Actions.forward]  # Robot tries forward
    tp4 = env.transition_probabilities(state_after_trigger, actions_4)
    assert len(tp4) == 1
    state_final = tp4[0][1]
    
    # Verify robot turned left (forced override), not moved forward
    robot_state_final = state_final[1][1]
    expected_dir = (robot_dir_before - 1) % 4
    assert robot_state_final[2] == expected_dir, (
        f"Robot should have turned left. Expected dir={expected_dir}, got dir={robot_state_final[2]}"
    )
    assert robot_state_final[8] is None, "forced_next_action should be cleared"


def test_control_button_awaiting_action_in_state():
    """Verify that _awaiting_action is included in get_state() and set_state() round-trips."""
    env = _setup_cb_env_facing_button()
    
    # Robot toggles → enters programming mode
    env.step([Actions.still, Actions.toggle])
    assert env.control_button._awaiting_action == True
    
    state = env.get_state()
    
    # Verify _awaiting_action is in state
    cb_state = [m for m in state[3] if m[0] == 'controlbutton'][0]
    assert len(cb_state) >= 7, f"ControlButton state should have 7+ elements, got {len(cb_state)}"
    assert cb_state[6] == True, "_awaiting_action should be True in state"
    
    # Modify the live object
    env.control_button._awaiting_action = False
    assert env.control_button._awaiting_action == False
    
    # Restore state
    env.set_state(state)
    assert env.control_button._awaiting_action == True, "_awaiting_action should be restored by set_state"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

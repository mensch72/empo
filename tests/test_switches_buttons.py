#!/usr/bin/env python3
"""
Test switches and buttons for Issue #25.

Tests for:
- KillButton: overlappable floor that kills agents when stepped on
- PauseSwitch: non-overlappable switch that pauses agents when toggled
- DisablingSwitch: non-overlappable switch that disables KillButtons or PauseSwitches
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

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
        
        # Place yellow agent at (1, 2)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # facing right
        self.grid.set(1, 2, self.agents[0])
        
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
    """Test that KillButton kills target agents when stepped on by trigger agent."""
    env = KillButtonEnv()
    env.reset()
    
    # Verify initial state
    assert not env.agents[0].terminated, "Yellow agent should not be terminated initially"
    assert not env.agents[1].terminated, "Grey agent should not be terminated initially"
    
    # Yellow agent moves forward onto kill button
    actions = [Actions.forward, Actions.still]  # Yellow moves, grey stays
    env.step(actions)
    
    # Verify grey agent is now terminated (killed)
    assert not env.agents[0].terminated, "Yellow agent should not be terminated"
    assert env.agents[1].terminated, "Grey agent should be terminated after yellow stepped on kill button"


def test_kill_button_disabled():
    """Test that disabled KillButton does not kill agents."""
    env = KillButtonEnv()
    env.reset()
    
    # Disable the kill button
    env.kill_button.enabled = False
    
    # Yellow agent moves forward onto kill button
    actions = [Actions.forward, Actions.still]
    env.step(actions)
    
    # Verify grey agent is NOT terminated
    assert not env.agents[1].terminated, "Grey agent should not be terminated when kill button is disabled"


def test_kill_button_wrong_trigger_color():
    """Test that KillButton only responds to correct trigger color."""
    env = KillButtonEnv()
    env.reset()
    
    # Set trigger color to something else
    env.kill_button.trigger_color = 'blue'
    
    # Yellow agent moves forward onto kill button
    actions = [Actions.forward, Actions.still]
    env.step(actions)
    
    # Grey agent should NOT be terminated because yellow is not the trigger color
    assert not env.agents[1].terminated, "Grey agent should not be terminated when wrong trigger color steps on button"


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
    
    # Yellow steps on kill button, killing grey
    actions = [Actions.forward, Actions.still]
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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

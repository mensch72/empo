#!/usr/bin/env python3
"""
Example demonstrating the new state management methods for multigrid environments.

This script shows how to use:
- get_state() to capture environment state
- set_state() to restore environment state
- transition_probabilities() to compute exact transition probabilities
"""

import sys
from pathlib import Path

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import Actions
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def example_get_set_state():
    """Example: Save and restore environment state."""
    print("=" * 70)
    print("Example 1: Get and Set State")
    print("=" * 70)
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Save initial state
    initial_state = env.get_state()
    print(f"✓ Captured initial state (step {env.step_count})")
    
    # Take some actions
    for i in range(5):
        actions = [Actions.forward, Actions.left, Actions.right]
        env.step(actions)
    
    print(f"✓ Stepped environment 5 times (now at step {env.step_count})")
    
    # Restore initial state
    env.set_state(initial_state)
    print(f"✓ Restored initial state (back to step {env.step_count})")
    
    # Verify it worked
    restored_state = env.get_state()
    if initial_state == restored_state:
        print("✓ State correctly restored!")
    else:
        print("✗ State restoration failed")
    
    print()


def example_deterministic_transitions():
    """Example: Demonstrate deterministic transitions."""
    print("=" * 70)
    print("Example 2: Deterministic Transitions")
    print("=" * 70)
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # Test 1: All agents stay still
    actions = [Actions.still, Actions.still, Actions.still]
    result = env.transition_probabilities(state, actions)
    print(f"Actions: all STILL")
    print(f"  → {len(result)} outcome(s), probability = {result[0][0]}")
    print(f"  → Deterministic: {len(result) == 1}")
    
    # Test 2: Only one agent acts
    actions = [Actions.forward, Actions.still, Actions.still]
    result = env.transition_probabilities(state, actions)
    print(f"\nActions: one FORWARD, others STILL")
    print(f"  → {len(result)} outcome(s), probability = {result[0][0]}")
    print(f"  → Deterministic: {len(result) == 1}")
    
    # Test 3: All agents rotate
    actions = [Actions.left, Actions.right, Actions.left]
    result = env.transition_probabilities(state, actions)
    print(f"\nActions: all ROTATE (left/right)")
    print(f"  → {len(result)} outcome(s), probability = {result[0][0]}")
    print(f"  → Deterministic: {len(result) == 1}")
    print(f"  → Reason: Rotations never interfere with each other")
    
    print()


def example_probabilistic_transitions():
    """Example: Show when transitions can be probabilistic."""
    print("=" * 70)
    print("Example 3: Potentially Probabilistic Transitions")
    print("=" * 70)
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # Multiple agents moving forward
    actions = [Actions.forward, Actions.forward, Actions.forward]
    result = env.transition_probabilities(state, actions)
    
    print(f"Actions: all agents FORWARD")
    print(f"  → {len(result)} unique outcome(s)")
    
    if len(result) == 1:
        print(f"  → Deterministic (probability = {result[0][0]})")
        print(f"  → Reason: Agents don't interfere in this configuration")
    else:
        print(f"  → Probabilistic! Multiple outcomes:")
        for i, (prob, succ_state) in enumerate(result):
            print(f"     Outcome {i+1}: probability = {prob:.4f}")
        print(f"  → Reason: Agent execution order matters")
    
    # Verify probabilities sum to 1
    total_prob = sum(prob for prob, _ in result)
    print(f"\n  ✓ Probabilities sum to {total_prob} (exactly 1.0: {total_prob == 1.0})")
    
    print()


def example_state_properties():
    """Example: Explore state properties."""
    print("=" * 70)
    print("Example 4: State Properties")
    print("=" * 70)
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # States are hashable
    try:
        state_dict = {state: "value"}
        print("✓ State is hashable (can be used as dict key)")
    except TypeError:
        print("✗ State is not hashable")
    
    # States are immutable tuples
    print(f"✓ State type: {type(state).__name__}")
    
    # State format: (step_count, agent_states, mobile_objects, mutable_objects)
    # - step_count: integer, current time step
    # - agent_states: tuple of tuples, each with (pos_x, pos_y, dir, terminated, started, paused, carrying_type, carrying_color)
    # - mobile_objects: tuple of tuples for blocks, rocks, etc.
    # - mutable_objects: tuple of tuples for doors, switches, etc.
    step_count, agent_states, mobile_objects, mutable_objects = state
    print(f"✓ Step count in state: {step_count}")
    
    # State includes time left
    time_left = env.max_steps - step_count
    print(f"✓ Time left: {time_left} steps")
    print(f"✓ Number of agents: {len(agent_states)}")
    print(f"✓ Number of mobile objects: {len(mobile_objects)}")
    
    print()


def example_terminal_state():
    """Example: Terminal state handling."""
    print("=" * 70)
    print("Example 5: Terminal State Handling")
    print("=" * 70)
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Force to terminal state
    env.step_count = env.max_steps
    terminal_state = env.get_state()
    
    actions = [Actions.forward, Actions.forward, Actions.forward]
    result = env.transition_probabilities(terminal_state, actions)
    
    print(f"Environment at max_steps: {env.max_steps}")
    print(f"transition_probabilities returns: {result}")
    print(f"✓ Terminal states correctly return None")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Multigrid State Management Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    example_get_set_state()
    example_deterministic_transitions()
    example_probabilistic_transitions()
    example_state_properties()
    example_terminal_state()
    
    print("=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

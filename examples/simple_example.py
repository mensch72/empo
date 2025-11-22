#!/usr/bin/env python3
"""
Simple example demonstrating the basic usage of the EMPO framework.

This script shows how to:
1. Set up an environment
2. Initialize agents
3. Run a simple training loop
4. Save and visualize results
"""

import torch
import numpy as np
from pathlib import Path


def main():
    """Run a simple demonstration of the framework."""
    print("=" * 60)
    print("EMPO Framework - Simple Example")
    print("=" * 60)
    
    # Check system setup
    print("\n1. System Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   NumPy version: {np.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path("./outputs/example")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Output directory: {output_dir}")
    
    # Simulate a simple agent-environment interaction
    print("\n3. Running simple agent-environment interaction...")
    state_dim = 4
    action_dim = 2
    num_steps = 100
    
    # Create a simple random policy (placeholder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.nn.Linear(state_dim, action_dim).to(device)
    
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Device: {device}")
    
    # Simulate episode
    total_reward = 0.0
    for step in range(num_steps):
        # Random state (placeholder for actual environment observation)
        state = torch.randn(1, state_dim).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action_logits = policy(state)
            action = torch.argmax(action_logits, dim=1)
        
        # Simulate reward
        reward = np.random.random()
        total_reward += reward
        
        if step % 20 == 0:
            print(f"   Step {step:3d}: action={action.item()}, reward={reward:.3f}")
    
    avg_reward = total_reward / num_steps
    print(f"\n4. Results:")
    print(f"   Total steps: {num_steps}")
    print(f"   Average reward: {avg_reward:.3f}")
    
    # Save a dummy checkpoint
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'avg_reward': avg_reward,
    }, checkpoint_path)
    print(f"\n5. Checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

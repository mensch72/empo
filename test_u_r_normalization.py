#!/usr/bin/env python
"""Quick diagnostic test for U_r normalization."""

import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "vendor/multigrid")
sys.path.insert(0, "vendor/ai_transport")
sys.path.insert(0, "multigrid_worlds")

from collections import deque

# Simulate U_r values during warmup
print("[TEST] Simulating warmup phase U_r collection...")
warmup_buffer = deque(maxlen=100)

# Simulate raw U_r values from network (should be in [-1, 0] range)
np.random.seed(42)
for i in range(100):
    # Most values in [-0.5, -0.01], some near boundary
    u_r_raw = np.random.uniform(-0.5, -0.01)
    if i % 20 == 0:
        print(f"  Step {i}: u_r_raw={u_r_raw:.6f}")
    warmup_buffer.append(u_r_raw)

print(f"\n[TEST] Warmup buffer collected {len(warmup_buffer)} samples")

# Freeze statistics
if len(warmup_buffer) > 0:
    u_r_array = np.array(list(warmup_buffer))
    u_r_mean = float(np.mean(u_r_array))
    u_r_std = float(np.std(u_r_array))
    
    print(f"[TEST] Statistics computed:")
    print(f"  μ = {u_r_mean:.6f}")
    print(f"  σ = {u_r_std:.6f}")
    print(f"  min = {np.min(u_r_array):.6f}")
    print(f"  max = {np.max(u_r_array):.6f}")

# Simulate main phase with normalization
print(f"\n[TEST] Simulating main phase with normalization...")
u_r_scale = 1995.262315  # From diagnostic output above

for i in range(5):
    # New U_r value from network
    u_r_raw = np.random.uniform(-0.5, -0.01)
    
    # Normalize
    u_r_normalized = (u_r_raw - u_r_mean) / (u_r_std + 1e-8)
    
    # Scale
    u_r_final = u_r_normalized / u_r_scale
    
    print(f"  Step {i}: raw={u_r_raw:.6f} → norm={u_r_normalized:.6f} → final={u_r_final:.6f}")

print("\n[TEST] After fix (skip scale during normalization):")
for i in range(5):
    # New U_r value from network
    u_r_raw = np.random.uniform(-0.5, -0.01)
    
    # Normalize (NO scale division after normalization)
    u_r_normalized = (u_r_raw - u_r_mean) / (u_r_std + 1e-8)
    
    print(f"  Step {i}: raw={u_r_raw:.6f} → norm={u_r_normalized:.6f}")

print("\n[TEST] Done")

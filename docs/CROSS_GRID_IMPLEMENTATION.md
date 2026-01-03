# Cross-Grid Policy Loading Implementation Summary

## Overview

This implementation enables loading policies trained on larger multigrid environments for use on smaller grids. This is achieved by:
1. Padding smaller grids with grey walls to match the trained policy's expected dimensions
2. Using absolute integer coordinates (not normalized) throughout all encoders

## Changes Made

### 1. Modified `_validate_grid_dimensions()` in `neural_policy_prior.py`
- **Before**: Rejected any grid dimension mismatch
- **After**: Only rejects when environment grid is LARGER than trained grid
- **Rationale**: Allows loading from large→small, prevents small→large (which would cause out-of-bounds errors)

### 2. Updated `_encode_grid()` in `state_encoder.py`
- **Before**: Assumed world dimensions match encoder dimensions
- **After**: Detects actual world dimensions and pads extra space with walls
- **Implementation**:
  - Gets actual world height/width from world_model
  - If smaller than encoder dimensions, fills outer area with wall channel
  - Only encodes objects/agents within actual world bounds

### 3. Updated Tests
- Modified `test_load_dimension_mismatch()` to expect new error message
- Added comprehensive test suite in `test_cross_grid_loading.py`:
  - Tests state encoder padding with walls
  - Tests saving on 15x15 and loading on 10x10
  - Tests rejection of small→large loading
  - Tests that equal dimensions still work

### 4. Documentation
- Updated `docs/ENCODER_ARCHITECTURE.md` with cross-grid loading section
- Enhanced `load()` docstring with detailed explanation
- Created example script `examples/cross_grid_policy_demo.py`

## Coordinate System Verification

All multigrid encoders use **absolute integer coordinates** (not normalized):

### State Encoder - Agent Positions
```python
features[0] = float(agent_state[0])  # x (absolute)
features[1] = float(agent_state[1])  # y (absolute)
```

### Goal Encoder - Goal Coordinates
```python
# Input: (x1, y1, x2, y2) as absolute grid coordinates
coords = torch.tensor([[float(x1), float(y1), float(x2), float(y2)]])
```

This was already the case before our changes, which is why the implementation works correctly.

## How It Works

### Example: Load 15x15 policy on 10x10 world

```
Original training grid (15x15):    Deployment world (10x10):
┌─────────────────┐                ┌───────────┐
│                 │                │           │
│                 │                │  Actual   │
│                 │                │  World    │
│                 │                │  (10x10)  │
│                 │                │           │
│                 │                └───────────┘
│                 │
│                 │
└─────────────────┘

Encoded state (15x15):
┌───────────┬─────┐
│           │ W W │  W = Walls
│  Actual   │ W W │  (padding)
│  World    │ W W │
│  (10x10)  │ W W │
│           │ W W │
├───────────┼─────┤
│ W W W W W W W W │
│ W W W W W W W W │
└─────────────────┘
   5x15      5x5
   walls     walls
```

The encoder maintains 15x15 dimensions, but only the 10x10 area contains the actual world. The rest is padded with walls (channel 0 = 1.0).

## Benefits

1. **Transfer Learning**: Train once on large diverse grids, deploy on smaller grids
2. **Efficient Training**: Use large training environments without retraining for deployment
3. **Backward Compatibility**: Upgrade training environments without breaking smaller deployments
4. **No Coordinate Issues**: Absolute coordinates remain valid across grid sizes

## Limitations

- **One Direction**: Can only load large→small, not small→large
  - Attempting small→large raises `ValueError` with clear message
  - Rationale: Small grid policies would have coordinates out of bounds in large grids
  
- **Wall Padding Only**: Padding uses grey walls (channel 0)
  - Could potentially be extended to use other object types if needed

## Testing

All tests pass:
- ✅ Core neural policy prior tests
- ✅ Cross-grid loading tests
- ✅ State encoder padding tests
- ✅ Equal dimension tests (backward compatibility)
- ✅ Rejection of unsupported direction (small→large)

## Example Usage

```python
from empo.nn_based.multigrid import MultiGridNeuralHumanPolicyPrior

# Train on large grid (15x15)
large_world = create_world(15, 15)
prior = train_policy(large_world)
prior.save('policy_15x15.pt')

# Load on small grid (10x10)
small_world = create_world(10, 10)
loaded_prior = MultiGridNeuralHumanPolicyPrior.load(
    'policy_15x15.pt',
    world_model=small_world,
    human_agent_indices=[0],
    goal_sampler=goal_sampler,
    device='cpu'
)

# Use loaded policy - it works!
q_values = loaded_prior.q_network.encode_and_forward(
    state, small_world, agent_idx=0, goal=goal
)
```

See `examples/cross_grid_policy_demo.py` for a complete working example.

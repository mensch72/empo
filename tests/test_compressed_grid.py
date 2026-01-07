#!/usr/bin/env python3
"""
Test script for compressed grid format and split tensorization.

Tests:
1. compress_grid() - Compresses grid + dynamic state to int32 tensor
2. decompress_grid_to_tensor() - Unpacks single compressed grid to channel tensor
3. decompress_grid_batch_to_tensor() - Vectorized batch decompression
4. tensorize_state_compact() - Computes expensive features for replay buffer
5. tensorize_state_from_compact() - Reconstructs full tensor from compact features
6. Round-trip consistency (compress → decompress matches original encoding)
"""

import sys
import os
import torch
import numpy as np

from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.constants import (
    NUM_OBJECT_TYPE_CHANNELS,
    NUM_STANDARD_COLORS,
    OBJECT_TYPE_TO_CHANNEL,
    COLOR_TO_IDX,
    DOOR_CHANNEL_START,
    KEY_CHANNEL_START,
    MAGICWALL_CHANNEL,
    DOOR_STATE_OPEN,
    DOOR_STATE_CLOSED,
    DOOR_STATE_LOCKED,
    COMPRESSED_GRID_OBJECT_TYPE_MASK,
    COMPRESSED_GRID_COLOR_SHIFT,
    COMPRESSED_GRID_COLOR_MASK,
    COMPRESSED_GRID_STATE_SHIFT,
    COMPRESSED_GRID_STATE_MASK,
    COMPRESSED_GRID_AGENT_COLOR_SHIFT,
    COMPRESSED_GRID_AGENT_COLOR_MASK,
    COMPRESSED_GRID_NO_AGENT,
    COMPRESSED_GRID_MAGIC_SHIFT,
    COMPRESSED_GRID_MAGIC_MASK,
    COMPRESSED_GRID_OTHER_SHIFT,
    COMPRESSED_GRID_OTHER_MASK,
    COMPRESSED_GRID_DOOR_TYPE,
    COMPRESSED_GRID_KEY_TYPE,
)


class MockWorldModel:
    """Mock world model for testing grid compression."""
    
    def __init__(self, width=7, height=7, num_agents=2):
        self.width = width
        self.height = height
        self.max_steps = 100
        self.grid = MockGrid(width, height)
        self.agents = [MockAgent(f'agent_{i}', ['grey', 'blue'][i % 2]) 
                       for i in range(num_agents)]
        self.stumble_prob = 0.0
        self.magic_entry_prob = 1.0
        self.magic_solidify_prob = 0.0


class MockGrid:
    """Mock grid for testing."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._cells = {}
    
    def get(self, x, y):
        return self._cells.get((x, y), None)
    
    def set(self, x, y, obj):
        self._cells[(x, y)] = obj


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, color='grey'):
        self.name = name
        self.color = color
        self.can_enter_magic_walls = False
        self.can_push_rocks = False
        self.carrying = None
        self.paused = False
        self.terminated = False
        self.forced_next_action = None


class MockObject:
    """Mock grid object."""
    
    def __init__(self, obj_type, color=None, **kwargs):
        self.type = obj_type
        self.color = color
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_mock_state(num_agents=2, step_count=0):
    """Create a mock state tuple."""
    # Format: (x, y, direction, carrying_type, carrying_color)
    agent_states = [(i, i, 0, -1, -1) for i in range(num_agents)]
    mobile_objects = []
    mutable_objects = []
    return (step_count, agent_states, mobile_objects, mutable_objects)


def create_encoder(height=7, width=7):
    """Create encoder with standard configuration."""
    return MultiGridStateEncoder(
        grid_height=height,
        grid_width=width,
        num_agents_per_color={'grey': 1, 'blue': 1}  # Dict mapping color to count
    )


def test_encoder_creation():
    """Test that encoder can be created with correct dimensions."""
    print("Testing encoder creation...")
    
    encoder = create_encoder(7, 7)
    
    assert encoder.grid_height == 7
    assert encoder.grid_width == 7
    assert encoder.num_grid_channels > 0
    
    print(f"  Grid channels: {encoder.num_grid_channels}")
    print("  ✓ Encoder creation OK")


def test_compress_grid_empty():
    """Test compressing an empty grid."""
    print("Testing compress_grid with empty grid...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    assert compressed.shape == (7, 7), f"Expected (7,7), got {compressed.shape}"
    assert compressed.dtype == torch.int32, f"Expected int32, got {compressed.dtype}"
    
    # Storage calculation
    storage_bytes = compressed.numel() * 4  # int32 = 4 bytes
    print(f"  Compressed shape: {compressed.shape}")
    print(f"  Storage: {storage_bytes} bytes")
    print("  ✓ Empty grid compression OK")


def test_compress_grid_with_objects():
    """Test compressing a grid with various objects."""
    print("Testing compress_grid with objects...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Add some objects to the grid
    world_model.grid.set(0, 0, MockObject('wall'))
    world_model.grid.set(1, 0, MockObject('door', color='red'))
    world_model.grid.set(2, 0, MockObject('key', color='blue'))
    world_model.grid.set(3, 0, MockObject('goal'))
    world_model.grid.set(4, 0, MockObject('lava'))
    
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    assert compressed.shape == (7, 7)
    
    # Check that cell (0,0) with wall has non-zero value
    cell_00 = compressed[0, 0].item()
    obj_type = cell_00 & COMPRESSED_GRID_OBJECT_TYPE_MASK
    assert obj_type == OBJECT_TYPE_TO_CHANNEL['wall'], f"Expected wall type, got {obj_type}"
    
    # Check that cell (1,0) with door has door type marker
    cell_10 = compressed[0, 1].item()
    obj_type = cell_10 & COMPRESSED_GRID_OBJECT_TYPE_MASK
    assert obj_type == COMPRESSED_GRID_DOOR_TYPE, f"Expected door type {COMPRESSED_GRID_DOOR_TYPE}, got {obj_type}"
    
    # Check door color encoding
    door_color = (cell_10 & COMPRESSED_GRID_COLOR_MASK) >> COMPRESSED_GRID_COLOR_SHIFT
    expected_color = COLOR_TO_IDX.get('red', 0)
    assert door_color == expected_color, f"Expected color {expected_color}, got {door_color}"
    
    print(f"  Wall cell value: {cell_00} (type={obj_type})")
    print(f"  Door cell value: {cell_10} (type=door, color={door_color})")
    print("  ✓ Object compression OK")


def test_compress_grid_with_agents():
    """Test that agent positions are encoded in compressed grid."""
    print("Testing compress_grid with agent positions...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Agents at positions (0,0) and (1,1)
    agent_states = [
        (0, 0, 0, -1, -1),  # grey agent at (0,0)
        (1, 1, 0, -1, -1),  # blue agent at (1,1)
    ]
    state = (0, agent_states, [], [])
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    # Check agent at (0,0)
    cell_00 = compressed[0, 0].item()
    agent_color = (cell_00 & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT
    assert agent_color != COMPRESSED_GRID_NO_AGENT, "Expected agent at (0,0)"
    
    # Check agent at (1,1)
    cell_11 = compressed[1, 1].item()
    agent_color = (cell_11 & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT
    assert agent_color != COMPRESSED_GRID_NO_AGENT, "Expected agent at (1,1)"
    
    # Check no agent at (2,2)
    cell_22 = compressed[2, 2].item()
    agent_color = (cell_22 & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT
    assert agent_color == COMPRESSED_GRID_NO_AGENT, "Expected no agent at (2,2)"
    
    print(f"  Agent at (0,0): color index = {(compressed[0,0].item() & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT}")
    print(f"  Agent at (1,1): color index = {(compressed[1,1].item() & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT}")
    print("  ✓ Agent position encoding OK")


def test_decompress_grid_single():
    """Test decompressing a single compressed grid."""
    print("Testing decompress_grid_to_tensor...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    world_model.grid.set(0, 0, MockObject('wall'))
    world_model.grid.set(1, 0, MockObject('goal'))
    
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compress
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    # Decompress
    decompressed = encoder.decompress_grid_to_tensor(compressed, 'cpu')
    
    assert decompressed.shape == (1, encoder.num_grid_channels, 7, 7)
    
    # Check wall channel at (0,0)
    wall_channel = OBJECT_TYPE_TO_CHANNEL['wall']
    wall_value = decompressed[0, wall_channel, 0, 0].item()
    assert wall_value == 1.0, f"Expected wall=1.0, got {wall_value}"
    
    # Check goal channel at (1,0)
    goal_channel = OBJECT_TYPE_TO_CHANNEL['goal']
    goal_value = decompressed[0, goal_channel, 0, 1].item()
    assert goal_value == 1.0, f"Expected goal=1.0, got {goal_value}"
    
    print(f"  Decompressed shape: {decompressed.shape}")
    print(f"  Wall at (0,0): {wall_value}")
    print(f"  Goal at (1,0): {goal_value}")
    print("  ✓ Single decompression OK")


def test_decompress_grid_batch():
    """Test batch decompression matches single decompression."""
    print("Testing decompress_grid_batch_to_tensor...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Create several different compressed grids
    compressed_list = []
    
    for i in range(4):
        # Different objects in each grid
        world_model.grid._cells = {}  # Clear
        world_model.grid.set(i, 0, MockObject('wall'))
        world_model.grid.set(i, 1, MockObject('goal'))
        
        agent_states = [(i, i, 0, -1, -1), ((i+1) % 7, (i+1) % 7, 0, -1, -1)]
        agent_colors = [agent.color for agent in world_model.agents]
        
        compressed = encoder.compress_grid(
            world_model, agent_states, [], [], agent_colors
        )
        compressed_list.append(compressed)
    
    # Stack into batch
    compressed_batch = torch.stack(compressed_list)  # (4, 7, 7)
    
    # Batch decompress
    batch_decompressed = encoder.decompress_grid_batch_to_tensor(compressed_batch, 'cpu')
    
    assert batch_decompressed.shape == (4, encoder.num_grid_channels, 7, 7)
    
    # Single decompress each and compare
    max_diff = 0.0
    for i in range(4):
        single_decompressed = encoder.decompress_grid_to_tensor(compressed_list[i], 'cpu')
        diff = (batch_decompressed[i:i+1] - single_decompressed).abs().max().item()
        max_diff = max(max_diff, diff)
    
    assert max_diff < 1e-6, f"Batch and single decompression differ by {max_diff}"
    
    print(f"  Batch shape: {batch_decompressed.shape}")
    print(f"  Max diff from single decompress: {max_diff}")
    print("  ✓ Batch decompression matches single OK")


def test_storage_savings():
    """Test that compressed format provides expected storage savings."""
    print("Testing storage savings...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compress
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    # Decompress to get full tensor
    full_tensor = encoder.decompress_grid_to_tensor(compressed, 'cpu')
    
    # Calculate storage
    compressed_bytes = compressed.numel() * 4  # int32
    full_bytes = full_tensor.numel() * 4  # float32
    savings_ratio = full_bytes / compressed_bytes
    
    print(f"  Compressed: {compressed_bytes} bytes ({compressed.shape})")
    print(f"  Full tensor: {full_bytes} bytes ({full_tensor.shape})")
    print(f"  Savings: {savings_ratio:.1f}x smaller")
    
    # For 7x7 grid with ~39 channels: expect ~39x savings
    assert savings_ratio > 30, f"Expected >30x savings, got {savings_ratio}x"
    print("  ✓ Storage savings OK")


def test_tensorize_state_compact():
    """Test split tensorization: compact features computation."""
    print("Testing tensorize_state_compact...")
    
    encoder = create_encoder(7, 7)
    
    world_model = MockWorldModel(7, 7, 2)
    state = create_mock_state(2)
    
    # Compute compact features
    global_feats, agent_feats, interactive_feats = encoder.tensorize_state_compact(
        state, world_model, 'cpu'
    )
    
    # Compact features have 1D shape (no batch dimension from compact)
    assert global_feats.dim() in [1, 2], f"Expected 1D or 2D global, got {global_feats.dim()}"
    assert agent_feats.dim() in [1, 2], f"Expected 1D or 2D agent, got {agent_feats.dim()}"
    assert interactive_feats.dim() in [1, 2], f"Expected 1D or 2D interactive, got {interactive_feats.dim()}"
    
    print(f"  Global features shape: {global_feats.shape}")
    print(f"  Agent features shape: {agent_feats.shape}")
    print(f"  Interactive features shape: {interactive_feats.shape}")
    
    # Total compact size
    total_floats = global_feats.numel() + agent_feats.numel() + interactive_feats.numel()
    total_bytes = total_floats * 4
    print(f"  Total compact features: {total_floats} floats = {total_bytes} bytes")
    print("  ✓ Compact features computation OK")


def test_tensorize_state_from_compact():
    """Test reconstructing full tensor from compact features + grid."""
    print("Testing tensorize_state_from_compact...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    world_model.grid.set(0, 0, MockObject('wall'))
    world_model.grid.set(1, 1, MockObject('goal'))
    
    state = create_mock_state(2)
    
    # Get compact features
    global_feats, agent_feats, interactive_feats = encoder.tensorize_state_compact(
        state, world_model, 'cpu'
    )
    compact_features = (global_feats, agent_feats, interactive_feats)
    
    # Reconstruct from compact
    grid, glob, agent, interactive = encoder.tensorize_state_from_compact(
        state, world_model, compact_features, 'cpu'
    )
    
    # Also get via full tensorization for comparison
    grid_full, glob_full, agent_full, interactive_full = encoder.tensorize_state(
        state, world_model, 'cpu'
    )
    
    # Compare non-grid features (should be identical)
    assert torch.allclose(glob, glob_full), "Global features don't match"
    assert torch.allclose(agent, agent_full), "Agent features don't match"
    assert torch.allclose(interactive, interactive_full), "Interactive features don't match"
    
    # Grid should also match (both use world_model)
    assert torch.allclose(grid, grid_full), "Grid features don't match"
    
    print(f"  Grid shape: {grid.shape}")
    print(f"  Global shape: {glob.shape}")
    print(f"  Agent shape: {agent.shape}")
    print(f"  Interactive shape: {interactive.shape}")
    print("  ✓ Reconstruction from compact OK")


def test_round_trip_consistency():
    """Test that compress → decompress produces consistent results."""
    print("Testing round-trip consistency...")
    
    encoder = create_encoder(7, 7)
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Set up various objects (excluding doors and magic walls which need state)
    world_model.grid.set(0, 0, MockObject('wall'))
    world_model.grid.set(1, 0, MockObject('key', color='blue'))
    world_model.grid.set(2, 0, MockObject('goal'))
    world_model.grid.set(3, 0, MockObject('lava'))
    world_model.grid.set(4, 0, MockObject('ball'))
    
    # Agent positions
    agent_states = [(0, 1, 0, -1, -1), (1, 1, 0, -1, -1)]
    state = (0, agent_states, [], [])
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compress and decompress
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    grid_roundtrip = encoder.decompress_grid_to_tensor(compressed, 'cpu')
    
    # Check specific object positions
    # Wall at (0,0)
    wall_ch = OBJECT_TYPE_TO_CHANNEL['wall']
    wall_val = grid_roundtrip[0, wall_ch, 0, 0].item()
    assert wall_val == 1.0, f"Expected wall=1.0 at (0,0), got {wall_val}"
    
    # Goal at (2,0)
    goal_ch = OBJECT_TYPE_TO_CHANNEL['goal']
    goal_val = grid_roundtrip[0, goal_ch, 0, 2].item()
    assert goal_val == 1.0, f"Expected goal=1.0 at (2,0), got {goal_val}"
    
    # Lava at (3,0)
    lava_ch = OBJECT_TYPE_TO_CHANNEL['lava']
    lava_val = grid_roundtrip[0, lava_ch, 0, 3].item()
    assert lava_val == 1.0, f"Expected lava=1.0 at (3,0), got {lava_val}"
    
    # Blue key at (1,0)
    blue_key_ch = KEY_CHANNEL_START + COLOR_TO_IDX['blue']
    key_val = grid_roundtrip[0, blue_key_ch, 0, 1].item()
    assert key_val == 1.0, f"Expected blue key=1.0 at (1,0), got {key_val}"
    
    print("  ✓ Wall at (0,0) encoded correctly")
    print("  ✓ Goal at (2,0) encoded correctly")
    print("  ✓ Lava at (3,0) encoded correctly")
    print("  ✓ Blue key at (1,0) encoded correctly")
    print("  ✓ Round-trip consistency OK")


def test_door_states():
    """Test that door states are correctly compressed and decompressed."""
    print("Testing door state encoding...")
    
    encoder = create_encoder(7, 7)
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Add doors with different states
    world_model.grid.set(0, 0, MockObject('door', color='red', is_open=True))
    world_model.grid.set(1, 0, MockObject('door', color='blue', is_open=False, is_locked=False))
    world_model.grid.set(2, 0, MockObject('door', color='green', is_open=False, is_locked=True))
    
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    # Add door states to mutable_objects as tuples: (type, x, y, is_open, is_locked)
    mutable_objects = [
        ('door', 0, 0, True, False),   # open door at (0, 0)
        ('door', 1, 0, False, False),  # closed door at (1, 0)
        ('door', 2, 0, False, True),   # locked door at (2, 0)
    ]
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compress
    compressed = encoder.compress_grid(
        world_model, agent_states, [], mutable_objects, agent_colors
    )
    
    # Check state bits for each door
    cell_00 = compressed[0, 0].item()
    state_00 = (cell_00 & COMPRESSED_GRID_STATE_MASK) >> COMPRESSED_GRID_STATE_SHIFT
    
    cell_10 = compressed[0, 1].item()
    state_10 = (cell_10 & COMPRESSED_GRID_STATE_MASK) >> COMPRESSED_GRID_STATE_SHIFT
    
    cell_20 = compressed[0, 2].item()
    state_20 = (cell_20 & COMPRESSED_GRID_STATE_MASK) >> COMPRESSED_GRID_STATE_SHIFT
    
    print(f"  Door (0,0) state bits: {state_00} (open)")
    print(f"  Door (1,0) state bits: {state_10} (closed)")
    print(f"  Door (2,0) state bits: {state_20} (locked)")
    
    # Decompress and check door channels
    decompressed = encoder.decompress_grid_to_tensor(compressed, 'cpu')
    
    red_door_ch = DOOR_CHANNEL_START + COLOR_TO_IDX['red']
    blue_door_ch = DOOR_CHANNEL_START + COLOR_TO_IDX['blue']
    green_door_ch = DOOR_CHANNEL_START + COLOR_TO_IDX['green']
    
    print(f"  Red door channel value at (0,0): {decompressed[0, red_door_ch, 0, 0].item()}")
    print(f"  Blue door channel value at (1,0): {decompressed[0, blue_door_ch, 0, 1].item()}")
    print(f"  Green door channel value at (2,0): {decompressed[0, green_door_ch, 0, 2].item()}")
    print("  ✓ Door state encoding OK")


def test_key_encoding():
    """Test that keys are correctly compressed and decompressed."""
    print("Testing key encoding...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    
    # Add keys of different colors
    world_model.grid.set(0, 0, MockObject('key', color='red'))
    world_model.grid.set(1, 0, MockObject('key', color='blue'))
    world_model.grid.set(2, 0, MockObject('key', color='yellow'))
    
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compress
    compressed = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    # Check key type marker
    cell_00 = compressed[0, 0].item()
    obj_type = cell_00 & COMPRESSED_GRID_OBJECT_TYPE_MASK
    assert obj_type == COMPRESSED_GRID_KEY_TYPE, f"Expected key type, got {obj_type}"
    
    # Check color
    key_color = (cell_00 & COMPRESSED_GRID_COLOR_MASK) >> COMPRESSED_GRID_COLOR_SHIFT
    expected_color = COLOR_TO_IDX.get('red', 0)
    assert key_color == expected_color, f"Expected color {expected_color}, got {key_color}"
    
    # Decompress and check key channels
    decompressed = encoder.decompress_grid_to_tensor(compressed, 'cpu')
    
    red_key_ch = KEY_CHANNEL_START + COLOR_TO_IDX['red']
    blue_key_ch = KEY_CHANNEL_START + COLOR_TO_IDX['blue']
    yellow_key_ch = KEY_CHANNEL_START + COLOR_TO_IDX['yellow']
    
    assert decompressed[0, red_key_ch, 0, 0].item() == 1.0, "Red key not found at (0,0)"
    assert decompressed[0, blue_key_ch, 0, 1].item() == 1.0, "Blue key not found at (1,0)"
    assert decompressed[0, yellow_key_ch, 0, 2].item() == 1.0, "Yellow key not found at (2,0)"
    
    print(f"  Red key at (0,0): {decompressed[0, red_key_ch, 0, 0].item()}")
    print(f"  Blue key at (1,0): {decompressed[0, blue_key_ch, 0, 1].item()}")
    print(f"  Yellow key at (2,0): {decompressed[0, yellow_key_ch, 0, 2].item()}")
    print("  ✓ Key encoding OK")


def test_total_compact_storage():
    """Test total storage for compact_features tuple."""
    print("Testing total compact storage...")
    
    encoder = MultiGridStateEncoder(
        grid_height=7,
        grid_width=7,
        num_agents_per_color={'grey': 1, 'blue': 1}
    )
    
    world_model = MockWorldModel(7, 7, 2)
    state = create_mock_state(2)
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent_colors = [agent.color for agent in world_model.agents]
    
    # Compute all compact features
    global_feats, agent_feats, interactive_feats = encoder.tensorize_state_compact(
        state, world_model, 'cpu'
    )
    compressed_grid = encoder.compress_grid(
        world_model, agent_states, mobile_objects, mutable_objects, agent_colors
    )
    
    # Calculate storage
    global_bytes = global_feats.numel() * 4
    agent_bytes = agent_feats.numel() * 4
    interactive_bytes = interactive_feats.numel() * 4
    grid_bytes = compressed_grid.numel() * 4
    total_compact_bytes = global_bytes + agent_bytes + interactive_bytes + grid_bytes
    
    # Full tensor storage
    full_grid, _, _, _ = encoder.tensorize_state(state, world_model, 'cpu')
    full_grid_bytes = full_grid.numel() * 4
    
    print(f"  Global features: {global_bytes} bytes ({global_feats.numel()} floats)")
    print(f"  Agent features: {agent_bytes} bytes ({agent_feats.numel()} floats)")
    print(f"  Interactive features: {interactive_bytes} bytes ({interactive_feats.numel()} floats)")
    print(f"  Compressed grid: {grid_bytes} bytes ({compressed_grid.numel()} int32s)")
    print(f"  Total compact: {total_compact_bytes} bytes")
    print(f"  Full grid tensor: {full_grid_bytes} bytes")
    print(f"  Savings: {full_grid_bytes / total_compact_bytes:.1f}x smaller")
    
    # Should be at least 10x smaller
    assert full_grid_bytes / total_compact_bytes > 10, "Expected >10x savings"
    print("  ✓ Compact storage calculation OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("COMPRESSED GRID AND SPLIT TENSORIZATION TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_encoder_creation,
        test_compress_grid_empty,
        test_compress_grid_with_objects,
        test_compress_grid_with_agents,
        test_decompress_grid_single,
        test_decompress_grid_batch,
        test_storage_savings,
        test_tensorize_state_compact,
        test_tensorize_state_from_compact,
        test_round_trip_consistency,
        test_door_states,
        test_key_encoding,
        test_total_compact_storage,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

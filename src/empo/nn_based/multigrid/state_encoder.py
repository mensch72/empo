"""
Unified state encoder for multigrid environments.

Encodes the complete world state in an agent-agnostic way:
- Grid-based spatial information (objects, doors, magic walls, agents by color)
- Agent features (per-color agent lists)
- Interactive object features (buttons/switches)
- Global world features

This encoder is fully query-agent agnostic - it encodes the world state
without any agent-specific perspective. Agent identity is handled separately
by the AgentIdentityEncoder.

The encoder supports internal caching of raw tensor extraction (before NN forward)
to avoid redundant computation when the same state is encoded multiple times.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ..state_encoder import BaseStateEncoder
from .constants import (
    NUM_STANDARD_COLORS,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
    DOOR_CHANNEL_START,
    KEY_CHANNEL_START,
    MAGICWALL_CHANNEL,
    DOOR_STATE_OPEN,
    DOOR_STATE_CLOSED,
    DOOR_STATE_LOCKED,
    MAGICWALL_STATE_ACTIVE_BASE,
    MAGICWALL_STATE_INACTIVE,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
    NUM_GLOBAL_WORLD_FEATURES,
    AGENT_FEATURE_SIZE,
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
    # Compressed grid constants
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
    COMPRESSED_GRID_OTHER_NONE,
    COMPRESSED_GRID_OTHER_OVERLAPPABLE,
    COMPRESSED_GRID_OTHER_IMMOBILE,
    COMPRESSED_GRID_OTHER_MOBILE,
    COMPRESSED_GRID_DOOR_TYPE,
    COMPRESSED_GRID_KEY_TYPE,
)
from .feature_extraction import (
    extract_global_world_features,
    extract_all_agent_features,
    extract_interactive_objects,
)


def _make_hashable(obj):
    """
    Recursively convert lists to tuples to make state hashable for caching.
    
    The state tuple contains lists (agent_states, mobile_objects, mutable_objects)
    which are not hashable. This function converts them to tuples.
    """
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(_make_hashable(item) for item in obj)
    else:
        return obj


class MultiGridStateEncoder(BaseStateEncoder):
    """
    Unified encoder for multigrid environment states.
    
    Combines:
    1. Grid-based CNN for spatial information (objects, doors, agents by color)
    2. MLP for agent features (query agent + per-color lists)
    3. MLP for interactive objects (buttons/switches)
    4. Global world features
    
    .. warning:: ASYNC TRAINING / PICKLE COMPATIBILITY
    
        This class is pickled and sent to spawned actor processes during async
        training. To avoid breaking async functionality:
        
        1. **Do NOT create large unused nn.Module layers.** When use_encoders=False,
           we skip creating CNN/MLP layers entirely and use nn.Identity() placeholders.
           Creating unused layers bloats pickle size (130MB vs 10MB) and can exceed
           Docker's default 64MB shared memory, causing SIGBUS errors.
        
        2. **All attributes must be picklable.** Avoid lambdas, local functions,
           open file handles, or non-picklable objects as instance attributes.
        
        3. **Cache contents are NOT preserved across pickle.** The _raw_cache dict
           will be empty in worker processes (this is fine, it's just a performance
           optimization).
        
        4. **Test with async mode after changes:** Always verify changes work with
           ``--async`` flag in the phase2 demo.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents of that color.
        num_agent_colors: Number of distinct agent colors for grid channels.
        feature_dim: Total output feature dimension.
        max_kill_buttons: Max KillButtons to encode.
        max_pause_switches: Max PauseSwitches to encode.
        max_disabling_switches: Max DisablingSwitches to encode.
        max_control_buttons: Max ControlButtons to encode.
        share_cache_with: Optional encoder instance to share raw tensor cache with.
            If provided, this encoder will use the other encoder's cache instead
            of creating its own. Useful for "own" encoders to reuse shared encoder caches.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = NUM_STANDARD_COLORS,
        feature_dim: int = 256,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        include_step_count: bool = True,
        use_encoders: bool = True,
        share_cache_with: Optional['MultiGridStateEncoder'] = None
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_agents_per_color = num_agents_per_color
        self.num_agent_colors = num_agent_colors
        self.max_kill_buttons = max_kill_buttons
        self.max_pause_switches = max_pause_switches
        self.max_disabling_switches = max_disabling_switches
        self.max_control_buttons = max_control_buttons
        self.include_step_count = include_step_count
        self.use_encoders = use_encoders
        
        # Grid channel structure
        # Use all standard colors for agent channels to support any color combination
        # No query agent channel - state encoding is fully agent-agnostic
        self.num_object_channels = NUM_OBJECT_TYPE_CHANNELS
        self.num_other_channels = 3  # overlappable, immobile, mobile
        self.agent_channels_start = self.num_object_channels + self.num_other_channels
        self.num_grid_channels = self.agent_channels_start + NUM_STANDARD_COLORS
        
        # Compute input sizes for agent and interactive features
        self.color_order = sorted(num_agents_per_color.keys())
        total_agents = sum(num_agents_per_color.values())
        agent_input_size = AGENT_FEATURE_SIZE * total_agents
        interactive_input_size = (
            max_kill_buttons * KILLBUTTON_FEATURE_SIZE +
            max_pause_switches * PAUSESWITCH_FEATURE_SIZE +
            max_disabling_switches * DISABLINGSWITCH_FEATURE_SIZE +
            max_control_buttons * CONTROLBUTTON_FEATURE_SIZE
        )
        
        # Store input sizes
        self._agent_input_size = agent_input_size
        self._interactive_input_size = interactive_input_size
        self._global_features_size = NUM_GLOBAL_WORLD_FEATURES
        
        # When use_encoders=False, compute identity output dim and skip creating NN layers
        if not use_encoders:
            # Identity mode: output is flattened concatenation of all inputs
            identity_dim = (
                self.num_grid_channels * grid_height * grid_width +
                NUM_GLOBAL_WORLD_FEATURES +
                agent_input_size +
                interactive_input_size
            )
            # Override feature_dim to match actual identity output
            self.feature_dim = identity_dim
            # Create dummy attributes so state_dict works (empty modules)
            self.grid_conv = nn.Identity()
            self.grid_fc = nn.Identity()
            self.agent_fc = nn.Identity()
            self.interactive_fc = nn.Identity()
            self.output_fc = nn.Identity()
            self._grid_feature_dim = 0
            self._agent_feature_dim = 0
            self._interactive_feature_dim = 0
        else:
            # Normal mode: create full NN layers
            # Scale intermediate dimensions based on feature_dim
            # For small feature_dim (e.g., 16), use proportionally smaller networks
            conv_channels = max(8, feature_dim // 4)  # e.g., 16->4->8, 64->16, 256->64
            grid_feature_dim = max(16, feature_dim // 2)  # e.g., 16->8->16, 64->32, 256->128
            agent_feature_dim = max(8, feature_dim // 4)  # e.g., 16->4->8, 64->16, 256->64
            interactive_feature_dim = max(4, feature_dim // 8)  # e.g., 16->2->4, 64->8, 256->32
            
            # Grid encoder (CNN) - reduced architecture for speed
            # 2 conv layers with pooling instead of 3 conv layers without pooling
            self.grid_conv = nn.Sequential(
                nn.Conv2d(self.num_grid_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Halves spatial dimensions -> 4x fewer values to process
                nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            # Compute actual output size with a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.num_grid_channels, grid_height, grid_width)
                dummy_output = self.grid_conv(dummy_input)
                grid_conv_out_size = dummy_output.numel()
            
            self.grid_fc = nn.Sequential(
                nn.Linear(grid_conv_out_size + NUM_GLOBAL_WORLD_FEATURES, grid_feature_dim),
                nn.ReLU(),
            )
            
            # Agent encoder (MLP)
            # Encodes all agents organized by color (no separate query agent features)
            self.agent_fc = nn.Sequential(
                nn.Linear(agent_input_size, agent_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(agent_feature_dim * 2, agent_feature_dim),
                nn.ReLU(),
            )
            
            # Interactive object encoder (MLP)
            self.interactive_fc = nn.Sequential(
                nn.Linear(interactive_input_size, interactive_feature_dim),
                nn.ReLU(),
            )
            
            # Combined feature dimension
            combined_dim = grid_feature_dim + agent_feature_dim + interactive_feature_dim
            
            # Final projection to feature_dim
            self.output_fc = nn.Sequential(
                nn.Linear(combined_dim, feature_dim),
                nn.ReLU(),
            )
            
            # Store sub-feature dimensions for get_config
            self._grid_feature_dim = grid_feature_dim
            self._agent_feature_dim = agent_feature_dim
            self._interactive_feature_dim = interactive_feature_dim
        
        # Internal cache for raw tensor extraction (before NN forward)
        # Keys are state_id (int), values are raw tensor tuples
        # Cache is query-agent agnostic since state encoding doesn't depend on query agent
        # If share_cache_with is provided, reuse that encoder's cache
        if share_cache_with is not None:
            self._raw_cache = share_cache_with._raw_cache
            self._shared_cache = True
        else:
            self._raw_cache: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
            self._shared_cache = False
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_cache(self):
        """Clear the internal raw tensor cache."""
        self._raw_cache.clear()
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) cache statistics."""
        return self._cache_hits, self._cache_misses
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        self._cache_hits = 0
        self._cache_misses = 0
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode complete state.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Feature tensor (batch, feature_dim)
        
        If use_encoders=False, bypasses neural network and returns flattened
        concatenation of inputs (true identity mode for debugging).
        """
        batch_size = grid_tensor.shape[0]
        
        if not self.use_encoders:
            # Identity mode: flatten and concatenate all inputs unchanged
            flat_grid = grid_tensor.view(batch_size, -1)
            return torch.cat([flat_grid, global_features, agent_features, interactive_features], dim=1)
        
        # Grid encoding
        conv_out = self.grid_conv(grid_tensor)
        conv_flat = conv_out.view(batch_size, -1)
        grid_combined = torch.cat([conv_flat, global_features], dim=1)
        grid_emb = self.grid_fc(grid_combined)
        
        # Agent encoding
        agent_emb = self.agent_fc(agent_features)
        
        # Interactive object encoding
        interactive_emb = self.interactive_fc(interactive_features)
        
        # Combine all features
        combined = torch.cat([grid_emb, agent_emb, interactive_emb], dim=1)
        return self.output_fc(combined)
    
    def tensorize_state(
        self,
        state: Tuple,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw state to input tensors (preprocessing, NOT neural network encoding).
        
        This method extracts grid, global, agent, and interactive features from
        the state tuple as tensors suitable for the forward() pass. Results are
        cached by state_id to avoid redundant extraction.
        
        This is tensorization/featurization - it converts Python objects to tensors.
        Call forward() on these tensors to get the actual neural network encoding.
        
        The tensorization is fully agent-agnostic - it does not depend on any
        particular agent's perspective. Agent identity is handled separately
        by the AgentIdentityEncoder.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents (or None).
            device: Torch device.
        
        Returns:
            Tuple of (grid_tensor, global_features, agent_features, interactive_features)
        """
        # Check cache first (agent-agnostic, keyed by state content)
        # State contains lists, so convert to hashable form (tuples)
        cache_key = _make_hashable(state)
        if cache_key in self._raw_cache:
            self._cache_hits += 1
            # Clone cached tensors to avoid in-place operation conflicts
            # during gradient computation when same state is used multiple times
            cached = self._raw_cache[cache_key]
            return tuple(t.clone() for t in cached)
        
        self._cache_misses += 1
        
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        # Tensorize grid (agent-agnostic)
        grid_tensor = torch.zeros(1, self.num_grid_channels, H, W, device=device)
        self._tensorize_grid(grid_tensor, world_model, agent_states, 
                             mobile_objects, mutable_objects)
        
        # Global features (with optional step count)
        global_features = extract_global_world_features(
            state, world_model, device, include_step_count=self.include_step_count
        )
        global_features = global_features.unsqueeze(0)
        
        # Agent features (all agents by color, no query-specific features)
        agent_features = self._encode_agents(agent_states, world_model, device)
        
        # Interactive object features
        interactive_features = self._encode_interactive(state, world_model, device)
        
        result = (grid_tensor, global_features, agent_features, interactive_features)
        self._raw_cache[cache_key] = result
        return result
    
    def tensorize_state_compact(
        self,
        state: Tuple,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute small tensors (expensive but low-storage) for storing in replay buffer.
        
        This extracts features that are:
        - Expensive to compute (require iterating over agents, grid scanning for objects)
        - Small in storage (tens to hundreds of floats, not thousands)
        
        These can be pre-computed by the actor and stored in the replay buffer,
        avoiding redundant computation during training.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents.
            device: Torch device.
        
        Returns:
            Tuple of (global_features, agent_features, interactive_features)
            - global_features: (1, NUM_GLOBAL_WORLD_FEATURES) ~ 4 floats
            - agent_features: (1, agent_input_size) ~ 26 floats for 2 agents
            - interactive_features: (1, interactive_input_size) ~ 96 floats
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Global features (with optional step count) - cheap but included for completeness
        global_features = extract_global_world_features(
            state, world_model, device, include_step_count=self.include_step_count
        )
        global_features = global_features.unsqueeze(0)
        
        # Agent features (all agents by color) - EXPENSIVE: iterates over agents, checks attributes
        agent_features = self._encode_agents(agent_states, world_model, device)
        
        # Interactive object features - EXPENSIVE: scans entire grid for buttons/switches
        interactive_features = self._encode_interactive(state, world_model, device)
        
        return (global_features, agent_features, interactive_features)
    
    def tensorize_state_grid(
        self,
        state: Tuple,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Compute grid tensor (cheap but large-storage) during training.
        
        This builds the grid tensor which is:
        - Cheap to compute (simple position-to-channel mapping)
        - Large in storage (num_channels × H × W floats)
        
        Should be computed by the trainer, not stored in replay buffer.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents.
            device: Torch device.
        
        Returns:
            grid_tensor: (1, num_grid_channels, H, W)
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        grid_tensor = torch.zeros(1, self.num_grid_channels, H, W, device=device)
        self._tensorize_grid(grid_tensor, world_model, agent_states, 
                             mobile_objects, mutable_objects)
        
        return grid_tensor
    
    def tensorize_state_from_compact(
        self,
        state: Tuple,
        world_model: Any,
        compact_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build full tensorization using pre-computed compact features.
        
        This combines pre-computed compact features (from replay buffer)
        with freshly computed grid tensor to get full input tensors.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents.
            compact_features: (global_features, agent_features, interactive_features)
                from tensorize_state_compact().
            device: Torch device.
        
        Returns:
            Tuple of (grid_tensor, global_features, agent_features, interactive_features)
        """
        global_features, agent_features, interactive_features = compact_features
        
        # Build grid tensor (cheap)
        grid_tensor = self.tensorize_state_grid(state, world_model, device)
        
        # Ensure all tensors are on the right device
        global_features = global_features.to(device)
        agent_features = agent_features.to(device)
        interactive_features = interactive_features.to(device)
        
        return (grid_tensor, global_features, agent_features, interactive_features)

    def compress_grid(
        self,
        world_model: Any,
        agent_states: list,
        mobile_objects: list,
        mutable_objects: list,
        agent_colors: list,
    ) -> torch.Tensor:
        """
        Compress the entire grid into a single int32 tensor for replay buffer storage.
        
        This captures ALL grid information (static objects from world_model + dynamic 
        state) so that the grid can be reconstructed without access to world_model.
        
        Storage: H × W × 4 bytes (int32) vs H × W × num_channels × 4 bytes (float32)
        For 7×7 grid with 39 channels: 196 bytes vs 7644 bytes = 39x smaller!
        
        Args:
            world_model: Environment with grid and agents.
            agent_states: List of agent states from state tuple.
            mobile_objects: List of mobile objects from state tuple.
            mutable_objects: List of mutable objects from state tuple.
            agent_colors: List of agent colors (strings), one per agent.
        
        Returns:
            compressed_grid: (H, W) int32 tensor with packed cell information.
        """
        H, W = self.grid_height, self.grid_width
        compressed = torch.zeros(H, W, dtype=torch.int32)
        
        # Initialize all cells with "no agent" marker
        compressed[:, :] = COMPRESSED_GRID_NO_AGENT << COMPRESSED_GRID_AGENT_COLOR_SHIFT
        
        # First pass: encode static grid objects from world_model
        if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
            actual_height = getattr(world_model, 'height', H)
            actual_width = getattr(world_model, 'width', W)
            
            for y in range(min(actual_height, H)):
                for x in range(min(actual_width, W)):
                    cell = world_model.grid.get(x, y)
                    if cell is None:
                        continue
                    
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is None:
                        continue
                    
                    # Preserve the "no agent" marker
                    cell_value = COMPRESSED_GRID_NO_AGENT << COMPRESSED_GRID_AGENT_COLOR_SHIFT
                    
                    if cell_type == 'door':
                        # Doors: use special type code + color + state
                        color = getattr(cell, 'color', None)
                        color_idx = COLOR_TO_IDX.get(color, 0)
                        cell_value |= COMPRESSED_GRID_DOOR_TYPE
                        cell_value |= (color_idx << COMPRESSED_GRID_COLOR_SHIFT)
                        # State will be updated from mutable_objects below
                        
                    elif cell_type == 'key':
                        # Keys: use special type code + color
                        color = getattr(cell, 'color', None)
                        color_idx = COLOR_TO_IDX.get(color, 0)
                        cell_value |= COMPRESSED_GRID_KEY_TYPE
                        cell_value |= (color_idx << COMPRESSED_GRID_COLOR_SHIFT)
                        
                    elif cell_type == 'magicwall':
                        # Magic walls: encode in magic_shift bits
                        magic_side = getattr(cell, 'magic_side', 0)
                        active = getattr(cell, 'active', True)
                        if active:
                            magic_state = MAGICWALL_STATE_ACTIVE_BASE + magic_side
                        else:
                            magic_state = MAGICWALL_STATE_INACTIVE
                        cell_value |= (magic_state << COMPRESSED_GRID_MAGIC_SHIFT)
                        
                    elif cell_type in OBJECT_TYPE_TO_CHANNEL:
                        # Standard object types
                        cell_value |= OBJECT_TYPE_TO_CHANNEL[cell_type]
                        
                    else:
                        # Other object - categorize it
                        if cell_type in OVERLAPPABLE_OBJECTS:
                            cell_value |= (COMPRESSED_GRID_OTHER_OVERLAPPABLE << COMPRESSED_GRID_OTHER_SHIFT)
                        elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                            cell_value |= (COMPRESSED_GRID_OTHER_IMMOBILE << COMPRESSED_GRID_OTHER_SHIFT)
                        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                            cell_value |= (COMPRESSED_GRID_OTHER_MOBILE << COMPRESSED_GRID_OTHER_SHIFT)
                    
                    compressed[y, x] = cell_value
            
            # Fill cells outside actual world with walls
            wall_with_no_agent = OBJECT_TYPE_TO_CHANNEL['wall'] | (COMPRESSED_GRID_NO_AGENT << COMPRESSED_GRID_AGENT_COLOR_SHIFT)
            if actual_height < H:
                compressed[actual_height:, :] = wall_with_no_agent
            if actual_width < W:
                compressed[:, actual_width:] = wall_with_no_agent
        else:
            # No world model - fill with walls
            wall_with_no_agent = OBJECT_TYPE_TO_CHANNEL['wall'] | (COMPRESSED_GRID_NO_AGENT << COMPRESSED_GRID_AGENT_COLOR_SHIFT)
            compressed[:, :] = wall_with_no_agent
        
        # Second pass: update mutable object states (doors, magic walls)
        for obj_data in mutable_objects:
            obj_type = obj_data[0]
            x, y = obj_data[1], obj_data[2]
            if not (0 <= x < W and 0 <= y < H):
                continue
                
            if obj_type == 'door':
                is_open, is_locked = obj_data[3], obj_data[4]
                if is_open:
                    state = DOOR_STATE_OPEN
                elif is_locked:
                    state = DOOR_STATE_LOCKED
                else:
                    state = DOOR_STATE_CLOSED
                # Update state bits, preserving other bits
                current = int(compressed[y, x].item())
                current = (current & ~COMPRESSED_GRID_STATE_MASK) | (state << COMPRESSED_GRID_STATE_SHIFT)
                compressed[y, x] = current
                
            elif obj_type == 'magicwall':
                active = obj_data[3] if len(obj_data) > 3 else True
                # Read magic_side from existing encoding or default to 0
                current = int(compressed[y, x].item())
                old_magic = (current & COMPRESSED_GRID_MAGIC_MASK) >> COMPRESSED_GRID_MAGIC_SHIFT
                if old_magic >= MAGICWALL_STATE_ACTIVE_BASE and old_magic < MAGICWALL_STATE_INACTIVE:
                    magic_side = old_magic - MAGICWALL_STATE_ACTIVE_BASE
                else:
                    magic_side = 0
                if active:
                    magic_state = MAGICWALL_STATE_ACTIVE_BASE + magic_side
                else:
                    magic_state = MAGICWALL_STATE_INACTIVE
                current = (current & ~COMPRESSED_GRID_MAGIC_MASK) | (magic_state << COMPRESSED_GRID_MAGIC_SHIFT)
                compressed[y, x] = current
        
        # Third pass: encode mobile objects (overwrite object type at their positions)
        for obj_data in mobile_objects:
            obj_type, obj_x, obj_y = obj_data[0], obj_data[1], obj_data[2]
            if 0 <= obj_x < W and 0 <= obj_y < H:
                current = int(compressed[obj_y, obj_x].item())
                # Clear object type bits, preserve agent bits
                current = current & ~COMPRESSED_GRID_OBJECT_TYPE_MASK
                if obj_type in OBJECT_TYPE_TO_CHANNEL:
                    current |= OBJECT_TYPE_TO_CHANNEL[obj_type]
                else:
                    # Other mobile object - set in other_cat bits
                    current = current & ~COMPRESSED_GRID_OTHER_MASK
                    current |= (COMPRESSED_GRID_OTHER_MOBILE << COMPRESSED_GRID_OTHER_SHIFT)
                compressed[obj_y, obj_x] = current
        
        # Fourth pass: encode agents (in agent_color bits, don't overwrite object)
        for i, agent_state in enumerate(agent_states):
            if agent_state[0] is None:
                continue
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < W and 0 <= y < H and i < len(agent_colors):
                color = agent_colors[i]
                color_idx = COLOR_TO_IDX.get(color, 0)
                current = int(compressed[y, x].item())
                current = (current & ~COMPRESSED_GRID_AGENT_COLOR_MASK) | (color_idx << COMPRESSED_GRID_AGENT_COLOR_SHIFT)
                compressed[y, x] = current
        
        return compressed
    
    def decompress_grid_to_tensor(
        self,
        compressed_grid: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Decompress a compressed grid back into the full channel tensor.
        
        This is a VECTORIZED operation that unpacks the compressed int32 grid
        into the full float32 channel tensor without any Python loops over cells.
        
        Args:
            compressed_grid: (H, W) int32 tensor from compress_grid().
            device: Torch device for output tensor.
        
        Returns:
            grid_tensor: (1, num_grid_channels, H, W) float32 tensor.
        """
        H, W = compressed_grid.shape
        compressed_grid = compressed_grid.to(device)
        
        # Output tensor
        grid_tensor = torch.zeros(1, self.num_grid_channels, H, W, device=device)
        
        # Extract packed fields using bitwise operations (all vectorized!)
        obj_type = compressed_grid & COMPRESSED_GRID_OBJECT_TYPE_MASK
        obj_color = (compressed_grid & COMPRESSED_GRID_COLOR_MASK) >> COMPRESSED_GRID_COLOR_SHIFT
        obj_state = (compressed_grid & COMPRESSED_GRID_STATE_MASK) >> COMPRESSED_GRID_STATE_SHIFT
        agent_color = (compressed_grid & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT
        magic_state = (compressed_grid & COMPRESSED_GRID_MAGIC_MASK) >> COMPRESSED_GRID_MAGIC_SHIFT
        other_cat = (compressed_grid & COMPRESSED_GRID_OTHER_MASK) >> COMPRESSED_GRID_OTHER_SHIFT
        
        # Standard object type channels (vectorized scatter)
        for channel_idx in range(NUM_OBJECT_TYPE_CHANNELS - 1):  # Exclude magic wall channel
            mask = (obj_type == channel_idx)
            grid_tensor[0, channel_idx, :, :] = mask.float()
        
        # Door channels (per-color, with state encoding)
        door_mask = (obj_type == COMPRESSED_GRID_DOOR_TYPE)
        for color_idx in range(NUM_STANDARD_COLORS):
            color_door_mask = door_mask & (obj_color == color_idx)
            if color_door_mask.any():
                channel = DOOR_CHANNEL_START + color_idx
                # Encode state: 0.33=open, 0.67=closed, 1.0=locked
                state_values = torch.zeros_like(compressed_grid, dtype=torch.float32)
                state_values[color_door_mask & (obj_state == DOOR_STATE_OPEN)] = 0.33
                state_values[color_door_mask & (obj_state == DOOR_STATE_CLOSED)] = 0.67
                state_values[color_door_mask & (obj_state == DOOR_STATE_LOCKED)] = 1.0
                grid_tensor[0, channel, :, :] = state_values
        
        # Key channels (per-color)
        key_mask = (obj_type == COMPRESSED_GRID_KEY_TYPE)
        for color_idx in range(NUM_STANDARD_COLORS):
            color_key_mask = key_mask & (obj_color == color_idx)
            channel = KEY_CHANNEL_START + color_idx
            grid_tensor[0, channel, :, :] = color_key_mask.float()
        
        # Magic wall channel
        magic_active_mask = (magic_state >= MAGICWALL_STATE_ACTIVE_BASE) & (magic_state < MAGICWALL_STATE_INACTIVE)
        magic_inactive_mask = (magic_state == MAGICWALL_STATE_INACTIVE)
        # Encode: active with magic_side 0-4 -> values 1-5, inactive -> 6
        grid_tensor[0, MAGICWALL_CHANNEL, :, :] = torch.where(
            magic_active_mask,
            magic_state.float(),  # Already MAGICWALL_STATE_ACTIVE_BASE + magic_side
            torch.where(magic_inactive_mask, torch.tensor(MAGICWALL_STATE_INACTIVE, dtype=torch.float32, device=device), 
                       torch.tensor(0.0, device=device))
        )
        
        # Other category channels
        other_overlappable_idx = self.num_object_channels  # Channel for "other overlappable"
        other_immobile_idx = self.num_object_channels + 1  # Channel for "other immobile"
        other_mobile_idx = self.num_object_channels + 2    # Channel for "other mobile"
        
        grid_tensor[0, other_overlappable_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_OVERLAPPABLE).float()
        grid_tensor[0, other_immobile_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_IMMOBILE).float()
        grid_tensor[0, other_mobile_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_MOBILE).float()
        
        # Agent channels (per-color)
        agent_present = (agent_color != COMPRESSED_GRID_NO_AGENT)
        for color_idx in range(NUM_STANDARD_COLORS):
            agent_color_mask = agent_present & (agent_color == color_idx)
            channel = self.agent_channels_start + color_idx
            grid_tensor[0, channel, :, :] = agent_color_mask.float()
        
        return grid_tensor
    
    def decompress_grid_batch_to_tensor(
        self,
        compressed_grids: torch.Tensor,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Decompress a BATCH of compressed grids into the full channel tensor.
        
        This is fully VECTORIZED across the batch dimension - no Python loops
        over batch elements or grid cells!
        
        Args:
            compressed_grids: (batch_size, H, W) int32 tensor.
            device: Torch device for output tensor.
        
        Returns:
            grid_tensor: (batch_size, num_grid_channels, H, W) float32 tensor.
        """
        batch_size, H, W = compressed_grids.shape
        compressed_grids = compressed_grids.to(device)
        
        # Output tensor
        grid_tensor = torch.zeros(batch_size, self.num_grid_channels, H, W, device=device)
        
        # Extract packed fields using bitwise operations (all vectorized!)
        obj_type = compressed_grids & COMPRESSED_GRID_OBJECT_TYPE_MASK
        obj_color = (compressed_grids & COMPRESSED_GRID_COLOR_MASK) >> COMPRESSED_GRID_COLOR_SHIFT
        obj_state = (compressed_grids & COMPRESSED_GRID_STATE_MASK) >> COMPRESSED_GRID_STATE_SHIFT
        agent_color = (compressed_grids & COMPRESSED_GRID_AGENT_COLOR_MASK) >> COMPRESSED_GRID_AGENT_COLOR_SHIFT
        magic_state = (compressed_grids & COMPRESSED_GRID_MAGIC_MASK) >> COMPRESSED_GRID_MAGIC_SHIFT
        other_cat = (compressed_grids & COMPRESSED_GRID_OTHER_MASK) >> COMPRESSED_GRID_OTHER_SHIFT
        
        # Standard object type channels (vectorized scatter)
        for channel_idx in range(NUM_OBJECT_TYPE_CHANNELS - 1):  # Exclude magic wall channel
            mask = (obj_type == channel_idx)
            grid_tensor[:, channel_idx, :, :] = mask.float()
        
        # Door channels (per-color, with state encoding)
        door_mask = (obj_type == COMPRESSED_GRID_DOOR_TYPE)
        for color_idx in range(NUM_STANDARD_COLORS):
            color_door_mask = door_mask & (obj_color == color_idx)
            channel = DOOR_CHANNEL_START + color_idx
            # Encode state: 0.33=open, 0.67=closed, 1.0=locked
            state_values = torch.zeros_like(compressed_grids, dtype=torch.float32, device=device)
            state_values[color_door_mask & (obj_state == DOOR_STATE_OPEN)] = 0.33
            state_values[color_door_mask & (obj_state == DOOR_STATE_CLOSED)] = 0.67
            state_values[color_door_mask & (obj_state == DOOR_STATE_LOCKED)] = 1.0
            grid_tensor[:, channel, :, :] = state_values
        
        # Key channels (per-color)
        key_mask = (obj_type == COMPRESSED_GRID_KEY_TYPE)
        for color_idx in range(NUM_STANDARD_COLORS):
            color_key_mask = key_mask & (obj_color == color_idx)
            channel = KEY_CHANNEL_START + color_idx
            grid_tensor[:, channel, :, :] = color_key_mask.float()
        
        # Magic wall channel
        magic_active_mask = (magic_state >= MAGICWALL_STATE_ACTIVE_BASE) & (magic_state < MAGICWALL_STATE_INACTIVE)
        magic_inactive_mask = (magic_state == MAGICWALL_STATE_INACTIVE)
        magic_values = torch.zeros_like(compressed_grids, dtype=torch.float32, device=device)
        magic_values[magic_active_mask] = magic_state[magic_active_mask].float()
        magic_values[magic_inactive_mask] = MAGICWALL_STATE_INACTIVE
        grid_tensor[:, MAGICWALL_CHANNEL, :, :] = magic_values
        
        # Other category channels
        other_overlappable_idx = self.num_object_channels
        other_immobile_idx = self.num_object_channels + 1
        other_mobile_idx = self.num_object_channels + 2
        
        grid_tensor[:, other_overlappable_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_OVERLAPPABLE).float()
        grid_tensor[:, other_immobile_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_IMMOBILE).float()
        grid_tensor[:, other_mobile_idx, :, :] = (other_cat == COMPRESSED_GRID_OTHER_MOBILE).float()
        
        # Agent channels (per-color)
        agent_present = (agent_color != COMPRESSED_GRID_NO_AGENT)
        for color_idx in range(NUM_STANDARD_COLORS):
            agent_color_mask = agent_present & (agent_color == color_idx)
            channel = self.agent_channels_start + color_idx
            grid_tensor[:, channel, :, :] = agent_color_mask.float()
        
        return grid_tensor

    def _tensorize_grid(
        self,
        grid_tensor: torch.Tensor,
        world_model: Any,
        agent_states: list,
        mobile_objects: list,
        mutable_objects: list
    ):
        """Encode grid objects into tensor.
        
        If world_model has smaller dimensions than the encoder was configured for,
        only the actual world area is encoded and the rest is padded with walls.
        This allows policies trained on larger grids to work on smaller grids.
        """
        H, W = self.grid_height, self.grid_width
        
        # Get actual world dimensions (may be smaller than encoder dimensions)
        actual_height = getattr(world_model, 'height', H) if world_model is not None else H
        actual_width = getattr(world_model, 'width', W) if world_model is not None else W
        
        # Pre-fill entire grid with walls, then clear cells as we encode them
        # This efficiently handles padding when actual world is smaller than encoder grid
        wall_channel = OBJECT_TYPE_TO_CHANNEL['wall']
        grid_tensor[0, wall_channel, :, :] = 1.0
        
        # Encode grid objects from actual world - clears wall channel where objects exist
        if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
            for y in range(min(actual_height, H)):
                for x in range(min(actual_width, W)):
                    # Clear wall at this position (will be set if cell contains wall)
                    grid_tensor[0, wall_channel, y, x] = 0.0
                    
                    cell = world_model.grid.get(x, y)
                    if cell is None:
                        continue
                    
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is None:
                        continue
                    
                    if cell_type == 'door':
                        self._encode_door(cell, x, y, mutable_objects, grid_tensor)
                    elif cell_type == 'key':
                        self._encode_key(cell, x, y, grid_tensor)
                    elif cell_type == 'magicwall':
                        self._encode_magic_wall(cell, x, y, mutable_objects, grid_tensor)
                    elif cell_type in OBJECT_TYPE_TO_CHANNEL:
                        channel = OBJECT_TYPE_TO_CHANNEL[cell_type]
                        grid_tensor[0, channel, y, x] = 1.0
                    else:
                        self._encode_other_object(cell, cell_type, x, y, grid_tensor)
        
        # Encode mobile objects
        if mobile_objects:
            for obj_data in mobile_objects:
                obj_type, obj_x, obj_y = obj_data[0], obj_data[1], obj_data[2]
                if 0 <= obj_x < W and 0 <= obj_y < H:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        grid_tensor[0, channel, obj_y, obj_x] = 1.0
                    else:
                        other_mobile_idx = self.num_object_channels + 2
                        grid_tensor[0, other_mobile_idx, obj_y, obj_x] = 1.0
        
        # Encode agents by color
        if world_model is not None and hasattr(world_model, 'agents'):
            for i, agent_state in enumerate(agent_states):
                if agent_state[0] is None:
                    continue
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H and i < len(world_model.agents):
                    color = getattr(world_model.agents[i], 'color', 'grey')
                    if color in COLOR_TO_IDX:
                        channel = self.agent_channels_start + COLOR_TO_IDX[color]
                        grid_tensor[0, channel, y, x] = 1.0
    
    def _encode_door(self, cell, x: int, y: int, mutable_objects, grid_tensor: torch.Tensor):
        """Encode door into per-color channel."""
        color = getattr(cell, 'color', None)
        if color not in COLOR_TO_IDX:
            return
        
        channel = DOOR_CHANNEL_START + COLOR_TO_IDX[color]
        
        is_open, is_locked = False, False
        for obj_data in mutable_objects:
            if obj_data[0] == 'door' and obj_data[1] == x and obj_data[2] == y:
                is_open, is_locked = obj_data[3], obj_data[4]
                break
        else:
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
        
        if is_open:
            grid_tensor[0, channel, y, x] = DOOR_STATE_OPEN
        elif is_locked:
            grid_tensor[0, channel, y, x] = DOOR_STATE_LOCKED
        else:
            grid_tensor[0, channel, y, x] = DOOR_STATE_CLOSED
    
    def _encode_key(self, cell, x: int, y: int, grid_tensor: torch.Tensor):
        """Encode key into per-color channel."""
        color = getattr(cell, 'color', None)
        if color not in COLOR_TO_IDX:
            return
        channel = KEY_CHANNEL_START + COLOR_TO_IDX[color]
        grid_tensor[0, channel, y, x] = 1.0
    
    def _encode_magic_wall(self, cell, x: int, y: int, mutable_objects, grid_tensor: torch.Tensor):
        """Encode magic wall with state."""
        active = True
        for obj_data in mutable_objects:
            if obj_data[0] == 'magicwall' and obj_data[1] == x and obj_data[2] == y:
                active = obj_data[3] if len(obj_data) > 3 else True
                break
        else:
            active = getattr(cell, 'active', True)
        
        magic_side = getattr(cell, 'magic_side', 0)
        
        if active:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = MAGICWALL_STATE_ACTIVE_BASE + magic_side
        else:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = MAGICWALL_STATE_INACTIVE
    
    def _encode_other_object(self, cell, cell_type: str, x: int, y: int, grid_tensor: torch.Tensor):
        """Encode object not in main channels."""
        if cell_type in OVERLAPPABLE_OBJECTS:
            channel = self.num_object_channels
        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
            channel = self.num_object_channels + 2
        elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
            channel = self.num_object_channels + 1
        elif hasattr(cell, 'can_overlap') and cell.can_overlap():
            channel = self.num_object_channels
        else:
            channel = self.num_object_channels + 1
        
        grid_tensor[0, channel, y, x] = 1.0
    
    def _encode_agents(
        self,
        agent_states: list,
        world_model: Any,
        device: str
    ) -> torch.Tensor:
        """Encode agent features (all agents organized by color, agent-agnostic)."""
        color_features = extract_all_agent_features(
            agent_states, world_model, self.num_agents_per_color
        )
        
        all_features = []
        for color in self.color_order:
            if color in color_features:
                all_features.append(color_features[color].flatten())
        
        return torch.cat(all_features).unsqueeze(0).to(device)
    
    def _encode_interactive(
        self,
        state: Tuple,
        world_model: Any,
        device: str
    ) -> torch.Tensor:
        """Encode interactive object features."""
        objects = extract_interactive_objects(
            state, world_model,
            self.max_kill_buttons,
            self.max_pause_switches,
            self.max_disabling_switches,
            self.max_control_buttons
        )
        
        features = torch.cat([
            objects['kill_buttons'].flatten(),
            objects['pause_switches'].flatten(),
            objects['disabling_switches'].flatten(),
            objects['control_buttons'].flatten(),
        ]).unsqueeze(0).to(device)
        
        return features
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'num_agents_per_color': self.num_agents_per_color,
            'num_agent_colors': self.num_agent_colors,
            'feature_dim': self.feature_dim,
            'max_kill_buttons': self.max_kill_buttons,
            'max_pause_switches': self.max_pause_switches,
            'max_disabling_switches': self.max_disabling_switches,
            'max_control_buttons': self.max_control_buttons,
            'include_step_count': self.include_step_count,
            'use_encoders': self.use_encoders,
        }

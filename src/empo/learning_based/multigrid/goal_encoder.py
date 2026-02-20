"""
Goal encoder for multigrid environments.

Encodes goals into feature vectors using polymorphic goal encoding with
type discrimination. All goals are encoded as 6-dimensional tensors:
- Dimensions 0-3: Goal-specific features
  * Spatial: [x1, y1, x2, y2, ...]
  * Orientation: [d0, d1, d2, d3, ...] (one-hot direction)
- Dimensions 4-5: Goal type discriminator
  * [1, 0] for spatial goals (ReachCell, ReachRectangle)
  * [0, 1] for orientation goals (OrientationGoal)

For goal sampling and rendering, see empo.world_specific_helpers.multigrid module.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from empo.learning_based.phase1.goal_encoder import BaseGoalEncoder


class MultiGridGoalEncoder(BaseGoalEncoder):
    """
    Encoder for multigrid goals.

    This encoder works with any goal type that implements the PossibleGoal.encode()
    interface.

    Supported goal types:
    - ReachCellGoal: Point goals encoded as bounding boxes (x, y, x, y)
    - ReachRectangleGoal: Rectangle goals encoded as (x1, y1, x2, y2)
    - OrientationGoal: Direction goals encoded as one-hot vectors

    .. warning:: ASYNC TRAINING / PICKLE COMPATIBILITY

        This class is pickled and sent to spawned actor processes during async
        training. To avoid breaking async functionality:

        1. **Do NOT create large unused nn.Module layers.** When use_encoders=False,
           we use nn.Identity() placeholder instead of creating MLP layers.

        2. **All attributes must be picklable.** Avoid lambdas, local functions,
           or non-picklable objects as instance attributes.

        3. **Test with async mode after changes:** Always verify changes work with
           ``--async`` flag in the phase2 demo.

    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        feature_dim: Output feature dimension.
        use_encoders: If False, forward() returns identity (padded input).
        share_cache_with: Optional encoder instance to share raw tensor cache with.
            If provided, this encoder will use the other encoder's cache instead
            of creating its own. Useful for "own" encoders to reuse shared encoder caches.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        feature_dim: int = 32,
        use_encoders: bool = True,
        share_cache_with: Optional['MultiGridGoalEncoder'] = None
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.use_encoders = use_encoders

        if not use_encoders:
            # Identity mode: output is the 6-dim input (features + type discriminator)
            self.feature_dim = 6
            self.encoder = nn.Identity()  # Dummy for state_dict compatibility
        else:
            # Normal mode: create MLP
            # Input: 6-dim vector = [4 features, 2 type discriminator]
            # The network can learn to use the type discriminator
            self.encoder = nn.Sequential(
                nn.Linear(6, feature_dim),
                nn.ReLU(),
            )

        # Internal cache for goal tensors (after goal.encode() call)
        # Keys are cache_key tuples returned by goal.encode()
        # If share_cache_with is provided, reuse that encoder's cache
        if share_cache_with is not None:
            self._raw_cache = share_cache_with._raw_cache
            self._shared_cache = True
        else:
            self._raw_cache: Dict[Tuple, torch.Tensor] = {}
            self._shared_cache = False
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the internal goal tensor cache."""
        self._raw_cache.clear()

    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) cache statistics."""
        return self._cache_hits, self._cache_misses

    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        self._cache_hits = 0
        self._cache_misses = 0

    def forward(self, goal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode goal tensor through the neural network.

        Args:
            goal_tensor: (batch, 6) tensor with goal representation
                        Format: [feature0, feature1, feature2, feature3, spatial_type, orient_type]
                        - Spatial: [x1, y1, x2, y2, 1.0, 0.0]
                        - Orientation: [d0, d1, d2, d3, 0.0, 1.0] (one-hot direction)

        Returns:
            Feature tensor (batch, feature_dim)

        If use_encoders=False, returns input unchanged (identity for debugging).
        """
        if not self.use_encoders:
            # Identity mode: return input unchanged
            return goal_tensor

        # Unified encoder for all goal types
        # The network learns to use the type discriminator (dims 4-5)
        return self.encoder(goal_tensor)

    def tensorize_goal(
        self,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Convert goal to input tensor using goal.encode() method.

        This method delegates encoding to the goal object itself via goal.encode(),
        which returns (tensor, cache_key). Results are cached to avoid redundant
        computation. Call forward() on the returned tensor to get neural network features.

        Args:
            goal: PossibleGoal object with encode() method.
            device: Torch device.

        Returns:
            Tensor (1, 6) with goal encoding including type discriminator:
            - Spatial goals: [x1, y1, x2, y2, 1.0, 0.0]
            - Orientation goals: [d0, d1, d2, d3, 0.0, 1.0] (one-hot direction)
        """
        # Use polymorphic encode() method
        if hasattr(goal, 'encode') and callable(goal.encode):
            # Get encoding and cache key from goal
            tensor, cache_key = goal.encode(device='cpu')  

            if cache_key in self._raw_cache:
                self._cache_hits += 1
                cached = self._raw_cache[cache_key]
                if cached.device != torch.device(device):
                    return cached.clone().to(device)
                return cached.clone()

            self._cache_misses += 1

            # Cache the result (on CPU to save GPU memory)
            self._raw_cache[cache_key] = tensor.clone().to('cpu')

            # Move to requested device
            if tensor.device != torch.device(device):
                tensor = tensor.to(device)

            return tensor

        # In case the goal does not implement encode(), use legacy encoding
        return self._legacy_encode(goal, device)

    def _legacy_encode(self, goal: Any, device: str) -> torch.Tensor:
        """
        Legacy encoding for goals that don't implement encode().

        This provides backward compatibility for old code or tuple-based goals.
        New code should use PossibleGoal.encode() instead.

        Returns 6-dim tensor with spatial type discriminator [1.0, 0.0].
        """
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = float(x), float(y), float(x), float(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = goal
            elif len(goal) >= 2:
                x1, y1 = float(goal[0]), float(goal[1])
                x2, y2 = x1, y1
            else:
                return torch.zeros(1, 6, dtype=torch.float32, device=device)
        else:
            return torch.zeros(1, 6, dtype=torch.float32, device=device)

        # Assume spatial goal for legacy (type discriminator = [1.0, 0.0])
        return torch.tensor(
            [[float(x1), float(y1), float(x2), float(y2), 1.0, 0.0]],
            dtype=torch.float32,
            device=device
        )

    @staticmethod
    def compute_goal_weight(goal: Any) -> float:
        """
        Compute the weight of a goal using polymorphic compute_weight().

        Args:
            goal: Goal object.

        Returns:
            Weight as float.
        """
        # Use polymorphic method if available
        if hasattr(goal, 'compute_weight') and callable(goal.compute_weight):
            return goal.compute_weight()

        # Legacy fallback
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return float((1 + x2 - x1) * (1 + y2 - y1))
        elif hasattr(goal, 'target_pos'):
            return 1.0
        else:
            return 1.0

    @staticmethod
    def get_goal_bounding_box(goal: Any) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box from a spatial goal object.

        Only applicable to spatial goals (ReachCell, ReachRectangle).
        Returns None for non-spatial goals (e.g., OrientationGoal).

        Args:
            goal: Goal object.

        Returns:
            Bounding box as (x1, y1, x2, y2) with normalized coordinates,
            or None if goal is not a spatial goal.
        """
        # Check if it's a spatial goal
        if hasattr(goal, 'target_dir'):  # OrientationGoal or similar
            return None

        # Extract bounding box from spatial goals
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = goal
            elif len(goal) >= 2:
                x1, y1 = int(goal[0]), int(goal[1])
                x2, y2 = x1, y1
            else:
                return None
        else:
            return None

        # Normalize coordinates
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return (x1, y1, x2, y2)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'feature_dim': self.feature_dim,
            'use_encoders': self.use_encoders,
        }
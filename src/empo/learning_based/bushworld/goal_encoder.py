"""
BushWorld-specific goal encoder for Phase 2.

Encodes a goal as its normalised bounding box ``(x1, y1, x2, y2)``. This works
uniformly for both cell goals and rectangle goals (both expose
``target_rect``), removing the old limitation that the neural path only
supported cell goals.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..phase1.goal_encoder import BaseGoalEncoder
from .constants import GOAL_COORD_DIM
from .feature_extraction import extract_goal_coords


class BushWorldGoalEncoder(BaseGoalEncoder):
    """MLP goal encoder for BushWorld.

    Args:
        grid_height: Number of rows (for normalisation).
        grid_width: Number of columns (for normalisation).
        feature_dim: Output feature dimension when ``use_encoders=True``.
        hidden_dim: Hidden layer width.
        use_encoders: If ``False``, the encoder returns the raw 4-d coordinate
            vector and ``feature_dim`` equals ``GOAL_COORD_DIM``.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        feature_dim: int = 32,
        hidden_dim: int = 32,
        use_encoders: bool = True,
    ):
        super().__init__(feature_dim=feature_dim if use_encoders else GOAL_COORD_DIM)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.hidden_dim = hidden_dim
        self.use_encoders = use_encoders

        if use_encoders:
            self.net = nn.Sequential(
                nn.Linear(GOAL_COORD_DIM, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
                nn.ReLU(),
            )
        else:
            self.net = None

        self._cache: Dict[Any, torch.Tensor] = {}
        self._hits = 0
        self._misses = 0

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_stats(self) -> Tuple[int, int]:
        return self._hits, self._misses

    def reset_cache_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    def tensorize_goal(self, goal: Any, device: str = "cpu") -> torch.Tensor:
        """Return the raw goal-coordinate tensor of shape ``(1, GOAL_COORD_DIM)``."""
        cached = self._cache.get(goal)
        if cached is not None:
            self._hits += 1
            return cached.to(device)
        self._misses += 1
        coords = extract_goal_coords(goal, self.grid_width, self.grid_height)
        tensor = torch.from_numpy(coords).unsqueeze(0)
        self._cache[goal] = tensor
        return tensor.to(device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode raw goal-coordinate tensor(s) of shape ``(batch, GOAL_COORD_DIM)``."""
        if self.net is None:
            return coords
        return self.net(coords)

    def get_config(self) -> Dict[str, Any]:
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "use_encoders": self.use_encoders,
        }

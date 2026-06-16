"""
BushWorld-specific state encoder for Phase 2.

Encodes a BushWorld state ``(step_count, positions, densities)`` into a
fixed-size feature vector. BushWorld is small, so a simple MLP over the
flattened multi-channel grid (density + robot/human occupancy) plus the global
step feature is sufficient; there is no need for the CNN / multi-input design
used by multigrid.

Like the multigrid encoder, ``tensorize_state`` performs *featurisation only*
(returning a raw input tensor) and ``forward`` runs the neural network. A small
per-state cache of raw tensors is kept so that repeated tensorisation of the
same state is cheap; the cache stores raw tensors (not NN outputs) to preserve
gradient flow.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..phase1.state_encoder import BaseStateEncoder
from .constants import NUM_GLOBAL_WORLD_FEATURES, NUM_GRID_CHANNELS
from .feature_extraction import extract_state_vector


class BushWorldStateEncoder(BaseStateEncoder):
    """MLP state encoder for BushWorld.

    Args:
        grid_height: Number of rows.
        grid_width: Number of columns.
        B: Maximum bush density (for normalisation).
        num_robots: Number of robot agents (used to split occupancy channels).
        max_steps: Episode horizon (for normalising the step count).
        feature_dim: Output feature dimension when ``use_encoders=True``.
        hidden_dim: Hidden layer width.
        use_encoders: If ``False``, the encoder is the identity (the raw input
            vector is returned and ``feature_dim`` equals the input dimension).
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robots: int,
        max_steps: int,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        use_encoders: bool = True,
    ):
        input_dim = NUM_GRID_CHANNELS * grid_height * grid_width + NUM_GLOBAL_WORLD_FEATURES
        super().__init__(feature_dim=feature_dim if use_encoders else input_dim)

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        self.use_encoders = use_encoders
        self.input_dim = input_dim

        if use_encoders:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
                nn.ReLU(),
            )
        else:
            self.net = None

        self._cache: Dict[Any, torch.Tensor] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # Cache management (API-compatible with the multigrid encoder)
    # ------------------------------------------------------------------ #
    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_stats(self) -> Tuple[int, int]:
        return self._hits, self._misses

    def reset_cache_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    def tensorize_state(self, state: Any, world_model: Any = None, device: str = "cpu") -> torch.Tensor:
        """Return the raw input tensor of shape ``(1, input_dim)`` for ``state``."""
        cached = self._cache.get(state)
        if cached is not None:
            self._hits += 1
            return cached.to(device)
        self._misses += 1
        vec = extract_state_vector(
            state,
            self.grid_width,
            self.grid_height,
            self.num_robots,
            self.B,
            self.max_steps,
        )
        tensor = torch.from_numpy(vec).unsqueeze(0)
        self._cache[state] = tensor
        return tensor.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input tensor(s) of shape ``(batch, input_dim)``."""
        if self.net is None:
            return x
        return self.net(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "B": self.B,
            "num_robots": self.num_robots,
            "max_steps": self.max_steps,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "use_encoders": self.use_encoders,
        }

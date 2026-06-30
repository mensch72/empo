"""
BushWorld-specific agent encoder for Phase 2.

Encodes a single agent's identity/position as a small feature vector. Used by
the human goal-achievement network (V_h^e) and aggregate goal-ability network
(X_h) to condition on *which* human agent is being queried.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .constants import AGENT_FEATURE_SIZE
from .feature_extraction import extract_agent_features


class BushWorldAgentEncoder(nn.Module):
    """MLP encoder for a single agent's identity features.

    Args:
        num_agents: Total number of agents (for normalising the index).
        grid_height: Number of rows (for normalising the position).
        grid_width: Number of columns (for normalising the position).
        output_dim: Output feature dimension when ``use_encoders=True``.
        hidden_dim: Hidden layer width.
        use_encoders: If ``False``, the encoder returns the raw 3-d feature
            vector and ``output_dim`` equals ``AGENT_FEATURE_SIZE``.
    """

    def __init__(
        self,
        num_agents: int,
        grid_height: int,
        grid_width: int,
        output_dim: int = 16,
        hidden_dim: int = 16,
        use_encoders: bool = True,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.hidden_dim = hidden_dim
        self.use_encoders = use_encoders
        self.output_dim = output_dim if use_encoders else AGENT_FEATURE_SIZE

        if use_encoders:
            self.net = nn.Sequential(
                nn.Linear(AGENT_FEATURE_SIZE, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
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

    def tensorize_single(
        self, agent_index: int, state: Any, world_model: Any = None, device: str = "cpu"
    ) -> torch.Tensor:
        """Return the raw agent-feature tensor of shape ``(1, AGENT_FEATURE_SIZE)``."""
        _, positions, _ = state
        key = (agent_index, positions)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached.to(device)
        self._misses += 1
        feats = extract_agent_features(
            state, agent_index, self.num_agents, self.grid_width, self.grid_height
        )
        tensor = torch.from_numpy(feats).unsqueeze(0)
        self._cache[key] = tensor
        return tensor.to(device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode raw agent-feature tensor(s) of shape ``(batch, AGENT_FEATURE_SIZE)``."""
        if self.net is None:
            return t
        return self.net(t)

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "use_encoders": self.use_encoders,
        }

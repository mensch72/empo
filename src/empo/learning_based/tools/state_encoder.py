"""
State encoder for the Tools WorldModel environment.

Converts the hashable state tuple ``(remaining, workbench, holds, requested)``
into a flat float32 tensor and passes it through a small MLP to produce a
fixed-size feature vector.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ToolsStateEncoder(nn.Module):
    """MLP-based state encoder for :class:`ToolsWorldModel`.

    The raw state is a tuple::

        (remaining_steps, workbench, holds, requested)

    where ``workbench``, ``holds``, and ``requested`` are nested tuples of
    shape ``(n_agents, n_tools)`` with binary values.  This encoder
    flattens the state into a 1-D vector of dimension
    ``1 + 3 * n_agents * n_tools`` and passes it through a two-layer MLP
    to produce a feature vector of dimension ``feature_dim``.

    Parameters
    ----------
    n_agents : int
        Number of agents.
    n_tools : int
        Number of tools.
    max_steps : int
        Maximum episode length (used to normalise ``remaining_steps``).
    feature_dim : int
        Output feature dimension.
    hidden_dim : int
        Hidden-layer width of the encoder MLP.
    """

    def __init__(
        self,
        n_agents: int,
        n_tools: int,
        max_steps: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_tools = n_tools
        self.max_steps = max_steps
        self.feature_dim = feature_dim

        # 1 (normalised remaining) + 3 binary planes of (n_agents * n_tools)
        raw_dim = 1 + 3 * n_agents * n_tools

        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def tensorize_state(
        self,
        state: Any,
        world_model: Any = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Convert a hashable tools state to a flat float32 tensor.

        Returns a tensor of shape ``(1, raw_dim)`` on *device*.
        """
        remaining, workbench, holds, requested = state
        vec = [remaining / max(self.max_steps, 1)]
        for plane in (workbench, holds, requested):
            for agent_row in plane:
                vec.extend(float(v) for v in agent_row)
        return torch.tensor([vec], dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensorized state.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, raw_dim)`` — output of :meth:`tensorize_state`.

        Returns
        -------
        Tensor of shape ``(batch, feature_dim)``.
        """
        return self.mlp(x)

"""
Neural networks for the EMPO learning approach.

Three networks, one per learned quantity:
    QhNet  — Phase 1: Q_h^m(s, g, a_h) human Q-values (goal-conditioned)
    VheNet — Phase 2: V_h^e(s, g)       goal achievement probability
    QrNet  — Phase 2: Q_r(s, a_r)       robot Q-values

All networks accept the flat state vector produced by StateEncoder.
Internally they reshape the grid portion into (C, H, W) and run a CNN
to extract spatial features, then concatenate with the non-spatial
features (agent direction, carrying, time) before a final MLP head.

Design choices:
    - A shared GridFeatureExtractor handles the CNN + concat + projection.
      Each network instantiates its own copy (no weight sharing between
      networks — they learn different things).
    - Goal-conditioned networks (QhNet, VheNet) concatenate the goal
      encoding with the extracted features before the head MLP.
    - VheNet uses sigmoid output since V_h^e is a probability in [0, 1].
    - QhNet and QrNet have unbounded outputs (Q-values).
    - from_encoders() classmethods let you create a network directly from
      a StateEncoder + GoalEncoder without computing dimensions by hand.
"""

import torch
import torch.nn as nn
from typing import Optional

from empo.ali_learning_based.encoders import NUM_GRID_CHANNELS


class GridFeatureExtractor(nn.Module):
    """
    Extracts a fixed-size feature vector from a flat state encoding.

    Internally:
    1. Splits the flat vector into grid channels (C, H, W) and extra features
    2. CNN on grid → flatten  (outputs 32 × H × W)
    3. Concatenate with extra features (agent directions, carrying, time)
    4. Linear projection to feature_dim

    Args:
        grid_channels: Number of grid channels (8 from our encoder).
        grid_height: Grid height.
        grid_width: Grid width.
        extra_dim: Dimension of non-spatial features in the flat vector.
        feature_dim: Output feature dimension.
    """

    CONV_CHANNELS = 32  # channels in conv layers

    def __init__(
        self,
        grid_channels: int,
        grid_height: int,
        grid_width: int,
        extra_dim: int,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_size = grid_channels * grid_height * grid_width
        self.feature_dim = feature_dim

        C = self.CONV_CHANNELS

        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, C, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # padding=1 preserves spatial dims, so output is C × H × W
        conv_out_dim = C * grid_height * grid_width

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + extra_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, flat_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flat_state: (batch, state_dim) flat state encoding.

        Returns:
            (batch, feature_dim) feature vector.
        """
        batch = flat_state.shape[0]

        # Split: [grid_flat | extra_features]
        grid_flat = flat_state[:, : self.grid_size]
        extra = flat_state[:, self.grid_size :]

        # Reshape grid to spatial tensor and run CNN
        grid = grid_flat.view(batch, self.grid_channels, self.grid_height, self.grid_width)
        conv_out = self.conv(grid).view(batch, -1)

        # Combine spatial and non-spatial features
        combined = torch.cat([conv_out, extra], dim=-1)
        return self.fc(combined)


# ---------------------------------------------------------------------------
# Phase 1: Human Q-network
# ---------------------------------------------------------------------------

class QhNet(nn.Module):
    """
    Phase 1 network: Q_h^m(s, g, a_h) for all human actions.

    Takes a flat state encoding and a goal encoding, returns Q-values
    for each action the human can take.

    Architecture:
        state → GridFeatureExtractor → state_features (128)
        cat(state_features, goal_enc) → MLP → Q-values (num_actions)

    Args:
        grid_channels, grid_height, grid_width, extra_dim: Grid structure
            (see GridFeatureExtractor).
        goal_dim: Dimension of goal encoding (4 from our GoalEncoder).
        num_actions: Number of actions (6 for multigrid).
        feature_dim: Internal feature dimension.
        head_dim: Hidden dimension in the output MLP head.
    """

    def __init__(
        self,
        grid_channels: int,
        grid_height: int,
        grid_width: int,
        extra_dim: int,
        goal_dim: int,
        num_actions: int,
        feature_dim: int = 128,
        head_dim: int = 64,
    ):
        super().__init__()
        self.features = GridFeatureExtractor(
            grid_channels, grid_height, grid_width, extra_dim, feature_dim
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_actions),
        )

    def forward(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) flat state encoding.
            goal:  (batch, goal_dim)  goal encoding.

        Returns:
            (batch, num_actions) Q-values for each human action.
        """
        f = self.features(state)
        x = torch.cat([f, goal], dim=-1)
        return self.head(x)

    @classmethod
    def from_encoders(cls, state_encoder, goal_encoder, num_actions, **kwargs):
        """Create from encoder objects (avoids manual dimension math)."""
        grid_size = NUM_GRID_CHANNELS * state_encoder.height * state_encoder.width
        extra_dim = state_encoder.dim - grid_size
        return cls(
            grid_channels=NUM_GRID_CHANNELS,
            grid_height=state_encoder.height,
            grid_width=state_encoder.width,
            extra_dim=extra_dim,
            goal_dim=goal_encoder.dim,
            num_actions=num_actions,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Phase 2: Goal achievement probability
# ---------------------------------------------------------------------------

class VheNet(nn.Module):
    """
    Phase 2 network: V_h^e(s, g) — probability that human achieves goal g.

    Same architecture as QhNet but outputs a single probability via sigmoid.

    Args:
        Same as QhNet, minus num_actions.
    """

    def __init__(
        self,
        grid_channels: int,
        grid_height: int,
        grid_width: int,
        extra_dim: int,
        goal_dim: int,
        feature_dim: int = 128,
        head_dim: int = 64,
    ):
        super().__init__()
        self.features = GridFeatureExtractor(
            grid_channels, grid_height, grid_width, extra_dim, feature_dim
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim + goal_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) flat state encoding.
            goal:  (batch, goal_dim)  goal encoding.

        Returns:
            (batch,) scalar probability in [0, 1].
        """
        f = self.features(state)
        x = torch.cat([f, goal], dim=-1)
        return self.head(x).squeeze(-1)

    @classmethod
    def from_encoders(cls, state_encoder, goal_encoder, **kwargs):
        """Create from encoder objects."""
        grid_size = NUM_GRID_CHANNELS * state_encoder.height * state_encoder.width
        extra_dim = state_encoder.dim - grid_size
        return cls(
            grid_channels=NUM_GRID_CHANNELS,
            grid_height=state_encoder.height,
            grid_width=state_encoder.width,
            extra_dim=extra_dim,
            goal_dim=goal_encoder.dim,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Phase 2: Robot Q-network
# ---------------------------------------------------------------------------

class QrNet(nn.Module):
    """
    Phase 2 network: Q_r(s, a_r) for all robot actions.

    Not goal-conditioned — the robot's Q-values depend on state only.
    Goal information is captured indirectly through the intrinsic reward
    U_r(s) which aggregates over all goals.

    Args:
        grid_channels, grid_height, grid_width, extra_dim: Grid structure.
        num_actions: Number of robot actions.
        feature_dim: Internal feature dimension.
        head_dim: Hidden dimension in the output MLP head.
    """

    def __init__(
        self,
        grid_channels: int,
        grid_height: int,
        grid_width: int,
        extra_dim: int,
        num_actions: int,
        feature_dim: int = 128,
        head_dim: int = 64,
    ):
        super().__init__()
        self.features = GridFeatureExtractor(
            grid_channels, grid_height, grid_width, extra_dim, feature_dim
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) flat state encoding.

        Returns:
            (batch, num_actions) Q-values for each robot action.
        """
        f = self.features(state)
        return self.head(f)

    @classmethod
    def from_encoders(cls, state_encoder, num_actions, **kwargs):
        """Create from encoder object."""
        grid_size = NUM_GRID_CHANNELS * state_encoder.height * state_encoder.width
        extra_dim = state_encoder.dim - grid_size
        return cls(
            grid_channels=NUM_GRID_CHANNELS,
            grid_height=state_encoder.height,
            grid_width=state_encoder.width,
            extra_dim=extra_dim,
            num_actions=num_actions,
            **kwargs,
        )

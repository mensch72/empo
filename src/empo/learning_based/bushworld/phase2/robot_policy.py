"""
BushWorld Robot Policy for Phase 2.

Provides a deployable robot policy that wraps a trained Q_r network (neural or
lookup table), mirroring :class:`MultiGridRobotPolicy`.
"""

from typing import Any, Optional, Union

import torch

from empo.robot_policy import RobotPolicy
from ...phase2.lookup.robot_q_network import LookupTableRobotQNetwork
from .robot_q_network import BushWorldRobotQNetwork


class BushWorldRobotPolicy(RobotPolicy):
    """Deployable robot policy for BushWorld.

    Load from a checkpoint saved by ``trainer.save_policy()``::

        policy = BushWorldRobotPolicy(path="policy.pt")
        policy.reset(env)
        action = policy.sample(env.get_state())
    """

    def __init__(
        self,
        q_network: Optional[Union[BushWorldRobotQNetwork, LookupTableRobotQNetwork]] = None,
        beta_r: float = 10.0,
        device: str = "cpu",
        path: Optional[str] = None,
    ):
        self.device = device
        self._world_model = None
        self._is_lookup_table = False

        if q_network is None:
            if path is None:
                raise ValueError(
                    "Either q_network or path must be provided. "
                    "Use: BushWorldRobotPolicy(path='policy.pt')"
                )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if "q_r_config" not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'q_r_config'. Was it saved with trainer.save_policy()?"
                )
            config = checkpoint["q_r_config"]
            self.beta_r = checkpoint.get("beta_r", beta_r)

            if config.get("type") == "lookup_table":
                self._is_lookup_table = True
                self.q_network = LookupTableRobotQNetwork(
                    num_actions=config["num_actions"],
                    num_robots=config["num_robots"],
                    beta_r=self.beta_r,
                    default_q_r=config.get("default_q_r", -1.0),
                    feasible_range=config.get("feasible_range"),
                )
                self.q_network.load_state_dict(checkpoint["q_r"])
            else:
                self.q_network = BushWorldRobotQNetwork(
                    grid_height=config["grid_height"],
                    grid_width=config["grid_width"],
                    B=config["B"],
                    num_robot_actions=config["num_robot_actions"],
                    num_robots=config["num_robots"],
                    num_humans=config["num_humans"],
                    max_steps=config["max_steps"],
                    state_feature_dim=config["state_feature_dim"],
                    hidden_dim=config["hidden_dim"],
                    beta_r=self.beta_r,
                    feasible_range=config.get("feasible_range"),
                    use_encoders=config.get("use_encoders", True),
                    use_z_space=config.get("use_z_space", False),
                    eta=config.get("eta", 1.1),
                    xi=config.get("xi", 1.0),
                )
                self.q_network.load_state_dict(checkpoint["q_r"])
        else:
            self.q_network = q_network
            self.beta_r = beta_r
            self._is_lookup_table = isinstance(q_network, LookupTableRobotQNetwork)
            if path is not None:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                self.q_network.load_state_dict(checkpoint["q_r"])
                self.beta_r = checkpoint.get("beta_r", beta_r)

        self.q_network.to(device)
        self.q_network.eval()

    def reset(self, world_model: Any) -> None:
        self._world_model = world_model

    def sample(self, state: Any) -> Any:
        if self._world_model is None:
            raise RuntimeError(
                "Must call reset(world_model) before sample(). "
                "The world model is needed to tensorize the state."
            )
        with torch.no_grad():
            q_values = self.q_network.forward(state, self._world_model, self.device)
            return self.q_network.sample_action(q_values, beta_r=self.beta_r)

    def get_distribution(self, state: Any) -> dict:
        """Return the robot policy distribution ``{action_profile: prob}``.

        Mirrors the call signature of :class:`TabularRobotPolicy` so the two
        policy types can be compared with the same code.
        """
        import itertools

        if self._world_model is None:
            raise RuntimeError(
                "Must call reset(world_model) before get_distribution()."
            )
        num_actions = self._world_model.action_space.n
        num_robots = self._world_model.num_robots
        with torch.no_grad():
            q_values = self.q_network.forward(state, self._world_model, self.device)
            probs = (
                self.q_network.get_policy(q_values, beta_r=self.beta_r)
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )
        profiles = list(itertools.product(range(num_actions), repeat=num_robots))
        return {profile: float(probs[i]) for i, profile in enumerate(profiles)}

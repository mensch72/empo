"""
PufferLib-based Phase 2 trainer for EMPO.

This module orchestrates Phase 2 training using **PufferLib's PuffeRL** class
for the core PPO loop (rollout collection, advantage computation, clipped
surrogate update).  EMPO-specific auxiliary network training (V_h^e, X_h,
U_r) runs alongside PufferLib as a post-train hook.

Key PufferLib integration points:

- ``pufferlib.vector.make()`` creates vectorised environments.
- ``pufferlib.pufferl.PuffeRL`` drives ``evaluate()`` + ``train()``.
- ``EMPOActorCritic.forward(observations, state)`` → ``(logits, value)``
  follows PufferLib's expected policy interface.
- ``pufferlib.pytorch.sample_logits`` handles action sampling.

This module does NOT import or modify the existing DQN-path trainer
(``learning_based.phase2.trainer``).  It reads only base-class interfaces
(``Phase2Transition``, ``Phase2ReplayBuffer``, ``BaseHumanGoalAchievement-
Network``, etc.) that are stable public API.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.vector
import pufferlib.pufferl
import pufferlib.emulation

# Read-only imports from existing DQN path (shared data structures)
from empo.learning_based.phase2.replay_buffer import (
    Phase2ReplayBuffer,
)
from empo.learning_based.phase2.human_goal_ability import (
    BaseHumanGoalAchievementNetwork,
)
from empo.learning_based.phase2.aggregate_goal_ability import (
    BaseAggregateGoalAbilityNetwork,
)
from empo.learning_based.phase2.intrinsic_reward_network import (
    BaseIntrinsicRewardNetwork,
)

from .actor_critic import EMPOActorCritic
from .config import PPOPhase2Config

logger = logging.getLogger(__name__)

# Floor for X_h values to prevent numerical instability in X_h^{-ξ}
# (division by zero / explosion when X_h → 0).  Matches the DQN path.
_X_H_MIN = 1e-3


# ======================================================================
# Container for auxiliary networks (PPO path)
# ======================================================================


@dataclass
class PPOAuxiliaryNetworks:
    """Container for the auxiliary networks used in the PPO path.

    These are the *same* base-class types as the DQN path but are
    instantiated independently.  The PPO trainer never touches
    ``Phase2Networks`` from the DQN path.
    """

    v_h_e: BaseHumanGoalAchievementNetwork
    x_h: Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r: Optional[BaseIntrinsicRewardNetwork] = None

    # Frozen copies for reward computation during rollouts
    v_h_e_target: Optional[BaseHumanGoalAchievementNetwork] = None
    x_h_target: Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r_target: Optional[BaseIntrinsicRewardNetwork] = None


# ======================================================================
# PPO Phase 2 Trainer  (PufferLib-backed)
# ======================================================================


class PPOPhase2Trainer:
    """PufferLib-backed PPO Phase 2 trainer.

    This trainer uses ``pufferlib.pufferl.PuffeRL`` for the core PPO loop
    and adds EMPO-specific auxiliary-network training on top.

    Parameters
    ----------
    actor_critic : EMPOActorCritic
        The PPO actor-critic network (follows PufferLib policy convention).
    auxiliary_networks : PPOAuxiliaryNetworks
        Auxiliary networks for V_h^e, X_h, U_r.
    config : PPOPhase2Config
        Full PPO Phase 2 configuration.
    device : str
        Torch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(
        self,
        actor_critic: EMPOActorCritic,
        auxiliary_networks: PPOAuxiliaryNetworks,
        config: PPOPhase2Config,
        device: str = "cpu",
    ):
        self.actor_critic = actor_critic.to(device)
        self.auxiliary_networks = auxiliary_networks
        self.config = config
        self.device = device

        # Move auxiliary networks to the same device as the actor-critic
        # to prevent device-mismatch errors in forward() calls.
        for net_name in ("v_h_e", "x_h", "u_r"):
            net = getattr(auxiliary_networks, net_name, None)
            if net is not None and hasattr(net, "to"):
                net.to(device)

        # Auxiliary optimisers
        self.aux_optimizers: Dict[str, optim.Optimizer] = {}
        self.aux_optimizers["v_h_e"] = optim.Adam(
            auxiliary_networks.v_h_e.parameters(),
            lr=config.lr_v_h_e,
            weight_decay=config.v_h_e_weight_decay,
        )
        if auxiliary_networks.x_h is not None:
            self.aux_optimizers["x_h"] = optim.Adam(
                auxiliary_networks.x_h.parameters(),
                lr=config.lr_x_h,
                weight_decay=config.x_h_weight_decay,
            )
        if auxiliary_networks.u_r is not None:
            self.aux_optimizers["u_r"] = optim.Adam(
                auxiliary_networks.u_r.parameters(),
                lr=config.lr_u_r,
                weight_decay=config.u_r_weight_decay,
            )

        # Auxiliary replay buffer
        self.aux_replay_buffer = Phase2ReplayBuffer(
            capacity=config.aux_buffer_size
        )

        # Counters
        self.training_step_count: int = 0
        self.ppo_iteration: int = 0

    # ------------------------------------------------------------------
    # Target-network management
    # ------------------------------------------------------------------

    def freeze_auxiliary_networks(self) -> None:
        """Create frozen copies of auxiliary networks for reward computation."""
        nets = self.auxiliary_networks
        nets.v_h_e_target = copy.deepcopy(nets.v_h_e)
        nets.v_h_e_target.eval()
        for p in nets.v_h_e_target.parameters():
            p.requires_grad = False
        if hasattr(nets.v_h_e_target, "to"):
            nets.v_h_e_target.to(self.device)
        if nets.x_h is not None:
            nets.x_h_target = copy.deepcopy(nets.x_h)
            nets.x_h_target.eval()
            for p in nets.x_h_target.parameters():
                p.requires_grad = False
            if hasattr(nets.x_h_target, "to"):
                nets.x_h_target.to(self.device)
        if nets.u_r is not None:
            nets.u_r_target = copy.deepcopy(nets.u_r)
            nets.u_r_target.eval()
            for p in nets.u_r_target.parameters():
                p.requires_grad = False
            if hasattr(nets.u_r_target, "to"):
                nets.u_r_target.to(self.device)

    # ------------------------------------------------------------------
    # Auxiliary training
    # ------------------------------------------------------------------

    def push_transition_to_aux_buffer(
        self,
        state: Any,
        next_state: Any,
        robot_action: tuple,
        goals: dict,
        goal_weights: dict,
        human_actions: list,
        transition_probs: Optional[dict],
        terminal: bool,
    ) -> None:
        """Push a single transition into the auxiliary replay buffer."""
        self.aux_replay_buffer.push(
            state=state,
            robot_action=robot_action,
            goals=goals,
            goal_weights=goal_weights,
            human_actions=human_actions,
            next_state=next_state,
            transition_probs_by_action=transition_probs,
            terminal=terminal,
        )

    def train_auxiliary_step(
        self,
        world_model: Any = None,
    ) -> Dict[str, float]:
        """Run one gradient step on the auxiliary networks.

        Mirrors the DQN path's ``compute_losses`` but uses the base-class
        ``forward()`` interface (single-item, not ``forward_batch``).

        Parameters
        ----------
        world_model : WorldModel or None
            A world model instance used for ``forward()`` calls on the
            auxiliary networks.  When ``None``, V_h^e and X_h forward passes
            are skipped (useful for unit testing the plumbing).

        Returns
        -------
        losses : dict[str, float]
        """
        if len(self.aux_replay_buffer) < self.config.batch_size:
            return {}

        batch = self.aux_replay_buffer.sample(self.config.batch_size)
        cfg = self.config
        nets = self.auxiliary_networks
        losses: Dict[str, float] = {}

        # -----------------------------------------------------------------
        # V_h^e loss: MSE between V_h^e(s, g_h) and TD target
        # -----------------------------------------------------------------
        if world_model is not None:
            v_h_e_preds: List[torch.Tensor] = []
            v_h_e_targets: List[torch.Tensor] = []

            for t in batch:
                for h_idx, goal in t.goals.items():
                    # Forward pass (online network)
                    pred = nets.v_h_e(
                        t.state, world_model, h_idx, goal, self.device
                    )
                    v_h_e_preds.append(pred.squeeze())

                    # Target from target network
                    with torch.no_grad():
                        target_net = nets.v_h_e_target or nets.v_h_e
                        # Check if goal achieved in next_state
                        achieved = (
                            goal.is_achieved(t.next_state)
                            if hasattr(goal, "is_achieved")
                            else 0
                        )
                        if achieved:
                            target = torch.tensor(1.0, device=self.device)
                        elif t.terminal:
                            target = torch.tensor(0.0, device=self.device)
                        else:
                            next_v = target_net(
                                t.next_state,
                                world_model,
                                h_idx,
                                goal,
                                self.device,
                            )
                            next_v = target_net.apply_hard_clamp(next_v)
                            target = cfg.gamma_h * next_v.squeeze()
                    v_h_e_targets.append(target)

            if v_h_e_preds:
                preds_t = torch.stack(v_h_e_preds)
                targets_t = torch.stack(v_h_e_targets)
                loss_v = ((preds_t - targets_t) ** 2).mean()

                opt = self.aux_optimizers["v_h_e"]
                opt.zero_grad()
                loss_v.backward()
                if cfg.v_h_e_grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        nets.v_h_e.parameters(), cfg.v_h_e_grad_clip
                    )
                opt.step()
                losses["v_h_e_loss"] = loss_v.item()

        # -----------------------------------------------------------------
        # X_h loss: MSE between X_h(s, h) and w_h * V_h^e(s, g_h)^ζ
        # -----------------------------------------------------------------
        if (
            nets.x_h is not None
            and world_model is not None
            and "x_h" in self.aux_optimizers
        ):
            x_h_preds: List[torch.Tensor] = []
            x_h_targets_list: List[torch.Tensor] = []

            for t in batch:
                for h_idx, goal in t.goals.items():
                    weight = t.goal_weights.get(h_idx, 1.0)
                    pred = nets.x_h(
                        t.state, world_model, h_idx, self.device
                    )
                    x_h_preds.append(pred.squeeze())

                    with torch.no_grad():
                        v_target_net = nets.v_h_e_target or nets.v_h_e
                        v_for_x = v_target_net(
                            t.state, world_model, h_idx, goal, self.device
                        )
                        v_for_x = v_target_net.apply_hard_clamp(v_for_x)
                        x_target = nets.x_h.compute_target(
                            v_for_x.squeeze(),
                            goal_weight=weight,
                        )
                    x_h_targets_list.append(x_target)

            if x_h_preds:
                xp = torch.stack(x_h_preds)
                xt = torch.stack(x_h_targets_list)
                loss_x = ((xp - xt) ** 2).mean()

                opt = self.aux_optimizers["x_h"]
                opt.zero_grad()
                loss_x.backward()
                if cfg.x_h_grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        nets.x_h.parameters(), cfg.x_h_grad_clip
                    )
                opt.step()
                losses["x_h_loss"] = loss_x.item()

        # -----------------------------------------------------------------
        # U_r loss: MSE between predicted y and target y = E_h[X_h^{-ξ}]
        # -----------------------------------------------------------------
        if (
            nets.u_r is not None
            and world_model is not None
            and "u_r" in self.aux_optimizers
        ):
            y_preds: List[torch.Tensor] = []
            y_targets: List[torch.Tensor] = []

            x_h_target_net = nets.x_h_target or nets.x_h
            for t in batch:
                y_pred, _ = nets.u_r(t.state, world_model, self.device)
                y_preds.append(y_pred.squeeze())

                with torch.no_grad():
                    x_vals: List[float] = []
                    if x_h_target_net is not None:
                        for h_idx in t.goals.keys():
                            xv = x_h_target_net(
                                t.state, world_model, h_idx, self.device
                            )
                            xv = x_h_target_net.apply_hard_clamp(xv)
                            x_vals.append(
                                max(xv.squeeze().item(), _X_H_MIN)
                            )
                    if x_vals:
                        x_t = torch.tensor(x_vals, device=self.device)
                        y_t = (x_t ** (-cfg.xi)).mean()
                    else:
                        y_t = torch.tensor(1.0, device=self.device)
                y_targets.append(y_t)

            if y_preds:
                yp = torch.stack(y_preds)
                yt = torch.stack(y_targets)
                loss_u = ((yp - yt) ** 2).mean()

                opt = self.aux_optimizers["u_r"]
                opt.zero_grad()
                loss_u.backward()
                if cfg.u_r_grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        nets.u_r.parameters(), cfg.u_r_grad_clip
                    )
                opt.step()
                losses["u_r_loss"] = loss_u.item()

        return losses

    # ------------------------------------------------------------------
    # Auxiliary network ↔ environment synchronisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_aux_nets_to_envs(vecenv: Any) -> None:
        """Inject the current (frozen) auxiliary_networks into each env.

        After ``freeze_auxiliary_networks()`` creates frozen target copies,
        this method pushes the ``auxiliary_networks`` reference into every
        ``EMPOMultiGridEnv`` instance managed by the vectorised env pool.
        This ensures that intrinsic rewards computed during rollouts use
        the frozen copies.
        """
        if not hasattr(vecenv, "envs"):
            return
        for env in vecenv.envs:
            # PufferLib wraps envs; unwrap to the Gymnasium env
            inner = env
            while hasattr(inner, "env"):
                inner = inner.env
            if hasattr(inner, "auxiliary_networks"):
                # The env already holds a ref — the trainer can update it
                # by mutating the PPOAuxiliaryNetworks dataclass in place
                # (freeze creates new target attrs on the same object).
                pass

    def _collect_aux_data_from_rollout(
        self, pufferl: Any, vecenv: Any
    ) -> None:
        """Extract auxiliary transition data from PufferLib's rollout infos.

        PufferLib stores per-step info dicts on its ``infos`` attribute
        after ``evaluate()``.  We iterate over them and push transitions
        into the auxiliary replay buffer so that ``train_auxiliary_step()``
        has data to train on.
        """
        infos = getattr(pufferl, "infos", None)
        if infos is None:
            return

        # PufferLib stores infos as a list-of-dicts (one per agent-step)
        if isinstance(infos, dict):
            # Sometimes it's a dict-of-lists; skip if not iterable as expected
            return

        for info in infos:
            if not isinstance(info, dict):
                continue
            # Only push if the info contains the required EMPO auxiliary fields
            state = info.get("state")
            next_state = info.get("next_state")
            goals = info.get("goals")
            if state is None or next_state is None or goals is None:
                continue

            goal_weights = info.get("goal_weights", {})
            human_actions = info.get("human_actions", [])
            transition_probs = info.get("transition_probs")
            terminal = info.get("terminated", False) or info.get("terminal", False)

            # Store robot action as a tuple (for multi-robot compat)
            robot_action = info.get("robot_action", (0,))
            if isinstance(robot_action, int):
                robot_action = (robot_action,)

            self.push_transition_to_aux_buffer(
                state=state,
                next_state=next_state,
                robot_action=robot_action,
                goals=goals,
                goal_weights=goal_weights,
                human_actions=human_actions,
                transition_probs=transition_probs,
                terminal=terminal,
            )

    # ------------------------------------------------------------------
    # Full training loop  (PufferLib-backed)
    # ------------------------------------------------------------------

    def train(
        self,
        env_creator: Callable[[], Any],
        num_iterations: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the full PufferLib PPO Phase 2 training loop.

        Parameters
        ----------
        env_creator : callable
            A zero-argument callable that returns a new Gymnasium-compatible
            :class:`EMPOMultiGridEnv` instance.  The trainer wraps each
            instance with ``pufferlib.emulation.GymnasiumPufferEnv`` and
            passes the wrapped creator to ``pufferlib.vector.make()`` for
            vectorised execution.
        num_iterations : int or None
            Override ``config.num_ppo_iterations``.

        Returns
        -------
        metrics : list[dict]
            Per-iteration loss / metric dictionaries.
        """
        cfg = self.config
        n_iters = num_iterations or cfg.num_ppo_iterations

        # Build PufferLib config dict (PuffeRL expects a flat dict)
        puffer_config = cfg.to_pufferlib_config()
        total_timesteps = n_iters * puffer_config["batch_size"]
        puffer_config["total_timesteps"] = total_timesteps

        # Wrap the Gymnasium env creator with PufferLib's emulation layer
        def puffer_env_creator(buf=None, seed=0):
            return pufferlib.emulation.GymnasiumPufferEnv(
                env_creator=env_creator, buf=buf, seed=seed
            )

        # Create vectorised environments via PufferLib
        vecenv = pufferlib.vector.make(
            puffer_env_creator,
            num_envs=cfg.num_envs,
            backend="Serial",
        )

        # Initialise PuffeRL training driver
        pufferl = pufferlib.pufferl.PuffeRL(
            puffer_config, vecenv, self.actor_critic
        )

        # Initial freeze of auxiliary networks and sync into envs
        self.freeze_auxiliary_networks()
        self._sync_aux_nets_to_envs(vecenv)

        all_metrics: List[Dict[str, float]] = []
        iteration = 0
        # Use a driver env for auxiliary forward passes (world_model access)
        driver_env = vecenv.driver_env
        wm = getattr(driver_env, "world_model", None)
        if wm is None and hasattr(driver_env, "env"):
            wm = getattr(driver_env.env, "world_model", None)

        while pufferl.global_step < total_timesteps and iteration < n_iters:
            self.ppo_iteration = iteration

            # --- PufferLib: collect rollout + PPO update ---
            pufferl.evaluate()

            # --- Extract auxiliary data from rollout info dicts ---
            self._collect_aux_data_from_rollout(pufferl, vecenv)

            pufferl.train()

            # --- EMPO-specific: train auxiliary networks ---
            aux_losses: Dict[str, float] = {}
            for _ in range(cfg.aux_training_steps_per_iteration):
                step_losses = self.train_auxiliary_step(world_model=wm)
                for k, v in step_losses.items():
                    aux_losses[k] = aux_losses.get(k, 0.0) + v

            # Re-freeze auxiliary networks periodically and sync to envs
            if (iteration + 1) % cfg.reward_freeze_interval == 0:
                self.freeze_auxiliary_networks()
                self._sync_aux_nets_to_envs(vecenv)

            metrics = {
                **pufferl.losses,
                **aux_losses,
                "iteration": iteration,
                "global_step": pufferl.global_step,
            }
            all_metrics.append(metrics)
            self.training_step_count = pufferl.global_step
            iteration += 1

            if iteration % 100 == 0:
                logger.info(
                    "PPO iter %d (step %d): policy_loss=%.4f value_loss=%.4f",
                    iteration,
                    pufferl.global_step,
                    pufferl.losses.get("policy_loss", 0),
                    pufferl.losses.get("value_loss", 0),
                )

        pufferl.close()
        return all_metrics

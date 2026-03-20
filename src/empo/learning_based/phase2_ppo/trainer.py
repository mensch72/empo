"""
PPO-based Phase 2 trainer for EMPO.

This module orchestrates PPO-based Phase 2 training:

- Collect on-policy rollouts with the current actor-critic.
- Compute advantages using GAE with the intrinsic reward U_r(s).
- Update the actor-critic via PPO.
- Train the auxiliary networks (V_h^e, X_h, U_r) from the same rollout data.

Earlier design docs for EMPO describe a separate **warm-up** stage that
would train auxiliary networks under a uniform random robot policy before
PPO updates begin, controlled by ``warmup_*_steps`` configuration fields.
That explicit warm-up stage is **not implemented** in this trainer; any
``warmup_*_steps`` fields in the config are currently ignored here, and
all training performed by this module happens within the PPO loop itself.

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
# Rollout buffer (on-policy, separate from the DQN replay buffer)
# ======================================================================


@dataclass
class PPORolloutEntry:
    """A single step from a PPO rollout."""

    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    # Auxiliary data carried via the env ``info`` dict
    info: Dict[str, Any]


class PPORolloutBuffer:
    """Simple on-policy rollout buffer for a single PPO iteration."""

    def __init__(self) -> None:
        self.entries: List[PPORolloutEntry] = []

    def add(self, entry: PPORolloutEntry) -> None:
        self.entries.append(entry)

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)


# ======================================================================
# PPO Phase 2 Trainer
# ======================================================================


class PPOPhase2Trainer:
    """PPO-based Phase 2 trainer.

    This trainer is the PPO-path counterpart of ``BasePhase2Trainer`` in the
    DQN path.  It manages:

    * An :class:`EMPOActorCritic` (actor + critic)
    * A set of :class:`PPOAuxiliaryNetworks` (V_h^e, X_h, U_r)
    * A :class:`Phase2ReplayBuffer` for auxiliary-network training
    * A :class:`PPORolloutBuffer` for on-policy PPO data

    Parameters
    ----------
    actor_critic : EMPOActorCritic
        The PPO actor-critic network.
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

        # PPO optimiser
        self.ppo_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=config.lr_ppo
        )

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

        # On-policy rollout buffer
        self.rollout_buffer = PPORolloutBuffer()

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
        if nets.x_h is not None:
            nets.x_h_target = copy.deepcopy(nets.x_h)
            nets.x_h_target.eval()
            for p in nets.x_h_target.parameters():
                p.requires_grad = False
        if nets.u_r is not None:
            nets.u_r_target = copy.deepcopy(nets.u_r)
            nets.u_r_target.eval()
            for p in nets.u_r_target.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Parameters
        ----------
        rewards : list[float]
            Per-step rewards from the rollout (U_r(s_t)).
        values : list[float]
            Per-step value estimates V_r(s_t) from the critic.
        dones : list[bool]
            Per-step done flags.
        last_value : float
            Bootstrap value V_r(s_{T+1}) for the final step.

        Returns
        -------
        advantages : ndarray, shape (T,)
        returns : ndarray, shape (T,)
        """
        gamma = self.config.gamma_r
        lam = self.config.ppo_gae_lambda
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = (
                rewards[t]
                + gamma * next_value * next_non_terminal
                - values[t]
            )
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def ppo_update(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one PPO update over the collected rollout data.

        Parameters
        ----------
        obs_batch : Tensor, (N, obs_dim)
        actions_batch : Tensor, (N,)
        old_log_probs : Tensor, (N,)
        advantages : Tensor, (N,)
        returns : Tensor, (N,)

        Returns
        -------
        losses : dict[str, float]
            Dictionary with ``'policy_loss'``, ``'value_loss'``,
            ``'entropy'``, and ``'total_loss'`` scalars.
        """
        cfg = self.config
        N = obs_batch.shape[0]
        batch_size = max(1, N // max(1, cfg.ppo_num_minibatches))

        # Normalize advantages over the full rollout (before minibatch split).
        # This is the convention used by CleanRL/PufferLib PPO: normalize once
        # globally rather than per-minibatch, to keep the advantage scale
        # consistent across all minibatches within the same update.
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _epoch in range(cfg.ppo_update_epochs):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                mb_idx = indices[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                _, new_lp, entropy, new_val = (
                    self.actor_critic.get_action_and_value(
                        mb_obs, mb_actions
                    )
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - cfg.ppo_clip_coef, 1.0 + cfg.ppo_clip_coef
                    )
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                # Entropy bonus
                ent_coef = cfg.get_entropy_coef(self.training_step_count)
                entropy_loss = -ent_coef * entropy.mean()

                loss = (
                    policy_loss
                    + cfg.ppo_vf_coef * value_loss
                    + entropy_loss
                )

                self.ppo_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.ppo_max_grad_norm
                )
                self.ppo_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.training_step_count += num_updates
        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_loss": (
                (total_policy_loss + total_value_loss)
                / max(num_updates, 1)
            ),
        }

    # ------------------------------------------------------------------
    # Auxiliary training
    # ------------------------------------------------------------------

    def push_rollout_to_aux_buffer(
        self, rollout: PPORolloutBuffer
    ) -> None:
        """Extract auxiliary transitions from a PPO rollout and push them
        into the replay buffer for V_h^e / X_h / U_r training."""
        for entry in rollout.entries:
            info = entry.info
            state = info.get("state")
            next_state = info.get("next_state")
            goals = info.get("goals", {})
            goal_weights = info.get("goal_weights", {})
            human_actions = info.get("human_actions", [])
            transition_probs = info.get("transition_probs")

            if state is None or next_state is None:
                continue

            self.aux_replay_buffer.push(
                state=state,
                robot_action=(entry.action,),
                goals=goals,
                goal_weights=goal_weights,
                human_actions=human_actions,
                next_state=next_state,
                transition_probs_by_action=transition_probs,
                terminal=entry.done,
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
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env_factory: Callable[[], Any],
        num_iterations: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the full PPO Phase 2 training loop.

        Parameters
        ----------
        env_factory : callable
            A zero-argument callable that returns a new
            :class:`EMPOMultiGridEnv` instance.
        num_iterations : int or None
            Override ``config.num_ppo_iterations``.

        Returns
        -------
        metrics : list[dict]
            Per-iteration loss / metric dictionaries.
        """
        cfg = self.config
        n_iters = num_iterations or cfg.num_ppo_iterations
        env = env_factory()
        all_metrics: List[Dict[str, float]] = []

        # Initial freeze of auxiliary networks
        self.freeze_auxiliary_networks()

        for iteration in range(n_iters):
            self.ppo_iteration = iteration

            # --- Step 1: collect rollout ---
            self.rollout_buffer.clear()
            obs, info = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            for _t in range(cfg.ppo_rollout_length):
                with torch.no_grad():
                    action, log_prob, _ent, value = (
                        self.actor_critic.get_action_and_value(
                            obs_t.unsqueeze(0)
                        )
                    )
                action_int = action.item()
                lp = log_prob.item()
                val = value.item()

                next_obs, reward, terminated, truncated, step_info = env.step(
                    action_int
                )
                done = terminated or truncated

                self.rollout_buffer.add(
                    PPORolloutEntry(
                        obs=obs,
                        action=action_int,
                        log_prob=lp,
                        value=val,
                        reward=reward,
                        done=done,
                        info=step_info,
                    )
                )

                if done:
                    obs, info = env.reset()
                else:
                    obs = next_obs

                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                )

            # Bootstrap value for last observation
            with torch.no_grad():
                last_val = self.actor_critic.get_value(
                    obs_t.unsqueeze(0)
                ).item()

            # --- Step 2: compute advantages ---
            rewards = [e.reward for e in self.rollout_buffer.entries]
            values = [e.value for e in self.rollout_buffer.entries]
            dones = [e.done for e in self.rollout_buffer.entries]
            advantages, returns = self.compute_gae(
                rewards, values, dones, last_val
            )

            # --- Step 3: PPO update ---
            obs_batch = torch.as_tensor(
                np.stack([e.obs for e in self.rollout_buffer.entries]),
                dtype=torch.float32,
                device=self.device,
            )
            actions_batch = torch.as_tensor(
                [e.action for e in self.rollout_buffer.entries],
                dtype=torch.long,
                device=self.device,
            )
            old_lps = torch.as_tensor(
                [e.log_prob for e in self.rollout_buffer.entries],
                dtype=torch.float32,
                device=self.device,
            )
            adv_t = torch.as_tensor(
                advantages, dtype=torch.float32, device=self.device
            )
            ret_t = torch.as_tensor(
                returns, dtype=torch.float32, device=self.device
            )

            ppo_losses = self.ppo_update(
                obs_batch, actions_batch, old_lps, adv_t, ret_t
            )

            # --- Step 4: auxiliary training ---
            self.push_rollout_to_aux_buffer(self.rollout_buffer)
            # Use the env's world_model for auxiliary forward passes
            wm = getattr(env, "world_model", None)
            aux_losses: Dict[str, float] = {}
            for _ in range(cfg.aux_training_steps_per_iteration):
                step_losses = self.train_auxiliary_step(world_model=wm)
                for k, v in step_losses.items():
                    aux_losses[k] = aux_losses.get(k, 0.0) + v

            # --- Step 5: re-freeze auxiliary networks ---
            if (iteration + 1) % cfg.reward_freeze_interval == 0:
                self.freeze_auxiliary_networks()

            metrics = {**ppo_losses, **aux_losses, "iteration": iteration}
            all_metrics.append(metrics)

            if iteration % 100 == 0:
                logger.info(
                    "PPO iter %d: policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                    iteration,
                    ppo_losses.get("policy_loss", 0),
                    ppo_losses.get("value_loss", 0),
                    ppo_losses.get("entropy", 0),
                )

        return all_metrics

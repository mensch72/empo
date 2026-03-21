"""
PufferLib-based Phase 1 trainer for EMPO.

This module orchestrates Phase 1 training using **PufferLib's PuffeRL** class
for the core PPO loop.  The goal-conditioned policy π_h(a|s,g) is approximated
by a :class:`GoalConditionedActorCritic` that receives state + goal features
as its observation.

Phase 1 is simpler than Phase 2:

- No auxiliary networks or warm-up stages.
- The reward is binary: ``goal.is_achieved(state)`` (0 or 1).
- Training amounts to standard goal-conditioned PPO.

Key PufferLib integration points:

- ``pufferlib.vector.make()`` creates vectorised environments.
- ``pufferlib.pufferl.PuffeRL`` drives ``evaluate()`` + ``train()``.
- ``GoalConditionedActorCritic.forward(observations, state)`` → ``(logits, value)``
  follows PufferLib's expected policy interface.

This module does NOT import or modify the existing DQN-path Phase 1 trainer.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import pufferlib
    import pufferlib.emulation
    import pufferlib.pufferl
    import pufferlib.vector

    HAS_PUFFERLIB = True
except ImportError:  # pragma: no cover

    class _PufferLibShim:
        """Minimal shim so the module can be imported without pufferlib."""

        class pufferl:
            class PuffeRL:
                pass

        class vector:
            @staticmethod
            def make(*_a, **_kw):
                raise ImportError("pufferlib is required for PPOPhase1Trainer.train()")

        class emulation:
            class GymnasiumPufferEnv:
                pass

    pufferlib = _PufferLibShim()  # type: ignore[assignment]
    HAS_PUFFERLIB = False


try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    HAS_TENSORBOARD = False

from .config import PPOPhase1Config
from .actor_critic import GoalConditionedActorCritic


class PPOPhase1Trainer:
    """PufferLib-backed PPO Phase 1 trainer.

    This trainer uses ``pufferlib.pufferl.PuffeRL`` for the core PPO loop
    to approximate a goal-conditioned human policy prior.

    Parameters
    ----------
    actor_critic : GoalConditionedActorCritic
        The PPO actor-critic network (follows PufferLib policy convention).
    config : PPOPhase1Config
        Full PPO Phase 1 configuration.
    device : str
        Torch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(
        self,
        actor_critic: GoalConditionedActorCritic,
        config: PPOPhase1Config,
        device: str = "cpu",
    ):
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = torch.device(device)

        # Counters
        self.global_env_step: int = 0
        self.ppo_iteration: int = 0

        # TensorBoard
        self._tb_writer: Optional[Any] = None
        if config.tensorboard_dir and HAS_TENSORBOARD:
            self._init_tensorboard()

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------

    def _init_tensorboard(self) -> None:
        """Create the TensorBoard SummaryWriter."""
        tb_dir = self.config.tensorboard_dir
        if tb_dir:
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """Persist Phase 1 network state and counters to *path*.

        Saves the actor-critic weights and EMPO-specific training counters.
        This allows ``load_checkpoint`` to restore these components for
        evaluation or further fine-tuning, but does not include PuffeRL/PPO
        driver optimiser state, rollout buffers, or driver-specific training
        counters.  PPO training will therefore restart from a fresh driver
        state after loading.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "actor_critic": self.actor_critic.state_dict(),
            "global_env_step": self.global_env_step,
            "ppo_iteration": self.ppo_iteration,
            "gamma_h": self.config.gamma_h,
            "beta_h": self.config.beta_h,
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Restore Phase 1 network state and counters from *path*.

        .. warning::

           ``torch.load`` with ``weights_only=False`` is used here so that
           optimiser state dicts (which contain non-tensor objects) can be
           restored.  Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.global_env_step = checkpoint.get("global_env_step", 0)
        self.ppo_iteration = checkpoint.get("ppo_iteration", 0)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env_creator: Callable[[], Any],
        num_iterations: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the PufferLib PPO Phase 1 training loop.

        Parameters
        ----------
        env_creator : callable
            A zero-argument callable that returns a new Gymnasium-compatible
            :class:`Phase1PPOEnv` instance.  The trainer wraps each instance
            with ``pufferlib.emulation.GymnasiumPufferEnv`` and passes the
            wrapped creator to ``pufferlib.vector.make()`` for vectorised
            stepping.
        num_iterations : int or None
            Override the number of PPO iterations from config.

        Returns
        -------
        metrics : list[dict[str, float]]
            Per-iteration metrics (policy_loss, value_loss, entropy, etc.).
        """
        cfg = self.config
        n_iters = num_iterations if num_iterations is not None else cfg.num_ppo_iterations

        # Build PufferLib config
        puffer_config = cfg.to_pufferlib_config()
        puffer_config["device"] = str(self.device)
        puffer_config["seed"] = cfg.seed
        total_timesteps = n_iters * puffer_config["batch_size"]
        puffer_config["total_timesteps"] = total_timesteps

        # Wrap the Gymnasium env creator with PufferLib's emulation layer.
        def puffer_env_creator(buf=None, seed=0):
            def _create():
                return env_creator()

            return pufferlib.emulation.GymnasiumPufferEnv(
                env_creator=_create, buf=buf, seed=seed
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

        all_metrics: List[Dict[str, float]] = []
        iteration = 0

        while pufferl.global_step < total_timesteps and iteration < n_iters:
            self.ppo_iteration = iteration

            # PufferLib: collect rollout + PPO update
            pufferl.evaluate()
            pufferl.train()

            # Track env steps
            self.global_env_step = pufferl.global_step

            # Collect metrics
            metrics: Dict[str, float] = {
                "iteration": iteration,
                "global_env_step": self.global_env_step,
            }
            all_metrics.append(metrics)

            # Logging
            if cfg.log_interval > 0 and iteration % cfg.log_interval == 0:
                self._log_scalar("phase1/global_env_step", self.global_env_step, iteration)

            # Checkpointing
            if (
                cfg.checkpoint_interval > 0
                and cfg.checkpoint_dir
                and iteration > 0
                and iteration % cfg.checkpoint_interval == 0
            ):
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"phase1_ppo_iter_{iteration}.pt"
                )
                self.save_checkpoint(ckpt_path)

            iteration += 1

        # Cleanup
        pufferl.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

        return all_metrics

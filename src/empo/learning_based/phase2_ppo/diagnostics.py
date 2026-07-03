"""Plasticity-loss diagnostics for the PPO-based Phase 2 actor-critic.

Thin wrapper around :mod:`empo.learning_based.plasticity_diagnostics`,
which holds the shared, trainer-agnostic implementation (dormant/dead
neurons, effective rank of the shared representation, weight-norm growth).
See that module for the theory background and references.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from empo.learning_based.plasticity_diagnostics import measure_plasticity

__all__ = ["compute_plasticity_diagnostics"]


def compute_plasticity_diagnostics(
    actor_critic: nn.Module,
    obs_batch: torch.Tensor,
    *,
    dormant_tau: float = 0.025,
) -> Dict[str, float]:
    """Compute plasticity-loss diagnostics for the Phase 2 actor-critic.

    Parameters
    ----------
    actor_critic : nn.Module
        The Phase 2 policy network (e.g. :class:`EMPOActorCritic`).  Only its
        ReLU activations and parameters are inspected; the module is left
        unchanged (train/eval mode is preserved).
    obs_batch : Tensor, shape (B, *obs_shape)
        A batch of observations to probe the network with.  Should be
        representative of the current rollout distribution.
    dormant_tau : float
        Normalised-activation threshold for counting a neuron as dormant
        (Sokar et al., 2023).  ``0.0`` counts only strictly dead neurons.

    Returns
    -------
    dict[str, float]
        Flat metric dictionary.  Keys include per-layer and overall
        ``dormant_frac/*`` and ``dead_frac/*``, ``effective_rank/{srank,erank}``
        for the shared representation, and ``weight_norm/*``.
    """
    # ``actor_critic.encoder`` is the shared trunk whose (post-ReLU) output
    # feeds both the actor and critic heads: the natural effective-rank probe.
    feature_module = getattr(actor_critic, "encoder", None)
    return measure_plasticity(
        actor_critic,
        lambda: actor_critic(obs_batch),
        feature_module=feature_module,
        dormant_tau=dormant_tau,
    )

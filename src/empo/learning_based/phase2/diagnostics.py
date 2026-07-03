"""Plasticity-loss diagnostics for the DQN-based Phase 2 robot networks.

Thin wrapper around :mod:`empo.learning_based.plasticity_diagnostics`,
which holds the shared, trainer-agnostic implementation (dormant/dead
neurons, effective rank of the shared representation, weight-norm growth).
See that module for the theory background and references.

The DQN-path robot networks (``q_r``, ``v_r``) expose a
``forward_batch(states, world_model, device)`` method and own a
``state_encoder`` submodule whose output is the shared feature
representation used for the effective-rank measure.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from empo.learning_based.plasticity_diagnostics import measure_plasticity

__all__ = ["compute_robot_network_plasticity"]


def compute_robot_network_plasticity(
    net: nn.Module,
    states: List[Any],
    world_model: Any,
    device: str = "cpu",
    *,
    dormant_tau: float = 0.025,
) -> Dict[str, float]:
    """Compute plasticity-loss diagnostics for a Phase 2 robot network.

    Parameters
    ----------
    net : nn.Module
        A robot Q- or value-network with a ``forward_batch(states,
        world_model, device)`` method and a ``state_encoder`` submodule.
    states : list
        A batch of (hashable) world-model states to probe the network with.
        Should be representative of the current replay distribution.
    world_model : WorldModel
        World model used by ``forward_batch`` to tensorise states.
    device : str
        Device on which to run the forward pass.
    dormant_tau : float
        Normalised-activation threshold for counting a neuron as dormant
        (Sokar et al., 2023).  ``0.0`` counts only strictly dead neurons.

    Returns
    -------
    dict[str, float]
        Flat metric dictionary with ``dormant_frac/*``, ``dead_frac/*``,
        ``effective_rank/{srank,erank}`` and ``weight_norm/*`` keys.  Empty
        if ``net`` is not a probeable ``nn.Module`` (e.g. a lookup table).
    """
    if not isinstance(net, nn.Module) or not hasattr(net, "forward_batch"):
        return {}
    if not states:
        return {}
    # Skip lookup tables / identity encoders: without ReLU layers there is no
    # plasticity signal to measure.
    if not any(isinstance(m, nn.ReLU) for m in net.modules()):
        return {}

    # The state encoder's (post-ReLU) output is the shared representation
    # fed to the network head: the natural effective-rank probe point.
    feature_module = getattr(net, "state_encoder", None)
    return measure_plasticity(
        net,
        lambda: net.forward_batch(states, world_model, device),
        feature_module=feature_module,
        dormant_tau=dormant_tau,
    )

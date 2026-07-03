"""Shared plasticity-loss diagnostics for Phase 2 neural networks.

Plasticity loss is the gradual erosion of a network's ability to keep
learning as training progresses.  It is a real concern for EMPO Phase 2,
where the robot policy is *computed* over many training steps against a
slowly-shifting intrinsic reward (the mutual-dependency loop of equations
4-9), so the networks are trained on a highly non-stationary target.

This module provides light-weight, forward-pass-only measurements of the
three most widely-used plasticity indicators, agnostic to which Phase 2
trainer (DQN- or PPO-based) is in use:

* **Dormant / dead neurons** (Sokar et al., 2023, "The Dormant Neuron
  Phenomenon in Deep Reinforcement Learning").  A ReLU unit is
  ``tau``-dormant when its mean absolute activation, normalised by the
  layer mean, falls at or below ``tau``.  ``tau = 0`` recovers strictly
  *dead* units that never fire on the batch.

* **Effective rank of the representation** (Kumar et al., 2020, ``srank``;
  Roy & Vetterli, 2007, entropy-based ``erank``).  A collapsing feature
  representation loses rank, which correlates with loss of plasticity.

* **Weight-norm growth**.  Parameter norms tend to grow without bound as
  plasticity is lost; tracking them gives a cheap early-warning signal.

The generic entry point is :func:`measure_plasticity`, which registers
forward hooks on the ReLU layers of a module, runs a caller-supplied
forward pass, and returns a flat ``dict[str, float]`` suitable for scalar
logging.  It leaves the module unchanged (train/eval mode preserved) and
runs under ``torch.no_grad``, so it is safe to call inside a training loop.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["measure_plasticity", "effective_ranks", "weight_norms"]

_EPS = 1e-9


def _clean_name(name: str) -> str:
    """Turn a ``named_modules`` path into a compact, log-friendly key."""
    return name.replace(".", "_") if name else "root"


def effective_ranks(feats: torch.Tensor, srank_delta: float = 0.01) -> Dict[str, float]:
    """Effective-rank measures of a (batch, features) representation matrix.

    Parameters
    ----------
    feats : Tensor, shape (B, D)
        Feature/activation matrix.
    srank_delta : float
        Energy tolerance for ``srank``: the smallest ``k`` whose top-``k``
        singular values retain at least ``(1 - srank_delta)`` of the total
        singular-value mass (Kumar et al., 2020).

    Returns
    -------
    dict with keys ``srank`` and ``erank`` (empty if the SVD is not
    applicable / fails).
    """
    if feats.ndim != 2 or feats.shape[0] < 2 or feats.shape[1] < 1:
        return {}
    f = feats.float()
    # Centre features so the rank reflects representational spread, not the
    # (uninformative) common mean offset.
    f = f - f.mean(dim=0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(f)
    except Exception:  # pragma: no cover - numerical guard
        return {}
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return {"srank": 0.0, "erank": 0.0}

    total = sv.sum()
    # srank: smallest k with cumulative energy >= (1 - delta) * total.
    csum = torch.cumsum(sv, dim=0)
    srank = int((csum < (1.0 - srank_delta) * total).sum().item()) + 1

    # erank: exp(entropy) of the normalised singular-value distribution.
    p = sv / total
    entropy = -(p * (p + 1e-12).log()).sum()
    erank = float(torch.exp(entropy).item())
    return {"srank": float(srank), "erank": erank}


def weight_norms(module: nn.Module) -> Dict[str, float]:
    """Total and per-top-level-submodule L2 parameter norms."""
    total_sq = 0.0
    per_module: Dict[str, float] = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        n2 = float(p.data.norm(2).item())
        sq = n2 * n2
        total_sq += sq
        top = name.split(".", 1)[0]
        per_module[top] = per_module.get(top, 0.0) + sq
    metrics = {"weight_norm/total": total_sq ** 0.5}
    for k, v in per_module.items():
        metrics[f"weight_norm/{k}"] = v ** 0.5
    return metrics


def _dormant_dead_fractions(
    activations: Dict[str, torch.Tensor], dormant_tau: float
) -> Dict[str, float]:
    """Per-layer and overall dormant / dead ReLU-neuron fractions."""
    metrics: Dict[str, float] = {}
    total_neurons = 0
    total_dormant = 0
    total_dead = 0
    for name, act in activations.items():
        a = act.reshape(act.shape[0], -1)
        score = a.abs().mean(dim=0)  # mean absolute activation per unit
        denom = score.mean() + _EPS
        normalised = score / denom
        n = int(normalised.numel())
        dormant = int((normalised <= dormant_tau).sum().item())
        dead = int((score <= _EPS).sum().item())
        key = _clean_name(name)
        metrics[f"dormant_frac/{key}"] = dormant / n
        metrics[f"dead_frac/{key}"] = dead / n
        total_neurons += n
        total_dormant += dormant
        total_dead += dead
    if total_neurons > 0:
        metrics["dormant_frac/overall"] = total_dormant / total_neurons
        metrics["dead_frac/overall"] = total_dead / total_neurons
    return metrics


def measure_plasticity(
    module: nn.Module,
    forward_fn: Callable[[], object],
    *,
    feature_module: Optional[nn.Module] = None,
    dormant_tau: float = 0.025,
) -> Dict[str, float]:
    """Measure plasticity-loss indicators for an ``nn.Module``.

    Parameters
    ----------
    module : nn.Module
        The network to inspect.  Its ReLU activations and parameters are
        probed; train/eval mode is preserved.
    forward_fn : callable
        A zero-argument callable that runs a forward pass through ``module``
        on a representative batch (e.g. ``lambda: net.forward_batch(states,
        wm, device)``).  Its return value is ignored; the ReLU hooks capture
        what is needed.
    feature_module : nn.Module or None
        Submodule whose output is treated as the shared representation for
        the effective-rank measure (e.g. the state encoder).  When ``None``
        the effective rank falls back to the first captured ReLU activation.
    dormant_tau : float
        Normalised-activation threshold for counting a neuron as dormant
        (Sokar et al., 2023).  ``0.0`` counts only strictly dead neurons.

    Returns
    -------
    dict[str, float]
        Flat metric dictionary with ``dormant_frac/*``, ``dead_frac/*``,
        ``effective_rank/{srank,erank}`` and ``weight_norm/*`` keys.
    """
    activations: Dict[str, torch.Tensor] = {}
    feat_holder: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _capture(store: Dict[str, torch.Tensor], key: str):
        def _hook(_module: nn.Module, _inp, out) -> None:
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(tensor, torch.Tensor):
                store[key] = tensor.detach()

        return _hook

    for name, sub in module.named_modules():
        if isinstance(sub, nn.ReLU):
            hooks.append(sub.register_forward_hook(_capture(activations, name)))
    if feature_module is not None:
        hooks.append(
            feature_module.register_forward_hook(_capture(feat_holder, "features"))
        )

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            module.train()

    metrics = _dormant_dead_fractions(activations, dormant_tau)

    feats = feat_holder.get("features")
    if feats is None and activations:
        feats = next(iter(activations.values()))
    if feats is not None:
        for k, v in effective_ranks(feats.reshape(feats.shape[0], -1)).items():
            metrics[f"effective_rank/{k}"] = v

    metrics.update(weight_norms(module))
    return metrics

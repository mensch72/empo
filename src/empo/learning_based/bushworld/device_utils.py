"""Small shared utilities for BushWorld learning-based modules."""

from __future__ import annotations

import torch
from torch import nn


def module_device(module: nn.Module) -> torch.device:
    """Return the device of ``module``'s first parameter or buffer.

    Falls back to CPU when the module has neither parameters nor buffers
    (e.g. an encoder running in identity / ``use_encoders=False`` mode).
    """
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")

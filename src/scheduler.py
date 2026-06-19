from __future__ import annotations

import math
import torch

def linear_schedule(tao: torch.Tensor) -> torch.Tensor:
    return tao

def cosine_schedule(tao: torch.Tensor) -> torch.Tensor:
    """
    Cosine mask scheduling.

    Args:
        tao: Tensor with values in (0, 1]. Shape: arbitrary.

    Returns:
        Gamma tensor with values in [0, 1). Computed as:
            gamma = cos((pi / 2) * tao)
    """
    # Use a tensor-typed scale so dtype/device follow `tao`.
    scale = tao.new_tensor(math.pi / 2.0)
    return torch.cos(scale * tao)
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The width-sweep MLP and the two loss/activation registers.

A single fully-connected architecture ``in_dim -> (hidden -> act)^depth -> classes``
whose *width* is the double-descent knob. The **register** fixes the activation and
the loss:

* ``ce_relu``  -- ReLU activation, cross-entropy loss (paper-faithful).
* ``mse_tanh`` -- tanh activation, one-hot mean-squared-error loss. tanh is a
  Riccati activation, so its closed-form ``sigma', sigma'', sigma'''`` power the
  exact curvature tower, and the MSE residual ``f(x) - onehot(y)`` is what the
  Gauss-Newton optimizer family and the certified enclosures consume.

The model itself only ever emits ``(N, num_classes)`` real outputs; whether those
are logits (CE) or a regression target (MSE) is the caller's choice, so the same
module serves both registers.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

Register = Literal["ce_relu", "mse_tanh"]
REGISTERS: tuple[Register, ...] = ("ce_relu", "mse_tanh")

_ACTIVATIONS: dict[Register, type[nn.Module]] = {
    "ce_relu": nn.ReLU,
    "mse_tanh": nn.Tanh,
}


def register_activation(register: Register) -> str:
    """Activation name (``'relu'`` / ``'tanh'``) implied by ``register``."""
    if register not in _ACTIVATIONS:
        raise ValueError(f"unknown register {register!r}; choose from {REGISTERS}")
    return "relu" if register == "ce_relu" else "tanh"


class MLP1D(nn.Module):
    """Fully-connected MNIST-1D classifier; ``hidden`` is the double-descent knob."""

    def __init__(
        self,
        *,
        in_dim: int = 40,
        hidden: int = 100,
        num_classes: int = 10,
        depth: int = 1,
        register: Register = "ce_relu",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        act_cls = _ACTIVATIONS[register]
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act_cls()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_classes = num_classes
        self.depth = depth
        self.register = register

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_model(
    register: Register,
    *,
    in_dim: int = 40,
    hidden: int = 100,
    num_classes: int = 10,
    depth: int = 1,
    seed: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> MLP1D:
    """Build an :class:`MLP1D` for ``register`` (optionally seeded / on ``device``)."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MLP1D(
        in_dim=in_dim, hidden=hidden, num_classes=num_classes, depth=depth, register=register
    )
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters ``P`` (the curvature dimension)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


__all__ = [
    "MLP1D",
    "REGISTERS",
    "Register",
    "build_model",
    "count_parameters",
    "register_activation",
]

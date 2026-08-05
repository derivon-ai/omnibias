# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable-temperature activation modules.

Thin ``nn.Module`` wrappers that hold a differentiable temperature ``beta`` (or
negative slope ``alpha``) and call the closed-form tempered / piecewise tower
each forward -- mirroring the learnable-``beta`` pattern in
``omnibias-binary``'s ``binarize``. The whole derivative tower stays available
through ``fastpath(z, n)``, so these compose with the operator layers.

* :class:`TemperedActivation` -- ``softplus(beta z)/beta -> relu``,
  ``sigmoid(beta z) -> step``, ``tanh(beta z) -> sign`` (pick the base + scale);
  ``beta`` is learnable when ``learnable_beta=True``.
* :class:`LearnablePReLU` -- leaky ReLU with a learnable negative slope
  ``alpha`` on the almost-everywhere tower.

``beta`` (and ``alpha``) should stay positive; wrap with your own
reparameterisation (``softplus`` / ``exp``) if you need to guarantee it.
"""

from __future__ import annotations

from omnibias.core.spec import make_tempered_fastpath
from omnibias.torch.activations.registry import (
    ActivationSpec,
    NthDerivativeFn,
    get_activation,
)

import torch
import torch.nn as nn
from torch import Tensor


class TemperedActivation(nn.Module):
    """Beta-tempered smooth surrogate of a base activation's tower.

    Parameters
    ----------
    base : str or :class:`ActivationSpec`, default ``"softplus"``
        Base activation carrying a fastpath (e.g. ``"softplus"``, ``"sigmoid"``,
        ``"tanh"``). The surrogate tower is
        ``beta**(n - p) * base.fastpath(beta * z, n)``.
    beta : float, default 1.0
        Initial temperature (larger = sharper; ``beta -> inf`` approaches the
        hard activation).
    scale : {"one_over_beta", "unit"}, default ``"one_over_beta"``
        ``"one_over_beta"`` (``p = 1``) for ``softplus(beta z)/beta -> relu``;
        ``"unit"`` (``p = 0``) for bounded surrogates like ``sigmoid`` / ``tanh``.
    learnable_beta : bool, default False
        If True, ``beta`` is an ``nn.Parameter`` (differentiable temperature);
        otherwise it is a frozen buffer.
    """

    beta: Tensor
    _base_fastpath: NthDerivativeFn

    def __init__(
        self,
        base: str | ActivationSpec[Tensor] = "softplus",
        beta: float = 1.0,
        *,
        scale: str = "one_over_beta",
        learnable_beta: bool = False,
    ) -> None:
        super().__init__()
        spec = get_activation(base)
        if spec.fastpath is None:
            raise ValueError(
                f"TemperedActivation requires a base with a fastpath; {spec.name!r} has none."
            )
        if scale == "one_over_beta":
            self._scale_power = 1
        elif scale == "unit":
            self._scale_power = 0
        else:
            raise ValueError(f"scale must be 'unit' or 'one_over_beta', got {scale!r}")
        self.base_name = spec.name
        self.scale = scale
        self._base_fastpath = spec.fastpath
        beta_init = torch.as_tensor(float(beta))
        if learnable_beta:
            self.beta = nn.Parameter(beta_init)
        else:
            self.register_buffer("beta", beta_init)

    def fastpath(self, z: Tensor, n: int) -> Tensor:
        """Closed-form ``n``-th derivative of the tempered surrogate at current ``beta``."""
        kernel = make_tempered_fastpath(
            self._base_fastpath, self.beta, scale_power=self._scale_power
        )
        return kernel(z, n)

    def forward(self, z: Tensor) -> Tensor:
        return self.fastpath(z, 0)

    def extra_repr(self) -> str:
        learnable = isinstance(self.beta, nn.Parameter)
        return f"base={self.base_name!r}, scale={self.scale!r}, learnable_beta={learnable}"


class LearnablePReLU(nn.Module):
    """Leaky ReLU with a (learnable) negative slope ``alpha`` on the a.e. tower.

    ``forward(z) = where(z > 0, z, alpha * z)``; ``fastpath`` gives the
    almost-everywhere tower (``n = 1`` step-with-slope, ``n >= 2 -> 0``).
    """

    alpha: Tensor

    def __init__(self, negative_slope: float = 0.25, *, learnable: bool = True) -> None:
        super().__init__()
        alpha_init = torch.as_tensor(float(negative_slope))
        if learnable:
            self.alpha = nn.Parameter(alpha_init)
        else:
            self.register_buffer("alpha", alpha_init)

    def fastpath(self, z: Tensor, n: int) -> Tensor:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return torch.where(z > 0, z, self.alpha * z)
        if n == 1:
            return torch.where(z > 0, torch.ones_like(z), self.alpha * torch.ones_like(z))
        return torch.zeros_like(z)

    def forward(self, z: Tensor) -> Tensor:
        return self.fastpath(z, 0)

    def extra_repr(self) -> str:
        return f"learnable={isinstance(self.alpha, nn.Parameter)}"


__all__ = ["LearnablePReLU", "TemperedActivation"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal transverse filter (torch; theory 05-02 G5).

A designed FIR whose taps are one closed-form ``sigma^(n)`` pack. Default
``order=0`` is the logistic tail (integral / survival role). Inference is
``O(T K)`` with no recurrence. Founding tower taps, not temperature
collapse; not :mod:`omnibias.struct`.
"""

from __future__ import annotations

import math

import omnibias.torch.activations  # noqa: F401 — register fastpaths
from omnibias.core.sequence import leaky_integrator_init
from omnibias.torch.activations.registry import get_activation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_ALPHA_EPS = 1e-3


def _log_alpha_from_timescale(alpha: float) -> float:
    """Invert ``softplus(log_alpha) + eps`` so init lands on ``alpha``."""
    target = max(float(alpha) - _ALPHA_EPS, 1e-6)
    return float(math.log(math.expm1(target)))


class CausalTransverseFilter(nn.Module):
    """Causal FIR ``y = (c * sigma^(n)(...)) * x + bias``.

    Parameters
    ----------
    order
        Pack order. ``0`` is the leaky-integrator tail; ``>= 1`` is a
        mid-lag band-pass bump (01-07 band selector).
    alpha
        Initial positive timescale.
    width
        FIR support. For a length-``T`` window, ``width=T`` is the
        zero-truncation horizon (still an FIR, not an SSM).
    """

    def __init__(
        self,
        *,
        order: int,
        alpha: float,
        width: int,
        coeff: float = 1.0,
        tau: float = 0.0,
        bias: float = 0.0,
        base: str = "sigmoid",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if int(order) < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if int(width) < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        spec = get_activation(base)
        if spec.fastpath is None:
            raise RuntimeError(f"{base!r} fastpath is required")
        resolved = torch.get_default_dtype() if dtype is None else dtype
        self.order = int(order)
        self.width = int(width)
        self.base = str(base)
        self.coeff = nn.Parameter(torch.tensor(float(coeff), dtype=resolved))
        self.log_alpha = nn.Parameter(
            torch.tensor(_log_alpha_from_timescale(float(alpha)), dtype=resolved)
        )
        self.tau = nn.Parameter(torch.tensor(float(tau), dtype=resolved))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=resolved))

    @classmethod
    def from_leaky_integrator(
        cls,
        rho: float,
        *,
        width: int,
        dtype: torch.dtype | None = None,
    ) -> CausalTransverseFilter:
        """Four-parameter init near ``y_t = rho y_{t-1} + (1-rho) x_t``."""
        coeff, alpha, tau = leaky_integrator_init(rho)
        return cls(
            order=0,
            alpha=alpha,
            width=width,
            coeff=coeff,
            tau=tau,
            bias=0.0,
            dtype=dtype,
        )

    def timescale(self) -> Tensor:
        return F.softplus(self.log_alpha) + _ALPHA_EPS

    def taps(self) -> Tensor:
        spec = get_activation(self.base)
        if spec.fastpath is None:
            raise RuntimeError(f"{self.base!r} fastpath is required")
        lags = torch.arange(self.width, dtype=self.coeff.dtype, device=self.coeff.device)
        alpha = self.timescale()
        if self.order == 0:
            argument = self.tau - alpha * lags
        else:
            argument = alpha * (lags - self.tau)
        return self.coeff * spec.fastpath(argument, self.order)

    def forward(self, x: Tensor) -> Tensor:
        xx = x if x.ndim == 2 else x.unsqueeze(0)
        if xx.ndim != 2:
            raise ValueError(f"x must be (batch, time) or (time,), got {tuple(x.shape)}")
        kern = self.taps()
        pred = F.conv1d(
            xx.unsqueeze(1),
            kern.flip(0).view(1, 1, -1),
            padding=self.width - 1,
        ).squeeze(1)[:, : xx.shape[1]]
        pred = pred + self.bias
        return pred if x.ndim == 2 else pred.squeeze(0)

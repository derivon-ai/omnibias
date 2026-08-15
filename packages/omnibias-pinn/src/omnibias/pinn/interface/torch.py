# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-interface transmission field (torch; theory 02-05)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from omnibias.pinn.interface._core import Interface
from torch import Tensor

_LOG2 = math.log(2.0)


def _log_cosh(z: Tensor) -> Tensor:
    az = z.abs()
    return az + torch.log1p(torch.exp(-2.0 * az)) - _LOG2


def _int_log_cosh(z: Tensor, terms: int = 24) -> Tensor:
    sign = torch.sign(z)
    az = z.abs()
    acc = 0.5 * az * az - _LOG2 * az
    for k in range(1, terms + 1):
        sk = 1.0 if k % 2 == 1 else -1.0
        acc = acc + sk * (1.0 - torch.exp(-2.0 * k * az)) / (2.0 * k * k)
    return sign * acc


def profile_tensor(order: int, z: Tensor, alpha: float) -> Tensor:
    az = float(alpha) * z
    n = int(order)
    if n == 0:
        return torch.tanh(az)
    if n == 1:
        return _log_cosh(az) / float(alpha)
    if n == 2:
        return _int_log_cosh(az) / (float(alpha) ** 2)
    raise ValueError(f"unsupported profile order {order}")


class MultiInterfaceField(nn.Module):
    """``u = base + sum_g c_g profile_{n_g}(alpha_g (w·x + mu_g))``.

    ``hard=True`` sets ``c_g = jump_g / 2`` (tanh-family sharp jump is 2).
    ``interface_residuals`` is always on. Parallel interfaces only.
    ``alpha -> inf`` is sharpening, not a collapse.
    """

    def __init__(
        self,
        base_field: Callable[[Tensor], Tensor] | nn.Module,
        interfaces: Sequence[Interface],
        *,
        hard: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not interfaces:
            raise ValueError("at least one interface is required")
        self.base_field = base_field
        self.interfaces = tuple(interfaces)
        self.hard = bool(hard)
        dt = torch.get_default_dtype() if dtype is None else dtype
        jumps = torch.tensor([float(i.jump) for i in self.interfaces], dtype=dt)
        if hard:
            self.register_buffer("coeffs", 0.5 * jumps)
        else:
            self.coeffs = nn.Parameter(0.5 * jumps.clone())

    def _transverse(self, x: Tensor, iface: Interface) -> Tensor:
        w = torch.as_tensor(iface.normal, dtype=x.dtype, device=x.device)
        return (x * w).sum(dim=-1) + float(iface.offset)

    def forward(self, x: Tensor) -> Tensor:
        u = self.base_field(x)
        for g, iface in enumerate(self.interfaces):
            z = self._transverse(x, iface)
            n = int(iface.order) if iface.order is not None else 1
            bump = profile_tensor(n, z, iface.alpha)
            while bump.ndim < u.ndim:
                bump = bump.unsqueeze(-1)
            u = u + self.coeffs[g] * bump
        return u

    def interface_residuals(self, x: Tensor) -> dict[int, Tensor]:
        """Far-field jump of the ``n``-th transverse derivative minus ``jump``.

        Every tanh-family profile has derivative jump ``2 c`` (the jump of
        ``tanh``), independent of order.
        """
        out: dict[int, Tensor] = {}
        eight = x.new_tensor(8.0)
        for g, iface in enumerate(self.interfaces):
            left = torch.tanh(-eight)
            right = torch.tanh(eight)
            out[g] = self.coeffs[g] * (right - left) - float(iface.jump)
        return out


__all__ = ["MultiInterfaceField", "profile_tensor"]

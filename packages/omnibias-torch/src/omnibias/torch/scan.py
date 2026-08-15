# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias scan: shared template on a bank of offsets (torch; theory 01-02).

``Scan[T](z)_j = F_T(z + tau_j)``. Parameters of ``T`` are shared; only the
offset varies. Equivariance is along the transverse coordinate and exact on
the bank lattice as an *interior* shift (``tanh'`` is not periodic, so the
response is not a circular wrap).

The template collapse is ``delta -> 0``. Soft-argmax ``gamma`` is a softmax
sharpness; ``gamma -> inf`` would be temperature collapse, a different limit.
"""

from __future__ import annotations

from typing import Literal

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.core.scan import BankSpec
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.multipack import multipack_response

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

OpName = Literal["identity", "grad", "laplacian", "derivative", "band", "integral"]
Readout = Literal["response", "pooled", "argmax"]

_COLLAPSE_ORDER: dict[str, int] = {
    "identity": 0,
    "grad": 1,
    "laplacian": 2,
}


def template_from_op(
    op: OpName | str,
    *,
    derivative_order: int = 1,
    gap: float = 1.0,
) -> MultiPackSpec | str:
    """Map an operator role to a :class:`MultiPackSpec`, or keep window ops as tags."""
    name = str(op).lower()
    if name in _COLLAPSE_ORDER:
        return MultiPackSpec((PackSpec(_COLLAPSE_ORDER[name], 0.0),))
    if name == "derivative":
        if int(derivative_order) < 0:
            raise ValueError("derivative_order must be >= 0")
        return MultiPackSpec((PackSpec(int(derivative_order), 0.0),))
    if name in ("band", "integral"):
        if gap <= 0.0:
            raise ValueError("gap must be positive")
        return name
    raise ValueError(f"unknown template op {op!r}")


def scan_response(
    z: Tensor,
    offsets: Tensor,
    scales: Tensor,
    spec: MultiPackSpec | str,
    base: ActivationSpec[Tensor],
    *,
    gap: float = 1.0,
) -> Tensor:
    """Evaluate the template at ``scale * (z + offset)``.

    ``z`` is ``(..., C)``, ``offsets`` ``(M,)``, ``scales`` ``(S,)``.
    Returns ``(..., C, M)`` when ``S == 1``, else ``(..., C, M, S)``.
    """
    off = offsets.reshape(-1)
    sc = scales.reshape(-1)
    bank = z.unsqueeze(-1) + off
    if sc.numel() == 1:
        u = sc.reshape(()) * bank
    else:
        u = bank.unsqueeze(-1) * sc
    if isinstance(spec, str):
        half = 0.5 * float(gap)
        if spec == "band":
            if base.forward is None:
                raise NotImplementedError("band template needs ActivationSpec.forward")
            out = base.forward(u + half) - base.forward(u - half)
        elif spec == "integral":
            if base.integral is None:
                raise NotImplementedError(
                    f"Activation {base.name!r} has no closed-form integral kernel"
                )
            out = base.integral(u + half) - base.integral(u - half)
        else:
            raise ValueError(f"unknown window template {spec!r}")
    else:
        orders = tuple(int(p.order) for p in spec.packs)
        means = torch.tensor([p.mean for p in spec.packs], dtype=u.dtype, device=u.device)
        weights = torch.tensor([p.weight for p in spec.packs], dtype=u.dtype, device=u.device)
        out = multipack_response(u, means, weights, orders, base)
    return out


def soft_argmax_offset(response: Tensor, offsets: Tensor, *, gamma: float = 8.0) -> Tensor:
    """``tau* = sum_j tau_j softmax(gamma R)_j`` over the offset axis.

    ``response`` is ``(..., M)`` or ``(..., M, S)`` (offset axis is -1 or -2).
    Softmax is over offsets. ``gamma -> inf`` would be a hard argmax
    (temperature collapse), not bias collapse.
    """
    off = offsets.reshape(-1).to(dtype=response.dtype, device=response.device)
    if response.shape[-1] == off.numel():
        logits = float(gamma) * response
        w = F.softmax(logits, dim=-1)
        return (w * off).sum(dim=-1)
    if response.ndim >= 2 and response.shape[-2] == off.numel():
        logits = float(gamma) * response
        w = F.softmax(logits, dim=-2)
        return (w * off.reshape(-1, 1)).sum(dim=-2)
    raise ValueError("response trailing shape does not match offsets")


def min_offset_separation(offsets: Tensor) -> Tensor:
    """Minimum adjacent spacing of ``offsets`` after sorting."""
    ordered, _ = torch.sort(offsets.reshape(-1))
    if ordered.numel() < 2:
        return offsets.new_tensor(float("inf"))
    return (ordered[1:] - ordered[:-1]).min()


class BiasScan(nn.Module):
    """Channel-wise bias scan.

    Parameters
    ----------
    num_channels:
        Trailing channel count of ``z``.
    bank:
        Offset / scale bank.
    template:
        :class:`~omnibias.core.multipack.MultiPackSpec` or an operator role name.
    readout:
        ``response`` -> ``(..., C, M)``; ``pooled`` / ``argmax`` -> ``(..., C)``.
    gamma:
        Soft-argmax sharpness (not a collapse parameter).
    """

    offsets: Tensor
    scales: Tensor
    pool_taps: Tensor

    def __init__(
        self,
        num_channels: int,
        bank: BankSpec,
        *,
        template: MultiPackSpec | OpName | str = "grad",
        base: str | ActivationSpec[Tensor] = "tanh",
        learnable_offsets: bool = True,
        learnable_scales: bool = False,
        readout: Readout = "response",
        gamma: float = 8.0,
        derivative_order: int = 1,
        gap: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if readout not in ("response", "pooled", "argmax"):
            raise ValueError(f"unknown readout {readout!r}")
        self.num_channels = int(num_channels)
        self.readout: Readout = readout
        self.gamma = float(gamma)
        self.gap = float(gap)
        self.act_spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if isinstance(template, MultiPackSpec):
            self.template: MultiPackSpec | str = template
        else:
            self.template = template_from_op(
                template, derivative_order=derivative_order, gap=gap
            )
        dt = torch.get_default_dtype() if dtype is None else dtype
        off0 = torch.tensor(list(bank.offsets), dtype=dt)
        sc0 = torch.tensor(list(bank.scales), dtype=dt)
        if learnable_offsets:
            self.offsets = nn.Parameter(off0)
        else:
            self.register_buffer("offsets", off0)
        if learnable_scales:
            self.scales = nn.Parameter(sc0)
        else:
            self.register_buffer("scales", sc0)
        taps0 = torch.full((off0.numel(),), 1.0 / max(off0.numel(), 1), dtype=dt)
        self.pool_taps = nn.Parameter(taps0)

    def min_offset_separation(self) -> Tensor:
        return min_offset_separation(self.offsets)

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected z[..., {self.num_channels}], got shape {tuple(z.shape)}"
            )
        resp = scan_response(
            z, self.offsets, self.scales, self.template, self.act_spec, gap=self.gap
        )
        if self.readout == "response":
            return resp
        if self.readout == "pooled":
            taps = self.pool_taps.to(dtype=resp.dtype, device=resp.device)
            if resp.shape[-1] == taps.numel():
                return (resp * taps).sum(dim=-1)
            if resp.ndim >= 2 and resp.shape[-2] == taps.numel():
                pooled_m = (resp * taps.reshape(-1, 1)).sum(dim=-2)
                return pooled_m.mean(dim=-1)
            raise ValueError("pooled readout shape mismatch")
        loc = soft_argmax_offset(resp, self.offsets, gamma=self.gamma)
        if loc.ndim == z.ndim:
            return loc
        return loc.mean(dim=-1)


__all__ = [
    "BankSpec",
    "BiasScan",
    "min_offset_separation",
    "scan_response",
    "soft_argmax_offset",
    "template_from_op",
]

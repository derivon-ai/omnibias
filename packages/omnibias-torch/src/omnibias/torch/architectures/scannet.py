# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scan-Net: stacked bias-scan banks, no pixel grid (theory 02-01).

Each layer is a learned direction ``z = w · [x; u] + b`` followed by a
shared-template :class:`~omnibias.torch.scan.BiasScan`. Equivariance is
**per-layer, per-direction, on-lattice** — not the translation group of
``R^D``. Templates reuse the six ``OperatorBlock`` roles; Scan-Net is not
a seventh role.

Soft-argmax ``gamma`` is a readout sharpness, not ``delta -> 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from omnibias.core.scan import BankSpec
from omnibias.core.spectral_design import BandPlan
from omnibias.torch.scan import BiasScan, OpName, Readout

import torch
import torch.nn as nn
from torch import Tensor

_VALID_READOUT: frozenset[str] = frozenset({"pooled", "response", "argmax"})


@dataclass(frozen=True)
class ScanNetConfig:
    """Layer stack for a grid-free scan network."""

    dim_in: int
    channels: tuple[int, ...]
    bank_sizes: tuple[int, ...]
    bank_extents: tuple[float, ...]
    template: OpName | str = "grad"
    base: str = "tanh"
    readout: Literal["pooled", "response", "argmax"] = "pooled"

    def __post_init__(self) -> None:
        if int(self.dim_in) < 1:
            raise ValueError(f"dim_in must be >= 1, got {self.dim_in}")
        if not self.channels:
            raise ValueError("channels must be non-empty")
        n = len(self.channels)
        if len(self.bank_sizes) != n or len(self.bank_extents) != n:
            raise ValueError("bank_sizes and bank_extents must match channels")
        if any(int(m) < 2 for m in self.bank_sizes):
            raise ValueError("each bank_size must be >= 2")
        if any(float(e) <= 0.0 for e in self.bank_extents):
            raise ValueError("each bank_extent must be positive")
        if self.readout not in _VALID_READOUT:
            raise ValueError(f"unknown readout {self.readout!r}")


def scannet_from_band_plan(
    plan: BandPlan,
    *,
    dim_in: int,
    width: int = 4,
    bank_size: int = 9,
    template: OpName | str = "grad",
    base: str = "tanh",
    readout: Literal["pooled", "response", "argmax"] = "pooled",
) -> ScanNetConfig:
    """One layer per band-plan channel; extents come from temper scales.

    01-06 wavelet frames stay concept. This only copies the 01-07 scales.
    """
    n = plan.n_channels
    return ScanNetConfig(
        dim_in=int(dim_in),
        channels=tuple(int(width) for _ in range(n)),
        bank_sizes=tuple(int(bank_size) for _ in range(n)),
        bank_extents=tuple(float(s) for s in plan.scales),
        template=template,
        base=base,
        readout=readout,
    )


class _ScanNetLayer(nn.Module):
    """One projection + :class:`BiasScan` + learned taps."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bank: BankSpec,
        *,
        template: OpName | str,
        base: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, dtype=dtype)
        self.scan = BiasScan(
            dim_out,
            bank,
            template=template,
            base=base,
            learnable_offsets=False,
            learnable_scales=False,
            readout="response",
            dtype=dtype,
        )
        n_off = int(self.scan.offsets.numel())
        taps0 = torch.full((dim_out, n_off), 1.0 / float(n_off), dtype=dtype)
        self.taps = nn.Parameter(taps0)

    def forward(self, h: Tensor, *, readout: Readout) -> Tensor:
        z = self.proj(h)
        resp = self.scan(z)
        if readout == "response":
            return cast(Tensor, resp)
        return cast(Tensor, (resp * self.taps).sum(dim=-1))


class ScanNet(nn.Module):
    """Stacked scan banks. ``forward(x, u=None)`` concatenates ``u`` when given."""

    def __init__(self, config: ScanNetConfig, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if config.readout not in _VALID_READOUT:
            raise ValueError(f"unknown readout {config.readout!r}")
        self.config = config
        dt = torch.get_default_dtype() if dtype is None else dtype
        layers: list[_ScanNetLayer] = []
        din = int(config.dim_in)
        for cout, m, extent in zip(
            config.channels, config.bank_sizes, config.bank_extents, strict=True
        ):
            bank = BankSpec.uniform(-float(extent), float(extent), int(m))
            layers.append(
                _ScanNetLayer(
                    din,
                    int(cout),
                    bank,
                    template=config.template,
                    base=config.base,
                    dtype=dt,
                )
            )
            din = int(cout)
        self.layers = nn.ModuleList(layers)

    def layer_preactivation(self, h: Tensor, index: int = 0) -> Tensor:
        """``z = W h + b`` of layer ``index`` (for on-lattice equivariance tests)."""
        layer = self.layers[int(index)]
        assert isinstance(layer, _ScanNetLayer)
        return cast(Tensor, layer.proj(h))

    def layer_scan_response(self, z: Tensor, index: int = 0) -> Tensor:
        """Bank response of layer ``index`` given a pre-activation ``z``."""
        layer = self.layers[int(index)]
        assert isinstance(layer, _ScanNetLayer)
        return cast(Tensor, layer.scan(z))

    def forward(self, x: Tensor, u: Tensor | None = None) -> Tensor:
        h = x if u is None else torch.cat((x, u), dim=-1)
        if h.shape[-1] != self.config.dim_in:
            raise ValueError(
                f"expected last dim {self.config.dim_in}, got {tuple(h.shape)}"
            )
        n = len(self.layers)
        out: Tensor = h
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, _ScanNetLayer)
            last = i == n - 1
            if last and self.config.readout == "response":
                out = cast(Tensor, layer.scan(layer.proj(out)))
            elif last and self.config.readout == "argmax":
                z = cast(Tensor, layer.proj(out))
                resp = cast(Tensor, layer.scan(z))
                from omnibias.torch.scan import soft_argmax_offset

                loc = soft_argmax_offset(resp, layer.scan.offsets, gamma=layer.scan.gamma)
                out = loc if loc.ndim == z.ndim else loc.mean(dim=-1)
            else:
                out = layer(out, readout="pooled")
        return out


__all__ = [
    "ScanNet",
    "ScanNetConfig",
    "scannet_from_band_plan",
]

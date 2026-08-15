# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scan-Net JAX twin (theory 02-01). Functional, bit-identical to torch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from omnibias.core.scan import BankSpec
from omnibias.core.spectral_design import BandPlan
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.scan import init_bias_scan, scan_response, soft_argmax_offset

import jax.numpy as jnp
from jax import Array

OpName = Literal["identity", "grad", "laplacian", "derivative", "band", "integral"]
Readout = Literal["pooled", "response", "argmax"]


@dataclass(frozen=True)
class ScanNetConfig:
    dim_in: int
    channels: tuple[int, ...]
    bank_sizes: tuple[int, ...]
    bank_extents: tuple[float, ...]
    template: OpName | str = "grad"
    base: str = "tanh"
    readout: Readout = "pooled"

    def __post_init__(self) -> None:
        if int(self.dim_in) < 1:
            raise ValueError(f"dim_in must be >= 1, got {self.dim_in}")
        if not self.channels:
            raise ValueError("channels must be non-empty")
        n = len(self.channels)
        if len(self.bank_sizes) != n or len(self.bank_extents) != n:
            raise ValueError("bank_sizes and bank_extents must match channels")


def scannet_from_band_plan(
    plan: BandPlan,
    *,
    dim_in: int,
    width: int = 4,
    bank_size: int = 9,
    template: OpName | str = "grad",
    base: str = "tanh",
    readout: Readout = "pooled",
) -> ScanNetConfig:
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


@dataclass(frozen=True)
class ScanNetParams:
    """Per-layer ``(W, b, offsets, scales, taps)``. ``W`` is ``(out, in)`` like ``nn.Linear``."""

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    offsets: tuple[Array, ...]
    scales: tuple[Array, ...]
    taps: tuple[Array, ...]


def init_scan_net(config: ScanNetConfig) -> tuple[JaxActivationSpec, ScanNetParams]:
    """Zero-init weights (parity tests overwrite from torch); uniform banks."""
    act = get_activation(config.base)
    weights: list[Array] = []
    biases: list[Array] = []
    offsets: list[Array] = []
    scales: list[Array] = []
    taps: list[Array] = []
    din = int(config.dim_in)
    for cout, m, extent in zip(
        config.channels, config.bank_sizes, config.bank_extents, strict=True
    ):
        w = jnp.zeros((int(cout), din), dtype=jnp.float64)
        b = jnp.zeros((int(cout),), dtype=jnp.float64)
        bank = BankSpec.uniform(-float(extent), float(extent), int(m))
        _act, off, sc, _tmpl, _pool = init_bias_scan(
            int(cout), bank, template=config.template, base=act
        )
        tap = jnp.full((int(cout), int(off.size)), 1.0 / float(off.size), dtype=jnp.float64)
        weights.append(w)
        biases.append(b)
        offsets.append(off)
        scales.append(sc)
        taps.append(tap)
        din = int(cout)
    params = ScanNetParams(
        weights=tuple(weights),
        biases=tuple(biases),
        offsets=tuple(offsets),
        scales=tuple(scales),
        taps=tuple(taps),
    )
    return act, params


def scan_net_apply(
    params: ScanNetParams,
    x: Array,
    u: Array | None = None,
    *,
    config: ScanNetConfig,
    base: str | JaxActivationSpec | None = None,
) -> Array:
    """Functional forward. ``nn.Linear`` convention: ``z = h @ W.T + b``."""
    act = get_activation(config.base) if base is None else (
        get_activation(base) if isinstance(base, str) else base
    )
    h = x if u is None else jnp.concatenate((x, u), axis=-1)
    if h.shape[-1] != config.dim_in:
        raise ValueError(f"expected last dim {config.dim_in}, got {tuple(h.shape)}")
    n = len(params.weights)
    from omnibias.jax.scan import template_from_op

    tmpl = template_from_op(config.template)
    out: Array = h
    for i in range(n):
        w = params.weights[i]
        z = out @ jnp.swapaxes(w, -1, -2) + params.biases[i]
        last = i == n - 1
        readout: Readout = config.readout if last else "pooled"
        if readout == "response":
            out = scan_response(z, params.offsets[i], params.scales[i], tmpl, act)
        elif readout == "argmax":
            resp = scan_response(z, params.offsets[i], params.scales[i], tmpl, act)
            loc = soft_argmax_offset(resp, params.offsets[i], gamma=8.0)
            out = loc if loc.ndim == z.ndim else loc.mean(axis=-1)
        else:
            resp = scan_response(z, params.offsets[i], params.scales[i], tmpl, act)
            out = (resp * params.taps[i]).sum(axis=-1)
    return out


def scan_net_from_torch_state(
    config: ScanNetConfig,
    state: Sequence[tuple[Any, Any, Any, Any, Any]],
) -> ScanNetParams:
    """``state`` is ``[(W, b, offsets, scales, taps), ...]`` numpy-compatible."""
    del config
    weights = tuple(jnp.asarray(w, dtype=jnp.float64) for w, _b, _o, _s, _t in state)
    biases = tuple(jnp.asarray(b, dtype=jnp.float64) for _w, b, _o, _s, _t in state)
    offsets = tuple(jnp.asarray(o, dtype=jnp.float64) for _w, _b, o, _s, _t in state)
    scales = tuple(jnp.asarray(s, dtype=jnp.float64) for _w, _b, _o, s, _t in state)
    taps = tuple(jnp.asarray(t, dtype=jnp.float64) for _w, _b, _o, _s, t in state)
    return ScanNetParams(weights, biases, offsets, scales, taps)


__all__ = [
    "ScanNetConfig",
    "ScanNetParams",
    "init_scan_net",
    "scan_net_apply",
    "scan_net_from_torch_state",
    "scannet_from_band_plan",
]

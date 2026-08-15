# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias scan: shared template on a bank of offsets (JAX twin; theory 01-02).

Functional twin of :mod:`omnibias.torch.scan`. Orders stay Python tuples so
the path is ``jit``-safe. Equivariance is an interior lattice shift, not a
circular wrap. Soft-argmax ``gamma`` is not bias collapse.
"""

from __future__ import annotations

from typing import Literal

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.core.scan import BankSpec
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.multipack import multipack_response

import jax.numpy as jnp
from jax import Array
from jax.nn import softmax

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
    z: Array,
    offsets: Array,
    scales: Array,
    spec: MultiPackSpec | str,
    base: JaxActivationSpec,
    *,
    gap: float = 1.0,
) -> Array:
    """Evaluate the template at ``scale * (z + offset)``."""
    off = jnp.reshape(offsets, (-1,))
    sc = jnp.reshape(scales, (-1,))
    bank = z[..., None] + off
    extra_scale = int(sc.size) != 1
    if extra_scale:
        u = bank[..., None] * sc
    else:
        u = sc.reshape(()) * bank
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
        means = jnp.asarray([p.mean for p in spec.packs], dtype=u.dtype)
        weights = jnp.asarray([p.weight for p in spec.packs], dtype=u.dtype)
        out = multipack_response(u, means, weights, orders, base)
    return out


def soft_argmax_offset(response: Array, offsets: Array, *, gamma: float = 8.0) -> Array:
    off = jnp.reshape(offsets, (-1,))
    if response.shape[-1] == off.size:
        w = softmax(float(gamma) * response, axis=-1)
        return (w * off).sum(axis=-1)
    if response.ndim >= 2 and response.shape[-2] == off.size:
        w = softmax(float(gamma) * response, axis=-2)
        return (w * off.reshape((-1, 1))).sum(axis=-2)
    raise ValueError("response trailing shape does not match offsets")


def min_offset_separation(offsets: Array) -> Array:
    ordered = jnp.sort(jnp.reshape(offsets, (-1,)))
    if ordered.size < 2:
        return jnp.asarray(float("inf"), dtype=offsets.dtype)
    return jnp.min(ordered[1:] - ordered[:-1])


def init_bias_scan(
    num_channels: int,
    bank: BankSpec,
    *,
    template: MultiPackSpec | OpName | str = "grad",
    base: str | JaxActivationSpec = "tanh",
    derivative_order: int = 1,
    gap: float = 1.0,
) -> tuple[JaxActivationSpec, Array, Array, MultiPackSpec | str, Array]:
    """Return ``(act, offsets, scales, template, pool_taps)``."""
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    act = get_activation(base) if isinstance(base, str) else base
    tmpl: MultiPackSpec | str
    if isinstance(template, MultiPackSpec):
        tmpl = template
    else:
        tmpl = template_from_op(template, derivative_order=derivative_order, gap=gap)
    offsets = jnp.asarray(list(bank.offsets), dtype=jnp.float64)
    scales = jnp.asarray(list(bank.scales), dtype=jnp.float64)
    n = int(offsets.size)
    pool_taps = jnp.full((n,), 1.0 / max(n, 1), dtype=jnp.float64)
    return act, offsets, scales, tmpl, pool_taps


def bias_scan(
    z: Array,
    offsets: Array,
    scales: Array,
    template: MultiPackSpec | str,
    base: str | JaxActivationSpec = "tanh",
    *,
    gap: float = 1.0,
    readout: Readout = "response",
    gamma: float = 8.0,
    pool_taps: Array | None = None,
) -> Array:
    """Functional scan forward (bit-identical twin of :class:`omnibias.torch.scan.BiasScan`)."""
    act = get_activation(base) if isinstance(base, str) else base
    resp = scan_response(z, offsets, scales, template, act, gap=gap)
    if readout == "response":
        return resp
    taps = (
        jnp.full((offsets.size,), 1.0 / max(int(offsets.size), 1), dtype=resp.dtype)
        if pool_taps is None
        else pool_taps
    )
    if readout == "pooled":
        if resp.shape[-1] == taps.size:
            return (resp * taps).sum(axis=-1)
        if resp.ndim >= 2 and resp.shape[-2] == taps.size:
            return (resp * taps.reshape((-1, 1))).sum(axis=-2).mean(axis=-1)
        raise ValueError("pooled readout shape mismatch")
    loc = soft_argmax_offset(resp, offsets, gamma=gamma)
    if loc.ndim == z.ndim:
        return loc
    return loc.mean(axis=-1)


__all__ = [
    "BankSpec",
    "bias_scan",
    "init_bias_scan",
    "min_offset_separation",
    "scan_response",
    "soft_argmax_offset",
    "template_from_op",
]

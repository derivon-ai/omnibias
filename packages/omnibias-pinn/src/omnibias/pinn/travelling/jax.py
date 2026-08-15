# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soliton field (JAX twin; theory 02-09)."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from omnibias.core.tanh_method import (
    PDESpec,
    TravellingWaveAnsatz,
    evaluate_ansatz,
    substitute,
)


def soliton_apply(
    ansatz: Sequence[TravellingWaveAnsatz],
    x: Array,
    t: Array,
    *,
    amps: Sequence[float] | None = None,
) -> Array:
    xs = jnp.asarray(x).reshape(-1)
    ts = jnp.asarray(t).reshape(-1)
    weights = tuple(1.0 for _ in ansatz) if amps is None else tuple(float(a) for a in amps)
    vals = []
    for xi, ti in zip(xs.tolist(), ts.tolist(), strict=True):
        acc = 0.0
        for a, wave in zip(weights, ansatz, strict=True):
            acc += a * evaluate_ansatz(wave, float(xi), float(ti))
        vals.append(acc)
    return jnp.asarray(vals, dtype=xs.dtype).reshape(jnp.asarray(x).shape)


def exact_residual(
    ansatz: Sequence[TravellingWaveAnsatz], x: Array, t: Array, pde: PDESpec
) -> Array:
    import math

    xs = jnp.asarray(x).reshape(-1)
    ts = jnp.asarray(t).reshape(-1)
    vals = []
    for xi, ti in zip(xs.tolist(), ts.tolist(), strict=True):
        acc = 0.0
        for wave in ansatz:
            coeffs = substitute(pde, wave)
            z = float(wave.wavenumber) * float(xi) - float(wave.frequency) * float(ti)
            tnh = math.tanh(z)
            res = 0.0
            p = 1.0
            for c in coeffs:
                res += float(c) * p
                p *= tnh
        vals.append(acc)
    return jnp.asarray(vals, dtype=xs.dtype).reshape(jnp.asarray(x).shape)


__all__ = ["exact_residual", "soliton_apply"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Activation-specific closed-form fractional derivatives (jax twin).

Bit-identical twin of :mod:`omnibias.fractional.torch.ops.activation`:
:func:`exp_fractional`, :func:`cosh_fractional`, :func:`sinh_fractional`, the
:data:`ACTIVATION_FRACTIONAL` registry, and
:func:`activation_fractional_derivative`. See the torch twin for the identities.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from omnibias.fractional.jax.ops.special import _recip_gamma, mittag_leffler


def exp_fractional(
    x: Array | float,
    *,
    alpha: float | Array,
    lam: float = 1.0,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Array:
    r"""Closed-form fractional derivative of ``e^{lam x}`` (jax twin; see torch twin)."""
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}")
    x_t = jnp.asarray(x)
    a = jnp.asarray(alpha, dtype=x_t.dtype)
    xma = jnp.exp(-a * jnp.log(x_t))
    rl: Array = xma * mittag_leffler(lam * x_t, 1.0, 1.0 - a, terms=terms)
    if kind == "riemann_liouville":
        return rl
    m = int(math.ceil(float(a)))
    head = jnp.zeros_like(rl)
    for k in range(m):
        coef = (lam**k) * _recip_gamma(k + 1.0 - a)
        head = head + coef * jnp.exp((k - a) * jnp.log(x_t))
    return rl - head


def cosh_fractional(
    x: Array | float,
    *,
    alpha: float | Array,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Array:
    r"""Closed-form fractional derivative of ``cosh`` (jax twin; see torch twin)."""
    pos = exp_fractional(x, alpha=alpha, lam=1.0, kind=kind, terms=terms)
    neg = exp_fractional(x, alpha=alpha, lam=-1.0, kind=kind, terms=terms)
    return 0.5 * (pos + neg)


def sinh_fractional(
    x: Array | float,
    *,
    alpha: float | Array,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Array:
    r"""Closed-form fractional derivative of ``sinh`` (jax twin; see torch twin)."""
    pos = exp_fractional(x, alpha=alpha, lam=1.0, kind=kind, terms=terms)
    neg = exp_fractional(x, alpha=alpha, lam=-1.0, kind=kind, terms=terms)
    return 0.5 * (pos - neg)


ACTIVATION_FRACTIONAL: dict[str, Callable[..., Array]] = {
    "exp": exp_fractional,
    "cosh": cosh_fractional,
    "sinh": sinh_fractional,
}


def activation_fractional_derivative(
    name: str,
    x: Array | float,
    *,
    alpha: float | Array,
    kind: str = "riemann_liouville",
    **kwargs: object,
) -> Array:
    r"""Dispatch to the registered closed-form fractional derivative (jax twin)."""
    if name not in ACTIVATION_FRACTIONAL:
        raise KeyError(
            f"no closed-form fractional derivative registered for {name!r}; "
            f"available: {sorted(ACTIVATION_FRACTIONAL)}"
        )
    return ACTIVATION_FRACTIONAL[name](x, alpha=alpha, kind=kind, **kwargs)


__all__ = [
    "ACTIVATION_FRACTIONAL",
    "activation_fractional_derivative",
    "cosh_fractional",
    "exp_fractional",
    "sinh_fractional",
]

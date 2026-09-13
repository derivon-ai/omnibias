# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable centered collision atoms, live through the attained rho=0 boundary.

Finite formulas are exact in real arithmetic. For abs(h*eta)<=series_radius an
explicit even Taylor polynomial is evaluated (series_terms coefficients). Its
remainder is bounded by the shared core confluence derivative bounds. The
inactive finite branch uses positive denominators so backward/JIT are safe.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import factorial
from typing import NamedTuple

from omnibias.jax.activations import get_activation

import jax
import jax.numpy as jnp
from jax import Array


class CenteredPair(NamedTuple):
    average: Array
    divided_difference: Array


def retract_spread(rho: Array) -> Array:
    """Project onto rho>=0; call between steps to attain zero exactly."""
    return jnp.where(rho >= 0, rho, jnp.zeros_like(rho))


def centered_pair(
    z: Array, rho: Array, eta: Array, *, activation: str = "sigmoid",
    series_terms: int = 6, series_radius: float = 0.01,
    tower: Array | Sequence[Array] | None = None,
) -> CenteredPair:
    """A=(sigma(z+h*eta)+sigma(z-h*eta))/2, B=odd difference/(2h).

    rho=h**2 is nonnegative (projected in this function). At rho=0, A=sigma(z)
    and B=eta*sigma'(z). Includes live derivatives in rho, z, and eta. Sigmoid
    and tanh have stable finite-spread formulas; other activations are rejected.
    The integer series_terms and radius are static compilation parameters.
    """
    if activation not in ("sigmoid", "tanh"):
        raise ValueError("centered_pair supports sigmoid and tanh")
    if series_terms < 1 or not 0 < series_radius <= 0.1:
        raise ValueError("positive terms and radius in (0,0.1] required")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError("activation requires a derivative provider")
    p = retract_spread(rho)
    small = p * eta**2 <= series_radius**2
    safe_rho = jnp.where(small, jnp.ones_like(p), p)
    h = jnp.sqrt(safe_rho)
    u = jnp.abs(h * eta)
    u = jnp.where(small, jnp.ones_like(u), u)
    # Three sigmoid evaluations in the finite path; no subtraction of nearby
    # activation values. tanh(z)=2*sigmoid(2*z)-1 supplies the same formula.
    scale = 1.0 if activation == "sigmoid" else 2.0
    zz = scale * z
    uu = scale * u
    splus = jax.nn.sigmoid(zz + uu)
    sminus = jax.nn.sigmoid(zz - uu)
    complement = jax.nn.sigmoid(-zz + uu)
    a_finite = (splus + sminus) * 0.5
    b_finite = eta * scale * splus * complement * (-jnp.expm1(-2 * uu) / (2 * uu))
    if activation == "tanh":
        a_finite = 2 * a_finite - 1
        b_finite = 2 * b_finite
    # Evaluate the near branch at safe spread even when it is inactive: huge
    # finite spreads must not overflow powers in a discarded Taylor polynomial.
    r = jnp.where(small, p, jnp.zeros_like(p))
    e = jnp.where(small, eta, jnp.zeros_like(eta))
    maximum = 2 * series_terms - 1
    if tower is None:
        tower = (spec.tower(z, maximum) if spec.tower is not None
                 else tuple(spec.fastpath(z, k) for k in range(maximum + 1)))
    if len(tower) <= maximum:
        raise ValueError("supplied tower is too short for the centered series")
    a_series = jnp.zeros_like(z)
    b_series = jnp.zeros_like(z)
    for k in range(series_terms):
        a_series = a_series + tower[2*k] * (r**k * e**(2*k)) / factorial(2*k)
        b_series = b_series + tower[2*k+1] * (r**k * e**(2*k+1)) / factorial(2*k+1)
    return CenteredPair(jnp.where(small, a_series, a_finite),
                        jnp.where(small, b_series, b_finite))


def collision_atom(z: Array, rho: Array, eta: Array, m0: Array, m1: Array,
                   *, activation: str = "sigmoid", series_terms: int = 6,
                   tower: Array | Sequence[Array] | None = None) -> Array:
    """m0*A + m1*B; moments remain bounded as a pair becomes a derivative atom."""
    pair = centered_pair(z, rho, eta, activation=activation, series_terms=series_terms, tower=tower)
    return m0 * pair.average + m1 * pair.divided_difference


def moment_atom(z: Array, moments: Array, *, activation: str = "sigmoid",
                tower: Array | Sequence[Array] | None = None) -> Array:
    """A truncated common-weight bias cluster, sum(m[k]*sigma**(k)(z)).

    The host initialize_moments/cluster_remainder API supplies its error bound.
    This is exact only when the selected atom architecture is itself the model.
    """
    if moments.ndim != 1 or moments.shape[0] < 1:
        raise ValueError("moments must be a nonempty vector")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError("activation requires a derivative provider")
    n = moments.shape[0] - 1
    if tower is None:
        tower = (spec.tower(z, n) if spec.tower is not None
                 else tuple(spec.fastpath(z, k) for k in range(n+1)))
    if len(tower) <= n:
        raise ValueError("supplied tower is too short for the moment expansion")
    out = jnp.zeros_like(z)
    for k in range(n+1):
        out = out + moments[k] * tower[k]
    return out



def polynomial_pair(z: Array, rho: Array, eta: Array, coefficients: Array) -> CenteredPair:
    """Exact finite polynomial control in the same centered coordinates.

    Coefficients are live and in ascending powers. No finite-spread truncation
    occurs: the even/odd binomial expansion terminates at the supplied degree.
    """
    from math import comb
    if coefficients.ndim != 1 or coefficients.shape[0] < 1:
        raise ValueError('nonempty polynomial coefficient vector required')
    p = retract_spread(rho)
    a, b = jnp.zeros_like(z), jnp.zeros_like(z)
    for degree in range(coefficients.shape[0]):
        for power in range(degree+1):
            term = coefficients[degree]*comb(degree,power)*z**(degree-power)
            if power % 2:
                b = b+term*p**((power-1)//2)*eta**power
            else:
                a = a+term*p**(power//2)*eta**power
    return CenteredPair(a,b)

__all__ = ["CenteredPair", "centered_pair", "collision_atom", "moment_atom", "polynomial_pair", "retract_spread"]

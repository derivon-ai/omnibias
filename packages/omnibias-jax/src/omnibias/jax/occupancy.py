# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Fermi-Dirac occupancy and thermodynamic potentials
(JAX twin; theory 04-03).

Point-for-point mirror of :mod:`omnibias.torch.occupancy` and bit-identical
(given the same ``sigmoid`` sample) twin of :mod:`omnibias.core.occupancy`,
built the same way as :mod:`omnibias.jax._fastpath`: one framework-native
``jax.nn.sigmoid`` / ``jax.nn.softplus`` call per function, then ``O(n)``
Horner multiply-adds over the coefficients shared from
:mod:`omnibias.core.polynomials` -- never re-derived per backend.

With ``z = -beta * (energy - mu)``:

.. math::

    f(e) = \sigma(z), \qquad
    s(z) = \mathrm{softplus}(z) - z\,\sigma(z), \qquad
    \omega(z) = -\tfrac{1}{\beta}\,\mathrm{softplus}(z).

Every function is pure ``jax.numpy`` arithmetic with Python-``int`` control
flow on the static order ``n`` only (no data-dependent branching), so every
function here is ``jit`` / ``vmap`` / ``grad`` safe -- no host coercion of a
value that might be a trace-time abstract array.

Two collapse senses are named (mirroring the core module) and must not be
conflated: the founding bias collapse (``delta -> 0``) never appears in
this module; ``beta -> inf`` is the founding **temperature collapse**
(feasibility sense) and is not evaluated or requested by anything here --
see :mod:`omnibias.core.occupancy` for the honest reference value and the
honesty payload. Do not conflate the two.

Scope: non-interacting fermions with a caller-supplied density of states;
not density-functional theory, not a many-body solve, no thermodynamic
limit. See :func:`omnibias.core.occupancy.honesty_payload`.

Note on the module name: :mod:`omnibias.jax` re-exports :func:`occupancy`
(this module's own function) at package level, per the repository's
"regenerate ``__all__``" convention. Because of that, once :mod:`omnibias.jax`
has been imported, ``omnibias.jax.occupancy`` is the *function*, not this
submodule -- ordinary Python attribute shadowing, not a bug. Import specific
names directly (``from omnibias.jax.occupancy import occupancy_derivative``),
which resolves against the fully-qualified module in ``sys.modules`` and is
unaffected by the shadow.
"""

from __future__ import annotations

from omnibias.core.polynomials import sigmoid_polynomial_coeffs

import jax.nn as jnn
import jax.numpy as jnp
from jax import Array


def _horner(coeffs: tuple[float, ...], x: Array) -> Array:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k`` on a JAX array."""
    deg = len(coeffs) - 1
    result = jnp.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def _sigma_tower(z: Array, order: int) -> list[Array]:
    """``[sigma(z), sigma'(z), ..., sigma^(order)(z)]``, one ``jax.nn.sigmoid``
    call then Horner over the shared Eulerian coefficients."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    s = jnn.sigmoid(z)
    rows = [s]
    for k in range(1, order + 1):
        rows.append(_horner(sigmoid_polynomial_coeffs(k), s))
    return rows


def reduced_argument(energy: Array, beta: Array | float, mu: Array | float) -> Array:
    """``z = -beta * (energy - mu)``, the sigmoid tower's native argument."""
    return -beta * (energy - mu)


def occupancy(energy: Array, beta: Array | float, mu: Array | float) -> Array:
    """Fermi-Dirac occupancy ``f(e) = sigma(z)``, ``z = -beta (e - mu)``."""
    z = reduced_argument(energy, beta, mu)
    return jnn.sigmoid(z)


def occupancy_derivative(
    energy: Array, beta: Array | float, mu: Array | float, *, order: int
) -> Array:
    """``d^order f / de^order = (-beta)^order * sigma^(order)(z)``.

    ``order`` is a static Python ``int`` (branches only at trace time, so
    this remains ``jit``/``vmap`` safe); the result stays differentiable
    (``jax.grad``) in ``beta`` and ``mu`` via the ``(-beta)**order``
    prefactor and ``z``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = reduced_argument(energy, beta, mu)
    tower = _sigma_tower(z, order)
    return ((-beta) ** order) * tower[order]


def occupancy_mu_derivative(
    energy: Array, beta: Array | float, mu: Array | float, *, order: int
) -> Array:
    """``d^order f / dmu^order = beta^order * sigma^(order)(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = reduced_argument(energy, beta, mu)
    tower = _sigma_tower(z, order)
    return (beta**order) * tower[order]


def thermal_broadening(energy: Array, beta: Array | float, mu: Array | float) -> Array:
    """``-df/de = beta * sigma'(z)``, the thermal smearing width of the step."""
    return -occupancy_derivative(energy, beta, mu, order=1)


def entropy_per_state(energy: Array, beta: Array | float, mu: Array | float) -> Array:
    """Fermi entropy per state, ``s(z) = softplus(z) - z sigma(z)``."""
    z = reduced_argument(energy, beta, mu)
    return jnn.softplus(z) - z * jnn.sigmoid(z)


def grand_potential_density(
    energy: Array, beta: Array | float, mu: Array | float
) -> Array:
    """Grand-potential density per state, ``omega(z) = -(1/beta) softplus(z)``."""
    z = reduced_argument(energy, beta, mu)
    return -jnn.softplus(z) / beta


def occupancy_window(
    e_lo: Array, e_hi: Array, beta: Array | float, mu: Array | float
) -> Array:
    r"""Exact electron count in ``[e_lo, e_hi]`` for a constant density of
    states (the ``band`` role); see :func:`omnibias.core.occupancy.
    occupancy_window`. Does not itself validate ``e_lo <= e_hi`` (a pure
    array op meant to stay ``jit``/``vmap`` safe); callers wanting that
    check should use the core module.
    """
    z_lo = reduced_argument(e_lo, beta, mu)
    z_hi = reduced_argument(e_hi, beta, mu)
    return (jnn.softplus(z_lo) - jnn.softplus(z_hi)) / beta


__all__ = [
    "entropy_per_state",
    "grand_potential_density",
    "occupancy",
    "occupancy_derivative",
    "occupancy_mu_derivative",
    "occupancy_window",
    "reduced_argument",
    "thermal_broadening",
]

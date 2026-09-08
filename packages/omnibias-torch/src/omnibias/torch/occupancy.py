# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Fermi-Dirac occupancy and thermodynamic potentials
(torch twin; theory 04-03).

Bit-identical (given the same ``sigmoid`` sample) twin of
:mod:`omnibias.core.occupancy`, built the same way as
:mod:`omnibias.torch.fastpath.eulerian`: one framework-native
``torch.sigmoid`` / ``torch.nn.functional.softplus`` call per function,
then ``O(n)`` Horner multiply-adds over the coefficients shared from
:mod:`omnibias.core.polynomials` -- never re-derived per backend.

With ``z = -beta * (energy - mu)``:

.. math::

    f(e) = \sigma(z), \qquad
    s(z) = \mathrm{softplus}(z) - z\,\sigma(z), \qquad
    \omega(z) = -\tfrac{1}{\beta}\,\mathrm{softplus}(z).

Every function here is differentiable in both ``energy``/``e_lo``/``e_hi``
*and* ``beta``/``mu`` (native autograd sees through the tensor arithmetic;
the closed-form Horner tower is only used for the ``order``-th *energy*
derivative requested by :func:`occupancy_derivative`, which itself remains
differentiable in ``beta`` and ``mu``). All ops are plain tensor arithmetic
with Python-``int`` control flow on the static order ``n`` only, so every
function here is ``torch.jit.script`` / ``torch.vmap`` safe -- no host
coercion of a tensor that might be a functorch/vmap batch tracer.

Two collapse senses are named (mirroring the core module) and must not be
conflated: the founding bias collapse (``delta -> 0``) never appears in
this module; ``beta -> inf`` is the founding **temperature collapse**
(feasibility sense) and is not evaluated or requested by anything here --
see :mod:`omnibias.core.occupancy` for the honest reference value and the
honesty payload. Do not conflate the two.

Scope: non-interacting fermions with a caller-supplied density of states;
not density-functional theory, not a many-body solve, no thermodynamic
limit. See :func:`omnibias.core.occupancy.honesty_payload`.

Note on the module name: :mod:`omnibias.torch` re-exports :func:`occupancy`
(this module's own function) at package level, per the repository's
"regenerate ``__all__``" convention. Because of that, once
:mod:`omnibias.torch` has been imported, ``omnibias.torch.occupancy`` is the
*function*, not this submodule -- ordinary Python attribute shadowing, not a
bug. Import specific names directly (``from omnibias.torch.occupancy import
occupancy_derivative``), which resolves against the fully-qualified module in
``sys.modules`` and is unaffected by the shadow.
"""

from __future__ import annotations

from omnibias.core.polynomials import sigmoid_polynomial_coeffs

import torch
from torch import Tensor


def _horner(coeffs: tuple[float, ...], x: Tensor) -> Tensor:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k`` on a tensor."""
    deg = len(coeffs) - 1
    result = torch.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def _sigma_tower(z: Tensor, order: int) -> list[Tensor]:
    """``[sigma(z), sigma'(z), ..., sigma^(order)(z)]``, one ``torch.sigmoid``
    call then Horner over the shared Eulerian coefficients."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    s = torch.sigmoid(z)
    rows = [s]
    for k in range(1, order + 1):
        rows.append(_horner(sigmoid_polynomial_coeffs(k), s))
    return rows


def reduced_argument(energy: Tensor, beta: Tensor | float, mu: Tensor | float) -> Tensor:
    """``z = -beta * (energy - mu)``, the sigmoid tower's native argument."""
    return -beta * (energy - mu)


def occupancy(energy: Tensor, beta: Tensor | float, mu: Tensor | float) -> Tensor:
    """Fermi-Dirac occupancy ``f(e) = sigma(z)``, ``z = -beta (e - mu)``."""
    z = reduced_argument(energy, beta, mu)
    return torch.sigmoid(z)


def occupancy_derivative(
    energy: Tensor, beta: Tensor | float, mu: Tensor | float, *, order: int
) -> Tensor:
    """``d^order f / de^order = (-beta)^order * sigma^(order)(z)``.

    ``order`` is a static Python ``int`` (branches only at trace time, so
    this remains ``jit``/``vmap`` safe); the result stays differentiable in
    ``beta`` and ``mu`` via the ``(-beta)**order`` prefactor and ``z``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = reduced_argument(energy, beta, mu)
    tower = _sigma_tower(z, order)
    return ((-beta) ** order) * tower[order]


def occupancy_mu_derivative(
    energy: Tensor, beta: Tensor | float, mu: Tensor | float, *, order: int
) -> Tensor:
    """``d^order f / dmu^order = beta^order * sigma^(order)(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = reduced_argument(energy, beta, mu)
    tower = _sigma_tower(z, order)
    return (beta**order) * tower[order]


def thermal_broadening(energy: Tensor, beta: Tensor | float, mu: Tensor | float) -> Tensor:
    """``-df/de = beta * sigma'(z)``, the thermal smearing width of the step."""
    return -occupancy_derivative(energy, beta, mu, order=1)


def entropy_per_state(energy: Tensor, beta: Tensor | float, mu: Tensor | float) -> Tensor:
    """Fermi entropy per state, ``s(z) = softplus(z) - z sigma(z)``."""
    z = reduced_argument(energy, beta, mu)
    return torch.nn.functional.softplus(z) - z * torch.sigmoid(z)


def grand_potential_density(
    energy: Tensor, beta: Tensor | float, mu: Tensor | float
) -> Tensor:
    """Grand-potential density per state, ``omega(z) = -(1/beta) softplus(z)``."""
    z = reduced_argument(energy, beta, mu)
    return -torch.nn.functional.softplus(z) / beta


def occupancy_window(
    e_lo: Tensor, e_hi: Tensor, beta: Tensor | float, mu: Tensor | float
) -> Tensor:
    r"""Exact electron count in ``[e_lo, e_hi]`` for a constant density of
    states (the ``band`` role); see :func:`omnibias.core.occupancy.
    occupancy_window`. Does not itself validate ``e_lo <= e_hi`` (a tensor
    op meant to stay ``jit``/``vmap`` safe); callers wanting that check
    should use the core module.
    """
    z_lo = reduced_argument(e_lo, beta, mu)
    z_hi = reduced_argument(e_hi, beta, mu)
    softplus = torch.nn.functional.softplus
    return (softplus(z_lo) - softplus(z_hi)) / beta


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

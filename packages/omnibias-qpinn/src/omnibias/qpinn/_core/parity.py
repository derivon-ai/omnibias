# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Parity-projection helpers for symmetric / antisymmetric eigenstates.

For a Hamiltonian that commutes with the parity operator
:math:`\mathcal{P}\psi(x) = \psi(-x)` (one-dimensional even
potentials, doubly-symmetric multidimensional traps, ...), every
non-degenerate eigenstate has a definite parity :math:`\varepsilon
\in \{+1, -1\}`. The parity-projected combinations

.. math::

    \psi_\pm(x) = \frac{1}{\sqrt{2}}\,\big[\,u(x) \pm u(-x)\,\big]

isolate the symmetric (+) and antisymmetric (-) sectors. They obey
the closed-form derivative identities

.. math::

    \frac{d^{2n}\psi_\pm}{dx^{2n}}(x)
      &= \frac{1}{\sqrt 2}\,\big[u^{(2n)}(x) \pm u^{(2n)}(-x)\big]\\
    \frac{d^{2n+1}\psi_\pm}{dx^{2n+1}}(x)
      &= \frac{1}{\sqrt 2}\,\big[u^{(2n+1)}(x) \mp u^{(2n+1)}(-x)\big]

(the second derivative is *parity-even* w.r.t. the projection -- the
mirror-image evaluation is added back with the **same** sign for the
Laplacian, regardless of which parity sector is being built; the
first derivative flips the sign on the mirror term).

These helpers are backend-agnostic and operate on the *values* / *n-th
derivatives* of a base field already evaluated at ``coords`` and at
``-coords`` (or the mirror image along the chosen axis). The full
:class:`ParityProjectedField` cage is deferred to v0.0.3; for now the
helpers cover the demo's Time-Independent Schrodinger usage.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

INV_SQRT2: float = 0.7071067811865475244008443621048


def _parity_sign(parity: str) -> float:
    p = parity.lower()
    if p in ("+", "even", "symmetric", "+1"):
        return +1.0
    if p in ("-", "odd", "antisymmetric", "-1"):
        return -1.0
    raise ValueError(f"unknown parity {parity!r}; use 'even' or 'odd'")


def project_parity_value(value_at_x: T, value_at_minus_x: T, *, parity: str) -> T:
    r"""Return :math:`\psi_\pm(x) = (u(x) \pm u(-x)) / \sqrt 2`.

    Parameters
    ----------
    value_at_x, value_at_minus_x
        Output of the base field at ``coords`` and at ``-coords``
        respectively (or any reflected coords along the chosen mirror
        axis). Backend-agnostic.
    parity
        ``"even"`` for symmetric, ``"odd"`` for antisymmetric.

    Returns
    -------
    T
        Same backend / shape as the inputs.
    """
    eps = _parity_sign(parity)
    return INV_SQRT2 * (value_at_x + eps * value_at_minus_x)  # type: ignore[operator,return-value]


def project_parity_even_derivative(
    deriv_at_x: T, deriv_at_minus_x: T, *, parity: str,
) -> T:
    r"""Project an *even-order* derivative (e.g. Laplacian, 4th derivative).

    For an even-order derivative the mirror-image evaluation contributes
    with the **same** sign as the value, regardless of the parity
    sector::

        psi_pm^{(2n)}(x) = (u^{(2n)}(x) + eps u^{(2n)}(-x)) / sqrt 2

    where ``eps = +1 (even sector)`` or ``-1 (odd sector)``.
    """
    eps = _parity_sign(parity)
    return INV_SQRT2 * (deriv_at_x + eps * deriv_at_minus_x)  # type: ignore[operator,return-value]


def project_parity_odd_derivative(
    deriv_at_x: T, deriv_at_minus_x: T, *, parity: str,
) -> T:
    r"""Project an *odd-order* derivative (e.g. gradient, 3rd derivative).

    For an odd-order derivative the mirror-image term enters with an
    extra ``-1`` from the chain rule, so the *relative* sign is
    flipped relative to the value projection::

        psi_pm^{(2n+1)}(x) = (u^{(2n+1)}(x) - eps u^{(2n+1)}(-x)) / sqrt 2.
    """
    eps = _parity_sign(parity)
    return INV_SQRT2 * (deriv_at_x - eps * deriv_at_minus_x)  # type: ignore[operator,return-value]


__all__ = [
    "project_parity_even_derivative",
    "project_parity_odd_derivative",
    "project_parity_value",
]

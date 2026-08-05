# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic pullback of a differential form (pure Python: no torch / jax).

Integrating a ``k``-form over a ``k``-dimensional parametrized submanifold reduces
to a pullback plus a quadrature: for an immersion :math:`\varphi:\mathbb R^d\to
\mathbb R^n` with Jacobian :math:`J = \partial\varphi/\partial x` (shape
``(n, d)``), the pullback of a ``k``-form is the ``k``-form on the ``d``-dimensional
parameter domain

.. math::

    (\varphi^*\omega)_{a_1\dots a_k}
        = \sum_{i_1 < \dots < i_k}
          \omega_{i_1\dots i_k}(\varphi(x))\,
          \det\!\big(\partial\varphi^{i_p}/\partial x^{a_q}\big)_{p,q},

the sum running over strictly increasing ambient index sets. Each :math:`k\times k`
minor determinant is expanded by the Leibniz permutation sum, so the only tensor
operations are ``+`` / ``*`` / indexing -- duck-typed across torch and jax. This
keeps the backend ops in ``{torch,jax}/ops/integration.py`` bit-identical by
construction (they only supply the evaluated form components and the Jacobian).
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from omnibias.geometry._core.forms import permutation_sign, sorted_index_sets


def _minor_det(jac: Any, rows: tuple[int, ...], cols: tuple[int, ...]) -> Any:
    r"""Leibniz determinant of the ``k x k`` minor ``M_{pq} = jac[:, rows[p], cols[q]]``.

    ``jac`` has shape ``(Q, n, d)``; ``rows`` are ambient indices and ``cols`` are
    domain indices, both of length ``k``. Expanded as
    :math:`\sum_\pi \operatorname{sgn}(\pi)\prod_p \mathrm{jac}[:, \mathrm{rows}[p],
    \mathrm{cols}[\pi(p)]]` so only ``+`` / ``*`` / indexing are used.
    """
    k = len(rows)
    acc: Any = None
    for perm in permutations(range(k)):
        term: Any = None
        for p in range(k):
            factor = jac[:, rows[p], cols[perm[p]]]
            term = factor if term is None else term * factor
        sign = permutation_sign(perm)
        contrib = term if sign == 1 else sign * term
        acc = contrib if acc is None else acc + contrib
    return acc


def pullback_form_components(
    values: dict[tuple[int, ...], Any],
    jac: Any,
    degree: int,
    domain_dim: int,
    ambient_dim: int,
) -> dict[tuple[int, ...], Any]:
    r"""Pull an evaluated ambient ``k``-form back through ``J = d phi / d x``.

    Parameters
    ----------
    values
        Mapping ``{sorted ambient k-index -> (Q,) tensor}`` giving the form
        components already evaluated at the image points ``phi(x_q)``. Missing
        index sets are treated as zero.
    jac
        The Jacobian ``d phi / d x`` of shape ``(Q, n, d)`` (``n = ambient_dim``,
        ``d = domain_dim``); ``jac[:, i, a] = d phi^i / d x^a``.
    degree
        The form degree ``k``.
    domain_dim, ambient_dim
        The chart domain / ambient dimensions ``d <= n``.

    Returns
    -------
    dict
        ``{sorted domain k-index -> (Q,) tensor}`` for the pulled-back ``k``-form.
        A ``k``-form with ``k > domain_dim`` pulls back to zero (empty dict); a
        ``0``-form passes through unchanged (``phi^* f = f o phi``).
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if domain_dim > ambient_dim:
        raise ValueError(
            f"domain_dim {domain_dim} must be <= ambient_dim {ambient_dim}"
        )
    for idx in values:
        if len(idx) != degree:
            raise ValueError(
                f"component index {idx!r} does not match form degree {degree}"
            )
    if degree == 0:
        v = values.get(())
        return {} if v is None else {(): v}
    out: dict[tuple[int, ...], Any] = {}
    for a_set in sorted_index_sets(domain_dim, degree):
        acc: Any = None
        for i_set, comp in values.items():
            det = _minor_det(jac, i_set, a_set)
            contrib = comp * det
            acc = contrib if acc is None else acc + contrib
        if acc is not None:
            out[a_set] = acc
    return out


__all__ = ["pullback_form_components"]

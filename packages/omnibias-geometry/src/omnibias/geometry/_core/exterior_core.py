# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic index logic for the Lie derivative and codifferential.

Both operators are closed-form: every term is a closed-form field derivative
(``state.ops.derivative`` on a *named* form/vector component) combined with the
analytic metric / Christoffel arrays. Because the only tensor operations are
``+`` / ``*`` / indexing -- duck-typed across torch and jax -- the bodies are
shared and the per-backend wrappers in ``{torch,jax}/ops/exterior.py`` only
supply the metric arrays. This guarantees torch/jax bit-identical numerics by
construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omnibias.geometry._core.forms import (
    DifferentialForm,
    permutation_sign,
    sorted_index_sets,
)

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _sorted_with_sign(idx: tuple[int, ...]) -> tuple[tuple[int, ...] | None, int]:
    """``(sorted_idx, sign)`` of an index tuple, or ``(None, 0)`` if repeated."""
    if len(set(idx)) != len(idx):
        return None, 0
    order = sorted(range(len(idx)), key=lambda p: idx[p])
    return tuple(sorted(idx)), permutation_sign(tuple(order))


def _form_value(state: FieldState, form: DifferentialForm, idx: tuple[int, ...]) -> Any:
    """Value of an antisymmetric form at an arbitrary index tuple (or ``None``)."""
    s, sign = _sorted_with_sign(idx)
    if s is None:
        return None
    name = form.comps.get(s)
    if name is None:
        return None
    v = state.ops.value(state, name)
    return v if sign == 1 else sign * v


def _form_partial(
    state: FieldState, form: DifferentialForm, idx: tuple[int, ...], m: int,
) -> Any:
    """``d_m`` of an antisymmetric form component at an arbitrary index (or ``None``)."""
    s, sign = _sorted_with_sign(idx)
    if s is None:
        return None
    name = form.comps.get(s)
    if name is None:
        return None
    dv = state.ops.derivative(state, name, axis=m, order=1)
    return dv if sign == 1 else sign * dv


def lie_derivative_components(
    state: FieldState,
    vector_names: tuple[str, ...],
    form: DifferentialForm,
) -> dict[tuple[int, ...], Any]:
    r"""Lie derivative :math:`\mathcal L_X\omega` of a name-form along a name-vector.

    Coordinate (Cartan-equivalent) formula

    .. math::

        (\mathcal L_X\omega)_{i_1\dots i_k}
            = X^m\,\partial_m\omega_{i_1\dots i_k}
            + \sum_{s} (\partial_{i_s}X^m)\,\omega_{i_1\dots m\dots i_k}.
    """
    d = form.dim
    k = form.degree
    if len(vector_names) != d:
        raise ValueError(
            f"vector field needs {d} component names, got {len(vector_names)}"
        )
    xv = [state.ops.value(state, n) for n in vector_names]
    out: dict[tuple[int, ...], Any] = {}
    for idx in sorted_index_sets(d, k):
        acc: Any = None
        for m in range(d):
            p = _form_partial(state, form, idx, m)
            if p is None:
                continue
            term = xv[m] * p
            acc = term if acc is None else acc + term
        for s, i_s in enumerate(idx):
            for m in range(d):
                w = _form_value(state, form, idx[:s] + (m,) + idx[s + 1:])
                if w is None:
                    continue
                d_xm = state.ops.derivative(state, vector_names[m], axis=i_s, order=1)
                term = d_xm * w
                acc = term if acc is None else acc + term
        if acc is not None:
            out[idx] = acc
    return out


def codifferential_components(
    state: FieldState,
    form: DifferentialForm,
    ginv: Any,
    gamma: Any,
) -> dict[tuple[int, ...], Any]:
    r"""Codifferential :math:`\delta\omega` of a name-form (covariant formula).

    .. math::

        (\delta\omega)_{i_1\dots i_{k-1}}
            = -g^{jm}\,\nabla_m\omega_{j i_1\dots i_{k-1}},

    with :math:`\nabla` the Levi-Civita connection. ``ginv`` is :math:`g^{jm}`
    of shape ``(B, d, d)`` and ``gamma`` is :math:`\Gamma^k_{ij}` of shape
    ``(B, d, d, d)`` (indexed ``gamma[:, k, i, j]``).
    """
    d = form.dim
    k = form.degree
    if k == 0:
        raise ValueError("codifferential of a 0-form is zero")
    out: dict[tuple[int, ...], Any] = {}
    for ip in sorted_index_sets(d, k - 1):
        acc: Any = None
        for j in range(d):
            for m in range(d):
                cov: Any = _form_partial(state, form, (j,) + ip, m)
                for b in range(d):
                    w = _form_value(state, form, (b,) + ip)
                    if w is None:
                        continue
                    term = gamma[:, b, m, j] * w
                    cov = -term if cov is None else cov - term
                for s, i_s in enumerate(ip):
                    for b in range(d):
                        w = _form_value(
                            state, form, (j,) + ip[:s] + (b,) + ip[s + 1:],
                        )
                        if w is None:
                            continue
                        term = gamma[:, b, m, i_s] * w
                        cov = -term if cov is None else cov - term
                if cov is None:
                    continue
                contrib = -(ginv[:, j, m] * cov)
                acc = contrib if acc is None else acc + contrib
        if acc is not None:
            out[ip] = acc
    return out


__all__ = [
    "codifferential_components",
    "lie_derivative_components",
]

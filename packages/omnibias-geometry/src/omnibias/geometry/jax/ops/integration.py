# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Surface integration of differential forms over parametrized submanifolds (jax).

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.integration`. See that
module for the pullback + quadrature construction and the honesty label (exact
integrand, numerical-quadrature integral).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import value
from omnibias.fields.jax.ops.integral import quadrature_nodes
from omnibias.geometry._core.integration_core import pullback_form_components
from omnibias.geometry.jax.ops.pullback import pullback_metric

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.charts import ChartSpec
    from omnibias.geometry._core.forms import DifferentialForm

__all__ = [
    "integrate_form",
    "integrate_form_values",
    "surface_area",
    "surface_integral",
    "volume_element",
]


def _jacobian(chart: ChartSpec[Array], x: Array) -> Array:
    """Batched chart Jacobian ``d phi / d x`` of shape ``(Q, n, d)`` at param nodes ``x``."""
    jac: Array = jax.vmap(jax.jacfwd(chart.phi))(x)
    return jac


def _integrate_values(vals: Array, rule: QuadratureSpec) -> Array:
    r"""Quadrature contraction :math:`\sum_q w_q\,v_q` of a ``(Q, ...)`` integrand."""
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrand has {vals.shape[0]} points but rule has {rule.n_nodes} "
            "nodes; evaluate at quadrature_nodes(rule)"
        )
    w = jnp.asarray(rule.weights, dtype=vals.dtype)
    return jnp.tensordot(w, vals, axes=([0], [0]))


def volume_element(chart: ChartSpec[Array], rule: QuadratureSpec, *, like: Array) -> Array:
    r"""Riemannian volume element :math:`\sqrt{|\det g|}` of the pullback metric.

    Evaluated at the rule's parameter nodes, shape ``(Q,)``. ``like`` supplies the
    dtype for the nodes.
    """
    x = quadrature_nodes(rule, like=like)
    g = pullback_metric(x, chart)
    return jnp.sqrt(jnp.abs(jnp.linalg.det(g)))


def surface_area(chart: ChartSpec[Array], rule: QuadratureSpec, *, like: Array) -> Array:
    r"""Riemannian ``d``-volume of the image submanifold: :math:`\int_M \sqrt{|\det g|}\,dx`."""
    return _integrate_values(volume_element(chart, rule, like=like), rule)


def integrate_form_values(
    values: dict[tuple[int, ...], Array],
    degree: int,
    jacobian: Array,
    *,
    rule: QuadratureSpec,
) -> Array:
    r"""Integrate an already-evaluated top-degree form over a chart (low-level entry).

    ``values`` maps each sorted ambient ``degree``-index to its ``(Q,)`` component
    evaluated at the image points; ``jacobian`` is the ``(Q, n, d)`` chart Jacobian
    at the same parameter nodes. ``degree`` must equal the chart domain dimension.
    """
    ambient_dim = int(jacobian.shape[1])
    domain_dim = int(jacobian.shape[2])
    if degree != domain_dim:
        raise ValueError(
            f"integrate_form: form degree {degree} must equal the submanifold "
            f"(chart domain) dimension {domain_dim}"
        )
    pulled = pullback_form_components(values, jacobian, degree, domain_dim, ambient_dim)
    comp = pulled.get(tuple(range(domain_dim)))
    if comp is None:  # the form pulls back to zero over this chart
        comp = jacobian[:, 0, 0] * 0.0
    return _integrate_values(comp, rule)


def integrate_form(
    state: FieldState[Array],
    form: DifferentialForm,
    chart: ChartSpec[Array],
    rule: QuadratureSpec,
) -> Array:
    r"""Integrate a ``k``-form over the image of a ``k``-dimensional chart.

    ``state`` must be the field evaluated at the chart image points
    ``phi(quadrature_nodes(rule))``; ``form.degree`` must equal ``chart.domain_dim``
    and ``form.dim`` must equal ``chart.ambient_dim``.
    """
    if form.dim != chart.ambient_dim:
        raise ValueError(
            f"form.dim {form.dim} must equal chart.ambient_dim {chart.ambient_dim}"
        )
    if form.degree != chart.domain_dim:
        raise ValueError(
            f"integrate_form: form.degree {form.degree} must equal chart.domain_dim "
            f"{chart.domain_dim} (a top-degree form over the submanifold)"
        )
    if not form.comps:
        raise ValueError("integrate_form: form has no components to integrate")
    values = {idx: value(state, nm) for idx, nm in form.comps.items()}
    ref = next(iter(values.values()))
    if ref.shape[0] != rule.n_nodes:
        raise ValueError(
            "integrate_form: state must be evaluated at quadrature_nodes(rule)"
        )
    x = quadrature_nodes(rule, like=ref)
    jac = _jacobian(chart, x)
    return integrate_form_values(values, form.degree, jac, rule=rule)


def surface_integral(
    state: FieldState[Array],
    name: str,
    chart: ChartSpec[Array],
    rule: QuadratureSpec,
) -> Array:
    r"""Integrate a scalar field over the image submanifold: :math:`\int_M f\,dA`."""
    f = value(state, name)
    if f.shape[0] != rule.n_nodes:
        raise ValueError(
            "surface_integral: state must be evaluated at quadrature_nodes(rule)"
        )
    x = quadrature_nodes(rule, like=f)
    g = pullback_metric(x, chart)
    vol = jnp.sqrt(jnp.abs(jnp.linalg.det(g)))
    return _integrate_values(f * vol, rule)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Surface integration of differential forms over parametrized submanifolds (torch).

Bit-identical twin of :mod:`omnibias.geometry.jax.ops.integration`.

Given a :class:`~omnibias.geometry._core.charts.ChartSpec` immersion
:math:`\varphi:\mathbb R^d\to\mathbb R^n`, a ``d``-form ``omega`` on the ambient
space integrates over the image submanifold by pullback + quadrature,

.. math::

    \int_M \omega = \int_{\text{param box}} \varphi^*\omega,

with :math:`\varphi^*\omega` the top-degree component from
:func:`~omnibias.geometry._core.integration_core.pullback_form_components`
(the generalized Jacobian determinant). The metric companions
:func:`volume_element` / :func:`surface_area` / :func:`surface_integral` integrate
against the Riemannian volume element :math:`\sqrt{|\det g|}` of the pullback
metric :math:`g = J^\top h J`.

Honesty label
-------------
The *integrand* is **exact**: form components are closed-form field values /
derivatives and the Jacobian ``J = d phi / d x`` is exact forward-mode autodiff of
the (analytic or neural) chart. The *integral* is a **numerical quadrature**
(:class:`~omnibias.fields._core.quadrature.QuadratureSpec`) -- exact for
polynomials up to the rule degree and convergent otherwise -- matching the
existing Betti / Gauss-Bonnet framing in this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import value
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.geometry._core.integration_core import pullback_form_components
from omnibias.geometry.torch.ops.pullback import pullback_metric
from torch import Tensor
from torch.func import jacfwd, vmap

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


def _jacobian(chart: ChartSpec[Tensor], x: Tensor) -> Tensor:
    """Batched chart Jacobian ``d phi / d x`` of shape ``(Q, n, d)`` at param nodes ``x``."""
    jac: Tensor = vmap(jacfwd(chart.phi))(x)
    return jac


def _integrate_values(vals: Tensor, rule: QuadratureSpec) -> Tensor:
    r"""Quadrature contraction :math:`\sum_q w_q\,v_q` of a ``(Q, ...)`` integrand."""
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrand has {vals.shape[0]} points but rule has {rule.n_nodes} "
            "nodes; evaluate at quadrature_nodes(rule)"
        )
    w = torch.as_tensor(rule.weights, dtype=vals.dtype, device=vals.device)
    result: Tensor = torch.tensordot(w, vals, dims=([0], [0]))
    return result


def volume_element(chart: ChartSpec[Tensor], rule: QuadratureSpec, *, like: Tensor) -> Tensor:
    r"""Riemannian volume element :math:`\sqrt{|\det g|}` of the pullback metric.

    Evaluated at the rule's parameter nodes, shape ``(Q,)``. ``like`` supplies the
    dtype / device for the nodes. ``g = J^T h J`` is the pullback metric of
    ``chart`` (see :func:`~omnibias.geometry.torch.ops.pullback.pullback_metric`).
    """
    x = quadrature_nodes(rule, like=like)
    g = pullback_metric(x, chart)
    return torch.sqrt(torch.abs(torch.linalg.det(g)))


def surface_area(chart: ChartSpec[Tensor], rule: QuadratureSpec, *, like: Tensor) -> Tensor:
    r"""Riemannian ``d``-volume of the image submanifold: :math:`\int_M \sqrt{|\det g|}\,dx`."""
    return _integrate_values(volume_element(chart, rule, like=like), rule)


def integrate_form_values(
    values: dict[tuple[int, ...], Tensor],
    degree: int,
    jacobian: Tensor,
    *,
    rule: QuadratureSpec,
) -> Tensor:
    r"""Integrate an already-evaluated top-degree form over a chart (low-level entry).

    ``values`` maps each sorted ambient ``degree``-index to its ``(Q,)`` component
    evaluated at the image points; ``jacobian`` is the ``(Q, n, d)`` chart Jacobian
    at the same parameter nodes. ``degree`` must equal the chart domain dimension
    ``d`` (a form integrates over a submanifold of its own degree). Returns the
    scalar :math:`\int_M \omega`.
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
    state: FieldState[Tensor],
    form: DifferentialForm,
    chart: ChartSpec[Tensor],
    rule: QuadratureSpec,
) -> Tensor:
    r"""Integrate a ``k``-form over the image of a ``k``-dimensional chart.

    ``state`` must be the field evaluated at the chart image points
    ``phi(quadrature_nodes(rule))`` (its named components hold ``form``'s
    components); ``form.degree`` must equal ``chart.domain_dim`` and ``form.dim``
    must equal ``chart.ambient_dim``. Returns the scalar :math:`\int_M \omega`.
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
    state: FieldState[Tensor],
    name: str,
    chart: ChartSpec[Tensor],
    rule: QuadratureSpec,
) -> Tensor:
    r"""Integrate a scalar field over the image submanifold: :math:`\int_M f\,dA`.

    ``state`` holds the scalar component ``name`` evaluated at the chart image
    points; the area element ``dA = \sqrt{|\det g|}\,dx`` uses the pullback metric.
    """
    f = value(state, name)
    if f.shape[0] != rule.n_nodes:
        raise ValueError(
            "surface_integral: state must be evaluated at quadrature_nodes(rule)"
        )
    x = quadrature_nodes(rule, like=f)
    g = pullback_metric(x, chart)
    vol = torch.sqrt(torch.abs(torch.linalg.det(g)))
    return _integrate_values(f * vol, rule)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fredholm & Volterra integral-equation residuals (torch).

Every other equation in this package is **local**: a residual at ``x`` reads the
field and its derivatives at ``x`` and nowhere else. These two are not. An
integral equation of the second kind

.. math::

    u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t)

couples every point to every other, which changes what a residual evaluation
costs and what the network must be able to do -- it has to be evaluable at the
quadrature nodes, not only at the collocation points. That is free here, because
an omnibias field *is* a function: ``state.field(nodes)`` re-evaluates it
anywhere.

Two residuals, with an asymmetry worth understanding before choosing:

* :class:`Fredholm` integrates over a **fixed** domain, so one node evaluation is
  shared by the whole batch -- ``n_nodes`` extra field evaluations per residual,
  regardless of batch size.
* :class:`Volterra` integrates over :math:`[a, x]`, a domain that *moves with the
  collocation point*, so nothing can be shared: ``batch * n_nodes`` extra
  evaluations. It stays mesh-free by pulling each interval back to a reference
  one, :math:`t = a + (x - a)s`, and reusing a single fixed rule on
  :math:`s \in [0, 1]`, which is why it converges at the rule's own order instead
  of at the second order a cumulative-trapezoid grid would give.

Local terms (a source, a derivative in a coupled system) go through
``state.ops.*`` and stay exact closed form; only the integral is quadrature. Its
error is the measure's own, so a Gauss-Legendre measure on a smooth kernel is
spectrally accurate on a handful of nodes -- worth far more here than in a local
problem, since nodes are the expensive axis.

The solver-side twins live in :mod:`omnibias.measure.torch.integraleq`, which
solves the *discretised* equation directly. Reach for these residuals instead
when the equation is coupled to a PDE, the kernel is learned, or the solution is
wanted as a differentiable function rather than nodal values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import FredholmOutput, VolterraOutput
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.measure import Measure

#: A kernel ``K(x, t)`` called with collocation points ``(B, d)`` and quadrature
#: nodes ``(n, d)``, returning ``(B, n)``. May carry learnable parameters.
KernelFn = Callable[[Tensor, Tensor], Tensor]

#: A causal kernel ``K(x, t)`` for :class:`Volterra`, called with the collocation
#: points broadcast to ``(B, n, d)`` and the per-point nodes ``(B, n, d)``,
#: returning ``(B, n)``.
CausalKernelFn = Callable[[Tensor, Tensor], Tensor]


def _require_measure(measure: Any) -> Measure:
    """Fail early, and with the install line, if ``omnibias-measure`` is absent."""
    try:
        from omnibias.measure import Measure as _Measure
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by install
        raise ModuleNotFoundError(
            "the integral-equation residuals need the quadrature from "
            "omnibias-measure, which is an optional dependency: "
            "pip install 'omnibias-pinn[integral]'"
        ) from exc
    if not isinstance(measure, _Measure):
        raise TypeError(f"measure must be a Measure, got {type(measure).__name__}")
    return measure


def _nodes_and_weights(
    measure: Measure, reference: Tensor, *, expect_dim: int | None
) -> tuple[Tensor, Tensor]:
    """The measure's nodes and weights as tensors matching ``reference``."""
    if expect_dim is not None and measure.dim != expect_dim:
        raise ValueError(
            f"the measure lives in {measure.dim}D but the field's coordinates are "
            f"{expect_dim}D; the Fredholm integral runs over the whole domain, so "
            "the two must agree (use Measure.product for a multi-dimensional box)"
        )
    nodes = torch.as_tensor(
        measure.nodes, dtype=reference.dtype, device=reference.device
    )
    weights = torch.as_tensor(
        measure.weights, dtype=reference.dtype, device=reference.device
    )
    return nodes, weights


def _kernel_matrix(
    kernel: Callable[..., Tensor], left: Tensor, right: Tensor, shape: tuple[int, ...]
) -> Tensor:
    k = kernel(left, right)
    if k.shape != shape:
        raise ValueError(
            f"the kernel returned shape {tuple(k.shape)}, expected {shape} for "
            "(collocation points, quadrature nodes)"
        )
    return k


def fredholm_residual_samples(
    u: Tensor,
    u_nodes: Tensor,
    k: Tensor,
    weights: Tensor,
    *,
    lam: float | Tensor = 1.0,
    source: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fredholm residual and nonlocal term from already-sampled values.

    The arithmetic of :class:`Fredholm` with the field evaluation lifted out:
    ``u`` at the collocation points ``(B,)``, ``u_nodes`` at the quadrature nodes
    ``(n,)``, the kernel matrix ``k`` ``(B, n)`` and the quadrature ``weights``
    ``(n,)``. Separating it means the operator can be checked against an analytic
    solution without a field in the way, which is how the tests pin it down.

    Returns ``(residual, integral)``.
    """
    integral = (k * weights.unsqueeze(0)) @ u_nodes
    lam_t = torch.as_tensor(lam, dtype=u.dtype, device=u.device)
    residual = u - lam_t * integral
    if source is not None:
        residual = residual - source
    return residual, integral


def volterra_residual_samples(
    u: Tensor,
    u_nodes: Tensor,
    k: Tensor,
    weights: Tensor,
    span: Tensor,
    *,
    lam: float | Tensor = 1.0,
    source: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Volterra residual and causal term from already-sampled values.

    As :func:`fredholm_residual_samples`, but every array is per collocation
    point because the domain is: ``u_nodes`` and ``k`` are ``(B, n)`` over that
    point's own pullback nodes, and ``span`` ``(B,)`` is the interval length
    ``x - a`` that the change of variables :math:`t = a + (x-a)s` factors out.

    Returns ``(residual, integral)``.
    """
    integral = span * ((k * u_nodes) * weights.unsqueeze(0)).sum(dim=1)
    lam_t = torch.as_tensor(lam, dtype=u.dtype, device=u.device)
    residual = u - lam_t * integral
    if source is not None:
        residual = residual - source
    return residual, integral


@dataclass
class Fredholm:
    r"""Residual of ``u(x) - f(x) - lam * int_Omega K(x,t) u(t) dmu(t)``.

    Parameters
    ----------
    kernel
        ``K(x, t)`` mapping ``(B, d)`` collocation points and ``(n, d)``
        quadrature nodes to a ``(B, n)`` matrix. Free to carry learnable
        parameters -- the residual is differentiable through them.
    measure
        Quadrature for the integral. Its dimension must match the field's, since
        the integral runs over the whole domain. Gauss-Legendre (``lebesgue``) is
        the rule to want: spectral on a smooth kernel, so few nodes.
    lam
        The coupling :math:`\lambda`. May be a tensor, including a learnable one.
    component
        Field component playing :math:`u`. Default ``"u"``.
    source
        Optional ``f(state) -> (B,)``. Absent means the homogeneous equation.

    Notes
    -----
    Costs ``n_nodes`` extra field evaluations per residual -- independent of the
    batch, because the integration domain is fixed and the node values are shared.
    Honesty label: **numerical** in the integral (the measure's own quadrature
    error), exact closed form in every local term.
    """

    kernel: KernelFn
    measure: Measure
    lam: float | Tensor = 1.0
    component: str = "u"
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> FredholmOutput:
        measure = _require_measure(self.measure)
        u = state.ops.value(state, self.component)
        nodes, weights = _nodes_and_weights(
            measure, state.coords, expect_dim=state.coords.shape[1]
        )
        node_state = state.field(nodes)
        u_nodes = node_state.ops.value(node_state, self.component)
        k = _kernel_matrix(
            self.kernel, state.coords, nodes, (u.shape[0], nodes.shape[0])
        )
        residual, integral = fredholm_residual_samples(
            u,
            u_nodes,
            k,
            weights,
            lam=self.lam,
            source=None if self.source is None else self.source(state),
        )
        return FredholmOutput(
            residual=residual,
            integral=integral,
            diag={
                "mean_sq_residual": float((residual.detach() ** 2).mean()),
                "max_abs_residual": float(residual.detach().abs().max()),
            },
        )


def fredholm(
    state: FieldState,
    *,
    kernel: KernelFn,
    measure: Measure,
    lam: float | Tensor = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Tensor] | None = None,
) -> FredholmOutput:
    """Stateless one-shot wrapper around :class:`Fredholm`."""
    return Fredholm(
        kernel=kernel,
        measure=measure,
        lam=lam,
        component=component,
        source=source,
    )(state)


@dataclass
class Volterra:
    r"""Residual of ``u(x) - f(x) - lam * int_a^x K(x,t) u(t) dt``.

    The causal twin of :class:`Fredholm`: the upper limit is the collocation
    point, so this is the residual of a *hereditary* law -- the state at ``x``
    depends on its own history over ``[a, x]`` and on nothing ahead of it.

    Parameters
    ----------
    kernel
        ``K(x, t)`` mapping two ``(B, n, d)`` tensors -- the collocation points
        broadcast against the per-point nodes -- to ``(B, n)``.
    measure
        A **1-D** rule on the reference interval :math:`[0, 1]`; each interval
        :math:`[a, x_i]` is pulled back to it. Not a rule on the problem domain,
        which is what makes this mesh-free.
    origin
        The lower limit :math:`a`.
    axis
        Name of the causal axis. Every other coordinate is held at the
        collocation point's own value, so in a space-time problem this is a
        memory term :math:`\int_0^t K(t,s)\, u(x, s)\, ds` at fixed ``x``.
    lam, component, source
        As in :class:`Fredholm`.

    Notes
    -----
    Costs ``batch * n_nodes`` extra field evaluations per residual: the
    integration domain moves with the collocation point, so unlike
    :class:`Fredholm` nothing can be shared across the batch. The pullback
    :math:`t = a + (x - a)s` buys the reference rule's own convergence order
    rather than the second order of a cumulative-trapezoid grid, which is what
    makes a small ``n_nodes`` viable -- and small ``n_nodes`` is the only thing
    that makes the cost viable.

    Honesty label: **numerical** in the integral, exact closed form elsewhere.
    """

    kernel: CausalKernelFn
    measure: Measure
    origin: float = 0.0
    axis: str | None = None
    lam: float | Tensor = 1.0
    component: str = "u"
    source: Callable[[FieldState], Tensor] | None = None

    def _causal_axis(self, state: FieldState) -> str:
        if self.axis is not None:
            return self.axis
        spec = state.coordinate_spec
        if spec.time_axis is not None:
            return str(spec.time_axis)
        if len(spec.axes) == 1:
            return str(spec.axes[0])
        raise ValueError(
            "Volterra needs to know which axis is causal: the field has axes "
            f"{spec.axes!r} and no time axis, so pass axis=... explicitly"
        )

    def __call__(self, state: FieldState) -> VolterraOutput:
        measure = _require_measure(self.measure)
        if measure.dim != 1:
            raise ValueError(
                f"the reference measure must be 1-D, got {measure.dim}D: it "
                "discretises the pullback variable s in [0, 1], not the domain"
            )
        axis = self._causal_axis(state)
        axis_i = state.coordinate_spec.axis_index(axis)
        coords = state.coords
        batch, dim = coords.shape
        u = state.ops.value(state, self.component)

        s, weights = _nodes_and_weights(measure, coords, expect_dim=None)
        s = s[:, 0]
        span = coords[:, axis_i] - self.origin

        # t_ij = a + (x_i - a) s_j, with every other coordinate frozen at x_i:
        # the causal integral is along one axis of a point's own history.
        nodes = coords.unsqueeze(1).expand(batch, s.shape[0], dim).clone()
        nodes[:, :, axis_i] = self.origin + span.unsqueeze(1) * s.unsqueeze(0)

        flat = nodes.reshape(-1, dim)
        node_state = state.field(flat)
        u_nodes = node_state.ops.value(node_state, self.component).reshape(
            batch, s.shape[0]
        )
        k = _kernel_matrix(
            self.kernel,
            coords.unsqueeze(1).expand_as(nodes),
            nodes,
            (batch, s.shape[0]),
        )
        residual, integral = volterra_residual_samples(
            u,
            u_nodes,
            k,
            weights,
            span,
            lam=self.lam,
            source=None if self.source is None else self.source(state),
        )
        return VolterraOutput(
            residual=residual,
            integral=integral,
            diag={
                "mean_sq_residual": float((residual.detach() ** 2).mean()),
                "max_abs_residual": float(residual.detach().abs().max()),
            },
        )


def volterra(
    state: FieldState,
    *,
    kernel: CausalKernelFn,
    measure: Measure,
    origin: float = 0.0,
    axis: str | None = None,
    lam: float | Tensor = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Tensor] | None = None,
) -> VolterraOutput:
    """Stateless one-shot wrapper around :class:`Volterra`."""
    return Volterra(
        kernel=kernel,
        measure=measure,
        origin=origin,
        axis=axis,
        lam=lam,
        component=component,
        source=source,
    )(state)


__all__ = [
    "CausalKernelFn",
    "Fredholm",
    "KernelFn",
    "Volterra",
    "fredholm",
    "fredholm_residual_samples",
    "volterra",
    "volterra_residual_samples",
]

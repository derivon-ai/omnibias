# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fredholm & Volterra integral-equation residuals (jax twin).

Bit-parity companion of :mod:`omnibias.pinn.torch.equations.integral`, which
carries the full contract. In brief, these are the two **nonlocal** residuals of
the equation of the second kind
:math:`u(x) = f(x) + \lambda \int K(x, t) u(t)\, d\mu(t)`: :class:`Fredholm` over
a fixed domain (one shared node evaluation for the whole batch) and
:class:`Volterra` over the moving domain :math:`[a, x]` (``batch * n_nodes``
evaluations, kept mesh-free by the pullback :math:`t = a + (x - a)s` onto a single
reference rule).

The integral is the only quadrature-approximated quantity; local terms stay exact
closed form through ``state.ops.*``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import FredholmOutput, VolterraOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.measure import Measure

#: A kernel ``K(x, t)`` from ``(B, d)`` points and ``(n, d)`` nodes to ``(B, n)``.
KernelFn = Callable[[Array, Array], Array]

#: A causal kernel ``K(x, t)`` from two ``(B, n, d)`` tensors to ``(B, n)``.
CausalKernelFn = Callable[[Array, Array], Array]


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
    measure: Measure, reference: Array, *, expect_dim: int | None
) -> tuple[Array, Array]:
    if expect_dim is not None and measure.dim != expect_dim:
        raise ValueError(
            f"the measure lives in {measure.dim}D but the field's coordinates are "
            f"{expect_dim}D; the Fredholm integral runs over the whole domain, so "
            "the two must agree (use Measure.product for a multi-dimensional box)"
        )
    nodes = jnp.asarray(measure.nodes, dtype=reference.dtype)
    weights = jnp.asarray(measure.weights, dtype=reference.dtype)
    return nodes, weights


def _kernel_matrix(
    kernel: Callable[..., Array], left: Array, right: Array, shape: tuple[int, ...]
) -> Array:
    k = kernel(left, right)
    if k.shape != shape:
        raise ValueError(
            f"the kernel returned shape {k.shape}, expected {shape} for "
            "(collocation points, quadrature nodes)"
        )
    return k


def fredholm_residual_samples(
    u: Array,
    u_nodes: Array,
    k: Array,
    weights: Array,
    *,
    lam: float | Array = 1.0,
    source: Array | None = None,
) -> tuple[Array, Array]:
    """Fredholm residual and nonlocal term from already-sampled values (jax)."""
    integral = (k * weights[None, :]) @ u_nodes
    residual = u - jnp.asarray(lam, dtype=u.dtype) * integral
    if source is not None:
        residual = residual - source
    return residual, integral


def volterra_residual_samples(
    u: Array,
    u_nodes: Array,
    k: Array,
    weights: Array,
    span: Array,
    *,
    lam: float | Array = 1.0,
    source: Array | None = None,
) -> tuple[Array, Array]:
    """Volterra residual and causal term from already-sampled values (jax)."""
    integral = span * jnp.sum((k * u_nodes) * weights[None, :], axis=1)
    residual = u - jnp.asarray(lam, dtype=u.dtype) * integral
    if source is not None:
        residual = residual - source
    return residual, integral


@dataclass
class Fredholm:
    """Residual of ``u(x) - f(x) - lam * int_Omega K(x,t) u(t) dmu(t)`` (jax)."""

    kernel: KernelFn
    measure: Measure
    lam: float | Array = 1.0
    component: str = "u"
    source: Callable[[FieldState], Array] | None = None

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
                "mean_sq_residual": jnp.mean(residual * residual),
                "max_abs_residual": jnp.max(jnp.abs(residual)),
            },
        )


def fredholm(
    state: FieldState,
    *,
    kernel: KernelFn,
    measure: Measure,
    lam: float | Array = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Array] | None = None,
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
    """Residual of ``u(x) - f(x) - lam * int_a^x K(x,t) u(t) dt`` (jax)."""

    kernel: CausalKernelFn
    measure: Measure
    origin: float = 0.0
    axis: str | None = None
    lam: float | Array = 1.0
    component: str = "u"
    source: Callable[[FieldState], Array] | None = None

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

        # t_ij = a + (x_i - a) s_j, with every other coordinate frozen at x_i.
        nodes = jnp.broadcast_to(coords[:, None, :], (batch, s.shape[0], dim))
        nodes = nodes.at[:, :, axis_i].set(
            self.origin + span[:, None] * s[None, :]
        )

        node_state = state.field(nodes.reshape(-1, dim))
        u_nodes = node_state.ops.value(node_state, self.component).reshape(
            batch, s.shape[0]
        )
        k = _kernel_matrix(
            self.kernel,
            jnp.broadcast_to(coords[:, None, :], nodes.shape),
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
                "mean_sq_residual": jnp.mean(residual * residual),
                "max_abs_residual": jnp.max(jnp.abs(residual)),
            },
        )


def volterra(
    state: FieldState,
    *,
    kernel: CausalKernelFn,
    measure: Measure,
    origin: float = 0.0,
    axis: str | None = None,
    lam: float | Array = 1.0,
    component: str = "u",
    source: Callable[[FieldState], Array] | None = None,
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

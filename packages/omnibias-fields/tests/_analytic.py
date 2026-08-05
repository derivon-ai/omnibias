# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reusable separable-polynomial analytic field for op tests.

A component ``f(x_0, ..., x_{D-1}) = prod_d p_d(x_d)`` is a product of per-axis
polynomials, so value / derivative / mixed-partial all have exact closed forms
and the *same* pure-Python arithmetic runs bit-identically on torch and jax. The
field uses the state-method dispatch path (marker ``"spectral"``), so it exercises
the foundational ops without depending on omnibias-pinn.

This generalises the 2-D fixture in ``conftest.py`` to arbitrary axes (including a
time axis), which the vector-calculus / conservation / wave ops need.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState


class Poly:
    """A 1-D polynomial ``sum_k c[k] x^k`` with exact derivatives."""

    def __init__(self, coeffs: Sequence[float]) -> None:
        self.coeffs = tuple(float(c) for c in coeffs)

    def value(self, x):  # type: ignore[no-untyped-def]
        acc = None
        for k, c in enumerate(self.coeffs):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def _deriv_coeffs(self, order: int) -> tuple[float, ...]:
        c = list(self.coeffs)
        for _ in range(order):
            c = [k * c[k] for k in range(1, len(c))] if len(c) > 1 else [0.0]
        return tuple(c) if c else (0.0,)

    def deriv(self, x, order: int):  # type: ignore[no-untyped-def]
        dc = self._deriv_coeffs(order)
        acc = None
        for k, c in enumerate(dc):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def squared(self) -> Poly:
        """Coefficients of ``p(x)^2`` (still separable, for independent checks)."""
        n = len(self.coeffs)
        out = [0.0] * (2 * n - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(self.coeffs):
                out[i + j] += a * b
        return Poly(out)


class AnalyticField:
    """Separable-polynomial field implementing the state-method op path."""

    _omnibias_dispatch = "spectral"

    def __init__(self, coordinate_spec, components, comp_polys, ops_module):  # type: ignore[no-untyped-def]
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._polys = comp_polys
        self._ops = ops_module

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    def __call__(self, coords):  # type: ignore[no-untyped-def]
        return self.evaluate(coords)

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x = state.coords
        acc = None
        for d, poly in enumerate(self._polys[name]):
            term = poly.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x = state.coords
        acc = None
        for d, poly in enumerate(self._polys[name]):
            term = poly.deriv(x[:, d], order) if d == axis else poly.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        order_by_axis = dict(zip(axes, orders, strict=False))
        x = state.coords
        acc = None
        for d, poly in enumerate(self._polys[name]):
            o = order_by_axis.get(d, 0)
            term = poly.deriv(x[:, d], o) if o > 0 else poly.value(x[:, d])
            acc = term if acc is None else acc * term
        return acc


def make_field(
    backend: str,
    axes: Sequence[str],
    comp_polys: dict[str, tuple[Poly, ...]],
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
    time_axis: str | None | object = ...,
) -> AnalyticField:
    """Build an :class:`AnalyticField` for ``"torch"`` or ``"jax"``."""
    if backend == "torch":
        from omnibias.fields.torch import _ops_dispatch as ops
    elif backend == "jax":
        from omnibias.fields.jax import _ops_dispatch as ops
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown backend {backend!r}")
    coord_spec = CoordinateSpec(tuple(axes), time_axis=time_axis)
    components = ComponentSpec(tuple(comp_polys), groups=groups)
    return AnalyticField(coord_spec, components, comp_polys, ops)

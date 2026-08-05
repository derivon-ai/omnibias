# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared test fixtures: a separable-polynomial analytic field.

Each component is a product of per-axis polynomials, so value, derivatives,
gradient, Hessian, integral, and norms all have closed-form references. The same
pure-Python ``_Poly`` arithmetic runs on torch and jax tensors, so the two
backends are bit-identical by construction. The field uses the state-method
dispatch path (marker ``"spectral"``), exercising the foundational ops without
depending on omnibias-pinn.

The suite also runs in double precision on both backends -- see
:func:`_double_precision_default` for why torch's half of that is a fixture
rather than an import-time set.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

try:  # enable float64 in jax for the cross-backend bit-parity tests
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover - jax optional
    pass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState


class _Poly:
    """A 1D polynomial ``sum_k c[k] x^k`` with exact derivative and integral."""

    def __init__(self, coeffs: Sequence[float]) -> None:
        self.coeffs = tuple(float(c) for c in coeffs)

    def value(self, x):  # type: ignore[no-untyped-def]
        acc = None
        for k, c in enumerate(self.coeffs):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def deriv_coeffs(self, order: int) -> tuple[float, ...]:
        c = list(self.coeffs)
        for _ in range(order):
            c = [k * c[k] for k in range(1, len(c))] if len(c) > 1 else [0.0]
        return tuple(c) if c else (0.0,)

    def deriv(self, x, order: int):  # type: ignore[no-untyped-def]
        dc = self.deriv_coeffs(order)
        acc = None
        for k, c in enumerate(dc):
            term = c * x**k
            acc = term if acc is None else acc + term
        return acc

    def integral(self, lo: float, hi: float) -> float:
        total = 0.0
        for k, c in enumerate(self.coeffs):
            total += c / (k + 1) * (hi ** (k + 1) - lo ** (k + 1))
        return total


class AnalyticField:
    """Separable-polynomial field implementing the state-method op path."""

    _omnibias_dispatch = "spectral"

    def __init__(self, coordinate_spec, components, comp_polys, ops_module):  # type: ignore[no-untyped-def]
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._polys = comp_polys  # dict[name -> tuple[_Poly, ...]]
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

    # --- state-method op surface --------------------------------------
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

    def integral_component(self, name, bounds) -> float:
        total = 1.0
        for d, poly in enumerate(self._polys[name]):
            total *= poly.integral(bounds[d][0], bounds[d][1])
        return total


# Component polynomial table shared by both backends.
#   u(x, y) = (1 + 2x + x^2) * (1 - y)
#   v(x, y) = (x) * (1 + y + y^2)
_COMP_POLYS = {
    "u": (_Poly((1.0, 2.0, 1.0)), _Poly((1.0, -1.0))),
    "v": (_Poly((0.0, 1.0)), _Poly((1.0, 1.0, 1.0))),
}


def _coordinate_spec() -> CoordinateSpec:
    # Two spatial axes, no time axis.
    return CoordinateSpec(("x", "y"), time_axis=None)


def _component_spec() -> ComponentSpec:
    return ComponentSpec(("u", "v"), groups={"vec": ("u", "v")})


@pytest.fixture(autouse=True)
def _double_precision_default() -> Iterator[None]:
    """Run every test in this suite with torch defaulting to ``float64``.

    The cross-backend parity tests compare torch against a jax side that
    ``jax_enable_x64`` has already put in double precision, and the elasticity /
    MHD / kinetic modules need the headroom for their finite-difference and
    autodiff references.

    Setting it here rather than at a test module's import time is what makes it
    survive collection order. ``torch.set_default_dtype`` is a process-global
    mutation, so an import-time set lands during collection and can then be
    reverted by another suite's own dtype fixture before these tests actually
    run -- which is exactly what happens when this package and omnibias-torch
    share one pytest session.
    """
    if torch is None:  # pragma: no cover - torch optional
        yield
        return
    prev = torch.get_default_dtype()
    if prev is not torch.float64:
        torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        if torch.get_default_dtype() is not prev:
            torch.set_default_dtype(prev)


@pytest.fixture
def analytic_polys():  # type: ignore[no-untyped-def]
    return _COMP_POLYS


@pytest.fixture
def make_torch_field():  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    def _make() -> AnalyticField:
        return AnalyticField(
            _coordinate_spec(), _component_spec(), _COMP_POLYS, _ops_dispatch,
        )

    return _make


@pytest.fixture
def make_jax_field():  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch

    def _make() -> AnalyticField:
        return AnalyticField(
            _coordinate_spec(), _component_spec(), _COMP_POLYS, _ops_dispatch,
        )

    return _make


@pytest.fixture
def grid_nodes():  # type: ignore[no-untyped-def]
    """A deterministic (B, 2) numpy point cloud for pointwise op checks."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(32, 2)).astype(np.float64)


# --- spacetime (x, y, t) field for axis-scope regression tests -----------
#   u(x, y, t) = (1 + 2x + x^2) * (1 - y) * (1 + t + t^2)
# The degree-2 time factor makes d^2u/dt^2 and the mixed space-time second
# derivatives nonzero, so a spatial-only Hessian differs from the full one.
_COMP_POLYS_SPACETIME = {
    "u": (_Poly((1.0, 2.0, 1.0)), _Poly((1.0, -1.0)), _Poly((1.0, 1.0, 1.0))),
}


def _coordinate_spec_spacetime() -> CoordinateSpec:
    return CoordinateSpec(("x", "y", "t"), time_axis="t")


def _component_spec_spacetime() -> ComponentSpec:
    return ComponentSpec(("u",))


@pytest.fixture
def make_spacetime_torch_field():  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    def _make() -> AnalyticField:
        return AnalyticField(
            _coordinate_spec_spacetime(),
            _component_spec_spacetime(),
            _COMP_POLYS_SPACETIME,
            _ops_dispatch,
        )

    return _make


@pytest.fixture
def make_spacetime_jax_field():  # type: ignore[no-untyped-def]
    from omnibias.fields.jax import _ops_dispatch

    def _make() -> AnalyticField:
        return AnalyticField(
            _coordinate_spec_spacetime(),
            _component_spec_spacetime(),
            _COMP_POLYS_SPACETIME,
            _ops_dispatch,
        )

    return _make

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for :class:`FieldState` and the attribute DSL views.

These tests use a *stub ops module* to verify that the view delegates
correctly to ``state.ops.*`` regardless of which backend is active.
"""

from __future__ import annotations

import types

import pytest
from omnibias.pinn import (
    ComponentSpec,
    ComponentView,
    CoordinateSpec,
    FieldState,
    SigmaCache,
    VectorView,
)


def _make_stub_ops():
    """Return a stub ops module that records every call."""
    calls: list[tuple[str, tuple, dict]] = []
    mod = types.ModuleType("stub_ops")

    def make(name):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return f"{name}({args}, {kwargs})"
        return fn

    for n in (
        "value",
        "derivative",
        "gradient",
        "laplacian",
        "hessian",
        "spatial_hessian",
        "gradient_of_derivative",
        "biharmonic",
        "polylaplacian",
        "p_laplacian",
        "directional_derivative",
        "mixed_partial",
        "stack_components",
        "jacobian",
        "spatial_jacobian",
        "divergence",
        "curl",
        "vector_laplacian",
        "vector_biharmonic",
        "vector_hessian",
        "vector_polylaplacian",
        "strain_rate",
        "deformation_gradient",
        "vector_derivative",
        "advection",
        "material_derivative",
    ):
        setattr(mod, n, make(n))
    mod.list_ops = lambda: tuple([])  # type: ignore[attr-defined]
    return mod, calls


class _StubField:
    def __init__(self, components, coordinate_spec):
        self.components = components
        self.coordinate_spec = coordinate_spec

    def evaluate(self, coords):
        return None  # not used here

    def __call__(self, coords):
        return self.evaluate(coords)


def _make_state(components=None, coords_dim=3):
    if components is None:
        components = ComponentSpec(
            ("u", "v", "w", "p"),
            groups={"velocity": ("u", "v", "w")},
        )
    coord_spec = CoordinateSpec(("x", "y", "z", "t")[:coords_dim] + ("t",))
    field = _StubField(components, coord_spec)
    ops, calls = _make_stub_ops()
    state = FieldState(
        coords=("coords-tensor",),  # opaque to the schema
        field=field,
        components=components,
        coordinate_spec=coord_spec,
        ops=ops,
        sigma_cache=SigmaCache(z=("z-tensor",)),
    )
    return state, calls


def test_state_attribute_dispatch_to_component_view():
    state, _ = _make_state()
    cv = state.u
    assert isinstance(cv, ComponentView)
    assert cv.name == "u"


def test_state_attribute_dispatch_to_vector_view():
    state, _ = _make_state()
    vv = state.velocity
    assert isinstance(vv, VectorView)
    assert vv.names == ("u", "v", "w")


def test_state_did_you_mean():
    state, _ = _make_state()
    with pytest.raises(AttributeError) as ei:
        _ = state.velcty  # typo
    msg = str(ei.value)
    assert "velocity" in msg


def test_state_contains_and_getitem():
    state, _ = _make_state()
    assert "u" in state
    assert "velocity" in state
    assert "phi" not in state
    assert isinstance(state["u"], ComponentView)
    assert isinstance(state["velocity"], VectorView)


def test_state_repr():
    state, _ = _make_state()
    r = repr(state)
    assert "FieldState" in r
    assert "_StubField" in r


def test_state_is_frozen():
    state, _ = _make_state()
    with pytest.raises(AttributeError):
        state.coords = ("nope",)


def test_state_extra_is_mutable():
    state, _ = _make_state()
    state.extra["my_intermediate"] = 42  # should not raise
    assert state.extra["my_intermediate"] == 42


def test_component_view_dispatches():
    state, calls = _make_state()
    _ = state.u.value
    _ = state.u.dt
    _ = state.u.dx
    _ = state.u.lap
    _ = state.u.grad
    _ = state.u.hess
    _ = state.u.hess_spatial
    _ = state.u.biharm
    _ = state.u.d("y", order=2)
    _ = state.u.grad_of_d("x")
    _ = state.u.dn(("x", "y"), (1, 1))
    _ = state.u.polylap(2)

    op_names = [c[0] for c in calls]
    assert "value" in op_names
    assert "derivative" in op_names
    assert "laplacian" in op_names
    assert "gradient" in op_names
    assert "hessian" in op_names
    assert "spatial_hessian" in op_names
    assert "gradient_of_derivative" in op_names
    assert "biharmonic" in op_names
    assert "mixed_partial" in op_names
    assert "polylaplacian" in op_names

    # derivative was called with the right axis tag for `dt`/`dx`
    deriv_calls = [c for c in calls if c[0] == "derivative"]
    axes = [c[2].get("axis") for c in deriv_calls]
    assert "t" in axes
    assert "x" in axes


def test_vector_view_dispatches():
    state, calls = _make_state()
    _ = state.velocity.value
    _ = state.velocity.div
    _ = state.velocity.curl
    _ = state.velocity.lap
    _ = state.velocity.jac
    _ = state.velocity.spatial_jac
    _ = state.velocity.hess
    _ = state.velocity.polylap(2)
    _ = state.velocity.strain_rate
    _ = state.velocity.deformation_gradient
    _ = state.velocity.dt
    _ = state.velocity.advect()
    _ = state.velocity.material_derivative()

    op_names = [c[0] for c in calls]
    assert "stack_components" in op_names
    assert "divergence" in op_names
    assert "curl" in op_names
    assert "vector_laplacian" in op_names
    assert "jacobian" in op_names
    assert "spatial_jacobian" in op_names
    assert "vector_hessian" in op_names
    assert "vector_polylaplacian" in op_names
    assert "strain_rate" in op_names
    assert "deformation_gradient" in op_names
    assert "vector_derivative" in op_names
    assert "advection" in op_names
    assert "material_derivative" in op_names


def test_vector_view_iteration():
    state, _ = _make_state()
    vv = state.velocity
    members = list(vv)
    assert all(isinstance(m, ComponentView) for m in members)
    assert [m.name for m in members] == ["u", "v", "w"]
    assert len(vv) == 3
    assert vv[0].name == "u"
    assert vv["v"].name == "v"
    with pytest.raises(KeyError):
        vv["q"]


def test_vector_view_advect_by_cross_state_raises():
    state_a, _ = _make_state()
    state_b, _ = _make_state()
    with pytest.raises(ValueError):
        state_a.velocity.advect_by(state_b.velocity)


def test_vector_view_alias_vort_equals_curl():
    state, calls = _make_state()
    _ = state.velocity.vort
    _ = state.velocity.curl
    op_names = [c[0] for c in calls]
    assert op_names.count("curl") == 2


def test_sigma_cache_lazy_computation():
    cache = SigmaCache(z="z-tensor")
    n_calls = [0]

    def build(order):
        n_calls[0] += 1
        return f"sigma^{order}"

    a = cache.get_or_compute(2, build)
    b = cache.get_or_compute(2, build)
    c = cache.get_or_compute(4, build)
    assert a == "sigma^2"
    assert a is b
    assert c == "sigma^4"
    assert n_calls[0] == 2  # only two builds for two distinct orders
    assert 2 in cache and 4 in cache
    assert cache.orders() == (2, 4)


def test_sigma_cache_negative_order_raises():
    cache = SigmaCache(z="z")
    with pytest.raises(ValueError):
        cache.get_or_compute(-1, lambda n: n)


def test_state_values_property_calls_ops_value_per_component():
    state, calls = _make_state()
    out = state.values
    op_names = [c[0] for c in calls]
    assert op_names.count("value") == 4
    assert set(out) == {"u", "v", "w", "p"}

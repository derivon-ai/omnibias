# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact tests for the planar angular-momentum operators L_z and L_z^2.

The eigenfunctions ``(x + i y)^m`` satisfy ``L_z psi = m hbar psi`` and
``L_z^2 psi = m^2 hbar^2 psi``. We build an additive-separable polynomial
wavefunction (so value / derivative / mixed-partial are all exact closed
forms, bit-identical on torch and jax) and check those eigenvalues directly,
plus an independent recomposition and torch<->jax parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.qpinn._core.complex import (
    apply_angular_momentum_squared,
    apply_angular_momentum_z,
)


class _Poly:
    """1-D polynomial ``sum_k c[k] x^k`` with exact derivatives."""

    def __init__(self, coeffs):  # type: ignore[no-untyped-def]
        self.coeffs = tuple(float(c) for c in coeffs)

    def value(self, x):  # type: ignore[no-untyped-def]
        acc = 0.0 * x
        for k, c in enumerate(self.coeffs):
            acc = acc + c * x**k
        return acc

    def deriv(self, x, order):  # type: ignore[no-untyped-def]
        c = list(self.coeffs)
        for _ in range(order):
            c = [k * c[k] for k in range(1, len(c))] if len(c) > 1 else [0.0]
        acc = 0.0 * x
        for k, cc in enumerate(c):
            acc = acc + cc * x**k
        return acc


class _AdditiveField:
    """Component = sum of separable terms ``prod_d poly_d(x_d)`` (non-separable)."""

    _omnibias_dispatch = "spectral"

    def __init__(self, coordinate_spec, components, comp_terms, ops_module):  # type: ignore[no-untyped-def]
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._terms = comp_terms  # name -> list[tuple[_Poly, ...]]
        self._ops = ops_module

    def __call__(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        x = state.coords
        total = 0.0 * x[:, 0]
        for polys in self._terms[name]:
            term = None
            for d, poly in enumerate(polys):
                t = poly.value(x[:, d])
                term = t if term is None else term * t
            total = total + term
        return total

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        x = state.coords
        total = 0.0 * x[:, 0]
        for polys in self._terms[name]:
            term = None
            for d, poly in enumerate(polys):
                t = poly.deriv(x[:, d], order) if d == axis else poly.value(x[:, d])
                term = t if term is None else term * t
            total = total + term
        return total

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        order_by_axis = dict(zip(axes, orders, strict=False))
        x = state.coords
        total = 0.0 * x[:, 0]
        for polys in self._terms[name]:
            term = None
            for d, poly in enumerate(polys):
                o = order_by_axis.get(d, 0)
                t = poly.deriv(x[:, d], o) if o > 0 else poly.value(x[:, d])
                term = t if term is None else term * t
            total = total + term
        return total


def _zpow_terms(m: int):
    """Real / imaginary additive-separable terms of ``(x + i y)^m``."""
    re_terms, im_terms = [], []
    from math import comb

    for k in range(m + 1):
        coeff = comb(m, k)  # binomial; i^k handles the phase
        xpow = m - k
        ypow = k
        px = _Poly([0.0] * xpow + [1.0])
        py_coeffs = [0.0] * ypow + [1.0]
        phase = (1j) ** k
        if phase.imag == 0:  # real contribution
            c = coeff * phase.real
            re_terms.append((_scaled(px, c), _Poly(py_coeffs)))
        else:  # imaginary contribution
            c = coeff * phase.imag
            im_terms.append((_scaled(px, c), _Poly(py_coeffs)))
    return re_terms, im_terms


def _scaled(poly: _Poly, c: float) -> _Poly:
    return _Poly([c * a for a in poly.coeffs])


def _ops(backend):  # type: ignore[no-untyped-def]
    if backend == "torch":
        from omnibias.fields.torch import _ops_dispatch as ops
    else:
        from omnibias.fields.jax import _ops_dispatch as ops
    return ops


def _coords(backend, arr):  # type: ignore[no-untyped-def]
    if backend == "torch":
        return torch.as_tensor(arr, dtype=torch.float64)
    return jnp.asarray(arr, dtype=jnp.float64)


def _np(x):  # type: ignore[no-untyped-def]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _zpow_field(backend, m):  # type: ignore[no-untyped-def]
    re_terms, im_terms = _zpow_terms(m)
    coord = CoordinateSpec(("x", "y"), time_axis=None)
    comp = ComponentSpec(("psi_re", "psi_im"), groups={"psi": ("psi_re", "psi_im")})
    terms = {"psi_re": re_terms, "psi_im": im_terms}
    return _AdditiveField(coord, comp, terms, _ops(backend))


def _nodes(seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(20, 2)).astype(np.float64)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_Lz_eigenvalue(m):
    nodes = _nodes()
    st = _zpow_field("torch", m)(_coords("torch", nodes))
    lz_re, lz_im = apply_angular_momentum_z(st, group="psi", hbar=1.0)
    psi_re = _np(st.ops.value(st, "psi_re"))
    psi_im = _np(st.ops.value(st, "psi_im"))
    assert np.allclose(_np(lz_re), m * psi_re, rtol=1e-11, atol=1e-11)
    assert np.allclose(_np(lz_im), m * psi_im, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_Lz_squared_eigenvalue(m):
    nodes = _nodes()
    st = _zpow_field("torch", m)(_coords("torch", nodes))
    l2_re, l2_im = apply_angular_momentum_squared(st, group="psi", hbar=1.0)
    psi_re = _np(st.ops.value(st, "psi_re"))
    psi_im = _np(st.ops.value(st, "psi_im"))
    assert np.allclose(_np(l2_re), m**2 * psi_re, rtol=1e-11, atol=1e-11)
    assert np.allclose(_np(l2_im), m**2 * psi_im, rtol=1e-11, atol=1e-11)


def test_Lz_recomposition_matches_inline_formula():
    """Guards the rotating-NLS refactor: op == hand-rolled x d_y - y d_x form."""
    nodes = _nodes(seed=3)
    st = _zpow_field("torch", 3)(_coords("torch", nodes))
    lz_re, lz_im = apply_angular_momentum_z(st, group="psi", hbar=1.0)
    x = _np(st.coords[:, 0])
    y = _np(st.coords[:, 1])
    dx_re = _np(st.ops.derivative(st, "psi_re", axis=0, order=1))
    dy_re = _np(st.ops.derivative(st, "psi_re", axis=1, order=1))
    dx_im = _np(st.ops.derivative(st, "psi_im", axis=0, order=1))
    dy_im = _np(st.ops.derivative(st, "psi_im", axis=1, order=1))
    assert np.allclose(_np(lz_re), x * dy_im - y * dx_im, rtol=1e-12, atol=1e-12)
    assert np.allclose(_np(lz_im), -(x * dy_re - y * dx_re), rtol=1e-12, atol=1e-12)


def test_hbar_scaling_and_axis_validation():
    nodes = _nodes()
    st = _zpow_field("torch", 2)(_coords("torch", nodes))
    a_re, _ = apply_angular_momentum_z(st, hbar=1.0)
    b_re, _ = apply_angular_momentum_z(st, hbar=2.5)
    assert np.allclose(_np(b_re), 2.5 * _np(a_re), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="must differ"):
        apply_angular_momentum_z(st, x_axis=0, y_axis=0)
    with pytest.raises(ValueError, match="must differ"):
        apply_angular_momentum_squared(st, x_axis=1, y_axis=1)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_cross_backend_parity(m):
    nodes = _nodes(seed=5)
    ts = _zpow_field("torch", m)(_coords("torch", nodes))
    js = _zpow_field("jax", m)(_coords("jax", nodes))
    tz = apply_angular_momentum_z(ts, hbar=1.3)
    jz = apply_angular_momentum_z(js, hbar=1.3)
    assert np.allclose(_np(tz[0]), _np(jz[0]), rtol=1e-12, atol=1e-12)
    assert np.allclose(_np(tz[1]), _np(jz[1]), rtol=1e-12, atol=1e-12)
    t2 = apply_angular_momentum_squared(ts, hbar=1.3)
    j2 = apply_angular_momentum_squared(js, hbar=1.3)
    assert np.allclose(_np(t2[0]), _np(j2[0]), rtol=1e-12, atol=1e-12)
    assert np.allclose(_np(t2[1]), _np(j2[1]), rtol=1e-12, atol=1e-12)

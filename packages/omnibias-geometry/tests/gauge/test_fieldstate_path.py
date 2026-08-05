# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The closed-form-derivative showcase: F, D_mu F^{mu nu} from a FieldState.

A separable analytic su(2) connection is built as a :class:`FieldState`; the
gauge ops obtain ``d_mu A_nu`` and ``d_rho d_mu A_nu`` from the omnibias
closed-form sigma-tower (no autodiff / finite differences), and the result must
match the same quantities assembled from the analytic derivatives.
"""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.geometry.gauge._core.connection import (
    connection_component_names,
    gauge_connection_spec,
)

SU2 = la.su(2)
DIM = 4
COUPLING = 0.85


class _Cos:
    """``amp * cos(freq * x + phase)`` with exact derivatives on any array module."""

    def __init__(self, amp: float, freq: float, phase: float) -> None:
        self.amp, self.freq, self.phase = amp, freq, phase

    def value(self, xp, x):
        return self.amp * xp.cos(self.freq * x + self.phase)

    def deriv(self, xp, x, order: int):
        amp = self.amp * (self.freq**order)
        ph = self.phase + order * (np.pi / 2.0)
        return amp * xp.cos(self.freq * x + ph)


def _axis_funcs(name: str) -> tuple[_Cos, ...]:
    # deterministic distinct separable factors per component name
    h = abs(hash(name)) % 997
    return tuple(
        _Cos(amp=0.2 + 0.03 * ((h + d) % 5), freq=0.4 + 0.07 * ((h + 2 * d) % 7), phase=0.1 * ((h + d) % 9))
        for d in range(DIM)
    )


class AnalyticConnectionField:
    _omnibias_dispatch = "spectral"

    def __init__(self, names: tuple[str, ...], ops_module) -> None:
        self.coordinate_spec = CoordinateSpec(tuple(f"x{d}" for d in range(DIM)), time_axis=None)
        self.components = ComponentSpec(names)
        self._axes = {nm: _axis_funcs(nm) for nm in names}
        self._ops = ops_module

    def evaluate(self, coords):
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def value_component(self, state, name):
        xp = _xp_of(state.coords)
        acc = None
        for d, ax in enumerate(self._axes[name]):
            term = ax.value(xp, state.coords[:, d])
            acc = term if acc is None else acc * term
        return acc

    def derivative(self, state, name, *, axis, order):
        xp = _xp_of(state.coords)
        acc = None
        for d, ax in enumerate(self._axes[name]):
            term = ax.deriv(xp, state.coords[:, d], order) if d == axis else ax.value(xp, state.coords[:, d])
            acc = term if acc is None else acc * term
        return acc

    def mixed_partial(self, state, name, axes, orders):
        xp = _xp_of(state.coords)
        ob = dict(zip(axes, orders, strict=False))
        acc = None
        for d, ax in enumerate(self._axes[name]):
            o = ob.get(d, 0)
            term = ax.deriv(xp, state.coords[:, d], o) if o > 0 else ax.value(xp, state.coords[:, d])
            acc = term if acc is None else acc * term
        return acc


def _xp_of(coords):
    mod = type(coords).__module__
    if mod.startswith("torch"):
        import torch

        return torch
    import jax.numpy as jnp

    return jnp


def _analytic_arrays(field: AnalyticConnectionField, names, coords_np: np.ndarray):
    """Build A, dA, ddA analytically (numpy) for cross-checking the field path."""
    B = coords_np.shape[0]
    n = SU2.dim
    name_grid = [[names[mu * n + a] for a in range(n)] for mu in range(DIM)]

    def comp_value(nm, deriv_axes=()):
        orders = {ax: deriv_axes.count(ax) for ax in set(deriv_axes)}
        acc = np.ones(B)
        for d, ax in enumerate(field._axes[nm]):
            o = orders.get(d, 0)
            acc = acc * (ax.deriv(np, coords_np[:, d], o) if o > 0 else ax.value(np, coords_np[:, d]))
        return acc

    A = np.zeros((B, DIM, n))
    dA = np.zeros((B, DIM, DIM, n))
    ddA = np.zeros((B, DIM, DIM, DIM, n))
    for mu in range(DIM):
        for a in range(n):
            nm = name_grid[mu][a]
            A[:, mu, a] = comp_value(nm)
            for rho in range(DIM):
                dA[:, rho, mu, a] = comp_value(nm, (rho,))
                for sig in range(DIM):
                    ddA[:, rho, sig, mu, a] = comp_value(nm, (rho, sig))
    return A, dA, ddA


def _setup(backend):
    if backend.name == "torch":
        from omnibias.fields.torch import _ops_dispatch
    else:
        from omnibias.fields.jax import _ops_dispatch
    names = connection_component_names(
        gauge_connection_spec(SU2, coupling=COUPLING, spacetime_dim=DIM)
    )
    field = AnalyticConnectionField(names, _ops_dispatch)
    conn = gauge_connection_spec(SU2, coupling=COUPLING, spacetime_dim=DIM)
    rng = np.random.default_rng(21)
    coords_np = rng.uniform(-1.5, 1.5, size=(16, DIM))
    coords = backend.asarray(coords_np)
    state = field(coords)
    return field, conn, names, coords_np, state


def test_field_strength_via_fieldstate_matches_analytic(backend) -> None:
    field, conn, names, coords_np, state = _setup(backend)
    A, dA, _ = _analytic_arrays(field, names, coords_np)
    F_state = backend.tonumpy(backend.ops.field_strength(state, conn))
    F_arr = backend.tonumpy(
        backend.ops.field_strength_from_arrays(
            backend.asarray(A), backend.asarray(dA), algebra=SU2, coupling=COUPLING
        )
    )
    np.testing.assert_allclose(F_state, F_arr, rtol=1e-9, atol=1e-10)


def test_yang_mills_operator_via_fieldstate_matches_analytic(backend) -> None:
    field, conn, names, coords_np, state = _setup(backend)
    A, dA, ddA = _analytic_arrays(field, names, coords_np)
    eom_state = backend.tonumpy(backend.ops.covariant_divergence(state, conn))
    eom_arr = backend.tonumpy(
        backend.ops.covariant_divergence_from_arrays(
            backend.asarray(A), backend.asarray(dA), backend.asarray(ddA),
            algebra=SU2, coupling=COUPLING, signature=(1, 1, 1, 1),
        )
    )
    np.testing.assert_allclose(eom_state, eom_arr, rtol=1e-8, atol=1e-9)

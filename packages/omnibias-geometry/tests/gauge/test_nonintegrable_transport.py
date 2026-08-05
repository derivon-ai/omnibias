# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite nonintegrability: parallel transport / Wilson line / holonomy.

The Wu-Yang nonintegrable phase factor ``U(C) = P exp(-i g \int_C A_mu^a T^a dx)``.
Checks the abelian Aharonov-Bohm phase, path-ordered composition, unitarity, the
non-abelian Stokes small-loop limit (holonomy -> closed-form ``field_strength``),
Wilson-loop gauge invariance, and torch <-> jax agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from _gauge_helpers import instanton_arrays
from omnibias.geometry.gauge._core.connection import (
    connection_component_names,
    gauge_connection_spec,
)
from omnibias.geometry.gauge._core.lie_algebra import su, u1
from test_fieldstate_path import AnalyticConnectionField

DIM = 4


def test_abelian_aharonov_bohm(backend) -> None:
    """U(1) straight-line transport is the closed-form phase ``exp(-i g (1/sqrt2) c L)``."""
    alg = u1()
    g, n_steps, length, amp = 0.9, 64, 1.3, 0.7
    a_path = np.zeros((n_steps, DIM, 1))
    a_path[:, 0, 0] = amp  # constant A_0^0 = amp
    tangents = np.zeros((n_steps, DIM))
    tangents[:, 0] = length  # r'(t) = length * e_0
    u = backend.ops.parallel_transport_from_arrays(
        backend.asarray(a_path), backend.asarray(tangents),
        algebra=alg, coupling=g, dt=1.0 / n_steps,
    )
    expected = np.exp(-1j * g * (1.0 / np.sqrt(2.0)) * amp * length)
    assert abs(complex(backend.tonumpy(u)[0, 0]) - expected) < 1e-10


def test_ordered_exponential_composition(backend) -> None:
    """``U(0->1) = U(0->1/2) @ U(1/2->1)`` -- earliest-parameter segment on the left."""
    alg = su(2)
    g, n_steps = 0.6, 50
    rng = np.random.default_rng(5)
    a_path = rng.normal(size=(n_steps, DIM, alg.dim)) * 0.3
    tangents = rng.normal(size=(n_steps, DIM))
    dt = 1.0 / n_steps
    half = n_steps // 2

    def transport(a: np.ndarray, t: np.ndarray) -> np.ndarray:
        return backend.tonumpy(
            backend.ops.parallel_transport_from_arrays(
                backend.asarray(a), backend.asarray(t), algebra=alg, coupling=g, dt=dt
            )
        )

    u_full = transport(a_path, tangents)
    u_first = transport(a_path[:half], tangents[:half])
    u_second = transport(a_path[half:], tangents[half:])
    np.testing.assert_allclose(u_full, u_first @ u_second, rtol=1e-9, atol=1e-11)


def test_zero_connection_and_unitarity(backend) -> None:
    alg = su(2)
    n_steps = 20
    zero = np.zeros((n_steps, DIM, alg.dim))
    ones = np.ones((n_steps, DIM))
    u0 = backend.tonumpy(
        backend.ops.parallel_transport_from_arrays(
            backend.asarray(zero), backend.asarray(ones), algebra=alg, coupling=1.0, dt=0.1
        )
    )
    np.testing.assert_allclose(u0, np.eye(2), atol=1e-12)

    rng = np.random.default_rng(9)
    a_path = rng.normal(size=(n_steps, DIM, alg.dim)) * 0.5
    tangents = rng.normal(size=(n_steps, DIM))
    u = backend.tonumpy(
        backend.ops.parallel_transport_from_arrays(
            backend.asarray(a_path), backend.asarray(tangents),
            algebra=alg, coupling=1.0, dt=1.0 / n_steps,
        )
    )
    dim = u.shape[-1]
    np.testing.assert_allclose(u.conj().T @ u, np.eye(dim), atol=1e-10)


def _circle_curve(backend, radius: float, center: np.ndarray):
    """A small circle in the ``(0, 1)`` plane, centered at ``center`` (backend callable)."""
    two_pi = 2.0 * np.pi
    if backend.name == "torch":
        import torch

        def curve(tt: torch.Tensor) -> torch.Tensor:
            th = two_pi * tt[..., 0]
            x0 = radius * torch.cos(th) + center[0]
            x1 = radius * torch.sin(th) + center[1]
            x2 = torch.full_like(th, center[2])
            x3 = torch.full_like(th, center[3])
            return torch.stack([x0, x1, x2, x3], dim=-1)

        return curve

    import jax.numpy as jnp

    def curve(tt):  # type: ignore[no-untyped-def]
        th = two_pi * tt[..., 0]
        x0 = radius * jnp.cos(th) + center[0]
        x1 = radius * jnp.sin(th) + center[1]
        x2 = jnp.full_like(th, center[2])
        x3 = jnp.full_like(th, center[3])
        return jnp.stack([x0, x1, x2, x3], axis=-1)

    return curve


def test_nonabelian_stokes_convergence(backend) -> None:
    """Holonomy of a shrinking loop reproduces the closed-form ``field_strength``.

    Non-abelian Stokes: ``(i / (g * area)) (U - I) -> F_{01}`` as ``area -> 0``. Uses
    the strong BPST instanton (``|F| ~ 1``) so the ``g[A, A]`` term is genuinely
    exercised; the residual is first order in the radius, so it halves when the
    radius halves.
    """
    alg = su(2)
    g = 1.0
    center = np.array([0.5, 0.2, 0.1, 0.3])
    substeps = 800

    a_c, da_c, _ = instanton_arrays(center[None, :])
    fld = backend.ops.field_strength_from_arrays(
        backend.asarray(a_c), backend.asarray(da_c), algebra=alg, coupling=g
    )
    f01 = backend.tonumpy(backend.ops.to_matrix(fld[:, 0, 1], alg))[0]

    errors = []
    for radius in (0.08, 0.04, 0.02):
        t = (np.arange(substeps) + 0.5) / substeps
        th = 2.0 * np.pi * t
        pts = np.stack(
            [radius * np.cos(th) + center[0], radius * np.sin(th) + center[1],
             np.full_like(th, center[2]), np.full_like(th, center[3])], axis=-1,
        )
        a_path = instanton_arrays(pts)[0]
        tangents = radius * 2.0 * np.pi * np.stack(
            [-np.sin(th), np.cos(th), np.zeros_like(th), np.zeros_like(th)], axis=-1
        )
        holonomy = backend.tonumpy(
            backend.ops.parallel_transport_from_arrays(
                backend.asarray(a_path), backend.asarray(tangents),
                algebra=alg, coupling=g, dt=1.0 / substeps,
            )
        )
        area = np.pi * radius**2
        dim = holonomy.shape[-1]
        f_extract = (1j / (g * area)) * (holonomy - np.eye(dim))
        errors.append(float(np.abs(f_extract - f01).max()))

    # first-order convergence: error roughly halves as the loop radius halves.
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] == pytest.approx(2.0, abs=0.2)
    assert errors[2] < 1e-2 * np.abs(f01).max()


def test_fieldstate_transport_matches_field_strength(backend) -> None:
    """Full ``parallel_transport(state, conn, curve)`` path reproduces ``field_strength``.

    Exercises the FieldState wrapper (closed-form connection value along the curve
    + autodiff tangent) end to end on a small loop.
    """
    alg = su(2)
    coupling = 0.8
    conn = gauge_connection_spec(alg, coupling=coupling, spacetime_dim=DIM)
    if backend.name == "torch":
        from omnibias.fields.torch import _ops_dispatch
    else:
        from omnibias.fields.jax import _ops_dispatch
    field = AnalyticConnectionField(connection_component_names(conn), _ops_dispatch)

    center = np.array([0.1, -0.05, 0.03, 0.02])
    radius, substeps = 0.05, 400
    like = backend.asarray(np.zeros(1))

    state_center = field(backend.asarray(center[None, :]))
    fld = backend.ops.field_strength(state_center, conn)
    f01 = backend.tonumpy(backend.ops.to_matrix(fld[:, 0, 1], alg))[0]

    curve = _circle_curve(backend, radius, center)
    mids = backend.ops.transport_nodes(0.0, 1.0, substeps, like=like)
    state = field(curve(mids))
    holonomy = backend.tonumpy(
        backend.ops.parallel_transport(state, conn, curve, t0=0.0, t1=1.0, substeps=substeps)
    )
    area = np.pi * radius**2
    dim = holonomy.shape[-1]
    f_extract = (1j / (coupling * area)) * (holonomy - np.eye(dim))
    np.testing.assert_allclose(f_extract, f01, atol=1e-3)


def test_wilson_loop_gauge_invariance(backend) -> None:
    """A global gauge transform conjugates the holonomy, leaving its trace invariant."""
    alg = su(2)
    g, n_steps = 0.7, 60
    rng = np.random.default_rng(13)
    a_path = rng.normal(size=(n_steps, DIM, alg.dim)) * 0.4
    tangents = rng.normal(size=(n_steps, DIM))
    dt = 1.0 / n_steps
    angle = 0.9
    v = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  # SO(2) in SU(2)

    a_t = backend.ops.global_gauge_transform(
        backend.asarray(a_path), backend.asarray(v), algebra=alg
    )
    u1_mat = backend.tonumpy(
        backend.ops.parallel_transport_from_arrays(
            backend.asarray(a_path), backend.asarray(tangents), algebra=alg, coupling=g, dt=dt
        )
    )
    u2_mat = backend.tonumpy(
        backend.ops.parallel_transport_from_arrays(
            a_t, backend.asarray(tangents), algebra=alg, coupling=g, dt=dt
        )
    )
    tr1 = np.trace(u1_mat).real / u1_mat.shape[-1]
    tr2 = np.trace(u2_mat).real / u2_mat.shape[-1]
    assert abs(tr1 - tr2) < 1e-9


def test_transport_cross_backend() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.geometry.gauge.jax.ops as jops
    import omnibias.geometry.gauge.torch.ops as tops

    torch.set_default_dtype(torch.float64)
    alg = su(3)
    g, n_steps = 0.6, 40
    rng = np.random.default_rng(21)
    a_path = rng.normal(size=(n_steps, DIM, alg.dim)) * 0.2
    tangents = rng.normal(size=(n_steps, DIM))
    dt = 1.0 / n_steps

    t = tops.parallel_transport_from_arrays(
        torch.as_tensor(a_path), torch.as_tensor(tangents), algebra=alg, coupling=g, dt=dt
    )
    j = jops.parallel_transport_from_arrays(
        jnp.asarray(a_path), jnp.asarray(tangents), algebra=alg, coupling=g, dt=dt
    )
    np.testing.assert_allclose(t.detach().numpy(), np.asarray(j), rtol=1e-9, atol=1e-11)

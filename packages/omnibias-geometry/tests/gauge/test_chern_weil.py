# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Chern-Weil characteristic numbers: c1 (abelian), c2, Pontryagin, Euler.

Oracles:

- Dirac monopole of charge ``n`` on ``S^2`` -> first Chern number ``c1 = n``
  (an integer), with a **verified** :class:`~omnibias.core.verified.Interval`
  enclosure that brackets exactly one integer (integer-quantisation certificate).
- BPST instanton -> second Chern number ``c2 = 1`` and Pontryagin number
  ``p1 = -2 c2 = -2``.
- ``euler_number_so2`` is the ``c1`` alias; ``first_chern_class = F/(2 pi)``.
- torch vs jax parity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _gauge_helpers import thooft_eta
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.quadrature import midpoint_integral

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Dirac monopole: abelian first Chern number
# --------------------------------------------------------------------------- #
def _monopole_field_strength(backend, nodes, charge):  # type: ignore[no-untyped-def]
    r"""``F_{theta phi} = (n/2) sin(theta)`` on the ``(theta, phi)`` chart."""
    theta = nodes[:, 0]
    b = nodes.shape[0]
    f01 = 0.5 * charge * np.sin(theta)
    F = np.zeros((b, 2, 2), dtype=np.float64)
    F[:, 0, 1] = f01
    F[:, 1, 0] = -f01
    return backend.asarray(F)


def _gl_2d(n):  # type: ignore[no-untyped-def]
    from omnibias.fields._core.quadrature import gauss_legendre

    return gauss_legendre([(0.0, math.pi), (0.0, TWO_PI)], n)


@pytest.mark.parametrize("charge", [1, 2, -1])
def test_monopole_first_chern_number(backend, charge) -> None:  # type: ignore[no-untyped-def]
    rule = _gl_2d(24)
    F = _monopole_field_strength(backend, rule.nodes, charge)
    w = backend.asarray(rule.weights)
    c1 = backend.ops.first_chern_number(F, plane=(0, 1), weights=w)
    assert backend.tonumpy(c1) == pytest.approx(float(charge), abs=1e-6)
    assert backend.ops.is_quantized(c1)


def test_first_chern_class_is_F_over_2pi(backend) -> None:  # type: ignore[no-untyped-def]
    rule = _gl_2d(8)
    F = _monopole_field_strength(backend, rule.nodes, 1)
    c1_form = backend.ops.first_chern_class(F)
    np.testing.assert_allclose(
        backend.tonumpy(c1_form), backend.tonumpy(F) / TWO_PI, rtol=1e-12, atol=1e-12,
    )


def test_euler_number_so2_is_first_chern(backend) -> None:  # type: ignore[no-untyped-def]
    rule = _gl_2d(20)
    F = _monopole_field_strength(backend, rule.nodes, 2)
    w = backend.asarray(rule.weights)
    euler = backend.ops.euler_number_so2(F, plane=(0, 1), weights=w)
    c1 = backend.ops.first_chern_number(F, plane=(0, 1), weights=w)
    assert backend.tonumpy(euler) == pytest.approx(backend.tonumpy(c1), abs=1e-12)


# --------------------------------------------------------------------------- #
# Verified integer-quantisation certificate for the monopole c1
# --------------------------------------------------------------------------- #
def _unique_integer(iv: Interval) -> int | None:
    """The unique integer inside ``[lo, hi]``, else ``None``."""
    k_lo = math.ceil(iv.lo)
    k_hi = math.floor(iv.hi)
    return k_lo if k_lo == k_hi else None


@pytest.mark.parametrize("charge", [1, 2, 3])
def test_monopole_chern_number_verified_integer(charge) -> None:
    r"""Enclose ``c1 = int_0^pi (n/2) sin(theta) dtheta`` and bracket one integer.

    The ``phi`` integral contributes a factor ``2 pi`` that cancels the ``1/2 pi``
    normalisation, so ``c1`` equals the ``theta`` integral of ``(n/2) sin(theta)``.
    """
    a, b = 0.0, math.pi
    n_panels = 200
    h = (b - a) / n_panels
    mids = [a + (k + 0.5) * h for k in range(n_panels)]
    values = [Interval.point(0.5 * charge * math.sin(t)) for t in mids]
    # f'' = -(n/2) sin(theta) in [-(|n|/2), |n|/2] over [0, pi]
    amp = 0.5 * abs(charge)
    m2 = Interval(-amp, amp)
    enclosure = midpoint_integral(values, a, b, m2)
    assert enclosure.contains(float(charge))
    assert _unique_integer(enclosure) == charge


# --------------------------------------------------------------------------- #
# BPST instanton: second Chern number c2 = 1, Pontryagin p1 = -2
# --------------------------------------------------------------------------- #
def _instanton_grid_F(grid: np.ndarray, rho: float = 1.0) -> np.ndarray:
    eta = np.transpose(thooft_eta(), (1, 2, 0))  # (mu, nu, a)
    d = (grid**2).sum(1) + rho**2
    return -4.0 * rho**2 * eta[None] / (d[:, None, None, None] ** 2)


def _instanton_c2_and_p1(ops, asarray, tonumpy):  # type: ignore[no-untyped-def]
    lo, hi, npts = -6.0, 6.0, 24
    axis = np.linspace(lo, hi, npts)
    cell = ((hi - lo) / (npts - 1)) ** 4
    grid = np.stack(np.meshgrid(*([axis] * 4), indexing="ij"), -1).reshape(-1, 4)
    c2 = p1 = 0.0
    for chunk in np.array_split(grid, 64):
        F = asarray(_instanton_grid_F(chunk))
        c2 += float(tonumpy(ops.topological_charge_density(F)).sum()) * cell
        p1 += float(tonumpy(ops.pontryagin_density(F)).sum()) * cell
    return c2, p1


def test_instanton_pontryagin_number(backend) -> None:  # type: ignore[no-untyped-def]
    c2, p1 = _instanton_c2_and_p1(backend.ops, backend.asarray, backend.tonumpy)
    assert c2 == pytest.approx(1.0, abs=0.03)
    assert p1 == pytest.approx(-2.0, abs=0.06)


def test_pontryagin_density_is_minus_two_c2(backend) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(11)
    pts = rng.uniform(-2.0, 2.0, size=(32, 4))
    F = backend.asarray(_instanton_grid_F(pts))
    p1 = backend.tonumpy(backend.ops.pontryagin_density(F))
    c2 = backend.tonumpy(backend.ops.topological_charge_density(F))
    np.testing.assert_allclose(p1, -2.0 * c2, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Cross-backend parity
# --------------------------------------------------------------------------- #
def test_chern_weil_cross_backend() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.geometry.gauge.jax.ops as jops
    import omnibias.geometry.gauge.torch.ops as tops

    rule = _gl_2d(16)
    F = np.zeros((rule.n_nodes, 2, 2), dtype=np.float64)
    F[:, 0, 1] = 0.5 * np.sin(rule.nodes[:, 0])
    F[:, 1, 0] = -F[:, 0, 1]
    wt = torch.as_tensor(rule.weights, dtype=torch.float64)
    wj = jnp.asarray(rule.weights, dtype=jnp.float64)
    c1t = float(tops.first_chern_number(torch.as_tensor(F, dtype=torch.float64), weights=wt))
    c1j = float(jops.first_chern_number(jnp.asarray(F, dtype=jnp.float64), weights=wj))
    assert c1t == pytest.approx(c1j, abs=1e-12)

    pts = np.random.default_rng(5).uniform(-2.0, 2.0, size=(24, 4))
    Fi = _instanton_grid_F(pts)
    p1t = float(tops.pontryagin_number(torch.as_tensor(Fi, dtype=torch.float64)))
    p1j = float(jops.pontryagin_number(jnp.asarray(Fi, dtype=jnp.float64)))
    assert p1t == pytest.approx(p1j, abs=1e-11)

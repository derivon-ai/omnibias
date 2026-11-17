# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Declared-demand residuals: engines read what a residual declares and refuse what they cannot provide."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.demand import Demand  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    Quantities,
    Residual,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
    legacy_full_gradient,
    legacy_projected,
    masked_residual,
    pad_block,
    projected_residual_gram,
)


def _layers(seed, D, H, depth=2):
    rng = np.random.default_rng(seed)
    dims = [D] + [H] * depth + [1]
    return [
        (
            jnp.asarray(rng.standard_normal((b, a)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
            "tanh" if i < depth else None,
        )
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
    ]


def _points(seed, B, D):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((B, D)) * 0.6)


def test_undeclared_quantities_are_hidden() -> None:
    seen = {}

    def fn(q: Quantities):
        seen.update({name: getattr(q, name) for name in Quantities._fields})
        return q.u[:, 0]

    res = Residual(fn, ["lap"])
    u = jnp.ones((3, 1))
    res(
        Quantities(
            u=u,
            x=jnp.ones((3, 2)),
            grad=jnp.ones((3, 2, 1)),
            grad_sq=u,
            lap=u,
            bilap=u,
            directional=u,
        )
    )
    assert seen["lap"] is not None
    assert all(seen[name] is None for name in ("grad", "grad_sq", "bilap", "directional"))
    assert res.order == 2 and res.demand == Demand.of("lap")


def test_declared_residual_matches_legacy_callback_bitwise() -> None:
    layers = _layers(0, 4, 6)
    x, xb = _points(1, 5, 4), _points(2, 3, 4)
    target = jnp.asarray(np.random.default_rng(3).standard_normal(5))

    def old(u, g, lap, x):
        return lap[:, 0] - jnp.sum(g[:, :, 0] ** 2, axis=1) + target

    new = Residual(
        lambda q: q.lap[:, 0] - jnp.sum(q.grad[:, :, 0] ** 2, axis=1) + target, ["grad", "lap"]
    )
    old_b = Residual(lambda q: q.u[:, 0], [])
    r0, K0 = deep_residual_gram([(x, old), (xb, lambda u, g, lap, x: u[:, 0], 0)], layers)
    r1, K1 = deep_residual_gram([(x, new), (xb, old_b)], layers)
    np.testing.assert_array_equal(np.asarray(r0), np.asarray(r1))
    np.testing.assert_array_equal(np.asarray(K0), np.asarray(K1))
    # wrappers give the same program as the raw callback
    r2, K2 = deep_residual_gram([(x, legacy_full_gradient(old))], layers)
    r3, K3 = deep_residual_gram([(x, old)], layers)
    np.testing.assert_array_equal(np.asarray(K2), np.asarray(K3))


def test_grad_sq_is_served_by_both_engines() -> None:
    D, H = 9, 4
    layers = _layers(4, D, H)
    x = _points(5, 6, D)
    diffusion = jnp.asarray(np.random.default_rng(6).uniform(0.5, 1.5, D))
    res = Residual(lambda q: q.lap[:, 0] - q.grad_sq[:, 0] + q.u[:, 0] - 0.3, ["grad_sq", "lap"])
    r_full, K_full = deep_residual_gram([(x, res)], layers, diffusion=diffusion)
    r_proj, K_proj = projected_residual_gram([(x, res)], layers, diffusion=diffusion)
    np.testing.assert_allclose(np.asarray(r_proj), np.asarray(r_full), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(K_proj), np.asarray(K_full), rtol=1e-9, atol=1e-11)
    d_r, d_full = deep_residual_ntk_diagonal([(x, res)], layers, diffusion=diffusion)
    np.testing.assert_allclose(
        np.asarray(d_full), np.diag(np.asarray(K_full)), rtol=1e-9, atol=1e-11
    )


def test_projected_engine_refuses_the_full_gradient_and_full_engine_refuses_bilap() -> None:
    layers = _layers(7, 6, 4)
    x = _points(8, 4, 6)
    needs_grad = Residual(lambda q: q.grad[:, 0, 0], ["grad"])
    with pytest.raises(ValueError, match="projected engine cannot provide"):
        projected_residual_gram([(x, needs_grad)], layers)
    needs_bilap = Residual(lambda q: q.bilap[:, 0], ["bilap"])
    with pytest.raises(ValueError, match="full engine cannot provide"):
        deep_residual_gram([(x, needs_bilap)], layers)
    with pytest.raises(ValueError, match="cannot provide"):
        deep_residual_gram(
            [(x, Residual(lambda q: q.directional[:, 0, 0], ["directional"]))], layers
        )


def test_order_is_inferred_and_validated() -> None:
    layers = _layers(9, 5, 4)
    x = _points(10, 4, 5)
    boundary = Residual(lambda q: q.u[:, 0] - 1.0, [])  # order 0: no gradient or Laplacian streams
    r, K = deep_residual_gram([(x, boundary)], layers)
    assert K.shape == (4, 4)
    lap = Residual(lambda q: q.lap[:, 0], ["lap"])
    with pytest.raises(ValueError, match="asks for order"):
        deep_residual_gram([(x, lap, 1)], layers)
    with pytest.raises(ValueError, match="serves blocks of order"):
        deep_residual_gram([(x, Residual(lambda q: q.lap[:, 0], ["lap"]), 4)], layers)
    four = Residual(lambda q: q.bilap[:, 0] + q.lap[:, 0] - 0.2, ["lap", "bilap"])
    r4, K4 = projected_residual_gram([(x, four)], layers)
    assert K4.shape == (4, 4) and np.all(np.isfinite(np.asarray(K4)))


def test_fourth_order_declared_matches_legacy_projected() -> None:
    layers = _layers(11, 8, 3)
    x = _points(12, 4, 8)

    def old(u, gs, lap, bil, dv, x):
        return bil[:, 0] + lap[:, 0] + 0.5 * u[:, 0] * gs[:, 0]

    new = Residual(
        lambda q: q.bilap[:, 0] + q.lap[:, 0] + 0.5 * q.u[:, 0] * q.grad_sq[:, 0],
        ["grad_sq", "lap", "bilap"],
    )
    _, K_old = projected_residual_gram([(x, old, 4)], layers)
    _, K_new = projected_residual_gram([(x, new)], layers)
    _, K_wrap = projected_residual_gram([(x, legacy_projected(old, order=4))], layers)
    np.testing.assert_array_equal(np.asarray(K_old), np.asarray(K_new))
    np.testing.assert_array_equal(np.asarray(K_old), np.asarray(K_wrap))


def test_directional_demand_reaches_the_residual() -> None:
    layers = _layers(13, 7, 3)
    x = _points(14, 4, 7)
    V = jnp.asarray(np.random.default_rng(15).standard_normal((7, 2)))
    res = Residual(
        lambda q: q.lap[:, 0] + q.directional[:, 0, 0] + q.directional[:, 1, 0],
        ["lap", "directional"],
    )

    def old(u, gs, lap, dv, x):
        return lap[:, 0] + dv[:, 0, 0] + dv[:, 1, 0]

    _, K_old = projected_residual_gram([(x, old)], layers, directions=V)
    _, K_new = projected_residual_gram([(x, res)], layers, directions=V)
    np.testing.assert_array_equal(np.asarray(K_old), np.asarray(K_new))


def test_masked_residual_keeps_the_demand_and_padding_is_exact() -> None:
    layers = _layers(16, 4, 5)
    x = _points(17, 5, 4)
    target = jnp.asarray(np.random.default_rng(18).standard_normal(5))
    res = Residual(lambda q: q.lap[:, 0] - target_pad[: q.u.shape[0]], ["lap"])
    target_pad = target
    r, K = deep_residual_gram([(x, res)], layers)
    xp, mask, tp = pad_block(x, target, bucket=8)
    target_pad = tp
    padded = masked_residual(res, mask)
    assert isinstance(padded, Residual) and padded.demand == res.demand
    rp, Kp = deep_residual_gram([(xp, padded)], layers)
    np.testing.assert_allclose(np.asarray(rp[:5]), np.asarray(r), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(Kp[:5, :5]), np.asarray(K), rtol=1e-11, atol=1e-12)
    assert float(jnp.max(jnp.abs(Kp[5:]))) == 0.0 and float(jnp.max(jnp.abs(rp[5:]))) == 0.0


def test_legacy_wrappers_validate_order() -> None:
    with pytest.raises(ValueError, match="order must be"):
        legacy_full_gradient(lambda *a: a[0], order=4)
    with pytest.raises(ValueError, match="order must be"):
        legacy_projected(lambda *a: a[0], order=3)
    assert legacy_projected(lambda *a: a[0], order=4, directional=True).demand == Demand.of(
        "grad_sq", "lap", "bilap", "directional"
    )
    with pytest.raises(TypeError):
        Residual("not callable", [])

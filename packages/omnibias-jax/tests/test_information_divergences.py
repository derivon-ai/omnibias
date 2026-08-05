# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generalized divergences + optimal-transport extensions (JAX) and parity.

Truth oracles, the analytic limit relations between families (Renyi -> KL,
Tsallis/Renyi entropy -> Shannon, f-divergence specialisations), enclosure by
the certified core, differentiability, and bit-identical torch parity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.information import (  # noqa: E402
    chi_squared_divergence,
    entropy,
    f_divergence,
    hellinger_distance,
    kl_divergence,
    renyi_divergence,
    renyi_entropy,
    sinkhorn_distance,
    sliced_wasserstein,
    total_variation_distance,
    tsallis_entropy,
    wasserstein1,
    wasserstein2_gaussian,
    wassersteinp,
)

_P = jnp.asarray([0.5, 0.3, 0.2])
_Q = jnp.asarray([0.2, 0.5, 0.3])


# ----- generalized divergences: truths --------------------------------------


def test_total_variation_truth_and_self_zero() -> None:
    assert float(total_variation_distance(_P, _Q)) == pytest.approx(
        0.5 * float(jnp.abs(_P - _Q).sum())
    )
    assert float(total_variation_distance(_P, _P)) == pytest.approx(0.0, abs=1e-15)


def test_hellinger_truth_self_zero_and_bhattacharyya() -> None:
    bc = float(jnp.sqrt(_P * _Q).sum())
    assert float(hellinger_distance(_P, _Q)) == pytest.approx(math.sqrt(1.0 - bc), rel=1e-12)
    assert float(hellinger_distance(_P, _P)) == pytest.approx(0.0, abs=1e-12)
    disjoint = float(hellinger_distance(jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])))
    assert disjoint == pytest.approx(1.0, rel=1e-12)


def test_chi_squared_truth_and_self_zero() -> None:
    truth = float((((_P - _Q) ** 2) / _Q).sum())
    assert float(chi_squared_divergence(_P, _Q)) == pytest.approx(truth, rel=1e-12)
    assert float(chi_squared_divergence(_P, _P)) == pytest.approx(0.0, abs=1e-15)


# ----- limit relations ------------------------------------------------------


def test_renyi_divergence_limit_is_kl() -> None:
    kl = float(kl_divergence(_P, _Q))
    near = float(renyi_divergence(_P, _Q, 1.0 + 1e-6))
    assert near == pytest.approx(kl, rel=1e-3)


def test_renyi_half_matches_hellinger_identity() -> None:
    # D_{1/2}(p||q) = -2 ln(1 - H^2(p,q))
    h2 = float(hellinger_distance(_P, _Q)) ** 2
    assert float(renyi_divergence(_P, _Q, 0.5)) == pytest.approx(
        -2.0 * math.log(1.0 - h2), rel=1e-12
    )


def test_renyi_entropy_limit_is_shannon_and_collision() -> None:
    near = float(renyi_entropy(_P, 1.0 + 1e-6))
    assert near == pytest.approx(float(entropy(_P)), rel=1e-3)
    assert float(renyi_entropy(_P, 2.0)) == pytest.approx(
        -math.log(float((_P**2).sum())), rel=1e-12
    )


def test_tsallis_limit_is_shannon_and_gini() -> None:
    near = float(tsallis_entropy(_P, 1.0 + 1e-6))
    assert near == pytest.approx(float(entropy(_P)), rel=1e-3)
    assert float(tsallis_entropy(_P, 2.0)) == pytest.approx(1.0 - float((_P**2).sum()), rel=1e-12)


@pytest.mark.parametrize(
    ("f", "ref"),
    [
        (lambda t: t * jnp.log(t), "kl"),
        (lambda t: (t - 1.0) ** 2, "chi2"),
        (lambda t: jnp.abs(t - 1.0), "tv2"),
        (lambda t: (jnp.sqrt(t) - 1.0) ** 2, "hellinger2"),
    ],
)
def test_f_divergence_specialisations(f, ref: str) -> None:  # type: ignore[no-untyped-def]
    val = float(f_divergence(_P, _Q, f))
    if ref == "kl":
        assert val == pytest.approx(float(kl_divergence(_P, _Q)), rel=1e-12)
    elif ref == "chi2":
        assert val == pytest.approx(float(chi_squared_divergence(_P, _Q)), rel=1e-12)
    elif ref == "tv2":  # sum |p - q| = 2 TV
        assert val == pytest.approx(2.0 * float(total_variation_distance(_P, _Q)), rel=1e-12)
    else:  # sum (sqrt p - sqrt q)^2 = 2 H^2
        assert val == pytest.approx(2.0 * float(hellinger_distance(_P, _Q)) ** 2, rel=1e-12)


def test_renyi_rejects_alpha_one() -> None:
    with pytest.raises(ValueError, match="alpha != 1"):
        renyi_divergence(_P, _Q, 1.0)
    with pytest.raises(ValueError, match="alpha != 1"):
        renyi_entropy(_P, 1.0)


# ----- optimal transport ----------------------------------------------------


def test_wassersteinp_p1_matches_w1_and_p2_rms() -> None:
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.normal(size=40))
    v = jnp.asarray(rng.normal(size=40))
    assert float(wassersteinp(u, v, p=1.0)) == pytest.approx(float(wasserstein1(u, v)), rel=1e-12)
    us, vs = np.sort(np.asarray(u)), np.sort(np.asarray(v))
    rms = math.sqrt(float(np.mean((us - vs) ** 2)))
    assert float(wassersteinp(u, v, p=2.0)) == pytest.approx(rms, rel=1e-12)


def test_wassersteinp_rejects_bad_p_and_shape() -> None:
    with pytest.raises(ValueError, match="p >= 1"):
        wassersteinp(jnp.zeros(3), jnp.zeros(3), p=0.5)
    with pytest.raises(ValueError, match="equal length"):
        wassersteinp(jnp.zeros(3), jnp.zeros(4))


def test_wasserstein2_gaussian_closed_form() -> None:
    assert float(wasserstein2_gaussian(0.0, 1.0, 3.0, 2.0)) == pytest.approx(math.sqrt(10.0))
    assert float(wasserstein2_gaussian(2.0, 0.5, 2.0, 0.5)) == pytest.approx(0.0, abs=1e-15)


def test_sliced_wasserstein_identity_and_translation() -> None:
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.normal(size=(30, 3)))
    dirs = np.asarray(rng.normal(size=(8, 3)))
    dirs = jnp.asarray(dirs / np.linalg.norm(dirs, axis=1, keepdims=True))
    assert float(sliced_wasserstein(X, X, dirs)) == pytest.approx(0.0, abs=1e-12)
    shift = jnp.asarray([1.0, 0.0, 0.0])
    assert float(sliced_wasserstein(X, X + shift, dirs)) > 0.0


def test_sliced_wasserstein_rejects_shapes() -> None:
    with pytest.raises(ValueError, match="equal-shape"):
        sliced_wasserstein(jnp.zeros((4, 2)), jnp.zeros((5, 2)), jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="directions"):
        sliced_wasserstein(jnp.zeros((4, 2)), jnp.zeros((4, 2)), jnp.zeros((3, 5)))


def test_sinkhorn_approaches_exact_ot_as_epsilon_shrinks() -> None:
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(2)
    n = 6
    xs = np.sort(rng.normal(size=n))
    ys = np.sort(rng.normal(size=n))
    cost = np.abs(xs[:, None] - ys[None, :])
    a = jnp.full((n,), 1.0 / n)
    b = jnp.full((n,), 1.0 / n)
    ri, ci = scipy_opt.linear_sum_assignment(cost)
    exact = float(cost[ri, ci].mean())
    coarse = float(sinkhorn_distance(a, b, jnp.asarray(cost), epsilon=0.5, num_iters=500))
    fine = float(sinkhorn_distance(a, b, jnp.asarray(cost), epsilon=0.005, num_iters=2000))
    assert fine == pytest.approx(exact, abs=1e-3)
    assert fine <= coarse + 1e-9  # entropic bias shrinks with epsilon


def test_sinkhorn_rejects_bad_cost_shape_and_epsilon() -> None:
    a = jnp.full((3,), 1.0 / 3)
    b = jnp.full((2,), 0.5)
    with pytest.raises(ValueError, match="cost must have shape"):
        sinkhorn_distance(a, b, jnp.zeros((3, 3)))
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_distance(a, b, jnp.zeros((3, 2)), epsilon=0.0)


# ----- differentiability ----------------------------------------------------


def test_new_operators_are_differentiable() -> None:
    g_renyi = jax.grad(lambda p: renyi_divergence(p, _Q, 0.5).sum())(_P)
    assert bool(jnp.all(jnp.isfinite(g_renyi)))
    rng = np.random.default_rng(3)
    cost = jnp.asarray(rng.normal(size=(4, 4)) ** 2)
    a = jnp.full((4,), 0.25)
    g_sink = jax.grad(lambda c: sinkhorn_distance(a, a, c, epsilon=0.2, num_iters=100))(cost)
    assert bool(jnp.all(jnp.isfinite(g_sink)))


# ----- certified enclosure --------------------------------------------------


def test_certified_core_encloses_differentiable() -> None:
    from omnibias.core.verified.information import (
        chi_squared_enclosure,
        hellinger_enclosure,
        total_variation_enclosure,
    )
    from omnibias.core.verified.transport import certified_wasserstein2_samples

    p = [0.5, 0.3, 0.2]
    q = [0.2, 0.5, 0.3]
    pj, qj = jnp.asarray(p), jnp.asarray(q)
    tv = total_variation_enclosure(p, q)
    assert tv.lo - 1e-12 <= float(total_variation_distance(pj, qj)) <= tv.hi + 1e-12
    he = hellinger_enclosure(p, q)
    assert he.lo - 1e-12 <= float(hellinger_distance(pj, qj)) <= he.hi + 1e-12
    ch = chi_squared_enclosure(p, q)
    assert ch.lo - 1e-12 <= float(chi_squared_divergence(pj, qj)) <= ch.hi + 1e-12
    u = [0.0, 1.0, 2.0, 3.5]
    v = [0.4, 1.1, 2.2, 3.0]
    w2 = certified_wasserstein2_samples(u, v)
    assert w2.lo - 1e-12 <= float(wassersteinp(jnp.asarray(u), jnp.asarray(v), p=2.0)) <= w2.hi + 1e-12


# ----- cross-backend parity (torch <-> jax) ---------------------------------


def test_divergence_and_ot_parity_with_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch import information as ti

    rng = np.random.default_rng(7)
    p_np = rng.dirichlet(np.ones(6))
    q_np = rng.dirichlet(np.ones(6))
    p_j, q_j = jnp.asarray(p_np), jnp.asarray(q_np)
    p_t = torch.tensor(p_np, dtype=torch.float64)
    q_t = torch.tensor(q_np, dtype=torch.float64)

    pairs = [
        (total_variation_distance(p_j, q_j), ti.total_variation_distance(p_t, q_t)),
        (hellinger_distance(p_j, q_j), ti.hellinger_distance(p_t, q_t)),
        (chi_squared_divergence(p_j, q_j), ti.chi_squared_divergence(p_t, q_t)),
        (renyi_divergence(p_j, q_j, 0.5), ti.renyi_divergence(p_t, q_t, 0.5)),
        (renyi_divergence(p_j, q_j, 2.0), ti.renyi_divergence(p_t, q_t, 2.0)),
        (renyi_entropy(p_j, 2.0), ti.renyi_entropy(p_t, 2.0)),
        (tsallis_entropy(p_j, 2.0), ti.tsallis_entropy(p_t, 2.0)),
    ]
    for j_val, t_val in pairs:
        assert float(j_val) == pytest.approx(float(t_val), rel=1e-12, abs=1e-12)

    u_np, v_np = rng.normal(size=33), rng.normal(size=33)
    for p in (1.0, 2.0, 3.0):
        wj = float(wassersteinp(jnp.asarray(u_np), jnp.asarray(v_np), p=p))
        wt = float(ti.wassersteinp(torch.tensor(u_np), torch.tensor(v_np), p=p))
        assert wj == pytest.approx(wt, rel=1e-12)
    g_args_t = [torch.tensor(v, dtype=torch.float64) for v in (0.1, 1.2, -0.3, 0.7)]
    assert float(wasserstein2_gaussian(0.1, 1.2, -0.3, 0.7)) == pytest.approx(
        float(ti.wasserstein2_gaussian(*g_args_t)), rel=1e-12
    )

    X = rng.normal(size=(20, 4))
    Y = rng.normal(size=(20, 4))
    dirs = rng.normal(size=(6, 4))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    swj = float(sliced_wasserstein(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(dirs)))
    swt = float(ti.sliced_wasserstein(torch.tensor(X), torch.tensor(Y), torch.tensor(dirs)))
    assert swj == pytest.approx(swt, rel=1e-10, abs=1e-12)

    a_np = rng.dirichlet(np.ones(5))
    b_np = rng.dirichlet(np.ones(7))
    cost = rng.normal(size=(5, 7)) ** 2
    skj = float(sinkhorn_distance(jnp.asarray(a_np), jnp.asarray(b_np), jnp.asarray(cost), epsilon=0.2, num_iters=300))
    skt = float(ti.sinkhorn_distance(torch.tensor(a_np), torch.tensor(b_np), torch.tensor(cost), epsilon=0.2, num_iters=300))
    assert skj == pytest.approx(skt, rel=1e-10, abs=1e-12)

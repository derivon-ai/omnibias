# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Information theory + information geometry operators (JAX) + cross-backend parity."""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.information import (  # noqa: E402
    cross_entropy,
    entropy,
    exponential_family_cumulants,
    fisher_information,
    fit_natural_parameter,
    glm_mean,
    glm_variance,
    js_divergence,
    kl_divergence,
    moment_match,
    mutual_information,
    wasserstein1,
    wasserstein1_cdf,
)


def _logistic_quantiles(n: int, loc: float = 0.0, scale: float = 1.0) -> list[float]:
    return [loc + scale * math.log(u / (1.0 - u)) for u in ((i + 0.5) / n for i in range(n))]


def _w1_model_numeric(
    name: str, xs: list[float], loc: float, scale: float, *, half_width: float = 80.0
) -> float:
    """Independent fine-grid oracle for ``int |F - F_n| dx`` (manual trapezoid)."""
    n_grid = 200_001
    a = loc - half_width * scale
    b = loc + half_width * scale
    dx = (b - a) / (n_grid - 1)
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def model_cdf(x: float) -> float:
        u = (x - loc) / scale
        if name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-u))
        return 0.5 * math.tanh(u) + 0.5

    import bisect

    total = 0.0
    prev = None
    for i in range(n_grid):
        x = a + i * dx
        fn = bisect.bisect_right(xs_sorted, x) / n
        val = abs(model_cdf(x) - fn)
        if prev is not None:
            total += 0.5 * (prev + val) * dx
        prev = val
    return total


# ----- discrete information theory ------------------------------------------


def test_entropy_uniform_is_log_k() -> None:
    for k in (2, 5, 17):
        p = jnp.full((k,), 1.0 / k)
        assert float(entropy(p)) == pytest.approx(math.log(k))


def test_cross_entropy_equals_entropy_plus_kl() -> None:
    p = jnp.asarray([0.1, 0.6, 0.3])
    q = jnp.asarray([0.2, 0.5, 0.3])
    assert float(cross_entropy(p, q)) == pytest.approx(
        float(entropy(p) + kl_divergence(p, q)), rel=1e-12
    )


def test_kl_nonnegative_and_diverges_off_support() -> None:
    p = jnp.asarray([0.2, 0.5, 0.3])
    q = jnp.asarray([0.5, 0.2, 0.3])
    assert float(kl_divergence(p, p)) == pytest.approx(0.0, abs=1e-12)
    assert float(kl_divergence(p, q)) > 0.0
    assert math.isinf(float(kl_divergence(jnp.asarray([0.5, 0.5]), jnp.asarray([1.0, 0.0]))))


def test_js_symmetric_and_bounded() -> None:
    p = jnp.asarray([0.1, 0.7, 0.2])
    q = jnp.asarray([0.6, 0.1, 0.3])
    assert float(js_divergence(p, q)) == pytest.approx(float(js_divergence(q, p)), rel=1e-12)
    assert 0.0 <= float(js_divergence(p, q)) <= math.log(2.0) + 1e-12


def test_mutual_information_independent_and_correlated() -> None:
    px = jnp.asarray([0.3, 0.7])
    py = jnp.asarray([0.2, 0.3, 0.5])
    joint = px[:, None] * py[None, :]
    assert float(mutual_information(joint)) == pytest.approx(0.0, abs=1e-12)
    p = jnp.asarray([0.25, 0.25, 0.5])
    assert float(mutual_information(jnp.diag(p))) == pytest.approx(float(entropy(p)), rel=1e-12)


# ----- optimal transport ----------------------------------------------------


def test_wasserstein1_shift() -> None:
    u = jnp.linspace(0.0, 1.0, 50)
    v = u + 0.75
    assert float(wasserstein1(u, v)) == pytest.approx(0.75, rel=1e-12)
    assert float(wasserstein1(u, u)) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein1_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        wasserstein1(jnp.zeros(3), jnp.zeros(4))


# ----- model-vs-empirical W1 (wasserstein1_cdf) -----------------------------


@pytest.mark.parametrize("name", ["sigmoid", "tanh"])
def test_w1_cdf_matches_numeric_oracle(name: str) -> None:
    loc, scale = 0.3, 1.4
    xs = _logistic_quantiles(48, loc=0.0, scale=1.0)
    val = float(wasserstein1_cdf(name, jnp.asarray(xs), loc=loc, scale=scale))
    assert val == pytest.approx(_w1_model_numeric(name, xs, loc, scale), abs=2e-3)
    assert val >= 0.0


@pytest.mark.parametrize("name", ["sigmoid", "tanh"])
def test_w1_cdf_is_enclosed_by_certified(name: str) -> None:
    from omnibias.core.verified.transport import certified_wasserstein1

    loc, scale = 0.2, 1.1
    xs = _logistic_quantiles(40, loc=0.0, scale=1.0)
    val = float(wasserstein1_cdf(name, jnp.asarray(xs), loc=loc, scale=scale))
    enc = certified_wasserstein1(name, xs, loc=loc, scale=scale)
    assert enc.lo - 1e-9 <= val <= enc.hi + 1e-9


def test_w1_cdf_tanh_equals_sigmoid_at_half_scale() -> None:
    xs = jnp.asarray(_logistic_quantiles(60, loc=0.1, scale=0.9))
    tanh_val = float(wasserstein1_cdf("tanh", xs, loc=0.2, scale=2.0))
    sig_val = float(wasserstein1_cdf("sigmoid", xs, loc=0.2, scale=1.0))
    assert tanh_val == pytest.approx(sig_val, rel=1e-12)


def test_w1_cdf_is_differentiable_in_loc() -> None:
    xs = jnp.asarray(_logistic_quantiles(64, loc=0.0, scale=1.0))
    grad = jax.grad(lambda loc: wasserstein1_cdf("sigmoid", xs, loc=loc, scale=1.3))(0.7)
    assert jnp.isfinite(grad)
    assert float(jnp.abs(grad)) > 0.0


def test_w1_cdf_rejects_arctan() -> None:
    with pytest.raises(NotImplementedError, match="finite"):
        wasserstein1_cdf("arctan", jnp.asarray([0.0, 1.0]))


def test_w1_cdf_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D samples"):
        wasserstein1_cdf("sigmoid", jnp.zeros((2, 3)))


# ----- exponential families / information geometry --------------------------


def test_softplus_cumulants_are_bernoulli_moments() -> None:
    theta = jnp.linspace(-3.0, 3.0, 13)
    cum = exponential_family_cumulants(theta, base="softplus", order=2)
    assert cum.shape == (3, 13)
    s = jax.nn.sigmoid(theta)
    assert jnp.allclose(cum[1], s, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(cum[2], s * (1.0 - s), rtol=1e-12, atol=1e-12)


def test_glm_helpers() -> None:
    theta = jnp.linspace(-2.0, 2.0, 9)
    s = jax.nn.sigmoid(theta)
    assert jnp.allclose(glm_mean(theta), s, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(glm_variance(theta), s * (1.0 - s), rtol=1e-12, atol=1e-12)
    assert jnp.allclose(fisher_information(theta), glm_variance(theta), rtol=1e-12, atol=1e-12)


def test_cumulants_reject_bad_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 1"):
        exponential_family_cumulants(jnp.zeros(3), order=0)


# ----- cross-backend parity (torch <-> jax) ---------------------------------


# ----- MLE / moment matching (the inverse map) ------------------------------


def test_moment_match_softplus_is_the_logit() -> None:
    means = jnp.asarray([0.1, 0.27, 0.5, 0.73, 0.9])
    theta = moment_match(means, base="softplus")
    logit = jnp.log(means / (1.0 - means))
    assert np.allclose(np.asarray(theta), np.asarray(logit), rtol=1e-10, atol=1e-12)


def test_moment_match_poisson_is_the_log() -> None:
    means = jnp.asarray([0.5, 2.0, 7.0])
    theta = moment_match(means, base="exp")
    assert np.allclose(np.asarray(theta), np.asarray(jnp.log(means)), rtol=1e-10, atol=1e-12)


def test_fit_natural_parameter_recovers_logit_of_sample_mean() -> None:
    samples = jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    theta = fit_natural_parameter(samples, base="softplus")
    p = float(samples.mean())
    assert float(theta) == pytest.approx(math.log(p / (1.0 - p)), rel=1e-10)


def test_moment_match_round_trips_through_glm_mean() -> None:
    means = jnp.asarray([0.05, 0.4, 0.95])
    theta = moment_match(means, base="softplus")
    assert np.allclose(np.asarray(glm_mean(theta, base="softplus")), np.asarray(means))


def test_cross_backend_parity_with_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch import information as ti

    rng = np.random.default_rng(0)
    p_np = rng.dirichlet(np.ones(5))
    q_np = rng.dirichlet(np.ones(5))
    theta_np = np.linspace(-3.0, 3.0, 11)

    p_j, q_j = jnp.asarray(p_np), jnp.asarray(q_np)
    p_t = torch.tensor(p_np, dtype=torch.float64)
    q_t = torch.tensor(q_np, dtype=torch.float64)
    theta_j = jnp.asarray(theta_np)
    theta_t = torch.tensor(theta_np, dtype=torch.float64)

    assert float(entropy(p_j)) == pytest.approx(float(ti.entropy(p_t)), rel=1e-12)
    assert float(kl_divergence(p_j, q_j)) == pytest.approx(
        float(ti.kl_divergence(p_t, q_t)), rel=1e-12
    )
    assert float(js_divergence(p_j, q_j)) == pytest.approx(
        float(ti.js_divergence(p_t, q_t)), rel=1e-12
    )

    cum_j = np.asarray(exponential_family_cumulants(theta_j, base="softplus", order=3))
    cum_t = ti.exponential_family_cumulants(theta_t, base="softplus", order=3).numpy()
    assert np.allclose(cum_j, cum_t, rtol=1e-12, atol=1e-12)

    u_np, v_np = rng.normal(size=40), rng.normal(size=40)
    w_j = float(wasserstein1(jnp.asarray(u_np), jnp.asarray(v_np)))
    w_t = float(ti.wasserstein1(torch.tensor(u_np), torch.tensor(v_np)))
    assert w_j == pytest.approx(w_t, rel=1e-12)

    # model-vs-empirical W1 parity (sigmoid + tanh)
    samp = rng.normal(size=48)
    for name in ("sigmoid", "tanh"):
        wc_j = float(wasserstein1_cdf(name, jnp.asarray(samp), loc=0.2, scale=1.3))
        wc_t = float(
            ti.wasserstein1_cdf(
                name,
                torch.tensor(samp, dtype=torch.float64),
                loc=torch.tensor(0.2, dtype=torch.float64),
                scale=torch.tensor(1.3, dtype=torch.float64),
            )
        )
        assert wc_j == pytest.approx(wc_t, rel=1e-12)

    # MLE / moment matching parity
    means_np = np.array([0.15, 0.45, 0.8])
    th_j = np.asarray(moment_match(jnp.asarray(means_np), base="softplus"))
    th_t = ti.moment_match(torch.tensor(means_np, dtype=torch.float64)).numpy()
    assert np.allclose(th_j, th_t, rtol=1e-12, atol=1e-12)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Information theory + information geometry operators (PyTorch)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.torch.information import (
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

F64 = torch.float64


def _logistic_quantiles(n: int, loc: float = 0.0, scale: float = 1.0) -> list[float]:
    """Deterministic logistic samples (no RNG, no flakiness)."""
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
        p = torch.full((k,), 1.0 / k, dtype=F64)
        assert float(entropy(p)) == pytest.approx(math.log(k))


def test_entropy_of_point_mass_is_zero() -> None:
    p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F64)
    assert float(entropy(p)) == pytest.approx(0.0, abs=1e-12)


def test_entropy_is_batched_over_last_axis() -> None:
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=F64)
    out = entropy(p)
    assert out.shape == (2,)
    assert float(out[0]) == pytest.approx(math.log(2.0))
    assert float(out[1]) == pytest.approx(0.0, abs=1e-12)


def test_cross_entropy_equals_entropy_plus_kl() -> None:
    p = torch.tensor([0.1, 0.6, 0.3], dtype=F64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=F64)
    lhs = cross_entropy(p, q)
    rhs = entropy(p) + kl_divergence(p, q)
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-12)


def test_kl_is_zero_iff_equal_and_nonnegative() -> None:
    p = torch.tensor([0.2, 0.5, 0.3], dtype=F64)
    q = torch.tensor([0.5, 0.2, 0.3], dtype=F64)
    assert float(kl_divergence(p, p)) == pytest.approx(0.0, abs=1e-12)
    assert float(kl_divergence(p, q)) > 0.0  # Gibbs' inequality


def test_kl_diverges_when_support_not_covered() -> None:
    p = torch.tensor([0.5, 0.5], dtype=F64)
    q = torch.tensor([1.0, 0.0], dtype=F64)
    assert math.isinf(float(kl_divergence(p, q)))


def test_js_is_symmetric_and_bounded() -> None:
    p = torch.tensor([0.1, 0.7, 0.2], dtype=F64)
    q = torch.tensor([0.6, 0.1, 0.3], dtype=F64)
    js_pq = float(js_divergence(p, q))
    js_qp = float(js_divergence(q, p))
    assert js_pq == pytest.approx(js_qp, rel=1e-12)
    assert 0.0 <= js_pq <= math.log(2.0) + 1e-12
    assert float(js_divergence(p, p)) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_zero_for_independent_joint() -> None:
    px = torch.tensor([0.3, 0.7], dtype=F64)
    py = torch.tensor([0.2, 0.3, 0.5], dtype=F64)
    joint = px[:, None] * py[None, :]
    assert float(mutual_information(joint)) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_of_perfectly_correlated_equals_marginal_entropy() -> None:
    p = torch.tensor([0.25, 0.25, 0.5], dtype=F64)
    joint = torch.diag(p)
    assert float(mutual_information(joint)) == pytest.approx(float(entropy(p)), rel=1e-12)


def test_entropy_is_differentiable() -> None:
    p = torch.tensor([0.2, 0.3, 0.5], dtype=F64, requires_grad=True)
    entropy(p).backward()
    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))


# ----- optimal transport ----------------------------------------------------


def test_wasserstein1_identical_samples_is_zero() -> None:
    u = torch.linspace(-2.0, 3.0, 64, dtype=F64)
    assert float(wasserstein1(u, u)) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein1_of_shift_equals_shift() -> None:
    u = torch.linspace(0.0, 1.0, 50, dtype=F64)
    v = u + 0.75
    assert float(wasserstein1(u, v)) == pytest.approx(0.75, rel=1e-12)
    assert float(wasserstein1(u, v)) == pytest.approx(float(wasserstein1(v, u)), rel=1e-12)


def test_wasserstein1_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        wasserstein1(torch.zeros(3, dtype=F64), torch.zeros(4, dtype=F64))


# ----- model-vs-empirical W1 (wasserstein1_cdf) -----------------------------


@pytest.mark.parametrize("name", ["sigmoid", "tanh"])
def test_w1_cdf_matches_numeric_oracle(name: str) -> None:
    loc, scale = 0.3, 1.4
    xs = _logistic_quantiles(48, loc=0.0, scale=1.0)
    val = float(
        wasserstein1_cdf(
            name,
            torch.tensor(xs, dtype=F64),
            loc=torch.tensor(loc, dtype=F64),
            scale=torch.tensor(scale, dtype=F64),
        )
    )
    assert val == pytest.approx(_w1_model_numeric(name, xs, loc, scale), abs=2e-3)
    assert val >= 0.0


@pytest.mark.parametrize("name", ["sigmoid", "tanh"])
def test_w1_cdf_is_enclosed_by_certified(name: str) -> None:
    # The differentiable value must lie inside the rigorous certified enclosure.
    from omnibias.core.verified.transport import certified_wasserstein1

    loc, scale = 0.2, 1.1
    xs = _logistic_quantiles(40, loc=0.0, scale=1.0)
    val = float(
        wasserstein1_cdf(
            name,
            torch.tensor(xs, dtype=F64),
            loc=torch.tensor(loc, dtype=F64),
            scale=torch.tensor(scale, dtype=F64),
        )
    )
    enc = certified_wasserstein1(name, xs, loc=loc, scale=scale)
    assert enc.lo - 1e-9 <= val <= enc.hi + 1e-9


def test_w1_cdf_tanh_equals_sigmoid_at_half_scale() -> None:
    xs = torch.tensor(_logistic_quantiles(60, loc=0.1, scale=0.9), dtype=F64)
    loc = torch.tensor(0.2, dtype=F64)
    tanh_val = wasserstein1_cdf("tanh", xs, loc=loc, scale=torch.tensor(2.0, dtype=F64))
    sig_val = wasserstein1_cdf("sigmoid", xs, loc=loc, scale=torch.tensor(1.0, dtype=F64))
    assert float(tanh_val) == pytest.approx(float(sig_val), rel=1e-12)


def test_w1_cdf_grows_when_model_is_misplaced() -> None:
    xs = torch.tensor(_logistic_quantiles(200, loc=0.0, scale=1.0), dtype=F64)
    centered = wasserstein1_cdf("sigmoid", xs, loc=torch.tensor(0.0, dtype=F64))
    shifted = wasserstein1_cdf("sigmoid", xs, loc=torch.tensor(3.0, dtype=F64))
    assert float(shifted) > float(centered)


def test_w1_cdf_is_differentiable_in_loc_and_scale() -> None:
    xs = torch.tensor(_logistic_quantiles(64, loc=0.0, scale=1.0), dtype=F64)
    loc = torch.tensor(0.7, dtype=F64, requires_grad=True)
    scale = torch.tensor(1.3, dtype=F64, requires_grad=True)
    wasserstein1_cdf("sigmoid", xs, loc=loc, scale=scale).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad)
    assert scale.grad is not None and torch.isfinite(scale.grad)
    assert float(loc.grad.abs()) > 0.0  # the model is misplaced -> nonzero gradient


def test_w1_cdf_is_differentiable_in_samples() -> None:
    xs = torch.tensor(_logistic_quantiles(32, loc=0.5, scale=1.0), dtype=F64, requires_grad=True)
    wasserstein1_cdf("sigmoid", xs).backward()
    assert xs.grad is not None and torch.all(torch.isfinite(xs.grad))


def test_w1_cdf_single_sample_is_finite() -> None:
    val = wasserstein1_cdf("sigmoid", torch.tensor([0.4], dtype=F64))
    assert torch.isfinite(val) and float(val) > 0.0


def test_w1_cdf_rejects_arctan() -> None:
    with pytest.raises(NotImplementedError, match="finite"):
        wasserstein1_cdf("arctan", torch.tensor([0.0, 1.0], dtype=F64))


def test_w1_cdf_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D samples"):
        wasserstein1_cdf("sigmoid", torch.zeros((2, 3), dtype=F64))


def test_w1_cdf_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        wasserstein1_cdf("sigmoid", torch.zeros(0, dtype=F64))


# ----- exponential families / information geometry --------------------------


def test_softplus_cumulants_are_the_bernoulli_moments() -> None:
    theta = torch.linspace(-3.0, 3.0, 13, dtype=F64)
    cum = exponential_family_cumulants(theta, base="softplus", order=2)
    assert cum.shape == (3, 13)
    s = torch.sigmoid(theta)
    # kappa_0 = A = softplus, kappa_1 = mean = sigmoid, kappa_2 = variance = s(1-s)
    assert torch.allclose(cum[0], torch.nn.functional.softplus(theta), rtol=1e-12, atol=1e-12)
    assert torch.allclose(cum[1], s, rtol=1e-12, atol=1e-12)
    assert torch.allclose(cum[2], s * (1.0 - s), rtol=1e-12, atol=1e-12)


def test_glm_helpers_match_cumulants() -> None:
    theta = torch.linspace(-2.0, 2.0, 9, dtype=F64)
    s = torch.sigmoid(theta)
    assert torch.allclose(glm_mean(theta), s, rtol=1e-12, atol=1e-12)
    assert torch.allclose(glm_variance(theta), s * (1.0 - s), rtol=1e-12, atol=1e-12)
    assert torch.allclose(fisher_information(theta), glm_variance(theta), rtol=1e-12, atol=1e-12)


def test_fisher_information_is_positive() -> None:
    theta = torch.linspace(-5.0, 5.0, 21, dtype=F64)
    assert torch.all(fisher_information(theta) > 0.0)


def test_cumulants_match_finite_difference_oracle() -> None:
    theta = torch.linspace(-2.0, 2.0, 11, dtype=F64)
    h = 1e-4
    sp = torch.nn.functional.softplus
    d1 = (sp(theta + h) - sp(theta - h)) / (2.0 * h)
    d2 = (sp(theta + h) - 2.0 * sp(theta) + sp(theta - h)) / (h * h)
    cum = exponential_family_cumulants(theta, base="softplus", order=2)
    assert torch.allclose(cum[1], d1, rtol=1e-6, atol=1e-7)
    assert torch.allclose(cum[2], d2, rtol=1e-4, atol=1e-6)


def test_cumulants_are_differentiable() -> None:
    theta = torch.linspace(-1.0, 1.0, 5, dtype=F64, requires_grad=True)
    glm_variance(theta).sum().backward()
    assert theta.grad is not None
    assert torch.all(torch.isfinite(theta.grad))


def test_cumulants_reject_bad_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 1"):
        exponential_family_cumulants(torch.zeros(3, dtype=F64), order=0)


def test_cumulants_accept_python_scalar() -> None:
    val = exponential_family_cumulants(0.0, base="softplus", order=1)
    assert float(val[1]) == pytest.approx(0.5, rel=1e-12)  # sigmoid(0) = 1/2


# ----- MLE / moment matching (the inverse map) ------------------------------


def test_moment_match_softplus_is_the_logit() -> None:
    # Bernoulli: A'(theta) = sigmoid(theta), so the inverse link is the logit.
    means = torch.tensor([0.1, 0.27, 0.5, 0.73, 0.9], dtype=F64)
    theta = moment_match(means, base="softplus")
    logit = torch.log(means / (1.0 - means))
    assert torch.allclose(theta, logit, rtol=1e-10, atol=1e-12)


def test_moment_match_round_trips_through_glm_mean() -> None:
    means = torch.tensor([0.05, 0.4, 0.95], dtype=F64)
    theta = moment_match(means, base="softplus")
    assert torch.allclose(glm_mean(theta, base="softplus"), means, rtol=1e-10, atol=1e-12)


def test_moment_match_poisson_is_the_log() -> None:
    # Poisson: A = exp, A'(theta) = exp(theta), inverse link is log.
    means = torch.tensor([0.5, 2.0, 7.0], dtype=F64)
    theta = moment_match(means, base="exp")
    assert torch.allclose(theta, torch.log(means), rtol=1e-10, atol=1e-12)


def test_fit_natural_parameter_recovers_logit_of_sample_mean() -> None:
    samples = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=F64)
    theta = fit_natural_parameter(samples, base="softplus")
    p = samples.mean()
    assert float(theta) == pytest.approx(math.log(p / (1.0 - p)), rel=1e-10)


def test_fit_natural_parameter_is_batched_over_last_axis() -> None:
    samples = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=F64)
    theta = fit_natural_parameter(samples, base="softplus")  # mean = [0.5, 0.75]
    expected = torch.log(torch.tensor([0.5, 0.75], dtype=F64) / torch.tensor([0.5, 0.25], dtype=F64))
    assert torch.allclose(theta, expected, rtol=1e-10, atol=1e-12)


def test_moment_match_is_differentiable() -> None:
    mean = torch.tensor(0.3, dtype=F64, requires_grad=True)
    moment_match(mean, base="softplus").backward()
    assert mean.grad is not None and torch.isfinite(mean.grad)

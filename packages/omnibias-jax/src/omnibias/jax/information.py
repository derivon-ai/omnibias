# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Information theory + information geometry operators (JAX).

Bit-identical twin of :mod:`omnibias.torch.information`:

* **Discrete information theory**: :func:`entropy`, :func:`cross_entropy`,
  :func:`kl_divergence`, :func:`js_divergence`, :func:`mutual_information`
  (``xlogy``-based, exact ``0 log 0`` convention).
* **Optimal transport**: :func:`wasserstein1` (1-D, equal-size empirical samples)
  and :func:`wasserstein1_cdf` (differentiable model-vs-empirical ``W_1`` for a
  ``sigmoid`` / ``tanh`` CDF).
* **Information geometry / exponential families**:
  :func:`exponential_family_cumulants` (the log-partition derivative tower is the
  cumulant tower), with :func:`glm_mean`, :func:`glm_variance`, and
  :func:`fisher_information`.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax.numpy as jnp
from jax import Array
from jax.nn import softplus
from jax.scipy.special import logsumexp, xlogy


def _as_array(x: Array | float) -> Array:
    return jnp.asarray(x)


# ----- discrete information theory ------------------------------------------


def entropy(p: Array, *, dim: int = -1) -> Array:
    r"""Shannon entropy ``H(p) = -sum_i p_i ln p_i`` (nats) along ``dim``."""
    out: Array = -xlogy(p, p).sum(axis=dim)
    return out


def cross_entropy(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Cross-entropy ``H(p, q) = -sum_i p_i ln q_i`` (nats) along ``dim``."""
    out: Array = -xlogy(p, q).sum(axis=dim)
    return out


def kl_divergence(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Kullback-Leibler divergence ``D(p || q) = sum_i p_i ln(p_i / q_i)``."""
    out: Array = (xlogy(p, p) - xlogy(p, q)).sum(axis=dim)
    return out


def js_divergence(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Jensen-Shannon divergence ``0.5 D(p||m) + 0.5 D(q||m)``, ``m = (p+q)/2``."""
    m = 0.5 * (p + q)
    out: Array = 0.5 * kl_divergence(p, m, dim=dim) + 0.5 * kl_divergence(q, m, dim=dim)
    return out


def mutual_information(p_joint: Array) -> Array:
    r"""Mutual information ``I(X; Y)`` of a joint table over the last two axes."""
    px = p_joint.sum(axis=-1, keepdims=True)
    py = p_joint.sum(axis=-2, keepdims=True)
    outer = px * py
    out: Array = (xlogy(p_joint, p_joint) - xlogy(p_joint, outer)).sum(axis=(-2, -1))
    return out


# ----- generalized divergences ----------------------------------------------


def total_variation_distance(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Total-variation distance ``TV = 1/2 sum_i |p_i - q_i|`` along ``dim``.

    Symmetric, a proper metric, and bounded in ``[0, 1]``.
    """
    out: Array = 0.5 * jnp.abs(p - q).sum(axis=dim)
    return out


def hellinger_distance(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Hellinger distance ``H = sqrt(1/2 sum_i (sqrt(p_i) - sqrt(q_i))^2)``.

    Symmetric, a proper metric, bounded in ``[0, 1]``; finite even where ``p`` or
    ``q`` has zeros.
    """
    out: Array = jnp.sqrt((0.5 * (jnp.sqrt(p) - jnp.sqrt(q)) ** 2).sum(axis=dim))
    return out


def chi_squared_divergence(p: Array, q: Array, *, dim: int = -1) -> Array:
    r"""Pearson ``chi^2(p || q) = sum_i (p_i - q_i)^2 / q_i`` along ``dim``.

    Requires ``q_i > 0`` wherever evaluated; the local curvature of the KL
    divergence (twice the chi-square is its second-order expansion).
    """
    out: Array = ((p - q) ** 2 / q).sum(axis=dim)
    return out


def renyi_divergence(p: Array, q: Array, alpha: float, *, dim: int = -1) -> Array:
    r"""Renyi divergence ``D_alpha = 1/(alpha-1) ln sum_i p_i^alpha q_i^{1-alpha}``.

    Defined for ``alpha > 0``, ``alpha != 1``; the limit ``alpha -> 1`` is the KL
    divergence (:func:`kl_divergence`) and ``alpha = 1/2`` gives ``-2 ln(1 - H^2)``
    for the Hellinger affinity. ``q`` must cover the support of ``p``.
    """
    if alpha <= 0.0 or alpha == 1.0:
        raise ValueError(f"renyi_divergence needs alpha > 0 and alpha != 1, got {alpha}")
    s = (p**alpha * q ** (1.0 - alpha)).sum(axis=dim)
    out: Array = jnp.log(s) / (alpha - 1.0)
    return out


def renyi_entropy(p: Array, alpha: float, *, dim: int = -1) -> Array:
    r"""Renyi entropy ``H_alpha = 1/(1-alpha) ln sum_i p_i^alpha`` (nats).

    Defined for ``alpha > 0``, ``alpha != 1``; recovers Shannon entropy as
    ``alpha -> 1``, the collision entropy at ``alpha = 2`` and the max-entropy
    ``ln |support|`` as ``alpha -> 0``.
    """
    if alpha <= 0.0 or alpha == 1.0:
        raise ValueError(f"renyi_entropy needs alpha > 0 and alpha != 1, got {alpha}")
    out: Array = jnp.log((p**alpha).sum(axis=dim)) / (1.0 - alpha)
    return out


def tsallis_entropy(p: Array, order: float, *, dim: int = -1) -> Array:
    r"""Tsallis entropy ``S_q = (1 - sum_i p_i^q) / (q - 1)``.

    The non-extensive (``q``-deformed) entropy; ``order = q > 0``, ``q != 1``, with
    the Shannon entropy as the ``q -> 1`` limit.
    """
    if order <= 0.0 or order == 1.0:
        raise ValueError(f"tsallis_entropy needs order > 0 and order != 1, got {order}")
    out: Array = (1.0 - (p**order).sum(axis=dim)) / (order - 1.0)
    return out


def f_divergence(
    p: Array, q: Array, f: Callable[[Array], Array], *, dim: int = -1
) -> Array:
    r"""Generic Csiszar ``f``-divergence ``D_f(p || q) = sum_i q_i f(p_i / q_i)``.

    ``f`` is a convex function with ``f(1) = 0``; choosing ``f(t) = t ln t`` gives
    the KL divergence, ``f(t) = (t - 1)^2`` Pearson ``chi^2``,
    ``f(t) = |t - 1|/2`` total variation and ``f(t) = (sqrt(t) - 1)^2`` the squared
    Hellinger distance. ``q`` must be strictly positive.
    """
    out: Array = (q * f(p / q)).sum(axis=dim)
    return out


# ----- optimal transport ----------------------------------------------------


def wasserstein1(u: Array, v: Array) -> Array:
    r"""1-D Wasserstein-1 distance between two equal-size empirical samples."""
    if u.ndim != 1 or v.ndim != 1 or u.shape != v.shape:
        raise ValueError(
            f"wasserstein1 expects two 1-D samples of equal length, got "
            f"{tuple(u.shape)} and {tuple(v.shape)}"
        )
    us = jnp.sort(u)
    vs = jnp.sort(v)
    out: Array = jnp.abs(us - vs).mean()
    return out


#: Location-scale CDF bases with a finite first moment (hence finite ``W_1``).
SUPPORTED_W1_CDFS: tuple[str, ...] = ("sigmoid", "tanh")


def wasserstein1_cdf(
    name: str,
    samples: Array,
    *,
    loc: Array | float = 0.0,
    scale: Array | float = 1.0,
) -> Array:
    r"""Differentiable ``W_1(F, F_n)`` between a model CDF ``F`` and the data.

    Model-vs-empirical companion of :func:`wasserstein1`: ``F`` is the
    ``sigmoid`` / ``tanh`` location-scale CDF and ``F_n`` the empirical CDF of
    ``samples`` (1-D). Evaluates the exact integral ``W_1 = int |F - F_n| dx``
    through the closed-form antiderivative ``Phi(x) = s * softplus((x - loc)/s)``
    (two tails plus the per-panel crossing formula), differentiable in ``loc`` /
    ``scale`` / ``samples``. Bit-identical twin of
    :func:`omnibias.torch.information.wasserstein1_cdf`; the certified
    :func:`omnibias.core.verified.transport.certified_wasserstein1` encloses this
    value. ``arctan`` is rejected (no finite mean).
    """
    if samples.ndim != 1:
        raise ValueError(
            f"wasserstein1_cdf expects 1-D samples, got shape {tuple(samples.shape)}"
        )
    n = samples.shape[0]
    if n == 0:
        raise ValueError("wasserstein1_cdf needs at least one sample")
    key = name.lower()
    if key == "sigmoid":
        s = _as_array(scale)
    elif key == "tanh":
        s = _as_array(scale) * 0.5  # tanh CDF == sigmoid CDF at half the scale
    else:
        raise NotImplementedError(
            f"wasserstein1_cdf supports {SUPPORTED_W1_CDFS} (finite first moment); "
            f"{name!r} has no finite mean so W1 is infinite"
        )
    loc_a = _as_array(loc)
    xs = jnp.sort(samples)

    def phi(x: Array) -> Array:
        return s * softplus((x - loc_a) / s)

    total = phi(xs[0]) + s * softplus(-((xs[-1] - loc_a) / s))
    if n > 1:
        a = xs[:-1]
        b = xs[1:]
        i = jnp.arange(1, n, dtype=xs.dtype)
        c = i / n  # empirical CDF level on panel [x_(i), x_(i+1))
        xstar = loc_a + s * (jnp.log(c) - jnp.log1p(-c))  # F^{-1}(c)
        t = jnp.minimum(jnp.maximum(xstar, a), b)  # crossing clamped to panel
        panel = c * (2.0 * t - (a + b)) + phi(a) + phi(b) - 2.0 * phi(t)
        total = total + jnp.maximum(panel, 0.0).sum()
    return total


def wassersteinp(u: Array, v: Array, *, p: float = 1.0) -> Array:
    r"""1-D Wasserstein-``p`` distance between two equal-size empirical samples.

    ``W_p = (mean_i |u_(i) - v_(i)|^p)^{1/p}`` from the sorted order statistics
    (exact in one dimension). ``p = 1`` reproduces :func:`wasserstein1`; ``p = 2``
    is the quadratic transport distance whose certified twin is
    :func:`omnibias.core.verified.transport.certified_wasserstein2_samples`.
    """
    if u.ndim != 1 or v.ndim != 1 or u.shape != v.shape:
        raise ValueError(
            f"wassersteinp expects two 1-D samples of equal length, got "
            f"{tuple(u.shape)} and {tuple(v.shape)}"
        )
    if p < 1.0:
        raise ValueError(f"wassersteinp needs p >= 1, got {p}")
    us = jnp.sort(u)
    vs = jnp.sort(v)
    out: Array = (jnp.abs(us - vs) ** p).mean() ** (1.0 / p)
    return out


def wasserstein2_gaussian(
    mu1: Array | float,
    sigma1: Array | float,
    mu2: Array | float,
    sigma2: Array | float,
) -> Array:
    r"""Closed-form 1-D Gaussian ``W_2`` distance.

    ``W_2(N(mu1, sigma1^2), N(mu2, sigma2^2)) = sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2)``
    (standard deviations ``sigma >= 0``). Differentiable in all four arguments --
    the matched-Gaussian objective used by the residual diagnostics; the certified
    twin is :func:`omnibias.core.verified.transport.certified_wasserstein2_gaussian`.
    """
    dmu = _as_array(mu1) - _as_array(mu2)
    dsigma = _as_array(sigma1) - _as_array(sigma2)
    out: Array = jnp.sqrt(dmu**2 + dsigma**2)
    return out


def sliced_wasserstein(
    X: Array, Y: Array, directions: Array, *, p: float = 2.0
) -> Array:
    r"""Sliced Wasserstein-``p`` distance between two equal-size point clouds.

    Projects both clouds onto each unit ``directions`` row, averages the 1-D
    :func:`wassersteinp` over projections: ``SW_p = mean_theta W_p(theta . X, theta . Y)``.
    ``X`` and ``Y`` are ``(n, d)`` (equal ``n``) and ``directions`` is ``(k, d)``
    -- supply them (e.g. random unit vectors) so the estimate is deterministic and
    bit-identical across backends.
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError(
            f"sliced_wasserstein expects equal-shape (n, d) clouds, got "
            f"{tuple(X.shape)} and {tuple(Y.shape)}"
        )
    if directions.ndim != 2 or directions.shape[1] != X.shape[1]:
        raise ValueError(
            f"directions must be (k, d) with d = {X.shape[1]}, got "
            f"{tuple(directions.shape)}"
        )
    if p < 1.0:
        raise ValueError(f"sliced_wasserstein needs p >= 1, got {p}")
    px = jnp.sort(X @ directions.T, axis=0)
    py = jnp.sort(Y @ directions.T, axis=0)
    per_dir = (jnp.abs(px - py) ** p).mean(axis=0) ** (1.0 / p)
    out: Array = per_dir.mean()
    return out


def sinkhorn_distance(
    a: Array,
    b: Array,
    cost: Array,
    *,
    epsilon: float = 0.1,
    num_iters: int = 200,
) -> Array:
    r"""Entropic optimal-transport (Sinkhorn) cost between two histograms.

    Solves the entropy-regularised OT problem
    ``min_P <P, cost> - epsilon H(P)`` over couplings with marginals ``a``, ``b``
    by ``num_iters`` log-domain Sinkhorn iterations (numerically stable for small
    ``epsilon``) and returns the transport cost ``<P*, cost>``. ``a`` is ``(n,)``,
    ``b`` is ``(m,)`` (each summing to one) and ``cost`` is the ``(n, m)`` ground
    cost. Differentiable in ``a``, ``b`` and ``cost``; as ``epsilon -> 0`` the
    value approaches the exact (unregularised) optimal-transport cost.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("sinkhorn_distance expects 1-D marginals a, b")
    if cost.shape != (a.shape[0], b.shape[0]):
        raise ValueError(
            f"cost must have shape {(a.shape[0], b.shape[0])}, got {tuple(cost.shape)}"
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    log_a = jnp.log(a)
    log_b = jnp.log(b)
    f = jnp.zeros_like(a)
    g = jnp.zeros_like(b)
    for _ in range(num_iters):
        f = epsilon * (log_a - logsumexp((g[None, :] - cost) / epsilon, axis=1))
        g = epsilon * (log_b - logsumexp((f[:, None] - cost) / epsilon, axis=0))
    plan = jnp.exp((f[:, None] + g[None, :] - cost) / epsilon)
    out: Array = jnp.sum(plan * cost)
    return out


# ----- exponential families / information geometry --------------------------


def _glm_spec(base: str | JaxActivationSpec) -> JaxActivationSpec:
    spec = get_activation(base)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no derivative fastpath; cannot form "
            "the exponential-family cumulant tower"
        )
    return spec


def exponential_family_cumulants(
    theta: Array | float,
    *,
    base: str | JaxActivationSpec = "softplus",
    order: int = 2,
) -> Array:
    r"""Cumulant tower ``[kappa_0, ..., kappa_order]`` of a GLM log-partition ``A``.

    ``kappa_k = A^(k)(theta)``: mean (``k=1``), variance / Fisher information
    (``k=2``), ... stacked on a leading axis of size ``order + 1``.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    spec = _glm_spec(base)
    assert spec.fastpath is not None  # narrowed by _glm_spec
    t = _as_array(theta)
    rows = [spec.forward(t)]
    for k in range(1, order + 1):
        rows.append(spec.fastpath(t, k))
    out: Array = jnp.stack(rows, axis=0)
    return out


def glm_mean(
    theta: Array | float, *, base: str | JaxActivationSpec = "softplus"
) -> Array:
    r"""GLM mean ``kappa_1 = A'(theta)`` (the first cumulant / link inverse)."""
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    out: Array = spec.fastpath(_as_array(theta), 1)
    return out


def glm_variance(
    theta: Array | float, *, base: str | JaxActivationSpec = "softplus"
) -> Array:
    r"""GLM variance ``kappa_2 = A''(theta)`` (the second cumulant)."""
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    out: Array = spec.fastpath(_as_array(theta), 2)
    return out


def fisher_information(
    theta: Array | float, *, base: str | JaxActivationSpec = "softplus"
) -> Array:
    r"""Fisher information ``A''(theta)`` of the natural parameter (= :func:`glm_variance`)."""
    return glm_variance(theta, base=base)


def moment_match(
    mean: Array | float,
    *,
    base: str | JaxActivationSpec = "softplus",
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Array:
    r"""Natural parameter ``theta`` with ``A'(theta) == mean`` (the inverse link).

    Fisher-scoring / Newton iteration ``theta <- theta - (A'(theta) - mean) /
    A''(theta)`` whose step is the closed-form cumulant ratio (mean ``kappa_1`` /
    variance ``kappa_2``). For an exponential family this is the MLE (MLE ==
    moment matching). ``mean`` must lie in the interior of the mean range and may
    be scalar or a tensor (solved elementwise).
    """
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    mu = _as_array(mean)
    theta = jnp.zeros_like(mu)
    for _ in range(max_iter):
        step = (spec.fastpath(theta, 1) - mu) / spec.fastpath(theta, 2)
        theta = theta - step
        if float(jnp.max(jnp.abs(step))) <= tol:
            break
    if not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError(
            "moment_match failed to converge; is `mean` inside the GLM mean range?"
        )
    return theta


def fit_natural_parameter(
    samples: Array,
    *,
    base: str | JaxActivationSpec = "softplus",
    dim: int = -1,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Array:
    r"""MLE of the natural parameter from ``samples`` (mean over ``dim``).

    Reduces ``samples`` along ``dim`` and calls :func:`moment_match`; returns
    ``theta_hat`` (scalar for 1-D ``samples``, else batched).
    """
    mu = samples.mean(axis=dim)
    return moment_match(mu, base=base, max_iter=max_iter, tol=tol)


__all__ = [
    "SUPPORTED_W1_CDFS",
    "chi_squared_divergence",
    "cross_entropy",
    "entropy",
    "exponential_family_cumulants",
    "f_divergence",
    "fisher_information",
    "fit_natural_parameter",
    "glm_mean",
    "glm_variance",
    "hellinger_distance",
    "js_divergence",
    "kl_divergence",
    "moment_match",
    "mutual_information",
    "renyi_divergence",
    "renyi_entropy",
    "sinkhorn_distance",
    "sliced_wasserstein",
    "total_variation_distance",
    "tsallis_entropy",
    "wasserstein1",
    "wasserstein1_cdf",
    "wasserstein2_gaussian",
    "wassersteinp",
]

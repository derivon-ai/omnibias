# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Information theory + information geometry operators (PyTorch).

Three groups, all differentiable:

* **Discrete information theory** over distributions on the last axis:
  :func:`entropy`, :func:`cross_entropy`, :func:`kl_divergence`,
  :func:`js_divergence`, :func:`mutual_information`. All use ``xlogy`` so the
  ``0 log 0 = 0`` convention is exact and a ``q = 0`` where ``p > 0`` correctly
  yields ``+inf``.
* **Optimal transport** (the CDF-native distance): :func:`wasserstein1`, the 1-D
  Wasserstein-1 distance between two equal-size empirical samples, and
  :func:`wasserstein1_cdf`, the differentiable ``W_1`` between a ``sigmoid`` /
  ``tanh`` model CDF and the data (the companion of the certified
  :func:`omnibias.core.verified.transport.certified_wasserstein1`).
* **Information geometry / exponential families**: the closed-form derivative
  tower of a GLM log-partition activation *is* its cumulant tower, so
  :func:`exponential_family_cumulants` returns ``[A, A', A'', ...]`` exactly from
  one evaluation -- :func:`glm_mean` (``= A'``), :func:`glm_variance` and
  :func:`fisher_information` (``= A''``, the 1-D Fisher-Rao metric of the natural
  parameter).

Bit-identical twin: :mod:`omnibias.jax.information`.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.torch.activations.registry import ActivationSpec, get_activation

import torch
from torch import Tensor


def _as_tensor(x: Tensor | float) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


# ----- discrete information theory ------------------------------------------


def entropy(p: Tensor, *, dim: int = -1) -> Tensor:
    r"""Shannon entropy ``H(p) = -sum_i p_i ln p_i`` (nats) along ``dim``."""
    out: Tensor = -torch.xlogy(p, p).sum(dim=dim)
    return out


def cross_entropy(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Cross-entropy ``H(p, q) = -sum_i p_i ln q_i`` (nats) along ``dim``."""
    out: Tensor = -torch.xlogy(p, q).sum(dim=dim)
    return out


def kl_divergence(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Kullback-Leibler divergence ``D(p || q) = sum_i p_i ln(p_i / q_i)``."""
    out: Tensor = (torch.xlogy(p, p) - torch.xlogy(p, q)).sum(dim=dim)
    return out


def js_divergence(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Jensen-Shannon divergence ``0.5 D(p||m) + 0.5 D(q||m)``, ``m = (p+q)/2``.

    Symmetric and bounded in ``[0, ln 2]``; finite even where ``p`` or ``q`` has
    zeros.
    """
    m = 0.5 * (p + q)
    out: Tensor = 0.5 * kl_divergence(p, m, dim=dim) + 0.5 * kl_divergence(q, m, dim=dim)
    return out


def mutual_information(p_joint: Tensor) -> Tensor:
    r"""Mutual information ``I(X; Y)`` of a joint table over the last two axes.

    ``p_joint`` has shape ``(..., X, Y)`` and sums to 1 over the last two axes;
    ``I = sum p log(p / (p_x p_y))`` is reduced over those axes.
    """
    px = p_joint.sum(dim=-1, keepdim=True)
    py = p_joint.sum(dim=-2, keepdim=True)
    outer = px * py
    out: Tensor = (torch.xlogy(p_joint, p_joint) - torch.xlogy(p_joint, outer)).sum(
        dim=(-2, -1)
    )
    return out


# ----- generalized divergences ----------------------------------------------


def total_variation_distance(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Total-variation distance ``TV = 1/2 sum_i |p_i - q_i|`` along ``dim``.

    Symmetric, a proper metric, and bounded in ``[0, 1]``.
    """
    out: Tensor = 0.5 * (p - q).abs().sum(dim=dim)
    return out


def hellinger_distance(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Hellinger distance ``H = sqrt(1/2 sum_i (sqrt(p_i) - sqrt(q_i))^2)``.

    Symmetric, a proper metric, bounded in ``[0, 1]``; finite even where ``p`` or
    ``q`` has zeros.
    """
    out: Tensor = (0.5 * (p.sqrt() - q.sqrt()) ** 2).sum(dim=dim).sqrt()
    return out


def chi_squared_divergence(p: Tensor, q: Tensor, *, dim: int = -1) -> Tensor:
    r"""Pearson ``chi^2(p || q) = sum_i (p_i - q_i)^2 / q_i`` along ``dim``.

    Requires ``q_i > 0`` wherever evaluated; the local curvature of the KL
    divergence (twice the chi-square is its second-order expansion).
    """
    out: Tensor = ((p - q) ** 2 / q).sum(dim=dim)
    return out


def renyi_divergence(p: Tensor, q: Tensor, alpha: float, *, dim: int = -1) -> Tensor:
    r"""Renyi divergence ``D_alpha = 1/(alpha-1) ln sum_i p_i^alpha q_i^{1-alpha}``.

    Defined for ``alpha > 0``, ``alpha != 1``; the limit ``alpha -> 1`` is the KL
    divergence (:func:`kl_divergence`) and ``alpha = 1/2`` gives ``-2 ln(1 - H^2)``
    for the Hellinger affinity. ``q`` must cover the support of ``p``.
    """
    if alpha <= 0.0 or alpha == 1.0:
        raise ValueError(f"renyi_divergence needs alpha > 0 and alpha != 1, got {alpha}")
    s = (p**alpha * q ** (1.0 - alpha)).sum(dim=dim)
    out: Tensor = torch.log(s) / (alpha - 1.0)
    return out


def renyi_entropy(p: Tensor, alpha: float, *, dim: int = -1) -> Tensor:
    r"""Renyi entropy ``H_alpha = 1/(1-alpha) ln sum_i p_i^alpha`` (nats).

    Defined for ``alpha > 0``, ``alpha != 1``; recovers Shannon entropy as
    ``alpha -> 1``, the collision entropy at ``alpha = 2`` and the max-entropy
    ``ln |support|`` as ``alpha -> 0``.
    """
    if alpha <= 0.0 or alpha == 1.0:
        raise ValueError(f"renyi_entropy needs alpha > 0 and alpha != 1, got {alpha}")
    out: Tensor = torch.log((p**alpha).sum(dim=dim)) / (1.0 - alpha)
    return out


def tsallis_entropy(p: Tensor, order: float, *, dim: int = -1) -> Tensor:
    r"""Tsallis entropy ``S_q = (1 - sum_i p_i^q) / (q - 1)``.

    The non-extensive (``q``-deformed) entropy; ``order = q > 0``, ``q != 1``, with
    the Shannon entropy as the ``q -> 1`` limit.
    """
    if order <= 0.0 or order == 1.0:
        raise ValueError(f"tsallis_entropy needs order > 0 and order != 1, got {order}")
    out: Tensor = (1.0 - (p**order).sum(dim=dim)) / (order - 1.0)
    return out


def f_divergence(
    p: Tensor, q: Tensor, f: Callable[[Tensor], Tensor], *, dim: int = -1
) -> Tensor:
    r"""Generic Csiszar ``f``-divergence ``D_f(p || q) = sum_i q_i f(p_i / q_i)``.

    ``f`` is a convex function with ``f(1) = 0``; choosing ``f(t) = t ln t`` gives
    the KL divergence, ``f(t) = (t - 1)^2`` Pearson ``chi^2``,
    ``f(t) = |t - 1|/2`` total variation and ``f(t) = (sqrt(t) - 1)^2`` the squared
    Hellinger distance. ``q`` must be strictly positive.
    """
    out: Tensor = (q * f(p / q)).sum(dim=dim)
    return out


# ----- optimal transport ----------------------------------------------------


def wasserstein1(u: Tensor, v: Tensor) -> Tensor:
    r"""1-D Wasserstein-1 distance ``W_1 = int |F_u - F_v|`` between equal-size samples.

    For two 1-D samples of equal length this equals the mean absolute difference
    of their order statistics, ``mean_i |u_(i) - v_(i)|`` (exact).
    """
    if u.ndim != 1 or v.ndim != 1 or u.shape != v.shape:
        raise ValueError(
            f"wasserstein1 expects two 1-D samples of equal length, got "
            f"{tuple(u.shape)} and {tuple(v.shape)}"
        )
    us, _ = torch.sort(u)
    vs, _ = torch.sort(v)
    out: Tensor = (us - vs).abs().mean()
    return out


#: Location-scale CDF bases with a finite first moment (hence finite ``W_1``).
SUPPORTED_W1_CDFS: tuple[str, ...] = ("sigmoid", "tanh")


def wasserstein1_cdf(
    name: str,
    samples: Tensor,
    *,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Differentiable ``W_1(F, F_n)`` between a model CDF ``F`` and the data.

    The model-vs-empirical companion of :func:`wasserstein1`: ``F`` is the
    ``sigmoid`` / ``tanh`` location-scale CDF and ``F_n`` is the empirical CDF of
    ``samples`` (1-D). Computes the exact CDF integral
    ``W_1 = int |F - F_n| dx`` via the closed-form antiderivative
    ``Phi(x) = s * softplus((x - loc)/s)`` (two tails plus the per-panel crossing
    formula) -- the same construction as the certified
    :func:`omnibias.core.verified.transport.certified_wasserstein1`, but with
    differentiable tensors so it is usable as a fitting loss / regulariser in
    ``loc``, ``scale`` (learnable) and ``samples``. ``arctan`` is rejected (no
    finite mean). The certified enclosure encloses this value.
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
        s = _as_tensor(scale)
    elif key == "tanh":
        s = _as_tensor(scale) * 0.5  # tanh CDF == sigmoid CDF at half the scale
    else:
        raise NotImplementedError(
            f"wasserstein1_cdf supports {SUPPORTED_W1_CDFS} (finite first moment); "
            f"{name!r} has no finite mean so W1 is infinite"
        )
    loc_t = _as_tensor(loc)
    xs, _ = torch.sort(samples)
    softplus = torch.nn.functional.softplus

    def phi(x: Tensor) -> Tensor:
        return s * softplus((x - loc_t) / s)

    # tails: int_{-inf}^{x_(1)} F = Phi(x_(1)); int_{x_(n)}^{inf} (1 - F).
    total = phi(xs[0]) + s * softplus(-((xs[-1] - loc_t) / s))
    if n > 1:
        a = xs[:-1]
        b = xs[1:]
        i = torch.arange(1, n, dtype=xs.dtype, device=xs.device)
        c = i / n  # empirical CDF level on panel [x_(i), x_(i+1))
        xstar = loc_t + s * (torch.log(c) - torch.log1p(-c))  # F^{-1}(c)
        t = torch.minimum(torch.maximum(xstar, a), b)  # crossing clamped to panel
        panel = c * (2.0 * t - (a + b)) + phi(a) + phi(b) - 2.0 * phi(t)
        total = total + torch.clamp_min(panel, 0.0).sum()
    return total


def wassersteinp(u: Tensor, v: Tensor, *, p: float = 1.0) -> Tensor:
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
    us, _ = torch.sort(u)
    vs, _ = torch.sort(v)
    out: Tensor = ((us - vs).abs() ** p).mean() ** (1.0 / p)
    return out


def wasserstein2_gaussian(
    mu1: Tensor | float,
    sigma1: Tensor | float,
    mu2: Tensor | float,
    sigma2: Tensor | float,
) -> Tensor:
    r"""Closed-form 1-D Gaussian ``W_2`` distance.

    ``W_2(N(mu1, sigma1^2), N(mu2, sigma2^2)) = sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2)``
    (standard deviations ``sigma >= 0``). Differentiable in all four arguments --
    the matched-Gaussian objective used by the residual diagnostics; the certified
    twin is :func:`omnibias.core.verified.transport.certified_wasserstein2_gaussian`.
    """
    dmu = _as_tensor(mu1) - _as_tensor(mu2)
    dsigma = _as_tensor(sigma1) - _as_tensor(sigma2)
    out: Tensor = (dmu**2 + dsigma**2).sqrt()
    return out


def sliced_wasserstein(
    X: Tensor, Y: Tensor, directions: Tensor, *, p: float = 2.0
) -> Tensor:
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
    px, _ = torch.sort(X @ directions.T, dim=0)
    py, _ = torch.sort(Y @ directions.T, dim=0)
    per_dir = ((px - py).abs() ** p).mean(dim=0) ** (1.0 / p)
    out: Tensor = per_dir.mean()
    return out


def sinkhorn_distance(
    a: Tensor,
    b: Tensor,
    cost: Tensor,
    *,
    epsilon: float = 0.1,
    num_iters: int = 200,
) -> Tensor:
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
    log_a = torch.log(a)
    log_b = torch.log(b)
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(num_iters):
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - cost) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - cost) / epsilon, dim=0))
    plan = torch.exp((f[:, None] + g[None, :] - cost) / epsilon)
    out: Tensor = (plan * cost).sum()
    return out


# ----- exponential families / information geometry --------------------------


def _glm_spec(base: str | ActivationSpec[Tensor]) -> ActivationSpec[Tensor]:
    spec = get_activation(base)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no derivative fastpath; cannot form "
            "the exponential-family cumulant tower"
        )
    return spec


def exponential_family_cumulants(
    theta: Tensor | float,
    *,
    base: str | ActivationSpec[Tensor] = "softplus",
    order: int = 2,
) -> Tensor:
    r"""Cumulant tower ``[kappa_0, ..., kappa_order]`` of a GLM log-partition ``A``.

    With ``A = sigma`` (the ``base`` activation) the cumulants are its derivatives,
    ``kappa_k = A^(k)(theta)``: ``kappa_1`` is the mean, ``kappa_2`` the variance
    (Fisher information), ``kappa_3`` relates to skewness, etc. Returned stacked on
    a leading axis of size ``order + 1`` (shape ``(order+1,) + theta.shape``),
    from a single closed-form evaluation of the omnibias derivative tower.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    spec = _glm_spec(base)
    assert spec.fastpath is not None  # narrowed by _glm_spec
    t = _as_tensor(theta)
    rows = [spec.forward(t)]
    for k in range(1, order + 1):
        rows.append(spec.fastpath(t, k))
    out: Tensor = torch.stack(rows, dim=0)
    return out


def glm_mean(
    theta: Tensor | float, *, base: str | ActivationSpec[Tensor] = "softplus"
) -> Tensor:
    r"""GLM mean ``kappa_1 = A'(theta)`` (the first cumulant / link inverse)."""
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    out: Tensor = spec.fastpath(_as_tensor(theta), 1)
    return out


def glm_variance(
    theta: Tensor | float, *, base: str | ActivationSpec[Tensor] = "softplus"
) -> Tensor:
    r"""GLM variance ``kappa_2 = A''(theta)`` (the second cumulant)."""
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    out: Tensor = spec.fastpath(_as_tensor(theta), 2)
    return out


def fisher_information(
    theta: Tensor | float, *, base: str | ActivationSpec[Tensor] = "softplus"
) -> Tensor:
    r"""Fisher information ``A''(theta)`` of the natural parameter (= :func:`glm_variance`).

    For a 1-parameter exponential family the Fisher-Rao metric equals the
    variance ``kappa_2``; this is the information-geometry view of the same tower.
    """
    return glm_variance(theta, base=base)


def moment_match(
    mean: Tensor | float,
    *,
    base: str | ActivationSpec[Tensor] = "softplus",
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tensor:
    r"""Natural parameter ``theta`` with ``A'(theta) == mean`` (the inverse link).

    Solves the moment equation ``glm_mean(theta) = mean`` by Fisher-scoring /
    Newton iteration ``theta <- theta - (A'(theta) - mean) / A''(theta)`` -- the
    Newton step *is* the cumulant ratio, so each update uses the closed-form mean
    (``kappa_1``) and variance (``kappa_2``). For an exponential family this is
    also the maximum-likelihood estimate (MLE == moment matching). ``mean`` must
    lie in the interior of the mean range (e.g. ``(0, 1)`` for ``softplus`` /
    Bernoulli); it may be a scalar or a tensor (solved elementwise).
    """
    spec = _glm_spec(base)
    assert spec.fastpath is not None
    mu = _as_tensor(mean)
    theta = torch.zeros_like(mu)
    for _ in range(max_iter):
        step = (spec.fastpath(theta, 1) - mu) / spec.fastpath(theta, 2)
        theta = theta - step
        if float(step.detach().abs().max()) <= tol:
            break
    if not bool(torch.isfinite(theta).all()):
        raise ValueError(
            "moment_match failed to converge; is `mean` inside the GLM mean range?"
        )
    return theta


def fit_natural_parameter(
    samples: Tensor,
    *,
    base: str | ActivationSpec[Tensor] = "softplus",
    dim: int = -1,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tensor:
    r"""MLE of the natural parameter from ``samples`` (mean over ``dim``).

    For an exponential family the MLE equates the model mean to the sample mean,
    so this reduces ``samples`` along ``dim`` and calls :func:`moment_match`.
    Returns ``theta_hat`` (a scalar if ``samples`` is 1-D, else batched).
    """
    mu = samples.mean(dim=dim)
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

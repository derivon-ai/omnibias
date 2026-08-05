# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Residual / distributional diagnostics for symbolic discovery.

This module brings the omnibias **information theory** and **optimal transport**
operators into the symbolic-regression engines as *scoring* primitives. The
guiding principle (SINDy / minimum-description-length): a *correct* law leaves
**white, structureless, near-Gaussian** residuals, so a good model's residual
distribution has

* near-maximal (Gaussian-reference) differential entropy,
* near-zero divergence (``KL`` / ``W_1``) to a mean/variance-matched Gaussian, and
* near-zero mutual information with every input feature (no leftover dependence).

These functionals let information theory / optimal transport act either as an
opt-in **selection objective** or as an always-reported **fit diagnostic** inside
:func:`omnibias.symbolic.discover_interpretable_surrogate` and
:class:`omnibias.symbolic.NeuralJetDiscoverer`.

The discrete functionals (:func:`entropy`, :func:`kl_divergence`,
:func:`js_divergence`, :func:`mutual_information`) are the *numpy point-estimate
twins* of :mod:`omnibias.torch.information` / :mod:`omnibias.jax.information` --
they share the ``xlogy`` (``0 log 0 = 0``) convention bit-for-bit on a common
``pmf`` (parity-tested), so nothing is forked: only the histogram / Gaussian glue
needed to turn a residual sample into a distribution lives here.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

__all__ = [
    "DIVERGENCE_OBJECTIVES",
    "chi_squared_divergence",
    "chi_squared_to_gaussian",
    "design_conditioning_report",
    "differential_entropy",
    "divergence_objective_term",
    "entropy",
    "feature_residual_mutual_information",
    "gaussian_entropy",
    "hellinger_distance",
    "hellinger_to_gaussian",
    "histogram_pmf",
    "js_divergence",
    "js_to_gaussian",
    "kl_divergence",
    "kl_to_gaussian",
    "mutual_information",
    "renyi_divergence",
    "renyi_to_gaussian",
    "residual_dependence_report",
    "residual_distribution_report",
    "surrogate_residual_diagnostics",
    "total_variation_distance",
    "total_variation_to_gaussian",
    "wasserstein2_to_gaussian",
    "wasserstein_to_gaussian",
]


# ----- discrete functionals (numpy twins of omnibias.{torch,jax}.information) --


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""``x * ln(y)`` with the exact ``0 * ln(0) = 0`` convention.

    Mirrors ``torch.xlogy`` / ``jax.scipy.special.xlogy``: ``0`` wherever
    ``x == 0`` (even if ``y == 0``), and ``-inf`` where ``x > 0`` and ``y == 0``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prod = x * np.log(y)
    return np.where(x > 0.0, prod, 0.0)


def entropy(p: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Shannon entropy ``H(p) = -sum_i p_i ln p_i`` (nats) along ``axis``."""
    pa = np.asarray(p, dtype=float)
    return np.asarray(-_xlogy(pa, pa).sum(axis=axis), dtype=float)


def kl_divergence(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Kullback-Leibler divergence ``D(p || q) = sum_i p_i ln(p_i / q_i)``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.asarray((_xlogy(p, p) - _xlogy(p, q)).sum(axis=axis), dtype=float)


def js_divergence(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Jensen-Shannon divergence ``0.5 D(p||m) + 0.5 D(q||m)``, ``m = (p+q)/2``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, axis=axis) + 0.5 * kl_divergence(q, m, axis=axis)


def mutual_information(joint: np.ndarray) -> np.ndarray:
    r"""Mutual information ``I(X; Y)`` of a joint table over the last two axes."""
    joint = np.asarray(joint, dtype=float)
    px = joint.sum(axis=-1, keepdims=True)
    py = joint.sum(axis=-2, keepdims=True)
    value = (_xlogy(joint, joint) - _xlogy(joint, px * py)).sum(axis=(-2, -1))
    return np.asarray(value, dtype=float)


def total_variation_distance(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Total-variation distance ``1/2 sum_i |p_i - q_i|`` (twin of the backends)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.asarray(0.5 * np.abs(p - q).sum(axis=axis), dtype=float)


def hellinger_distance(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Hellinger distance ``sqrt(1/2 sum_i (sqrt(p_i) - sqrt(q_i))^2)`` (twin)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    value = np.sqrt((0.5 * (np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=axis))
    return np.asarray(value, dtype=float)


def chi_squared_divergence(p: np.ndarray, q: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""Pearson ``chi^2(p || q) = sum_i (p_i - q_i)^2 / q_i`` (twin; needs ``q > 0``)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.asarray(((p - q) ** 2 / q).sum(axis=axis), dtype=float)


def renyi_divergence(
    p: np.ndarray, q: np.ndarray, alpha: float, *, axis: int = -1
) -> np.ndarray:
    r"""Renyi divergence ``1/(alpha-1) ln sum_i p_i^alpha q_i^{1-alpha}`` (twin).

    ``alpha > 0``, ``alpha != 1``; the ``alpha -> 1`` limit is :func:`kl_divergence`.
    """
    if alpha <= 0.0 or alpha == 1.0:
        raise ValueError(f"renyi_divergence needs alpha > 0 and alpha != 1, got {alpha}")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    s = (p**alpha * q ** (1.0 - alpha)).sum(axis=axis)
    return np.asarray(np.log(s) / (alpha - 1.0), dtype=float)


# ----- distribution glue ----------------------------------------------------


def histogram_pmf(
    samples: np.ndarray,
    *,
    bins: int = 32,
    value_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical probability-mass function ``(pmf, edges)`` of ``samples``."""
    s = np.asarray(samples, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("histogram_pmf needs at least one sample")
    counts, edges = np.histogram(s, bins=bins, range=value_range)
    total = float(counts.sum())
    if total == 0.0:
        raise ValueError("histogram_pmf produced an empty histogram")
    return counts / total, edges


def gaussian_entropy(std: float) -> float:
    r"""Differential entropy ``0.5 ln(2 pi e sigma^2)`` of a Gaussian (nats)."""
    if std <= 0.0:
        raise ValueError(f"gaussian_entropy needs std > 0, got {std}")
    return 0.5 * math.log(2.0 * math.pi * math.e * std * std)


def differential_entropy(samples: np.ndarray, *, bins: int = 32) -> float:
    r"""Binned plug-in differential entropy ``-sum p_i ln(p_i / width)`` (nats)."""
    pmf, edges = histogram_pmf(samples, bins=bins)
    width = float(edges[1] - edges[0])
    return float(entropy(pmf)) + math.log(width)


def _gaussian_bin_masses(edges: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    nd = NormalDist(mu, sigma)
    cdf = np.array([nd.cdf(float(e)) for e in edges], dtype=float)
    return np.diff(cdf)


def _matched_gaussian_pmfs(
    samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""Binned ``(empirical pmf, matched-Gaussian pmf)`` over a shared grid.

    The Gaussian shares the empirical mean / std; its bin masses are floored at
    ``eps`` and renormalised (so divergences needing ``q > 0`` are well defined).
    Returns ``None`` for a degenerate (zero-variance) sample.
    """
    s = np.asarray(samples, dtype=float).reshape(-1)
    sigma = float(s.std())
    if sigma <= 0.0:
        return None
    pmf, edges = histogram_pmf(s, bins=bins)
    q = _gaussian_bin_masses(edges, float(s.mean()), sigma)
    q = np.clip(q, eps, None)
    q = q / q.sum()
    return pmf, q


def kl_to_gaussian(samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12) -> float:
    r"""``KL(empirical || matched Gaussian)`` -- a non-Gaussianity / structure score.

    ``0`` only when the (binned) residuals already match a Gaussian with the same
    mean and variance; positive otherwise. Returns ``0.0`` for a degenerate
    (zero-variance) residual.
    """
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(kl_divergence(pq[0], pq[1]))


def js_to_gaussian(samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12) -> float:
    r"""Jensen-Shannon divergence of the residuals to a matched Gaussian (in ``[0, ln 2]``)."""
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(js_divergence(pq[0], pq[1]))


def total_variation_to_gaussian(
    samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12
) -> float:
    r"""Total-variation distance of the residuals to a matched Gaussian (in ``[0, 1]``)."""
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(total_variation_distance(pq[0], pq[1]))


def hellinger_to_gaussian(
    samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12
) -> float:
    r"""Hellinger distance of the residuals to a matched Gaussian (in ``[0, 1]``)."""
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(hellinger_distance(pq[0], pq[1]))


def chi_squared_to_gaussian(
    samples: np.ndarray, *, bins: int = 32, eps: float = 1e-12
) -> float:
    r"""Pearson ``chi^2`` of the residuals to a matched Gaussian (``q`` floored at ``eps``)."""
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(chi_squared_divergence(pq[0], pq[1]))


def renyi_to_gaussian(
    samples: np.ndarray, alpha: float, *, bins: int = 32, eps: float = 1e-12
) -> float:
    r"""Renyi-``alpha`` divergence of the residuals to a matched Gaussian (``alpha != 1``)."""
    pq = _matched_gaussian_pmfs(samples, bins=bins, eps=eps)
    if pq is None:
        return 0.0
    return float(renyi_divergence(pq[0], pq[1], alpha))


def wasserstein_to_gaussian(samples: np.ndarray) -> float:
    r"""1-D Wasserstein-1 distance to a mean/variance-matched Gaussian.

    Computed exactly from the order statistics versus the Gaussian quantiles,
    ``(1/n) sum_i |x_(i) - Phi^{-1}((i+1/2)/n)||`` -- ``0`` for a degenerate
    residual, ``-> 0`` as Gaussian residuals grow.
    """
    s = np.asarray(samples, dtype=float).reshape(-1)
    n = s.size
    if n == 0:
        raise ValueError("wasserstein_to_gaussian needs at least one sample")
    sigma = float(s.std())
    if sigma <= 0.0:
        return 0.0
    nd = NormalDist(float(s.mean()), sigma)
    xs = np.sort(s)
    probs = (np.arange(n) + 0.5) / n
    gq = np.array([nd.inv_cdf(float(p)) for p in probs], dtype=float)
    return float(np.abs(xs - gq).mean())


def wasserstein2_to_gaussian(samples: np.ndarray) -> float:
    r"""1-D Wasserstein-2 distance to a mean/variance-matched Gaussian.

    Quadratic-cost companion of :func:`wasserstein_to_gaussian`: the
    root-mean-square of the order statistics against the matched Gaussian
    quantiles, ``(mean_i (x_(i) - Phi^{-1}((i+1/2)/n))^2)^{1/2}``. ``0`` for a
    degenerate residual.
    """
    s = np.asarray(samples, dtype=float).reshape(-1)
    n = s.size
    if n == 0:
        raise ValueError("wasserstein2_to_gaussian needs at least one sample")
    sigma = float(s.std())
    if sigma <= 0.0:
        return 0.0
    nd = NormalDist(float(s.mean()), sigma)
    xs = np.sort(s)
    probs = (np.arange(n) + 0.5) / n
    gq = np.array([nd.inv_cdf(float(p)) for p in probs], dtype=float)
    return float(np.sqrt(((xs - gq) ** 2).mean()))


def feature_residual_mutual_information(
    feature: np.ndarray,
    residuals: np.ndarray,
    *,
    bins: int = 16,
    bias_correction: bool = True,
) -> float:
    r"""Mutual information ``I(feature; residual)`` via a 2-D histogram (nats).

    Leftover dependence between an input and the model residual: ``~0`` for a
    correct model (residual independent of the input -- e.g. white noise),
    positive when structure remains. The plug-in estimator over-estimates ``I``
    for independent variables by roughly ``(B_x - 1)(B_y - 1) / 2N``; with
    ``bias_correction`` (default) the standard **Miller-Madow** term
    ``(m_x + m_y - m_xy - 1) / 2N`` (``m`` = occupied bin counts) is added and the
    result clamped to ``>= 0``, so genuinely independent residuals score ``~0``
    while real dependence (deterministic or noisy) is preserved.

    Note ``I`` is *scale-invariant*: a smooth residual that is a deterministic
    function of the input scores high even at negligible amplitude, so read it
    alongside the amplitude (``rmse`` / ``std``) and the scale-sensitive
    :func:`wasserstein_to_gaussian`.
    """
    f = np.asarray(feature, dtype=float).reshape(-1)
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if f.size == 0 or f.size != r.size:
        raise ValueError("feature and residuals must be non-empty and equal length")
    counts, _, _ = np.histogram2d(f, r, bins=bins)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    mi = float(mutual_information(counts / total))
    if bias_correction:
        m_x = int(np.count_nonzero(counts.sum(axis=1)))
        m_y = int(np.count_nonzero(counts.sum(axis=0)))
        m_xy = int(np.count_nonzero(counts))
        mi += (m_x + m_y - m_xy - 1) / (2.0 * total)
        mi = max(0.0, mi)
    return mi


# ----- residual reports + objective dispatch --------------------------------


def residual_distribution_report(residuals: np.ndarray, *, bins: int = 32) -> dict[str, float]:
    """Distributional summary of a residual vector (entropy / Gaussian divergences)."""
    r = np.asarray(residuals, dtype=float).reshape(-1)
    sigma = float(r.std())
    can_bin = r.size >= 2 and sigma > 0.0
    return {
        "rmse": float(np.sqrt(np.mean(r**2))) if r.size else float("nan"),
        "std": sigma,
        "differential_entropy": differential_entropy(r, bins=bins) if can_bin else float("nan"),
        "gaussian_reference_entropy": gaussian_entropy(sigma) if sigma > 0.0 else float("nan"),
        "kl_to_gaussian": kl_to_gaussian(r, bins=bins) if can_bin else 0.0,
        "wasserstein_to_gaussian": wasserstein_to_gaussian(r) if r.size else float("nan"),
    }


def residual_dependence_report(
    x: np.ndarray, residuals: np.ndarray, *, bins: int = 16
) -> dict[str, object]:
    """Per-feature mutual information between inputs and residuals (leftover structure)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    r = np.asarray(residuals, dtype=float).reshape(-1)
    mis = [feature_residual_mutual_information(x[:, j], r, bins=bins) for j in range(x.shape[1])]
    return {
        "feature_residual_mi": mis,
        "max_feature_residual_mi": max(mis) if mis else 0.0,
    }


def design_conditioning_report(design: np.ndarray) -> dict[str, float]:
    r"""How well-posed is the least-squares problem this library poses?

    Two condition numbers, because they answer different questions:

    * ``design_condition_number`` -- of the raw matrix. Dominated by column
      *scaling*, so it blows up as soon as columns live on different scales. An
      integral column grows with the domain while a derivative column does not,
      which is exactly that situation.
    * ``standardized_condition_number`` -- of the centered, unit-variance matrix,
      which is the space :func:`~omnibias.symbolic.discovery.fit_sparse_equation`
      actually thresholds in. Scaling has been divided out, so what is left is
      genuine near-**collinearity**: two columns carrying the same information.

    The second is the one that governs whether term selection is trustworthy. A
    large first number with a small second is benign (badly scaled but
    independent); a large second number means the fit cannot tell two candidate
    terms apart, and a sparse solver will pick between them on noise.

    ``max_column_scale_ratio`` reports the scale spread directly, since that is
    the readable form of the gap between the two.
    """
    a = np.asarray(design, dtype=float)
    if a.ndim != 2 or a.size == 0 or a.shape[1] == 0:
        return {
            "design_condition_number": float("nan"),
            "standardized_condition_number": float("nan"),
            "max_column_scale_ratio": float("nan"),
        }
    scales = a.std(axis=0)
    positive = scales[scales > 1e-12]
    ratio = (
        float(positive.max() / positive.min()) if positive.size else float("inf")
    )
    z = (a - a.mean(axis=0)) / np.where(scales < 1e-12, 1.0, scales)

    def cond(m: np.ndarray) -> float:
        try:
            return float(np.linalg.cond(m))
        except np.linalg.LinAlgError:  # pragma: no cover -- non-convergent SVD
            return float("inf")

    return {
        "design_condition_number": cond(a),
        "standardized_condition_number": cond(z),
        "max_column_scale_ratio": ratio,
    }


def surrogate_residual_diagnostics(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bins: int = 32,
    dependence_bins: int = 16,
) -> dict[str, object]:
    """Full residual report for a discovered model: distribution + dependence."""
    r = np.asarray(y_true, dtype=float).reshape(-1) - np.asarray(y_pred, dtype=float).reshape(-1)
    report: dict[str, object] = dict(residual_distribution_report(r, bins=bins))
    report.update(residual_dependence_report(x, r, bins=dependence_bins))
    return report


#: Divergence-based selection objectives usable by the discovery engines.
DIVERGENCE_OBJECTIVES: tuple[str, ...] = (
    "kl_gaussian",
    "js_gaussian",
    "tv_gaussian",
    "hellinger_gaussian",
    "chi2_gaussian",
    "wasserstein_gaussian",
    "wasserstein2_gaussian",
    "residual_mi",
)


def divergence_objective_term(
    name: str,
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    bins: int = 32,
    dependence_bins: int = 16,
) -> float:
    r"""One scalar penalty (``>= 0``) for a divergence-aware selection objective.

    Distance of the residual distribution to a mean/variance-matched Gaussian
    (white-residual reference), or leftover input dependence:

    * ``"kl_gaussian"`` -- ``KL(residual || Gaussian)``;
    * ``"js_gaussian"`` -- Jensen-Shannon divergence to the Gaussian;
    * ``"tv_gaussian"`` -- total-variation distance to the Gaussian;
    * ``"hellinger_gaussian"`` -- Hellinger distance to the Gaussian;
    * ``"chi2_gaussian"`` -- Pearson ``chi^2`` to the Gaussian;
    * ``"wasserstein_gaussian"`` -- ``W_1`` to the Gaussian;
    * ``"wasserstein2_gaussian"`` -- ``W_2`` to the Gaussian;
    * ``"residual_mi"`` -- the maximum mutual information between any input feature
      and the residual.

    All are minimised (``-> 0``) by a model whose residuals are white and
    Gaussian, so adding ``weight * term`` to a validation-RMSE score steers the
    search toward genuinely structure-free fits.
    """
    if name == "kl_gaussian":
        return kl_to_gaussian(residuals, bins=bins)
    if name == "js_gaussian":
        return js_to_gaussian(residuals, bins=bins)
    if name == "tv_gaussian":
        return total_variation_to_gaussian(residuals, bins=bins)
    if name == "hellinger_gaussian":
        return hellinger_to_gaussian(residuals, bins=bins)
    if name == "chi2_gaussian":
        return chi_squared_to_gaussian(residuals, bins=bins)
    if name == "wasserstein_gaussian":
        return wasserstein_to_gaussian(residuals)
    if name == "wasserstein2_gaussian":
        return wasserstein2_to_gaussian(residuals)
    if name == "residual_mi":
        report = residual_dependence_report(x, residuals, bins=dependence_bins)
        return float(report["max_feature_residual_mi"])  # type: ignore[arg-type]
    raise ValueError(
        f"unknown divergence objective {name!r}; supported: {DIVERGENCE_OBJECTIVES}"
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sliced optimal transport of activation mixtures (theory 03-04).

A tempered-activation mixture has a closed-form CDF (the founding bias collapse,
``delta -> 0``, supplies ``sigma'`` and its antiderivative ``sigma``).
One-dimensional ``W_1`` is an integral of ``|F-G|`` split at sign-change roots.
Temperature collapse (``beta -> inf``, feasibility) does not appear. Do not conflate the two.
Exact per slice; the average over directions is sampled (``O(1/L)``) and is
not sample-free. Sliced Wasserstein is not Wasserstein.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from omnibias.fields._core.quadrature import gauss_legendre

FloatArray = NDArray[np.float64]
_SUPPORTED = ("logistic", "sigmoid", "tanh")
_WORKED_W1 = 2.0 * (
    -math.log(2.0)
    + 0.5 * math.log(1.0 + math.e)
    + 0.5 * math.log(1.0 + math.exp(-1.0))
)


def _sigmoid(z: object) -> FloatArray:
    x = np.asarray(z, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softplus(z: object) -> FloatArray:
    x = np.asarray(z, dtype=np.float64)
    return np.logaddexp(0.0, x)


def _sig_prime(z: object) -> FloatArray:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _as_alpha(base: str, scales: FloatArray) -> FloatArray:
    if base == "tanh":
        return 2.0 * scales
    return scales


@dataclass(frozen=True)
class ActivationMixture:
    """Isotropic tempered-activation mixture. Closed under 1-D projection."""

    weights: FloatArray
    means: FloatArray
    scales: FloatArray
    base: str = "logistic"

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        m = np.asarray(self.means, dtype=np.float64)
        a = np.asarray(self.scales, dtype=np.float64).reshape(-1)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if w.size != m.shape[0] or a.size != m.shape[0]:
            raise ValueError("weights, means, and scales must share a component axis")
        if np.any(w < 0.0) or abs(float(np.sum(w)) - 1.0) > 1e-12:
            raise ValueError("weights must be nonnegative and sum to 1")
        if np.any(a <= 0.0):
            raise ValueError("scales (alpha) must be > 0")
        key = str(self.base).lower()
        if key not in _SUPPORTED:
            raise ValueError(f"base must be one of {_SUPPORTED}, got {self.base!r}")
        if key == "sigmoid":
            key = "logistic"
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "means", m)
        object.__setattr__(self, "scales", a)
        object.__setattr__(self, "base", key)

    @property
    def n_components(self) -> int:
        return int(self.weights.size)

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def alpha(self) -> FloatArray:
        return _as_alpha(self.base, self.scales)

    def require_1d(self) -> None:
        if self.dim != 1:
            raise ValueError("cdf / pdf / W_1 need a 1-D mixture; project first")

    def loc(self) -> FloatArray:
        return self.means.reshape(self.n_components)

    def mean(self) -> float:
        self.require_1d()
        return float(np.dot(self.weights, self.loc()))

    def cdf(self, x: object) -> FloatArray:
        self.require_1d()
        xv = np.asarray(x, dtype=np.float64)
        z = self.alpha * (xv[..., None] - self.loc())
        return np.sum(self.weights * _sigmoid(z), axis=-1)

    def pdf(self, x: object) -> FloatArray:
        self.require_1d()
        xv = np.asarray(x, dtype=np.float64)
        z = self.alpha * (xv[..., None] - self.loc())
        return np.sum(self.weights * self.alpha * _sig_prime(z), axis=-1)

    def cdf_antiderivative(self, x: object) -> FloatArray:
        """``int^x F`` with ``Phi(-inf) = 0``. Logistic: ``(c/alpha) softplus``."""
        self.require_1d()
        xv = np.asarray(x, dtype=np.float64)
        z = self.alpha * (xv[..., None] - self.loc())
        return np.sum(self.weights / self.alpha * _softplus(z), axis=-1)

    def project(self, direction: object) -> ActivationMixture:
        w = np.asarray(direction, dtype=np.float64).reshape(-1)
        if w.size != self.dim:
            raise ValueError(f"direction has size {w.size}, mixture dim is {self.dim}")
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-15:
            raise ValueError("direction must be nonzero")
        unit = w / nrm
        locs = self.means @ unit
        return ActivationMixture(self.weights, locs, self.scales, base=self.base)


@dataclass(frozen=True)
class SlicedResult:
    """Exact per slice; ``direction_stderr`` is the remaining sampling error."""

    value: float
    direction_stderr: float
    n_directions: int
    p: int
    slices: FloatArray


def honesty_payload() -> dict[str, bool]:
    return {
        "sample_free": False,
        "exact_per_slice": True,
        "sliced_is_wasserstein": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
    }


def named_worked_pair() -> tuple[ActivationMixture, ActivationMixture]:
    """Spec §5: two logistic bumps at ``±1`` versus one at ``0``."""
    mu = ActivationMixture([0.5, 0.5], [-1.0, 1.0], [1.0, 1.0])
    nu = ActivationMixture([1.0], [0.0], [1.0])
    return mu, nu


def worked_w1() -> float:
    return float(_WORKED_W1)


def _diff_cdf(mu: ActivationMixture, nu: ActivationMixture, x: object) -> FloatArray:
    return mu.cdf(x) - nu.cdf(x)


def _diff_anti(mu: ActivationMixture, nu: ActivationMixture, x: object) -> FloatArray:
    return mu.cdf_antiderivative(x) - nu.cdf_antiderivative(x)


def sign_change_roots(
    mu: ActivationMixture,
    nu: ActivationMixture,
    *,
    n_grid: int = 512,
    pad: float = 16.0,
) -> FloatArray:
    """Bracket ``F-G`` on a padded grid and refine by bisection."""
    mu.require_1d()
    nu.require_1d()
    locs = np.concatenate([mu.loc(), nu.loc()])
    amin = float(min(np.min(mu.alpha), np.min(nu.alpha)))
    lo = float(np.min(locs) - pad / amin)
    hi = float(np.max(locs) + pad / amin)
    grid = np.linspace(lo, hi, int(n_grid), dtype=np.float64)
    vals = _diff_cdf(mu, nu, grid)
    roots: list[float] = []
    for i in range(grid.size - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        if fa == 0.0:
            roots.append(a)
            continue
        if fa * fb > 0.0:
            continue
        left, right, fleft = a, b, fa
        for _ in range(80):
            mid = 0.5 * (left + right)
            fm = float(_diff_cdf(mu, nu, mid))
            if fleft * fm <= 0.0:
                right = mid
            else:
                left, fleft = mid, fm
            if right - left < 1e-15:
                break
        roots.append(0.5 * (left + right))
    # Dedup clustered roots (nearly tangent CDFs).
    uniq: list[float] = []
    for r in roots:
        if not uniq or abs(r - uniq[-1]) > 1e-10:
            uniq.append(r)
    return np.asarray(uniq, dtype=np.float64)


def krawczyk_root_count(
    mu: ActivationMixture,
    nu: ActivationMixture,
    roots: FloatArray,
) -> int:
    """IVT-certified sign-change count, plus a 1-D Krawczyk uniqueness attempt.

    Each isolated sign change is a guaranteed root (intermediate-value
    theorem). Krawczyk uniqueness is attempted when ``F'-G'`` is bounded
    away from zero; a failed uniqueness test does not drop the IVT count.
    """
    from omnibias.core.verified.interval import Interval
    from omnibias.core.verified.kantorovich import krawczyk_certificate
    from omnibias.core.verified.transcend import sigmoid_iv

    def cdf_iv(mix: ActivationMixture, x: Interval) -> Interval:
        acc = Interval.point(0.0)
        for c, loc, a in zip(mix.weights, mix.loc(), mix.alpha, strict=True):
            z = (x - Interval.point(float(loc))) * Interval.point(float(a))
            acc = acc + Interval.point(float(c)) * sigmoid_iv(z)
        return acc

    def pdf_iv(mix: ActivationMixture, x: Interval) -> Interval:
        acc = Interval.point(0.0)
        for c, loc, a in zip(mix.weights, mix.loc(), mix.alpha, strict=True):
            z = (x - Interval.point(float(loc))) * Interval.point(float(a))
            s = sigmoid_iv(z)
            acc = acc + Interval.point(float(c) * float(a)) * s * (Interval.point(1.0) - s)
        return acc

    def func(xs: Sequence[Interval]) -> list[Interval]:
        return [cdf_iv(mu, xs[0]) - cdf_iv(nu, xs[0])]

    def jac(xs: Sequence[Interval]) -> list[list[Interval]]:
        return [[pdf_iv(mu, xs[0]) - pdf_iv(nu, xs[0])]]

    for r in roots:
        deriv = float(mu.pdf(r) - nu.pdf(r))
        if abs(deriv) < 1e-12:
            continue
        krawczyk_certificate(func, jac, [float(r)], [[1.0 / deriv]], 1e-5)
    return int(roots.size)


def w1_exact(mu: ActivationMixture, nu: ActivationMixture) -> float:
    """Exact 1-D ``W_1`` via sign-change roots and closed-form antiderivatives."""
    roots = sign_change_roots(mu, nu)
    mean_gap = nu.mean() - mu.mean()
    points = [-math.inf, *[float(r) for r in roots], math.inf]
    total = 0.0
    for a, b in zip(points[:-1], points[1:], strict=True):
        if math.isinf(a) and math.isinf(b):
            continue
        if math.isinf(a):
            s_a = 0.0
        else:
            s_a = float(_diff_anti(mu, nu, a))
        s_b = mean_gap if math.isinf(b) else float(_diff_anti(mu, nu, b))
        total += abs(s_b - s_a)
    return float(total)


def quantile(mix: ActivationMixture, q: object, *, steps: int = 40) -> FloatArray:
    """Newton inversion of ``F``. Derivative is ``1 / rho(Q)`` (closed form)."""
    mix.require_1d()
    qq = np.asarray(q, dtype=np.float64)
    if np.any((qq <= 0.0) | (qq >= 1.0)):
        raise ValueError("quantile q must lie in (0, 1)")
    x = np.full_like(qq, mix.mean(), dtype=np.float64)
    spread = 1.0 / float(np.mean(mix.alpha))
    x = x + spread * (2.0 * qq - 1.0)
    for _ in range(int(steps)):
        dens = mix.pdf(x)
        dens = np.maximum(dens, 1e-30)
        x = x - (mix.cdf(x) - qq) / dens
    return x


def quantile_dtheta_means(mix: ActivationMixture, q: float) -> FloatArray:
    """``d Q / d mu_g = - (dF/d mu_g) / rho`` at ``x = Q(q)``."""
    x = float(quantile(mix, q))
    rho = float(mix.pdf(x))
    z = mix.alpha * (x - mix.loc())
    dF_dmu = mix.weights * mix.alpha * (-_sig_prime(z))
    return -dF_dmu / max(rho, 1e-30)


def wp_quantile(
    mu: ActivationMixture,
    nu: ActivationMixture,
    *,
    p: int = 2,
    n_quad: int = 48,
) -> float:
    """``W_p`` via Gauss-Legendre in ``q`` and Newton quantiles."""
    if int(p) < 1:
        raise ValueError("p must be >= 1")
    spec = gauss_legendre([(1e-8, 1.0 - 1e-8)], int(n_quad))
    qs = spec.nodes.reshape(-1)
    wts = spec.weights.reshape(-1)
    delta = np.abs(quantile(mu, qs) - quantile(nu, qs))
    moment = float(np.dot(wts, np.power(delta, float(p))))
    return moment ** (1.0 / float(p))


def _sphere(rng: Generator, n: int, dim: int) -> FloatArray:
    z = rng.normal(size=(int(n), int(dim)))
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-15)
    return z / nrm


def sliced_wasserstein(
    mu: ActivationMixture,
    nu: ActivationMixture,
    *,
    p: int = 1,
    directions: int = 32,
    seed: int | None = None,
) -> SlicedResult:
    """Average of exact 1-D slices. ``direction_stderr`` is always filled."""
    if mu.dim != nu.dim:
        raise ValueError("mixtures must share dimension")
    rng = np.random.default_rng(seed)
    if mu.dim == 1:
        dirs = np.array([[1.0], [-1.0]], dtype=np.float64)
    else:
        dirs = _sphere(rng, int(directions), mu.dim)
    slices = np.empty(dirs.shape[0], dtype=np.float64)
    for i, w in enumerate(dirs):
        a, b = mu.project(w), nu.project(w)
        slices[i] = w1_exact(a, b) if int(p) == 1 else wp_quantile(a, b, p=int(p))
    value = float(np.mean(slices))
    if slices.size < 2:
        stderr = 0.0
    else:
        stderr = float(np.std(slices, ddof=1) / math.sqrt(slices.size))
    return SlicedResult(
        value=value,
        direction_stderr=stderr,
        n_directions=int(slices.size),
        p=int(p),
        slices=slices,
    )


def bootstrap_direction_stderr(
    slices: object,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> float:
    """Independent bootstrap SE of the slice mean (G5 check)."""
    s = np.asarray(slices, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        means[i] = float(np.mean(rng.choice(s, size=s.size, replace=True)))
    return float(np.std(means, ddof=1))


def sample_mixture(mix: ActivationMixture, n: int, rng: Generator) -> FloatArray:
    """Inverse-CDF samples of a 1-D logistic / tanh mixture (MC baseline)."""
    mix.require_1d()
    idx = rng.choice(mix.n_components, size=int(n), p=mix.weights)
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=int(n))
    # Logistic quantile: mu + (1/alpha) log(u/(1-u)); tanh uses alpha_eff = 2 alpha.
    z = np.log(u / (1.0 - u))
    return mix.loc()[idx] + z / mix.alpha[idx]


def mc_w1(x: object, y: object) -> float:
    """Empirical 1-D ``W_1`` between equal-length samples (sorted mean abs)."""
    a = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if a.size != b.size:
        raise ValueError("mc_w1 needs equal-length samples")
    return float(np.mean(np.abs(a - b)))


def mc_sliced_wasserstein(
    mu: ActivationMixture,
    nu: ActivationMixture,
    *,
    n_samples: int,
    directions: int = 32,
    seed: int | None = None,
) -> SlicedResult:
    """Monte Carlo sliced ``W_1`` (particle noise + direction noise)."""
    rng = np.random.default_rng(seed)
    if mu.dim == 1:
        dirs = np.array([[1.0]], dtype=np.float64)
    else:
        dirs = _sphere(rng, int(directions), mu.dim)
    slices = np.empty(dirs.shape[0], dtype=np.float64)
    for i, w in enumerate(dirs):
        xs = sample_mixture(mu.project(w), int(n_samples), rng)
        ys = sample_mixture(nu.project(w), int(n_samples), rng)
        slices[i] = mc_w1(xs, ys)
    value = float(np.mean(slices))
    stderr = 0.0 if slices.size < 2 else float(np.std(slices, ddof=1) / math.sqrt(slices.size))
    return SlicedResult(
        value=value,
        direction_stderr=stderr,
        n_directions=int(slices.size),
        p=1,
        slices=slices,
    )


def highprec_w1(
    mu: ActivationMixture,
    nu: ActivationMixture,
    *,
    n_quad: int = 64,
    pad: float = 80.0,
    chunk: float = 3.0,
) -> float:
    """Independent Gauss-Legendre integral of ``|F-G|`` (G1 oracle)."""
    roots = sign_change_roots(mu, nu)
    locs = np.concatenate([mu.loc(), nu.loc()])
    amin = float(min(np.min(mu.alpha), np.min(nu.alpha)))
    lo = float(np.min(locs) - pad / amin)
    hi = float(np.max(locs) + pad / amin)
    cuts = np.unique(np.concatenate(([lo], roots, [hi])))
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if b <= a + 1e-15:
            continue
        n_chunk = max(1, int(math.ceil((float(b) - float(a)) / float(chunk))))
        edges = np.linspace(float(a), float(b), n_chunk + 1, dtype=np.float64)
        for u, v in zip(edges[:-1], edges[1:], strict=True):
            spec = gauss_legendre([(float(u), float(v))], int(n_quad))
            xs = spec.nodes.reshape(-1)
            wts = spec.weights.reshape(-1)
            total += float(np.dot(wts, np.abs(_diff_cdf(mu, nu, xs))))
    return float(total)


__all__ = [
    "ActivationMixture",
    "SlicedResult",
    "bootstrap_direction_stderr",
    "honesty_payload",
    "krawczyk_root_count",
    "mc_sliced_wasserstein",
    "mc_w1",
    "named_worked_pair",
    "quantile",
    "quantile_dtheta_means",
    "sample_mixture",
    "sign_change_roots",
    "sliced_wasserstein",
    "highprec_w1",
    "w1_exact",
    "worked_w1",
    "wp_quantile",
]

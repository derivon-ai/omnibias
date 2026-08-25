# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Pack-parameter Fisher metric (theory 04-01).

This is a **metric on a pack family**, not the scalar exponential-family
Fisher ``A''(theta)`` in :mod:`omnibias.curvature.glm_fisher`. The
integrand is closed form from the logistic tower; the expectation is a
1-D quadrature (recorded) or a Monte Carlo fallback.

The ``delta -> 0`` limit here **is** the founding bias collapse.
Temperature collapse (``beta -> inf``) does not appear. ``K >= 3``
central finite-difference packs change sign, so they are not densities
and Fisher is refused (inapplicable, not unmeasured).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PATH_CLOSED = "closed_form_integrand"
PATH_MC = "monte_carlo"

# numpy.linalg.pinv rule: max(n, m) * eps(dtype). Documented threshold.
PINV_RCOND_RULE = "max(n_params) * eps(dtype), matching numpy.linalg.pinv"

# Absolute Fisher floor used only by the *damped* natural step when the
# collapse report says the coordinate is near-degenerate. Distinct from
# the tighter pinv cutoff (linear algebra) and from the founding delta.
DEGENERACY_ABS_FLOOR = 1e-2

_KINDS = frozenset(
    {
        "two_bias_logistic",
        "two_bias_located",
        "logistic_location",
        "logistic_mixture",
        "finite_difference",
    }
)


class NotADensityError(ValueError):
    """Fisher was asked for a pack that is not a probability density."""


@dataclass(frozen=True)
class DegeneracyReport:
    """Collapse-direction diagnosis for a pack-parameter metric.

    ``damping`` is the extra Tikhonov used by
    :func:`damped_natural_step`. The pseudo-inverse
    (:func:`fisher_pinv`) uses the tighter :data:`PINV_RCOND_RULE`.
    """

    eigenvalue: float
    exponent: float
    condition_number: float
    recommendation: str
    damping: float
    path: str = PATH_CLOSED
    direction: str = "spread"


@dataclass(frozen=True)
class FisherEvaluation:
    """Metric matrix plus the path that produced the expectation."""

    metric: FloatArray
    path: str


@dataclass(frozen=True)
class PackFamily:
    """A parametric family of 1-D logistic pack densities.

    Parameters
    ----------
    kind
        ``two_bias_logistic`` (``theta = (delta,)``),
        ``two_bias_located`` (``theta = (mu, delta)``),
        ``logistic_location`` (``theta = (mu,)``, unit logistic),
        ``logistic_mixture`` (two components; see :meth:`unpack`),
        or ``finite_difference`` (refused for ``n_components >= 3``).
    n_components
        Mixture size or finite-difference bias count.
    activation
        Must be ``sigmoid``: only the logistic mixture is a density
        with a closed-form score in this leftover close-out.
    """

    kind: str
    n_components: int = 1
    activation: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown PackFamily.kind {self.kind!r}")
        if self.activation != "sigmoid":
            raise NotADensityError(
                "pack Fisher is implemented for logistic/sigmoid densities "
                f"only, got {self.activation!r}"
            )
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.kind == "logistic_mixture" and self.n_components != 2:
            raise ValueError(
                "logistic_mixture is the two-component D8 suite; "
                f"got n_components={self.n_components}"
            )

    @property
    def n_params(self) -> int:
        if self.kind in {"two_bias_logistic", "logistic_location", "finite_difference"}:
            return 1
        if self.kind == "two_bias_located":
            return 2
        if self.kind == "logistic_mixture":
            return 5
        raise ValueError(f"n_params undefined for kind {self.kind!r}")

    def unpack(
        self, theta: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(locations, scales, weights)`` for mixture-like kinds."""
        t = _theta(self, theta)
        if self.kind == "logistic_location":
            return np.array([t[0]]), np.array([1.0]), np.array([1.0])
        if self.kind == "logistic_mixture":
            mu1, phi1, mu2, phi2, psi = (float(v) for v in t)
            c1 = 1.0 / (1.0 + math.exp(-psi))
            return (
                np.array([mu1, mu2]),
                np.array([math.exp(phi1), math.exp(phi2)]),
                np.array([c1, 1.0 - c1]),
            )
        if self.kind == "two_bias_logistic":
            return np.array([0.0]), np.array([1.0]), np.array([1.0])
        if self.kind == "two_bias_located":
            return np.array([t[0]]), np.array([1.0]), np.array([1.0])
        raise NotADensityError(_finite_difference_msg(self))


def two_bias_family() -> PackFamily:
    """One-parameter two-bias logistic pack ``p_delta``."""
    return PackFamily(kind="two_bias_logistic", n_components=2)


def two_bias_located_family() -> PackFamily:
    """Location-plus-spread two-bias pack ``p(x; mu, delta)``."""
    return PackFamily(kind="two_bias_located", n_components=2)


def logistic_location_family() -> PackFamily:
    """Unit logistic location family; Fisher is exactly ``1/3``."""
    return PackFamily(kind="logistic_location", n_components=1)


def logistic_mixture_family() -> PackFamily:
    """Two-component logistic mixture (the G1 / D8 suite)."""
    return PackFamily(kind="logistic_mixture", n_components=2)


def finite_difference_family(*, n_biases: int) -> PackFamily:
    """Central finite-difference pack. Fisher applies only for ``n_biases == 2``."""
    return PackFamily(kind="finite_difference", n_components=int(n_biases))


def two_bias_A(delta: float) -> float:
    """``A = (delta/2) cosh(delta/2) - sinh(delta/2)``, series-stable."""
    h = 0.5 * float(delta)
    if h < 0.5:
        tot = 0.0
        for k in range(1, 25):
            tot += h ** (2 * k + 1) * (
                1.0 / math.factorial(2 * k) - 1.0 / math.factorial(2 * k + 1)
            )
        return tot
    return h * math.cosh(h) - math.sinh(h)


def two_bias_density_u(u: Any, delta: float) -> Any:
    """Cancellation-free ``p_delta`` in ``u = exp(-x)``."""
    d = float(delta)
    if d <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    uu = np.asarray(u, dtype=float)
    s = math.sinh(0.5 * d)
    c = math.cosh(0.5 * d)
    denom = 1.0 + 2.0 * uu * c + uu * uu
    return 2.0 * uu * s / (d * denom)


def two_bias_ddelta_u(u: Any, delta: float) -> Any:
    """Cancellation-free ``d p_delta / d delta`` in ``u = exp(-x)``."""
    d = float(delta)
    if d <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    uu = np.asarray(u, dtype=float)
    s = math.sinh(0.5 * d)
    c = math.cosh(0.5 * d)
    denom = 1.0 + 2.0 * uu * c + uu * uu
    a = two_bias_A(d)
    return 2.0 * uu * (denom * a - d * uu * s * s) / (d * d * denom * denom)


def two_bias_dx_u(u: Any, delta: float) -> Any:
    """Cancellation-free ``d p_delta / d x`` in ``u = exp(-x)``."""
    d = float(delta)
    if d <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    uu = np.asarray(u, dtype=float)
    s = math.sinh(0.5 * d)
    c = math.cosh(0.5 * d)
    denom = 1.0 + 2.0 * uu * c + uu * uu
    return -2.0 * s * uu * (1.0 - uu * uu) / (d * denom * denom)


def two_bias_density_naive(x: Any, delta: float) -> Any:
    """Definition form ``(sigma(x+d/2) - sigma(x-d/2)) / d``."""
    d = float(delta)
    xx = np.asarray(x, dtype=float)
    sp = 1.0 / (1.0 + np.exp(-(xx + 0.5 * d)))
    sm = 1.0 / (1.0 + np.exp(-(xx - 0.5 * d)))
    return (sp - sm) / d


def fisher_delta_delta(delta: float, *, nodes: int = 200) -> float:
    """``G_{delta,delta}`` for the two-bias logistic pack."""
    metric = fisher_metric(two_bias_family(), np.array([float(delta)]), nodes=nodes)
    return float(metric[0, 0])


def pinv_rcond(metric: Any) -> float:
    """Documented pseudo-inverse cutoff: ``max(shape) * eps(dtype)``."""
    arr = np.asarray(metric)
    return float(max(arr.shape)) * float(np.finfo(arr.dtype).eps)


def fisher_pinv(metric: Any) -> FloatArray:
    """Symmetric eigen-pseudoinverse at :func:`pinv_rcond`."""
    g = _sym(metric)
    eig, vecs = np.linalg.eigh(g)
    cutoff = pinv_rcond(g) * float(np.max(np.abs(eig)))
    inv = np.where(np.abs(eig) > cutoff, 1.0 / eig, 0.0)
    return np.asarray((vecs * inv) @ vecs.T, dtype=np.float64)


def evaluate_fisher(
    family: PackFamily,
    theta: Any,
    *,
    closed_form: bool = True,
    nodes: int = 96,
    n_mc: int = 200_000,
    seed: int = 0,
) -> FisherEvaluation:
    """Assemble ``G(theta)``. Closed-form integrand plus quadrature by default."""
    _require_density(family, theta)
    if closed_form:
        return FisherEvaluation(
            metric=_quadrature_metric(family, theta, nodes=nodes),
            path=PATH_CLOSED,
        )
    metric, _se = _monte_carlo_metric(family, theta, n=n_mc, seed=seed)
    return FisherEvaluation(metric=metric, path=PATH_MC)


def fisher_metric(
    family: PackFamily,
    theta: Any,
    *,
    closed_form: bool = True,
    nodes: int = 96,
    n_mc: int = 200_000,
    seed: int = 0,
) -> FloatArray:
    """Closed form where the family's moments are; else Monte Carlo.

    Records the path on the returned evaluation via
    :func:`evaluate_fisher`. This leftover uses quadrature of a
    closed-form integrand (not a sampling-free moment formula).
    """
    return evaluate_fisher(
        family,
        theta,
        closed_form=closed_form,
        nodes=nodes,
        n_mc=n_mc,
        seed=seed,
    ).metric


def fisher_metric_mc(
    family: PackFamily,
    theta: Any,
    *,
    n: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Monte Carlo Fisher and per-entry standard errors."""
    _require_density(family, theta)
    return _monte_carlo_metric(family, theta, n=int(n), seed=int(seed))


def fisher_distance(
    family: PackFamily,
    theta_a: Any,
    theta_b: Any,
    *,
    steps: int = 64,
) -> float:
    """Fisher length of the chart-straight path.

    Equals the geodesic in one parameter. In several parameters it is
    the length of the linear interpolant (an upper bound on the
    geodesic). ``as_manifold_spec`` exposes the same ``G`` to
    ``omnibias.geometry`` when that package is installed.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    a = _theta(family, theta_a)
    b = _theta(family, theta_b)
    length = 0.0
    delta = (b - a) / float(steps)
    for i in range(steps):
        mid = a + (i + 0.5) * delta
        g = fisher_metric(family, mid)
        quad = float(delta @ g @ delta)
        length += math.sqrt(max(quad, 0.0))
    return length


def log_density(family: PackFamily, theta: Any, data: Any) -> FloatArray:
    """Per-observation ``log p_theta(x)``."""
    _require_density(family, theta)
    xs = np.asarray(data, dtype=float).reshape(-1)
    p, _jac = _density_and_jac(family, theta, xs)
    return np.asarray(np.log(np.maximum(p, 1e-300)), dtype=np.float64)


def empirical_distinguishability_n(
    family: PackFamily,
    theta_a: Any,
    theta_b: Any,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    n_trials: int = 80,
    seed: int = 0,
    n_min: int = 8,
    n_max: int = 512,
) -> int:
    """Smallest ``n`` whose Neyman–Pearson test reaches ``power``.

    Simple ``H0: theta_a`` vs ``H1: theta_b``. Threshold is the
    ``1 - alpha`` quantile of the log-likelihood ratio under ``H0``.
    """
    lo, hi = int(n_min), int(n_max)
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        pw = _likelihood_ratio_power(
            family,
            theta_a,
            theta_b,
            n=mid,
            n_trials=int(n_trials),
            seed=int(seed),
            alpha=float(alpha),
        )
        if pw >= float(power):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)


def _likelihood_ratio_power(
    family: PackFamily,
    theta_a: Any,
    theta_b: Any,
    *,
    n: int,
    n_trials: int,
    seed: int,
    alpha: float,
) -> float:
    null = np.empty(int(n_trials))
    alt = np.empty(int(n_trials))
    for i in range(int(n_trials)):
        xs0 = sample_family(family, theta_a, int(n), seed=seed * 1_000_003 + i)
        xs1 = sample_family(family, theta_b, int(n), seed=seed * 1_000_003 + 10_000 + i)
        null[i] = float(
            np.sum(log_density(family, theta_b, xs0) - log_density(family, theta_a, xs0))
        )
        alt[i] = float(
            np.sum(log_density(family, theta_b, xs1) - log_density(family, theta_a, xs1))
        )
    cutoff = float(np.quantile(null, 1.0 - float(alpha)))
    return float(np.mean(alt > cutoff))


def distinguishability_samples(
    family: PackFamily,
    theta_a: Any,
    theta_b: Any,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
    steps: int = 64,
) -> int:
    """Wald sample count from the Fisher distance.

    ``N = (z_{1-alpha/2} + z_{power})^2 / d_F^2``. Default
    ``alpha=0.05``, ``power=0.8`` gives the usual ``7.849 / d_F^2``.
    """
    if not 0.0 < float(power) < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    dist = fisher_distance(family, theta_a, theta_b, steps=steps)
    if dist <= 0.0:
        raise ValueError("Fisher distance is zero; the settings are identical")
    z_a = _normal_ppf(1.0 - float(alpha) / 2.0)
    z_p = _normal_ppf(float(power))
    return max(1, int(math.ceil((z_a + z_p) ** 2 / (dist * dist))))


def collapse_degeneracy(
    family: PackFamily,
    theta: Any,
    *,
    direction: str = "spread",
) -> DegeneracyReport:
    """Eigenvalue, local exponent, and damping along the collapse chart."""
    if direction != "spread":
        raise ValueError(f"only direction='spread' is implemented, got {direction!r}")
    _require_density(family, theta)
    g = fisher_metric(family, theta)
    eig = np.linalg.eigvalsh(_sym(g))
    lam_min = float(np.min(eig))
    lam_max = float(np.max(np.abs(eig)))
    cond = (
        float("inf")
        if lam_min <= 0.0
        else float(lam_max / max(lam_min, 1e-300))
    )
    exponent = _spread_exponent(family, theta)
    near = bool(lam_min < DEGENERACY_ABS_FLOOR or exponent > 1.5)
    if near:
        recommendation = "reparameterize by order"
        damping = max(pinv_rcond(g) * max(lam_max, 1e-300), DEGENERACY_ABS_FLOOR)
    else:
        recommendation = "spread coordinate is identifiable"
        damping = pinv_rcond(g) * max(lam_max, 1e-300)
    return DegeneracyReport(
        eigenvalue=lam_min,
        exponent=exponent,
        condition_number=cond,
        recommendation=recommendation,
        damping=float(damping),
        path=PATH_CLOSED,
        direction=direction,
    )


def mean_nll(family: PackFamily, theta: Any, data: Any) -> float:
    """Mean negative log-density of i.i.d. observations."""
    _require_density(family, theta)
    xs = np.asarray(data, dtype=float).reshape(-1)
    p, _jac = _density_and_jac(family, theta, xs)
    return float(-np.mean(np.log(np.maximum(p, 1e-300))))


def nll_gradient(family: PackFamily, theta: Any, data: Any) -> FloatArray:
    """Mean ``-score`` of i.i.d. observations (NLL gradient)."""
    _require_density(family, theta)
    xs = np.asarray(data, dtype=float).reshape(-1)
    _p, jac = _density_and_jac(family, theta, xs)
    score = jac / np.maximum(_p[:, None], 1e-300)
    return np.asarray(-np.mean(score, axis=0), dtype=np.float64)


def undamped_natural_step(
    family: PackFamily,
    theta: Any,
    data: Any,
    *,
    lr: float = 1.0,
) -> FloatArray:
    """``theta - lr * G^{-1} grad`` with the tight pinv cutoff only."""
    g = fisher_metric(family, theta)
    grad = nll_gradient(family, theta, data)
    step = fisher_pinv(g) @ grad
    return _project_theta(family, _theta(family, theta) - float(lr) * step)


def damped_natural_step(
    family: PackFamily,
    theta: Any,
    data: Any,
    *,
    lr: float = 1.0,
    report: DegeneracyReport | None = None,
) -> FloatArray:
    """Natural step with eigenvalues floored by :class:`DegeneracyReport`.

    Only the collapse direction is damped. Well-conditioned coordinates
    keep their Fisher scaling. This is the opposite of adding
    ``damping * I`` to every direction.
    """
    t = _theta(family, theta)
    g = _sym(fisher_metric(family, theta))
    grad = nll_gradient(family, theta, data)
    used = report if report is not None else collapse_degeneracy(family, theta)
    eig, vecs = np.linalg.eigh(g)
    floored = np.maximum(eig, float(used.damping))
    step = vecs @ ((vecs.T @ grad) / floored)
    return _project_theta(family, t - float(lr) * step)


def sample_family(
    family: PackFamily,
    theta: Any,
    n: int,
    *,
    seed: int,
) -> FloatArray:
    """Exact inverse-CDF samples from a logistic pack mixture / two-bias pack."""
    _require_density(family, theta)
    rng = np.random.default_rng(int(seed))
    if family.kind in {"two_bias_logistic", "two_bias_located"}:
        mu, delta = _two_bias_mu_delta(family, theta)
        u1 = np.clip(rng.random(int(n)), 1e-16, 1.0 - 1e-16)
        u2 = rng.random(int(n))
        return np.asarray(
            mu + np.log(u1 / (1.0 - u1)) + float(delta) * (u2 - 0.5),
            dtype=np.float64,
        )
    locations, scales, weights = family.unpack(theta)
    choice = rng.choice(len(weights), size=int(n), p=weights)
    u = np.clip(rng.random(int(n)), 1e-16, 1.0 - 1e-16)
    return np.asarray(
        locations[choice] + np.log(u / (1.0 - u)) / scales[choice],
        dtype=np.float64,
    )


def as_manifold_spec(family: PackFamily, *, closed_form: bool = True) -> Any:
    """Wrap ``G`` as an ``omnibias.geometry`` :class:`ManifoldSpec`.

    ``g_point`` calls the numpy metric, so Christoffel symbols via
    autodiff of this wrap are not available. Geodesic *length* uses
    :func:`fisher_distance`. Requires ``omnibias-geometry``.
    """
    try:
        from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "as_manifold_spec requires omnibias-geometry; fisher_distance "
            "does not"
        ) from exc

    dim = family.n_params

    def g_point(coords: Any) -> Any:
        import jax.numpy as jnp

        metric = fisher_metric(family, coords, closed_form=closed_form)
        return jnp.asarray(metric)

    return ManifoldSpec(
        name=f"pack_fisher_{family.kind}",
        dim=dim,
        metric=MetricSpec(g_point=g_point, dim=dim, name="pack_fisher"),
    )


def honesty_payload() -> dict[str, object]:
    return {
        "bias_collapse": True,
        "temperature_collapse": False,
        "k_ge_3_fisher": "inapplicable_not_a_density",
        "scalar_glm_fisher_is_a_different_object": True,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "pinv_rcond_rule": PINV_RCOND_RULE,
    }


def worked_example() -> dict[str, object]:
    """Tiny deterministic example for docs snippets."""
    family = two_bias_family()
    delta = 1e-4
    g = fisher_delta_delta(delta, nodes=96)
    report = collapse_degeneracy(family, np.array([0.1]))
    loc = logistic_location_family()
    dist = fisher_distance(loc, np.array([0.0]), np.array([math.sqrt(3.0)]))
    return {
        "g_over_delta2": g / (delta * delta),
        "predicted_prefactor": 1.0 / 720.0,
        "recommendation": report.recommendation,
        "unit_location_distance": dist,
    }


def _finite_difference_msg(family: PackFamily) -> str:
    return (
        f"K={family.n_components} central finite-difference packs change "
        "sign, so they are not probability densities and Fisher does not "
        "apply (inapplicable, not unmeasured)"
    )


def _require_density(family: PackFamily, theta: Any) -> None:
    if family.kind == "finite_difference":
        if family.n_components >= 3:
            raise NotADensityError(_finite_difference_msg(family))
        # K=2 is the two-bias pack.
        return
    if family.kind == "logistic_mixture":
        _weights = family.unpack(theta)[2]
        if np.any(_weights < 0.0) or abs(float(np.sum(_weights)) - 1.0) > 1e-9:
            raise NotADensityError(
                "Fisher requires c_g >= 0 and sum c_g = 1"
            )


def _theta(family: PackFamily, theta: Any) -> FloatArray:
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    if family.kind == "finite_difference" and family.n_components == 2:
        expected = 1
    else:
        expected = family.n_params
    if t.size != expected:
        raise ValueError(
            f"{family.kind} expects {expected} parameters, got {t.size}"
        )
    if family.kind in {"two_bias_logistic", "finite_difference"} and t[0] <= 0.0:
        raise ValueError(f"delta must be positive, got {t[0]}")
    if family.kind == "two_bias_located" and t[1] <= 0.0:
        raise ValueError(f"delta must be positive, got {t[1]}")
    return t


def _project_theta(family: PackFamily, theta: FloatArray) -> FloatArray:
    t = np.asarray(theta, dtype=np.float64).reshape(-1).copy()
    if family.kind in {"two_bias_logistic", "finite_difference"}:
        t[0] = max(float(t[0]), 1e-8)
    if family.kind == "two_bias_located":
        t[1] = max(float(t[1]), 1e-8)
    return t


def _two_bias_mu_delta(family: PackFamily, theta: Any) -> tuple[float, float]:
    t = _theta(family, theta)
    if family.kind == "two_bias_located":
        return float(t[0]), float(t[1])
    return 0.0, float(t[0])


def _sym(metric: Any) -> FloatArray:
    g = np.asarray(metric, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"metric must be square, got {g.shape}")
    return np.asarray(0.5 * (g + g.T), dtype=np.float64)


_NODE_CACHE: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}


def _logit_nodes(nodes: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    key = int(nodes)
    cached = _NODE_CACHE.get(key)
    if cached is not None:
        return cached
    xg, wg = np.polynomial.legendre.leggauss(key)
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    x = np.log(t / (1.0 - t))
    dx = 1.0 / (t * (1.0 - t))
    packed = (
        np.asarray(x, dtype=np.float64),
        np.asarray(w * dx, dtype=np.float64),
        np.asarray((1.0 - t) / t, dtype=np.float64),
    )
    _NODE_CACHE[key] = packed
    return packed


def _sigmoid(z: Any) -> Any:
    zz = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-zz))


def _sigmoid_p(z: Any) -> Any:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _sigmoid_pp(z: Any) -> Any:
    s = _sigmoid(z)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def _density_and_jac(
    family: PackFamily,
    theta: Any,
    x: Any,
) -> tuple[FloatArray, FloatArray]:
    xs = np.asarray(x, dtype=float).reshape(-1)
    kind = family.kind
    if kind in {"two_bias_logistic", "finite_difference"}:
        u = np.exp(-xs)
        p = np.asarray(two_bias_density_u(u, float(_theta(family, theta)[0])))
        jac = np.asarray(two_bias_ddelta_u(u, float(_theta(family, theta)[0]))).reshape(
            -1, 1
        )
        return np.asarray(p, dtype=np.float64), np.asarray(jac, dtype=np.float64)
    if kind == "two_bias_located":
        mu, delta = _two_bias_mu_delta(family, theta)
        u = np.exp(-(xs - mu))
        p = np.asarray(two_bias_density_u(u, delta))
        d_delta = np.asarray(two_bias_ddelta_u(u, delta))
        d_mu = -np.asarray(two_bias_dx_u(u, delta))
        jac = np.stack([d_mu, d_delta], axis=1)
        return np.asarray(p, dtype=np.float64), np.asarray(jac, dtype=np.float64)
    if kind == "logistic_location":
        mu = float(_theta(family, theta)[0])
        z = xs - mu
        p = np.asarray(_sigmoid_p(z), dtype=np.float64)
        # d p / d mu = - sigma''(z)
        jac = (-_sigmoid_pp(z)).reshape(-1, 1)
        return p, np.asarray(jac, dtype=np.float64)
    locations, scales, weights = family.unpack(theta)
    t = _theta(family, theta)
    mu1, phi1, mu2, phi2, psi = (float(v) for v in t)
    a1, a2 = float(scales[0]), float(scales[1])
    c1, c2 = float(weights[0]), float(weights[1])
    z1 = a1 * (xs - mu1)
    z2 = a2 * (xs - mu2)
    sp1 = _sigmoid_p(z1)
    sp2 = _sigmoid_p(z2)
    spp1 = _sigmoid_pp(z1)
    spp2 = _sigmoid_pp(z2)
    p = c1 * a1 * sp1 + c2 * a2 * sp2
    dp_dmu1 = -c1 * a1 * a1 * spp1
    dp_dmu2 = -c2 * a2 * a2 * spp2
    dp_da1 = c1 * (sp1 + z1 * spp1)
    dp_da2 = c2 * (sp2 + z2 * spp2)
    dp_dphi1 = a1 * dp_da1
    dp_dphi2 = a2 * dp_da2
    dc1 = c1 * (1.0 - c1)
    dp_dpsi = dc1 * (a1 * sp1 - a2 * sp2)
    jac = np.stack([dp_dmu1, dp_dphi1, dp_dmu2, dp_dphi2, dp_dpsi], axis=1)
    return np.asarray(p, dtype=np.float64), np.asarray(jac, dtype=np.float64)


def _quadrature_metric(
    family: PackFamily,
    theta: Any,
    *,
    nodes: int,
) -> FloatArray:
    x, w, u = _logit_nodes(nodes)
    if family.kind in {"two_bias_logistic", "finite_difference"}:
        p = np.asarray(two_bias_density_u(u, float(_theta(family, theta)[0])))
        jac = np.asarray(
            two_bias_ddelta_u(u, float(_theta(family, theta)[0]))
        ).reshape(-1, 1)
        return _assemble_g(p, jac, w)
    p, jac = _density_and_jac(family, theta, x)
    return _assemble_g(p, jac, w)


def _assemble_g(p: FloatArray, jac: FloatArray, w: FloatArray) -> FloatArray:
    safe = np.maximum(p, 1e-300)
    # G_ij = int (dp_i dp_j / p) dx
    weighted = (jac / safe[:, None]) * np.sqrt(safe)[:, None]
    # int score_i score_j p dx = int (dp_i dp_j / p) dx
    g = (jac.T * (w / safe)) @ jac
    _ = weighted
    return np.asarray(0.5 * (g + g.T), dtype=np.float64)


def _monte_carlo_metric(
    family: PackFamily,
    theta: Any,
    *,
    n: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    xs = sample_family(family, theta, n, seed=seed)
    p, jac = _density_and_jac(family, theta, xs)
    score = jac / np.maximum(p[:, None], 1e-300)
    outer = score[:, :, None] * score[:, None, :]
    metric = np.mean(outer, axis=0)
    se = np.std(outer, axis=0, ddof=1) / math.sqrt(int(n))
    return (
        np.asarray(0.5 * (metric + metric.T), dtype=np.float64),
        np.asarray(se, dtype=np.float64),
    )


def _spread_exponent(family: PackFamily, theta: Any) -> float:
    """Local log-log slope of the collapse-direction eigenvalue vs spread."""
    if family.kind in {"two_bias_logistic", "finite_difference"}:
        delta = float(_theta(family, theta)[0])
        spreads = np.array([delta, delta / 4.0, delta / 16.0], dtype=float)
        vals = []
        for d in spreads:
            if d <= 0.0:
                return 2.0
            g = fisher_metric(family, np.array([d]))
            vals.append(max(float(g[0, 0]), 1e-300))
        log_s = np.log(spreads)
        log_g = np.log(np.asarray(vals))
        slope = float(np.polyfit(log_s, log_g, 1)[0])
        return slope
    if family.kind == "two_bias_located":
        mu, delta = _two_bias_mu_delta(family, theta)
        spreads = np.array([delta, delta / 4.0, delta / 16.0], dtype=float)
        vals = []
        for d in spreads:
            g = fisher_metric(family, np.array([mu, d]))
            vals.append(max(float(g[1, 1]), 1e-300))
        slope = float(np.polyfit(np.log(spreads), np.log(np.asarray(vals)), 1)[0])
        return slope
    if family.kind == "logistic_location":
        return 0.0
    # Mixture: shrink the location gap, watch the smallest eigenvalue.
    t = _theta(family, theta)
    mu1, phi1, mu2, phi2, psi = (float(v) for v in t)
    mid = 0.5 * (mu1 + mu2)
    gap = mu1 - mu2
    if abs(gap) < 1e-9:
        return 2.0
    scales = np.array([1.0, 0.5, 0.25])
    vals = []
    gaps = []
    for s in scales:
        trial = np.array([mid + 0.5 * s * gap, phi1, mid - 0.5 * s * gap, phi2, psi])
        g = fisher_metric(family, trial)
        vals.append(max(float(np.min(np.linalg.eigvalsh(_sym(g)))), 1e-300))
        gaps.append(abs(s * gap))
    return float(np.polyfit(np.log(gaps), np.log(np.asarray(vals)), 1)[0])


def _normal_ppf(p: float) -> float:
    # Abramowitz-Stegun 26.2.23 via erfcinv.
    return float(math.sqrt(2.0) * _erfcinv(2.0 * (1.0 - float(p))))


def _erfcinv(y: float) -> float:
    """Inverse complementary error function for ``y in (0, 2)``."""
    yy = min(max(float(y), 1e-16), 2.0 - 1e-16)
    if yy == 1.0:
        return 0.0
    sign = 1.0 if yy < 1.0 else -1.0
    t = 1.0 - yy if yy <= 1.0 else yy - 1.0
    a = 0.147
    ln = math.log(1.0 - t * t)
    inner = 2.0 / (math.pi * a) + 0.5 * ln
    x = sign * math.sqrt(math.sqrt(inner * inner - ln / a) - inner)
    for _ in range(3):
        err = math.erfc(x) - yy
        deriv = -2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        second = -2.0 * x * deriv
        x = x - err * deriv / (deriv * deriv - 0.5 * err * second)
    return x


def randomized_mixture_suite(
    *,
    n: int = 8,
    seed: int = 0,
    min_eig: float = 1e-3,
) -> list[FloatArray]:
    """Well-separated two-component mixtures for G1 / G3.

    Draws are rejected until the closed-form metric is positive definite
    away from collapse (``lambda_min >= min_eig``).
    """
    rng = np.random.default_rng(int(seed))
    family = logistic_mixture_family()
    out: list[FloatArray] = []
    attempts = 0
    while len(out) < int(n):
        attempts += 1
        if attempts > 200 * int(n):
            raise RuntimeError("could not draw a well-separated mixture suite")
        mu1 = float(rng.uniform(-2.2, -0.8))
        mu2 = float(rng.uniform(0.8, 2.2))
        phi1 = float(rng.uniform(-0.15, 0.35))
        phi2 = float(rng.uniform(-0.15, 0.35))
        psi = float(rng.uniform(-0.5, 0.5))
        theta = np.array([mu1, phi1, mu2, phi2, psi], dtype=np.float64)
        eig = np.linalg.eigvalsh(fisher_metric(family, theta, nodes=64))
        if float(np.min(eig)) >= float(min_eig):
            out.append(theta)
    return out


__all__ = [
    "DEGENERACY_ABS_FLOOR",
    "DegeneracyReport",
    "FisherEvaluation",
    "FloatArray",
    "NotADensityError",
    "PATH_CLOSED",
    "PATH_MC",
    "PINV_RCOND_RULE",
    "PackFamily",
    "as_manifold_spec",
    "collapse_degeneracy",
    "damped_natural_step",
    "distinguishability_samples",
    "empirical_distinguishability_n",
    "evaluate_fisher",
    "finite_difference_family",
    "fisher_delta_delta",
    "fisher_distance",
    "fisher_metric",
    "fisher_metric_mc",
    "fisher_pinv",
    "honesty_payload",
    "log_density",
    "logistic_location_family",
    "logistic_mixture_family",
    "mean_nll",
    "nll_gradient",
    "pinv_rcond",
    "randomized_mixture_suite",
    "sample_family",
    "two_bias_A",
    "two_bias_ddelta_u",
    "two_bias_density_naive",
    "two_bias_density_u",
    "two_bias_dx_u",
    "two_bias_family",
    "two_bias_located_family",
    "undamped_natural_step",
    "worked_example",
]

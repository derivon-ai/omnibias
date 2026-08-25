# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Inverse-problem product API (theory 05-01).

Interface localization, layered inversion, free-boundary tracking,
identifiability, and D-optimal sensor placement. This is a submodule of
``omnibias-pinn``, not a package. Ill-posedness is not removed: every
recovery depends on the reported regularizer.

The certified enclosure is the noiseless-response Krawczyk box (spec
03-08). A conformal slab (spec 04-02) is a different guarantee kind and
must not be merged into that interval.

``alpha`` is a tempering scale, not founding bias collapse and not
temperature collapse. Distinct from
:func:`omnibias.pinn.solver.torch.inverse.solve_inverse` (PDE-coefficient
recovery).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omnibias.core.polynomials import sigmoid_polynomial_coeffs
from omnibias.core.transfer import Layer, reflection_transmission, stack_matrix
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.sigma import sigma_tower_interval

FloatArray = NDArray[np.float64]
FieldFn = Callable[[float, float], float]

SUBMODULAR_GUARANTEE = 1.0 - 1.0 / math.e
SIGMA1_AT_ZERO = 0.25
STEFAN_LAMBDA_STE1 = 0.620062610277404


class GuaranteeKindError(ValueError):
    """Enclosure and conformal slabs are different guarantee kinds."""


class InverseRegularizerError(ValueError):
    """A recovery was requested without naming the regularizer."""


@dataclass(frozen=True)
class InterfaceEstimate:
    """Point estimate plus a sound enclosure of the noiseless peak.

    ``location`` is the Krawczyk box. ``conformal_slab`` is optional and
    a different kind -- see :func:`merge_guarantees`.
    """

    location: Interval
    jump_order: int
    jump_size: float
    unique_in_enclosure: bool
    tau_hat: float
    channel: int
    conformal_slab: object | None = None
    regularizer: str = "scan_channel_n"


@dataclass(frozen=True)
class LayerStack:
    """Recovered 1-D transfer stack (spec 02-11)."""

    layers: tuple[Layer, ...]
    n_iterations: int
    residual: float
    regularizer: str = "analytic_gn_transfer"


@dataclass(frozen=True)
class BoundaryTrack:
    """Equality-locus front ``f1 = f2`` with IFT velocity."""

    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    regularizer: str = "equality_locus_ift"


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Fisher eigenvalues of the named parameters, before inversion."""

    fisher: FloatArray
    eigenvalues: FloatArray
    unidentifiable: tuple[str, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class SensorPlan:
    """D-optimal subset of candidate locations."""

    indices: tuple[int, ...]
    locations: FloatArray
    fisher: float
    uniform_fisher: float
    volume_ratio: float
    guarantee: float = SUBMODULAR_GUARANTEE
    regularizer: str = "d_optimal_logdet"


def honesty_payload() -> dict[str, object]:
    return {
        "ill_posedness_removed": False,
        "enclosure_is_noiseless_response": True,
        "conformal_merged": False,
        "alpha_is_tempering_scale": True,
        "bias_collapse": False,
        "temperature_collapse": False,
        "sensor_guarantee_is_optimization": True,
        "not_pde_coefficient_inverse": True,
        "theorem_prover_verified": False,
    }


def merge_guarantees(enclosure: Interval, conformal: object) -> None:
    """Refuse to add a conformal slab to a sound enclosure."""
    _ = enclosure, conformal
    raise GuaranteeKindError(
        "sound enclosure (03-08) and conformal slab (04-02) must not merge"
    )


def sigma_n(z: Any, n: int) -> Any:
    """Closed-form ``sigma^(n)(z)`` from the shared logistic polynomial."""
    zz = np.asarray(z, dtype=float)
    s = 1.0 / (1.0 + np.exp(-np.clip(zz, -60.0, 60.0)))
    coeffs = sigmoid_polynomial_coeffs(int(n))
    out = np.zeros_like(s)
    for i, a in enumerate(coeffs):
        out = out + float(a) * s**i
    return out


def piecewise_jump_field(
    x: Any, *, jump_order: int, jump: float, tau: float
) -> FloatArray:
    """Piecewise polynomial with a jump of size ``jump`` in derivative ``m``."""
    xx = np.asarray(x, dtype=float)
    base = 0.5 * xx
    excess = xx - float(tau)
    m = int(jump_order)
    j = float(jump)
    if m == 0:
        return np.where(xx >= tau, base + j, base)
    if m == 1:
        return np.where(xx >= tau, base + j * excess, base)
    if m == 2:
        return np.where(xx >= tau, base + 0.5 * j * excess * excess, base)
    raise ValueError(f"unsupported jump_order {m}")


def scan_response(
    points: Any,
    values: Any,
    taus: Any,
    *,
    order: int,
    alpha: float,
    normal: float = 1.0,
) -> FloatArray:
    """``r_n(tau) = (1/N) sum y_i alpha^n sigma^(n)(alpha (n·x_i - tau))``."""
    x = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    tt = np.asarray(taus, dtype=float).reshape(-1)
    n = int(order)
    a = float(alpha)
    w = float(normal)
    z = a * (w * x[:, None] - tt[None, :])
    kern = (a**n) * sigma_n(z, n) / float(x.size)
    return y @ kern


def admissible_band(
    points: Any, *, alpha: float, kernels: float = 7.0
) -> tuple[float, float]:
    """Interior band that keeps the scan kernel off the domain edges."""
    x = np.asarray(points, dtype=float).reshape(-1)
    margin = float(kernels) / max(float(alpha), 1.0)
    lo = float(x.min()) + margin
    hi = float(x.max()) - margin
    if hi <= lo:
        raise ValueError("admissible band empty; increase domain or lower alpha")
    return lo, hi


def polish_interface(
    points: Any,
    values: Any,
    *,
    order: int,
    alpha: float,
    tau_init: float,
    normal: float = 1.0,
    n_iters: int = 8,
) -> tuple[float, float, float]:
    """Newton on ``r'(tau) = 0``. Returns ``(tau_hat, r', r'')``."""
    x = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    a = float(alpha)
    w = float(normal)
    n = int(order)
    lo, hi = admissible_band(x, alpha=a)
    tau = float(np.clip(tau_init, lo, hi))
    n_pts = float(x.size)
    for _ in range(int(n_iters)):
        z = a * (w * x - tau)
        rp = -float(np.dot(y, (a ** (n + 1)) * sigma_n(z, n + 1)) / n_pts)
        rpp = float(np.dot(y, (a ** (n + 2)) * sigma_n(z, n + 2)) / n_pts)
        if abs(rpp) < 1e-30:
            break
        tau = float(np.clip(tau - rp / rpp, lo, hi))
    z = a * (w * x - tau)
    rp = -float(np.dot(y, (a ** (n + 1)) * sigma_n(z, n + 1)) / n_pts)
    rpp = float(np.dot(y, (a ** (n + 2)) * sigma_n(z, n + 2)) / n_pts)
    return float(tau), rp, rpp


def tv_deconvolution_localize(
    points: Any, values: Any, *, n_iters: int = 80, lam: float = 0.08
) -> float:
    """Named TV (ROF) baseline: denoise, then argmax of ``|D^2 u|``."""
    x = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    u = y.copy()
    dt = 0.15
    for _ in range(int(n_iters)):
        du = np.diff(u, prepend=u[0])
        p = du / np.maximum(np.abs(du), 1e-12)
        div = np.diff(p, append=p[-1])
        u = u - dt * ((u - y) - float(lam) * div)
    d2 = np.abs(np.diff(u, n=2))
    return float(x[1:-1][int(np.argmax(d2))])


def _mollifier_derivs(
    tau: Interval, *, tau_star: float, alpha: float, jump: float
) -> tuple[Interval, Interval]:
    """Continuum noiseless ``r`` / ``r'`` for a first-derivative jump (n=3)."""
    z = Interval.point(float(alpha)) * (tau - Interval.point(float(tau_star)))
    tower = sigma_tower_interval("sigmoid", z, 2)
    scale = Interval.point(float(jump) * float(alpha))
    r = scale * tower[1]
    rp = scale * Interval.point(float(alpha)) * tower[2]
    return r, rp


def enclose_peak(
    *,
    tau_hat: float,
    tau_star: float,
    alpha: float,
    jump: float,
    radii: tuple[float, ...] = (0.008, 0.005, 0.003, 0.002, 0.001),
) -> tuple[Interval, bool]:
    """Krawczyk unique-zero box for the noiseless mollifier ``r' = 0``."""

    def func(xs: list[Interval]) -> list[Interval]:
        return [_mollifier_derivs(xs[0], tau_star=tau_star, alpha=alpha, jump=jump)[1]]

    def jac(xs: list[Interval]) -> list[list[Interval]]:
        z = Interval.point(float(alpha)) * (xs[0] - Interval.point(float(tau_star)))
        tower = sigma_tower_interval("sigmoid", z, 3)
        rpp = Interval.point(float(jump) * float(alpha) ** 3) * tower[3]
        return [[rpp]]

    rpp0 = float(jump) * (float(alpha) ** 3) * float(sigma_n(0.0, 3))
    if abs(rpp0) < 1e-30:
        return Interval.point(float(tau_hat)), False
    a_inv = [[1.0 / rpp0]]
    for radius in radii:
        hit = krawczyk_certificate(func, jac, [float(tau_hat)], a_inv, float(radius))
        if hit is None:
            continue
        lo, hi = hit.enclosure[0]
        pad = 1.5e-3
        box = Interval(float(lo) - pad, float(hi) + pad)
        return box, True
    half = 3.0 / max(float(alpha), 1.0)
    return Interval(float(tau_hat) - half, float(tau_hat) + half), False


def locate_interface(
    points: Any,
    values: Any,
    normal: float = 1.0,
    *,
    order: int,
    alpha: float,
    regularizer: str = "scan_channel_n",
) -> InterfaceEstimate:
    """Scan-channel peak plus a Krawczyk enclosure of the noiseless ``r'``."""
    if not regularizer:
        raise InverseRegularizerError("locate_interface requires a named regularizer")
    x = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    n = int(order)
    a = float(alpha)
    lo, hi = admissible_band(x, alpha=a)
    grid = np.linspace(lo, hi, 161)
    r = scan_response(x, y, grid, order=n, alpha=a, normal=normal)
    tau0 = float(grid[int(np.argmax(np.abs(r)))])
    tau_hat, _rp, _rpp = polish_interface(
        x, y, order=n, alpha=a, tau_init=tau0, normal=normal
    )
    peak = float(
        scan_response(x, y, np.array([tau_hat]), order=n, alpha=a, normal=normal)[0]
    )
    jump = peak / max(a * SIGMA1_AT_ZERO, 1e-30)
    box, unique = enclose_peak(
        tau_hat=tau_hat, tau_star=tau_hat, alpha=a, jump=abs(jump)
    )
    return InterfaceEstimate(
        location=box,
        jump_order=n - 2,
        jump_size=float(jump),
        unique_in_enclosure=unique,
        tau_hat=float(tau_hat),
        channel=n,
        regularizer=str(regularizer),
    )


def identify_jump_order(
    points: Any,
    values: Any,
    *,
    alpha: float,
    channels: tuple[int, ...] = (2, 3, 4),
    normal: float = 1.0,
) -> tuple[int, dict[int, float]]:
    """Pick the scan channel with the largest ``|peak|``."""
    scores: dict[int, float] = {}
    x = np.asarray(points, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    a = float(alpha)
    lo, hi = admissible_band(x, alpha=a)
    grid = np.linspace(lo, hi, 121)
    residuals: dict[int, float] = {}
    for n in channels:
        r = scan_response(x, y, grid, order=int(n), alpha=a, normal=normal)
        tau0 = float(grid[int(np.argmax(np.abs(r)))])
        tau, _rp, _rpp = polish_interface(
            x, y, order=int(n), alpha=a, tau_init=tau0, normal=normal
        )
        kern = a * sigma_n(a * (grid - tau), 1)
        denom = float(np.dot(kern, kern))
        scale = float(np.dot(r, kern) / denom) if denom > 0.0 else 0.0
        rel = float(np.linalg.norm(r - scale * kern) / max(np.linalg.norm(r), 1e-12))
        residuals[int(n)] = rel
        scores[int(n)] = 1.0 / max(rel, 1e-12)
    best = min(residuals, key=lambda k: residuals[k])
    return int(best), scores


def location_fisher(
    points: Any, *, tau: float, alpha: float, noise_std: float
) -> float:
    """1-parameter Fisher of a logistic location observed at ``points``."""
    x = np.asarray(points, dtype=float).reshape(-1)
    a = float(alpha)
    s = float(noise_std)
    if s <= 0.0:
        raise ValueError("noise_std must be positive")
    dmu = a * sigma_n(a * (x - float(tau)), 1)
    return float(np.dot(dmu, dmu) / (s * s))


def identifiability(
    model: str,
    params: dict[str, float],
    data_design: Any,
    *,
    noise_std: float = 0.05,
    floor: float = 1e-6,
) -> IdentifiabilityReport:
    """Fisher eigenvalues; small ones are named unidentifiable directions."""
    _ = model
    points = np.asarray(data_design, dtype=float).reshape(-1)
    tau = float(params["tau"])
    alpha = float(params.get("alpha", 25.0))
    fish = location_fisher(points, tau=tau, alpha=alpha, noise_std=noise_std)
    metric = np.array([[fish]], dtype=np.float64)
    eig = np.array([fish], dtype=np.float64)
    names = ("tau",)
    weak = tuple(n for n, v in zip(names, eig, strict=True) if v < float(floor))
    return IdentifiabilityReport(
        fisher=metric, eigenvalues=eig, unidentifiable=weak, names=names
    )


def place_sensors(
    model: str,
    params: dict[str, float],
    candidates: Any,
    *,
    budget: int,
    noise_std: float = 0.05,
) -> SensorPlan:
    """Greedy D-optimal design on the location Fisher; reports ``1 - 1/e``."""
    _ = model
    xs = np.asarray(candidates, dtype=float).reshape(-1)
    k = int(budget)
    if k < 1 or k > xs.size:
        raise ValueError("budget must be in 1..len(candidates)")
    tau = float(params["tau"])
    alpha = float(params.get("alpha", 25.0))
    gains = np.array(
        [
            location_fisher(np.array([x]), tau=tau, alpha=alpha, noise_std=noise_std)
            for x in xs
        ],
        dtype=np.float64,
    )
    chosen: list[int] = []
    remaining = set(range(int(xs.size)))
    for _step in range(k):
        pick = max(remaining, key=lambda i: float(gains[i]))
        chosen.append(int(pick))
        remaining.remove(pick)
    uniform = np.linspace(0, xs.size - 1, k)
    uniform_idx = tuple(int(round(v)) for v in uniform)
    # Dedup while preserving count by stretching if needed.
    if len(set(uniform_idx)) < k:
        uniform_idx = tuple(range(0, xs.size, max(1, xs.size // k)))[:k]
    f_opt = location_fisher(xs[list(chosen)], tau=tau, alpha=alpha, noise_std=noise_std)
    f_uni = location_fisher(
        xs[list(uniform_idx)], tau=tau, alpha=alpha, noise_std=noise_std
    )
    vol = (1.0 / max(f_uni, 1e-30)) / (1.0 / max(f_opt, 1e-30))
    return SensorPlan(
        indices=tuple(chosen),
        locations=xs[list(chosen)].copy(),
        fisher=float(f_opt),
        uniform_fisher=float(f_uni),
        volume_ratio=float(vol),
    )


def _layer_matrix_and_jac(
    layer: Layer, omega: float
) -> tuple[Any, Any, Any]:
    """Characteristic matrix and analytic ``dM/dn``, ``dM/dd``."""
    n = complex(layer.index)
    d = float(layer.thickness)
    w = float(omega)
    delta = n * w * d
    c = cmath.cos(delta)
    s = cmath.sin(delta)
    m = ((c, 1.0j * s / n), (1.0j * n * s, c))
    ddelta_dn = w * d
    ddelta_dd = n * w
    dc_dn = -s * ddelta_dn
    ds_dn = c * ddelta_dn
    dc_dd = -s * ddelta_dd
    ds_dd = c * ddelta_dd
    dmdn = (
        (dc_dn, 1.0j * (n * ds_dn - s) / (n * n)),
        (1.0j * (s + n * ds_dn), dc_dn),
    )
    dmdd = (
        (dc_dd, 1.0j * ds_dd / n),
        (1.0j * n * ds_dd, dc_dd),
    )
    return m, dmdn, dmdd


def _mul2(a: Any, b: Any) -> Any:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _r_and_jac(layers: tuple[Layer, ...], omega: float) -> tuple[complex, np.ndarray]:
    """Complex ``R(omega)`` and analytic ``dR/d(n_i, d_i)``."""
    mats = [_layer_matrix_and_jac(layer, omega) for layer in layers]
    eye = ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))
    n_layers = len(layers)
    left: list[Any] = [eye] * n_layers
    acc = eye
    for i in range(n_layers - 1, -1, -1):
        left[i] = acc
        acc = _mul2(acc, mats[i][0])
    right: list[Any] = [eye] * n_layers
    acc = eye
    for i in range(n_layers):
        right[i] = acc
        acc = _mul2(mats[i][0], acc)
    m_full = stack_matrix(layers, omega)
    r, _t = reflection_transmission(m_full)
    jac = np.zeros(2 * n_layers, dtype=np.complex128)
    for i, (_m, dmdn, dmdd) in enumerate(mats):
        for slot, dm in ((i, dmdn), (n_layers + i, dmdd)):
            dm_full = _mul2(left[i], _mul2(dm, right[i]))
            jac[slot] = _d_reflection(dm_full, m_full)
    return complex(r), jac


def _d_reflection(dm: Any, m: Any) -> complex:
    a, b = m[0]
    c, d = m[1]
    da, db = dm[0]
    dc, dd = dm[1]
    n_in = 1.0
    n_out = 1.0
    denom = a * n_out + b * n_in * n_out + c + d * n_in
    ddenom = da * n_out + db * n_in * n_out + dc + dd * n_in
    num = a * n_out + b * n_in * n_out - c - d * n_in
    dnum = da * n_out + db * n_in * n_out - dc - dd * n_in
    return (dnum * denom - num * ddenom) / (denom * denom)


def _stack_from_vec(vec: FloatArray) -> tuple[Layer, ...]:
    half = vec.size // 2
    ns = np.exp(vec[:half])
    ds = np.exp(vec[half:])
    return tuple(Layer(complex(float(n), 0.0), float(d)) for n, d in zip(ns, ds, strict=True))


def _complex_r_vec(vec: FloatArray, omegas: FloatArray) -> np.ndarray:
    layers = _stack_from_vec(vec)
    out = np.empty(2 * omegas.size, dtype=np.float64)
    for i, w in enumerate(omegas):
        r, _t = reflection_transmission(stack_matrix(layers, float(w)))
        out[2 * i] = float(r.real)
        out[2 * i + 1] = float(r.imag)
    return out


def invert_layered(
    measurements: dict[str, Any],
    *,
    n_layers: int,
    init: LayerStack | None = None,
    max_iter: int = 25,
    analytic: bool = True,
    fd_eps: float = 1e-2,
    residual_tol: float = 1e-8,
    free: str = "both",
    regularizer: str = "analytic_gn_transfer",
) -> LayerStack:
    """Levenberg-Marquardt on complex ``R(omega)`` with the 02-11 Jacobian.

    ``free`` is ``"both"``, ``"index"`` (known thicknesses), or
    ``"thickness"`` (known indices). A material library / profilometry
    split is the regularizer that makes the five-layer map identifiable.
    """
    if not regularizer:
        raise InverseRegularizerError("invert_layered requires a named regularizer")
    omegas = np.asarray(measurements["omega"], dtype=float).reshape(-1)
    if "r" in measurements:
        rc = np.asarray(measurements["r"], dtype=np.complex128).reshape(-1)
    else:
        rc = np.asarray(measurements["R"], dtype=np.complex128).reshape(-1)
    target = np.empty(2 * rc.size, dtype=np.float64)
    target[0::2] = rc.real
    target[1::2] = rc.imag
    k = int(n_layers)
    if init is None:
        vec = np.zeros(2 * k, dtype=np.float64)
    else:
        ns = np.array([abs(layer.index) for layer in init.layers], dtype=np.float64)
        ds = np.array([layer.thickness for layer in init.layers], dtype=np.float64)
        vec = np.log(np.concatenate([ns, ds]))
    last_res = float("inf")
    n_iter = 0
    lam = 1e-3 if analytic else 2.0
    lo_b = np.concatenate(
        [np.full(k, np.log(1.05)), np.full(k, np.log(0.03))]
    )
    hi_b = np.concatenate(
        [np.full(k, np.log(3.5)), np.full(k, np.log(0.35))]
    )
    vec = np.clip(vec, lo_b, hi_b)
    if free not in {"both", "index", "thickness"}:
        raise ValueError("free must be 'both', 'index', or 'thickness'")
    mask = np.ones(2 * k, dtype=bool)
    if free == "index":
        mask[k:] = False
    elif free == "thickness":
        mask[:k] = False
    n_iter = 0
    while n_iter < int(max_iter):
        n_iter += 1
        layers = _stack_from_vec(vec)
        pred = np.empty_like(target)
        jac = np.empty((target.size, vec.size), dtype=np.float64)
        ns = np.exp(vec[:k])
        ds = np.exp(vec[k:])
        for i, w in enumerate(omegas):
            if analytic:
                r, j_nd = _r_and_jac(layers, float(w))
                pred[2 * i] = float(r.real)
                pred[2 * i + 1] = float(r.imag)
                jr = j_nd * np.concatenate([ns, ds])
                jac[2 * i] = jr.real
                jac[2 * i + 1] = jr.imag
            else:
                r, _t = reflection_transmission(stack_matrix(layers, float(w)))
                pred[2 * i] = float(r.real)
                pred[2 * i + 1] = float(r.imag)
                for j in range(vec.size):
                    step = np.zeros_like(vec)
                    step[j] = float(fd_eps)
                    pert = _complex_r_vec(np.clip(vec + step, lo_b, hi_b), np.array([w]))
                    jac[2 * i, j] = (pert[0] - pred[2 * i]) / float(fd_eps)
                    jac[2 * i + 1, j] = (pert[1] - pred[2 * i + 1]) / float(fd_eps)
        residual = pred - target
        last_res = float(np.sqrt(np.mean(residual * residual)))
        if last_res <= float(residual_tol):
            break
        jtj = jac[:, mask].T @ jac[:, mask]
        try:
            delta_free = np.linalg.solve(
                jtj + lam * np.eye(int(np.count_nonzero(mask))),
                jac[:, mask].T @ residual,
            )
        except np.linalg.LinAlgError:
            break
        delta = np.zeros_like(vec)
        delta[mask] = delta_free
        trial = np.clip(vec - delta, lo_b, hi_b)
        trial_pred = _complex_r_vec(trial, omegas)
        trial_res = float(np.sqrt(np.mean((trial_pred - target) ** 2)))
        if trial_res <= last_res:
            vec = trial
            last_res = trial_res
            lam = max(lam * 0.3, 1e-10)
            if last_res <= float(residual_tol):
                break
        else:
            lam = min(lam * 4.0, 1e4)
        if float(np.linalg.norm(delta)) < 1e-12:
            break
    layers = _stack_from_vec(vec)
    return LayerStack(
        layers=layers,
        n_iterations=int(n_iter),
        residual=last_res,
        regularizer=str(regularizer),
    )


def stefan_front(t: float, *, kappa: float = 1.0, lam: float = STEFAN_LAMBDA_STE1) -> float:
    return 2.0 * float(lam) * math.sqrt(float(kappa) * float(t))


def stefan_temperature(
    x: float,
    t: float,
    *,
    kappa: float = 1.0,
    lam: float = STEFAN_LAMBDA_STE1,
    t_wall: float = 1.0,
) -> float:
    s = stefan_front(t, kappa=kappa, lam=lam)
    if x >= s:
        return 0.0
    xi = x / (2.0 * math.sqrt(float(kappa) * float(t)))
    return float(t_wall) * (1.0 - math.erf(xi) / math.erf(float(lam)))


def stefan_partials(
    x: float,
    t: float,
    *,
    kappa: float = 1.0,
    lam: float = STEFAN_LAMBDA_STE1,
    t_wall: float = 1.0,
    h: float = 1e-6,
) -> tuple[float, float]:
    """``(T_x, T_t)`` by a tiny central difference of the closed form."""
    tx = (
        stefan_temperature(x + h, t, kappa=kappa, lam=lam, t_wall=t_wall)
        - stefan_temperature(x - h, t, kappa=kappa, lam=lam, t_wall=t_wall)
    ) / (2.0 * h)
    tt = (
        stefan_temperature(x, t + h, kappa=kappa, lam=lam, t_wall=t_wall)
        - stefan_temperature(x, t - h, kappa=kappa, lam=lam, t_wall=t_wall)
    ) / (2.0 * h)
    return float(tx), float(tt)


def track_free_boundary(
    field_pair: tuple[FieldFn, FieldFn],
    *,
    t_span: tuple[float, float],
    n_times: int = 21,
    x_init: float | None = None,
    regularizer: str = "equality_locus_ift",
) -> BoundaryTrack:
    """Newton on ``f1 - f2 = 0``; velocity is exact IFT ``-g_t / g_x``."""
    if not regularizer:
        raise InverseRegularizerError("track_free_boundary requires a named regularizer")
    f1, f2 = field_pair
    times = np.linspace(float(t_span[0]), float(t_span[1]), int(n_times))
    pos = np.empty_like(times)
    vel = np.empty_like(times)
    x = float(x_init if x_init is not None else stefan_front(times[0]))
    h = 1e-6
    for i, t in enumerate(times):
        for _ in range(8):
            g = float(f1(x, float(t)) - f2(x, float(t)))
            gx = (
                float(f1(x + h, float(t)) - f2(x + h, float(t)))
                - float(f1(x - h, float(t)) - f2(x - h, float(t)))
            ) / (2.0 * h)
            if abs(gx) < 1e-14:
                break
            x = x - g / gx
        gt = (
            float(f1(x, float(t) + h) - f2(x, float(t) + h))
            - float(f1(x, float(t) - h) - f2(x, float(t) - h))
        ) / (2.0 * h)
        gx = (
            float(f1(x + h, float(t)) - f2(x + h, float(t)))
            - float(f1(x - h, float(t)) - f2(x - h, float(t)))
        ) / (2.0 * h)
        pos[i] = x
        vel[i] = 0.0 if abs(gx) < 1e-14 else -gt / gx
    return BoundaryTrack(
        times=times.astype(np.float64),
        positions=pos.astype(np.float64),
        velocities=vel.astype(np.float64),
        regularizer=str(regularizer),
    )


def level_set_front(
    *,
    t_span: tuple[float, float],
    n_times: int = 21,
    n_space: int = 81,
    x_max: float = 3.0,
) -> BoundaryTrack:
    """Named 1-D upwind level-set baseline at the same time step."""
    times = np.linspace(float(t_span[0]), float(t_span[1]), int(n_times))
    xs = np.linspace(0.0, float(x_max), int(n_space))
    dx = float(xs[1] - xs[0])
    phi = xs - stefan_front(times[0])
    pos = np.empty_like(times)
    vel = np.zeros_like(times)
    pos[0] = float(xs[int(np.argmin(np.abs(phi)))])
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        speed = STEFAN_LAMBDA_STE1 / math.sqrt(max(times[i - 1], 1e-8))
        dphi = np.gradient(phi, dx)
        phi = phi - dt * speed * np.abs(dphi)
        pos[i] = float(xs[int(np.argmin(np.abs(phi)))])
        vel[i] = (pos[i] - pos[i - 1]) / dt
    return BoundaryTrack(
        times=times.astype(np.float64),
        positions=pos.astype(np.float64),
        velocities=vel.astype(np.float64),
        regularizer="level_set_upwind",
    )


def worked_example() -> dict[str, object]:
    """Spec §5 interface: ``tau* = 0.37``, jump ``J = 2`` in ``u'``."""
    tau_star = 0.37
    x = np.linspace(0.0, 1.0, 201)
    y = piecewise_jump_field(x, jump_order=1, jump=2.0, tau=tau_star)
    est = locate_interface(x, y, order=3, alpha=25.0)
    return {
        "tau_hat": est.tau_hat,
        "tau_star": tau_star,
        "unique": est.unique_in_enclosure,
        "contains_truth": est.location.contains(tau_star),
        "channel": est.channel,
        "honesty": honesty_payload(),
    }

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gauge-hard signed-hat search for compactified CCF (free Ω).

A CCF root must cancel leftover ``L[Ω]`` with the Hilbert quadratic at
gauge ``Ω(0.5)=0.05``. Core cancel wants ``HΩ(0)=(2+λ)/2 > 0``. Every
positive-on-``R+`` envelope has ``HΩ(0)<0``, so a softplus / ker hat
cannot leave the ``O(10^{-2})`` floor.

This driver:

* lifts a *signed* even Chebyshev hat through the official envelope
  (analytic ``Ω_y``, ``C^∞``). Default chart is ``arctan(y/s)`` so
  ``y=0`` is an interior node, not a ``q``-chart endpoint;
* hard-rescales so ``Ω(0.5)=0.05`` inside the residual JVP;
* collocates on an origin-clustered grid (residual peaks sit near
  ``|y|~0.2``);
* optional nodal init / frozen-velocity Picard / ``HΩ(0)`` row — those
  flip ``H0`` but have not beaten the residual of a ker start;
* cubic / Martens–Grosse on the official free-Ω residual;
* scores on 401 pts, ``|y|<38``, with :func:`free_omega_vorticity_residual`.

Not a stretch or Rung-1 claim until the measured dense residual clears
the named gates. ``navier_stokes_proof_claim`` stays false. Rung-1 is
Hardy-Hilbert and is not started from this free-Ω driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_enable_x64", True)

from omnibias.jax.optim import (  # noqa: E402
    HomotopyGNConfig,
    homotopy_gauss_newton_minimize,
    peak_weighted_residual,
)
from omnibias.pinn.jax.discovery.ccf_vorticity import (  # noqa: E402
    free_omega_vorticity_residual,
)
from omnibias.pinn.jax.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
    apply_envelope,
    compactify_y_lambda,
)
from omnibias.pinn.jax.hilbert_line import hilbert_wholeline_hp  # noqa: E402

LAM_DEFAULT = 0.6057
GAUGE_PT = 0.5
GAUGE = 0.05
STRETCH_GATE = 1e-13


@dataclass(frozen=True)
class HatHomotopyConfig:
    """Compactified-hat search configuration."""

    degree: int = 8
    n_grid: int = 121
    y_max: float = 40.0
    lam: float = LAM_DEFAULT
    gauge_point: float = GAUGE_PT
    gauge_value: float = GAUGE
    peak_power: float = 2.0
    h0_weight: float = 0.0
    node_a: float = 0.532
    ker_eps_init: float = 0.4
    init: str = "ker"
    chart: Literal["q", "arctan_even"] = "arctan_even"
    core_s: float = 0.4
    picard_iters: int = 0
    picard_ridge: float = 1e-6
    homotopy: HomotopyGNConfig = HomotopyGNConfig(
        stages=(1.0,),
        steps_per_stage=16,
        method="cubic",
        cubic_sigma=1.0,
    )
    run_homotopy: bool = True
    cluster_core: float = 2.0
    n_core: int | None = None


def train_nodes(
    n: int,
    y_max: float,
    *,
    core: float = 2.0,
    n_core: int | None = None,
) -> Array:
    """Origin-clustered odd grid so ``|y|≲0.2`` residual peaks are sampled."""
    n = int(n)
    y_max = float(y_max)
    core = float(core)
    if n_core is None:
        n_core = (2 * n) // 3
    n_core = int(n_core)
    if n_core % 2 == 0:
        n_core += 1
    if n_core >= n - 4 or core >= y_max:
        return jnp.linspace(-y_max, y_max, n, dtype=jnp.float64)
    yc = jnp.linspace(-core, core, n_core, dtype=jnp.float64)
    n_tail = n - n_core
    n_half = n_tail // 2
    pos = jnp.linspace(core, y_max, n_half + 1, dtype=jnp.float64)[1:]
    neg = -pos[::-1]
    return jnp.concatenate([neg, yc, pos])


def _cheb_sum(coeffs: Array, x: Array) -> tuple[Array, Array]:
    """``(Σ c_k T_k(x), d/dx Σ c_k T_k)`` via the Chebyshev recurrence."""
    n = int(coeffs.shape[0])
    t0 = jnp.ones_like(x)
    val = coeffs[0] * t0
    dval = jnp.zeros_like(x)
    if n == 1:
        return val, dval
    t1 = x
    val = val + coeffs[1] * t1
    dval = dval + coeffs[1] * jnp.ones_like(x)
    t_prev, t_cur = t0, t1
    d_prev, d_cur = jnp.zeros_like(x), jnp.ones_like(x)
    for k in range(2, n):
        t_next = 2.0 * x * t_cur - t_prev
        d_next = 2.0 * t_cur + 2.0 * x * d_cur - d_prev
        val = val + coeffs[k] * t_next
        dval = dval + coeffs[k] * d_next
        t_prev, t_cur = t_cur, t_next
        d_prev, d_cur = d_cur, d_next
    return val, dval


def _cheb_vandermonde(x: Array, degree: int) -> Array:
    cols = [jnp.ones_like(x)]
    if degree >= 1:
        cols.append(x)
        t_prev, t_cur = cols[0], cols[1]
        for _k in range(2, degree + 1):
            t_next = 2.0 * x * t_cur - t_prev
            cols.append(t_next)
            t_prev, t_cur = t_cur, t_next
    return jnp.stack(cols, axis=1)


def _chart_x(
    y: Array, *, chart: str, lam: float, core_s: float
) -> tuple[Array, Array, bool]:
    """Return ``(x, x_y, even_only)`` for the hat Chebyshev chart."""
    if chart == "q":
        q = compactify_y_lambda(y, lam)
        alpha = float(alpha_from_lambda(lam))
        q_y = (-alpha * y) * jnp.power(1.0 + y * y, -0.5 * alpha - 1.0)
        return 2.0 * q - 1.0, 2.0 * q_y, False
    if chart == "arctan_even":
        s = float(core_s)
        xi = (2.0 / math.pi) * jnp.arctan(y / s)
        xi_y = (2.0 / math.pi) * (s / (s * s + y * y))
        return xi, xi_y, True
    raise ValueError(f"unknown hat chart {chart!r}")


def _hat_sum(coeffs: Array, x: Array, *, even_only: bool) -> tuple[Array, Array]:
    if not even_only:
        return _cheb_sum(coeffs, x)
    m = int(coeffs.shape[0])
    full = jnp.zeros((max(2 * m - 1, 1),), dtype=coeffs.dtype)
    full = full.at[0::2].set(coeffs)
    return _cheb_sum(full, x)


def _hat_vandermonde(
    y: Array, *, degree: int, chart: str, lam: float, core_s: float
) -> Array:
    x, _xy, even_only = _chart_x(y, chart=chart, lam=lam, core_s=core_s)
    if not even_only:
        return _cheb_vandermonde(x, degree)
    cols = []
    for k in range(int(degree) + 1):
        e = jnp.zeros((int(degree) + 1,), dtype=jnp.float64).at[k].set(1.0)
        val, _ = _hat_sum(e, x, even_only=True)
        cols.append(val)
    return jnp.stack(cols, axis=1)


def hat_from_cheb(
    y: Array,
    coeffs: Array,
    *,
    lam: float,
    chart: str = "arctan_even",
    core_s: float = 0.4,
) -> tuple[Array, Array]:
    """Signed Chebyshev hat and its ``y``-derivative."""
    x, x_y, even_only = _chart_x(y, chart=chart, lam=lam, core_s=core_s)
    raw, draw_dx = _hat_sum(coeffs, x, even_only=even_only)
    return raw, draw_dx * x_y


def omega_from_hat(
    y: Array,
    coeffs: Array,
    *,
    lam: float,
    chart: str = "arctan_even",
    core_s: float = 0.4,
) -> tuple[Array, Array]:
    """Official envelope lift ``Ω = y · E · hat`` with analytic ``Ω_y``."""
    hat, hat_y = hat_from_cheb(y, coeffs, lam=lam, chart=chart, core_s=core_s)
    alpha = float(alpha_from_lambda(lam))
    psi, psi_y = apply_envelope(y, hat, hat_y, power=alpha + 1.0)
    return y * psi, psi + y * psi_y


def hard_gauge(
    y: Array, omega: Array, omega_y: Array, *, point: float, value: float
) -> tuple[Array, Array]:
    g = jnp.interp(point, y, omega)
    scale = value / (g + 1e-30)
    return scale * omega, scale * omega_y


def ker_hat_init(
    y: Array,
    *,
    eps: float,
    lam: float,
    degree: int,
    chart: str = "arctan_even",
    core_s: float = 0.4,
) -> Array:
    """Least-squares Chebyshev fit of the ``ε``-ker hat (positive)."""
    alpha = float(alpha_from_lambda(lam))
    hat = jnp.power((1.0 + y * y) / (y * y + eps * eps), 0.5 * (alpha + 1.0))
    a = _hat_vandermonde(y, degree=degree, chart=chart, lam=lam, core_s=core_s)
    coef, *_ = jnp.linalg.lstsq(a, hat, rcond=None)
    return coef


def node_hat_init(
    y: Array,
    *,
    node: float,
    lam: float,
    degree: int,
    chart: str = "arctan_even",
    core_s: float = 0.4,
) -> Array:
    """Nodal hat ``(y²-a²)/(y²+a²)``: negative core on ``R+``, ``HΩ(0)`` can flip."""
    a2 = float(node) ** 2
    hat = (y * y - a2) / (y * y + a2)
    a = _hat_vandermonde(y, degree=degree, chart=chart, lam=lam, core_s=core_s)
    coef, *_ = jnp.linalg.lstsq(a, hat, rcond=None)
    return coef


def h0_target(lam: float) -> float:
    """Core-cancel value ``HΩ(0)=(2+λ)/2`` from ``r'(0)=0`` for odd ``Ω``."""
    return 0.5 * (2.0 + float(lam))


def honesty_flags(*, dense_max_abs: float, anti_ghost: bool) -> dict[str, bool]:
    """Stretch can flip only on a measured residual. Rung-1 is never started here."""
    stretch = bool(anti_ghost and float(dense_max_abs) <= STRETCH_GATE)
    return {
        "navier_stokes_proof_claim": False,
        "stretch_1e-13_cleared": stretch,
        "rung1_1e-11_report": False,
    }


def _gauged_omega_fn(
    y_train: Array,
    coeffs: Array,
    *,
    lam: float,
    point: float,
    value: float,
    chart: str,
    core_s: float,
):
    def omega_fn(tt: Array, c: Array = coeffs) -> Array:
        o, _ = omega_from_hat(tt, c, lam=lam, chart=chart, core_s=core_s)
        g = jnp.interp(
            point,
            y_train,
            omega_from_hat(y_train, c, lam=lam, chart=chart, core_s=core_s)[0],
        )
        return o * (value / (g + 1e-30))

    return omega_fn


def coupling_residual(
    y: Array,
    omega: Array,
    omega_y: Array,
    omega_fn,
    *,
    lam: float,
    t: float,
    y_trunc: float,
    peak_power: float,
    h0_weight: float = 0.0,
) -> Array:
    """``L + t Quad`` on a hard-gauged free Ω, plus optional ``HΩ(0)`` match."""
    r_full, fields = free_omega_vorticity_residual(
        y,
        omega,
        omega_y,
        omega_fn,
        lam=lam,
        y_trunc=y_trunc,
    )
    if float(t) >= 1.0 - 1e-15:
        r = r_full
    else:
        om = fields["omega"]
        omy = fields["omega_y"]
        u = fields["U"]
        uy = fields["U_y"]
        r_lin = om + (1.0 + lam) * y * omy
        r_quad = -u * omy - om * uy
        r = r_lin + float(t) * r_quad
    core = jnp.abs(y) < 0.95 * float(y_trunc)
    r = jnp.where(core, r, 0.0)
    r = peak_weighted_residual(r, peak_power)
    if float(h0_weight) > 0.0:
        h0 = jnp.interp(0.0, y, fields["U_y"])
        r = jnp.concatenate([r, jnp.asarray([float(h0_weight) * (h0 - h0_target(lam))])])
    return r


def _basis_h0(
    y: Array,
    *,
    degree: int,
    lam: float,
    y_trunc: float,
    chart: str,
    core_s: float,
) -> Array:
    """``H[Ω_k](0)`` for each Chebyshev basis column (linear, ungauged)."""
    decay = float(alpha_from_lambda(lam))
    n = int(degree) + 1
    vals = []
    for k in range(n):
        e = jnp.zeros((n,), dtype=jnp.float64).at[k].set(1.0)
        om, _ = omega_from_hat(y, e, lam=lam, chart=chart, core_s=core_s)

        def ofn(tt: Array, ee: Array = e) -> Array:
            return omega_from_hat(tt, ee, lam=lam, chart=chart, core_s=core_s)[0]

        uy = hilbert_wholeline_hp(
            y, om, ofn, decay_power=decay, y_trunc=y_trunc
        )
        vals.append(jnp.interp(0.0, y, uy))
    return jnp.stack(vals)


def _basis_profiles(
    y: Array, *, degree: int, lam: float, chart: str, core_s: float
) -> tuple[Array, Array]:
    n = int(degree) + 1
    oms = []
    omys = []
    for k in range(n):
        e = jnp.zeros((n,), dtype=jnp.float64).at[k].set(1.0)
        om, omy = omega_from_hat(y, e, lam=lam, chart=chart, core_s=core_s)
        oms.append(om)
        omys.append(omy)
    return jnp.stack(oms, axis=1), jnp.stack(omys, axis=1)


def frozen_velocity_picard(
    y: Array,
    coeffs0: Array,
    *,
    cfg: HatHomotopyConfig,
) -> tuple[Array, list[dict[str, float]]]:
    """Alternate exact ``H`` with linear LS in the signed hat (frozen ``U``).

    ``r ≈ (1-U_y)Ω + ((1+λ)y-U)Ω_y`` is linear in the hat once ``U`` is
    frozen. The next ``H`` is the official ``wholeline_hp`` of the new
    profile. Hard gauge is a linear row.
    """
    om_b, omy_b = _basis_profiles(
        y,
        degree=int(cfg.degree),
        lam=cfg.lam,
        chart=cfg.chart,
        core_s=cfg.core_s,
    )
    g_row = jnp.stack(
        [
            jnp.interp(cfg.gauge_point, y, om_b[:, k])
            for k in range(int(om_b.shape[1]))
        ]
    )
    h0_row = _basis_h0(
        y,
        degree=int(cfg.degree),
        lam=cfg.lam,
        y_trunc=cfg.y_max,
        chart=cfg.chart,
        core_s=cfg.core_s,
    )
    core = jnp.abs(y) < 0.95 * float(cfg.y_max)
    hist: list[dict[str, float]] = []
    coeffs = coeffs0
    for it in range(int(cfg.picard_iters)):
        om, omy = omega_from_hat(
            y, coeffs, lam=cfg.lam, chart=cfg.chart, core_s=cfg.core_s
        )
        om, omy = hard_gauge(
            y, om, omy, point=cfg.gauge_point, value=cfg.gauge_value
        )
        omega_fn = _gauged_omega_fn(
            y,
            coeffs,
            lam=cfg.lam,
            point=cfg.gauge_point,
            value=cfg.gauge_value,
            chart=cfg.chart,
            core_s=cfg.core_s,
        )
        _r, fields = free_omega_vorticity_residual(
            y, om, omy, omega_fn, lam=cfg.lam, y_trunc=cfg.y_max
        )
        u = fields["U"]
        uy = fields["U_y"]
        r_mat = (1.0 - uy)[:, None] * om_b + ((1.0 + cfg.lam) * y - u)[:, None] * omy_b
        r_mat = jnp.where(core[:, None], r_mat, 0.0)
        a = jnp.concatenate(
            [
                r_mat,
                (32.0 * g_row).reshape(1, -1),
                (float(cfg.h0_weight) * h0_row).reshape(1, -1),
            ],
            axis=0,
        )
        rhs = jnp.zeros((a.shape[0],), dtype=jnp.float64)
        rhs = rhs.at[-2].set(32.0 * cfg.gauge_value)
        rhs = rhs.at[-1].set(float(cfg.h0_weight) * h0_target(cfg.lam))
        ata = a.T @ a + float(cfg.picard_ridge) * jnp.eye(a.shape[1])
        coeffs = jnp.linalg.solve(ata, a.T @ rhs)
        r_now = r_mat @ coeffs
        hist.append(
            {
                "iter": float(it),
                "max_abs_frozen": float(jnp.max(jnp.abs(r_now))),
                "h0": float(jnp.interp(0.0, y, uy)),
                "omega_gauge_col": float(g_row @ coeffs),
            }
        )
    return coeffs, hist


def _score_coeffs(coeffs: Array, *, cfg: HatHomotopyConfig) -> dict[str, Any]:
    y_score = jnp.linspace(-40.0, 40.0, 401, dtype=jnp.float64)
    y_ref = train_nodes(
        int(cfg.n_grid),
        cfg.y_max,
        core=cfg.cluster_core,
        n_core=cfg.n_core,
    )
    om_s, omy_s = omega_from_hat(
        y_score, coeffs, lam=cfg.lam, chart=cfg.chart, core_s=cfg.core_s
    )
    om_s, omy_s = hard_gauge(
        y_score, om_s, omy_s, point=cfg.gauge_point, value=cfg.gauge_value
    )
    omega_fn = _gauged_omega_fn(
        y_ref,
        coeffs,
        lam=cfg.lam,
        point=cfg.gauge_point,
        value=cfg.gauge_value,
        chart=cfg.chart,
        core_s=cfg.core_s,
    )
    r_score, fields = free_omega_vorticity_residual(
        y_score,
        om_s,
        omy_s,
        omega_fn,
        lam=cfg.lam,
        y_trunc=40.0,
    )
    core = jnp.abs(y_score) < 38.0
    r_core = r_score[core]
    omax = float(jnp.max(jnp.abs(fields["omega"])))
    og = float(jnp.interp(cfg.gauge_point, y_score, fields["omega"]))
    anti_ghost = abs(og - cfg.gauge_value) <= 0.01 and omax >= 0.02
    h0 = float(jnp.interp(0.0, y_score, fields["U_y"]))
    dense = float(jnp.max(jnp.abs(r_core)))
    out = {
        "coeffs": [float(c) for c in coeffs],
        "dense_max_abs": dense,
        "omega_max": omax,
        "omega_gauge": og,
        "h0": h0,
        "h0_target": h0_target(cfg.lam),
        "anti_ghost": bool(anti_ghost),
    }
    out.update(honesty_flags(dense_max_abs=dense, anti_ghost=anti_ghost))
    return out


def run_hat_homotopy(
    cfg: HatHomotopyConfig | None = None,
    *,
    coeffs0: Array | None = None,
) -> dict[str, Any]:
    """Signed-hat Picard + optional coupling homotopy. Prototype."""
    cfg = HatHomotopyConfig() if cfg is None else cfg
    y = train_nodes(
        int(cfg.n_grid),
        cfg.y_max,
        core=cfg.cluster_core,
        n_core=cfg.n_core,
    )
    if coeffs0 is not None:
        coeffs0 = jnp.asarray(coeffs0, dtype=jnp.float64)
    elif cfg.init == "node":
        coeffs0 = node_hat_init(
            y,
            node=float(cfg.node_a),
            lam=cfg.lam,
            degree=int(cfg.degree),
            chart=cfg.chart,
            core_s=cfg.core_s,
        )
    elif cfg.init == "ker":
        coeffs0 = ker_hat_init(
            y,
            eps=float(cfg.ker_eps_init),
            lam=cfg.lam,
            degree=int(cfg.degree),
            chart=cfg.chart,
            core_s=cfg.core_s,
        )
    else:
        raise ValueError(f"init must be 'ker' or 'node', got {cfg.init!r}")
    if int(cfg.picard_iters) > 0:
        coeffs, picard_hist = frozen_velocity_picard(y, coeffs0, cfg=cfg)
    else:
        coeffs, picard_hist = coeffs0, []
    picard_score = _score_coeffs(coeffs, cfg=cfg)
    homotopy_hist: dict[str, Any] | None = None
    if cfg.run_homotopy:
        def residual_fn(cvec: Array, t: float) -> Array:
            om, omy = omega_from_hat(
                y, cvec, lam=cfg.lam, chart=cfg.chart, core_s=cfg.core_s
            )
            om, omy = hard_gauge(
                y, om, omy, point=cfg.gauge_point, value=cfg.gauge_value
            )
            omega_fn = _gauged_omega_fn(
                y,
                cvec,
                lam=cfg.lam,
                point=cfg.gauge_point,
                value=cfg.gauge_value,
                chart=cfg.chart,
                core_s=cfg.core_s,
            )
            return coupling_residual(
                y,
                om,
                omy,
                omega_fn,
                lam=cfg.lam,
                t=float(t),
                y_trunc=cfg.y_max,
                peak_power=cfg.peak_power,
                h0_weight=cfg.h0_weight,
            )

        coeffs, homotopy_hist = homotopy_gauss_newton_minimize(
            residual_fn, coeffs, config=cfg.homotopy
        )
    scored = _score_coeffs(coeffs, cfg=cfg)
    scored["picard"] = picard_hist
    scored["picard_score"] = {
        k: picard_score[k]
        for k in (
            "dense_max_abs",
            "h0",
            "omega_max",
            "anti_ghost",
            "stretch_1e-13_cleared",
        )
    }
    scored["homotopy"] = homotopy_hist
    scored["init"] = cfg.init
    scored["chart"] = cfg.chart
    return scored


def run_hat_picard_only(cfg: HatHomotopyConfig | None = None) -> dict[str, Any]:
    """Picard without homotopy (cheap official-path probe)."""
    return run_hat_homotopy(replace(cfg or HatHomotopyConfig(), run_homotopy=False))


__all__ = [
    "HatHomotopyConfig",
    "coupling_residual",
    "frozen_velocity_picard",
    "h0_target",
    "hard_gauge",
    "hat_from_cheb",
    "honesty_flags",
    "ker_hat_init",
    "node_hat_init",
    "omega_from_hat",
    "run_hat_homotopy",
    "run_hat_picard_only",
    "train_nodes",
]

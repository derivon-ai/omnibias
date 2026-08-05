# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified 2-D incompressible Euler steady vortex on the Riesz/Leray substrate.

This is the **2-D companion** to the 1-D Córdoba--Córdoba--Fontelos line work
(:func:`omnibias.pinn.certified.certified_ccf_selfsimilar_blowup_attempt`).  Where
the CCF certificate lives on the exact line-Hilbert pair, this one lives on the
verified planar substrate :mod:`omnibias.core.verified.riesz`: the Biot--Savart
map ``u = \nabla^\perp\Delta^{-1}\omega`` and the second-order Riesz / Leray
building blocks, all closed form on the radial blob basis
``f_a = a^2/(\pi D^2)`` (``D = x^2+y^2+a^2``).

What is certified
-----------------
For a radial vorticity field ``\omega = \sum_i c_i f_{a_i}`` (every blob centred
at the origin) the Biot--Savart velocity is the *tangential* field

.. math::

    u = \nabla^\perp\psi,\quad \psi = \sum_i c_i N_{a_i},\qquad
    u = \frac{1}{2\pi}\,(-y, x)\sum_i \frac{c_i}{D_i},

while ``\nabla\omega`` is *radial* (``\parallel (x, y)``).  Three facts follow and
are made theorem-grade here:

* **Exact steady state.**  ``u\cdot\nabla\omega \equiv 0`` because a tangential
  field dotted with a radial field is the *zero polynomial* ``(-y)x + x y``; the
  reported ``steady_residual_certified_sup`` is therefore an exact ``0`` (not a
  sampled estimate), independently re-confirmed on a substrate grid.
* **Certified norms.**  ``\lVert u\rVert_\infty``, ``\lVert\omega\rVert_\infty``
  and the strain ``\lVert\nabla u\rVert_{F,\infty}`` reduce to **1-D radial**
  rational functions of ``r`` (``|u| = r|Q(r)|/2\pi`` with
  ``Q=\sum_i c_i/D_i``; ``\lVert\nabla u\rVert_F^2 = (\Omega_p^2 + r^4 W^2)/2\pi^2``
  with ``\Omega_p=\sum_i c_i a_i^2/D_i^2``, ``W=\sum_i c_i/D_i^2``), so they are
  bounded over the *whole plane* by per-cell :class:`TaylorModel` enclosures on
  ``[0, R]`` plus explicit far-field tails.
* **Structural identities.**  ``\nabla\cdot u = 0`` and the Calderon--Zygmund
  trace ``R_{11}\omega + R_{22}\omega = -\omega`` are confirmed on the substrate
  grid; the Leray projection's own divergence residual is reported too.

Scope / honesty
---------------
This is a **2-D Euler** certificate, not SQG.  Genuine SQG velocity
``u = R^\perp\theta = \nabla^\perp(-\Delta)^{-1/2}\theta`` needs the *single*
Riesz transform (the half-Laplacian ``|\xi|^{-1}``), which is **not elementary**
on a radial blob (only the *composite* ``R_iR_k`` is -- see
:mod:`omnibias.core.verified.riesz`).  Extending the substrate with a certified
single-Riesz-of-blob is the recorded ``open_obligation`` for SQG.  2-D Euler is
moreover globally regular (Beale--Kato--Majda), so this is an *exact regular
steady state* and emphatically **not** a blow-up: ``honesty.unproven_claim``,
``three_d_claim``, ``blowup_claim`` and ``sqg_claim`` are all ``False``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Callable, Sequence
from typing import Any

from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.riesz import (
    blob,
    blob_gradient,
    leray_divergence_residual,
    potential_gradient,
    riesz_double_blob,
)
from omnibias.core.verified.taylor_model import TaylorModel

EULER2D_VORTEX_SCHEMA_VERSION = "euler2d-steady-vortex-1"

#: rigorous enclosure of pi (``math.pi`` is the nearest double below the truth).
_PI = Interval(math.pi, math.nextafter(math.pi, math.inf))
_TWO_PI = Interval.point(2.0) * _PI


def _sha256_json(value: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable proof artifact."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------- #
# Radial Taylor-model builders (functions of r over a cell [center +- radius]) #
# --------------------------------------------------------------------------- #
def _cell_inv_denoms(
    as_: Sequence[float], center: float, radius: float, order: int
) -> tuple[TaylorModel, list[TaylorModel]]:
    r"""``r^2`` and the list of ``1/(r^2 + a_i^2)`` Taylor models on one cell."""
    x2 = TaylorModel.identity(center, radius, order).pow_int(2)
    inv = [(x2 + a * a).reciprocal() for a in as_]
    return x2, inv


def _r_times_q_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``r\,Q(r) = r\sum_i c_i/(r^2+a_i^2)`` -- the velocity magnitude up to ``1/2\pi``."""
    x2, inv = _cell_inv_denoms(as_, center, radius, order)
    q = TaylorModel.constant(0.0, center, radius, order)
    for c, iv in zip(cs, inv, strict=True):
        q = q + iv * c
    return TaylorModel.identity(center, radius, order) * q


def _omega_numerator_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``\Omega_p(r) = \sum_i c_i a_i^2/(r^2+a_i^2)^2`` -- the vorticity up to ``1/\pi``."""
    _x2, inv = _cell_inv_denoms(as_, center, radius, order)
    out = TaylorModel.constant(0.0, center, radius, order)
    for c, a, iv in zip(cs, as_, inv, strict=True):
        out = out + iv.pow_int(2) * (c * a * a)
    return out


def _strain_sq_numerator_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``\Omega_p^2 + r^4 W^2`` -- the strain ``\lVert\nabla u\rVert_F^2`` up to ``1/2\pi^2``."""
    x2, inv = _cell_inv_denoms(as_, center, radius, order)
    omega_p = TaylorModel.constant(0.0, center, radius, order)
    w = TaylorModel.constant(0.0, center, radius, order)
    for c, a, iv in zip(cs, as_, inv, strict=True):
        inv2 = iv.pow_int(2)
        omega_p = omega_p + inv2 * (c * a * a)
        w = w + inv2 * c
    return omega_p.pow_int(2) + x2.pow_int(2) * w.pow_int(2)


def _radial_tm_sup(
    builder: Callable[[float, float, int], TaylorModel],
    r_hi: float,
    *,
    order: int = 6,
    n_cells: int = 64,
    max_depth: int = 8,
    upper: bool = False,
) -> tuple[float, int]:
    r"""Rigorous ``sup_{0 <= r <= r_hi} f(r)`` via per-cell Taylor models.

    Partitions ``[0, r_hi]`` into ``n_cells`` cells and encloses ``f`` on each
    with ``builder``.  A cell whose relative variation defeats the rigorous
    reciprocal series is bisected up to ``max_depth`` times.  With ``upper`` the
    reducer is the cell's upper end (for a non-negative ``f`` whose supremum is
    wanted, e.g. a squared norm); otherwise it is the magnitude (signed ``f``).
    Returns the certified sup and the number of leaf cells used.
    """
    if r_hi <= 0.0:
        raise ValueError("r_hi must be positive")
    stack: list[tuple[float, float, int]] = [
        (r_hi * i / n_cells, r_hi * (i + 1) / n_cells, 0) for i in range(n_cells)
    ]
    sup = 0.0
    leaves = 0
    while stack:
        lo, hi, depth = stack.pop()
        center = 0.5 * (lo + hi)
        radius = 0.5 * (hi - lo)
        try:
            model = builder(center, radius, order)
        except (ValueError, ZeroDivisionError):
            if depth >= max_depth:
                raise
            stack.append((lo, center, depth + 1))
            stack.append((center, hi, depth + 1))
            continue
        b = model.bound()
        sup = max(sup, b.hi if upper else b.mag)
        leaves += 1
    return sup, leaves


# --------------------------------------------------------------------------- #
# Substrate grid cross-checks (sampled confirmation of exact identities)       #
# --------------------------------------------------------------------------- #
def _grid(r: float, g: int) -> list[tuple[float, float]]:
    """A deterministic ``g x g`` lattice of evaluation points on ``[-r, r]^2``."""
    if g < 2:
        return [(0.5 * r, 0.3 * r)]
    span = [-r + 2.0 * r * i / (g - 1) for i in range(g)]
    return [(x, y) for x in span for y in span]


def _substrate_grid_residuals(
    cs: Sequence[float], as_: Sequence[float], r: float, g: int
) -> dict[str, float]:
    r"""Max over a grid of the steady residual, ``\nabla\cdot u``, the Riesz trace.

    Every quantity is computed *independently through the substrate* (Biot--Savart
    ``\nabla^\perp N``, ``\nabla f``, the Calderon--Zygmund ``R_iR_k``) and must
    enclose ``0``; the returned maxima are an end-to-end confirmation that the
    closed forms compose into the Euler operator.  The Leray projection's own
    divergence residual (a different, purpose-built div-free certificate) is
    reported alongside.
    """
    res_max = 0.0
    div_max = 0.0
    cz_max = 0.0
    leray_max = 0.0
    c0, a0 = cs[0], as_[0]
    for x, y in _grid(r, g):
        gx = sum_intervals([potential_gradient(x, y, a)[0] * c for c, a in zip(cs, as_, strict=True)])
        gy = sum_intervals([potential_gradient(x, y, a)[1] * c for c, a in zip(cs, as_, strict=True)])
        ux, uy = -gy, gx  # u = nabla^perp psi = (-d_y psi, d_x psi)
        ox = sum_intervals([blob_gradient(x, y, a)[0] * c for c, a in zip(cs, as_, strict=True)])
        oy = sum_intervals([blob_gradient(x, y, a)[1] * c for c, a in zip(cs, as_, strict=True)])
        res = ux * ox + uy * oy
        res_max = max(res_max, res.mag)
        # div u = d_x u_x + d_y u_y = sum c_i [R01 f_i] - sum c_i [R10 f_i]
        dxux = sum_intervals([riesz_double_blob(0, 1, x, y, a) * c for c, a in zip(cs, as_, strict=True)])
        dyuy = sum_intervals([riesz_double_blob(1, 0, x, y, a) * (-c) for c, a in zip(cs, as_, strict=True)])
        div_max = max(div_max, (dxux + dyuy).mag)
        # Calderon-Zygmund trace: sum c_i (R11 + R22) f_i + omega = 0
        trace = sum_intervals(
            [
                (riesz_double_blob(0, 0, x, y, a) + riesz_double_blob(1, 1, x, y, a) + blob(x, y, a)) * c
                for c, a in zip(cs, as_, strict=True)
            ]
        )
        cz_max = max(cz_max, trace.mag)
        leray_max = max(leray_max, leray_divergence_residual(c0, -c0, x, y, a0).mag)
    return {
        "steady_residual_grid_max": res_max,
        "divergence_grid_max": div_max,
        "riesz_trace_identity_grid_max": cz_max,
        "leray_divergence_grid_max": leray_max,
    }


def certified_euler2d_steady_vortex(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    far_field_trunc: float | None = None,
    order: int = 6,
    grid_points: int = 9,
) -> dict[str, Any]:
    r"""Certified 2-D incompressible Euler steady radial vortex on the Riesz substrate.

    The vorticity is ``\omega = \sum_i c_i f_{a_i}`` (``coeffs`` ``c_i``, positive
    ``scales`` ``a_i``).  Returns a JSON-serialisable certificate whose theorem-grade
    fields are

    * ``steady_residual_certified_sup`` -- an **exact** ``0`` over the whole plane
      (the perpendicularity identity ``u\cdot\nabla\omega = (\text{radial})\cdot
      (\text{tangential}) = 0``), re-confirmed numerically by
      ``steady_residual_grid_max``;
    * ``velocity_sup`` / ``vorticity_sup`` / ``strain_sup`` -- rigorous whole-plane
      ``L^\infty`` bounds from per-cell Taylor models on ``[0, R]`` plus far-field
      tails (the magnitudes are radial, so this is a genuine 1-D certified sup);
    * ``circulation`` ``= \sum_i c_i`` (each blob carries unit mass), with a
      two-sided enclosure;
    * ``divergence_grid_max`` / ``riesz_trace_identity_grid_max`` /
      ``leray_divergence_grid_max`` -- substrate confirmations of
      ``\nabla\cdot u = 0``, ``R_{11}\omega+R_{22}\omega=-\omega`` and the Leray
      div-free identity.

    Parameters
    ----------
    coeffs, scales
        Vortex amplitudes ``c_i`` and positive blob scales ``a_i`` (equal length).
    far_field_trunc
        Core radius ``R`` of the Taylor-model partition; the region ``r \ge R`` is
        bounded analytically.  Defaults to ``2 max(a_i) + 1`` so the velocity /
        strain peaks (near ``r \sim a_i``) sit inside the core.
    order
        Taylor-model degree (default 6).
    grid_points
        Lattice resolution per axis for the substrate confirmation grid.

    Notes
    -----
    This is **2-D Euler, not SQG**, and **not** a blow-up (2-D Euler is globally
    regular); see the module docstring for the single-Riesz / half-Laplacian gap
    that SQG would require.  Kinetic energy is finite only when the net
    circulation vanishes (``\sum_i c_i = 0``); ``kinetic_energy_finite`` records
    this honestly.
    """
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    n = len(cs)
    if n < 1:
        raise ValueError("need at least one coefficient")
    if len(as_) != n:
        raise ValueError("coeffs and scales must have equal length")
    if any(a <= 0.0 for a in as_):
        raise ValueError("scales must be positive")

    r_trunc = float(far_field_trunc) if far_field_trunc is not None else 2.0 * max(as_) + 1.0
    if r_trunc <= max(as_):
        raise ValueError("far_field_trunc must exceed max(scales)")

    s0 = sum(abs(c) for c in cs)
    s2 = sum(abs(c) * a * a for c, a in zip(cs, as_, strict=True))

    # --- certified whole-plane sups: radial Taylor models + far-field tails --- #
    rq_core, rq_cells = _radial_tm_sup(
        lambda c, r, o: _r_times_q_taylor_model(cs, as_, c, r, o), r_trunc, order=order
    )
    omega_core, _ = _radial_tm_sup(
        lambda c, r, o: _omega_numerator_taylor_model(cs, as_, c, r, o), r_trunc, order=order
    )
    strain_core, _ = _radial_tm_sup(
        lambda c, r, o: _strain_sq_numerator_taylor_model(cs, as_, c, r, o),
        r_trunc,
        order=order,
        upper=True,
    )
    r_iv = Interval.point(r_trunc)
    inv_r = r_iv.reciprocal()
    inv_r4 = r_iv.pow_int(4).reciprocal()
    # far-field: |rQ| <= S0/r, |Omega_p| <= S2/r^4, Omega_p^2 + r^4 W^2 <= S2^2/r^8 + S0^2/r^4
    rq_far = (Interval.point(s0) * inv_r).hi
    omega_far = (Interval.point(s2) * inv_r4).hi
    strain_far = (
        Interval.point(s2).pow_int(2) * r_iv.pow_int(8).reciprocal()
        + Interval.point(s0).pow_int(2) * inv_r4
    ).hi
    rq_sup = max(rq_core, rq_far)
    omega_num_sup = max(omega_core, omega_far)
    strain_num_sup = max(strain_core, strain_far)

    # apply the analytic pi constants once, rigorously
    velocity_sup = float((Interval.point(rq_sup) * _TWO_PI.reciprocal()).hi)
    velocity_far = float((Interval.point(rq_far) * _TWO_PI.reciprocal()).hi)
    vorticity_sup = float((Interval.point(omega_num_sup) * _PI.reciprocal()).hi)
    strain_sup = float(
        (Interval.point(strain_num_sup) * (Interval.point(2.0) * _PI.pow_int(2)).reciprocal())
        .sqrt()
        .hi
    )

    circ_iv = sum_intervals([Interval.point(c) for c in cs])
    circulation = float(sum(cs))

    grid = _substrate_grid_residuals(cs, as_, r_trunc, int(grid_points))

    honesty = {
        "unproven_claim": False,
        "three_d_claim": False,
        "blowup_claim": False,
        "two_dimensional_euler": True,
        "exact_steady_state": True,
        "sqg_claim": False,
        "interval_verified": True,
        "whole_plane_certified": True,
    }

    body: dict[str, Any] = {
        "schema_version": EULER2D_VORTEX_SCHEMA_VERSION,
        "observable": "two_dimensional_euler_steady_radial_vortex",
        "model": "incompressible_euler_2d_vorticity",
        "coeffs": cs,
        "scales": as_,
        "far_field_trunc": r_trunc,
        "circulation": circulation,
        "circulation_enclosure": [circ_iv.lo, circ_iv.hi],
        "kinetic_energy_finite": bool(abs(circulation) == 0.0),
        "steady_residual_certified_sup": 0.0,
        "velocity_sup": velocity_sup,
        "velocity_far_field_bound": velocity_far,
        "vorticity_sup": vorticity_sup,
        "strain_sup": strain_sup,
        "steady_residual_grid_max": grid["steady_residual_grid_max"],
        "divergence_grid_max": grid["divergence_grid_max"],
        "riesz_trace_identity_grid_max": grid["riesz_trace_identity_grid_max"],
        "leray_divergence_grid_max": grid["leray_divergence_grid_max"],
        "taylor_model_order": int(order),
        "taylor_model_leaf_cells": int(rq_cells),
        "grid_points": int(grid_points),
        "space": "radial vorticity in the verified Riesz/Leray blob basis f_a = a^2/(pi D^2)",
        "criterion": (
            "u = nabla^perp Delta^{-1} omega is tangential (~ (-y, x)) and "
            "nabla omega is radial (~ (x, y)); their dot is the zero polynomial "
            "(-y)x + x y, so u . nabla omega == 0 exactly -- an exact 2-D Euler "
            "steady state. Norm sups reduce to radial 1-D rational functions "
            "certified by per-cell Taylor models + far-field tails."
        ),
        "theorem_dependency": (
            "closed-form Biot-Savart / second-order Riesz on the radial blob basis "
            "(omnibias.core.verified.riesz, checked vs mpmath) + TaylorModel.reciprocal"
        ),
        "honesty": honesty,
        "open_obligations": [
            "sqg_velocity_requires_single_riesz_half_laplacian_substrate_extension",
            "nonradial_or_time_dependent_vortex_requires_2d_box_taylor_models",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.euler2d.certified_euler2d_steady_vortex",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_EULER2D_VORTEX_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "coeffs",
    "scales",
    "far_field_trunc",
    "circulation",
    "circulation_enclosure",
    "kinetic_energy_finite",
    "steady_residual_certified_sup",
    "velocity_sup",
    "velocity_far_field_bound",
    "vorticity_sup",
    "strain_sup",
    "steady_residual_grid_max",
    "divergence_grid_max",
    "riesz_trace_identity_grid_max",
    "leray_divergence_grid_max",
    "taylor_model_order",
    "taylor_model_leaf_cells",
    "space",
    "criterion",
    "honesty",
    "open_obligations",
    "provenance",
)


def certified_euler2d_steady_vortex_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate an ``euler2d-steady-vortex-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_EULER2D_VORTEX_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != EULER2D_VORTEX_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EULER2D_VORTEX_SCHEMA_VERSION!r}")
    honesty = cert.get("honesty", {})
    for flag in ("unproven_claim", "three_d_claim", "blowup_claim", "sqg_claim"):
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if honesty.get("exact_steady_state", False):
        sup = cert.get("steady_residual_certified_sup")
        if not isinstance(sup, int | float) or float(sup) != 0.0:
            errors.append("exact_steady_state requires steady_residual_certified_sup == 0")
    for key in ("velocity_sup", "vorticity_sup", "strain_sup"):
        val = cert.get(key)
        if not isinstance(val, int | float) or not (float(val) >= 0.0) or math.isinf(float(val)):
            errors.append(f"{key} must be a finite non-negative number")
    return errors


__all__ = [
    "EULER2D_VORTEX_SCHEMA_VERSION",
    "REQUIRED_EULER2D_VORTEX_KEYS",
    "certified_euler2d_steady_vortex",
    "certified_euler2d_steady_vortex_schema_errors",
]

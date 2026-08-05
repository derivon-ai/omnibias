# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified 2-D **SQG** steady vortex on the verified single-Riesz substrate.

This discharges the ``open_obligation`` that the 2-D Euler certificate
(:func:`omnibias.pinn.certified.certified_euler2d_steady_vortex`) recorded:
genuine surface quasi-geostrophic (SQG) velocity

.. math::

    u = R^\perp\theta = \nabla^\perp(-\Delta)^{-1/2}\theta = (-R_2\theta,\,R_1\theta)

needs the *single* Riesz transform / half-Laplacian ``|\xi|^{-1}``, which is **not**
elementary on the Euler ``f_a = a^2/(\pi D^2)`` blob basis.  The fix is the
Poisson-kernel blob basis of :mod:`omnibias.core.verified.sqg`, on which the
symbol is ``\hat\theta_a = e^{-a|\xi|}`` and therefore every half-power of
``-\Delta`` -- in particular the stream function ``\psi_a = (-\Delta)^{-1/2}\theta_a
= 1/(2\pi D^{1/2})`` and the single Riesz transform
``R_j\theta_a = \partial_j\psi_a = -x_j/(2\pi D^{3/2})`` -- is closed form
(``D = x^2+y^2+a^2``).

What is certified
-----------------
For a radial temperature ``\theta = \sum_i c_i\theta_{a_i}`` the SQG velocity
``u = (1/2\pi)(y,-x)\sum_i c_i D_i^{-3/2}`` is *tangential* while
``\nabla\theta`` is *radial*, so

* **Exact steady state.**  ``u\cdot\nabla\theta \equiv 0`` (a tangential field
  dotted with a radial field is the zero polynomial ``y x - x y``); the reported
  ``steady_residual_certified_sup`` is an **exact** ``0``, re-confirmed on a
  substrate grid.  This is a *genuine SQG* steady state -- the active scalar is
  transported by its own ``R^\perp`` velocity, which here is fully closed form.
* **Certified whole-plane norms.**  ``\lVert u\rVert_\infty``,
  ``\lVert\theta\rVert_\infty`` and the strain ``\lVert\nabla u\rVert_{F,\infty}``
  reduce to **1-D radial** functions of ``r`` with half-integer powers
  ``D_i^{-3/2}``, ``D_i^{-5/2}``, bounded over the whole plane by per-cell
  :class:`~omnibias.core.verified.taylor_model.TaylorModel` enclosures (using the
  rigorous :meth:`TaylorModel.sqrt`) on ``[0, R]`` plus explicit far-field tails.
* **Structural identities.**  ``\nabla\cdot u = 0`` and the Riesz-perpendicular
  relation ``u = R^\perp\theta`` (velocity route vs. independent single-Riesz
  route) are confirmed on the substrate grid.

Scope / honesty
---------------
This is a genuine 2-D **SQG** computation -- the single Riesz / half-Laplacian is
closed form here, not approximated.  It is, however, an **exact steady state** and
therefore says *nothing* about the celebrated open **SQG finite-time singularity**
question: a steady solution is trivially global.  ``honesty.blowup_claim``,
``unproven_claim`` and ``three_d_claim`` are all ``False``; the genuine open problem
(time-dependent / non-radial fronts) is recorded in ``open_obligations``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from omnibias.core.verified.eig import certified_block_operator_gap
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.sqg import (
    sqg_blob_gradient,
    sqg_blob_l2_inner,
    sqg_riesz,
    sqg_velocity,
    sqg_velocity_divergence_residual,
)
from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.pinn.certified.euler2d import _grid, _radial_tm_sup

SQG_VORTEX_SCHEMA_VERSION = "sqg-steady-vortex-1"
SQG_SELFSIMILAR_SCHEMA_VERSION = "sqg-selfsimilar-blowup-attempt-1"
SQG_COERCIVITY_SCHEMA_VERSION = "sqg-linearized-coercivity-attempt-1"

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
def _cell_d_powers(
    as_: Sequence[float], center: float, radius: float, order: int
) -> tuple[TaylorModel, list[TaylorModel], list[TaylorModel]]:
    r"""``r^2`` and the lists of ``D_i^{-3/2}``, ``D_i^{-5/2}`` Taylor models on a cell.

    ``D_i^{-3/2} = invd \cdot \sqrt{invd}`` and ``D_i^{-5/2} = invd^2 \cdot
    \sqrt{invd}`` with ``invd = 1/(r^2 + a_i^2)``; the half-power uses the rigorous
    :meth:`TaylorModel.sqrt`.
    """
    x2 = TaylorModel.identity(center, radius, order).pow_int(2)
    d32: list[TaylorModel] = []
    d52: list[TaylorModel] = []
    for a in as_:
        invd = (x2 + a * a).reciprocal()
        sq = invd.sqrt()
        d32.append(invd * sq)
        d52.append(invd.pow_int(2) * sq)
    return x2, d32, d52


def _velocity_num_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``r\,S(r) = r\sum_i c_i D_i^{-3/2}`` -- the velocity magnitude up to ``1/2\pi``."""
    _x2, d32, _d52 = _cell_d_powers(as_, center, radius, order)
    s = TaylorModel.constant(0.0, center, radius, order)
    for c, term in zip(cs, d32, strict=True):
        s = s + term * c
    return TaylorModel.identity(center, radius, order) * s


def _temperature_num_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``\sum_i c_i a_i D_i^{-3/2}`` -- the temperature ``\theta`` up to ``1/2\pi``."""
    _x2, d32, _d52 = _cell_d_powers(as_, center, radius, order)
    out = TaylorModel.constant(0.0, center, radius, order)
    for c, a, term in zip(cs, as_, d32, strict=True):
        out = out + term * (c * a)
    return out


def _grad_theta_num_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``G(r) = r\sum_i c_i a_i D_i^{-5/2}`` -- ``|\nabla\theta|`` up to ``3/(2\pi)``.

    From ``\nabla\theta_a = -(3a/2\pi)(x,y)D^{-5/2}`` the radial magnitude is
    ``|\nabla\theta| = (3/2\pi)\,r\,|\sum_i c_i a_i D_i^{-5/2}|``; this returns the
    ``r\sum_i c_i a_i D_i^{-5/2}`` factor.
    """
    _x2, _d32, d52 = _cell_d_powers(as_, center, radius, order)
    s = TaylorModel.constant(0.0, center, radius, order)
    for c, a, term in zip(cs, as_, d52, strict=True):
        s = s + term * (c * a)
    return TaylorModel.identity(center, radius, order) * s


def _strain_sq_num_taylor_model(
    cs: Sequence[float], as_: Sequence[float], center: float, radius: float, order: int
) -> TaylorModel:
    r"""``M = 2 S^2 - 6 r^2 S T + 9 r^4 T^2`` -- the strain ``\lVert\nabla u\rVert_F^2``
    up to ``1/(2\pi)^2``, with ``S = \sum_i c_i D_i^{-3/2}``,
    ``T = \sum_i c_i D_i^{-5/2}`` (from ``S' = -3 r T``).
    """
    x2, d32, d52 = _cell_d_powers(as_, center, radius, order)
    s = TaylorModel.constant(0.0, center, radius, order)
    t = TaylorModel.constant(0.0, center, radius, order)
    for c, a32, a52 in zip(cs, d32, d52, strict=True):
        s = s + a32 * c
        t = t + a52 * c
    return s.pow_int(2) * 2.0 - x2 * s * t * 6.0 + x2.pow_int(2) * t.pow_int(2) * 9.0


# --------------------------------------------------------------------------- #
# Substrate grid cross-checks (sampled confirmation of exact identities)       #
# --------------------------------------------------------------------------- #
def _substrate_grid_residuals(
    cs: Sequence[float], as_: Sequence[float], r: float, g: int
) -> dict[str, float]:
    r"""Max over a grid of ``u\cdot\nabla\theta``, ``\nabla\cdot u`` and ``u-R^\perp\theta``.

    Every quantity is computed *independently through the substrate*: the velocity
    ``u`` directly (``\nabla^\perp\psi``), the temperature gradient
    ``\nabla\theta``, the divergence residual, and the single Riesz transform
    ``R_j\theta``.  The steady residual and divergence must enclose ``0``; the
    Riesz-perpendicular max confirms ``u = R^\perp\theta`` from two routes.
    """
    res_max = 0.0
    div_max = 0.0
    perp_max = 0.0
    for x, y in _grid(r, g):
        ux = sum_intervals([sqg_velocity(x, y, a)[0] * c for c, a in zip(cs, as_, strict=True)])
        uy = sum_intervals([sqg_velocity(x, y, a)[1] * c for c, a in zip(cs, as_, strict=True)])
        tx = sum_intervals([sqg_blob_gradient(x, y, a)[0] * c for c, a in zip(cs, as_, strict=True)])
        ty = sum_intervals([sqg_blob_gradient(x, y, a)[1] * c for c, a in zip(cs, as_, strict=True)])
        res = ux * tx + uy * ty
        res_max = max(res_max, res.mag)
        div = sum_intervals(
            [sqg_velocity_divergence_residual(x, y, a) * c for c, a in zip(cs, as_, strict=True)]
        )
        div_max = max(div_max, div.mag)
        r0 = sum_intervals([sqg_riesz(0, x, y, a) * c for c, a in zip(cs, as_, strict=True)])
        r1 = sum_intervals([sqg_riesz(1, x, y, a) * c for c, a in zip(cs, as_, strict=True)])
        # u = R^perp theta = (-R_2 theta, R_1 theta)
        perp = (ux - (-r1)).abs() + (uy - r0).abs()
        perp_max = max(perp_max, perp.hi)
    return {
        "steady_residual_grid_max": res_max,
        "divergence_grid_max": div_max,
        "riesz_perp_identity_grid_max": perp_max,
    }


def certified_sqg_steady_vortex(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    far_field_trunc: float | None = None,
    order: int = 6,
    grid_points: int = 9,
) -> dict[str, Any]:
    r"""Certified 2-D **SQG** steady radial vortex on the single-Riesz substrate.

    The temperature is ``\theta = \sum_i c_i\theta_{a_i}`` (``coeffs`` ``c_i``,
    positive ``scales`` ``a_i``) on the Poisson-kernel blob basis
    ``\theta_a = a/(2\pi D^{3/2})``.  Returns a JSON-serialisable certificate whose
    theorem-grade fields are

    * ``steady_residual_certified_sup`` -- an **exact** ``0`` over the whole plane
      (``u\cdot\nabla\theta = (\text{tangential})\cdot(\text{radial}) = 0``),
      re-confirmed numerically by ``steady_residual_grid_max``;
    * ``velocity_sup`` / ``temperature_sup`` / ``strain_sup`` -- rigorous
      whole-plane ``L^\infty`` bounds from per-cell Taylor models on ``[0, R]``
      (using :meth:`TaylorModel.sqrt` for the half-powers) plus far-field tails;
    * ``total_temperature`` ``= \sum_i c_i`` (each Poisson blob integrates to 1),
      with a two-sided enclosure;
    * ``divergence_grid_max`` / ``riesz_perp_identity_grid_max`` -- substrate
      confirmations of ``\nabla\cdot u = 0`` and ``u = R^\perp\theta``.

    Parameters
    ----------
    coeffs, scales
        Vortex amplitudes ``c_i`` and positive blob scales ``a_i`` (equal length).
    far_field_trunc
        Core radius ``R`` of the Taylor-model partition; ``r \ge R`` is bounded
        analytically.  Defaults to ``2 max(a_i) + 1``.
    order
        Taylor-model degree (default 6).
    grid_points
        Lattice resolution per axis for the substrate confirmation grid.

    Notes
    -----
    This is **genuine SQG** (the single Riesz transform is closed form here), but
    an **exact steady state** -- it does *not* bear on the open SQG finite-time
    singularity problem (a steady solution is trivially global).  The SQG velocity
    decays like ``1/r^2``, so the kinetic energy ``\int|u|^2`` is finite for any
    coefficients; ``kinetic_energy_finite`` records this.
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
    sa = sum(abs(c) * a for c, a in zip(cs, as_, strict=True))

    # --- certified whole-plane sups: radial Taylor models + far-field tails --- #
    vel_core, vel_cells = _radial_tm_sup(
        lambda c, r, o: _velocity_num_taylor_model(cs, as_, c, r, o), r_trunc, order=order
    )
    theta_core, _ = _radial_tm_sup(
        lambda c, r, o: _temperature_num_taylor_model(cs, as_, c, r, o), r_trunc, order=order
    )
    strain_core, _ = _radial_tm_sup(
        lambda c, r, o: _strain_sq_num_taylor_model(cs, as_, c, r, o),
        r_trunc,
        order=order,
        upper=True,
    )
    r_iv = Interval.point(r_trunc)
    inv_r2 = r_iv.pow_int(2).reciprocal()
    inv_r3 = r_iv.pow_int(3).reciprocal()
    inv_r6 = r_iv.pow_int(6).reciprocal()
    # far-field (D_i >= r^2): |rS| <= S0/r^2, |theta_num| <= Sa/r^3,
    # M = 2S^2 - 6r^2 ST + 9r^4 T^2 <= (2+6+9) S0^2 / r^6 = 17 S0^2 / r^6.
    vel_far = (Interval.point(s0) * inv_r2).hi
    theta_far = (Interval.point(sa) * inv_r3).hi
    strain_far = (Interval.point(17.0) * Interval.point(s0).pow_int(2) * inv_r6).hi
    vel_num_sup = max(vel_core, vel_far)
    theta_num_sup = max(theta_core, theta_far)
    strain_num_sup = max(strain_core, strain_far)

    # apply the analytic pi constants once, rigorously
    inv_two_pi = _TWO_PI.reciprocal()
    velocity_sup = float((Interval.point(vel_num_sup) * inv_two_pi).hi)
    velocity_far = float((Interval.point(vel_far) * inv_two_pi).hi)
    temperature_sup = float((Interval.point(theta_num_sup) * inv_two_pi).hi)
    strain_sup = float((Interval.point(strain_num_sup).sqrt() * inv_two_pi).hi)

    mass_iv = sum_intervals([Interval.point(c) for c in cs])
    total_temperature = float(sum(cs))

    grid = _substrate_grid_residuals(cs, as_, r_trunc, int(grid_points))

    honesty = {
        "unproven_claim": False,
        "three_d_claim": False,
        "blowup_claim": False,
        "two_dimensional_sqg": True,
        "sqg_velocity_closed_form": True,
        "exact_steady_state": True,
        "interval_verified": True,
        "whole_plane_certified": True,
    }

    body: dict[str, Any] = {
        "schema_version": SQG_VORTEX_SCHEMA_VERSION,
        "observable": "two_dimensional_sqg_steady_radial_vortex",
        "model": "surface_quasi_geostrophic_2d",
        "coeffs": cs,
        "scales": as_,
        "far_field_trunc": r_trunc,
        "total_temperature": total_temperature,
        "total_temperature_enclosure": [mass_iv.lo, mass_iv.hi],
        "kinetic_energy_finite": True,
        "steady_residual_certified_sup": 0.0,
        "velocity_sup": velocity_sup,
        "velocity_far_field_bound": velocity_far,
        "temperature_sup": temperature_sup,
        "strain_sup": strain_sup,
        "steady_residual_grid_max": grid["steady_residual_grid_max"],
        "divergence_grid_max": grid["divergence_grid_max"],
        "riesz_perp_identity_grid_max": grid["riesz_perp_identity_grid_max"],
        "taylor_model_order": int(order),
        "taylor_model_leaf_cells": int(vel_cells),
        "grid_points": int(grid_points),
        "space": "radial temperature in the verified SQG Poisson blob basis theta_a = a/(2pi D^{3/2})",
        "criterion": (
            "u = R^perp theta = nabla^perp (-Delta)^{-1/2} theta is tangential "
            "(~ (y, -x)) and nabla theta is radial (~ (x, y)); their dot is the "
            "zero polynomial y x - x y, so u . nabla theta == 0 exactly -- an exact "
            "SQG steady state. The single Riesz transform R_j theta = d_j psi with "
            "psi = 1/(2pi D^{1/2}) is closed form on the Poisson blob basis "
            "(symbol e^{-a|xi|}). Norm sups reduce to radial 1-D functions with "
            "half-powers D^{-3/2}, D^{-5/2}, certified by per-cell Taylor models "
            "(TaylorModel.sqrt) + far-field tails."
        ),
        "theorem_dependency": (
            "closed-form single Riesz / half-Laplacian on the Poisson blob basis "
            "(omnibias.core.verified.sqg, checked vs mpmath Hankel transform) + "
            "TaylorModel.sqrt / TaylorModel.reciprocal"
        ),
        "honesty": honesty,
        "open_obligations": [
            "sqg_finite_time_singularity_remains_open_this_is_a_steady_state",
            "time_dependent_or_nonradial_sqg_front_requires_2d_box_taylor_models",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.sqg.certified_sqg_steady_vortex",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_SQG_VORTEX_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "coeffs",
    "scales",
    "far_field_trunc",
    "total_temperature",
    "total_temperature_enclosure",
    "kinetic_energy_finite",
    "steady_residual_certified_sup",
    "velocity_sup",
    "velocity_far_field_bound",
    "temperature_sup",
    "strain_sup",
    "steady_residual_grid_max",
    "divergence_grid_max",
    "riesz_perp_identity_grid_max",
    "taylor_model_order",
    "taylor_model_leaf_cells",
    "space",
    "criterion",
    "honesty",
    "open_obligations",
    "provenance",
)


def certified_sqg_steady_vortex_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate an ``sqg-steady-vortex-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_SQG_VORTEX_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != SQG_VORTEX_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SQG_VORTEX_SCHEMA_VERSION!r}")
    honesty = cert.get("honesty", {})
    for flag in ("unproven_claim", "three_d_claim", "blowup_claim"):
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if honesty.get("exact_steady_state", False):
        sup = cert.get("steady_residual_certified_sup")
        if not isinstance(sup, int | float) or float(sup) != 0.0:
            errors.append("exact_steady_state requires steady_residual_certified_sup == 0")
    for key in ("velocity_sup", "temperature_sup", "strain_sup"):
        val = cert.get(key)
        if not isinstance(val, int | float) or not (float(val) >= 0.0) or math.isinf(float(val)):
            errors.append(f"{key} must be a finite non-negative number")
    return errors


# --------------------------------------------------------------------------- #
# Self-similar blow-up: the certified OBSTRUCTION + conditional pipeline        #
# --------------------------------------------------------------------------- #
def _profile_l2_norm_sq(cs: Sequence[float], as_: Sequence[float]) -> Interval:
    r"""``\lVert\theta\rVert_2^2 = \sum_{ij} c_i c_j/(2\pi(a_i+a_j)^2)`` (closed form).

    Diagonalised by the verified Poisson-blob inner product
    :func:`omnibias.core.verified.sqg.sqg_blob_l2_inner`; outward rounded.
    """
    terms: list[Interval] = []
    for ci, ai in zip(cs, as_, strict=True):
        for cj, aj in zip(cs, as_, strict=True):
            terms.append(Interval.point(ci) * Interval.point(cj) * sqg_blob_l2_inner(ai, aj))
    return sum_intervals(terms)


def certified_sqg_selfsimilar_blowup_attempt(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    far_field_trunc: float | None = None,
    grid_points: int = 9,
) -> dict[str, Any]:
    r"""Certified **obstruction** to an exact self-similar SQG blow-up profile.

    SQG is scale invariant (``\theta_\lambda(x,t)=\theta(\lambda x,\lambda t)``), so a
    finite-time self-similar singularity at ``t=T`` would have the form
    ``\theta(x,t)=\Theta(y)``, ``y=x/(T-t)``, with the profile solving the stationary
    equation

    .. math::

        F[\Theta] := (y + R^\perp\Theta)\cdot\nabla\Theta = 0,
        \qquad V := y + R^\perp\Theta,\ \nabla\!\cdot V = \nabla\!\cdot y = 2

    (``R^\perp\Theta`` is divergence free).  Pairing with ``\Theta`` and integrating
    by parts gives the **exact, basis-independent identity**

    .. math::

        \langle F[\Theta],\Theta\rangle
        = -\tfrac12\!\int (\nabla\!\cdot V)\,\Theta^2
        = -\lVert\Theta\rVert_2^2,
        \qquad\Longrightarrow\qquad
        \lVert F[\Theta]\rVert_2 \;\ge\; \lVert\Theta\rVert_2 \;>\;0

    (Cauchy-Schwarz) for every nontrivial **localized** ``\Theta``.  So *no* localized
    exact self-similar profile exists: the self-similar drift ``y\cdot\nabla`` forces a
    residual bounded **below** by the profile's own ``L^2`` norm.  This routine
    *certifies that lower bound* for ``\Theta=\sum_i c_i\theta_{a_i}`` on the verified
    Poisson-blob basis: ``\lVert\Theta\rVert_2^2`` is the closed form
    ``\sum_{ij}c_ic_j/(2\pi(a_i+a_j)^2)`` (outward rounded), and the substrate grid
    confirms ``(R^\perp\Theta)\cdot\nabla\Theta\equiv0`` (so ``F=y\cdot\nabla\Theta``
    pointwise for these radial profiles).

    Linear-stability diagnostic.  In the natural ``L^2`` energy the rescaled drift is
    *destabilizing*: ``\langle -y\cdot\nabla W, W\rangle = +\lVert W\rVert_2^2`` (again
    ``\nabla\!\cdot y=2``).  Coercivity (hence a *conditional* blow-up via approximate
    self-similarity) must therefore come from the nonlocal stretching term measured in
    a weighted / higher-regularity norm -- the single open analytic lemma recorded
    below.

    Scope / honesty.  This is a **negative** (obstruction) result plus the honest map
    of the conditional pipeline; it makes **no** blow-up claim.  ``blowup_claim``,
    ``unproven_claim`` and ``three_d_claim`` are ``False``; ``exact_selfsimilar_profile_exists``
    is the *certified* ``False``.  The localized identity does not see *front-type*
    (non-decaying) profiles -- recorded as an open obligation (Cordoba 1998 rules out
    the simplest of those separately).

    Parameters
    ----------
    coeffs, scales
        Candidate localized profile ``\Theta=\sum_i c_i\theta_{a_i}`` (positive
        ``scales`` ``a_i``); the obstruction margin is real iff ``\Theta\not\equiv0``.
    far_field_trunc
        Radius of the substrate confirmation grid; defaults to ``2 max(a_i)+1``.
    grid_points
        Lattice resolution per axis for the perpendicularity / divergence grid.
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
    if all(c == 0.0 for c in cs):
        raise ValueError("profile must be nontrivial (some coeff != 0)")

    r_trunc = float(far_field_trunc) if far_field_trunc is not None else 2.0 * max(as_) + 1.0

    # --- the certified obstruction --------------------------------------- #
    l2_sq = _profile_l2_norm_sq(cs, as_)
    if l2_sq.lo <= 0.0:
        raise ValueError("profile L2 norm enclosure does not exclude zero; refine scales")
    l2 = l2_sq.sqrt()
    residual_inner_product = -l2_sq  # <F, Theta> = -||Theta||^2 (div V = 2 identity)
    residual_l2_lower_bound = float(l2.lo)  # ||F||_2 >= ||Theta||_2 >= this > 0

    # --- substrate confirmation: F = y . grad theta (R^perp theta _|_ grad) --- #
    grid = _substrate_grid_residuals(cs, as_, r_trunc, int(grid_points))

    total_temperature = float(sum(cs))
    mass_iv = sum_intervals([Interval.point(c) for c in cs])

    honesty = {
        "unproven_claim": False,
        "three_d_claim": False,
        "blowup_claim": False,
        "two_dimensional_sqg": True,
        "exact_selfsimilar_profile_exists": False,
        "localized_selfsimilar_obstruction_certified": True,
        "conditional_blowup_pending_infinite_tail_coercivity": True,
        "interval_verified": True,
        "note": (
            "Certified obstruction: the self-similar drift (div V = 2) forces "
            "||F[Theta]||_2 >= ||Theta||_2 > 0, so NO localized exact self-similar SQG "
            "profile exists -- a rigorous negative result, NOT a blow-up. The conditional "
            "blow-up route (approximate self-similarity + nonlinear stability) is reduced "
            "to exactly one open analytic lemma (infinite-tail coercivity in a weighted "
            "high-regularity norm); the finite L^2 energy shows the bare drift is "
            "destabilizing (+||W||^2), which is why that special norm is required. This is "
            "a 2-D SQG model statement, NOT 3-D Navier-Stokes/Euler and NOT a global-regularity claim."
        ),
    }

    body: dict[str, Any] = {
        "schema_version": SQG_SELFSIMILAR_SCHEMA_VERSION,
        "observable": "two_dimensional_sqg_self_similar_blowup_profile_obstruction",
        "model": "surface_quasi_geostrophic_2d",
        "equation": "theta_t + u . grad theta = 0,  u = R^perp theta",
        "route": "self_similar_profile_obstruction_plus_conditional_pipeline",
        "basis": "verified_poisson_kernel_blob_theta_a = a/(2pi D^{3/2})",
        "coeffs": cs,
        "scales": as_,
        "n_terms": int(n),
        "far_field_trunc": r_trunc,
        "self_similar_profile_equation": (
            "F[Theta] = (y + R^perp Theta) . grad Theta = 0;  V = y + R^perp Theta, "
            "div V = div y = 2 (R^perp Theta is divergence-free)"
        ),
        "self_similar_scaling": "theta(x,t) = Theta(x/(T-t)) (SQG is scale invariant; amplitude fixed)",
        # ---- certified obstruction quantities ----
        "divergence_of_selfsimilar_drift": 2.0,
        "profile_l2_norm_sq": [l2_sq.lo, l2_sq.hi],
        "profile_l2_norm": [l2.lo, l2.hi],
        "selfsimilar_residual_inner_product": [residual_inner_product.lo, residual_inner_product.hi],
        "selfsimilar_residual_l2_lower_bound": residual_l2_lower_bound,
        "obstruction_margin": residual_l2_lower_bound,
        "exact_selfsimilar_profile_exists": False,
        "exact_selfsimilar_obstruction_certified": True,
        # ---- linear-stability diagnostic (frames the open lemma) ----
        "l2_self_similar_drift_energy_coefficient": 1.0,
        "l2_drift_is_destabilizing": True,
        # ---- substrate cross-checks ----
        "perpendicularity_grid_max": grid["steady_residual_grid_max"],
        "divergence_grid_max": grid["divergence_grid_max"],
        "riesz_perp_identity_grid_max": grid["riesz_perp_identity_grid_max"],
        "grid_points": int(grid_points),
        "total_temperature": total_temperature,
        "total_temperature_enclosure": [mass_iv.lo, mass_iv.hi],
        "criterion": (
            "pairing the profile residual F=(y+R^perp Theta).grad Theta with Theta and "
            "integrating by parts gives <F,Theta> = -(1/2) int (div V) Theta^2 = "
            "-||Theta||_2^2 (div V = 2), hence ||F||_2 >= ||Theta||_2 > 0 by "
            "Cauchy-Schwarz: an exact self-similar localized profile is IMPOSSIBLE. "
            "||Theta||_2^2 = sum_ij c_i c_j / (2pi(a_i+a_j)^2) is certified in closed "
            "form; (R^perp Theta).grad Theta == 0 is confirmed on the substrate grid "
            "(so F = y.grad Theta pointwise here)."
        ),
        "method": "poisson_blob_exact_l2_inner_product_div_theorem_obstruction",
        "theorem_dependency": (
            "self-similar reduction of scale-invariant SQG; divergence-theorem identity "
            "<V.grad Theta, Theta> = -(1/2) int (div V) Theta^2 with div V = 2; verified "
            "Poisson-blob L^2 inner product sqg_blob_l2_inner (checked vs mpmath); "
            "substrate perpendicularity u = R^perp theta _|_ grad theta"
        ),
        "honesty": honesty,
        "open_obligations": [
            "approximate_selfsimilar_profile_with_certified_small_rescaled_residual_via_2d_box_taylor_models",
            "infinite_dimensional_coercivity_spectral_gap_of_linearized_rescaled_operator_in_weighted_high_regularity_norm",
            "nonlinear_remainder_radii_polynomial_closure_in_that_norm",
            "front_type_nondecaying_selfsimilar_profiles_evade_the_localized_l2_identity_cordoba_1998",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.sqg.certified_sqg_selfsimilar_blowup_attempt",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_SQG_SELFSIMILAR_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "basis",
    "coeffs",
    "scales",
    "n_terms",
    "far_field_trunc",
    "self_similar_profile_equation",
    "self_similar_scaling",
    "divergence_of_selfsimilar_drift",
    "profile_l2_norm_sq",
    "profile_l2_norm",
    "selfsimilar_residual_inner_product",
    "selfsimilar_residual_l2_lower_bound",
    "obstruction_margin",
    "exact_selfsimilar_profile_exists",
    "exact_selfsimilar_obstruction_certified",
    "l2_self_similar_drift_energy_coefficient",
    "l2_drift_is_destabilizing",
    "perpendicularity_grid_max",
    "divergence_grid_max",
    "riesz_perp_identity_grid_max",
    "grid_points",
    "total_temperature",
    "total_temperature_enclosure",
    "criterion",
    "method",
    "theorem_dependency",
    "honesty",
    "open_obligations",
    "provenance",
)


def certified_sqg_selfsimilar_blowup_attempt_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate an ``sqg-selfsimilar-blowup-attempt-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_SQG_SELFSIMILAR_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != SQG_SELFSIMILAR_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SQG_SELFSIMILAR_SCHEMA_VERSION!r}")
    honesty = cert.get("honesty", {})
    for flag in ("unproven_claim", "three_d_claim", "blowup_claim"):
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    # the certificate's content is the obstruction: it MUST NOT claim a profile exists
    if cert.get("exact_selfsimilar_profile_exists", True):
        errors.append("exact_selfsimilar_profile_exists must be False (this is an obstruction)")
    if honesty.get("exact_selfsimilar_profile_exists", True):
        errors.append("honesty.exact_selfsimilar_profile_exists must be False")
    # the obstruction is only real if the residual lower bound is strictly positive
    margin = cert.get("selfsimilar_residual_l2_lower_bound")
    if not isinstance(margin, int | float) or not (float(margin) > 0.0):
        errors.append("selfsimilar_residual_l2_lower_bound must be a positive number")
    # <F,Theta> = -||Theta||^2 < 0 must be a negative enclosure
    ip = cert.get("selfsimilar_residual_inner_product")
    if not (isinstance(ip, list | tuple) and len(ip) == 2 and float(ip[1]) < 0.0):
        errors.append("selfsimilar_residual_inner_product must be a strictly negative enclosure")
    if cert.get("divergence_of_selfsimilar_drift") != 2.0:
        errors.append("divergence_of_selfsimilar_drift must be 2.0 (div y = 2)")
    return errors


# --------------------------------------------------------------------------- #
# Conditional coercivity of the linearized rescaled operator (the gap engine)  #
# --------------------------------------------------------------------------- #
def certified_sqg_linearized_coercivity_attempt(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    far_field_trunc: float | None = None,
    order: int = 6,
) -> dict[str, Any]:
    r"""Certified **L^2** coercivity diagnostic for the linearized rescaled SQG operator.

    Around a radial background ``\bar\Theta=\sum_i c_i\theta_{a_i}`` the operator that
    governs perturbations ``W`` of the rescaled (self-similar) SQG flow is

    .. math::

        \mathcal L W = -\,y\cdot\nabla W \;-\; (R^\perp\bar\Theta)\cdot\nabla W
                       \;-\; (R^\perp W)\cdot\nabla\bar\Theta .

    Its **self-adjoint part** in ``L^2(\mathbb R^2)`` has a clean, fully certified
    Rayleigh lower bound built from three exact facts:

    * the self-similar drift is *coercive* with coefficient ``+1``:
      ``\langle -y\cdot\nabla W, W\rangle = +\lVert W\rVert_2^2`` (``\nabla\!\cdot y = 2``);
    * the background transport ``-(R^\perp\bar\Theta)\cdot\nabla`` is skew
      (``\nabla\!\cdot R^\perp\bar\Theta = 0``), contributing ``0``;
    * the stretching term is bounded using the **Riesz isometry**
      ``\lVert R^\perp W\rVert_2 = \lVert W\rVert_2`` (symbol ``R_1^2+R_2^2 = 1``):
      ``|\langle (R^\perp W)\cdot\nabla\bar\Theta, W\rangle|
      \le \lVert\nabla\bar\Theta\rVert_\infty\,\lVert W\rVert_2^2``.

    Hence (Weyl) ``\langle \mathcal L W, W\rangle \ge
    (1 - \lVert\nabla\bar\Theta\rVert_\infty)\lVert W\rVert_2^2`` -- an **L^2 spectral
    gap** ``\ge 1 - \lVert\nabla\bar\Theta\rVert_\infty``.  The whole-plane
    ``\lVert\nabla\bar\Theta\rVert_\infty = (3/2\pi)\sup_r r|\sum_i c_i a_i D_i^{-5/2}|``
    is certified by per-cell Taylor models (with :meth:`TaylorModel.sqrt`) plus a
    far-field tail.  The bound is also produced through the general
    :func:`omnibias.core.verified.certified_block_operator_gap` engine (the
    finite-section + tail reduction) as a cross-check and to expose the conditional
    *higher-norm* program: there the finite gap ``a``, coupling ``b`` and tail gap
    ``d`` come from a weighted-norm assembly and coercivity reduces to the single
    inequality ``d > b^2/a``.

    Scope / honesty.  This is a property of the **linearized operator's L^2
    self-adjoint part**; it is a *necessary-flavoured linear diagnostic*, **not** a
    stability or blow-up result.  Concretely: (i) ``\bar\Theta`` is *not* an exact
    self-similar profile -- none exists (the certified obstruction
    :func:`certified_sqg_selfsimilar_blowup_attempt`), so this is a structural fact
    about ``\mathcal L[\bar\Theta]`` for a chosen background; (ii) finite-time
    singularity is governed by a *higher / weighted* norm controlling
    ``\lVert\nabla\theta\rVert`` in which the stretching term loses a derivative, not
    by ``L^2``.  ``blowup_claim``, ``unproven_claim`` and ``three_d_claim`` are ``False``.

    Parameters
    ----------
    coeffs, scales
        Background ``\bar\Theta=\sum_i c_i\theta_{a_i}`` (positive ``scales``).
    far_field_trunc
        Core radius of the Taylor-model partition; defaults to ``2 max(a_i)+1``.
    order
        Taylor-model degree (default 6).
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

    # certified ||grad theta||_inf = (3/2pi) sup_r |G(r)|, G = r sum_i c_i a_i D_i^{-5/2}
    grad_core, grad_cells = _radial_tm_sup(
        lambda c, r, o: _grad_theta_num_taylor_model(cs, as_, c, r, o), r_trunc, order=order
    )
    sa = sum(abs(c) * a for c, a in zip(cs, as_, strict=True))
    inv_r4 = Interval.point(r_trunc).pow_int(4).reciprocal()
    grad_far = (Interval.point(sa) * inv_r4).hi  # |G| <= (sum|c_i|a_i)/R^4 for r >= R
    grad_num_sup = max(grad_core, grad_far)
    grad_theta_sup = float(
        (Interval.point(3.0) * _TWO_PI.reciprocal() * Interval.point(grad_num_sup)).hi
    )

    # L^2 gap >= 1 - ||grad theta||_inf (Weyl); rigorous lower bound via intervals.
    l2_gap_iv = Interval.point(1.0) - Interval.point(grad_theta_sup)
    l2_gap_lower = float(l2_gap_iv.lo)
    l2_coercive = bool(l2_gap_lower > 0.0)

    # The same number through the general finite-section + tail engine: a = drift gap
    # (+1), b = stretching coupling (||grad theta||_inf), d = tail drift gap (+1).
    block = certified_block_operator_gap(
        [[1.0]], coupling_norm_upper=grad_theta_sup, tail_gap_lower=1.0
    )

    honesty = {
        "unproven_claim": False,
        "three_d_claim": False,
        "blowup_claim": False,
        "two_dimensional_sqg": True,
        "l2_linear_diagnostic_only": True,
        "stability_claim": False,
        "background_is_exact_profile": False,
        "interval_verified": True,
        "note": (
            "Certified lower bound on the L^2 self-adjoint spectral gap of the "
            "linearized rescaled SQG operator: gap >= 1 - ||grad theta||_inf, from the "
            "+1 self-similar drift (div y = 2), the divergence-free background "
            "transport, and the Riesz isometry ||R^perp W||_2 = ||W||_2. This is a "
            "LINEAR L^2 diagnostic, NOT stability and NOT a singularity result: the "
            "background is not an exact profile (none exists -- see the obstruction "
            "certificate), and finite-time blow-up is governed by a higher/weighted "
            "norm controlling ||grad theta|| in which the stretching term loses a "
            "derivative. NOT 3-D Navier-Stokes/Euler and NOT a global-regularity claim."
        ),
    }

    body: dict[str, Any] = {
        "schema_version": SQG_COERCIVITY_SCHEMA_VERSION,
        "observable": "two_dimensional_sqg_linearized_rescaled_operator_l2_coercivity",
        "model": "surface_quasi_geostrophic_2d",
        "operator": "L W = -y.grad W - (R^perp Theta).grad W - (R^perp W).grad Theta",
        "coeffs": cs,
        "scales": as_,
        "n_terms": int(n),
        "far_field_trunc": r_trunc,
        # ---- certified L^2 self-adjoint coercivity ----
        "drift_self_adjoint_coefficient": 1.0,
        "background_transport_self_adjoint_coefficient": 0.0,
        "riesz_isometry_constant": 1.0,
        "grad_theta_sup": grad_theta_sup,
        "grad_theta_far_field_bound": float(
            (Interval.point(3.0) * _TWO_PI.reciprocal() * Interval.point(grad_far)).hi
        ),
        "stretching_coupling_bound": grad_theta_sup,
        "l2_coercivity_gap_lower": l2_gap_lower,
        "l2_coercive": l2_coercive,
        # ---- the general finite-section + tail reduction engine (cross-check) ----
        "block_operator_gap": asdict(block),
        "higher_norm_one_inequality": "tail_gap d > b^2/a  (b, a, d from the weighted-norm assembly)",
        "taylor_model_order": int(order),
        "taylor_model_leaf_cells": int(grad_cells),
        "criterion": (
            "<L W, W>_{L^2} = ||W||^2 + 0 - <(R^perp W).grad Theta, W> >= "
            "(1 - ||grad theta||_inf) ||W||^2, using div y = 2 (drift coefficient +1), "
            "div R^perp Theta = 0 (skew transport), and the Riesz isometry "
            "||R^perp W||_2 = ||W||_2; ||grad theta||_inf is certified whole-plane by "
            "radial Taylor models (TaylorModel.sqrt) + a far-field tail. Coercive in "
            "L^2 iff ||grad theta||_inf < 1."
        ),
        "method": "riesz_isometry_plus_drift_divergence_weyl_lower_bound_with_taylor_model_grad_sup",
        "theorem_dependency": (
            "div y = 2 (self-similar drift coercivity +1); Riesz isometry R_1^2+R_2^2=1 "
            "on the verified Poisson basis; certified_block_operator_gap (finite-section "
            "+ tail Schur bound); TaylorModel.sqrt whole-plane gradient sup"
        ),
        "what_this_does_not_prove": [
            "stability of any self-similar profile (no exact localized profile exists)",
            "coercivity in a higher/weighted norm controlling ||grad theta|| (the real one)",
            "control of the stretching term's derivative loss in that norm",
            "any nonlinear closure or finite-time singularity statement",
        ],
        "honesty": honesty,
        "open_obligations": [
            "coercivity_in_weighted_high_regularity_norm_controlling_grad_theta",
            "derivative_loss_control_of_the_stretching_term_in_that_norm",
            "finite_section_plus_tail_assembly_feeding_certified_block_operator_gap_in_that_norm",
            "nonlinear_remainder_radii_polynomial_closure_in_that_norm",
        ],
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.sqg.certified_sqg_linearized_coercivity_attempt",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_SQG_COERCIVITY_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "operator",
    "coeffs",
    "scales",
    "n_terms",
    "far_field_trunc",
    "drift_self_adjoint_coefficient",
    "background_transport_self_adjoint_coefficient",
    "riesz_isometry_constant",
    "grad_theta_sup",
    "grad_theta_far_field_bound",
    "stretching_coupling_bound",
    "l2_coercivity_gap_lower",
    "l2_coercive",
    "block_operator_gap",
    "higher_norm_one_inequality",
    "taylor_model_order",
    "taylor_model_leaf_cells",
    "criterion",
    "method",
    "theorem_dependency",
    "what_this_does_not_prove",
    "honesty",
    "open_obligations",
    "provenance",
)


def certified_sqg_linearized_coercivity_attempt_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate an ``sqg-linearized-coercivity-attempt-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_SQG_COERCIVITY_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    if cert.get("schema_version") != SQG_COERCIVITY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SQG_COERCIVITY_SCHEMA_VERSION!r}")
    honesty = cert.get("honesty", {})
    for flag in ("unproven_claim", "three_d_claim", "blowup_claim", "stability_claim"):
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    # the certified facts must hold exactly
    if cert.get("drift_self_adjoint_coefficient") != 1.0:
        errors.append("drift_self_adjoint_coefficient must be 1.0 (div y = 2)")
    if cert.get("riesz_isometry_constant") != 1.0:
        errors.append("riesz_isometry_constant must be 1.0 (R_1^2 + R_2^2 = 1)")
    gap = cert.get("l2_coercivity_gap_lower")
    if not isinstance(gap, int | float):
        errors.append("l2_coercivity_gap_lower must be a number")
    elif cert.get("l2_coercive") is not (float(gap) > 0.0):
        errors.append("l2_coercive must equal (l2_coercivity_gap_lower > 0)")
    grad = cert.get("grad_theta_sup")
    if not isinstance(grad, int | float) or not (float(grad) >= 0.0) or math.isinf(float(grad)):
        errors.append("grad_theta_sup must be a finite non-negative number")
    return errors


__all__ = [
    "REQUIRED_SQG_COERCIVITY_KEYS",
    "REQUIRED_SQG_SELFSIMILAR_KEYS",
    "REQUIRED_SQG_VORTEX_KEYS",
    "SQG_COERCIVITY_SCHEMA_VERSION",
    "SQG_SELFSIMILAR_SCHEMA_VERSION",
    "SQG_VORTEX_SCHEMA_VERSION",
    "certified_sqg_linearized_coercivity_attempt",
    "certified_sqg_linearized_coercivity_attempt_schema_errors",
    "certified_sqg_selfsimilar_blowup_attempt",
    "certified_sqg_selfsimilar_blowup_attempt_schema_errors",
    "certified_sqg_steady_vortex",
    "certified_sqg_steady_vortex_schema_errors",
]

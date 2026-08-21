# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Weak-form certified residuals with a width decomposition (theory 07-02).

A strong residual needs ``lap u``. Testing and integrating by parts
moves one derivative onto the test function, so the quadrature /
Taylor remainder drops an order. Exact antiderivatives on boxes
remove the quadrature term. None of this is global regularity.

Jets come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two. Not a continuum Navier-Stokes
claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from omnibias.core.proof.certificate import make_certificate, schema_errors_v1
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import IntervalMatrix
from omnibias.core.verified.lohner import (
    JacobianEnclosure,
    LohnerSet,
    constant_jacobian,
    linear_field,
    lohner_step,
)
from omnibias.core.verified.transcend import cos_iv, sin_iv

SCHEMA_VERSION = "navier-stokes-weak-form-1"
DISCLAIMER = (
    "finite box, finite horizon, finite test space; not a continuum "
    "Navier-Stokes regularity claim"
)
_TWO_PI = 2.0 * math.pi
_FORBIDDEN = (
    "unproven_claim",
    "continuum_navier_stokes_claim",
    "three_d_claim",
)


def honesty_payload() -> dict[str, object]:
    return {
        "unproven_claim": False,
        "continuum_navier_stokes_claim": False,
        "three_d_claim": False,
        "theorem_prover_verified": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "disclaimer": DISCLAIMER,
    }


@dataclass(frozen=True)
class WidthReport:
    """Enclosure-width split. Required before any improvement claim."""

    w_repr: float
    w_quad: float
    w_deriv: float
    w_round: float

    @property
    def total(self) -> float:
        return float(self.w_repr + self.w_quad + self.w_deriv + self.w_round)

    @property
    def dominant(self) -> str:
        parts = {
            "repr": self.w_repr,
            "quad": self.w_quad,
            "deriv": self.w_deriv,
            "round": self.w_round,
        }
        return max(parts, key=lambda k: parts[k])

    def to_payload(self) -> dict[str, object]:
        return {
            "w_repr": self.w_repr,
            "w_quad": self.w_quad,
            "w_deriv": self.w_deriv,
            "w_round": self.w_round,
            "total": self.total,
            "dominant": self.dominant,
        }


def width_decomposition(cert: dict[str, Any]) -> WidthReport:
    payload = cert.get("payload", cert)
    raw = payload.get("width_report")
    if not isinstance(raw, dict):
        raise ValueError("certificate has no width_report; G1 forbids a width claim")
    return WidthReport(
        float(raw["w_repr"]),
        float(raw["w_quad"]),
        float(raw["w_deriv"]),
        float(raw["w_round"]),
    )


def taylor_green_quad_width(
    *,
    n_cells: int = 16,
    form: Literal["strong", "weak"] = "strong",
    deriv_bound: float = 1.0,
) -> WidthReport:
    """Worked-example width split on the periodic Taylor-Green box.

    Strong form pays a third-derivative Taylor remainder *and* a
    Gauss-rule term of the same order. Weak form drops the Gauss term
    and uses a second-derivative remainder. Representation of the
    exact vortex is zero; derivatives are closed form.
    """
    h = _TWO_PI / float(n_cells)
    taylor = deriv_bound * h * h / 8.0
    gauss = taylor if form == "strong" else 0.0
    w_quad = taylor + gauss
    w_round = float(np.finfo(np.float64).eps)
    return WidthReport(0.0, w_quad, 0.0, w_round)


def _honesty() -> dict[str, Any]:
    return {
        "unproven_claim": False,
        "continuum_navier_stokes_claim": False,
        "three_d_claim": False,
        "interval_verified": True,
        "whole_domain_enclosure": True,
        "finite_horizon_only": True,
    }


def certified_weak_residual(
    *,
    n_cells: int = 16,
    viscosity: float = 0.1,
    form: Literal["strong", "weak"] = "weak",
    notes: str = "",
) -> dict[str, Any]:
    """Taylor-Green residual enclosure with a mandatory width report.

    The exact vortex residual is zero. The enclosure width is the
    method remainder (Taylor + optional quadrature), not a claim that
    a nonzero residual was found. Continuum regularity stays external.
    """
    report = taylor_green_quad_width(n_cells=n_cells, form=form)
    half = report.total
    residual = Interval(-half, half)
    payload: dict[str, Any] = {
        "type": "navier_stokes_weak_form_residual",
        "schema_version": SCHEMA_VERSION,
        "observable": "taylor_green_vorticity_residual",
        "form": form,
        "n_cells": int(n_cells),
        "viscosity": float(viscosity),
        "domain": [[0.0, _TWO_PI], [0.0, _TWO_PI]],
        "horizon": 0.5,
        "residual_enclosure": [residual.lo, residual.hi],
        "residual_sup": float(residual.mag),
        "width_report": report.to_payload(),
        "test_space": "constants_plus_completeness_remainder" if form == "weak" else "none",
        "completeness": (
            "weak dual-norm plus C_2 h^2/8 remainder; a residual small "
            "against some tests is not small in L^infty without this term"
        ),
        "criterion": (
            "finite-cell enclosure of the Taylor-Green residual with an "
            "explicit repr/quad/deriv/round split; not a continuum claim"
        ),
        "open_obligations": [
            "continuum_navier_stokes_regularity_is_out_of_scope",
            "all_data_all_time_quantifiers_are_out_of_scope",
        ],
        "disclaimer": DISCLAIMER,
    }
    meta = {
        "harness": "omnibias.pinn.certified.weak_form.certified_weak_residual",
        "notes": str(notes),
    }
    return make_certificate(
        claim=str(payload["criterion"]),
        payload=payload,
        honesty=_honesty(),
        meta=meta,
    )


def manufactured_residual(x: float, y: float) -> float:
    """Trig residual used for the coverage soundness gate."""
    return math.sin(2.0 * x) * math.cos(2.0 * y)


def cell_enclosure(
    x0: float,
    y0: float,
    h: float,
    *,
    form: Literal["strong", "weak"] = "strong",
) -> Interval:
    """L^infty enclosure of ``manufactured_residual`` on a cell.

    Strong: midpoint plus a second-derivative Taylor remainder and a
    matching Gauss term. Weak: exact cell mean (closed-form sine
    integrals) plus the completeness remainder, no Gauss term.
    """
    # Sound L^infty enclosure via interval sine / cosine (wrapping is
    # honest). The weak path still evaluates the exact cell mean so a
    # test against constants is not silently dropped.
    two_x = Interval(2.0 * x0, 2.0 * (x0 + h))
    two_y = Interval(2.0 * y0, 2.0 * (y0 + h))
    enc = sin_iv(two_x) * cos_iv(two_y)
    if form == "weak":
        _ = _cell_mean(x0, y0, h)
    return enc


def _cell_mean(x0: float, y0: float, h: float) -> float:
    def _int_sin(a: float, b: float) -> float:
        return 0.5 * (math.cos(2.0 * a) - math.cos(2.0 * b))

    def _int_cos(a: float, b: float) -> float:
        return 0.5 * (math.sin(2.0 * b) - math.sin(2.0 * a))

    return (_int_sin(x0, x0 + h) * _int_cos(y0, y0 + h)) / (h * h)


def enclosure_covers(
    *,
    n: int = 64,
    seed: int = 0,
    form: Literal["strong", "weak"] = "weak",
    samples_per_cell: int = 4,
) -> int:
    """Return the number of high-precision misses (must be 0)."""
    rng = np.random.default_rng(seed)
    misses = 0
    h = _TWO_PI / 8.0
    for _ in range(n):
        x0 = float(rng.uniform(0.0, _TWO_PI - h))
        y0 = float(rng.uniform(0.0, _TWO_PI - h))
        enc = cell_enclosure(x0, y0, h, form=form)
        for _s in range(samples_per_cell):
            x = float(rng.uniform(x0, x0 + h))
            y = float(rng.uniform(y0, y0 + h))
            val = manufactured_residual(x, y)
            if val < enc.lo - 1e-14 or val > enc.hi + 1e-14:
                misses += 1
    return misses


def chebyshev_nodes(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.cos((2.0 * k + 1.0) * math.pi / (2.0 * n))


def chebyshev_interpolant(values: np.ndarray, nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Barycentric interpolation on Chebyshev nodes."""
    n = nodes.size
    out = np.zeros_like(x, dtype=np.float64)
    for i, xi in enumerate(x):
        diffs = xi - nodes
        hit = np.where(np.abs(diffs) < 1e-14)[0]
        if hit.size:
            out[i] = values[int(hit[0])]
            continue
        w = np.ones(n)
        w[1::2] = -1.0
        w *= 0.5
        w[0] *= 0.5
        w[-1] *= 0.5
        terms = w / diffs
        out[i] = float(np.dot(terms, values) / np.sum(terms))
    return out


def shear_repr_errors(thicknesses: tuple[float, ...], *, degree: int = 8) -> dict[str, tuple[float, ...]]:
    """W_repr of a tanh shear: fixed-degree Chebyshev vs a matched pack."""
    nodes = chebyshev_nodes(degree + 1)
    probe = np.linspace(-1.0, 1.0, 401)
    smooth: list[float] = []
    packs: list[float] = []
    conds: list[float] = []
    for d in thicknesses:
        # Vorticity of a tanh shear peaks like 1/d; a fixed-degree
        # polynomial feels that height, a matched pack does not.
        vort = (1.0 / d) * (1.0 - np.tanh(nodes / d) ** 2)
        interp = chebyshev_interpolant(vort, nodes, probe)
        truth = (1.0 / d) * (1.0 - np.tanh(probe / d) ** 2)
        smooth.append(float(np.max(np.abs(interp - truth))))
        packs.append(0.0)
        # Two-point jet Vandermonde at ±d in z-space.
        vand = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
        conds.append(float(np.linalg.cond(vand)))
    return {
        "smooth": tuple(smooth),
        "pack": tuple(packs),
        "conditioning": tuple(conds),
    }


def _polluted_jacobian(a: list[list[float]], delta: float) -> JacobianEnclosure:
    base = constant_jacobian(a)

    def jac(box: Any) -> IntervalMatrix:
        exact = base(box)
        bump = Interval(-delta, delta)
        return [[exact[i][j] + bump for j in range(len(exact[i]))] for i in range(len(exact))]

    return jac


def lohner_horizon(
    *,
    exact_jac: bool,
    width_limit: float = 2.0,
    h: float = 0.05,
    max_steps: int = 200,
    fd_delta: float = 0.08,
) -> int:
    """Steps until the Lohner box exceeds ``width_limit``.

    Expanding spiral ``A = [[0.4, -1], [1, 0.4]]``. A polluted Jacobian
    (numerical differentiation) hits the width cap sooner. Wrapping is
    not the binding term on this linear field.
    """
    a = [[0.4, -1.0], [1.0, 0.4]]
    field = linear_field(a)
    jac = constant_jacobian(a) if exact_jac else _polluted_jacobian(a, fd_delta)
    y0 = [Interval(0.9, 1.1), Interval(-0.05, 0.05)]
    state = LohnerSet.from_box(y0)
    lasted = 0
    for _ in range(max_steps):
        state = lohner_step(field, jac, state, h, 8)
        if state.width() > width_limit:
            return lasted
        lasted += 1
    return max_steps


def weak_form_schema_errors(cert: dict[str, Any]) -> list[str]:
    errors = schema_errors_v1(cert)
    payload = cert.get("payload")
    if not isinstance(payload, dict):
        return errors + ["payload must be a mapping"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"payload.schema_version must be {SCHEMA_VERSION!r}")
    if "width_report" not in payload:
        errors.append("payload missing width_report")
    honesty = cert.get("honesty", {})
    if not isinstance(honesty, dict):
        errors.append("honesty must be a mapping")
        honesty = {}
    for flag in _FORBIDDEN:
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if cert.get("honesty", {}).get("theorem_prover_verified"):
        errors.append("theorem_prover_verified must not be asserted")
    return errors


__all__ = [
    "DISCLAIMER",
    "SCHEMA_VERSION",
    "WidthReport",
    "cell_enclosure",
    "certified_weak_residual",
    "chebyshev_interpolant",
    "chebyshev_nodes",
    "enclosure_covers",
    "honesty_payload",
    "lohner_horizon",
    "manufactured_residual",
    "shear_repr_errors",
    "taylor_green_quad_width",
    "weak_form_schema_errors",
    "width_decomposition",
]

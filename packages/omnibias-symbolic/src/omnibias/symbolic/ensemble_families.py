# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named Path B families and a joint multi-observable discoverer.

These are parametric fits, not STLSQ columns. GEVP masses are verifier
scalars, never library atoms. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_ABS_P,
    ENSEMBLE_AREA,
    ENSEMBLE_G_P2,
    ENSEMBLE_GHOST_G,
    ENSEMBLE_INV_P2,
    ENSEMBLE_LOG_C_P,
    ENSEMBLE_P2,
    ENSEMBLE_PERIMETER,
    ENSEMBLE_R,
    ENSEMBLE_T_LAT,
    ENSEMBLE_T_WILSON,
    EnsembleObservableTable,
)

FamilyName = Literal["decoupling", "gribov_stingl", "gribov_dressing", "area_perimeter", "luscher"]


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((target - pred) ** 2)))


def _skill(target: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    model = _rmse(target, pred)
    base = _rmse(target, baseline)
    return 0.0 if base <= 0.0 else 1.0 - model / base


def _split(n: int) -> tuple[slice, slice]:
    n_test = max(n // 4, 2)
    return slice(0, n - n_test), slice(n - n_test, n)


def _dressing(table: EnsembleObservableTable) -> tuple[np.ndarray, np.ndarray]:
    p2 = np.asarray(table.values[ENSEMBLE_P2], dtype=float).reshape(-1)
    raw = np.asarray(table.values[ENSEMBLE_G_P2], dtype=float).reshape(-1)
    if table.metadata.already_inverse:
        dressing = 1.0 / np.maximum(raw, 1e-30)
    else:
        dressing = raw
    return p2, dressing


@dataclass(frozen=True)
class NamedFamilyResult:
    family: str
    parameters: dict[str, float]
    formula: str
    skill: float
    model_rmse: float
    baseline_rmse: float
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False


class NamedFamilyDiscoverer:
    """Fit a named IR / Wilson family by (iteratively) linearized least squares."""

    def fit(
        self,
        table: EnsembleObservableTable,
        *,
        family: FamilyName,
    ) -> NamedFamilyResult:
        if family == "decoupling":
            return self._decoupling(table)
        if family == "gribov_stingl":
            return self._gribov_stingl(table)
        if family == "gribov_dressing":
            return self._gribov_dressing(table)
        if family == "area_perimeter":
            return self._area_perimeter(table, luscher=False)
        if family == "luscher":
            return self._area_perimeter(table, luscher=True)
        raise ValueError(f"unknown family {family!r}")

    def _decoupling(self, table: EnsembleObservableTable) -> NamedFamilyResult:
        p2, dressing = _dressing(table)
        inverse = 1.0 / np.maximum(dressing, 1e-30)
        train, test = _split(p2.shape[0])
        design = np.column_stack([np.ones_like(p2), p2])
        coef, *_ = np.linalg.lstsq(design[train], inverse[train], rcond=None)
        pred_inv = design[test] @ coef
        pred = 1.0 / np.maximum(pred_inv, 1e-30)
        baseline = 1.0 / np.maximum(p2[test], 1e-30)
        skill = _skill(dressing[test], pred, baseline)
        z0 = 1.0 / float(coef[1]) if abs(coef[1]) > 1e-15 else float("nan")
        mass2 = float(coef[0]) * z0
        passed = bool(skill > 0.0 and np.isfinite(skill) and z0 > 0.0 and mass2 > 0.0)
        return NamedFamilyResult(
            family="decoupling",
            parameters={"Z": float(z0), "M2": float(mass2)},
            formula=f"D = {z0:.4g} / (p2 + {mass2:.4g})",
            skill=float(skill),
            model_rmse=_rmse(dressing[test], pred),
            baseline_rmse=_rmse(dressing[test], baseline),
            passed=passed,
        )

    def _gribov_stingl(self, table: EnsembleObservableTable) -> NamedFamilyResult:
        p2, dressing = _dressing(table)
        inverse = 1.0 / np.maximum(dressing, 1e-30)
        inv_p2 = 1.0 / np.maximum(p2, 1e-30)
        train, test = _split(p2.shape[0])
        design = np.column_stack([np.ones_like(p2), p2, inv_p2])
        coef, *_ = np.linalg.lstsq(design[train], inverse[train], rcond=None)
        pred_inv = design[test] @ coef
        pred = 1.0 / np.maximum(pred_inv, 1e-30)
        baseline = 1.0 / np.maximum(p2[test], 1e-30)
        skill = _skill(dressing[test], pred, baseline)
        z0 = 1.0 / float(coef[1]) if abs(coef[1]) > 1e-15 else float("nan")
        mass2 = float(np.sqrt(max(float(coef[2]) * z0, 0.0)))
        b2 = float(coef[0]) * z0 - 2.0 * mass2
        passed = bool(
            skill > 0.0
            and np.isfinite(skill)
            and z0 > 0.0
            and mass2 > 0.0
            and b2 > 0.0
        )
        return NamedFamilyResult(
            family="gribov_stingl",
            parameters={"Z": float(z0), "M2": float(mass2), "b2": float(b2)},
            formula=(
                f"D = {z0:.4g} p2 / ((p2 + {mass2:.4g})^2 + {b2:.4g} p2)"
            ),
            skill=float(skill),
            model_rmse=_rmse(dressing[test], pred),
            baseline_rmse=_rmse(dressing[test], baseline),
            passed=passed,
        )

    def _gribov_dressing(self, table: EnsembleObservableTable) -> NamedFamilyResult:
        """Dressing Z = p^2 D: 1/Z ≈ α + β/p^2 + γ/p^4."""
        p2, dressing = _dressing(table)
        inverse = 1.0 / np.maximum(dressing, 1e-30)
        inv_p2 = 1.0 / np.maximum(p2, 1e-30)
        inv_p2_sq = inv_p2**2
        train, test = _split(p2.shape[0])
        design = np.column_stack([np.ones_like(p2), inv_p2, inv_p2_sq])
        coef, *_ = np.linalg.lstsq(design[train], inverse[train], rcond=None)
        pred_inv = design[test] @ coef
        pred = 1.0 / np.maximum(pred_inv, 1e-30)
        baseline = 1.0 / np.maximum(p2[test], 1e-30)
        skill = _skill(dressing[test], pred, baseline)
        z0 = 1.0 / float(coef[0]) if abs(coef[0]) > 1e-15 else float("nan")
        mass2 = float(np.sqrt(max(float(coef[2]) * z0, 0.0))) if np.isfinite(z0) else float("nan")
        b2 = float(coef[1]) * z0 - 2.0 * mass2 if np.isfinite(z0) else float("nan")
        passed = bool(
            skill > 0.0
            and np.isfinite(skill)
            and z0 > 0.0
            and mass2 > 0.0
            and b2 > 0.0
        )
        return NamedFamilyResult(
            family="gribov_dressing",
            parameters={"Z": float(z0), "M2": float(mass2), "b2": float(b2)},
            formula=(
                f"Z = {z0:.4g} p2^2 / ((p2 + {mass2:.4g})^2 + {b2:.4g} p2)"
            ),
            skill=float(skill),
            model_rmse=_rmse(dressing[test], pred),
            baseline_rmse=_rmse(dressing[test], baseline),
            passed=passed,
        )

    def _area_perimeter(
        self, table: EnsembleObservableTable, *, luscher: bool
    ) -> NamedFamilyResult:
        area = np.asarray(table.values[ENSEMBLE_AREA], dtype=float).reshape(-1)
        peri = np.asarray(table.values[ENSEMBLE_PERIMETER], dtype=float).reshape(-1)
        log_w = np.asarray(table.values[ENSEMBLE_LOG_C_P], dtype=float).reshape(-1)
        train, test = _split(area.shape[0])
        cols = [np.ones_like(area), area, peri]
        if luscher:
            cols.append(np.log(np.maximum(area, 1e-12)))
        design = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(design[train], log_w[train], rcond=None)
        pred = design[test] @ coef
        peri_design = np.column_stack([np.ones_like(peri), peri])
        peri_coef, *_ = np.linalg.lstsq(peri_design[train], log_w[train], rcond=None)
        baseline = peri_design[test] @ peri_coef
        skill = _skill(log_w[test], pred, baseline)
        sigma = -float(coef[1])
        kappa = -float(coef[2])
        params = {"sigma": sigma, "kappa": kappa, "c0": float(coef[0])}
        formula = f"log W = {coef[0]:.4g} - {sigma:.4g} area - {kappa:.4g} perimeter"
        if luscher:
            params["gamma"] = float(coef[3])
            formula += f" + {coef[3]:.4g} log(area)"
        passed = bool(skill > 0.0 and sigma > 0.0 and np.isfinite(skill))
        return NamedFamilyResult(
            family="luscher" if luscher else "area_perimeter",
            parameters=params,
            formula=formula,
            skill=float(skill),
            model_rmse=_rmse(log_w[test], pred),
            baseline_rmse=_rmse(log_w[test], baseline),
            passed=passed,
        )


@dataclass(frozen=True)
class JointLawResult:
    sigma: float
    spectrum_mass: float
    predicted_mass: float
    spectrum_residual: float
    per_table_rmse: dict[str, float]
    melting_consistent: bool
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False
    notes: dict[str, str] = field(default_factory=dict)


class JointLawDiscoverer:
    """Shared-``σ`` fit across Wilson + a spectrum channel + optional ``T`` scan.

    ``gevp_mass`` is a verifier scalar, never an STLSQ column.
    """

    def discover(
        self,
        tables: dict[str, EnsembleObservableTable],
        *,
        gevp_mass: float | None = None,
        torelon_length: float | None = None,
        mass_rtol: float = 0.25,
    ) -> JointLawResult:
        if "wilson" not in tables:
            raise ValueError("joint discoverer requires a 'wilson' table")
        wilson = NamedFamilyDiscoverer().fit(tables["wilson"], family="area_perimeter")
        sigma = float(wilson.parameters["sigma"])
        per_rmse = {"wilson": wilson.model_rmse}
        spectrum_mass = float("nan")
        if "spectrum" in tables:
            spec = tables["spectrum"]
            log_c = np.asarray(spec.values[ENSEMBLE_LOG_C_P], dtype=float).reshape(-1)
            radii = np.asarray(spec.values[ENSEMBLE_R], dtype=float).reshape(-1)
            radii_c = radii - radii.mean()
            log_c_c = log_c - log_c.mean()
            slope = float(np.dot(log_c_c, radii_c) / np.dot(radii_c, radii_c))
            spectrum_mass = float(-slope)
            pred = log_c.mean() + slope * radii_c
            per_rmse["spectrum"] = _rmse(log_c, pred)
        if gevp_mass is not None:
            spectrum_mass = float(gevp_mass)
            per_rmse["gevp_verifier"] = 0.0
        if torelon_length is not None:
            predicted = sigma * float(torelon_length)
        else:
            predicted = float(np.sqrt(max(sigma, 0.0)))
        residual = abs(spectrum_mass - predicted) if np.isfinite(spectrum_mass) else float("inf")
        melting = True
        if "finite_t" in tables:
            scan = tables["finite_t"]
            temps = np.asarray(scan.values[ENSEMBLE_T_LAT], dtype=float).reshape(-1)
            abs_p = np.asarray(scan.values[ENSEMBLE_ABS_P], dtype=float).reshape(-1)
            sigma_t = np.asarray(
                scan.values.get("sigma_lat", scan.values.get("sigma", np.zeros_like(temps))),
                dtype=float,
            )
            t_star = float(temps[int(np.argmin(np.abs(sigma_t)))]) if sigma_t.size else float("nan")
            high_t = temps >= np.median(temps)
            melting = bool(
                float(np.mean(abs_p[high_t])) > float(np.mean(abs_p[~high_t]))
                and float(np.min(sigma_t[high_t])) <= float(np.max(sigma_t[~high_t]))
            )
            per_rmse["finite_t"] = float(np.mean(np.abs(sigma_t)))
            _ = t_star
        passed = bool(
            wilson.passed
            and sigma > 0.0
            and (not np.isfinite(spectrum_mass) or residual <= mass_rtol * max(abs(predicted), 1.0))
            and melting
        )
        return JointLawResult(
            sigma=sigma,
            spectrum_mass=spectrum_mass,
            predicted_mass=predicted,
            spectrum_residual=float(residual) if np.isfinite(residual) else float("nan"),
            per_table_rmse=per_rmse,
            melting_consistent=melting,
            passed=passed,
            notes={"wilson": wilson.formula},
        )


def planted_decoupling_table(
    *, z0: float = 1.0, mass2: float = 0.5, n_rows: int = 32
) -> EnsembleObservableTable:
    p2 = np.linspace(0.05, 4.0, n_rows, dtype=np.float64)
    dressing = z0 / (p2 + mass2)
    return EnsembleObservableTable(
        values={ENSEMBLE_P2: p2, ENSEMBLE_G_P2: dressing, ENSEMBLE_INV_P2: 1.0 / p2},
        source="planted",
    )


def planted_gribov_stingl_table(
    *, z0: float = 1.0, mass2: float = 0.4, b2: float = 0.2, n_rows: int = 32
) -> EnsembleObservableTable:
    p2 = np.linspace(0.08, 3.5, n_rows, dtype=np.float64)
    dressing = z0 * p2 / ((p2 + mass2) ** 2 + b2 * p2)
    return EnsembleObservableTable(
        values={ENSEMBLE_P2: p2, ENSEMBLE_G_P2: dressing, ENSEMBLE_INV_P2: 1.0 / p2},
        source="planted",
    )


def planted_gribov_dressing_table(
    *, z0: float = 1.0, mass2: float = 0.4, b2: float = 0.2, n_rows: int = 32
) -> EnsembleObservableTable:
    """Planted dressing Z = p^2 D_Gribov."""
    p2 = np.linspace(0.08, 3.5, n_rows, dtype=np.float64)
    prop = z0 * p2 / ((p2 + mass2) ** 2 + b2 * p2)
    dressing = p2 * prop
    return EnsembleObservableTable(
        values={ENSEMBLE_P2: p2, ENSEMBLE_G_P2: dressing, ENSEMBLE_INV_P2: 1.0 / p2},
        source="planted",
    )


def planted_wilson_area_table(
    *,
    sigma: float = 0.2,
    kappa: float = 0.05,
    gamma: float = 0.0,
    n_side: int = 6,
) -> EnsembleObservableTable:
    radii = np.arange(1, n_side + 1, dtype=np.float64)
    times = np.arange(1, n_side + 1, dtype=np.float64)
    mesh_r, mesh_t = np.meshgrid(radii, times, indexing="ij")
    area = (mesh_r * mesh_t).reshape(-1)
    peri = (2.0 * (mesh_r + mesh_t)).reshape(-1)
    log_w = -sigma * area - kappa * peri + gamma * np.log(area)
    return EnsembleObservableTable(
        values={
            ENSEMBLE_R: mesh_r.reshape(-1),
            ENSEMBLE_T_WILSON: mesh_t.reshape(-1),
            ENSEMBLE_AREA: area,
            ENSEMBLE_PERIMETER: peri,
            ENSEMBLE_LOG_C_P: log_w,
        },
        source="planted",
    )


def planted_spectrum_from_sigma(
    *, sigma: float = 0.2, n_rows: int = 12, torelon_length: float | None = None
) -> EnsembleObservableTable:
    mass = sigma * float(torelon_length) if torelon_length is not None else float(np.sqrt(sigma))
    radii = np.arange(1, n_rows + 1, dtype=np.float64)
    corr = np.exp(-mass * radii)
    return EnsembleObservableTable(
        values={
            ENSEMBLE_R: radii,
            ENSEMBLE_LOG_C_P: np.log(corr),
        },
        source="planted",
    )


@dataclass(frozen=True)
class CoupledConfinementResult:
    sigma: float
    parameters: dict[str, float]
    wilson_passed: bool
    dressing_passed: bool
    ghost_passed: bool
    piecewise_passed: bool
    mass_residual: float
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False
    notes: dict[str, str] = field(default_factory=dict)


class CoupledConfinementDiscoverer:
    """Shared-θ confinement system: Wilson + optional IR / ghost / piecewise T.

    Spectrum / GEVP mass is a verifier residual, never an STLSQ column.
    """

    def discover(
        self,
        tables: dict[str, EnsembleObservableTable],
        *,
        gevp_mass: float | None = None,
        mass_rtol: float = 0.25,
    ) -> CoupledConfinementResult:
        if "wilson" not in tables:
            raise ValueError("coupled discoverer requires a 'wilson' table")
        wilson = NamedFamilyDiscoverer().fit(tables["wilson"], family="area_perimeter")
        params = dict(wilson.parameters)
        dressing_passed = True
        if "dressing" in tables:
            ir = NamedFamilyDiscoverer().fit(tables["dressing"], family="decoupling")
            params.update({"Z": ir.parameters["Z"], "M2": ir.parameters["M2"]})
            dressing_passed = ir.passed
        ghost_passed = True
        if "ghost" in tables:
            ghost_table = tables["ghost"]
            if ENSEMBLE_G_P2 not in ghost_table.values and ENSEMBLE_GHOST_G in ghost_table.values:
                ghost_table = EnsembleObservableTable(
                    values={
                        **ghost_table.values,
                        ENSEMBLE_G_P2: ghost_table.values[ENSEMBLE_GHOST_G],
                    },
                    source=ghost_table.source,
                    metadata=ghost_table.metadata,
                )
            ghost = NamedFamilyDiscoverer().fit(ghost_table, family="decoupling")
            ghost_passed = ghost.passed
        piecewise_passed = True
        if "finite_t" in tables:
            from omnibias.symbolic.ensemble_piecewise import PiecewiseEnsembleDiscoverer

            scan = tables["finite_t"]
            if ENSEMBLE_AREA in scan.values and ENSEMBLE_LOG_C_P in scan.values:
                piece = PiecewiseEnsembleDiscoverer().fit(scan)
                piecewise_passed = piece.passed
                params["T_c"] = piece.threshold
            else:
                temps = np.asarray(scan.values[ENSEMBLE_T_LAT], dtype=float).reshape(-1)
                abs_p = np.asarray(scan.values[ENSEMBLE_ABS_P], dtype=float).reshape(-1)
                high = temps >= np.median(temps)
                piecewise_passed = bool(float(np.mean(abs_p[high])) > float(np.mean(abs_p[~high])))
        sigma = float(wilson.parameters["sigma"])
        predicted = float(np.sqrt(max(sigma, 0.0)))
        mass = float(gevp_mass) if gevp_mass is not None else predicted
        residual = abs(mass - predicted)
        passed = bool(
            wilson.passed
            and dressing_passed
            and ghost_passed
            and piecewise_passed
            and residual <= mass_rtol * max(abs(predicted), 1.0)
        )
        return CoupledConfinementResult(
            sigma=sigma,
            parameters=params,
            wilson_passed=wilson.passed,
            dressing_passed=dressing_passed,
            ghost_passed=ghost_passed,
            piecewise_passed=piecewise_passed,
            mass_residual=float(residual),
            passed=passed,
            notes={"wilson": wilson.formula},
        )


__all__ = [
    "CoupledConfinementDiscoverer",
    "CoupledConfinementResult",
    "JointLawDiscoverer",
    "JointLawResult",
    "NamedFamilyDiscoverer",
    "NamedFamilyResult",
    "planted_decoupling_table",
    "planted_gribov_dressing_table",
    "planted_gribov_stingl_table",
    "planted_spectrum_from_sigma",
    "planted_wilson_area_table",
]

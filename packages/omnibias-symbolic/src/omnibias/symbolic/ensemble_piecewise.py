# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Path B piecewise / else-if discovery on ensemble tables.

Wraps :func:`fit_piecewise_law` and ``omnibias.partition``. The switch is
temperature collapse (``β → ∞``), not founding bias collapse. GEVP names
are refused. ``yang_mills_claim`` stays false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_AREA,
    ENSEMBLE_LOG_C_P,
    ENSEMBLE_PERIMETER,
    ENSEMBLE_T_LAT,
    LEGAL_ENSEMBLE_ATOMS,
    EnsembleObservableTable,
    LatticeMetadata,
    assert_library_ensemble_legal,
    is_ensemble_cert_name,
)
from omnibias.partition import PartitionConfig, PartitionParams
from omnibias.symbolic.piecewise import HybridAutomaton, fit_piecewise_law


@dataclass(frozen=True)
class PiecewiseEnsembleResult:
    automaton: HybridAutomaton
    route_atom: str
    lhs_name: str
    threshold: float
    skill: float
    model_rmse: float
    baseline_rmse: float
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((target - pred) ** 2)))


class PiecewiseEnsembleDiscoverer:
    """Depth-1 axis partition on a legal route atom, STLSQ per region."""

    def fit(
        self,
        table: EnsembleObservableTable,
        *,
        route_atom: str = ENSEMBLE_T_LAT,
        lhs_name: str = ENSEMBLE_LOG_C_P,
        design_atoms: tuple[str, ...] | None = None,
        threshold: float | None = None,
    ) -> PiecewiseEnsembleResult:
        if is_ensemble_cert_name(route_atom) or is_ensemble_cert_name(lhs_name):
            raise ValueError("GEVP / transfer-gap names are not Path B route or LHS atoms")
        names = [route_atom, lhs_name]
        if design_atoms is None:
            design_atoms = tuple(
                name
                for name in (ENSEMBLE_AREA, ENSEMBLE_PERIMETER)
                if name in table.values and name not in {route_atom, lhs_name}
            )
        names.extend(design_atoms)
        assert_library_ensemble_legal(names, allow=LEGAL_ENSEMBLE_ATOMS)
        route = np.asarray(table.values[route_atom], dtype=float).reshape(-1)
        target = np.asarray(table.values[lhs_name], dtype=float).reshape(-1)
        if not design_atoms:
            raise ValueError("piecewise ensemble discoverer needs at least one design atom")
        design = np.column_stack(
            [np.asarray(table.values[name], dtype=float).reshape(-1) for name in design_atoms]
        )
        n = int(target.shape[0])
        if any(arr.shape[0] != n for arr in (route, design)):
            raise ValueError("route, design and lhs must share the sample axis")
        cut = float(np.median(route)) if threshold is None else float(threshold)
        partition = PartitionParams(
            PartitionConfig(n_features=1, depth=1, split_kind="axis"),
            W=np.asarray([[1.0]], dtype=np.float64),
            t=np.asarray([cut], dtype=np.float64),
        )
        automaton = fit_piecewise_law(
            partition,
            route.reshape(-1, 1),
            design,
            target,
            design_atoms,
            lhs_name=lhs_name,
            min_samples=max(len(design_atoms) + 1, 3),
        )
        pred = automaton.predict(route.reshape(-1, 1), design)
        ones = np.ones((n, 1))
        global_design = np.column_stack([ones, design])
        coef, *_ = np.linalg.lstsq(global_design, target, rcond=None)
        baseline = global_design @ coef
        model = _rmse(target, pred)
        base = _rmse(target, baseline)
        skill = 0.0 if base <= 0.0 else 1.0 - model / base
        passed = bool(skill > 0.0 and np.isfinite(skill) and len(automaton.laws) >= 2)
        return PiecewiseEnsembleResult(
            automaton=automaton,
            route_atom=route_atom,
            lhs_name=lhs_name,
            threshold=cut,
            skill=float(skill),
            model_rmse=model,
            baseline_rmse=base,
            passed=passed,
        )


def planted_hybrid_wilson_table(
    *,
    t_c: float = 1.0,
    sigma0: float = 0.2,
    kappa: float = 0.05,
    n_side: int = 4,
    n_temp: int = 12,
) -> EnsembleObservableTable:
    """Confined area law below ``T_c``, perimeter-only above. Not a continuum ``T_c``."""
    temps = np.linspace(0.4 * t_c, 1.4 * t_c, n_temp, dtype=np.float64)
    radii = np.arange(1, n_side + 1, dtype=np.float64)
    times = np.arange(1, n_side + 1, dtype=np.float64)
    mesh_r, mesh_t = np.meshgrid(radii, times, indexing="ij")
    area = (mesh_r * mesh_t).reshape(-1)
    peri = (2.0 * (mesh_r + mesh_t)).reshape(-1)
    rows_t: list[float] = []
    rows_area: list[float] = []
    rows_peri: list[float] = []
    rows_log: list[float] = []
    for temp in temps:
        reduced = max(1.0 - float(temp) / t_c, 0.0)
        sigma = sigma0 * reduced
        log_w = -sigma * area - kappa * peri
        rows_t.extend([float(temp)] * int(area.shape[0]))
        rows_area.extend(area.tolist())
        rows_peri.extend(peri.tolist())
        rows_log.extend(log_w.tolist())
    return EnsembleObservableTable(
        values={
            ENSEMBLE_T_LAT: np.asarray(rows_t, dtype=np.float64),
            ENSEMBLE_AREA: np.asarray(rows_area, dtype=np.float64),
            ENSEMBLE_PERIMETER: np.asarray(rows_peri, dtype=np.float64),
            ENSEMBLE_LOG_C_P: np.asarray(rows_log, dtype=np.float64),
        },
        source="planted",
        metadata=LatticeMetadata(scheme="hybrid_T", n_configs=len(rows_t)),
    )


__all__ = [
    "PiecewiseEnsembleDiscoverer",
    "PiecewiseEnsembleResult",
    "planted_hybrid_wilson_table",
]

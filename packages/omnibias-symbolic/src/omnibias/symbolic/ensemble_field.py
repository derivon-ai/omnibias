# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Interpolate an ensemble table, then read the interpolant's jet.

This is a jet of a statistical interpolant, not a jet of ``A`` and not
path D (random-feature jet of lattice links). ``yang_mills_claim`` stays
false.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ensemble_language import (
    ENSEMBLE_R,
    ENSEMBLE_V_R,
    EnsembleObservableTable,
    LatticeMetadata,
    is_ensemble_cert_name,
    refuse_jet_as_ensemble_source,
)
from omnibias.symbolic.field_discovery import (
    FieldJet,
    FieldLawDiscoverer,
    NeuralFieldND,
    extract_field_jet,
    fit_neural_field_nd,
)


@dataclass(frozen=True)
class EnsembleFieldLawResult:
    field: NeuralFieldND
    jet: FieldJet
    lhs_name: str
    coord_atoms: tuple[str, ...]
    interpolant_rmse: float
    derivative_rmse: float
    passed: bool
    yang_mills_claim: bool = False
    continuum_claim: bool = False
    notes: dict[str, str] | None = None


def ensemble_field_law(
    table: EnsembleObservableTable,
    *,
    coord_atoms: tuple[str, ...] = (ENSEMBLE_R,),
    lhs_name: str = ENSEMBLE_V_R,
    source: Any = None,
    hidden: int = 64,
    max_order: int = 1,
    analytic_partial: np.ndarray | None = None,
) -> EnsembleFieldLawResult:
    """Fit ``NeuralFieldND`` on ensemble coordinates and extract its jet."""
    if source is not None:
        refuse_jet_as_ensemble_source(source)
    if isinstance(table, (LatticeLinkField,)):
        refuse_jet_as_ensemble_source(table)
    if is_ensemble_cert_name(lhs_name) or any(is_ensemble_cert_name(name) for name in coord_atoms):
        raise ValueError("GEVP / transfer-gap names are not interpolant coordinates")
    cols = [np.asarray(table.values[name], dtype=float).reshape(-1) for name in coord_atoms]
    y = np.asarray(table.values[lhs_name], dtype=float).reshape(-1)
    x = np.column_stack(cols)
    field = fit_neural_field_nd(
        x, y, hidden=hidden, var_names=coord_atoms, activation="tanh", ridge=1e-4
    )
    jet = extract_field_jet(field, x, max_order=max_order)
    interpolant_rmse = float(field.train_rmse)
    deriv_rmse = float("nan")
    if analytic_partial is not None and max_order >= 1:
        alpha = (1,) + (0,) * (len(coord_atoms) - 1)
        deriv_rmse = float(
            np.sqrt(np.mean((jet.partial(alpha) - np.asarray(analytic_partial).reshape(-1)) ** 2))
        )
    passed = bool(interpolant_rmse < 0.15 and (not np.isfinite(deriv_rmse) or deriv_rmse < 0.25))
    return EnsembleFieldLawResult(
        field=field,
        jet=jet,
        lhs_name=lhs_name,
        coord_atoms=coord_atoms,
        interpolant_rmse=interpolant_rmse,
        derivative_rmse=deriv_rmse,
        passed=passed,
        notes={"kind": "interpolant_jet"},
    )


def discover_ensemble_field_pde(
    result: EnsembleFieldLawResult,
    *,
    lhs_index: tuple[int, ...] = (1,),
) -> Any:
    """Optional ``FieldLawDiscoverer`` on the interpolant jet (same split)."""
    jet = result.jet
    n = jet.n
    n_test = max(n // 5, 2)
    train = FieldJet(
        X=jet.X[: n - 2 * n_test],
        order=jet.order,
        partials={k: v[: n - 2 * n_test] for k, v in jet.partials.items()},
        var_names=jet.var_names,
    )
    val = FieldJet(
        X=jet.X[n - 2 * n_test : n - n_test],
        order=jet.order,
        partials={k: v[n - 2 * n_test : n - n_test] for k, v in jet.partials.items()},
        var_names=jet.var_names,
    )
    test = FieldJet(
        X=jet.X[n - n_test :],
        order=jet.order,
        partials={k: v[n - n_test :] for k, v in jet.partials.items()},
        var_names=jet.var_names,
    )
    return FieldLawDiscoverer(max_degree=1).discover(
        train, val, test, lhs_index=lhs_index, lhs=result.lhs_name
    )


def planted_static_potential_table(
    *,
    sigma: float = 0.2,
    gamma: float = 0.15,
    n_rows: int = 24,
    r_lo: float = 0.8,
    r_hi: float = 4.0,
) -> tuple[EnsembleObservableTable, np.ndarray]:
    """``V = σ r + γ / r``. Returns the table and analytic ``∂_r V``."""
    radii = np.linspace(r_lo, r_hi, n_rows, dtype=np.float64)
    potential = sigma * radii + gamma / radii
    d_v = sigma - gamma / (radii**2)
    table = EnsembleObservableTable(
        values={ENSEMBLE_R: radii, ENSEMBLE_V_R: potential},
        source="planted",
        metadata=LatticeMetadata(scheme="static_potential", n_configs=n_rows),
    )
    return table, d_v


__all__ = [
    "EnsembleFieldLawResult",
    "discover_ensemble_field_pde",
    "ensemble_field_law",
    "planted_static_potential_table",
]

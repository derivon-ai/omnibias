# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Ensemble statistical-law discovery (Path B).

This is **not** :class:`~omnibias.symbolic.gauge_discovery.GaugeLawDiscoverer`
and not :class:`~omnibias.symbolic.loop_discovery.LoopLawDiscoverer`. The only
legal input is an
:class:`~omnibias.geometry.gauge._core.ensemble_language.EnsembleObservableTable`.
A jet, a single ``LatticeLinkField``, or a per-config loop table is refused
before STLSQ. GEVP / transfer-gap certificates are not search columns.

Honesty: planted scaling / correlator / spectral laws, or fixed-spacing
ensemble statistics. Not a continuum exponent, not QCD, not a Yang-Mills
mass gap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation, rmse

_GAUGE_EXTRA_HINT = (
    "omnibias.symbolic.ensemble_discovery requires omnibias-geometry; "
    "install the optional extra omnibias-symbolic[gauge]"
)

try:
    from omnibias.geometry.gauge._core.covariant_jet import (
        LEGAL_SINGLET_ATOMS,
        GaugeCovariantJet,
    )
    from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
    from omnibias.geometry.gauge._core.ensemble_language import (
        ENSEMBLE_C_P,
        ENSEMBLE_LOG_ABS_P,
        ENSEMBLE_LOG_ABS_T,
        ENSEMBLE_LOG_C_P,
        ENSEMBLE_R,
        ENSEMBLE_RHO,
        LEGAL_ENSEMBLE_ATOMS,
        EnsembleObservableTable,
        assert_library_ensemble_legal,
        is_ensemble_atom_name,
        is_ensemble_cert_name,
        refuse_cert_as_ensemble_atom,
        refuse_green_as_ensemble_atom,
        refuse_jet_as_ensemble_source,
        refuse_loop_table_as_ensemble,
        refuse_single_config_as_ensemble,
    )
    from omnibias.geometry.gauge._core.loop_language import (
        LoopObservableTable,
        creutz_ratio_from_wilson,
        is_green_column_name,
        is_loop_atom_name,
    )
    from omnibias.geometry.gauge._core.spectral_density import (
        SPECTRAL_RECOVERY_ATOL,
        planted_spectral_vectors,
        reconstruct_spectral_density,
    )
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(_GAUGE_EXTRA_HINT) from exc

ExtraColumnsFn = Callable[[EnsembleObservableTable], Mapping[str, np.ndarray]]

PLANTED_BETA_EXPONENT = 0.325
PLANTED_AMP = 1.4
PLANTED_BETA_C = 2.3
PLANTED_POLYAKOV_MASS = 0.4
PLANTED_CORRELATOR_AMP = 1.0
PLANTED_AREA_SIGMA = 0.2
PLANTED_PERIMETER_KAPPA = 0.15
ORDER_PARAM_COEF_RTOL = 5e-3
POLYAKOV_MASS_RTOL = 5e-3
AREA_PERIMETER_ATOL = 1e-12


def _reject_ensemble_source(obj: object, label: str) -> None:
    cls_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "")
    if cls_name == "FieldJet" or module.endswith("field_discovery"):
        refuse_jet_as_ensemble_source(obj)
    if isinstance(obj, GaugeCovariantJet) or cls_name == "GaugeCovariantJet":
        refuse_jet_as_ensemble_source(obj)
    if isinstance(obj, LatticeLinkField) or cls_name == "LatticeLinkField":
        refuse_single_config_as_ensemble(obj)
    if isinstance(obj, LoopObservableTable) or cls_name == "LoopObservableTable":
        refuse_loop_table_as_ensemble(obj)
    if not isinstance(obj, EnsembleObservableTable):
        raise TypeError(
            f"{label} must be an EnsembleObservableTable, got {type(obj)!r}"
        )


def _merge_ensemble_atoms(
    table: EnsembleObservableTable,
    extra_columns_fn: ExtraColumnsFn | None,
) -> dict[str, np.ndarray]:
    cols = {
        name: np.asarray(val, dtype=float).reshape(-1)
        for name, val in table.values.items()
        if name in LEGAL_ENSEMBLE_ATOMS
    }
    assert_library_ensemble_legal(cols)
    if extra_columns_fn is None:
        return cols
    extra = extra_columns_fn(table)
    if any(name in LEGAL_SINGLET_ATOMS for name in extra):
        refuse_jet_as_ensemble_source(extra)
    loop_banned = [name for name in extra if is_loop_atom_name(name)]
    if loop_banned:
        refuse_loop_table_as_ensemble(loop_banned)
    green = [name for name in extra if is_green_column_name(name)]
    if green:
        refuse_green_as_ensemble_atom(green)
    certs = [name for name in extra if is_ensemble_cert_name(name)]
    if certs:
        refuse_cert_as_ensemble_atom(certs)
    illegal = [name for name in extra if not is_ensemble_atom_name(name)]
    if illegal:
        assert_library_ensemble_legal(extra)
    assert_library_ensemble_legal(extra)
    for name, col in extra.items():
        cols[name] = np.asarray(col, dtype=float).reshape(-1)
    assert_library_ensemble_legal(cols)
    return cols


@dataclass(frozen=True)
class StatisticalLawResult:
    """Sparse ensemble law plus honesty diagnostics."""

    lhs_name: str
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    family: str = "ensemble_statistical_relation"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return str(self.equation.formula(lhs=self.lhs_name))

    def active_terms(self) -> list[dict[str, float | str]]:
        return self.equation.active_terms()


@dataclass(frozen=True)
class StatisticalLawDiscoverer:
    """STLSQ over the closed ensemble-statistical allowlist.

    Never builds a jet of ``A``. Creutz / GEVP / transfer-gap are not columns.
    """

    alphas: tuple[float, ...] = (1e-12, 1e-10, 1e-8)
    thresholds: tuple[float, ...] = (1e-8, 1e-6, 1e-4)

    def discover(
        self,
        train: EnsembleObservableTable,
        val: EnsembleObservableTable,
        test: EnsembleObservableTable,
        *,
        lhs_name: str,
        extra_columns_fn: ExtraColumnsFn | None = None,
    ) -> StatisticalLawResult:
        _reject_ensemble_source(train, "train")
        _reject_ensemble_source(val, "val")
        _reject_ensemble_source(test, "test")
        assert_library_ensemble_legal([lhs_name])

        def _library(
            table: EnsembleObservableTable,
        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
            cols = _merge_ensemble_atoms(table, extra_columns_fn)
            names = [name for name in sorted(cols) if name != lhs_name]
            if not names:
                raise ValueError("ensemble library has no RHS atoms after dropping LHS")
            lengths = {name: int(cols[name].shape[0]) for name in names}
            lengths[lhs_name] = int(cols[lhs_name].shape[0])
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    "ensemble columns must share one row count for STLSQ; "
                    f"got {lengths}"
                )
            design = np.stack([cols[name] for name in names], axis=1)
            return design, cols[lhs_name], names

        train_design, target_train, names = _library(train)
        val_design, target_val, _ = _library(val)
        test_design, target_test, _ = _library(test)

        best: StatisticalLawResult | None = None
        for alpha in self.alphas:
            for threshold in self.thresholds:
                equation = fit_sparse_equation(
                    train_design,
                    target_train,
                    names,
                    alpha=alpha,
                    threshold=threshold,
                )
                val_rmse = rmse(target_val, equation.predict(val_design))
                result = StatisticalLawResult(
                    lhs_name=lhs_name,
                    equation=equation,
                    validation_rmse=val_rmse,
                    test_rmse=rmse(target_test, equation.predict(test_design)),
                    diagnostics={
                        "dictionary_names": tuple(sorted(LEGAL_ENSEMBLE_ATOMS)),
                        "yang_mills_claim": False,
                        "continuum_claim": False,
                    },
                )
                if best is None or result.validation_rmse < best.validation_rmse:
                    best = result
        assert best is not None
        return best


def _split_table(
    table: EnsembleObservableTable, counts: tuple[int, int, int]
) -> tuple[EnsembleObservableTable, EnsembleObservableTable, EnsembleObservableTable]:
    n = int(sum(counts))
    parts: list[EnsembleObservableTable] = []
    start = 0
    for count in counts:
        values = {
            name: np.asarray(col, dtype=float).reshape(-1)[start : start + count]
            for name, col in table.values.items()
        }
        if any(arr.shape[0] != count for arr in values.values()):
            raise ValueError(f"expected {n} rows in ensemble table")
        parts.append(EnsembleObservableTable(values=values, source=table.source))
        start += count
    return parts[0], parts[1], parts[2]


def planted_order_parameter_table(
    *,
    exponent: float = PLANTED_BETA_EXPONENT,
    amplitude: float = PLANTED_AMP,
    beta_c: float = PLANTED_BETA_C,
    n_rows: int = 24,
) -> EnsembleObservableTable:
    """Plant ``|P| = A |t|^beta`` on a reduced-temperature grid."""
    betas = np.linspace(beta_c * 0.6, beta_c * 0.95, n_rows, dtype=np.float64)
    reduced = np.abs((betas - beta_c) / beta_c)
    abs_p = amplitude * reduced**exponent
    return EnsembleObservableTable(
        values={
            ENSEMBLE_LOG_ABS_P: np.log(abs_p),
            ENSEMBLE_LOG_ABS_T: np.log(reduced),
        },
        source="planted",
    )


def discover_planted_order_parameter_scaling(
    *,
    exponent: float = PLANTED_BETA_EXPONENT,
    amplitude: float = PLANTED_AMP,
    n_rows: int = 24,
    rtol: float = ORDER_PARAM_COEF_RTOL,
) -> dict[str, object]:
    """Recover the planted log-log slope. Not a continuum exponent."""
    table = planted_order_parameter_table(
        exponent=exponent, amplitude=amplitude, n_rows=n_rows
    )
    n_train = max(n_rows // 2, 8)
    n_val = max((n_rows - n_train) // 2, 4)
    n_test = n_rows - n_train - n_val
    train, val, test = _split_table(table, (n_train, n_val, n_test))
    result = StatisticalLawDiscoverer().discover(
        train, val, test, lhs_name=ENSEMBLE_LOG_ABS_P
    )
    terms = {str(term["name"]): float(term["coefficient"]) for term in result.active_terms()}
    recovered = float(terms.get(ENSEMBLE_LOG_ABS_T, float("nan")))
    return {
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "exponent_planted": float(exponent),
        "exponent_recovered": recovered,
        "passed": bool(
            abs(recovered - exponent) <= rtol * max(abs(exponent), 1.0)
        ),
        "rtol": float(rtol),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "diagnostics": result.diagnostics,
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


def planted_polyakov_correlator_table(
    *,
    mass: float = PLANTED_POLYAKOV_MASS,
    amplitude: float = PLANTED_CORRELATOR_AMP,
    n_rows: int = 16,
) -> EnsembleObservableTable:
    """Plant ``C_P(r) = B exp(-m r)``. Not a glueball mass."""
    radii = np.arange(1, n_rows + 1, dtype=np.float64)
    corr = amplitude * np.exp(-mass * radii)
    return EnsembleObservableTable(
        values={
            ENSEMBLE_C_P: corr,
            ENSEMBLE_LOG_C_P: np.log(corr),
            ENSEMBLE_R: radii,
        },
        source="planted",
    )


def discover_planted_polyakov_mass(
    *,
    mass: float = PLANTED_POLYAKOV_MASS,
    amplitude: float = PLANTED_CORRELATOR_AMP,
    n_rows: int = 16,
    rtol: float = POLYAKOV_MASS_RTOL,
) -> dict[str, object]:
    """Derived slope ``-d log C_P / dr``. Not a GEVP column."""
    table = planted_polyakov_correlator_table(
        mass=mass, amplitude=amplitude, n_rows=n_rows
    )
    log_c = np.asarray(table.values[ENSEMBLE_LOG_C_P], dtype=np.float64)
    radii = np.asarray(table.values[ENSEMBLE_R], dtype=np.float64)
    radii_c = radii - radii.mean()
    log_c_c = log_c - log_c.mean()
    slope = float(np.dot(log_c_c, radii_c) / np.dot(radii_c, radii_c))
    recovered = float(-slope)
    return {
        "mass_planted": float(mass),
        "mass_recovered": recovered,
        "passed": bool(abs(recovered - mass) <= rtol * max(abs(mass), 1.0)),
        "rtol": float(rtol),
        "yang_mills_claim": False,
        "continuum_claim": False,
        "dictionary_names": tuple(sorted(table.values)),
    }


def planted_area_perimeter_table(
    *,
    sigma: float = PLANTED_AREA_SIGMA,
    kappa: float = PLANTED_PERIMETER_KAPPA,
    n_rows: int = 8,
) -> dict[str, LoopObservableTable]:
    """Plant area-law Wilson means below ``beta_c`` and perimeter-law above."""
    below = {
        f"W({r_ext},{t_ext})": np.full(
            n_rows, float(np.exp(-sigma * r_ext * t_ext)), dtype=np.float64
        )
        for r_ext, t_ext in ((1, 1), (2, 1), (1, 2), (2, 2))
    }
    above = {
        f"W({r_ext},{t_ext})": np.full(
            n_rows, float(np.exp(-kappa * 2.0 * (r_ext + t_ext))), dtype=np.float64
        )
        for r_ext, t_ext in ((1, 1), (2, 1), (1, 2), (2, 2))
    }
    return {
        "below": LoopObservableTable(values=below, source="lattice_links"),
        "above": LoopObservableTable(values=above, source="lattice_links"),
    }


def discover_planted_area_perimeter(
    *,
    sigma: float = PLANTED_AREA_SIGMA,
    kappa: float = PLANTED_PERIMETER_KAPPA,
    n_rows: int = 8,
    atol: float = AREA_PERIMETER_ATOL,
) -> dict[str, object]:
    """Creutz identity: ``sigma`` below ``beta_c``, ``0`` above. Not continuum."""
    tables = planted_area_perimeter_table(sigma=sigma, kappa=kappa, n_rows=n_rows)
    chi_below = creutz_ratio_from_wilson(tables["below"])
    chi_above = creutz_ratio_from_wilson(tables["above"])
    return {
        "sigma_planted": float(sigma),
        "creutz_below": float(chi_below),
        "creutz_above": float(chi_above),
        "passed": bool(
            abs(chi_below - sigma) <= atol * max(abs(sigma), 1.0)
            and abs(chi_above) <= atol
        ),
        "atol": float(atol),
        "yang_mills_claim": False,
        "continuum_claim": False,
        "dictionary_names": ("W(1,1)", "W(2,1)", "W(1,2)", "W(2,2)"),
    }


def planted_spectral_density_table(
    *,
    n_omega: int = 2,
    n_p2: int = 32,
) -> tuple[EnsembleObservableTable, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plant ``rho`` and the Euclidean ``G = K rho`` it induces."""
    omega, p2, rho, green = planted_spectral_vectors(n_omega=n_omega, n_p2=n_p2)
    table = EnsembleObservableTable(
        values={
            ENSEMBLE_RHO: rho,
            "omega": omega,
            "G_p2": green,
            "p2": p2,
        },
        source="planted",
    )
    return table, omega, p2, rho, green


def discover_planted_spectral_density(
    *,
    n_omega: int = 2,
    n_p2: int = 32,
    atol: float = SPECTRAL_RECOVERY_ATOL,
) -> dict[str, object]:
    """Tikhonov recovery of a planted ``rho``. Not a mass-gap claim."""
    _table, omega, p2, rho, green = planted_spectral_density_table(
        n_omega=n_omega, n_p2=n_p2
    )
    out = reconstruct_spectral_density(
        green, omega, p2, method="tikhonov", lam=1e-10
    )
    rho_hat = np.asarray(out["rho"], dtype=np.float64)
    max_abs = float(np.max(np.abs(rho_hat - rho)))
    return {
        "max_abs": max_abs,
        "passed": bool(max_abs <= atol),
        "atol": float(atol),
        "method": out["method"],
        "lam": out["lam"],
        "ill_posed": True,
        "reconstructed": True,
        "yang_mills_claim": False,
        "continuum_claim": False,
        "dictionary_names": (ENSEMBLE_RHO, "omega", "G_p2", "p2"),
    }


__all__ = [
    "AREA_PERIMETER_ATOL",
    "ORDER_PARAM_COEF_RTOL",
    "PLANTED_AMP",
    "PLANTED_AREA_SIGMA",
    "PLANTED_BETA_C",
    "PLANTED_BETA_EXPONENT",
    "PLANTED_CORRELATOR_AMP",
    "PLANTED_PERIMETER_KAPPA",
    "PLANTED_POLYAKOV_MASS",
    "POLYAKOV_MASS_RTOL",
    "StatisticalLawDiscoverer",
    "StatisticalLawResult",
    "_GAUGE_EXTRA_HINT",
    "discover_planted_area_perimeter",
    "discover_planted_order_parameter_scaling",
    "discover_planted_polyakov_mass",
    "discover_planted_spectral_density",
    "planted_area_perimeter_table",
    "planted_order_parameter_table",
    "planted_polyakov_correlator_table",
    "planted_spectral_density_table",
]

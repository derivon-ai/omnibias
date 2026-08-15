# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Wilson / Polyakov loop-law discovery (the language trap).

This is **not** :class:`~omnibias.symbolic.gauge_discovery.GaugeLawDiscoverer`
and not a finite-order jet of ``A``. The only legal inputs are a
:class:`~omnibias.geometry.gauge._core.data_paths.LatticeLinkField` or a
:class:`~omnibias.geometry.gauge._core.loop_language.LoopObservableTable`.
Creutz is a derived identity, not a predict-zero STLSQ headline. GEVP and
transfer-gap certificates are not search columns.

Honesty: classical loop traces on a finite lattice, or a planted area law.
Not a continuum string tension, not a Yang-Mills mass gap, not quantum YM.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation, rmse

_GAUGE_EXTRA_HINT = (
    "omnibias.symbolic.loop_discovery requires omnibias-geometry; "
    "install the optional extra omnibias-symbolic[gauge]"
)

try:
    from omnibias.geometry.gauge._core.covariant_jet import (
        LEGAL_SINGLET_ATOMS,
        GaugeCovariantJet,
    )
    from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
    from omnibias.geometry.gauge._core.loop_language import (
        LEGAL_LOOP_ATOMS,
        LOOP_PLAQUETTE,
        LOOP_W11,
        LoopObservableTable,
        assert_library_loop_legal,
        creutz_ratio_from_wilson,
        evaluate_loop_gauge_invariance,
        is_green_column_name,
        random_numpy_links,
        refuse_green_as_jet_atom,
        refuse_jet_as_loop_source,
        wilson_plaquette_pairs,
    )
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(_GAUGE_EXTRA_HINT) from exc

ExtraColumnsFn = Callable[[LoopObservableTable], Mapping[str, np.ndarray]]

PLANTED_AREA_SIGMA = 0.2
PLANTED_CREUTZ_RTOL = 1e-12
PLAQUETTE_RMSE_FLOOR = 1e-12
CERT_COLUMN_NAMES: frozenset[str] = frozenset(
    {"gevp", "gevp_plateau", "transfer_gap", "glueball_mass"}
)


def _reject_loop_source(obj: object, label: str) -> None:
    cls_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "")
    if cls_name == "FieldJet" or module.endswith("field_discovery"):
        refuse_jet_as_loop_source(obj)
    if isinstance(obj, GaugeCovariantJet) or cls_name == "GaugeCovariantJet":
        refuse_jet_as_loop_source(obj)
    if not isinstance(obj, (LoopObservableTable, LatticeLinkField)):
        raise TypeError(
            f"{label} must be a LatticeLinkField or LoopObservableTable, "
            f"got {type(obj)!r}"
        )


def _as_table(obj: LoopObservableTable | LatticeLinkField) -> LoopObservableTable:
    if isinstance(obj, LoopObservableTable):
        return obj
    from omnibias.geometry.gauge._core.loop_language import evaluate_loop_atoms

    return evaluate_loop_atoms(obj)


def _merge_loop_atoms(
    table: LoopObservableTable,
    extra_columns_fn: ExtraColumnsFn | None,
) -> dict[str, np.ndarray]:
    cols = {
        name: np.asarray(val, dtype=float).reshape(-1)
        for name, val in table.values.items()
        if name in LEGAL_LOOP_ATOMS
    }
    assert_library_loop_legal(cols)
    if extra_columns_fn is None:
        return cols
    extra = extra_columns_fn(table)
    if any(name in LEGAL_SINGLET_ATOMS for name in extra):
        refuse_jet_as_loop_source(extra)
    green = [name for name in extra if is_green_column_name(name)]
    if green:
        refuse_green_as_jet_atom(green)
    if any(name in CERT_COLUMN_NAMES for name in extra):
        raise ValueError(
            "GEVP / transfer-gap certificates are not loop search columns; "
            f"rejected {sorted(set(extra) & CERT_COLUMN_NAMES)}"
        )
    assert_library_loop_legal(extra)
    for name, col in extra.items():
        cols[name] = np.asarray(col, dtype=float).reshape(-1)
    assert_library_loop_legal(cols)
    return cols


@dataclass(frozen=True)
class LoopLawResult:
    """Sparse loop law plus honesty diagnostics."""

    lhs_name: str
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    family: str = "loop_trace_relation"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return str(self.equation.formula(lhs=self.lhs_name))

    def active_terms(self) -> list[dict[str, float | str]]:
        return self.equation.active_terms()


@dataclass(frozen=True)
class LoopLawDiscoverer:
    """STLSQ over the closed Wilson / Polyakov allowlist.

    Never builds a jet of ``A``. Creutz is not a search column.
    """

    alphas: tuple[float, ...] = (1e-12, 1e-10, 1e-8)
    thresholds: tuple[float, ...] = (1e-8, 1e-6, 1e-4)

    def discover(
        self,
        train: LoopObservableTable | LatticeLinkField,
        val: LoopObservableTable | LatticeLinkField,
        test: LoopObservableTable | LatticeLinkField,
        *,
        lhs_name: str = LOOP_W11,
        extra_columns_fn: ExtraColumnsFn | None = None,
    ) -> LoopLawResult:
        _reject_loop_source(train, "train")
        _reject_loop_source(val, "val")
        _reject_loop_source(test, "test")
        assert_library_loop_legal([lhs_name])
        train_t, val_t, test_t = _as_table(train), _as_table(val), _as_table(test)

        def _library(
            table: LoopObservableTable,
        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
            cols = _merge_loop_atoms(table, extra_columns_fn)
            if any(name in CERT_COLUMN_NAMES for name in cols):
                raise ValueError("GEVP / transfer-gap are not discoverer columns")
            names = [name for name in sorted(cols) if name != lhs_name]
            if not names:
                raise ValueError("loop library has no RHS atoms after dropping LHS")
            design = np.stack([cols[name] for name in names], axis=1)
            return design, cols[lhs_name], names

        train_design, target_train, names = _library(train_t)
        val_design, target_val, _ = _library(val_t)
        test_design, target_test, _ = _library(test_t)

        best: LoopLawResult | None = None
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
                result = LoopLawResult(
                    lhs_name=lhs_name,
                    equation=equation,
                    validation_rmse=val_rmse,
                    test_rmse=rmse(target_test, equation.predict(test_design)),
                    diagnostics={
                        "dictionary_names": tuple(sorted(LEGAL_LOOP_ATOMS)),
                        "yang_mills_claim": False,
                        "continuum_claim": False,
                    },
                )
                if best is None or result.validation_rmse < best.validation_rmse:
                    best = result
        assert best is not None
        return best


def _split_table(
    table: LoopObservableTable, counts: tuple[int, int, int]
) -> tuple[LoopObservableTable, LoopObservableTable, LoopObservableTable]:
    n = int(sum(counts))
    parts: list[LoopObservableTable] = []
    start = 0
    for count in counts:
        values = {
            name: np.asarray(col, dtype=float).reshape(-1)[start : start + count]
            for name, col in table.values.items()
        }
        if any(arr.shape[0] != count for arr in values.values()):
            raise ValueError(f"expected {n} rows in loop table")
        parts.append(LoopObservableTable(values=values, source=table.source))
        start += count
    return parts[0], parts[1], parts[2]


def discover_wilson_plaquette_law(
    *,
    seed: int = 0,
    lattice_shape: tuple[int, int, int, int] = (4, 4, 4, 4),
    n_configs: int = 3,
) -> dict[str, object]:
    """Recover ``W(1,1) = plaquette`` on random unit-quaternion links."""
    rng = np.random.default_rng(seed)
    wilson_parts: list[np.ndarray] = []
    plaq_parts: list[np.ndarray] = []
    for _ in range(n_configs):
        field = LatticeLinkField(links=random_numpy_links(lattice_shape, rng))
        w_arr, p_arr = wilson_plaquette_pairs(field)
        wilson_parts.append(w_arr)
        plaq_parts.append(p_arr)
    table = LoopObservableTable(
        values={
            LOOP_W11: np.concatenate(wilson_parts),
            LOOP_PLAQUETTE: np.concatenate(plaq_parts),
        },
        source="lattice_links",
    )
    n = int(table.values[LOOP_W11].shape[0])
    n_train = max(n // 2, 8)
    n_val = max((n - n_train) // 2, 4)
    n_test = n - n_train - n_val
    train, val, test = _split_table(table, (n_train, n_val, n_test))
    result = LoopLawDiscoverer().discover(train, val, test, lhs_name=LOOP_W11)
    return {
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "diagnostics": result.diagnostics,
        "lhs_name": result.lhs_name,
        "yang_mills_claim": False,
        "continuum_claim": False,
    }


def planted_area_law_table(
    *,
    sigma: float = PLANTED_AREA_SIGMA,
    n_rows: int = 16,
) -> LoopObservableTable:
    """Plant ``W(R,T) = exp(-sigma R T)``. Not a Monte-Carlo vacuum."""
    values = {
        f"W({r_ext},{t_ext})": np.full(
            n_rows, float(np.exp(-sigma * r_ext * t_ext)), dtype=np.float64
        )
        for r_ext, t_ext in ((1, 1), (2, 1), (1, 2), (2, 2))
    }
    return LoopObservableTable(values=values, source="lattice_links")


def discover_planted_area_law(
    *,
    sigma: float = PLANTED_AREA_SIGMA,
    n_rows: int = 16,
    rtol: float = PLANTED_CREUTZ_RTOL,
) -> dict[str, object]:
    """Creutz identity on a planted area law. Not a continuum string tension."""
    table = planted_area_law_table(sigma=sigma, n_rows=n_rows)
    chi = creutz_ratio_from_wilson(table)
    return {
        "sigma_planted": float(sigma),
        "creutz": float(chi),
        "passed": bool(abs(chi - sigma) <= rtol * max(abs(sigma), 1.0)),
        "rtol": float(rtol),
        "yang_mills_claim": False,
        "continuum_claim": False,
        "dictionary_names": tuple(sorted(table.values)),
    }


__all__ = [
    "CERT_COLUMN_NAMES",
    "LoopLawDiscoverer",
    "LoopLawResult",
    "PLANTED_AREA_SIGMA",
    "PLANTED_CREUTZ_RTOL",
    "PLAQUETTE_RMSE_FLOOR",
    "_GAUGE_EXTRA_HINT",
    "discover_planted_area_law",
    "discover_wilson_plaquette_law",
    "evaluate_loop_gauge_invariance",
    "planted_area_law_table",
]

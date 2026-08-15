# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Singlet-only Yang-Mills law discovery on a gauge-covariant jet.

This is **not** :class:`~omnibias.symbolic.field_discovery.FieldLawDiscoverer`
applied to a coordinate jet of ``A_mu^a``. The only legal input is a
:class:`~omnibias.geometry.gauge._core.covariant_jet.GaugeCovariantJet`; the
design matrix is the closed Weyl-singlet allowlist. Coordinate partials,
flattened ``F_{mu nu}^a`` components, and adjoint/singlet mixing are rejected
before STLSQ.

Honesty: recovers classical *local* singlet identities (e.g. the self-dual
relation between the action and topological densities on a BPST instanton).
Not Wilson / Polyakov language, not a continuum mass-gap claim, not quantum
Yang-Mills. Operator columns are closed form; the sparse fit is numerical
STLSQ.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation, rmse

_GAUGE_EXTRA_HINT = (
    "omnibias.symbolic.gauge_discovery requires omnibias-geometry; "
    "install the optional extra omnibias-symbolic[gauge]"
)

try:
    from omnibias.geometry.gauge._core.connection import EUCLIDEAN_4D
    from omnibias.geometry.gauge._core.covariant_jet import (
        LEGAL_ADJOINT_1FORM_ATOMS,
        SELF_DUAL_ACTION_OVER_TOPOLOGICAL,
        SINGLET_TR_F2,
        SINGLET_TR_F_FTILDE,
        GaugeCovariantJet,
        assert_library_gauge_legal,
        evaluate_gauge_law_gate,
    )
    from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
    from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra, su
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(_GAUGE_EXTRA_HINT) from exc

ConnectionArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
ExtraColumnsFn = Callable[[GaugeCovariantJet], Mapping[str, np.ndarray]]


def _reject_field_jet(obj: object, label: str) -> None:
    cls_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "")
    if cls_name == "FieldJet" or module.endswith("field_discovery"):
        raise TypeError(
            f"{label} must be a GaugeCovariantJet, not a coordinate FieldJet "
            f"of A (got {module}.{cls_name})"
        )
    if not isinstance(obj, GaugeCovariantJet):
        raise TypeError(
            f"{label} must be a GaugeCovariantJet, got {type(obj)!r}"
        )


def _merge_singlets(
    jet: GaugeCovariantJet,
    extra_columns_fn: ExtraColumnsFn | None,
) -> dict[str, np.ndarray]:
    cols = dict(jet.singlets())
    if extra_columns_fn is None:
        assert_library_gauge_legal(cols)
        return cols
    extra = extra_columns_fn(jet)
    if any(name in LEGAL_ADJOINT_1FORM_ATOMS for name in extra):
        raise TypeError(
            "adjoint 1-form atoms cannot be mixed into the singlet discoverer; "
            f"rejected {sorted(set(extra) & LEGAL_ADJOINT_1FORM_ATOMS)}"
        )
    assert_library_gauge_legal(extra)
    for name, col in extra.items():
        cols[name] = np.asarray(col, dtype=float).reshape(-1)
    assert_library_gauge_legal(cols)
    return cols


@dataclass(frozen=True)
class GaugeLawResult:
    """Sparse singlet law plus a fail-closed gauge-equivariance diagnostic."""

    lhs_name: str
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    selection_score: float
    target_scale: float
    family: str = "gauge_singlet_relation"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return str(self.equation.formula(lhs=self.lhs_name))

    def active_terms(self) -> list[dict[str, float | str]]:
        return self.equation.active_terms()


@dataclass(frozen=True)
class GaugeLawDiscoverer:
    """STLSQ over Weyl singlets of a :class:`GaugeCovariantJet`.

    Never builds a coordinate library of ``partial^alpha A``. ``extra_columns_fn``
    may only return allowlisted singlet names.
    """

    max_degree: int = 1
    alphas: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6)
    thresholds: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-3)
    complexity_weight: float = 2e-3
    random_state: int = 0
    gate_atol: float = 1e-10

    def discover(
        self,
        train: GaugeCovariantJet,
        val: GaugeCovariantJet,
        test: GaugeCovariantJet,
        *,
        lhs_name: str = SINGLET_TR_F2,
        extra_columns_fn: ExtraColumnsFn | None = None,
        connections: tuple[ConnectionArrays, ConnectionArrays, ConnectionArrays]
        | None = None,
    ) -> GaugeLawResult:
        _reject_field_jet(train, "train")
        _reject_field_jet(val, "val")
        _reject_field_jet(test, "test")
        if self.max_degree != 1:
            raise ValueError("GaugeLawDiscoverer ships max_degree=1 only")
        assert_library_gauge_legal([lhs_name])

        def _library(
            jet: GaugeCovariantJet,
        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
            cols = _merge_singlets(jet, extra_columns_fn)
            names = [name for name in sorted(cols) if name != lhs_name]
            if not names:
                raise ValueError("singlet library has no RHS atoms after dropping LHS")
            design = np.stack([cols[name] for name in names], axis=1)
            return design, cols[lhs_name], names

        train_design, target_train, names = _library(train)
        val_design, target_val, _ = _library(val)
        test_design, target_test, _ = _library(test)
        scale = float(np.std(target_val))
        if scale < 1e-12:
            scale = 1.0

        best: GaugeLawResult | None = None
        for alpha in self.alphas:
            for threshold in self.thresholds:
                equation = fit_sparse_equation(
                    train_design,
                    target_train,
                    names,
                    alpha=alpha,
                    threshold=threshold,
                )
                val_pred = equation.predict(val_design)
                val_rmse = rmse(target_val, val_pred)
                active_count = len(equation.active_terms())
                score = val_rmse / scale + self.complexity_weight * active_count
                test_pred = equation.predict(test_design)
                result = GaugeLawResult(
                    lhs_name=lhs_name,
                    equation=equation,
                    validation_rmse=val_rmse,
                    test_rmse=rmse(target_test, test_pred),
                    selection_score=score,
                    target_scale=scale,
                )
                if best is None or result.selection_score < best.selection_score:
                    best = result
        assert best is not None

        if connections is None:
            diagnostics: dict[str, object] = {
                "gauge_equivariance": {
                    "passed": False,
                    "reason": "connections_required",
                    "yang_mills_claim": False,
                    "continuum_claim": False,
                }
            }
            return GaugeLawResult(
                lhs_name=best.lhs_name,
                equation=best.equation,
                validation_rmse=best.validation_rmse,
                test_rmse=best.test_rmse,
                selection_score=best.selection_score,
                target_scale=best.target_scale,
                diagnostics=diagnostics,
            )

        a, da, dda = connections[0]
        extra = None if extra_columns_fn is None else dict(extra_columns_fn(train))
        gate = evaluate_gauge_law_gate(
            best.equation,
            lhs_name=lhs_name,
            A=a,
            dA=da,
            ddA=dda,
            algebra=train.algebra,
            coupling=train.coupling,
            signature=train.signature,
            extra_columns=extra,
            rng=np.random.default_rng(self.random_state),
            atol=self.gate_atol,
        )
        if not gate["passed"]:
            raise ValueError(
                "gauge-equivariance gate failed; refusing a gauge-variant law "
                f"(residual_defect={gate['residual_defect']})"
            )
        return GaugeLawResult(
            lhs_name=best.lhs_name,
            equation=best.equation,
            validation_rmse=best.validation_rmse,
            test_rmse=best.test_rmse,
            selection_score=best.selection_score,
            target_scale=best.target_scale,
            diagnostics={"gauge_equivariance": gate},
        )


def _split_points(
    points: np.ndarray, counts: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(sum(counts))
    if points.shape[0] != n:
        raise ValueError(f"expected {n} points, got {points.shape[0]}")
    i, j, k = counts
    return points[:i], points[i : i + j], points[i + j :]


def make_yang_mills_bpst_split(
    *,
    seed: int = 0,
    counts: tuple[int, int, int] = (48, 24, 24),
    coupling: float = 1.0,
) -> tuple[
    GaugeCovariantJet,
    GaugeCovariantJet,
    GaugeCovariantJet,
    tuple[ConnectionArrays, ConnectionArrays, ConnectionArrays],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Disjoint BPST point splits and their covariant jets."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.6, 1.6, size=(sum(counts), 4))
    # Reject near-origin samples where the regular-gauge core is stiff.
    norms = np.linalg.norm(points, axis=1)
    points = points[norms > 0.35]
    while points.shape[0] < sum(counts):
        extra = rng.uniform(-1.6, 1.6, size=(sum(counts), 4))
        extra = extra[np.linalg.norm(extra, axis=1) > 0.35]
        points = np.concatenate([points, extra], axis=0)
    points = points[: sum(counts)]
    train_x, val_x, test_x = _split_points(points, counts)
    algebra = su(2)
    jets: list[GaugeCovariantJet] = []
    conns: list[ConnectionArrays] = []
    for xs in (train_x, val_x, test_x):
        a, da, dda = bpst_instanton_arrays(xs)
        conns.append((a, da, dda))
        jets.append(
            GaugeCovariantJet.from_arrays(
                a,
                da,
                dda,
                algebra=algebra,
                coupling=coupling,
                signature=EUCLIDEAN_4D,
            )
        )
    return jets[0], jets[1], jets[2], (conns[0], conns[1], conns[2]), (train_x, val_x, test_x)


def make_yang_mills_polynomial_split(
    *,
    seed: int = 1,
    counts: tuple[int, int, int] = (48, 24, 24),
    algebra: LieAlgebra | None = None,
    coupling: float = 0.8,
) -> tuple[
    GaugeCovariantJet,
    GaugeCovariantJet,
    GaugeCovariantJet,
    tuple[ConnectionArrays, ConnectionArrays, ConnectionArrays],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Disjoint splits of a generic (non-self-dual) quadratic connection."""
    alg = su(2) if algebra is None else algebra
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.8, 0.8, size=(sum(counts), 4))
    train_x, val_x, test_x = _split_points(points, counts)
    c0 = rng.normal(size=(4, alg.dim))
    c1 = rng.normal(size=(4, 4, alg.dim))
    c2 = rng.normal(size=(4, 4, 4, alg.dim))
    c2 = 0.5 * (c2 + np.swapaxes(c2, 0, 1))

    def _at(xs: np.ndarray) -> ConnectionArrays:
        a = (
            c0[None]
            + np.einsum("lma,Bl->Bma", c1, xs)
            + 0.5 * np.einsum("slma,Bs,Bl->Bma", c2, xs, xs)
        )
        da = c1[None] + np.einsum("rlna,Bl->Brna", c2, xs)
        dda = np.broadcast_to(c2[None], (xs.shape[0], 4, 4, 4, alg.dim)).copy()
        return a, da, dda

    jets: list[GaugeCovariantJet] = []
    conns: list[ConnectionArrays] = []
    for xs in (train_x, val_x, test_x):
        a, da, dda = _at(xs)
        conns.append((a, da, dda))
        jets.append(
            GaugeCovariantJet.from_arrays(
                a, da, dda, algebra=alg, coupling=coupling, signature=EUCLIDEAN_4D
            )
        )
    return jets[0], jets[1], jets[2], (conns[0], conns[1], conns[2]), (train_x, val_x, test_x)


def discover_yang_mills_singlet_law(
    train: GaugeCovariantJet,
    val: GaugeCovariantJet,
    test: GaugeCovariantJet,
    connections: tuple[ConnectionArrays, ConnectionArrays, ConnectionArrays],
    *,
    lhs_name: str = SINGLET_TR_F2,
    random_state: int = 0,
) -> dict[str, object]:
    """Recover a sparse singlet law (BPST: ``tr(F^2) ~ 8 pi^2 tr(F*Ftilde)``)."""
    discoverer = GaugeLawDiscoverer(random_state=random_state)
    result = discoverer.discover(
        train, val, test, lhs_name=lhs_name, connections=connections
    )
    return {
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "target_scale": result.target_scale,
        "diagnostics": result.diagnostics,
        "lhs_name": result.lhs_name,
        "expected_self_dual_coefficient": SELF_DUAL_ACTION_OVER_TOPOLOGICAL,
        "self_dual_rhs": SINGLET_TR_F_FTILDE,
    }


__all__ = [
    "ConnectionArrays",
    "GaugeLawDiscoverer",
    "GaugeLawResult",
    "_GAUGE_EXTRA_HINT",
    "discover_yang_mills_singlet_law",
    "make_yang_mills_bpst_split",
    "make_yang_mills_polynomial_split",
]

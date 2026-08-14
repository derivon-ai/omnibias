# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scientific equation discovery and interpretable surrogate modeling.

This module is the core of :mod:`omnibias.symbolic`:

* learn a compact surrogate by selecting among Taylor, Fourier, and hybrid
  series libraries;
* recover a PDE-style operator law from derivative/operator columns.

The derivative columns are generated analytically in this first PoC. In a full
omnibias workflow they can be supplied by ``omnibias.pinn`` field operators
after fitting a smooth neural field to observations.

Honesty: the derivative columns are **exact closed form** (the omnibias jet),
but the sparse-relation fit itself (:func:`fit_sparse_equation`, STLSQ) is a
**numerical, non-differentiable** numpy least-squares step -- not a closed-form
identity.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from omnibias.symbolic.diagnostics import (
    design_conditioning_report,
    divergence_objective_term,
    residual_dependence_report,
    residual_distribution_report,
    surrogate_residual_diagnostics,
)


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class LibrarySpec:
    """One candidate representation family for AutoML selection."""

    name: str
    builder: Callable[[np.ndarray], tuple[np.ndarray, list[str]]]
    description: str


@dataclass(frozen=True)
class FeatureLibraryPlan:
    """Train-only screened feature transform plan for high-dimensional data."""

    feature_names: tuple[str, ...]
    raw_indices: tuple[int, ...]
    square_indices: tuple[int, ...]
    sin_indices: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        x = np.asarray(x, dtype=float)
        cols: list[np.ndarray] = []
        names: list[str] = []
        for index in self.raw_indices:
            cols.append(x[:, index])
            names.append(self.feature_names[index])
        for index in self.square_indices:
            cols.append(x[:, index] ** 2)
            names.append(f"{self.feature_names[index]}^2")
        for index in self.sin_indices:
            cols.append(np.sin(x[:, index]))
            names.append(f"sin({self.feature_names[index]})")
        for left, right in self.pair_indices:
            cols.append(x[:, left] * x[:, right])
            names.append(f"{self.feature_names[left]}*{self.feature_names[right]}")
        if not cols:
            raise ValueError("feature library plan produced no columns")
        return np.stack(cols, axis=1), names


@dataclass(frozen=True)
class SparseEquation:
    """A fitted sparse linear equation over named candidate terms.

    The optional ``coefficient_ci`` / ``coefficient_intervals`` /
    ``selection_frequency`` fields carry per-term uncertainty (aligned to
    ``term_names``) when populated by
    :func:`omnibias.symbolic.uncertainty.attach_uncertainty`; they default to
    ``None`` so the plain point-estimate fit is unchanged.
    """

    term_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    threshold: float
    active_mask: np.ndarray
    #: Bootstrap percentile confidence interval ``(lo, hi)`` per term, or ``None``.
    coefficient_ci: tuple[tuple[float, float], ...] | None = None
    #: Certified (verified-interval) enclosure ``(lo, hi)`` per term, or ``None``.
    coefficient_intervals: tuple[tuple[float, float], ...] | None = None
    #: Bootstrap selection frequency in ``[0, 1]`` per term, or ``None``.
    selection_frequency: tuple[float, ...] | None = None

    def predict(self, design: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(design) @ self.coefficients

    def active_terms(self, *, min_abs: float = 1e-10) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for index, (name, coef) in enumerate(
            zip(self.term_names, self.coefficients, strict=False)
        ):
            value = float(coef)
            if abs(value) >= min_abs:
                row: dict[str, float | str] = {"name": name, "coefficient": value}
                if self.coefficient_ci is not None:
                    lo, hi = self.coefficient_ci[index]
                    row["ci_lower"] = float(lo)
                    row["ci_upper"] = float(hi)
                if self.coefficient_intervals is not None:
                    lo, hi = self.coefficient_intervals[index]
                    row["certified_lower"] = float(lo)
                    row["certified_upper"] = float(hi)
                if self.selection_frequency is not None:
                    row["selection_frequency"] = float(self.selection_frequency[index])
                rows.append(row)
        return sorted(rows, key=lambda row: -abs(float(row["coefficient"])))

    def uncertainty_formula(self, *, lhs: str = "y", digits: int = 4) -> str:
        """Like :meth:`formula` but annotates each term with its bootstrap CI.

        Falls back to :meth:`formula` when no bootstrap CI is attached.
        """
        if self.coefficient_ci is None:
            return self.formula(lhs=lhs, digits=digits)
        pieces = [f"{self.intercept:.{digits}g}"] if abs(self.intercept) > 1e-10 else []
        for row in self.active_terms():
            coef = float(row["coefficient"])
            name = str(row["name"])
            lo = float(row.get("ci_lower", coef))
            hi = float(row.get("ci_upper", coef))
            half = 0.5 * (hi - lo)
            term = f"({coef:.{digits}g} +/- {half:.{digits}g})*{name}"
            pieces.append(f"+ {term}" if pieces else term)
        rhs = " ".join(pieces) if pieces else "0"
        return f"{lhs} = {rhs}"

    def formula(self, *, lhs: str = "y", digits: int = 4) -> str:
        pieces = [f"{self.intercept:.{digits}g}"] if abs(self.intercept) > 1e-10 else []
        for row in self.active_terms():
            coef = float(row["coefficient"])
            name = str(row["name"])
            sign = "+" if coef >= 0 else "-"
            mag = abs(coef)
            term = f"{mag:.{digits}g}*{name}"
            if pieces:
                pieces.append(f"{sign} {term}")
            else:
                pieces.append(term if sign == "+" else f"-{term}")
        rhs = " ".join(pieces) if pieces else "0"
        return f"{lhs} = {rhs}"


@dataclass(frozen=True)
class NeuralField1D:
    """One-dimensional random-feature neural field with closed-form derivatives."""

    W: np.ndarray
    beta: np.ndarray
    c: np.ndarray
    b: float
    x_mean: float
    x_scale: float
    activation: str
    train_rmse: float = 0.0


@dataclass(frozen=True)
class JetBundle:
    """Samples of a function and its derivative jet."""

    x: np.ndarray
    jets: np.ndarray

    def name(self, order: int) -> str:
        return jet_name(order)


@dataclass(frozen=True)
class JetDiscoveryResult:
    """Best compressed differential identity found from neural jets."""

    lhs_order: int
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    selection_score: float
    target_scale: float
    family: str = "implicit_jet_polynomial"
    #: Residual distribution / dependence diagnostics of the selected relation
    #: (entropy, KL / W1 to a matched Gaussian, max input-residual MI). Empty
    #: until populated by :meth:`NeuralJetDiscoverer.discover`.
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return self.equation.formula(lhs=jet_name(self.lhs_order))

    def active_terms(self) -> list[dict[str, float | str]]:
        return self.equation.active_terms()


@dataclass(frozen=True)
class FractionalOrderDiscoveryResult:
    """Best fractional differential law found by searching the fractional order.

    The twin of :class:`JetDiscoveryResult` for
    :func:`discover_fractional_order_law`: it additionally records the recovered
    fractional order :attr:`fractional_order` and the grid of
    :attr:`candidate_orders` searched.
    """

    fractional_order: float
    lhs_order: int
    equation: SparseEquation
    validation_rmse: float
    test_rmse: float
    selection_score: float
    target_scale: float
    candidate_orders: tuple[float, ...]
    family: str = "fractional_jet_relation"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def formula(self) -> str:
        return self.equation.formula(lhs=jet_name(self.lhs_order))

    def active_terms(self) -> list[dict[str, float | str]]:
        return self.equation.active_terms()


@dataclass(frozen=True)
class NeuralJetDiscoverer:
    """Library-free-ish differential identity discovery over neural jets.

    The discoverer does not search named functions such as ``sin`` or ``exp``.
    It searches compact implicit relations among generic jet coordinates:
    ``x``, ``y``, ``dy``, ``d2y``, ...
    """

    max_library_degree: int = 2
    alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4)
    thresholds: tuple[float, ...] = (1e-7, 1e-5, 1e-4, 1e-3)
    complexity_weight: float = 2e-3
    #: Include the independent variable ``x`` in the polynomial library. Default
    #: ``True`` preserves existing callers. Set ``False`` when ``x`` is collinear
    #: with ``y`` on a short interval (per-leaf ODE identities ``dy ≈ c y``).
    include_x: bool = True
    #: Probability-operator features: CDF bases mixed into the jet library. Empty
    #: (the default) keeps the pure polynomial behaviour; set e.g.
    #: ``("sigmoid", "tanh")`` to also discover saturating differential laws.
    cdf_feature_bases: tuple[str, ...] = ()
    cdf_n_locations: int = 5
    cdf_scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0)
    #: Information-operator features: surprisal bases mixed into the jet library
    #: (the twin of ``cdf_feature_bases``). Empty keeps the default behaviour; set
    #: e.g. ``("sigmoid", "arctan")`` to also discover log-likelihood / energy laws.
    info_feature_bases: tuple[str, ...] = ()
    info_n_locations: int = 5
    info_scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0)
    #: Fractional-calculus features: Grunwald-Letnikov derivatives ``D^alpha`` of
    #: the ``fractional_source_order`` jet (``y`` by default) over the x grid, mixed
    #: into the jet library. Empty keeps the default behaviour; set e.g.
    #: ``(0.25, 0.5, 0.75)`` to also discover non-local / fractional differential
    #: laws. Requires a (near-)uniform sorted x grid (see :func:`split_x_grid`).
    fractional_orders: tuple[float, ...] = ()
    fractional_source_order: int = 0
    #: Piecewise (multi-terminal) *closed-form* fractional features: one Taylor jet
    #: per terminal, selected or blended by
    #: :func:`build_jet_piecewise_fractional_features`. Use instead of
    #: ``fractional_orders`` when a single expansion cannot cover the grid.
    #: ``piecewise_fractional_terminals`` must be set alongside the orders.
    piecewise_fractional_orders: tuple[float, ...] = ()
    piecewise_fractional_terminals: tuple[float, ...] = ()
    piecewise_fractional_source_order: int = 0
    piecewise_fractional_kind: str = "caputo"
    piecewise_fractional_blend: float = 0.0
    #: Leading jet columns that make up the derivative tower. ``None`` uses the
    #: whole jet width, which is wrong whenever the bundle carries appended target /
    #: LHS columns -- those would be read as high derivatives. Set it to the true
    #: tower width in that case (the same knob as
    #: :func:`discover_fractional_order_law`).
    piecewise_fractional_tower_width: int | None = None
    #: Closed-form fractional derivatives of a *named activation* evaluated on the x
    #: grid (see :func:`build_jet_activation_fractional_features`). The name must be
    #: registered in ``ACTIVATION_FRACTIONAL`` (``exp`` / ``cosh`` / ``sinh``).
    #: These columns depend only on ``x``, so no LHS-order guard applies.
    activation_fractional_orders: tuple[float, ...] = ()
    activation_fractional_name: str = "exp"
    activation_fractional_kind: str = "riemann_liouville"
    #: Spectral fractional features on a bounded non-periodic grid (see
    #: :func:`build_jet_spectral_fractional_features`). Numerical-spectral, not
    #: closed form, and the grid must match the transform's basis layout.
    spectral_fractional_orders: tuple[float, ...] = ()
    spectral_fractional_source_order: int = 0
    spectral_fractional_bc: str = "dirichlet"
    spectral_fractional_length: float | None = None
    spectral_fractional_windowed: bool = False
    #: Integral (non-local) features -- the running integral, plus Fredholm and
    #: causal Volterra columns for the supplied kernels (see
    #: :func:`build_jet_integral_features`). These are what let the discoverer
    #: express an integro-differential law; they are off unless
    #: ``integral_running`` is set or a kernel mapping is given.
    #: ``integral_measure`` (an :class:`omnibias.measure.Measure` on the bundle's
    #: own grid) is required by the Fredholm family only. The causal families are
    #: dropped when ``integral_source_order == lhs_order + 1``, where integrating
    #: would reproduce the left-hand side; the Fredholm family is not, since a
    #: global column carries no such identity and putting the unknown on both
    #: sides is the whole shape of a Fredholm equation.
    integral_running: bool = False
    integral_kernels: Mapping[str, IntegralKernel] | None = None
    integral_volterra_kernels: Mapping[str, IntegralKernel] | None = None
    integral_measure: object | None = None
    integral_source_order: int = 0
    #: Shared lower terminal for the causal integral columns. Leave ``None`` only
    #: when every split starts at the same ``x``; otherwise the running-integral
    #: column means something different on each split and the fitted law will not
    #: transfer. See :func:`build_jet_integral_features`.
    integral_origin: float | None = None
    #: Opt-in divergence-aware selection objective (see
    #: :data:`omnibias.symbolic.diagnostics.DIVERGENCE_OBJECTIVES`). ``None`` scores
    #: by validation RMSE + complexity only; otherwise ``weight * term`` is added.
    divergence_objective: str | None = None
    divergence_weight: float = 1.0
    diagnostics_bins: int = 32
    dependence_bins: int = 16

    def discover(
        self,
        train: JetBundle,
        val: JetBundle,
        test: JetBundle,
        *,
        candidate_lhs_orders: tuple[int, ...] | None = None,
        selection_criterion: str | None = None,
    ) -> JetDiscoveryResult:
        """Search compact jet relations.

        ``selection_criterion`` (``None`` by default, preserving the validation
        RMSE + complexity score) may be ``"aic"`` / ``"aicc"`` / ``"bic"`` /
        ``"mdl"``; when set, candidates are ranked by that information criterion
        evaluated on the training residuals (lower is better), with any
        configured divergence objective still added.
        """
        if candidate_lhs_orders is None:
            candidate_lhs_orders = tuple(range(1, train.jets.shape[1]))
        if selection_criterion is not None:
            from omnibias.symbolic.selection import information_criterion
        best: JetDiscoveryResult | None = None
        best_diagnostics: dict[str, object] = {}
        for lhs_order in candidate_lhs_orders:
            train_design, names = build_jet_relation_library(
                train,
                lhs_order=lhs_order,
                max_degree=self.max_library_degree,
                include_x=self.include_x,
            )
            val_design, _ = build_jet_relation_library(
                val,
                lhs_order=lhs_order,
                max_degree=self.max_library_degree,
                include_x=self.include_x,
            )
            test_design, _ = build_jet_relation_library(
                test,
                lhs_order=lhs_order,
                max_degree=self.max_library_degree,
                include_x=self.include_x,
            )
            if self.cdf_feature_bases:
                per_locs, per_scales = fit_jet_cdf_plan(
                    train,
                    lhs_order=lhs_order,
                    bases=self.cdf_feature_bases,
                    n_locations=self.cdf_n_locations,
                    scale_mults=self.cdf_scale_mults,
                )
                cdf_kw = {
                    "lhs_order": lhs_order,
                    "bases": self.cdf_feature_bases,
                    "per_variable_locations": per_locs,
                    "per_variable_scales": per_scales,
                }
                tr_cdf, cdf_names = build_jet_cdf_features(train, **cdf_kw)
                va_cdf, _ = build_jet_cdf_features(val, **cdf_kw)
                te_cdf, _ = build_jet_cdf_features(test, **cdf_kw)
                train_design = np.concatenate([train_design, tr_cdf], axis=1)
                val_design = np.concatenate([val_design, va_cdf], axis=1)
                test_design = np.concatenate([test_design, te_cdf], axis=1)
                names = names + cdf_names
            if self.info_feature_bases:
                per_locs, per_scales = fit_jet_info_plan(
                    train,
                    lhs_order=lhs_order,
                    bases=self.info_feature_bases,
                    n_locations=self.info_n_locations,
                    scale_mults=self.info_scale_mults,
                )
                info_kw = {
                    "lhs_order": lhs_order,
                    "bases": self.info_feature_bases,
                    "per_variable_locations": per_locs,
                    "per_variable_scales": per_scales,
                }
                tr_info, info_names = build_jet_info_features(train, **info_kw)
                va_info, _ = build_jet_info_features(val, **info_kw)
                te_info, _ = build_jet_info_features(test, **info_kw)
                train_design = np.concatenate([train_design, tr_info], axis=1)
                val_design = np.concatenate([val_design, va_info], axis=1)
                test_design = np.concatenate([test_design, te_info], axis=1)
                names = names + info_names
            if self.fractional_orders and self.fractional_source_order != lhs_order:
                frac_kw = {
                    "orders": self.fractional_orders,
                    "source_order": self.fractional_source_order,
                }
                tr_frac, frac_names = build_jet_fractional_features(train, **frac_kw)
                va_frac, _ = build_jet_fractional_features(val, **frac_kw)
                te_frac, _ = build_jet_fractional_features(test, **frac_kw)
                train_design = np.concatenate([train_design, tr_frac], axis=1)
                val_design = np.concatenate([val_design, va_frac], axis=1)
                test_design = np.concatenate([test_design, te_frac], axis=1)
                names = names + frac_names
            if (
                self.piecewise_fractional_orders
                and self.piecewise_fractional_source_order != lhs_order
            ):
                pw_kw = {
                    "orders": self.piecewise_fractional_orders,
                    "terminals": self.piecewise_fractional_terminals,
                    "source_order": self.piecewise_fractional_source_order,
                    "kind": self.piecewise_fractional_kind,
                    "blend": self.piecewise_fractional_blend,
                    "tower_width": self.piecewise_fractional_tower_width,
                }
                tr_pw, pw_names = build_jet_piecewise_fractional_features(train, **pw_kw)
                va_pw, _ = build_jet_piecewise_fractional_features(val, **pw_kw)
                te_pw, _ = build_jet_piecewise_fractional_features(test, **pw_kw)
                train_design = np.concatenate([train_design, tr_pw], axis=1)
                val_design = np.concatenate([val_design, va_pw], axis=1)
                test_design = np.concatenate([test_design, te_pw], axis=1)
                names = names + pw_names
            # No lhs_order guard: these columns are a function of x alone, so they
            # can never smuggle the left-hand side in as its own feature.
            if self.activation_fractional_orders:
                act_kw = {
                    "orders": self.activation_fractional_orders,
                    "name": self.activation_fractional_name,
                    "kind": self.activation_fractional_kind,
                }
                tr_act, act_names = build_jet_activation_fractional_features(
                    train, **act_kw
                )
                va_act, _ = build_jet_activation_fractional_features(val, **act_kw)
                te_act, _ = build_jet_activation_fractional_features(test, **act_kw)
                train_design = np.concatenate([train_design, tr_act], axis=1)
                val_design = np.concatenate([val_design, va_act], axis=1)
                test_design = np.concatenate([test_design, te_act], axis=1)
                names = names + act_names
            if (
                self.spectral_fractional_orders
                and self.spectral_fractional_source_order != lhs_order
            ):
                spec_kw = {
                    "orders": self.spectral_fractional_orders,
                    "source_order": self.spectral_fractional_source_order,
                    "bc": self.spectral_fractional_bc,
                    "length": self.spectral_fractional_length,
                    "windowed": self.spectral_fractional_windowed,
                }
                tr_spec, spec_names = build_jet_spectral_fractional_features(
                    train, **spec_kw
                )
                va_spec, _ = build_jet_spectral_fractional_features(val, **spec_kw)
                te_spec, _ = build_jet_spectral_fractional_features(test, **spec_kw)
                train_design = np.concatenate([train_design, tr_spec], axis=1)
                val_design = np.concatenate([val_design, va_spec], axis=1)
                test_design = np.concatenate([test_design, te_spec], axis=1)
                names = names + spec_names
            # The leakage risk for an integral column is the mirror of a derivative
            # column's: integrating the jet one order *above* the LHS reproduces the
            # LHS itself (int y' = y up to a constant), so the causal families are
            # dropped in that case. A global Fredholm column carries no such
            # identity -- and is exactly how a Fredholm equation puts the unknown on
            # both sides -- so it is never dropped.
            causal_leaks = self.integral_source_order == lhs_order + 1
            wants_causal = (
                self.integral_running or self.integral_volterra_kernels
            ) and not causal_leaks
            if wants_causal or self.integral_kernels:
                int_kw = {
                    "measure": self.integral_measure,
                    "kernels": self.integral_kernels,
                    "volterra_kernels": (
                        self.integral_volterra_kernels if wants_causal else None
                    ),
                    "running": self.integral_running and wants_causal,
                    "source_order": self.integral_source_order,
                    "origin": self.integral_origin,
                }
                tr_int, int_names = build_jet_integral_features(train, **int_kw)
                va_int, _ = build_jet_integral_features(val, **int_kw)
                te_int, _ = build_jet_integral_features(test, **int_kw)
                train_design = np.concatenate([train_design, tr_int], axis=1)
                val_design = np.concatenate([val_design, va_int], axis=1)
                test_design = np.concatenate([test_design, te_int], axis=1)
                names = names + int_names
            target_train = train.jets[:, lhs_order]
            target_val = val.jets[:, lhs_order]
            target_test = test.jets[:, lhs_order]
            scale = float(np.std(target_val))
            if scale < 1e-12:
                scale = 1.0
            # Right-hand-side jet coordinates: the "inputs" a correct relation
            # should leave the residual independent of (used for MI diagnostics /
            # the residual_mi objective).
            val_feat = np.stack(
                _jet_relation_variables(
                    val,
                    lhs_order=lhs_order,
                    include_x=self.include_x,
                    lower_order_only=True,
                )[0],
                axis=1,
            )
            test_feat = np.stack(
                _jet_relation_variables(
                    test,
                    lhs_order=lhs_order,
                    include_x=self.include_x,
                    lower_order_only=True,
                )[0],
                axis=1,
            )
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
                    if selection_criterion is None:
                        score = val_rmse / scale + self.complexity_weight * active_count
                    else:
                        train_resid = target_train - equation.predict(train_design)
                        score = information_criterion(
                            selection_criterion,
                            int(target_train.shape[0]),
                            float(train_resid @ train_resid),
                            active_count + 1,
                            n_candidates=len(names),
                        )
                    if self.divergence_objective is not None:
                        score += self.divergence_weight * divergence_objective_term(
                            self.divergence_objective,
                            val_feat,
                            target_val - val_pred,
                            bins=self.diagnostics_bins,
                            dependence_bins=self.dependence_bins,
                        )
                    test_pred = equation.predict(test_design)
                    result = JetDiscoveryResult(
                        lhs_order=lhs_order,
                        equation=equation,
                        validation_rmse=val_rmse,
                        test_rmse=rmse(target_test, test_pred),
                        selection_score=score,
                        target_scale=scale,
                    )
                    if best is None or result.selection_score < best.selection_score:
                        best = result
                        resid = target_test - test_pred
                        best_diagnostics = residual_distribution_report(
                            resid, bins=self.diagnostics_bins
                        )
                        best_diagnostics.update(
                            residual_dependence_report(
                                test_feat, resid, bins=self.dependence_bins
                            )
                        )
                        best_diagnostics.update(
                            design_conditioning_report(train_design)
                        )
        if best is None:
            raise RuntimeError(
                "NeuralJetDiscoverer.search produced no candidates; "
                "check candidate_lhs_orders, alphas, and thresholds"
            )
        return replace(best, diagnostics=best_diagnostics)


def make_symbolic_regression_dataset(
    *,
    n_samples: int = 900,
    noise_std: float = 0.02,
    seed: int = 0,
) -> SplitData:
    """Synthetic law with both Taylor and Fourier structure.

    Hidden target:

        y = 1.5*x1^2 - 2*x2*x3 + sin(2*x4) + 0.4*cos(x4) + noise
    """

    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_samples, 4))
    x[:, 3] = rng.uniform(-math.pi, math.pi, size=n_samples)
    y = symbolic_hidden_law(x) + rng.normal(0.0, noise_std, size=n_samples)
    order = rng.permutation(n_samples)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
    )


def symbolic_hidden_law(x: np.ndarray) -> np.ndarray:
    return 1.5 * x[:, 0] ** 2 - 2.0 * x[:, 1] * x[:, 2] + np.sin(2.0 * x[:, 3]) + 0.4 * np.cos(x[:, 3])


def make_high_dim_sparse_dataset(
    *,
    n_samples: int = 1200,
    n_features: int = 60,
    noise_std: float = 0.05,
    seed: int = 0,
) -> tuple[SplitData, dict[str, object]]:
    """High-dimensional sparse law with only a few active variables."""

    if n_features < 42:
        raise ValueError("n_features must be at least 42 for the default hidden law")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_samples, n_features))
    x[:, 7] = rng.uniform(-math.pi, math.pi, size=n_samples)
    y_clean = 2.0 * x[:, 2] ** 2 - 1.5 * x[:, 16] * x[:, 41] + np.sin(x[:, 7]) + 0.75 * x[:, 4]
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)
    order = rng.permutation(n_samples)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    hidden = {
        "law": "y = 2*x3^2 - 1.5*x17*x42 + sin(x8) + 0.75*x5 + noise",
        "terms": ["x3^2", "x17*x42", "sin(x8)", "x5"],
        "active_feature_indices": [2, 4, 7, 16, 41],
    }
    return (
        SplitData(
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_val=x[val_idx],
            y_val=y[val_idx],
            x_test=x[test_idx],
            y_test=y[test_idx],
        ),
        hidden,
    )


def fit_screened_feature_library_plan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    max_raw: int = 20,
    max_square: int = 20,
    max_sin: int = 20,
    max_pairs: int = 40,
) -> FeatureLibraryPlan:
    """Screen candidate transforms from training data only."""

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    names = tuple(feature_names or [f"x{j + 1}" for j in range(x_train.shape[1])])
    if len(names) != x_train.shape[1]:
        raise ValueError("feature_names must match x_train width")

    raw_scores = np.asarray([abs(_corr(x_train[:, j], y_train)) for j in range(x_train.shape[1])])
    square_scores = np.asarray([abs(_corr(x_train[:, j] ** 2, y_train)) for j in range(x_train.shape[1])])
    sin_scores = np.asarray([abs(_corr(np.sin(x_train[:, j]), y_train)) for j in range(x_train.shape[1])])
    raw_indices = _top_indices(raw_scores, max_raw)
    square_indices = _top_indices(square_scores, max_square)
    sin_indices = _top_indices(sin_scores, max_sin)

    pair_scores: list[tuple[float, int, int]] = []
    for left in range(x_train.shape[1]):
        for right in range(left + 1, x_train.shape[1]):
            score = abs(_corr(x_train[:, left] * x_train[:, right], y_train))
            pair_scores.append((score, left, right))
    pair_scores.sort(key=lambda row: (-row[0], row[1], row[2]))
    pair_indices = tuple((left, right) for _, left, right in pair_scores[:max_pairs])

    return FeatureLibraryPlan(
        feature_names=names,
        raw_indices=tuple(raw_indices),
        square_indices=tuple(square_indices),
        sin_indices=tuple(sin_indices),
        pair_indices=pair_indices,
    )


def evaluate_high_dim_sparse_validation(
    *,
    n_samples: int = 1200,
    n_features: int = 60,
    noise_std: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Validate sparse discovery in many irrelevant dimensions."""

    data, hidden = make_high_dim_sparse_dataset(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )
    plan = fit_screened_feature_library_plan(
        data.x_train,
        data.y_train,
        max_raw=16,
        max_square=16,
        max_sin=16,
        max_pairs=48,
    )
    result = _fit_sparse_plan_with_validation(data, plan, complexity_weight=1e-3)
    selected_names = [str(row["name"]) for row in result["selected_terms"]]
    hidden_terms = list(hidden["terms"])
    recovered_terms = [term for term in hidden_terms if term in selected_names]
    false_positive_terms = [term for term in selected_names if term not in hidden_terms]
    return {
        "hidden_law": hidden["law"],
        "hidden_terms": hidden_terms,
        "equation": result["equation"],
        "selected_terms": result["selected_terms"],
        "recovered_terms": recovered_terms,
        "recovery_rate": len(recovered_terms) / len(hidden_terms),
        "false_positive_count": len(false_positive_terms),
        "metrics": result["metrics"],
        "selection": result["selection"],
        "screened_feature_count": len(result["feature_names"]),
    }


def fit_neural_field_1d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 192,
    ridge: float = 1e-5,
    activation: str = "tanh",
    seed: int = 0,
) -> NeuralField1D:
    """Fit a smooth 1D omnibias random-feature field by solving the output layer."""

    jnp = _jax_numpy()
    from omnibias.jax import get_activation

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_mean = float(np.mean(x))
    x_scale = float(np.std(x))
    if x_scale < 1e-12:
        x_scale = 1.0
    xs = (x - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=hidden)
    beta = rng.normal(0.0, 0.8, size=hidden)
    spec = get_activation(activation)
    z = xs[:, None] * W[None, :] + beta[None, :]
    phi = np.asarray(spec.forward(jnp.asarray(z)))
    design = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    reg = ridge * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ coef
    return NeuralField1D(
        W=W,
        beta=beta,
        c=coef[:-1],
        b=float(coef[-1]),
        x_mean=x_mean,
        x_scale=x_scale,
        activation=activation,
        train_rmse=rmse(y, pred),
    )


def exact_activation_field_1d(activation: str) -> NeuralField1D:
    """Represent a single activation exactly as a one-neuron neural field."""

    return NeuralField1D(
        W=np.asarray([1.0]),
        beta=np.asarray([0.0]),
        c=np.asarray([1.0]),
        b=0.0,
        x_mean=0.0,
        x_scale=1.0,
        activation=activation,
        train_rmse=0.0,
    )


def extract_neural_jets(field: NeuralField1D, x: np.ndarray, *, max_order: int = 3) -> JetBundle:
    """Extract closed-form jets ``[y, dy, d2y, ...]`` from a 1D omnibias field."""

    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    jnp = _jax_numpy()
    from omnibias.jax import get_activation

    xs_raw = np.asarray(x, dtype=float).reshape(-1)
    xs = (xs_raw - field.x_mean) / field.x_scale
    z = xs[:, None] * field.W[None, :] + field.beta[None, :]
    spec = get_activation(field.activation)
    jets: list[np.ndarray] = []
    for order in range(max_order + 1):
        if order == 0:
            values = np.asarray(spec.forward(jnp.asarray(z))) @ field.c + field.b
        else:
            if spec.fastpath is None:
                raise TypeError(f"activation {field.activation!r} does not expose a derivative fastpath")
            sigma_n = np.asarray(spec.fastpath(jnp.asarray(z), order))
            chain = (field.W / field.x_scale) ** order
            values = sigma_n @ (field.c * chain)
        jets.append(np.asarray(values, dtype=float))
    return JetBundle(x=xs_raw, jets=np.stack(jets, axis=1))


def split_x_grid(
    *,
    xmin: float = -1.0,
    xmax: float = 1.0,
    n_train: int = 160,
    n_val: int = 120,
    n_test: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.linspace(xmin, xmax, n_train)
    val = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), n_val)
    test = np.linspace(xmin + 0.05 * (xmax - xmin), xmax - 0.05 * (xmax - xmin), n_test)
    return train, val, test


def _jet_relation_variables(
    bundle: JetBundle,
    *,
    lhs_order: int,
    include_x: bool,
    lower_order_only: bool,
) -> tuple[list[np.ndarray], list[str]]:
    """The raw degree-1 jet variables used on the right-hand side (``x``, jets)."""
    if lhs_order < 0 or lhs_order >= bundle.jets.shape[1]:
        raise ValueError(f"lhs_order {lhs_order} is outside jet width {bundle.jets.shape[1]}")
    variables: list[np.ndarray] = []
    names: list[str] = []
    if include_x:
        variables.append(bundle.x)
        names.append("x")
    for order in range(bundle.jets.shape[1]):
        if order == lhs_order:
            continue
        if lower_order_only and order > lhs_order:
            continue
        variables.append(bundle.jets[:, order])
        names.append(jet_name(order))
    return variables, names


def build_jet_relation_library(
    bundle: JetBundle,
    *,
    lhs_order: int,
    max_degree: int = 2,
    include_x: bool = True,
    lower_order_only: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build generic polynomial features from jet coordinates, excluding the lhs.

    By default only lower-order jets are allowed on the right-hand side. This
    avoids discovering derivative-shift identities such as ``dy = d2y`` when
    several jet coordinates are collinear for a special function.
    """

    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")
    variables, names = _jet_relation_variables(
        bundle,
        lhs_order=lhs_order,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )
    cols: list[np.ndarray] = []
    term_names: list[str] = []
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(len(variables)), degree):
            cols.append(np.prod([variables[index] for index in combo], axis=0))
            term_names.append("*".join(_power_name(names[index], combo.count(index)) for index in sorted(set(combo))))
    return np.stack(cols, axis=1), term_names


def _fit_jet_variable_grid(
    train: JetBundle,
    *,
    lhs_order: int,
    n_locations: int,
    scale_mults: tuple[float, ...],
    include_x: bool,
    lower_order_only: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Train-quantile / train-std location-scale grid for the jet-relation variables."""
    if n_locations < 1:
        raise ValueError(f"n_locations must be >= 1, got {n_locations}")
    if len(scale_mults) == 0 or any(m <= 0.0 for m in scale_mults):
        raise ValueError("scale_mults must be a non-empty tuple of positive floats")
    variables, _ = _jet_relation_variables(
        train,
        lhs_order=lhs_order,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )
    mults = np.asarray(scale_mults, dtype=float)
    quantiles = np.linspace(0.1, 0.9, n_locations)
    per_locs: list[np.ndarray] = []
    per_scales: list[np.ndarray] = []
    for col in variables:
        per_locs.append(np.quantile(col, quantiles))
        std = float(np.std(col))
        per_scales.append((std if std > 1e-12 else 1.0) * mults)
    return per_locs, per_scales


def _build_jet_transform_features(
    bundle: JetBundle,
    *,
    lhs_order: int,
    bases: tuple[str, ...],
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]],
    name_prefix: str,
    per_variable_locations: list[np.ndarray],
    per_variable_scales: list[np.ndarray],
    include_x: bool,
    lower_order_only: bool,
) -> tuple[np.ndarray, list[str]]:
    """Apply per-variable location-scale ``transforms`` to the jet-relation variables."""
    variables, names = _jet_relation_variables(
        bundle,
        lhs_order=lhs_order,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )
    cols: list[np.ndarray] = []
    col_names: list[str] = []
    for vi, col in enumerate(variables):
        locs = per_variable_locations[vi]
        scales = per_variable_scales[vi]
        for base in bases:
            transform = transforms[base]
            for s in scales:
                vals = transform((col[:, None] - locs[None, :]) / s)
                for li in range(locs.shape[0]):
                    cols.append(vals[:, li])
                    col_names.append(f"{name_prefix}{base}(({names[vi]}-{locs[li]:.4g})/{s:.4g})")
    if not cols:
        raise ValueError("jet transform features produced no columns")
    return np.stack(cols, axis=1), col_names


def fit_jet_cdf_plan(
    train: JetBundle,
    *,
    lhs_order: int,
    bases: tuple[str, ...] = ("sigmoid", "tanh"),
    n_locations: int = 5,
    scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0),
    include_x: bool = True,
    lower_order_only: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-variable CDF locations/scales for jet features on the train split.

    Mirrors :func:`fit_cdf_band_library_plan` for the jet-relation variables
    (``x`` and the admissible lower-order jets): locations are train quantiles
    (``n_locations`` between the 10th and 90th percentile) and scales are
    ``scale_mults`` times the train standard deviation. Returned as
    ``(per_variable_locations, per_variable_scales)`` so the same fixed grid can
    be reused across train / val / test (no leakage) by
    :func:`build_jet_cdf_features`.
    """
    for base in bases:
        if base not in _CDF_BASES:
            raise ValueError(f"unknown CDF base {base!r}; supported: {tuple(_CDF_BASES)}")
    return _fit_jet_variable_grid(
        train,
        lhs_order=lhs_order,
        n_locations=n_locations,
        scale_mults=scale_mults,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )


def build_jet_cdf_features(
    bundle: JetBundle,
    *,
    lhs_order: int,
    bases: tuple[str, ...],
    per_variable_locations: list[np.ndarray],
    per_variable_scales: list[np.ndarray],
    include_x: bool = True,
    lower_order_only: bool = True,
) -> tuple[np.ndarray, list[str]]:
    r"""CDF/band transforms of the jet-relation variables on a fixed grid.

    The probability-operator twin of :func:`build_jet_relation_library`: every
    right-hand-side jet variable ``v`` (``x``, ``y``, ``dy``, ...) is mapped
    through the monotone CDF bases ``F((v - loc)/scale)`` on the *fixed* grid from
    :func:`fit_jet_cdf_plan`. Concatenated with the polynomial design this lets
    :class:`NeuralJetDiscoverer` recover saturating / sigmoidal differential laws
    (e.g. ``dy = sigma((x - loc)/scale)``) that no finite polynomial captures.
    """
    return _build_jet_transform_features(
        bundle,
        lhs_order=lhs_order,
        bases=bases,
        transforms=_CDF_BASES,
        name_prefix="",
        per_variable_locations=per_variable_locations,
        per_variable_scales=per_variable_scales,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )


def fit_jet_info_plan(
    train: JetBundle,
    *,
    lhs_order: int,
    bases: tuple[str, ...] = ("sigmoid", "arctan"),
    n_locations: int = 5,
    scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0),
    include_x: bool = True,
    lower_order_only: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-variable surprisal locations/scales for jet features (train only).

    The information-theoretic twin of :func:`fit_jet_cdf_plan`: an identical,
    leakage-free train-quantile / train-std grid over the jet-relation variables,
    validated against the surprisal bases. Reuse the returned grid across splits
    with :func:`build_jet_info_features`.
    """
    for base in bases:
        if base not in _INFO_BASES:
            raise ValueError(
                f"unknown information base {base!r}; supported: {tuple(_INFO_BASES)}"
            )
    return _fit_jet_variable_grid(
        train,
        lhs_order=lhs_order,
        n_locations=n_locations,
        scale_mults=scale_mults,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )


def build_jet_info_features(
    bundle: JetBundle,
    *,
    lhs_order: int,
    bases: tuple[str, ...],
    per_variable_locations: list[np.ndarray],
    per_variable_scales: list[np.ndarray],
    include_x: bool = True,
    lower_order_only: bool = True,
) -> tuple[np.ndarray, list[str]]:
    r"""Self-information (surprisal) transforms of the jet-relation variables.

    The information-theoretic twin of :func:`build_jet_cdf_features`: every
    right-hand-side jet variable ``v`` is mapped through the surprisal
    ``-ln f((v - loc)/scale)`` of each base density (see
    :func:`build_information_library`) on the *fixed* grid from
    :func:`fit_jet_info_plan`. Concatenated with the polynomial design this lets
    :class:`NeuralJetDiscoverer` recover differential laws with log-likelihood /
    energy structure that neither the polynomial nor the CDF jet library captures.
    """
    return _build_jet_transform_features(
        bundle,
        lhs_order=lhs_order,
        bases=bases,
        transforms=_INFO_BASES,
        name_prefix="surprisal_",
        per_variable_locations=per_variable_locations,
        per_variable_scales=per_variable_scales,
        include_x=include_x,
        lower_order_only=lower_order_only,
    )


def _gl_weights(alpha: float, n: int) -> np.ndarray:
    r"""Grunwald-Letnikov weights ``w_k = (-1)^k binom(alpha, k)``, ``k = 0..n-1``.

    Built by the stable recurrence ``w_0 = 1``, ``w_k = w_{k-1} (1 - (alpha+1)/k)``.
    The numpy twin of :func:`omnibias.fractional._core.kernels.gl_weights`
    (parity-tested), kept here so the symbolic engine has no hard dependency on the
    fractional-calculus package.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    w = np.empty(n, dtype=float)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
    return w


def gl_fractional_derivative(y: np.ndarray, *, alpha: float, h: float) -> np.ndarray:
    r"""Grunwald-Letnikov fractional derivative ``D^alpha y`` on a uniform grid.

    Causal discretisation ``D^alpha y[i] = h^{-alpha} sum_{k=0}^{i} w_k y[i-k]`` of
    the (Riemann-Liouville) fractional derivative of order ``alpha >= 0`` with grid
    spacing ``h`` -- a *non-local* operator (each output depends on the whole
    history), evaluated as a causal convolution with the
    :func:`_gl_weights`. Integer orders reduce to the backward finite differences
    (``alpha = 0`` is the identity, ``alpha = 1`` the backward difference, ...), so
    this smoothly interpolates between derivative orders. Left-edge values carry
    the usual GL truncation bias (few history terms).
    """
    ya = np.asarray(y, dtype=float).reshape(-1)
    n = ya.size
    if n < 1:
        raise ValueError("gl_fractional_derivative needs at least one sample")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if h <= 0.0:
        raise ValueError(f"grid spacing h must be > 0, got {h}")
    w = _gl_weights(alpha, n)
    conv = np.convolve(ya, w)[:n]
    return np.asarray(conv * (h ** (-alpha)), dtype=float)


def _uniform_grid_spacing(x: np.ndarray, *, rtol: float = 1e-4) -> float:
    """Spacing ``h`` of a strictly increasing, (near-)uniform 1-D grid (else raise)."""
    xs = np.asarray(x, dtype=float).reshape(-1)
    if xs.size < 2:
        raise ValueError("fractional features need at least two grid points")
    diffs = np.diff(xs)
    if np.any(diffs <= 0.0):
        raise ValueError(
            "fractional features need a strictly increasing x grid (sort the bundle)"
        )
    h = float(diffs.mean())
    if float(diffs.std()) > rtol * max(abs(h), 1.0):
        raise ValueError(
            "fractional features require a (near-)uniform x grid; "
            f"spacing varies by {float(diffs.std()):.3g} (h={h:.3g})"
        )
    return h


def build_jet_fractional_features(
    bundle: JetBundle,
    *,
    orders: tuple[float, ...],
    source_order: int = 0,
) -> tuple[np.ndarray, list[str]]:
    r"""Fractional-derivative columns ``D^alpha`` of a jet signal over the x grid.

    The non-local / fractional-calculus twin of :func:`build_jet_cdf_features`:
    every order ``alpha`` in ``orders`` adds a column ``D^alpha v`` where ``v`` is
    the ``source_order`` jet (``y`` by default) differentiated along the bundle's
    (uniform, sorted) ``x`` grid via :func:`gl_fractional_derivative`. Concatenated
    with the polynomial / CDF / info design this lets :class:`NeuralJetDiscoverer`
    recover **fractional differential laws** (e.g. fractional relaxation
    ``dy = -c D^{0.5} y``) that no integer-order jet library can express.

    ``x`` must be a strictly increasing, near-uniform grid (as produced by
    :func:`split_x_grid`); the closed-form omnibias jets supply ``v`` exactly, so
    only the fractional operator itself is discretised.
    """
    if len(orders) == 0:
        raise ValueError("build_jet_fractional_features needs at least one order")
    if source_order < 0 or source_order >= bundle.jets.shape[1]:
        raise ValueError(
            f"source_order {source_order} is outside jet width {bundle.jets.shape[1]}"
        )
    h = _uniform_grid_spacing(bundle.x)
    signal = bundle.jets[:, source_order]
    cols: list[np.ndarray] = []
    names: list[str] = []
    for alpha in orders:
        cols.append(gl_fractional_derivative(signal, alpha=alpha, h=h))
        names.append(f"D^{alpha:g}({jet_name(source_order)})")
    return np.stack(cols, axis=1), names


def build_jet_fractional_features_closed_form(
    bundle: JetBundle,
    *,
    orders: tuple[float, ...],
    source_order: int = 0,
    terminal: float | None = None,
    kind: str = "riemann_liouville",
    tower_width: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    r"""Closed-form (analytic-class) fractional-derivative columns from the jet tower.

    The closed-form twin of :func:`build_jet_fractional_features`: instead of the
    grid Grunwald-Letnikov convolution, every order ``alpha`` in ``orders`` adds a
    column ``D^alpha v`` evaluated by
    :func:`omnibias.fractional.jax.ops.analytic.fractional_derivative` -- the
    single vectorised gamma-ratio Taylor-jet sum, no grid and no history. ``v`` is
    the ``source_order`` jet, whose Taylor coefficients about the lower terminal
    ``a`` come from the derivative tower ``bundle.jets[j0]`` at the terminal
    (``a = bundle.x[0]`` by default, else the nearest grid sample). Exact for
    signals that are polynomials of degree ``<= (tower_width - 1 - source_order)``.

    ``tower_width`` is the number of leading jet columns that form ``v``'s
    derivative tower (``jets[j0, source_order:tower_width]``); it defaults to the
    full jet width. Set it to exclude appended target / LHS columns that are not
    part of the tower.

    ``kind="riemann_liouville"`` is singular at the terminal (``t = 0``); use
    ``kind="caputo"`` (regular there) when the grid includes the terminal -- the
    default of :func:`discover_fractional_order_law`. Requires the
    ``omnibias-fractional`` package (imported lazily).
    """
    if len(orders) == 0:
        raise ValueError(
            "build_jet_fractional_features_closed_form needs at least one order"
        )
    width = bundle.jets.shape[1]
    tw = width if tower_width is None else int(tower_width)
    if not (1 <= tw <= width):
        raise ValueError(
            f"tower_width {tw} must be in 1..{width} (the jet width)"
        )
    if source_order < 0 or source_order >= tw:
        raise ValueError(
            f"source_order {source_order} is outside the tower width {tw}"
        )
    x = np.asarray(bundle.x, dtype=float).reshape(-1)
    if x.size < 1:
        raise ValueError("closed-form fractional features need at least one sample")
    if x.size > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError(
            "closed-form fractional features need a strictly increasing x grid "
            "(sort the bundle)"
        )
    j0 = 0 if terminal is None else int(np.argmin(np.abs(x - float(terminal))))
    a = float(x[j0])
    # Taylor coefficients of v = f^(source_order) about the terminal a:
    # v^(m)(a)/m! = f^(source_order+m)(a)/m! = jets[j0, source_order+m] / m!.
    tower = np.asarray(bundle.jets[j0], dtype=float)
    coeffs = np.array(
        [tower[source_order + m] / math.factorial(m) for m in range(tw - source_order)],
        dtype=float,
    )

    try:
        from omnibias.fractional.jax.ops.analytic import (
            fractional_derivative as _closed_form_frac,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "build_jet_fractional_features_closed_form needs the omnibias-fractional "
            "package (closed-form analytic op); install 'omnibias-fractional'."
        ) from exc

    jnp = _jax_numpy()
    jet_j = jnp.asarray(coeffs, dtype=jnp.float64)
    x_j = jnp.asarray(x, dtype=jnp.float64)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for alpha in orders:
        col = np.asarray(
            _closed_form_frac(jet_j, x_j, alpha=float(alpha), a=a, kind=kind),
            dtype=float,
        )
        cols.append(col)
        names.append(f"D^{alpha:g}({jet_name(source_order)})")
    return np.stack(cols, axis=1), names


def build_jet_piecewise_fractional_features(
    bundle: JetBundle,
    *,
    orders: tuple[float, ...],
    terminals: Sequence[float],
    source_order: int = 0,
    kind: str = "caputo",
    blend: float = 0.0,
    gap: float = 1e-6,
    tower_width: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    r"""Multi-terminal (piecewise) closed-form fractional-derivative columns.

    :func:`build_jet_fractional_features_closed_form` expands the signal as *one*
    Taylor jet about a single terminal, which is exact only while the jet's radius
    of convergence covers the whole grid. This builder instead supplies a jet per
    terminal and lets
    :func:`omnibias.fractional.jax.ops.analytic.piecewise_fractional_derivative`
    select (``blend=0``) or smoothly blend (``blend>0``) between the patches, so a
    long or strongly varying grid is covered by several local expansions rather
    than one global one.

    Each terminal is **snapped to its nearest grid sample**, and the tower is read
    there. The snap is what keeps the operator self-consistent: the jet coefficients
    describe an expansion about the sample they were read at, so handing the
    operator a slightly different terminal would silently evaluate the wrong series
    (the same reason
    :func:`build_jet_fractional_features_closed_form` reports ``a = x[j0]`` rather
    than the requested terminal). Distinct terminals must therefore snap to distinct
    samples, and the lowest must snap to ``x[0]`` -- the operator is undefined to the
    left of it.

    Because the snap is per bundle, a terminal means "the sample nearest here on
    *this* grid". Train / validation / test splits on interleaved grids each anchor
    to their own nearest sample, which is the same convention the single-terminal
    builder uses with ``terminal=None``.

    Honesty label: **closed form**. The fractional operator is the analytic
    gamma-ratio jet sum, not a grid convolution, so there is no discretisation
    error in ``alpha`` -- only the truncation of each Taylor tower. With a single
    terminal this reduces *exactly* to
    :func:`build_jet_fractional_features_closed_form` (gated in the tests).

    Requires the ``omnibias-fractional`` package (imported lazily).
    """
    if len(orders) == 0:
        raise ValueError("build_jet_piecewise_fractional_features needs at least one order")
    terms = np.asarray(list(terminals), dtype=float).reshape(-1)
    if terms.size == 0:
        raise ValueError("build_jet_piecewise_fractional_features needs at least one terminal")
    if terms.size > 1 and np.any(np.diff(terms) <= 0.0):
        raise ValueError(
            f"terminals must be strictly increasing, got {terms.tolist()}"
        )
    width = bundle.jets.shape[1]
    tw = width if tower_width is None else int(tower_width)
    if not (1 <= tw <= width):
        raise ValueError(f"tower_width {tw} must be in 1..{width} (the jet width)")
    if source_order < 0 or source_order >= tw:
        raise ValueError(f"source_order {source_order} is outside the tower width {tw}")
    x = np.asarray(bundle.x, dtype=float).reshape(-1)
    if x.size < 1:
        raise ValueError("piecewise fractional features need at least one sample")
    if x.size > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError(
            "piecewise fractional features need a strictly increasing x grid "
            "(sort the bundle)"
        )
    # Snap to the grid: the tower is read at a sample, so the terminal handed to the
    # operator must be that same sample or the series is expanded about the wrong
    # point.
    indices = [int(np.argmin(np.abs(x - float(a)))) for a in terms]
    if indices[0] != 0:
        raise ValueError(
            f"the lowest terminal {float(terms[0]):g} snaps to x[{indices[0]}]="
            f"{float(x[indices[0]]):g}, above the first grid point {float(x[0]):g}; "
            "the operator is undefined to the left of the lowest terminal"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"terminals {terms.tolist()} snap to duplicate grid samples "
            f"{indices}; space them at least one grid step apart"
        )
    snapped = x[indices]

    try:
        from omnibias.fractional.jax.ops.analytic import (
            piecewise_fractional_derivative as _piecewise_frac,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "build_jet_piecewise_fractional_features needs the omnibias-fractional "
            "package (closed-form analytic op); install 'omnibias-fractional'."
        ) from exc

    # One Taylor tower per terminal. Coefficient m of v = f^(source_order) about a
    # is f^(source_order+m)(a)/m!.
    n_coeff = tw - source_order
    jets = np.empty((len(indices), n_coeff), dtype=float)
    for i, j0 in enumerate(indices):
        tower = np.asarray(bundle.jets[j0], dtype=float)
        jets[i] = [
            tower[source_order + m] / math.factorial(m) for m in range(n_coeff)
        ]

    jnp = _jax_numpy()
    jets_j = jnp.asarray(jets, dtype=jnp.float64)
    terms_j = jnp.asarray(snapped, dtype=jnp.float64)
    x_j = jnp.asarray(x, dtype=jnp.float64)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for alpha in orders:
        col = np.asarray(
            _piecewise_frac(
                jets_j,
                terms_j,
                x_j,
                alpha=float(alpha),
                kind=kind,
                blend=blend,
                gap=gap,
            ),
            dtype=float,
        )
        cols.append(col)
        names.append(f"Dpw^{alpha:g}({jet_name(source_order)})")
    return np.stack(cols, axis=1), names


def build_jet_activation_fractional_features(
    bundle: JetBundle,
    *,
    orders: tuple[float, ...],
    name: str,
    kind: str = "riemann_liouville",
    activation_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[str]]:
    r"""Closed-form fractional derivatives of a *named activation* over the x grid.

    Where the other fractional builders differentiate the *sampled signal*, this one
    differentiates the analytic activation itself, evaluated on ``bundle.x``. That
    makes the column an exact special-function value rather than any kind of
    approximation, which is what lets a discovery run test the hypothesis "the law
    involves ``D^alpha exp``" without ever discretising.

    ``name`` must be a key of
    :data:`~omnibias.fractional.torch.ops.activation.ACTIVATION_FRACTIONAL`
    (currently ``"exp"``, ``"cosh"``, ``"sinh"``; the logistic ``sigmoid`` is
    deliberately absent, having been dropped for numerical stability). An
    unregistered name raises with the available set rather than silently falling
    back to a numerical operator.

    ``activation_kwargs`` forwards identity-specific options (``lam`` for ``exp``,
    ``terms`` for the series-based identities).

    Honesty label: **closed form** -- these are the analytic
    Mittag-Leffler / hyperbolic identities, exact at every ``alpha``.

    The columns do not depend on ``bundle.jets`` at all, only on ``bundle.x``, so
    unlike the other builders there is no ``source_order`` and no interaction with
    the LHS-order guard.

    Requires the ``omnibias-fractional`` package (imported lazily).
    """
    if len(orders) == 0:
        raise ValueError(
            "build_jet_activation_fractional_features needs at least one order"
        )
    x = np.asarray(bundle.x, dtype=float).reshape(-1)
    if x.size < 1:
        raise ValueError("activation fractional features need at least one sample")

    try:
        from omnibias.fractional.jax.ops.activation import (
            ACTIVATION_FRACTIONAL,
            activation_fractional_derivative,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "build_jet_activation_fractional_features needs the omnibias-fractional "
            "package (closed-form activation identities); install "
            "'omnibias-fractional'."
        ) from exc

    # Validated here rather than left to the dispatcher so the error names this
    # builder and the caller learns the whole available set at once.
    if name not in ACTIVATION_FRACTIONAL:
        raise ValueError(
            f"no closed-form fractional derivative registered for activation "
            f"{name!r}; available: {sorted(ACTIVATION_FRACTIONAL)}"
        )

    jnp = _jax_numpy()
    x_j = jnp.asarray(x, dtype=jnp.float64)
    kwargs = dict(activation_kwargs or {})
    cols: list[np.ndarray] = []
    names: list[str] = []
    for alpha in orders:
        col = np.asarray(
            activation_fractional_derivative(
                name, x_j, alpha=float(alpha), kind=kind, **kwargs
            ),
            dtype=float,
        )
        cols.append(col)
        names.append(f"D^{alpha:g}({name}(x))")
    return np.stack(cols, axis=1), names


def build_jet_spectral_fractional_features(
    bundle: JetBundle,
    *,
    orders: tuple[float, ...],
    source_order: int = 0,
    bc: str = "dirichlet",
    length: float | None = None,
    windowed: bool = False,
    taper: float = 0.1,
) -> tuple[np.ndarray, list[str]]:
    r"""Spectral fractional columns of a jet signal on a bounded, non-periodic grid.

    Two operators, selected by ``windowed``:

    * ``windowed=False`` (default) -- the two-sided spectral fractional Laplacian
      ``(-Delta)^{alpha/2}`` via an orthonormal DST-I (``bc="dirichlet"``) or DCT-II
      (``bc="neumann"``) transform. Exact on the basis modes, so at ``alpha=2`` it
      reproduces the integer-order Laplacian on an eigenmode to round-off.
    * ``windowed=True`` -- the Tukey-windowed periodic operator, for a signal that
      merely decays toward the ends rather than satisfying a boundary condition.

    **Grid requirement.** The transform's basis is tied to a specific grid, and using
    the wrong one silently returns a plausible but wrong column, so it is checked:
    ``bc="dirichlet"`` needs the interior grid ``x_j = (j+1) L/(N+1)`` and
    ``bc="neumann"`` the midpoint grid ``x_j = (j+1/2) L/N``. ``length`` defaults to
    the ``L`` implied by the observed uniform spacing, which is the value those two
    layouts pin down; pass it explicitly to override.

    Honesty label: **numerical-spectral**, not closed form. The operator is exact on
    the truncated basis, so the error is the spectral truncation of the signal --
    much smaller than a Grunwald-Letnikov convolution at the same ``N``, but not
    zero, and it degrades for a signal the basis represents badly.

    The windowed path's operator returns a complex array; the column is its **real
    part**. That is the physically meaningful half for a real signal, and the
    imaginary part is a windowing artefact rather than a second feature, so it is
    dropped rather than reported.

    Requires the ``omnibias-fractional`` package (imported lazily).
    """
    if len(orders) == 0:
        raise ValueError("build_jet_spectral_fractional_features needs at least one order")
    if source_order < 0 or source_order >= bundle.jets.shape[1]:
        raise ValueError(
            f"source_order {source_order} is outside jet width {bundle.jets.shape[1]}"
        )
    if bc not in ("dirichlet", "neumann"):
        raise ValueError(f"bc must be 'dirichlet' or 'neumann', got {bc!r}")
    h = _uniform_grid_spacing(bundle.x)
    x = np.asarray(bundle.x, dtype=float).reshape(-1)
    n = x.size
    # The two layouts pin L down from the spacing: Dirichlet has N+1 gaps across
    # [0, L], Neumann exactly N.
    implied = h * (n + 1) if bc == "dirichlet" else h * n
    lam = implied if length is None else float(length)
    if lam <= 0.0:
        raise ValueError(f"length must be > 0, got {lam}")
    if not windowed:
        expected = (
            (np.arange(1, n + 1) * lam / (n + 1))
            if bc == "dirichlet"
            else ((np.arange(n) + 0.5) * lam / n)
        )
        if not np.allclose(x, expected, rtol=1e-6, atol=1e-8 * max(lam, 1.0)):
            layout = (
                "x_j = (j+1) L/(N+1)" if bc == "dirichlet" else "x_j = (j+1/2) L/N"
            )
            raise ValueError(
                f"bc={bc!r} requires the {layout} grid on [0, L] with L={lam:g}; the "
                f"bundle's grid runs {x[0]:g}..{x[-1]:g} (expected "
                f"{expected[0]:g}..{expected[-1]:g})"
            )

    try:
        if windowed:
            from omnibias.fractional.jax.ops.spectral import (
                windowed_spectral_fractional as _spectral_op,
            )
        else:
            from omnibias.fractional.jax.ops.spectral import (
                spectral_fractional_laplacian as _spectral_op,  # type: ignore[assignment]
            )
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "build_jet_spectral_fractional_features needs the omnibias-fractional "
            "package (spectral ops); install 'omnibias-fractional'."
        ) from exc

    jnp = _jax_numpy()
    signal = jnp.asarray(bundle.jets[:, source_order], dtype=jnp.float64)
    label = "Dwin" if windowed else "Dspec"
    cols: list[np.ndarray] = []
    names: list[str] = []
    for alpha in orders:
        raw = (
            _spectral_op(signal, alpha=float(alpha), length=lam, taper=taper)
            if windowed
            else _spectral_op(signal, alpha=float(alpha), length=lam, bc=bc)
        )
        cols.append(np.asarray(np.real(np.asarray(raw)), dtype=float))
        names.append(f"{label}^{alpha:g}({jet_name(source_order)})")
    return np.stack(cols, axis=1), names


#: A kernel ``K(x, t)`` for an integral column, called with a broadcast pair of
#: shape ``(n, n)`` (query grid down the rows, quadrature nodes across the
#: columns) and returning the same shape. Written this way so a kernel is one
#: vectorised numpy expression rather than a loop.
IntegralKernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _cumulative_trapezoid_matrix(x: np.ndarray) -> np.ndarray:
    r"""Weights ``C`` with ``(C @ v)_i = int_{x_0}^{x_i} v`` by the trapezoid rule.

    The causal counterpart of a quadrature weight vector: row ``i`` holds the
    composite-trapezoid weights for the prefix ``[x_0, x_i]``, so a single
    matrix-vector product gives the whole running integral. Exact for a piecewise
    linear integrand, second order otherwise.

    A *global* rule's weights cannot be reused for this. Gauss-Legendre weights,
    for instance, are coefficients of a rule for the whole interval, not the
    measure of a neighbourhood of each node, so masking them to a prefix does not
    integrate the prefix. That is why the causal families here build their own
    weights from the grid rather than slicing the measure's.
    """
    n = x.size
    c = np.zeros((n, n), dtype=float)
    h = np.diff(x)
    for i in range(1, n):
        c[i, :i] = c[i - 1, :i]
        c[i, i - 1] += 0.5 * h[i - 1]
        c[i, i] += 0.5 * h[i - 1]
    return c


def build_jet_integral_features(
    bundle: JetBundle,
    *,
    measure: object | None = None,
    kernels: Mapping[str, IntegralKernel] | None = None,
    volterra_kernels: Mapping[str, IntegralKernel] | None = None,
    running: bool = True,
    source_order: int = 0,
    origin: float | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> tuple[np.ndarray, list[str]]:
    r"""Integral (non-local) columns of a jet signal: running, Fredholm, Volterra.

    Every other library in this module is *local*: each row of a derivative or
    fractional column is determined by the signal near that point. These columns
    are not. They let a discovery run express an **integro-differential** law such
    as ``y'(x) = -int_0^x y`` or ``y(x) = f(x) + lam int_a^b K(x,t) y(t) dt``,
    which no local library can represent at all.

    Three families, all built from the ``source_order`` jet ``v`` (``y`` by
    default) sampled on the bundle's own grid:

    * ``running=True`` adds ``I(v)(x_i) = int_{a}^{x_i} v`` -- the indefinite
      integral, the natural partner of the derivative columns.
    * each ``name -> K`` in ``kernels`` adds the **Fredholm** column
      ``int_a^b K(x_i, t) v(t) dt`` over the whole domain.
    * each ``name -> K`` in ``volterra_kernels`` adds the **causal Volterra**
      column ``int_{a}^{x_i} K(x_i, t) v(t) dt``.

    ``origin`` is the lower terminal ``a`` of the two causal families, snapped to
    the nearest sample (the same convention as the piecewise fractional builder,
    and for the same reason: the quadrature can only start at a point where the
    signal is known). It defaults to the bundle's **first sample**, which is worth
    stating plainly because it is the one way these columns differ from every
    local library here: an indefinite integral is only defined up to a constant,
    and that constant is set by where the grid starts. Two bundles that begin at
    different ``x`` therefore carry *different* ``I(v)`` columns, so a law fitted
    on one will not transfer to the other. Pass an explicit ``origin`` shared by
    every split -- or give every split the same starting point -- whenever the
    columns must be comparable across bundles.

    ``measure`` supplies the global quadrature weights and is required only for
    the Fredholm family. Its nodes must **equal the bundle's grid**: the signal is
    known at the bundle's samples and nowhere else, so a measure on different
    nodes could only be honoured by interpolating, which would quietly make the
    column an approximation of an approximation. Mismatched nodes therefore raise.
    Pass either an :class:`omnibias.measure.Measure` directly, or -- when bundles
    differ in resolution, as train / validation / test splits normally do -- a
    factory ``x -> Measure`` that builds the rule on whichever grid it is handed.
    The factory form is what makes a high-order rule usable here: 24 Gauss-Legendre
    nodes evaluate a smooth Fredholm column to round-off, where a 401-point
    trapezoid rule reaches only ``1e-6``.

    Honesty label: **numerical**. Every column is a quadrature. The Fredholm
    family inherits the accuracy of whatever rule the measure carries (spectral
    for Gauss-Legendre, second order for a uniform composite rule); the two causal
    families are cumulative trapezoid, so second order, because a global rule's
    weights cannot be restricted to a prefix (see
    :func:`_cumulative_trapezoid_matrix`).

    These columns are *global and badly scaled* next to derivative columns -- an
    integral grows with the domain while a derivative does not -- which is why
    :meth:`NeuralJetDiscoverer.discover` reports a design-matrix condition number
    alongside the fit. :func:`fit_sparse_equation` thresholds in standardized
    space, so the scale mismatch does not by itself change term selection, but
    near-collinearity between a running integral and a smooth low-order column
    still can, and the diagnostic is what makes that visible rather than assumed.

    Requires the ``omnibias-measure`` package when ``kernels`` is given (imported
    lazily; it is a test-only extra, never a hard dependency).
    """
    kernels = dict(kernels or {})
    volterra_kernels = dict(volterra_kernels or {})
    if not running and not kernels and not volterra_kernels:
        raise ValueError(
            "build_jet_integral_features needs at least one column: set running=True "
            "or pass kernels / volterra_kernels"
        )
    width = bundle.jets.shape[1]
    if source_order < 0 or source_order >= width:
        raise ValueError(f"source_order {source_order} is outside jet width {width}")
    x = np.asarray(bundle.x, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError("integral features need at least two grid points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(
            "integral features need a strictly increasing x grid (sort the bundle)"
        )
    overlap = sorted(set(kernels) & set(volterra_kernels))
    if overlap:
        raise ValueError(
            f"kernel names {overlap} appear in both kernels and volterra_kernels; "
            "the two families would produce identically named columns"
        )

    v = np.asarray(bundle.jets[:, source_order], dtype=float)
    src = jet_name(source_order)
    cols: list[np.ndarray] = []
    names: list[str] = []

    if running or volterra_kernels:
        cum = _cumulative_trapezoid_matrix(x)
        if origin is not None:
            if not (x[0] - atol <= origin <= x[-1] + atol):
                raise ValueError(
                    f"origin {origin:g} is outside the grid [{x[0]:g}, {x[-1]:g}]; the "
                    "causal integral can only start where the signal is sampled"
                )
            # Re-base rather than re-integrate: shifting the terminal shifts the
            # whole column by a constant, and below the origin it goes negative.
            cum = cum - cum[int(np.argmin(np.abs(x - origin)))][None, :]
    if running:
        cols.append(cum @ v)
        names.append(f"I({src})")

    if kernels:
        weights = _measure_weights_on_grid(measure, x, atol=atol, rtol=rtol)
        # (n, n): query grid down the rows, quadrature nodes across the columns.
        xi, tj = np.meshgrid(x, x, indexing="ij")
        for name in kernels:
            k = np.asarray(kernels[name](xi, tj), dtype=float)
            if k.shape != (x.size, x.size):
                raise ValueError(
                    f"kernel {name!r} returned shape {k.shape}, expected "
                    f"{(x.size, x.size)}; it must broadcast over the (x, t) pair"
                )
            cols.append((k * weights[None, :]) @ v)
            names.append(f"F[{name}]({src})")

    if volterra_kernels:
        xi, tj = np.meshgrid(x, x, indexing="ij")
        for name in volterra_kernels:
            k = np.asarray(volterra_kernels[name](xi, tj), dtype=float)
            if k.shape != (x.size, x.size):
                raise ValueError(
                    f"kernel {name!r} returned shape {k.shape}, expected "
                    f"{(x.size, x.size)}; it must broadcast over the (x, t) pair"
                )
            # Causality is carried by ``cum``, which is lower triangular, so the
            # kernel is never evaluated as if t > x contributed.
            cols.append((k * cum) @ v)
            names.append(f"V[{name}]({src})")

    return np.stack(cols, axis=1), names


def _measure_weights_on_grid(
    measure: object | None, x: np.ndarray, *, atol: float, rtol: float
) -> np.ndarray:
    """The measure's quadrature weights, after checking its nodes are the grid."""
    if measure is None:
        raise ValueError(
            "build_jet_integral_features needs a measure to build Fredholm columns "
            "(the global quadrature weights come from it); pass one, or use "
            "volterra_kernels for the causal family, which builds its own"
        )
    try:  # pragma: no cover -- exercised, but the import guard itself is not
        from omnibias.measure import Measure
    except ModuleNotFoundError as exc:  # pragma: no cover -- optional dependency
        raise ModuleNotFoundError(
            "build_jet_integral_features needs the omnibias-measure package for "
            "Fredholm columns; install 'omnibias-measure'."
        ) from exc
    if not isinstance(measure, Measure) and callable(measure):
        measure = measure(x)
    if not isinstance(measure, Measure):
        raise TypeError(
            f"measure must be an omnibias.measure.Measure, got {type(measure).__name__}"
        )
    if measure.dim != 1:
        raise ValueError(
            f"integral features are 1-D; got a {measure.dim}-D measure. Build the "
            "kernel domain inside the kernel callable rather than as a product measure"
        )
    nodes = np.asarray(measure.nodes, dtype=float).reshape(-1)
    if nodes.shape != x.shape or not np.allclose(nodes, x, rtol=rtol, atol=atol):
        raise ValueError(
            f"the measure's {nodes.size} nodes must be the bundle's {x.size}-point "
            "grid: the signal is only known at the bundle's samples, so a measure on "
            "other nodes could only be used by interpolating. Build the measure on "
            "bundle.x (or the bundle on the measure's nodes)"
        )
    return np.asarray(measure.weights, dtype=float)


def discover_fractional_order_law(
    train: JetBundle,
    val: JetBundle,
    test: JetBundle,
    *,
    candidate_orders: Sequence[float],
    fractional_source_order: int = 0,
    terminal: float | None = None,
    kind: str = "caputo",
    tower_width: int | None = None,
    max_library_degree: int = 1,
    candidate_lhs_orders: tuple[int, ...] | None = None,
    ridge_alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4),
    thresholds: tuple[float, ...] = (1e-7, 1e-5, 1e-4, 1e-3),
    complexity_weight: float = 2e-3,
    order_penalty: float = 0.0,
) -> FractionalOrderDiscoveryResult:
    r"""Discover a fractional differential law, *recovering the fractional order*.

    Searches the discrete ``candidate_orders`` for the fractional order ``alpha``
    that best explains the data. For each candidate it adds the closed-form
    ``D^alpha`` column (:func:`build_jet_fractional_features_closed_form`) to the
    polynomial jet library and fits a sparse relation by STLSQ
    (:func:`fit_sparse_equation`), scoring by validation RMSE + complexity
    (+ ``order_penalty`` per active fractional term). Returns the best
    :class:`FractionalOrderDiscoveryResult`; its
    :attr:`~FractionalOrderDiscoveryResult.fractional_order` is the recovered order.

    The LHS is a jet column selected from ``candidate_lhs_orders`` (default: every
    order ``>= 1``); pin it to the observed target column for a fractional law
    whose LHS is not itself a derivative. The closed-form fractional feature is
    exact for polynomial signals, so on clean polynomial data the recovered order
    and coefficients are exact; on a fitted neural field it is the analytic-class
    (Taylor-jet) fractional derivative about the terminal.
    """
    orders = tuple(float(o) for o in candidate_orders)
    if not orders:
        raise ValueError(
            "discover_fractional_order_law needs at least one candidate order"
        )
    if candidate_lhs_orders is None:
        candidate_lhs_orders = tuple(range(1, train.jets.shape[1]))

    best: FractionalOrderDiscoveryResult | None = None
    for frac_alpha in orders:
        frac_kw = {
            "orders": (frac_alpha,),
            "source_order": fractional_source_order,
            "terminal": terminal,
            "kind": kind,
            "tower_width": tower_width,
        }
        tr_frac, frac_names = build_jet_fractional_features_closed_form(train, **frac_kw)
        va_frac, _ = build_jet_fractional_features_closed_form(val, **frac_kw)
        te_frac, _ = build_jet_fractional_features_closed_form(test, **frac_kw)
        frac_set = set(frac_names)
        for lhs_order in candidate_lhs_orders:
            if fractional_source_order == lhs_order:
                continue  # never inject the LHS as its own fractional feature
            train_design, names = build_jet_relation_library(
                train, lhs_order=lhs_order, max_degree=max_library_degree
            )
            val_design, _ = build_jet_relation_library(
                val, lhs_order=lhs_order, max_degree=max_library_degree
            )
            test_design, _ = build_jet_relation_library(
                test, lhs_order=lhs_order, max_degree=max_library_degree
            )
            train_design = np.concatenate([train_design, tr_frac], axis=1)
            val_design = np.concatenate([val_design, va_frac], axis=1)
            test_design = np.concatenate([test_design, te_frac], axis=1)
            names = names + frac_names
            target_train = train.jets[:, lhs_order]
            target_val = val.jets[:, lhs_order]
            target_test = test.jets[:, lhs_order]
            scale = float(np.std(target_val))
            if scale < 1e-12:
                scale = 1.0
            for ridge in ridge_alphas:
                for threshold in thresholds:
                    equation = fit_sparse_equation(
                        train_design,
                        target_train,
                        names,
                        alpha=ridge,
                        threshold=threshold,
                    )
                    val_pred = equation.predict(val_design)
                    val_rmse = rmse(target_val, val_pred)
                    active_count = len(equation.active_terms())
                    frac_active = sum(
                        1
                        for tname, coef in zip(
                            equation.term_names, equation.coefficients, strict=False
                        )
                        if tname in frac_set and abs(float(coef)) > 0.0
                    )
                    score = (
                        val_rmse / scale
                        + complexity_weight * active_count
                        + order_penalty * frac_active
                    )
                    test_pred = equation.predict(test_design)
                    result = FractionalOrderDiscoveryResult(
                        fractional_order=frac_alpha,
                        lhs_order=lhs_order,
                        equation=equation,
                        validation_rmse=val_rmse,
                        test_rmse=rmse(target_test, test_pred),
                        selection_score=score,
                        target_scale=scale,
                        candidate_orders=orders,
                    )
                    if best is None or result.selection_score < best.selection_score:
                        best = result
    if best is None:
        raise RuntimeError(
            "discover_fractional_order_law produced no candidates; "
            "check candidate_orders / fractional_orders / alphas / thresholds"
        )
    return best


def discover_activation_identity(
    activation: str,
    *,
    x_range: tuple[float, float] = (-1.0, 1.0),
    max_order: int = 3,
    candidate_lhs_orders: tuple[int, ...] | None = None,
) -> JetDiscoveryResult:
    """Discover an implicit differential identity for an exactly represented activation."""

    field = exact_activation_field_1d(activation)
    x_train, x_val, x_test = split_x_grid(xmin=x_range[0], xmax=x_range[1])
    train = extract_neural_jets(field, x_train, max_order=max_order)
    val = extract_neural_jets(field, x_val, max_order=max_order)
    test = extract_neural_jets(field, x_test, max_order=max_order)
    return NeuralJetDiscoverer().discover(
        train,
        val,
        test,
        candidate_lhs_orders=candidate_lhs_orders,
    )


def discover_from_noisy_observations(
    *,
    seed: int = 0,
    noise_std: float = 0.01,
    hidden: int = 256,
) -> dict[str, object]:
    """Fit a neural field to noisy ``sin(x)`` observations, then discover a jet law."""

    rng = np.random.default_rng(seed)
    x_train, x_val, x_test = split_x_grid(xmin=-math.pi, xmax=math.pi, n_train=220, n_val=140, n_test=140)
    y_train = np.sin(x_train) + rng.normal(0.0, noise_std, size=x_train.shape)
    field = fit_neural_field_1d(
        x_train,
        y_train,
        hidden=hidden,
        ridge=1e-4,
        activation="tanh",
        seed=seed,
    )
    train = extract_neural_jets(field, x_train, max_order=3)
    val = extract_neural_jets(field, x_val, max_order=3)
    test = extract_neural_jets(field, x_test, max_order=3)
    result = NeuralJetDiscoverer(
        max_library_degree=1,
        thresholds=(1e-5, 1e-4, 1e-3, 1e-2),
        complexity_weight=5e-3,
    ).discover(train, val, test, candidate_lhs_orders=(2,))
    design, _ = build_jet_relation_library(
        test,
        lhs_order=result.lhs_order,
        max_degree=1,
    )
    pred = result.equation.predict(design)
    true_residual_rmse = rmse(np.zeros_like(x_test), pred - (-np.sin(x_test)))
    return {
        "field_train_rmse": field.train_rmse,
        "equation": result.formula(),
        "selected_terms": result.active_terms(),
        "jet_test_rmse": result.test_rmse,
        "true_identity_rmse": true_residual_rmse,
    }


def default_surrogate_specs(
    *,
    max_degree: int = 2,
    max_frequency: int = 2,
) -> list[LibrarySpec]:
    return [
        LibrarySpec(
            name="taylor",
            builder=lambda x: build_taylor_library(x, max_degree=max_degree),
            description=f"polynomial Taylor-like monomials up to degree {max_degree}",
        ),
        LibrarySpec(
            name="fourier",
            builder=lambda x: build_fourier_library(x, max_frequency=max_frequency),
            description=f"per-coordinate Fourier modes up to frequency {max_frequency}",
        ),
        LibrarySpec(
            name="taylor_fourier",
            builder=lambda x: build_hybrid_library(
                x,
                max_degree=max_degree,
                max_frequency=max_frequency,
            ),
            description="union of Taylor monomials and Fourier modes",
        ),
    ]


def build_taylor_library(x: np.ndarray, *, max_degree: int = 2) -> tuple[np.ndarray, list[str]]:
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")
    x = np.asarray(x, dtype=float)
    cols: list[np.ndarray] = []
    names: list[str] = []
    dim = x.shape[1]
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(dim), degree):
            cols.append(np.prod(x[:, combo], axis=1))
            names.append(_monomial_name(combo))
    return np.stack(cols, axis=1), names


def build_fourier_library(x: np.ndarray, *, max_frequency: int = 2) -> tuple[np.ndarray, list[str]]:
    if max_frequency < 1:
        raise ValueError(f"max_frequency must be >= 1, got {max_frequency}")
    x = np.asarray(x, dtype=float)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for j in range(x.shape[1]):
        for k in range(1, max_frequency + 1):
            label = f"{k}*" if k > 1 else ""
            cols.append(np.sin(k * x[:, j]))
            names.append(f"sin({label}x{j + 1})")
            cols.append(np.cos(k * x[:, j]))
            names.append(f"cos({label}x{j + 1})")
    return np.stack(cols, axis=1), names


def build_hybrid_library(
    x: np.ndarray,
    *,
    max_degree: int = 2,
    max_frequency: int = 2,
) -> tuple[np.ndarray, list[str]]:
    taylor, taylor_names = build_taylor_library(x, max_degree=max_degree)
    fourier, fourier_names = build_fourier_library(x, max_frequency=max_frequency)
    return np.concatenate([taylor, fourier], axis=1), [*taylor_names, *fourier_names]


def _np_sigmoid(u: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid."""
    out = np.empty_like(u)
    pos = u >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    eu = np.exp(u[~pos])
    out[~pos] = eu / (1.0 + eu)
    return out


#: CDF-shaped (monotone ``R -> [0, 1]``) basis transforms for the band library.
_CDF_BASES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": _np_sigmoid,
    "tanh": lambda u: 0.5 * np.tanh(u) + 0.5,
    "arctan": lambda u: np.arctan(u) / math.pi + 0.5,
}


#: Self-information (surprisal) ``-ln f(u)`` of each CDF base's density ``f`` (the
#: derivative of the matching :data:`_CDF_BASES` transform), in the standardized
#: coordinate ``u = (x - loc)/scale``. The expectation of this column under the
#: model *is* the (differential) entropy, so it is the information-theoretic twin
#: of the probability features -- a smooth, even, non-monotone column a CDF or
#: polynomial basis cannot represent. Evaluated in overflow-safe log-domain form:
#:
#: * ``sigmoid``: ``f = s(1-s)`` -> ``softplus(u) + softplus(-u)``;
#: * ``tanh``:    ``f = (1 - t^2)/2`` -> ``2*logaddexp(u, -u) - ln 2``;
#: * ``arctan``:  ``f = 1/(pi(1+u^2))`` -> ``ln(pi) + ln(1 + u^2)``.
#:
#: (Scaling ``f`` to ``x``-space only adds the constant ``ln(scale)``, which the
#: regression intercept absorbs, so the standardized surprisal is used.)
_INFO_BASES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": lambda u: np.logaddexp(0.0, u) + np.logaddexp(0.0, -u),
    "tanh": lambda u: 2.0 * np.logaddexp(u, -u) - math.log(2.0),
    "arctan": lambda u: math.log(math.pi) + np.log1p(u * u),
}


def _cdf_band_design(
    x: np.ndarray,
    bases: tuple[str, ...],
    per_feature_locations: list[np.ndarray],
    per_feature_scales: list[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Assemble the CDF/band design matrix from fixed per-feature locations/scales."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        locs = per_feature_locations[j]
        scales = per_feature_scales[j]
        for base in bases:
            transform = _CDF_BASES[base]
            for s in scales:
                vals = transform((col[:, None] - locs[None, :]) / s)
                for li in range(locs.shape[0]):
                    cols.append(vals[:, li])
                    names.append(f"{base}((x{j + 1}-{locs[li]:.4g})/{s:.4g})")
    if not cols:
        raise ValueError("cdf_band library produced no columns")
    return np.stack(cols, axis=1), names


def build_cdf_band_library(
    x: np.ndarray,
    *,
    bases: tuple[str, ...] = ("sigmoid", "tanh"),
    locations: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    r"""Location-scale CDF features ``F((x_j - loc)/scale)`` on a shared grid.

    Each column is a monotone CDF transform (``sigmoid`` / ``tanh`` / ``arctan``)
    of an affine map of one feature -- the data-side twin of the OMBU *band*: a
    difference of two such columns is the empirical probability mass of a slab.
    This stateless primitive uses the same ``locations`` and ``scales`` for every
    feature; for train-fitted, data-adaptive placement use
    :func:`fit_cdf_band_library_plan`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    locs = np.asarray(locations, dtype=float).reshape(-1)
    scs = np.asarray(scales, dtype=float).reshape(-1)
    if locs.size == 0 or scs.size == 0:
        raise ValueError("locations and scales must be non-empty")
    if np.any(scs <= 0.0):
        raise ValueError("scales must be positive")
    for base in bases:
        if base not in _CDF_BASES:
            raise ValueError(f"unknown CDF base {base!r}; supported: {tuple(_CDF_BASES)}")
    dim = x.shape[1]
    return _cdf_band_design(x, tuple(bases), [locs] * dim, [scs] * dim)


def fit_cdf_band_library_plan(
    x_train: np.ndarray,
    *,
    bases: tuple[str, ...] = ("sigmoid", "tanh"),
    n_locations: int = 5,
    scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> LibrarySpec:
    r"""Fit a data-adaptive CDF/band :class:`LibrarySpec` on the training split.

    Per-feature locations are the train quantiles (``n_locations`` of them between
    the 10th and 90th percentile) and scales are ``scale_mults`` times the train
    standard deviation. The returned spec's builder closes over those *fixed*
    parameters, so it produces consistent columns across train / val / test (no
    leakage) -- the correct way to feed CDF features into
    :func:`discover_interpretable_surrogate`.
    """
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")
    if n_locations < 1:
        raise ValueError(f"n_locations must be >= 1, got {n_locations}")
    if len(scale_mults) == 0 or any(m <= 0.0 for m in scale_mults):
        raise ValueError("scale_mults must be a non-empty tuple of positive floats")
    for base in bases:
        if base not in _CDF_BASES:
            raise ValueError(f"unknown CDF base {base!r}; supported: {tuple(_CDF_BASES)}")
    bases_t = tuple(bases)
    mults = np.asarray(scale_mults, dtype=float)
    quantiles = np.linspace(0.1, 0.9, n_locations)
    per_locs: list[np.ndarray] = []
    per_scales: list[np.ndarray] = []
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        per_locs.append(np.quantile(col, quantiles))
        std = float(col.std())
        per_scales.append((std if std > 1e-12 else 1.0) * mults)

    def builder(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return _cdf_band_design(np.asarray(x, dtype=float), bases_t, per_locs, per_scales)

    label = "+".join(bases_t)
    return LibrarySpec(
        name="cdf_band",
        builder=builder,
        description=f"location-scale CDF band features ({label}) fit on train quantiles",
    )


def _information_design(
    x: np.ndarray,
    bases: tuple[str, ...],
    per_feature_locations: list[np.ndarray],
    per_feature_scales: list[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Assemble the surprisal design matrix from fixed per-feature locations/scales."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        locs = per_feature_locations[j]
        scales = per_feature_scales[j]
        for base in bases:
            transform = _INFO_BASES[base]
            for s in scales:
                vals = transform((col[:, None] - locs[None, :]) / s)
                for li in range(locs.shape[0]):
                    cols.append(vals[:, li])
                    names.append(f"surprisal_{base}((x{j + 1}-{locs[li]:.4g})/{s:.4g})")
    if not cols:
        raise ValueError("information library produced no columns")
    return np.stack(cols, axis=1), names


def build_information_library(
    x: np.ndarray,
    *,
    bases: tuple[str, ...] = ("sigmoid", "arctan"),
    locations: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    r"""Self-information (surprisal) features ``-ln f((x_j - loc)/scale)`` on a grid.

    The information-theoretic twin of :func:`build_cdf_band_library`: instead of
    the monotone CDF ``F``, each column is the *surprisal* ``-ln f`` of the
    matching density ``f = F'`` (``sigmoid`` / ``tanh`` / ``arctan``) -- a smooth,
    even, bump/log-quadratic-shaped feature whose model expectation is the
    (differential) entropy. These let a sparse surrogate express log-likelihood /
    energy-style terms (e.g. ``ln(1 + ((x - loc)/scale)^2)`` from ``arctan``) that
    monotone CDF or polynomial bases cannot. This stateless primitive shares one
    ``locations`` / ``scales`` grid across features; for train-fitted, data-
    adaptive placement use :func:`fit_information_library_plan`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    locs = np.asarray(locations, dtype=float).reshape(-1)
    scs = np.asarray(scales, dtype=float).reshape(-1)
    if locs.size == 0 or scs.size == 0:
        raise ValueError("locations and scales must be non-empty")
    if np.any(scs <= 0.0):
        raise ValueError("scales must be positive")
    for base in bases:
        if base not in _INFO_BASES:
            raise ValueError(
                f"unknown information base {base!r}; supported: {tuple(_INFO_BASES)}"
            )
    dim = x.shape[1]
    return _information_design(x, tuple(bases), [locs] * dim, [scs] * dim)


def fit_information_library_plan(
    x_train: np.ndarray,
    *,
    bases: tuple[str, ...] = ("sigmoid", "arctan"),
    n_locations: int = 5,
    scale_mults: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> LibrarySpec:
    r"""Fit a data-adaptive surprisal :class:`LibrarySpec` on the training split.

    The information-theoretic counterpart of :func:`fit_cdf_band_library_plan`:
    per-feature locations are train quantiles (``n_locations`` between the 10th
    and 90th percentile) and scales are ``scale_mults`` times the train standard
    deviation. The returned spec closes over those *fixed* parameters, so it emits
    consistent columns across train / val / test (no leakage). Pass it to
    :func:`discover_interpretable_surrogate` (or enable ``include_information``).
    """
    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be 2D, got shape {x_train.shape}")
    if n_locations < 1:
        raise ValueError(f"n_locations must be >= 1, got {n_locations}")
    if len(scale_mults) == 0 or any(m <= 0.0 for m in scale_mults):
        raise ValueError("scale_mults must be a non-empty tuple of positive floats")
    for base in bases:
        if base not in _INFO_BASES:
            raise ValueError(
                f"unknown information base {base!r}; supported: {tuple(_INFO_BASES)}"
            )
    bases_t = tuple(bases)
    mults = np.asarray(scale_mults, dtype=float)
    quantiles = np.linspace(0.1, 0.9, n_locations)
    per_locs: list[np.ndarray] = []
    per_scales: list[np.ndarray] = []
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        per_locs.append(np.quantile(col, quantiles))
        std = float(col.std())
        per_scales.append((std if std > 1e-12 else 1.0) * mults)

    def builder(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return _information_design(np.asarray(x, dtype=float), bases_t, per_locs, per_scales)

    label = "+".join(bases_t)
    return LibrarySpec(
        name="information",
        builder=builder,
        description=f"location-scale self-information (surprisal) features ({label})",
    )


def discover_interpretable_surrogate(
    data: SplitData,
    specs: list[LibrarySpec] | None = None,
    *,
    alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2),
    thresholds: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    complexity_weight: float = 1e-3,
    include_cdf_band: bool = True,
    cdf_bases: tuple[str, ...] = ("sigmoid", "tanh"),
    include_information: bool = False,
    information_bases: tuple[str, ...] = ("sigmoid", "arctan"),
    divergence_objective: str | None = None,
    divergence_weight: float = 1.0,
    diagnostics_bins: int = 32,
    dependence_bins: int = 16,
    selection_criterion: str | None = None,
) -> dict[str, object]:
    """Auto-select the best sparse surrogate family using validation data.

    When ``specs`` is not supplied the candidate families are the Taylor / Fourier
    / hybrid defaults plus (if ``include_cdf_band``) a train-fitted location-scale
    CDF/band library and (if ``include_information``) the self-information
    (surprisal) library, letting the search recover logistic / probit-style laws
    *and* log-likelihood / energy-style laws that a polynomial or Fourier basis
    cannot express. Passing an explicit ``specs`` list disables both automatic
    families.

    Setting ``divergence_objective`` (one of
    :data:`omnibias.symbolic.diagnostics.DIVERGENCE_OBJECTIVES`) adds
    ``divergence_weight * term`` -- a validation-residual divergence to a matched
    Gaussian, or the residual/input mutual information -- to the selection score,
    so information theory / optimal transport co-drive model selection. The
    returned dict always carries a ``"residual_diagnostics"`` report of the
    selected model's test residuals regardless.

    ``selection_criterion`` (``None`` by default, preserving the validation RMSE
    + complexity score) may be ``"aic"`` / ``"aicc"`` / ``"bic"`` / ``"mdl"``;
    when set, families are ranked by that information criterion on the training
    residuals (lower is better), with any divergence objective still added.
    """

    if selection_criterion is not None:
        from omnibias.symbolic.selection import information_criterion
    if specs is not None:
        candidates = specs
    else:
        candidates = default_surrogate_specs()
        if include_cdf_band:
            candidates = [*candidates, fit_cdf_band_library_plan(data.x_train, bases=cdf_bases)]
        if include_information:
            candidates = [
                *candidates,
                fit_information_library_plan(data.x_train, bases=information_bases),
            ]
    best: dict[str, object] | None = None
    tried: list[dict[str, object]] = []
    for spec in candidates:
        train_design, names = spec.builder(data.x_train)
        val_design, _ = spec.builder(data.x_val)
        test_design, _ = spec.builder(data.x_test)
        for alpha in alphas:
            for threshold in thresholds:
                equation = fit_sparse_equation(
                    train_design,
                    data.y_train,
                    names,
                    alpha=alpha,
                    threshold=threshold,
                )
                val_pred = equation.predict(val_design)
                val_rmse = rmse(data.y_val, val_pred)
                active_count = len(equation.active_terms())
                if selection_criterion is None:
                    score = val_rmse + complexity_weight * active_count
                else:
                    train_resid = data.y_train - equation.predict(train_design)
                    score = information_criterion(
                        selection_criterion,
                        int(data.y_train.shape[0]),
                        float(train_resid @ train_resid),
                        active_count + 1,
                        n_candidates=len(names),
                    )
                if divergence_objective is not None:
                    score += divergence_weight * divergence_objective_term(
                        divergence_objective,
                        data.x_val,
                        data.y_val - val_pred,
                        bins=diagnostics_bins,
                        dependence_bins=dependence_bins,
                    )
                row: dict[str, object] = {
                    "family": spec.name,
                    "alpha": alpha,
                    "threshold": threshold,
                    "validation_rmse": val_rmse,
                    "active_terms": active_count,
                    "selection_score": score,
                }
                tried.append(row)
                if best is None or score < float(best["selection_score"]):
                    best = row | {"spec": spec}

    if best is None:
        raise RuntimeError(
            "discover_interpretable_surrogate produced no candidates; "
            "check library_specs / alphas / thresholds"
        )
    selected_spec = best["spec"]
    if not isinstance(selected_spec, LibrarySpec):
        raise TypeError(
            f"selected surrogate library_spec must be LibrarySpec, got {type(selected_spec).__name__}"
        )
    fit_x = np.concatenate([data.x_train, data.x_val], axis=0)
    fit_y = np.concatenate([data.y_train, data.y_val], axis=0)
    fit_design, names = selected_spec.builder(fit_x)
    test_design, _ = selected_spec.builder(data.x_test)
    equation = fit_sparse_equation(
        fit_design,
        fit_y,
        names,
        alpha=float(best["alpha"]),
        threshold=float(best["threshold"]),
    )
    pred = equation.predict(test_design)
    return {
        "family": selected_spec.name,
        "description": selected_spec.description,
        "equation": equation.formula(lhs="y"),
        "selected_terms": equation.active_terms(),
        "metrics": {"rmse": rmse(data.y_test, pred), "mae": mae(data.y_test, pred)},
        "selection": {
            key: best[key]
            for key in ["alpha", "threshold", "validation_rmse", "active_terms", "selection_score"]
        },
        "residual_diagnostics": surrogate_residual_diagnostics(
            data.x_test, data.y_test, pred, bins=diagnostics_bins, dependence_bins=dependence_bins
        ),
        "tried": sorted(tried, key=lambda row: float(row["selection_score"]))[:8],
    }


def fit_sparse_equation(
    design: np.ndarray,
    target: np.ndarray,
    term_names: list[str],
    *,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    max_iter: int = 8,
) -> SparseEquation:
    """Sequential thresholded ridge regression (STLSQ).

    This is a **numerical, non-differentiable** least-squares routine (numpy):
    the library columns handed to it may be exact closed form, but the sparse
    fit itself is not a closed-form identity.

    All thresholding is performed in **standardized space** (columns centered
    and scaled to unit std), so ``threshold`` is a single, scale-invariant
    criterion: rescaling a library column does not change which terms survive.
    The surviving coefficients are unscaled back to raw units only at the end.
    """

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.ndim != 2:
        raise ValueError(f"design must be 2D, got shape {design.shape}")
    if design.shape[1] != len(term_names):
        raise ValueError("term_names must match design width")

    col_mean = design.mean(axis=0)
    col_scale = np.where(design.std(axis=0) < 1e-12, 1.0, design.std(axis=0))
    y_mean = float(target.mean())
    xz = (design - col_mean) / col_scale
    yz = target - y_mean
    active = np.ones(design.shape[1], dtype=bool)
    coef_z = np.zeros(design.shape[1])

    for _ in range(max_iter):
        if not np.any(active):
            break
        local = _ridge_coef(xz[:, active], yz, alpha)
        next_active = active.copy()
        active_indices = np.flatnonzero(active)
        next_active[active_indices[np.abs(local) < threshold]] = False
        coef_z[:] = 0.0
        coef_z[active] = local
        if np.array_equal(next_active, active):
            break
        active = next_active

    if np.any(active):
        local = _ridge_coef(xz[:, active], yz, alpha)
        coef_z[:] = 0.0
        coef_z[active] = local

    # Cull sub-threshold terms in standardized space (matching the STLSQ loop),
    # *then* unscale -- keeping every threshold comparison in one consistent
    # space. ``col_scale > 0`` so the unscale preserves the zero pattern.
    coef_z[np.abs(coef_z) < threshold] = 0.0
    coef = coef_z / col_scale
    intercept = y_mean - float(np.dot(coef, col_mean))
    active_mask = np.abs(coef) > 0.0
    return SparseEquation(
        term_names=tuple(term_names),
        coefficients=coef,
        intercept=intercept,
        alpha=alpha,
        threshold=threshold,
        active_mask=active_mask,
    )


def make_heat_equation_operator_data(
    *,
    n_x: int = 80,
    n_t: int = 40,
    diffusivity: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Build exact derivative columns for a two-mode heat-equation field."""

    x = np.linspace(0.0, 1.0, n_x)
    t = np.linspace(0.0, 0.4, n_t)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    mode1 = np.exp(-diffusivity * math.pi**2 * tt) * np.sin(math.pi * xx)
    mode2 = 0.5 * np.exp(-diffusivity * (2.0 * math.pi) ** 2 * tt) * np.sin(2.0 * math.pi * xx)
    u = mode1 + mode2
    ux = (
        math.pi * np.exp(-diffusivity * math.pi**2 * tt) * np.cos(math.pi * xx)
        + math.pi * np.exp(-diffusivity * (2.0 * math.pi) ** 2 * tt) * np.cos(2.0 * math.pi * xx)
    )
    uxx = (
        -(math.pi**2) * mode1
        - (2.0 * math.pi) ** 2 * mode2
    )
    ut = diffusivity * uxx
    library, names = build_pde_operator_library(u.ravel(), ux.ravel(), uxx.ravel())
    return library, ut.ravel(), names, f"u_t = {diffusivity:g}*u_xx"


def build_pde_operator_library(
    u: np.ndarray,
    ux: np.ndarray,
    uxx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    cols = [
        u,
        u * u,
        u * u * u,
        ux,
        uxx,
        u * ux,
        u * uxx,
        ux * ux,
    ]
    names = ["u", "u^2", "u^3", "u_x", "u_xx", "u*u_x", "u*u_xx", "u_x^2"]
    return np.stack(cols, axis=1), names


def discover_pde_operator_law(
    *,
    diffusivity: float = 0.12,
    alpha: float = 1e-10,
    threshold: float = 1e-5,
) -> dict[str, object]:
    library, target, names, hidden = make_heat_equation_operator_data(diffusivity=diffusivity)
    equation = fit_sparse_equation(library, target, names, alpha=alpha, threshold=threshold)
    pred = equation.predict(library)
    return {
        "hidden_law": hidden,
        "equation": equation.formula(lhs="u_t"),
        "selected_terms": equation.active_terms(),
        "metrics": {"rmse": rmse(target, pred), "mae": mae(target, pred)},
    }


def evaluate_real_world_tabular_validation(
    *,
    dataset: str = "diabetes",
    seed: int = 0,
) -> dict[str, object]:
    """Run the sparse AutoRegressor on a reproducible sklearn tabular dataset."""

    x, y, feature_names, dataset_label, available, reason = _load_sklearn_regression_dataset(dataset)
    if not available:
        return {"dataset": dataset, "available": False, "reason": reason}

    split = _split_arrays(x, y, seed=seed)
    plan = fit_screened_feature_library_plan(
        split.x_train,
        split.y_train,
        feature_names=feature_names,
        max_raw=min(12, x.shape[1]),
        max_square=min(8, x.shape[1]),
        max_sin=0,
        max_pairs=min(12, max(1, x.shape[1] * (x.shape[1] - 1) // 2)),
    )
    symbolic = _fit_sparse_plan_with_validation(
        split,
        plan,
        thresholds=(1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1),
        complexity_weight=5e-2,
        max_active_terms=10,
        lhs="target",
    )
    raw_plan = FeatureLibraryPlan(
        feature_names=tuple(feature_names),
        raw_indices=tuple(range(x.shape[1])),
        square_indices=(),
        sin_indices=(),
        pair_indices=(),
    )
    raw = _fit_sparse_plan_with_validation(split, raw_plan, complexity_weight=0.0, lhs="target")
    return {
        "dataset": dataset_label,
        "available": True,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "symbolic_autoregressor": symbolic,
        "raw_linear_baseline": raw,
        "improvement_vs_raw_rmse": raw["metrics"]["rmse"] - symbolic["metrics"]["rmse"],
    }


def evaluate_poc(
    *,
    n_samples: int = 900,
    noise_std: float = 0.02,
    seed: int = 0,
) -> dict[str, object]:
    data = make_symbolic_regression_dataset(n_samples=n_samples, noise_std=noise_std, seed=seed)
    surrogate = discover_interpretable_surrogate(data)
    pde = discover_pde_operator_law()
    high_dim = evaluate_high_dim_sparse_validation(noise_std=max(noise_std, 0.02), seed=seed)
    real_world = {
        "diabetes": evaluate_real_world_tabular_validation(dataset="diabetes", seed=seed),
        "california_housing": evaluate_real_world_tabular_validation(dataset="california_housing", seed=seed),
    }
    jet_identities = {
        "exp": _jet_result_payload(discover_activation_identity("exp", candidate_lhs_orders=(1,))),
        "sin": _jet_result_payload(discover_activation_identity("sin", candidate_lhs_orders=(2,))),
        "tanh": _jet_result_payload(discover_activation_identity("tanh", candidate_lhs_orders=(1,))),
    }
    noisy_neural_field = discover_from_noisy_observations(seed=seed, noise_std=noise_std)
    return {
        "name": "Scientific AutoML PoC",
        "claim": "closed-form neural calculus plus weak-prior symbolic compression",
        "fairness_protocol": {
            "surrogate_fit_split": "train",
            "family_selection_split": "validation",
            "final_scoring_split": "test",
            "neural_jet_discovery": "closed-form omnibias fastpaths generate y, dy, d2y, ...; compression searches generic jet relations, not named functions",
            "pde_derivatives": "exact analytic columns in PoC; omnibias field operators are the next PDE integration point",
        },
        "symbolic_surrogate": {
            "hidden_law": "y = 1.5*x1^2 - 2*x2*x3 + sin(2*x4) + 0.4*cos(x4) + noise",
            **surrogate,
        },
        "operator_discovery": pde,
        "high_dimensional_sparse_validation": high_dim,
        "real_world_tabular_validation": real_world,
        "neural_jet_discovery": {
            "identities": jet_identities,
            "noisy_observation_sin": noisy_neural_field,
        },
    }


def write_artifacts(results: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    surrogate = results["symbolic_surrogate"]
    operator = results["operator_discovery"]
    neural_jet = results["neural_jet_discovery"]
    high_dim = results["high_dimensional_sparse_validation"]
    real_world = results["real_world_tabular_validation"]
    if not isinstance(surrogate, dict):
        raise TypeError(f"results['symbolic_surrogate'] must be a dict, got {type(surrogate).__name__}")
    if not isinstance(operator, dict):
        raise TypeError(f"results['operator_discovery'] must be a dict, got {type(operator).__name__}")
    if not isinstance(neural_jet, dict):
        raise TypeError(
            f"results['neural_jet_discovery'] must be a dict, got {type(neural_jet).__name__}"
        )
    if not isinstance(high_dim, dict):
        raise TypeError(
            f"results['high_dimensional_sparse_validation'] must be a dict, got {type(high_dim).__name__}"
        )
    if not isinstance(real_world, dict):
        raise TypeError(
            f"results['real_world_tabular_validation'] must be a dict, got {type(real_world).__name__}"
        )
    report = [
        "# Scientific AutoML PoC",
        "",
        "This PoC demonstrates equation discovery, interpretable surrogate modeling, and neural-jet identity discovery.",
        "",
        "## Surrogate Discovery",
        "",
        f"Hidden law: `{surrogate['hidden_law']}`",
        f"Selected family: `{surrogate['family']}`",
        f"Discovered equation: `{surrogate['equation']}`",
        f"Test RMSE: `{surrogate['metrics']['rmse']:.6f}`",
        "",
        "Selected terms:",
    ]
    for row in surrogate["selected_terms"]:
        report.append(f"- `{row['name']}` with coefficient `{row['coefficient']:.6g}`")

    report.extend(
        [
            "",
            "## Operator Discovery",
            "",
            f"Hidden law: `{operator['hidden_law']}`",
            f"Discovered equation: `{operator['equation']}`",
            f"RMSE: `{operator['metrics']['rmse']:.6e}`",
            "",
            "Selected terms:",
        ]
    )
    for row in operator["selected_terms"]:
        report.append(f"- `{row['name']}` with coefficient `{row['coefficient']:.6g}`")

    report.extend(
        [
            "",
            "## High-Dimensional Sparse Validation",
            "",
            f"Hidden law: `{high_dim['hidden_law']}`",
            f"Discovered equation: `{high_dim['equation']}`",
            f"Recovered terms: `{', '.join(high_dim['recovered_terms'])}`",
            f"Test RMSE: `{high_dim['metrics']['rmse']:.6f}`",
        ]
    )

    report.extend(["", "## Real-World Tabular Validation", ""])
    for name, payload in real_world.items():
        if not isinstance(payload, dict):
            raise TypeError(
                f"real_world_tabular_validation[{name!r}] must be a dict, got {type(payload).__name__}"
            )
        if not payload.get("available", False):
            report.append(f"- `{name}`: unavailable ({payload.get('reason', 'unknown reason')})")
            continue
        symbolic = payload["symbolic_autoregressor"]
        raw = payload["raw_linear_baseline"]
        report.append(
            f"- `{payload['dataset']}`: symbolic RMSE `{symbolic['metrics']['rmse']:.6f}`, "
            f"raw RMSE `{raw['metrics']['rmse']:.6f}`, equation `{symbolic['equation']}`"
        )

    report.extend(["", "## Neural Jet Discovery", ""])
    identities = neural_jet["identities"]
    if not isinstance(identities, dict):
        raise TypeError(
            f"neural_jet_discovery['identities'] must be a dict, got {type(identities).__name__}"
        )
    for name, payload in identities.items():
        report.append(
            f"- `{name}`: `{payload['equation']}`, test RMSE `{payload['test_rmse']:.3e}`"
        )
    noisy = neural_jet["noisy_observation_sin"]
    if not isinstance(noisy, dict):
        raise TypeError(
            f"neural_jet_discovery['noisy_observation_sin'] must be a dict, got {type(noisy).__name__}"
        )
    report.extend(
        [
            "",
            "Noisy-observation smoke:",
            f"- equation: `{noisy['equation']}`",
            f"- neural field train RMSE: `{noisy['field_train_rmse']:.6f}`",
            f"- identity RMSE against true `d2y=-sin(x)`: `{noisy['true_identity_rmse']:.6f}`",
        ]
    )

    report.extend(
        [
            "",
            "## Next Integration Point",
            "",
            "The scalar jet path now uses closed-form omnibias fastpaths. The PDE columns are "
            "still analytic in this smoke test. The production path is to fit an omnibias field "
            "to noisy PDE observations, extract `u`, `u_x`, `u_xx`, `Delta^k u`, and other "
            "operators through `omnibias.pinn`, then run the same weak-prior law search over "
            "those columns.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report) + "\n")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(err**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(np.abs(err)))


def _fit_sparse_plan_with_validation(
    data: SplitData,
    plan: FeatureLibraryPlan,
    *,
    lhs: str = "y",
    alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0),
    thresholds: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 5e-2),
    complexity_weight: float = 1e-3,
    max_active_terms: int | None = None,
) -> dict[str, object]:
    train_design, names = plan.transform(data.x_train)
    val_design, _ = plan.transform(data.x_val)
    best: dict[str, object] | None = None
    y_scale = float(np.std(data.y_val))
    if y_scale < 1e-12:
        y_scale = 1.0
    tried: list[dict[str, object]] = []
    for alpha in alphas:
        for threshold in thresholds:
            equation = fit_sparse_equation(
                train_design,
                data.y_train,
                names,
                alpha=alpha,
                threshold=threshold,
            )
            keep_indices = np.arange(train_design.shape[1])
            if max_active_terms is not None:
                equation, keep_indices = _limit_sparse_equation_terms(
                    train_design,
                    data.y_train,
                    names,
                    equation,
                    max_terms=max_active_terms,
                    alpha=alpha,
                )
            pred_val = equation.predict(val_design[:, keep_indices])
            val_rmse = rmse(data.y_val, pred_val)
            active_count = len(equation.active_terms())
            score = val_rmse / y_scale + complexity_weight * active_count
            row = {
                "alpha": alpha,
                "threshold": threshold,
                "validation_rmse": val_rmse,
                "active_terms": active_count,
                "selection_score": score,
            }
            tried.append(row)
            if best is None or score < float(best["selection_score"]):
                best = row
    if best is None:
        raise RuntimeError(
            "sparse equation search produced no candidates; check alphas / thresholds"
        )

    fit_x = np.concatenate([data.x_train, data.x_val], axis=0)
    fit_y = np.concatenate([data.y_train, data.y_val], axis=0)
    fit_design, names = plan.transform(fit_x)
    test_design, _ = plan.transform(data.x_test)
    equation = fit_sparse_equation(
        fit_design,
        fit_y,
        names,
        alpha=float(best["alpha"]),
        threshold=float(best["threshold"]),
    )
    keep_indices = np.arange(fit_design.shape[1])
    if max_active_terms is not None:
        equation, keep_indices = _limit_sparse_equation_terms(
            fit_design,
            fit_y,
            names,
            equation,
            max_terms=max_active_terms,
            alpha=float(best["alpha"]),
        )
    pred_test = equation.predict(test_design[:, keep_indices])
    return {
        "equation": equation.formula(lhs=lhs),
        "selected_terms": equation.active_terms(),
        "metrics": {"rmse": rmse(data.y_test, pred_test), "mae": mae(data.y_test, pred_test)},
        "selection": best,
        "feature_names": list(equation.term_names),
        "tried": sorted(tried, key=lambda row: float(row["selection_score"]))[:8],
    }


def _split_arrays(x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    n_train = int(0.6 * x.shape[0])
    n_val = int(0.2 * x.shape[0])
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
    )


def _limit_sparse_equation_terms(
    design: np.ndarray,
    target: np.ndarray,
    names: list[str],
    equation: SparseEquation,
    *,
    max_terms: int,
    alpha: float,
) -> tuple[SparseEquation, np.ndarray]:
    active = np.flatnonzero(np.abs(equation.coefficients) > 0.0)
    if active.size <= max_terms:
        return equation, np.arange(design.shape[1])
    scale = np.std(design, axis=0)
    contribution = np.abs(equation.coefficients) * np.where(scale < 1e-12, 1.0, scale)
    keep = active[np.argsort(-contribution[active])[:max_terms]]
    keep = np.sort(keep)
    limited_names = [names[index] for index in keep]
    limited = fit_sparse_equation(
        design[:, keep],
        target,
        limited_names,
        alpha=alpha,
        threshold=0.0,
    )
    return limited, keep


def _load_sklearn_regression_dataset(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, list[str], str, bool, str]:
    try:
        if dataset == "diabetes":
            from sklearn.datasets import load_diabetes

            bunch = load_diabetes()
            return (
                np.asarray(bunch.data, dtype=float),
                np.asarray(bunch.target, dtype=float),
                [str(name) for name in bunch.feature_names],
                "sklearn diabetes",
                True,
                "",
            )
        if dataset == "california_housing":
            from sklearn.datasets import fetch_california_housing

            bunch = fetch_california_housing(download_if_missing=False)
            return (
                np.asarray(bunch.data, dtype=float),
                np.asarray(bunch.target, dtype=float),
                [str(name) for name in bunch.feature_names],
                "sklearn California Housing",
                True,
                "",
            )
    except Exception as exc:
        return np.empty((0, 0)), np.empty((0,)), [], dataset, False, f"{type(exc).__name__}: {exc}"
    return np.empty((0, 0)), np.empty((0,)), [], dataset, False, f"unknown dataset {dataset!r}"


def _ridge_coef(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(design.shape[1])
    return np.linalg.solve(design.T @ design + reg, design.T @ target)


def _monomial_name(combo: tuple[int, ...]) -> str:
    counts = Counter(combo)
    pieces = []
    for index, power in sorted(counts.items()):
        name = f"x{index + 1}"
        pieces.append(name if power == 1 else f"{name}^{power}")
    return "*".join(pieces)


def jet_name(order: int) -> str:
    if order == 0:
        return "y"
    if order == 1:
        return "dy"
    return f"d{order}y"


def _power_name(name: str, power: int) -> str:
    return name if power == 1 else f"{name}^{power}"


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=float))
    return [int(index) for index in order[: min(k, len(order))]]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return 0.0
    return float(aa @ bb / denom)


def _jet_result_payload(result: JetDiscoveryResult) -> dict[str, object]:
    return {
        "equation": result.formula(),
        "lhs": jet_name(result.lhs_order),
        "selected_terms": result.active_terms(),
        "validation_rmse": result.validation_rmse,
        "test_rmse": result.test_rmse,
        "selection_score": result.selection_score,
        "target_scale": result.target_scale,
    }


def _jax_numpy():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jnp

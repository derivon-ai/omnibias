# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.symbolic.piecewise: per-region (piecewise) symbolic law discovery.

A bridge on :mod:`omnibias.partition`: route samples into ``2**depth`` regions with a soft
partition, then run the existing STLSQ sparse regression
(:func:`omnibias.symbolic.discovery.fit_sparse_equation`) **inside each region**. A system
whose governing law switches across a surface is thus recovered as a **hybrid automaton** --
one :class:`~omnibias.symbolic.discovery.SparseEquation` per region plus the hardened
``if ... then`` switch conditions exported from the partition. Global SINDy on the same data
finds a single averaged law that fits neither regime; piecewise recovers both laws *and* the
boundary.

Driver:

* :func:`fit_piecewise_law` -- the generic core: (route coords, design library, target) ->
  per-region :class:`SparseEquation`, reusing ``fit_sparse_equation`` unchanged.
* :func:`fit_piecewise_ode_law` -- a convenience that fits a global omnibias field
  (:func:`~omnibias.symbolic.field_discovery.fit_neural_field_nd`), reads off the *exact*
  closed-form ``(u, u')`` jet (:func:`~omnibias.symbolic.field_discovery.extract_field_jet`),
  and discovers a first-order ODE ``du = f(u)`` per region.
* :func:`global_sparse_law` -- the single-law baseline for comparison.

Honesty: the STLSQ fit is **numpy and non-differentiable** (the partition itself is
differentiable, but this discovery driver does not backprop through it -- a differentiable
piecewise-discovery path is future work). The hardened switch conditions are the
``beta -> inf`` limit of the partition gates -- the **feasibility / temperature** sense of
"collapse" (a soft indicator becoming a 0/1 step), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit of an ``OMBU`` to the
closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.partition._core.params import PartitionParams
from omnibias.partition._core.weights import hard_assignment, hardened_rules, region_rule
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation, rmse
from omnibias.symbolic.field_discovery import (
    NeuralFieldND,
    extract_field_jet,
    fit_neural_field_nd,
)


@dataclass(frozen=True)
class PiecewiseLaw:
    r"""One region's fitted law: the region index, its hardened rule, and the equation."""

    region: int
    rule: str
    equation: SparseEquation
    n_samples: int
    train_rmse: float

    def formula(self, *, lhs: str = "y", digits: int = 4) -> str:
        return str(self.equation.formula(lhs=lhs, digits=digits))


@dataclass(frozen=True)
class HybridAutomaton:
    r"""A piecewise symbolic model: one :class:`PiecewiseLaw` per (non-empty) region.

    Routing is by the ``beta -> inf`` hard partition (:func:`hard_assignment`): a sample is
    assigned to the region whose gate conditions it satisfies, and that region's equation
    makes the prediction. :meth:`switch_conditions` returns the hardened ``if ...`` gate rules.
    """

    partition: PartitionParams
    laws: tuple[PiecewiseLaw, ...]
    term_names: tuple[str, ...]
    lhs_name: str

    @property
    def n_regions(self) -> int:
        return int(self.partition.n_regions)

    def _law_by_region(self) -> dict[int, PiecewiseLaw]:
        return {law.region: law for law in self.laws}

    def region_of(self, x_route: np.ndarray) -> np.ndarray:
        r"""The crisp region index of each routing sample (the ``beta -> inf`` assignment)."""
        idx: np.ndarray = hard_assignment(self.partition, np.asarray(x_route, dtype=float))
        return idx

    def switch_conditions(self) -> list[str]:
        r"""The hardened ``if ...`` split boundaries, one per gate."""
        return list(hardened_rules(self.partition))

    def predict(self, x_route: np.ndarray, design: np.ndarray) -> np.ndarray:
        r"""Route each row to its region and evaluate that region's equation.

        Rows landing in a region with no fitted law (too few training samples) fall back to
        the most-populated region's law.
        """
        idx = self.region_of(x_route)
        design = np.asarray(design, dtype=float)
        if design.shape[0] != idx.shape[0]:
            raise ValueError("x_route and design must share the sample axis")
        by_region = self._law_by_region()
        fallback = max(self.laws, key=lambda law: law.n_samples) if self.laws else None
        out = np.full(design.shape[0], np.nan, dtype=float)
        for i in range(design.shape[0]):
            law = by_region.get(int(idx[i]), fallback)
            if law is None:
                continue
            out[i] = float(law.equation.predict(design[i : i + 1])[0])
        return out

    def report(self, *, digits: int = 4) -> str:
        r"""A human-readable ``if [rule]: lhs = formula`` block, one line per region."""
        lines = []
        for law in self.laws:
            lines.append(
                f"if [{law.rule}]:  {law.formula(lhs=self.lhs_name, digits=digits)}"
                f"   (n={law.n_samples}, rmse={law.train_rmse:.3g})"
            )
        return "\n".join(lines)


def fit_piecewise_law(
    partition: PartitionParams,
    x_route: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    term_names: Sequence[str],
    *,
    lhs_name: str = "dy",
    min_samples: int | None = None,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    max_iter: int = 8,
) -> HybridAutomaton:
    r"""Route samples by ``partition`` and fit one sparse equation per region.

    Parameters
    ----------
    partition:
        The soft partition (its ``beta -> inf`` hard assignment routes the samples).
    x_route:
        ``(n, n_features)`` routing coordinates fed to the partition.
    design:
        ``(n, n_terms)`` candidate-term library (the same for every region).
    target:
        ``(n,)`` left-hand side to regress (e.g. a derivative jet coordinate).
    term_names:
        Names aligned to the columns of ``design``.
    min_samples:
        Regions with fewer routed samples are skipped (default ``n_terms + 1``).
    """
    xr = np.asarray(x_route, dtype=float)
    d = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    names = list(term_names)
    if d.ndim != 2:
        raise ValueError(f"design must be 2D (n, n_terms), got shape {d.shape}")
    if d.shape[1] != len(names):
        raise ValueError("term_names must match design width")
    if d.shape[0] != y.shape[0] or xr.shape[0] != y.shape[0]:
        raise ValueError("x_route, design and target must share the sample axis")
    need = min_samples if min_samples is not None else len(names) + 1
    idx = hard_assignment(partition, xr)
    laws: list[PiecewiseLaw] = []
    for region in range(partition.n_regions):
        mask = idx == region
        n = int(mask.sum())
        if n < need:
            continue
        eq = fit_sparse_equation(
            d[mask], y[mask], names, alpha=alpha, threshold=threshold, max_iter=max_iter
        )
        r = float(rmse(y[mask], eq.predict(d[mask])))
        laws.append(
            PiecewiseLaw(
                region=region,
                rule=region_rule(partition, region),
                equation=eq,
                n_samples=n,
                train_rmse=r,
            )
        )
    if not laws:
        raise ValueError(
            "no region had >= min_samples routed samples; check the partition or min_samples"
        )
    return HybridAutomaton(
        partition=partition, laws=tuple(laws), term_names=tuple(names), lhs_name=lhs_name
    )


def global_sparse_law(
    design: np.ndarray,
    target: np.ndarray,
    term_names: Sequence[str],
    *,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    max_iter: int = 8,
) -> SparseEquation:
    r"""The single-law baseline: one :func:`fit_sparse_equation` over all data (no routing)."""
    return fit_sparse_equation(
        np.asarray(design, dtype=float),
        np.asarray(target, dtype=float).reshape(-1),
        list(term_names),
        alpha=alpha,
        threshold=threshold,
        max_iter=max_iter,
    )


def polynomial_value_library(values: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    r"""Design ``[y, y^2, ..., y^degree]`` + names for a first-order ODE ``dy = f(y)``."""
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    v = np.asarray(values, dtype=float).reshape(-1)
    cols = [v**k for k in range(1, degree + 1)]
    names = ["y"] + [f"y^{k}" for k in range(2, degree + 1)]
    return np.stack(cols, axis=1), names


def fit_piecewise_ode_law(
    x: np.ndarray,
    y: np.ndarray,
    partition: PartitionParams,
    *,
    deriv_axis: int = 0,
    degree: int = 2,
    hidden: int = 256,
    ridge: float = 1e-5,
    activation: str = "tanh",
    bandwidth: float = 1.0,
    seed: int = 0,
    min_samples: int | None = None,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
) -> tuple[HybridAutomaton, NeuralFieldND]:
    r"""Fit a global field ``u(x)``, read its closed-form ``(u, du)`` jet, discover ``du = f(u)`` per region.

    Fits a smooth omnibias random-feature field to ``(x, y)``
    (:func:`~omnibias.symbolic.field_discovery.fit_neural_field_nd`), extracts the *exact*
    closed-form value ``u`` and first partial ``du/dx_{deriv_axis}``
    (:func:`~omnibias.symbolic.field_discovery.extract_field_jet`), then runs
    :func:`fit_piecewise_law` with a polynomial-in-``u`` library and the coordinates ``x`` as
    the routing features. Returns the :class:`HybridAutomaton` and the fitted field.

    The field derivatives are closed form; the per-region STLSQ is numpy / non-differentiable.
    """
    xa = np.asarray(x, dtype=float)
    if xa.ndim != 2:
        raise ValueError(f"x must be 2D (n, d), got shape {xa.shape}")
    dim = xa.shape[1]
    if not (0 <= deriv_axis < dim):
        raise ValueError(f"deriv_axis {deriv_axis} out of range for dim {dim}")
    field = fit_neural_field_nd(
        xa, y, hidden=hidden, ridge=ridge, activation=activation, bandwidth=bandwidth, seed=seed
    )
    jet = extract_field_jet(field, xa, max_order=1)
    u = jet.value()
    alpha_idx = tuple(1 if a == deriv_axis else 0 for a in range(dim))
    du = jet.partial(alpha_idx)
    design, names = polynomial_value_library(u, degree)
    automaton = fit_piecewise_law(
        partition,
        xa,
        design,
        du,
        names,
        lhs_name="du",
        min_samples=min_samples,
        alpha=alpha,
        threshold=threshold,
    )
    return automaton, field


__all__ = [
    "HybridAutomaton",
    "PiecewiseLaw",
    "fit_piecewise_law",
    "fit_piecewise_ode_law",
    "global_sparse_law",
    "polynomial_value_library",
]

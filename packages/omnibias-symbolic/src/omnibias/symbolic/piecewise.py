# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.symbolic.piecewise: per-region (piecewise) symbolic law discovery.

A bridge on :mod:`omnibias.partition`: route samples into ``2**depth`` regions with a soft
partition, then run the existing STLSQ sparse regression
(:func:`omnibias.symbolic.discovery.fit_sparse_equation`) **inside each region**. A system
whose governing law switches across a surface is thus recovered as a **hybrid automaton** --
one :class:`~omnibias.symbolic.discovery.SparseEquation` per region (or ``k`` equations for
a vector field) plus the hardened ``if ... then`` switch conditions exported from the
partition. Global SINDy on the same data finds a single averaged law that fits neither
regime; piecewise recovers both laws *and* the boundary.

Driver:

* :func:`fit_piecewise_law` -- the generic core: (route coords, design library, target) ->
  per-region :class:`SparseEquation`, reusing ``fit_sparse_equation`` unchanged.
* :func:`fit_piecewise_ode_law` -- a convenience that fits a global omnibias field
  (:func:`~omnibias.symbolic.field_discovery.fit_neural_field_nd`), reads off the *exact*
  closed-form ``(u, u')`` jet (:func:`~omnibias.symbolic.field_discovery.extract_field_jet`),
  and discovers a first-order ODE ``du = f(u)`` per region.
* :func:`fit_learned_piecewise_ode` -- **learn** the gates from data (differentiable
  soft-weighted residual), harden, then STLSQ-polish. Oracle partitions stay the control.
* :func:`global_sparse_law` -- the single-law baseline for comparison.

Honesty: STLSQ polish remains **numpy and non-differentiable**. The **gates and per-cell
coefficients** ``xi`` are differentiable (Adam on
``F = sum_l w_l(x; W, t, beta) (xi_l · phi(u))``, with L1 on ``W`` and optional entropy
on ``w_l``). The hardened switch conditions are the ``beta -> inf`` limit of the
partition gates -- the **feasibility / temperature** sense of "collapse" (a soft
indicator becoming a 0/1 step), distinct from the **founding bias collapse** (the
multi-bias ``delta -> 0`` limit of an ``OMBU`` to the closed-form derivative
``sigma^(K-1)``; see ``docs/theory.md``). Jets of a tree surrogate are the
``delta -> 0`` register; switches are the ``beta -> inf`` register.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
from omnibias.partition import PartitionConfig
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
    r"""One region's fitted law: the region index, its hardened rule, and the equation(s)."""

    region: int
    rule: str
    equation: SparseEquation
    n_samples: int
    train_rmse: float
    equations: tuple[SparseEquation, ...] = ()

    def formula(self, *, lhs: str = "y", digits: int = 4) -> str:
        return str(self.equation.formula(lhs=lhs, digits=digits))

    def formulas(self, *, lhs_names: Sequence[str], digits: int = 4) -> tuple[str, ...]:
        eqs = self.equations if self.equations else (self.equation,)
        names = list(lhs_names)
        if len(names) < len(eqs):
            names = names + [f"y{i}" for i in range(len(names), len(eqs))]
        return tuple(eq.formula(lhs=names[i], digits=digits) for i, eq in enumerate(eqs))


@dataclass(frozen=True)
class HybridAutomaton:
    r"""A piecewise symbolic model: one :class:`PiecewiseLaw` per (non-empty) region.

    Routing is by the ``beta -> inf`` hard partition (:func:`hard_assignment`): a sample is
    assigned to the region whose gate conditions it satisfies, and that region's equation
    makes the prediction. :meth:`switch_conditions` returns the hardened ``if ...`` gate rules.
    Vector systems store ``k`` equations per region and :meth:`report` prints every formula.
    """

    partition: PartitionParams
    laws: tuple[PiecewiseLaw, ...]
    term_names: tuple[str, ...]
    lhs_name: str
    lhs_names: tuple[str, ...] = ()

    @property
    def n_regions(self) -> int:
        return int(self.partition.n_regions)

    @property
    def n_outputs(self) -> int:
        if self.lhs_names:
            return len(self.lhs_names)
        law0 = self.laws[0] if self.laws else None
        if law0 is not None and law0.equations:
            return len(law0.equations)
        return 1

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
        the most-populated region's law. Scalar laws return ``(n,)``; vector laws ``(n, k)``.
        """
        idx = self.region_of(x_route)
        design = np.asarray(design, dtype=float)
        if design.shape[0] != idx.shape[0]:
            raise ValueError("x_route and design must share the sample axis")
        by_region = self._law_by_region()
        fallback = max(self.laws, key=lambda law: law.n_samples) if self.laws else None
        k = int(self.n_outputs)
        if k == 1:
            out = np.full(design.shape[0], np.nan, dtype=float)
            for i in range(design.shape[0]):
                law = by_region.get(int(idx[i]), fallback)
                if law is None:
                    continue
                out[i] = float(law.equation.predict(design[i : i + 1])[0])
            return out
        out_m = np.full((design.shape[0], k), np.nan, dtype=float)
        for i in range(design.shape[0]):
            law = by_region.get(int(idx[i]), fallback)
            if law is None:
                continue
            eqs = law.equations if law.equations else (law.equation,)
            for c, eq in enumerate(eqs[:k]):
                out_m[i, c] = float(eq.predict(design[i : i + 1])[0])
        return out_m

    def report(self, *, digits: int = 4) -> str:
        r"""A human-readable ``if [rule]: lhs = formula`` block, one line per region."""
        lines = []
        names = self.lhs_names if self.lhs_names else (self.lhs_name,)
        for law in self.laws:
            if law.equations and len(law.equations) > 1:
                body = " ; ".join(law.formulas(lhs_names=names, digits=digits))
            else:
                body = law.formula(lhs=names[0], digits=digits)
            lines.append(
                f"if [{law.rule}]:  {body}   (n={law.n_samples}, rmse={law.train_rmse:.3g})"
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
    lhs_names: Sequence[str] | None = None,
    min_samples: int | None = None,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    max_iter: int = 8,
) -> HybridAutomaton:
    r"""Route samples by ``partition`` and fit one sparse equation per region (and component).

    Parameters
    ----------
    partition:
        The soft partition (its ``beta -> inf`` hard assignment routes the samples).
    x_route:
        ``(n, n_features)`` routing coordinates fed to the partition.
    design:
        ``(n, n_terms)`` candidate-term library (the same for every region).
    target:
        ``(n,)`` or ``(n, k)`` left-hand side to regress.
    term_names:
        Names aligned to the columns of ``design``.
    min_samples:
        Regions with fewer routed samples are skipped (default ``n_terms + 1``).
    """
    xr = np.asarray(x_route, dtype=float)
    d = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    names = list(term_names)
    if d.ndim != 2:
        raise ValueError(f"design must be 2D (n, n_terms), got shape {d.shape}")
    if d.shape[1] != len(names):
        raise ValueError("term_names must match design width")
    if d.shape[0] != y.shape[0] or xr.shape[0] != y.shape[0]:
        raise ValueError("x_route, design and target must share the sample axis")
    need = min_samples if min_samples is not None else len(names) + 1
    idx = hard_assignment(partition, xr)
    k = int(y.shape[1])
    lhs = tuple(lhs_names) if lhs_names is not None else (
        tuple(f"{lhs_name}{c}" for c in range(k)) if k > 1 else (lhs_name,)
    )
    laws: list[PiecewiseLaw] = []
    for region in range(partition.n_regions):
        mask = idx == region
        n = int(mask.sum())
        if n < need:
            continue
        eqs: list[SparseEquation] = []
        rmses: list[float] = []
        for c in range(k):
            eq = fit_sparse_equation(
                d[mask],
                y[mask, c],
                names,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
            )
            eqs.append(eq)
            rmses.append(float(rmse(y[mask, c], eq.predict(d[mask]))))
        laws.append(
            PiecewiseLaw(
                region=region,
                rule=region_rule(partition, region),
                equation=eqs[0],
                n_samples=n,
                train_rmse=float(np.mean(rmses)),
                equations=tuple(eqs) if k > 1 else (),
            )
        )
    if not laws:
        raise ValueError(
            "no region had >= min_samples routed samples; check the partition or min_samples"
        )
    return HybridAutomaton(
        partition=partition,
        laws=tuple(laws),
        term_names=tuple(names),
        lhs_name=str(lhs[0]),
        lhs_names=tuple(lhs) if k > 1 else (),
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
    y = np.asarray(target, dtype=float)
    if y.ndim == 2:
        y = y[:, 0]
    return fit_sparse_equation(
        np.asarray(design, dtype=float),
        y.reshape(-1),
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


def polynomial_vector_library(values: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    r"""Polynomial library in ``k`` state components, including cross terms.

    ``values`` is ``(n,)`` (delegates to :func:`polynomial_value_library`) or
    ``(n, k)``. Degree-1 columns are ``u0, u1, ...``; higher degrees use
    combinations-with-replacement (e.g. ``u0*u1``).
    """
    u = np.asarray(values, dtype=float)
    if u.ndim == 1:
        return polynomial_value_library(u, degree)
    if u.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got shape {u.shape}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    n, k = u.shape
    cols: list[np.ndarray] = []
    names: list[str] = []
    for deg in range(1, degree + 1):
        for idx in combinations_with_replacement(range(k), deg):
            term = np.ones(n, dtype=float)
            counts: dict[int, int] = {}
            for i in idx:
                term = term * u[:, i]
                counts[i] = counts.get(i, 0) + 1
            label_parts = []
            for i in sorted(counts):
                p = counts[i]
                label_parts.append(f"u{i}" if p == 1 else f"u{i}^{p}")
            cols.append(term)
            names.append("*".join(label_parts))
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


def _phi_design(u: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    uv = np.asarray(u, dtype=float)
    if uv.ndim == 1:
        poly, names = polynomial_value_library(uv, degree)
        intercept = np.ones((poly.shape[0], 1), dtype=float)
        return np.concatenate([intercept, poly], axis=1), ["1"] + names
    poly, names = polynomial_vector_library(uv, degree)
    intercept = np.ones((poly.shape[0], 1), dtype=float)
    return np.concatenate([intercept, poly], axis=1), ["1"] + names


def soft_piecewise_forward_np(
    W: np.ndarray,
    t: np.ndarray,
    xi: np.ndarray,
    x: np.ndarray,
    phi: np.ndarray,
    beta: float,
) -> np.ndarray:
    r"""Numpy ``F = sum_l w_l(x) (xi_l · phi)`` (the differentiable ansatz, evaluated)."""
    from omnibias.partition._core.weights import partition_weights

    cfg = PartitionConfig(
        n_features=int(W.shape[1]),
        depth=int(W.shape[0]),
        split_kind="oblique",
        beta_final=float(beta),
    )
    part = PartitionParams(cfg, np.asarray(W, dtype=float), np.asarray(t, dtype=float))
    w = partition_weights(part, x, float(beta))
    if xi.ndim == 3:
        return np.einsum("nl,ltk,nt->nk", w, xi, phi)
    return np.einsum("nl,lt,nt->n", w, xi, phi)


def _refine_split_threshold(
    W: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    phi: np.ndarray,
    du: np.ndarray,
) -> np.ndarray:
    """1-D STLSQ-adjacent polish: grid-search ``t`` on the first gate given learned ``W``."""
    W = np.asarray(W, dtype=float)
    t = np.asarray(t, dtype=float).copy()
    if W.shape[0] != 1:
        return t
    scores = (np.asarray(x, dtype=float) @ W[0]).reshape(-1)
    y = np.asarray(du, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    P = np.asarray(phi, dtype=float)
    order = np.argsort(scores)
    ss = scores[order]
    n = scores.size
    cands = 0.5 * (ss[7 : n - 8] + ss[8 : n - 7])
    if cands.size == 0:
        return t
    # Downsample long candidate lists; keep endpoints + a uniform stride.
    if cands.size > 80:
        stride = max(1, cands.size // 80)
        cands = cands[::stride]
    best_t = float(t[0])
    best = float("inf")
    for cand in cands:
        left = scores <= cand
        n_left = int(left.sum())
        if n_left < 8 or (n - n_left) < 8:
            continue
        mse = 0.0
        for mask in (left, ~left):
            A, b = P[mask], y[mask]
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            mse += float(np.sum((A @ coef - b) ** 2))
        mse /= n
        if mse < best:
            best = mse
            best_t = float(cand)
    t[0] = best_t
    return t


def _train_soft_piecewise(
    x: np.ndarray,
    phi: np.ndarray,
    du: np.ndarray,
    *,
    n_gates: int,
    steps: int,
    lr: float,
    beta: float,
    l1: float,
    entropy: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from omnibias.partition.torch.weights import combine, partition_weights_arrays

    torch.manual_seed(int(seed))
    xt = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
    phit = torch.as_tensor(np.asarray(phi, dtype=np.float64), dtype=torch.float64)
    yt = torch.as_tensor(np.asarray(du, dtype=np.float64), dtype=torch.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    n_features = int(xt.shape[-1])
    n_terms = int(phit.shape[-1])
    k = int(yt.shape[-1])
    n_regions = 1 << int(n_gates)
    W = torch.zeros(n_gates, n_features, dtype=torch.float64)
    W[:, 0] = 1.0
    W = W + 0.05 * torch.randn(n_gates, n_features, dtype=torch.float64)
    W = torch.nn.Parameter(W)
    t = torch.nn.Parameter(0.4 * torch.ones(n_gates, dtype=torch.float64))
    xi = torch.nn.Parameter(0.1 * torch.randn(n_regions, n_terms, k, dtype=torch.float64))
    opt = torch.optim.Adam([W, t, xi], lr=float(lr))
    b_final = float(beta)
    b_init = max(1.0, b_final / 8.0)
    n_steps = int(steps)
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        b = float(b_init * (b_final / b_init) ** frac)
        opt.zero_grad(set_to_none=True)
        weights = partition_weights_arrays(W, t, xt, b, int(n_gates))
        cell = torch.einsum("ltk,nt->nlk", xi, phit)
        pred = combine(weights, cell)
        loss = torch.mean((pred - yt) ** 2)
        if l1 > 0.0:
            loss = loss + float(l1) * W.abs().mean()
        if entropy > 0.0:
            wclip = weights.clamp(min=1e-8)
            ent = -(wclip * wclip.log()).sum(dim=-1).mean()
            # Early: mix (avoid collapsing to one cell). Late: sharpen.
            loss = loss + float(entropy) * ent * (1.0 if frac >= 0.5 else -1.0)
        loss.backward()
        opt.step()
    return (
        W.detach().cpu().numpy().copy(),
        t.detach().cpu().numpy().copy(),
        xi.detach().cpu().numpy().copy(),
    )


def fit_learned_piecewise_ode(
    x: np.ndarray,
    u: np.ndarray,
    du: np.ndarray,
    *,
    n_gates: int = 1,
    degree: int = 2,
    steps: int = 400,
    lr: float = 0.05,
    beta: float = 8.0,
    l1: float = 1e-3,
    entropy: float = 1e-3,
    seed: int = 0,
    min_samples: int | None = None,
    alpha: float = 1e-12,
    threshold: float = 1e-5,
    lhs_name: str = "du",
) -> tuple[HybridAutomaton, dict[str, np.ndarray]]:
    r"""Learn gates from data, harden, STLSQ-polish into a :class:`HybridAutomaton`.

    Trains the differentiable ansatz
    ``F = sum_l w_l(x; W, t, beta) (xi_l · phi(u))``
    by Adam (L1 on ``W``, optional entropy on ``w_l``), then hardens the
    learned partition and runs :func:`fit_piecewise_law`. STLSQ stays numpy;
    gates and ``xi`` are differentiable. Do not pass an oracle
    :class:`~omnibias.partition.PartitionParams` -- the split is learned.
    """
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    ua = np.asarray(u, dtype=float)
    dua = np.asarray(du, dtype=float)
    phi, _phi_names = _phi_design(ua, int(degree))
    if ua.ndim == 2:
        design, names = polynomial_vector_library(ua, int(degree))
    else:
        design, names = polynomial_value_library(ua.reshape(-1), int(degree))
    W, t, xi = _train_soft_piecewise(
        xa,
        phi,
        dua,
        n_gates=int(n_gates),
        steps=int(steps),
        lr=float(lr),
        beta=float(beta),
        l1=float(l1),
        entropy=float(entropy),
        seed=int(seed),
    )
    t = _refine_split_threshold(W, t, xa, phi, dua)
    cfg = PartitionConfig(
        n_features=int(xa.shape[1]),
        depth=int(n_gates),
        split_kind="oblique",
        beta_final=float(beta),
    )
    partition = PartitionParams(cfg, W, t)
    lhs_names = None
    if dua.ndim == 2 and dua.shape[1] > 1:
        lhs_names = tuple(f"{lhs_name}{c}" for c in range(dua.shape[1]))
    automaton = fit_piecewise_law(
        partition,
        xa,
        design,
        dua,
        names,
        lhs_name=lhs_name,
        lhs_names=lhs_names,
        min_samples=min_samples,
        alpha=alpha,
        threshold=threshold,
    )
    return automaton, {"W": W, "t": t, "xi": xi, "beta": np.asarray(float(beta))}


def soft_piecewise_forward_jax(
    W: object,
    t: object,
    xi: object,
    x: object,
    phi: object,
    beta: float,
) -> object:
    r"""JAX twin of the soft-weighted per-cell linear map (``jit`` / ``grad``)."""
    import jax.numpy as jnp
    from omnibias.partition.jax.weights import combine, partition_weights_arrays

    depth = int(W.shape[0])  # type: ignore[union-attr]
    weights = partition_weights_arrays(W, t, x, float(beta), depth)
    if jnp.ndim(xi) == 3:
        cell = jnp.einsum("ltk,nt->nlk", xi, phi)
        return combine(weights, cell)
    cell2 = jnp.einsum("lt,nt->nl", xi, phi)
    return combine(weights, cell2)


__all__ = [
    "HybridAutomaton",
    "PiecewiseLaw",
    "fit_learned_piecewise_ode",
    "fit_piecewise_law",
    "fit_piecewise_ode_law",
    "global_sparse_law",
    "polynomial_value_library",
    "polynomial_vector_library",
    "soft_piecewise_forward_jax",
    "soft_piecewise_forward_np",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Lie point-symmetry discovery from exact jets (theory 03-11).

Determining equations are linear in the generator once the
prolongation is known. Prolongation is a jet computation, so
the algebra is a nullspace. Rank needs a reported threshold
and a separation. Point symmetries only. The ansatz bounds
what can be found.

Jets come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two. Not a continuum regularity
claim.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from omnibias.core.collapse.rank import rank_collapse
from omnibias.core.proof.lift import as_fraction

FloatArray = NDArray[np.float64]
ExactVerdict = Literal["PROVED", "DISPROVED", "BLOCKED"]
DISCLAIMER = "point symmetries in the declared ansatz only; not a blow-up or regularity proof"


def honesty_payload() -> dict[str, object]:
    return {
        "point_symmetries_only": True,
        "ansatz_bounds_search": True,
        "rank_threshold_reported": True,
        "float_svd_is_proposer": True,
        "exact_rank_accept_required": True,
        "exact_rank_scope": "bounded-denominator snapped finite determining matrix",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "ns_regularity": False,
        "disclaimer": DISCLAIMER,
    }


@dataclass(frozen=True)
class LinearPoly:
    """Affine ``c0 + cx x + ct t + cu u``."""

    c0: float = 0.0
    cx: float = 0.0
    ct: float = 0.0
    cu: float = 0.0

    def __call__(self, x: float, t: float, u: float) -> float:
        return self.c0 + self.cx * x + self.ct * t + self.cu * u

    def at(self, x: complex, t: complex, u: complex) -> complex:
        return self.c0 + self.cx * x + self.ct * t + self.cu * u


@dataclass(frozen=True)
class Generator:
    xi_x: LinearPoly
    xi_t: LinearPoly
    eta: LinearPoly
    label: str = ""


@dataclass(frozen=True)
class SymmetryBasis:
    generators: tuple[Generator, ...]
    name: str = "affine_xtu"

    @property
    def n_coeff(self) -> int:
        return len(self.generators)


@dataclass(frozen=True)
class Sample:
    x: float
    t: float
    u: float
    ux: float
    ut: float
    uxx: float
    uxt: float
    utt: float
    uxxx: float = 0.0
    uxxt: float = 0.0
    uxtt: float = 0.0
    uttt: float = 0.0


@dataclass(frozen=True)
class PDESpec:
    name: str
    residual: Callable[[Sample], float]
    restrict: Callable[[Sample], Sample]
    expected_dim: int
    variational: bool = False


@dataclass(frozen=True)
class ConservedCurrent:
    density: Callable[[Sample], float]
    flux: Callable[[Sample], float]


@dataclass(frozen=True)
class ExactSymmetryReport:
    """Exact rank result for a snapped finite determining matrix.

    The float SVD proposes a dimension.  This report accepts or rejects that
    proposal only for the explicitly recorded bounded-denominator rational
    matrix.  It is not a proof that the sampled float jets, or a PDE away from
    the declared samples, satisfy the exact relation.
    """

    verdict: ExactVerdict
    kernel: tuple[tuple[int, ...], ...]
    denom_bound: int
    max_snap_error: float | None
    row_denominators: tuple[int, ...]
    max_abs_integer: int | None
    detail: str

    @property
    def algebra_dim(self) -> int:
        """Dimension of the exact kernel of the snapped matrix."""
        return len(self.kernel)

    @property
    def proved(self) -> bool:
        return self.verdict == "PROVED"

    @property
    def disproved(self) -> bool:
        return self.verdict == "DISPROVED"

    @property
    def blocked(self) -> bool:
        return self.verdict == "BLOCKED"

    def accepts_dimension(self, proposed_dim: int) -> bool:
        """Whether this finite exact check accepts an SVD rank proposal."""
        if proposed_dim < 0:
            raise ValueError("proposed_dim must be non-negative")
        if proposed_dim == 0:
            return self.disproved
        return self.proved and self.algebra_dim == proposed_dim

    def to_payload(self) -> dict[str, object]:
        """JSON-ready finite-matrix evidence, including snap provenance."""
        return {
            "verdict": self.verdict,
            "algebra_dim": self.algebra_dim,
            "kernel": [list(vector) for vector in self.kernel],
            "denom_bound": self.denom_bound,
            "max_snap_error": self.max_snap_error,
            "row_denominators": list(self.row_denominators),
            "max_abs_integer": self.max_abs_integer,
            "detail": self.detail,
            "scope": "bounded-denominator snapped finite determining matrix",
        }


@dataclass(frozen=True)
class SymmetryResult:
    generators: tuple[Generator, ...]
    singular_values: tuple[float, ...]
    rank_threshold: float
    separation: float
    basis: SymmetryBasis
    algebra_dim: int
    infinite_in_ansatz: bool
    ambiguous: bool
    verified: bool
    float_verified: bool = False
    exact: ExactSymmetryReport | None = None
    disclaimer: str = DISCLAIMER


def affine_basis() -> SymmetryBasis:
    """Worked-example ansatz: ``ξ^x, ξ^t`` affine in ``(x,t)``, ``η`` affine in ``u``."""
    gens = (
        Generator(LinearPoly(c0=1.0), LinearPoly(), LinearPoly(), "d/dx"),
        Generator(LinearPoly(cx=1.0), LinearPoly(), LinearPoly(), "x d/dx"),
        Generator(LinearPoly(ct=1.0), LinearPoly(), LinearPoly(), "t d/dx"),
        Generator(LinearPoly(), LinearPoly(c0=1.0), LinearPoly(), "d/dt"),
        Generator(LinearPoly(), LinearPoly(cx=1.0), LinearPoly(), "x d/dt"),
        Generator(LinearPoly(), LinearPoly(ct=1.0), LinearPoly(), "t d/dt"),
        Generator(LinearPoly(), LinearPoly(), LinearPoly(c0=1.0), "d/du"),
        Generator(LinearPoly(), LinearPoly(), LinearPoly(cu=1.0), "u d/du"),
    )
    return SymmetryBasis(gens, "affine_xtu")


def _dx_q(gen: Generator, s: Sample) -> float:
    eta, xx, tt = gen.eta, gen.xi_x, gen.xi_t
    return (
        eta.cx
        - xx.cx * s.ux
        - tt.cx * s.ut
        + (eta.cu - xx.cu * s.ux - tt.cu * s.ut) * s.ux
        - xx(s.x, s.t, s.u) * s.uxx
        - tt(s.x, s.t, s.u) * s.uxt
    )


def _dt_q(gen: Generator, s: Sample) -> float:
    eta, xx, tt = gen.eta, gen.xi_x, gen.xi_t
    return (
        eta.ct
        - xx.ct * s.ux
        - tt.ct * s.ut
        + (eta.cu - xx.cu * s.ux - tt.cu * s.ut) * s.ut
        - xx(s.x, s.t, s.u) * s.uxt
        - tt(s.x, s.t, s.u) * s.utt
    )


def _dxx_q(gen: Generator, s: Sample) -> float:
    eta, xx, tt = gen.eta, gen.xi_x, gen.xi_t
    return (
        -xx.cx * s.uxx
        - tt.cx * s.uxt
        + s.ux * (-xx.cu * s.uxx - tt.cu * s.uxt)
        + s.uxx * (eta.cu - xx.cx - 2.0 * xx.cu * s.ux - tt.cu * s.ut)
        + s.uxt * (-tt.cx - tt.cu * s.ux)
        - xx(s.x, s.t, s.u) * s.uxxx
        - tt(s.x, s.t, s.u) * s.uxxt
    )


def _dtt_q(gen: Generator, s: Sample) -> float:
    eta, xx, tt = gen.eta, gen.xi_x, gen.xi_t
    return (
        -xx.ct * s.uxt
        - tt.ct * s.utt
        + s.ut * (-xx.cu * s.uxt - tt.cu * s.utt)
        + s.utt * (eta.cu - tt.ct - 2.0 * tt.cu * s.ut - xx.cu * s.ux)
        + s.uxt * (-xx.ct - xx.cu * s.ut)
        - xx(s.x, s.t, s.u) * s.uxtt
        - tt(s.x, s.t, s.u) * s.uttt
    )


def _dxxx_q(gen: Generator, s: Sample) -> float:
    """Complex-step ``D_x`` of the closed-form ``D_x D_x Q``."""
    h = 1e-20

    # D_x acts on the x-jet chain: bump each coordinate that D_x hits.
    def bump(key: str, step: complex) -> Sample:
        kw = s.__dict__.copy()
        kw[key] = kw[key] + step
        return Sample(**kw)

    def dxx_at(step: complex) -> complex:
        return _dxx_q(gen, bump("x", step))

    acc = (dxx_at(1j * h) - dxx_at(0.0)).imag / h
    for key, chain in (
        ("u", s.ux),
        ("ux", s.uxx),
        ("ut", s.uxt),
        ("uxx", s.uxxx),
        ("uxt", s.uxxt),
        ("utt", s.uxtt),
        ("uxxx", 0.0),
        ("uxxt", 0.0),
    ):
        if chain == 0.0 and key not in {"u", "ux", "uxx"}:
            continue
        plus = _dxx_q(gen, bump(key, 1j * h))
        acc += (complex(plus).imag / h) * chain
    return float(acc.real if isinstance(acc, complex) else acc)


def eta_t(gen: Generator, s: Sample) -> float:
    return _dt_q(gen, s) + gen.xi_x(s.x, s.t, s.u) * s.uxt + gen.xi_t(s.x, s.t, s.u) * s.utt


def eta_x(gen: Generator, s: Sample) -> float:
    return _dx_q(gen, s) + gen.xi_x(s.x, s.t, s.u) * s.uxx + gen.xi_t(s.x, s.t, s.u) * s.uxt


def eta_xx(gen: Generator, s: Sample) -> float:
    return _dxx_q(gen, s) + gen.xi_x(s.x, s.t, s.u) * s.uxxx + gen.xi_t(s.x, s.t, s.u) * s.uxxt


def eta_tt(gen: Generator, s: Sample) -> float:
    return _dtt_q(gen, s) + gen.xi_x(s.x, s.t, s.u) * s.uxtt + gen.xi_t(s.x, s.t, s.u) * s.uttt


def eta_xxx(gen: Generator, s: Sample) -> float:
    return _dxxx_q(gen, s) + gen.xi_x(s.x, s.t, s.u) * 0.0 + gen.xi_t(s.x, s.t, s.u) * 0.0


def pr_heat(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) - eta_xx(gen, s)


def pr_wave(gen: Generator, s: Sample) -> float:
    return eta_tt(gen, s) - eta_xx(gen, s)


def pr_transport(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) + eta_x(gen, s)


def pr_burgers(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) + gen.eta(s.x, s.t, s.u) * s.ux + s.u * eta_x(gen, s) - eta_xx(gen, s)


def pr_laplace(gen: Generator, s: Sample) -> float:
    return eta_xx(gen, s) + eta_tt(gen, s)


def pr_fisher(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) - eta_xx(gen, s) - gen.eta(s.x, s.t, s.u) * (1.0 - 2.0 * s.u)


def pr_kdv(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) + gen.eta(s.x, s.t, s.u) * s.ux + s.u * eta_x(gen, s) + eta_xxx(gen, s)


def pr_kdv_linear(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) + eta_xxx(gen, s)


def pr_boussinesq(gen: Generator, s: Sample) -> float:
    eta = gen.eta(s.x, s.t, s.u)
    return (
        eta_tt(gen, s)
        - eta_xx(gen, s)
        - 2.0 * eta * s.uxx
        - 2.0 * s.u * eta_xx(gen, s)
        - 4.0 * s.ux * eta_x(gen, s)
    )


def pr_negative(gen: Generator, s: Sample) -> float:
    return eta_t(gen, s) - eta_xx(gen, s) - 3.0 * s.u**2 * gen.eta(s.x, s.t, s.u) - (
        2.0 * s.x * s.t * gen.xi_x(s.x, s.t, s.u) + s.x**2 * gen.xi_t(s.x, s.t, s.u)
    )


def _restrict_heat(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, s.uxx, s.uxx, s.uxxx, s.uxxt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_wave(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, s.ut, s.uxx, s.uxt, s.uxx, s.uxxx, s.uxxt, s.uxtt, s.uxxt)


def _restrict_transport(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, -s.ux, s.uxx, -s.uxx, s.uxx, s.uxxx, -s.uxxx, s.uxxx, -s.uxxx)


def _restrict_burgers(s: Sample) -> Sample:
    ut = s.uxx - s.u * s.ux
    return Sample(s.x, s.t, s.u, s.ux, ut, s.uxx, s.uxt, s.utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_laplace(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, s.ut, s.uxx, s.uxt, -s.uxx, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_fisher(s: Sample) -> Sample:
    ut = s.uxx + s.u * (1.0 - s.u)
    return Sample(s.x, s.t, s.u, s.ux, ut, s.uxx, s.uxt, s.utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_kdv(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, -s.uxxx - s.u * s.ux, s.uxx, s.uxt, s.utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_kdv_linear(s: Sample) -> Sample:
    return Sample(s.x, s.t, s.u, s.ux, -s.uxxx, s.uxx, s.uxt, s.utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_boussinesq(s: Sample) -> Sample:
    utt = s.uxx + 2.0 * s.u * s.uxx + 2.0 * s.ux**2
    return Sample(s.x, s.t, s.u, s.ux, s.ut, s.uxx, s.uxt, utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def _restrict_negative(s: Sample) -> Sample:
    ut = s.uxx + s.u**3 + s.x**2 * s.t
    return Sample(s.x, s.t, s.u, s.ux, ut, s.uxx, s.uxt, s.utt, s.uxxx, s.uxxt, s.uxtt, s.uttt)


def designed_samples(n: int = 24, *, seed: int = 0) -> tuple[Sample, ...]:
    """Deterministic dyadic sample stencil for float proposal and Q acceptance.

    Every coordinate is a dyadic rational.  The determining equations in this
    module use only rational arithmetic, so their float evaluations can be
    reconstructed exactly by the bounded-denominator acceptance gate.  ``seed``
    rotates the finite stencil; it does not turn it into an untracked random
    cloud.
    """
    if n < 1:
        raise ValueError("n must be positive")
    out: list[Sample] = []
    for i in range(n):
        index = i + 17 * seed
        x = float(((index * 1) % 7 - 3) / 4)
        t = float(((index * 3) % 9 - 4) / 8)
        u = float(((index * 5) % 11 - 5) / 8)
        ux = float(((index * 7) % 13 - 6) / 8)
        uxx = float(((index * 11) % 15 - 7) / 8)
        uxt = float(((index * 13) % 17 - 8) / 8)
        utt = float(((index * 17) % 19 - 9) / 8)
        uxxx = float(((index * 19) % 21 - 10) / 8)
        uxxt = float(((index * 23) % 23 - 11) / 8)
        uxtt = float(((index * 29) % 25 - 12) / 8)
        uttt = float(((index * 31) % 27 - 13) / 8)
        out.append(Sample(x, t, u, ux, 0.0, uxx, uxt, utt, uxxx, uxxt, uxtt, uttt))
    return tuple(out)


def matrix_condition(mat: FloatArray) -> float:
    """Condition number of the nonzero singular-value block."""
    svals = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
    if svals.size == 0 or float(svals[0]) <= 0.0:
        return float("inf")
    kept = svals[svals > 1e-8 * float(svals[0])]
    if kept.size < 2:
        return 1.0
    return float(kept[0] / kept[-1])


def determining_matrix(
    pr: Callable[[Generator, Sample], float],
    restrict: Callable[[Sample], Sample],
    basis: SymmetryBasis,
    samples: Sequence[Sample],
) -> FloatArray:
    rows = []
    for raw in samples:
        s = restrict(raw)
        rows.append([float(pr(g, s)) for g in basis.generators])
    return np.asarray(rows, dtype=np.float64)


def exact_symmetry_report(
    matrix: Sequence[Sequence[float]],
    *,
    denom_bound: int = 1_000_000,
) -> ExactSymmetryReport:
    """Snap, integerize, and exactly adjudicate one finite determining matrix.

    Each float entry is snapped with :func:`as_fraction` at ``denom_bound``.
    Denominators are cleared independently per row, which preserves the
    nullspace while avoiding an unnecessary global common denominator.  The
    resulting integer matrix is accepted by :func:`rank_collapse`.

    A failure to construct this finite rational presentation returns
    ``BLOCKED`` rather than treating a float SVD as proof.
    """
    if denom_bound < 1:
        raise ValueError("denom_bound must be positive")
    try:
        rows = list(matrix)
        if not rows:
            raise ValueError("matrix must be non-empty")
        width: int | None = None
        integer_rows: list[list[int]] = []
        row_denominators: list[int] = []
        max_snap_error = 0.0
        for row in rows:
            values = list(row)
            if not values:
                raise ValueError("matrix rows must be non-empty")
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise ValueError("matrix rows must share a width")
            fractions: list[Fraction] = []
            for value in values:
                original = float(value)
                if not math.isfinite(original):
                    raise ValueError("matrix entries must be finite")
                snapped = as_fraction(original, denom_bound=denom_bound)
                fractions.append(snapped)
                max_snap_error = max(max_snap_error, abs(original - float(snapped)))
            denominator = 1
            for fraction in fractions:
                denominator = math.lcm(denominator, fraction.denominator)
            integers = [int(value * denominator) for value in fractions]
            common = 0
            for value in integers:
                common = math.gcd(common, abs(value))
            if common > 1:
                integers = [value // common for value in integers]
            integer_rows.append(integers)
            row_denominators.append(denominator)
        rank = rank_collapse(integer_rows)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        return ExactSymmetryReport(
            verdict="BLOCKED",
            kernel=(),
            denom_bound=denom_bound,
            max_snap_error=None,
            row_denominators=(),
            max_abs_integer=None,
            detail=f"could not construct an exact snapped matrix: {exc}",
        )
    return ExactSymmetryReport(
        verdict=rank.verdict.status,
        kernel=rank.kernel,
        denom_bound=denom_bound,
        max_snap_error=max_snap_error,
        row_denominators=tuple(row_denominators),
        max_abs_integer=max(abs(value) for row in integer_rows for value in row),
        detail=rank.verdict.detail,
    )


def _svd_dim(svals: FloatArray, threshold: float) -> tuple[int, float, bool]:
    s = np.sort(np.asarray(svals, dtype=np.float64))[::-1]
    above = s[s > threshold]
    below = s[s <= threshold]
    dim = int(below.size)
    if above.size == 0 or below.size == 0:
        sep = math.inf if below.size == s.size or above.size == s.size else 0.0
        return dim, float(sep), True
    sep = float(above[-1] / max(below[0], 1e-30))
    return dim, sep, False


def discover_symmetries(
    pr: Callable[[Generator, Sample], float],
    restrict: Callable[[Sample], Sample],
    basis: SymmetryBasis,
    samples: Sequence[Sample],
    *,
    threshold: float | Literal["auto"] = "auto",
) -> SymmetryResult:
    mat = determining_matrix(pr, restrict, basis, samples)
    svals = np.linalg.svd(mat, compute_uv=False)
    if threshold == "auto":
        if svals.size == 0 or float(svals[0]) <= 0.0:
            thr = 1e-12
        else:
            log = np.log10(np.maximum(svals, 1e-18))
            gaps = log[:-1] - log[1:]
            if gaps.size and float(gaps.max()) >= 6.0:
                cut = int(np.argmax(gaps)) + 1
                thr = float(math.sqrt(max(float(svals[cut - 1]), 1e-30) * max(float(svals[cut]), 1e-30)))
            else:
                thr = float(svals[0]) * 1e-8
    else:
        thr = float(threshold)
    dim, sep, amb = _svd_dim(svals, thr)
    _, _, vt = np.linalg.svd(mat, full_matrices=True)
    gens: list[Generator] = []
    scale = max(1.0, float(np.linalg.norm(mat)))
    if dim:
        for row in vt[-dim:]:
            if float(np.linalg.norm(mat @ row)) > 1e-8 * scale:
                continue
            gens.append(_combine(basis, row))
    float_verified = all(
        float(np.linalg.norm(mat @ _coeff_of(g, basis))) <= 1e-8 * scale for g in gens
    )
    exact = exact_symmetry_report(mat.tolist())
    # The SVD is a proposer.  ``verified`` names only agreement with the exact
    # bounded-denominator finite-matrix check, never a float residual.
    verified = exact.accepts_dimension(int(dim))
    infinite = dim >= basis.n_coeff - 1
    return SymmetryResult(
        tuple(gens),
        tuple(float(v) for v in svals),
        thr,
        sep,
        basis,
        int(dim),
        infinite,
        amb and sep < 1e6,
        verified,
        float_verified,
        exact,
    )


def _combine(basis: SymmetryBasis, coeff: FloatArray) -> Generator:
    xx = LinearPoly(
        float(sum(c * g.xi_x.c0 for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_x.cx for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_x.ct for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_x.cu for c, g in zip(coeff, basis.generators, strict=True))),
    )
    tt = LinearPoly(
        float(sum(c * g.xi_t.c0 for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_t.cx for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_t.ct for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.xi_t.cu for c, g in zip(coeff, basis.generators, strict=True))),
    )
    eta = LinearPoly(
        float(sum(c * g.eta.c0 for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.eta.cx for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.eta.ct for c, g in zip(coeff, basis.generators, strict=True))),
        float(sum(c * g.eta.cu for c, g in zip(coeff, basis.generators, strict=True))),
    )
    return Generator(xx, tt, eta, "linear_combo")


def _coeff_of(gen: Generator, basis: SymmetryBasis) -> FloatArray:
    out = []
    for g in basis.generators:
        if g.xi_x.c0:
            out.append(gen.xi_x.c0)
        elif g.xi_x.cx:
            out.append(gen.xi_x.cx)
        elif g.xi_x.ct:
            out.append(gen.xi_x.ct)
        elif g.xi_t.c0:
            out.append(gen.xi_t.c0)
        elif g.xi_t.cx:
            out.append(gen.xi_t.cx)
        elif g.xi_t.ct:
            out.append(gen.xi_t.ct)
        elif g.eta.c0:
            out.append(gen.eta.c0)
        elif g.eta.cu:
            out.append(gen.eta.cu)
        else:
            out.append(0.0)
    return np.asarray(out, dtype=np.float64)


def heat_fd_samples(n: int = 16, *, dx: float = 0.25, dt: float = 0.25) -> tuple[Sample, ...]:
    """Finite-difference jets of ``u = e^{-t} sin(x)``. Coarse by design."""

    def u(x: float, t: float) -> float:
        return math.exp(-t) * math.sin(x)

    out = []
    for i in range(n):
        x = 0.3 + 0.2 * i
        t = 0.2
        ux = (u(x + dx, t) - u(x - dx, t)) / (2.0 * dx)
        ut = (u(x, t + dt) - u(x, t - dt)) / (2.0 * dt)
        uxx = (u(x + dx, t) - 2.0 * u(x, t) + u(x - dx, t)) / (dx * dx)
        uxt = (u(x + dx, t + dt) - u(x + dx, t - dt) - u(x - dx, t + dt) + u(x - dx, t - dt)) / (4.0 * dx * dt)
        utt = (u(x, t + dt) - 2.0 * u(x, t) + u(x, t - dt)) / (dt * dt)
        uxxx = (u(x + 1.5 * dx, t) - 3.0 * u(x + 0.5 * dx, t) + 3.0 * u(x - 0.5 * dx, t) - u(x - 1.5 * dx, t)) / (dx**3)
        out.append(Sample(x, t, u(x, t), ux, ut, uxx, uxt, utt, uxxx, 0.0, 0.0, 0.0))
    return tuple(out)


def heat_exact_samples(n: int = 16) -> tuple[Sample, ...]:
    out = []
    for i in range(n):
        x = 0.3 + 0.2 * i
        t = 0.2
        e = math.exp(-t)
        out.append(
            Sample(
                x,
                t,
                e * math.sin(x),
                e * math.cos(x),
                -e * math.sin(x),
                -e * math.sin(x),
                -e * math.cos(x),
                e * math.sin(x),
                -e * math.cos(x),
                e * math.sin(x),
                e * math.cos(x),
                -e * math.sin(x),
            )
        )
    return tuple(out)


def _q(gen: Generator, s: Sample) -> float:
    return gen.eta(s.x, s.t, s.u) - gen.xi_x(s.x, s.t, s.u) * s.ux - gen.xi_t(s.x, s.t, s.u) * s.ut


def _shift(s: Sample, *, dx: float = 0.0, dt: float = 0.0) -> Sample:
    return Sample(
        s.x + dx,
        s.t + dt,
        s.u + s.ux * dx + s.ut * dt,
        s.ux + s.uxx * dx + s.uxt * dt,
        s.ut + s.uxt * dx + s.utt * dt,
        s.uxx + s.uxxx * dx + s.uxxt * dt,
        s.uxt + s.uxxt * dx + s.uxtt * dt,
        s.utt + s.uxtt * dx + s.uttt * dt,
        s.uxxx,
        s.uxxt,
        s.uxtt,
        s.uttt,
    )


def pr_heat_fd(gen: Generator, s: Sample, *, step: float = 0.08) -> float:
    """Same determining residual as ``pr_heat``, but total derivatives are centered FD."""
    h = float(step)
    dt_q = (_q(gen, _shift(s, dt=h)) - _q(gen, _shift(s, dt=-h))) / (2.0 * h)
    dxx_q = (_q(gen, _shift(s, dx=h)) - 2.0 * _q(gen, s) + _q(gen, _shift(s, dx=-h))) / (h * h)
    eta_t_fd = dt_q + gen.xi_x(s.x, s.t, s.u) * s.uxt + gen.xi_t(s.x, s.t, s.u) * s.utt
    eta_xx_fd = dxx_q + gen.xi_x(s.x, s.t, s.u) * s.uxxx + gen.xi_t(s.x, s.t, s.u) * s.uxxt
    return eta_t_fd - eta_xx_fd


def wave_solution_sample(x: float, t: float) -> Sample:
    """Exact 1-jet of ``u = sin(x - t)``."""
    phase = x - t
    s, c = math.sin(phase), math.cos(phase)
    return Sample(x, t, s, c, -c, -s, s, -s, -c, c, -c, s)


def noether_wave_current(gen: Generator, s: Sample) -> tuple[float, float]:
    """First-order Noether current for ``L = (u_t^2 - u_x^2)/2``."""
    q = _q(gen, s)
    lag = 0.5 * (s.ut**2 - s.ux**2)
    density = -s.ut * q - gen.xi_t(s.x, s.t, s.u) * lag
    flux = s.ux * q - gen.xi_x(s.x, s.t, s.u) * lag
    return float(density), float(flux)


def noether_wave_residual(gen: Generator, *, x: float = 0.4, t: float = 0.3) -> float:
    """``D_t P^t + D_x P^x`` on ``u = sin(x-t)``, complex-step total derivatives."""
    h = 1e-20
    dt = (_complex_density(gen, x, t + 1j * h) - _complex_density(gen, x, t)).imag / h
    dx = (_complex_flux(gen, x + 1j * h, t) - _complex_flux(gen, x, t)).imag / h
    return float(abs(dt + dx))


def _complex_density(gen: Generator, x: complex, t: complex) -> complex:
    phase = x - t
    sn, cs = cmath.sin(phase), cmath.cos(phase)
    q = gen.eta.at(x, t, sn) - gen.xi_x.at(x, t, sn) * cs - gen.xi_t.at(x, t, sn) * (-cs)
    lag = 0.5 * ((-cs) ** 2 - cs**2)
    return cs * q - gen.xi_t.at(x, t, sn) * lag


def _complex_flux(gen: Generator, x: complex, t: complex) -> complex:
    phase = x - t
    sn, cs = cmath.sin(phase), cmath.cos(phase)
    q = gen.eta.at(x, t, sn) - gen.xi_x.at(x, t, sn) * cs - gen.xi_t.at(x, t, sn) * (-cs)
    lag = 0.5 * ((-cs) ** 2 - cs**2)
    return cs * q - gen.xi_x.at(x, t, sn) * lag


def pr_for(name: str) -> Callable[[Generator, Sample], float]:
    return {
        "heat": pr_heat,
        "wave": pr_wave,
        "burgers": pr_burgers,
        "kdv": pr_kdv,
        "laplace": pr_laplace,
        "fisher": pr_fisher,
        "kdv_linear": pr_kdv_linear,
        "boussinesq": pr_boussinesq,
        "negative": pr_negative,
    }[name]


def suite() -> tuple[PDESpec, ...]:
    """Eight classical equations. Dimensions are in-ansatz, not the full published algebras."""
    return (
        PDESpec("heat", lambda s: s.ut - s.uxx, _restrict_heat, 5, False),
        PDESpec("wave", lambda s: s.utt - s.uxx, _restrict_wave, 6, True),
        PDESpec("burgers", lambda s: s.ut + s.u * s.ux - s.uxx, _restrict_burgers, 4, False),
        PDESpec("kdv", lambda s: s.ut + s.u * s.ux + s.uxxx, _restrict_kdv, 4, False),
        PDESpec("laplace", lambda s: s.uxx + s.utt, _restrict_laplace, 6, True),
        PDESpec("fisher", lambda s: s.ut - s.uxx - s.u * (1.0 - s.u), _restrict_fisher, 2, False),
        PDESpec("kdv_linear", lambda s: s.ut + s.uxxx, _restrict_kdv_linear, 5, False),
        PDESpec("boussinesq", lambda s: s.utt - s.uxx - 2.0 * s.u * s.uxx - 2.0 * s.ux**2, _restrict_boussinesq, 4, True),
    )


def negative_control() -> PDESpec:
    return PDESpec("negative", lambda s: s.ut - s.uxx - s.u**3 - s.x**2 * s.t, _restrict_negative, 0, False)


def heat_known_coeffs() -> FloatArray:
    """Published heat generators that sit inside ``affine_basis``."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


__all__ = [
    "ConservedCurrent",
    "DISCLAIMER",
    "ExactSymmetryReport",
    "Generator",
    "LinearPoly",
    "PDESpec",
    "Sample",
    "SymmetryBasis",
    "SymmetryResult",
    "affine_basis",
    "designed_samples",
    "determining_matrix",
    "discover_symmetries",
    "eta_t",
    "eta_tt",
    "eta_x",
    "eta_xx",
    "eta_xxx",
    "exact_symmetry_report",
    "heat_exact_samples",
    "heat_fd_samples",
    "heat_known_coeffs",
    "honesty_payload",
    "matrix_condition",
    "negative_control",
    "noether_wave_residual",
    "pr_boussinesq",
    "pr_burgers",
    "pr_fisher",
    "pr_for",
    "pr_heat",
    "pr_heat_fd",
    "pr_kdv",
    "pr_kdv_linear",
    "pr_laplace",
    "pr_negative",
    "pr_transport",
    "pr_wave",
    "suite",
]

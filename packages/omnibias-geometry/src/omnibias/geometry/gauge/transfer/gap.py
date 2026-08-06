# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified spectral gaps of a finite lattice transfer matrix.

Composes the rigorous engines in :mod:`omnibias.core.verified.eig` onto the
transfer matrices built in :mod:`.matrices`.  Two complementary directions:

* :func:`certified_transfer_matrix_gap` and
  :func:`certified_multistep_gap_refinement` produce a **lower** bound on the
  lattice-unit mass gap ``m a = -ln(|lambda_1| / lambda_0)``, from the symmetric
  power-sum engine (with the partner chain) or Birkhoff-Hopf, whichever is both
  applicable and tighter.
* :func:`certified_effective_mass_curve` produces rigorous **upper** bounds from
  the closed-form spectrum, decreasing toward the true gap.

Together they sandwich the true gap, so the looseness of a lower bound is itself
measurable rather than a matter of opinion.

Scope, non-negotiable
---------------------
Every quantity here is a statement about **one fixed finite matrix at one fixed
lattice spacing**.  :func:`heat_kernel_gap_scaling_report` collects such statements
across several spacings; that is *evidence about a trend*, not a continuum limit,
and nothing in this module is a claim about Yang-Mills.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from omnibias.core.verified.eig import (
    certified_perron_spectral_gap,
    certified_symmetric_spectral_gap,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import IntervalMatrix, matmul, to_interval_matrix
from omnibias.core.verified.transcend import ln_iv
from omnibias.geometry.gauge.transfer.matrices import TransferMatrix

#: Method tags a gap result can carry.
SYMMETRIC_METHOD = "symmetric_power_sum_partner_chain"
BIRKHOFF_METHOD = "birkhoff_hopf_projective_contraction"


@dataclass(frozen=True)
class GapCandidate:
    """One engine's attempt at the gap, kept even when it loses the comparison."""

    method: str
    subdominant_ratio_upper: float
    spectral_gap_lower: float
    partners_deflated: int = 0
    detail: str = ""


@dataclass(frozen=True)
class TransferGapResult:
    """A certified **lower** bound on the lattice-unit mass gap of a fixed matrix.

    ``spectral_gap_lower`` is ``-ln(subdominant_ratio_upper)``: since the ratio
    bound holds, so does the gap bound.  ``spectral_gap_lower_per_unit`` divides by
    the lattice spacing so gaps at different spacings are comparable -- which is a
    *unit conversion*, not an extrapolation.
    """

    model: str
    basis: str
    dimension: int
    method: str
    subdominant_ratio_upper: float
    spectral_gap_lower: float
    spectral_gap_lower_per_unit: float
    lattice_spacing: float
    partners_deflated: int
    candidates: tuple[GapCandidate, ...] = field(default_factory=tuple)

    @property
    def certified(self) -> bool:
        """Whether a strictly positive gap was established."""
        return self.spectral_gap_lower > 0.0


def _spacing(transfer: TransferMatrix, lattice_spacing: float | None) -> float:
    if lattice_spacing is not None:
        return float(lattice_spacing)
    recorded = transfer.parameters.get("lattice_spacing", 1.0)
    if not isinstance(recorded, int | float):
        raise ValueError(f"recorded lattice_spacing is not numeric: {recorded!r}")
    return float(recorded)


def _symmetric_candidate(
    matrix: IntervalMatrix,
    transfer: TransferMatrix,
    spacing: float,
    deflate: bool,
) -> GapCandidate | None:
    partners = transfer.subdominant_vectors if deflate else None
    try:
        cert = certified_symmetric_spectral_gap(
            matrix,
            list(transfer.perron_vector),
            subdominant_vectors=[list(v) for v in partners] if partners else None,
            lattice_spacing=spacing,
        )
    except ValueError as exc:
        return GapCandidate(
            method=SYMMETRIC_METHOD,
            subdominant_ratio_upper=float("inf"),
            spectral_gap_lower=0.0,
            detail=f"not applicable: {exc}",
        )
    return GapCandidate(
        method=SYMMETRIC_METHOD,
        subdominant_ratio_upper=float(cert.subdominant_ratio_upper),
        spectral_gap_lower=float(cert.spectral_gap_lower),
        partners_deflated=int(cert.partners_deflated),
        detail=f"deflated {cert.partners_deflated} partner(s)",
    )


def _birkhoff_candidate(matrix: IntervalMatrix, spacing: float) -> GapCandidate:
    try:
        cert = certified_perron_spectral_gap(matrix, lattice_spacing=spacing)
    except ValueError as exc:
        return GapCandidate(
            method=BIRKHOFF_METHOD,
            subdominant_ratio_upper=float("inf"),
            spectral_gap_lower=0.0,
            detail=f"not applicable: {exc}",
        )
    return GapCandidate(
        method=BIRKHOFF_METHOD,
        subdominant_ratio_upper=float(cert.subdominant_ratio_upper),
        spectral_gap_lower=float(cert.spectral_gap_lower),
        detail=f"projective diameter <= {cert.kappa_upper:.6g}",
    )


def certified_transfer_matrix_gap(
    transfer: TransferMatrix,
    *,
    lattice_spacing: float | None = None,
    deflate: bool = True,
) -> TransferGapResult:
    r"""Certify a lower bound on ``m a = -ln(|lambda_1| / lambda_0)`` for a fixed matrix.

    Runs every applicable engine and keeps the **tightest** (largest) certified
    lower bound.  All attempts are reported in ``candidates``, so a bound that
    looks weak can be traced to the engine that produced it rather than guessed at.

    * The symmetric power-sum engine applies to any symmetric matrix and is
      sharpened by the partner chain from
      :attr:`~.matrices.TransferMatrix.subdominant_vectors`, which removes both the
      ``sqrt(multiplicity)`` inflation of a degenerate subdominant mode and the
      pollution from the tail behind it.
    * Birkhoff-Hopf applies only to an **entrywise-positive** matrix (so, of the
      constructions here, the ``angle``-basis circulant).  It assumes nothing about
      symmetry, and is deliberately conservative.

    Taking the max is sound because each candidate is independently a valid lower
    bound; picking the largest of several valid lower bounds is still valid.
    """
    matrix = to_interval_matrix(transfer.matrix())
    spacing = _spacing(transfer, lattice_spacing)
    candidates: list[GapCandidate] = []
    if transfer.symmetric:
        symmetric = _symmetric_candidate(matrix, transfer, spacing, deflate)
        if symmetric is not None:
            candidates.append(symmetric)
    if transfer.entrywise_positive:
        candidates.append(_birkhoff_candidate(matrix, spacing))
    if not candidates:
        raise ValueError(
            "no certified gap engine applies: the matrix is neither marked "
            "symmetric nor certifiably entrywise positive"
        )
    best = max(candidates, key=lambda c: c.spectral_gap_lower)
    gap = best.spectral_gap_lower
    return TransferGapResult(
        model=transfer.model,
        basis=transfer.basis,
        dimension=transfer.dimension,
        method=best.method,
        subdominant_ratio_upper=best.subdominant_ratio_upper,
        spectral_gap_lower=gap,
        spectral_gap_lower_per_unit=gap / spacing if math.isfinite(gap) else gap,
        lattice_spacing=spacing,
        partners_deflated=best.partners_deflated,
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class MultistepGapResult:
    """The sharpened gap from certifying ``T^n`` instead of ``T``."""

    model: str
    best_power: int
    subdominant_ratio_upper: float
    spectral_gap_lower: float
    spectral_gap_lower_per_unit: float
    lattice_spacing: float
    per_power: tuple[tuple[int, float], ...]

    @property
    def certified(self) -> bool:
        return self.spectral_gap_lower > 0.0


def certified_multistep_gap_refinement(
    transfer: TransferMatrix,
    *,
    max_power: int = 4,
    lattice_spacing: float | None = None,
) -> MultistepGapResult:
    r"""Sharpen the gap by certifying ``T^n`` and taking the ``n``-th root.

    ``T^n`` has eigenvalues ``lambda_i^n``, so a certified ratio bound
    ``|lambda_1|^n / lambda_0^n <= tau_n`` gives ``|lambda_1| / lambda_0 <=
    tau_n^{1/n}`` and therefore ``m a >= -ln(tau_n) / n``.

    This helps because the *pollution* a power-sum bound suffers from the tail
    behind the subdominant mode shrinks geometrically under powering: the ratios
    ``(lambda_i / lambda_1)^n`` for ``i > 1`` collapse, so at large ``n`` the bound
    approaches the true ratio.  Reports the best power found and the per-power
    curve, which is what tells you whether powering was worth its cost.
    """
    if max_power < 1:
        raise ValueError(f"max_power must be >= 1, got {max_power}")
    spacing = _spacing(transfer, lattice_spacing)
    base = to_interval_matrix(transfer.matrix())
    powered = base
    curve: list[tuple[int, float]] = []
    best_gap = 0.0
    best_power = 1
    best_ratio = float("inf")
    for power in range(1, max_power + 1):
        if power > 1:
            powered = matmul(powered, base)
        stepped = TransferMatrix(
            model=transfer.model,
            basis=transfer.basis,
            entries=tuple(tuple(row) for row in powered),
            mode_labels=transfer.mode_labels,
            exact_eigenvalues=None,
            parameters=dict(transfer.parameters),
            perron_vector=transfer.perron_vector,
            subdominant_vectors=transfer.subdominant_vectors,
            symmetric=transfer.symmetric,
        )
        result = certified_transfer_matrix_gap(stepped, lattice_spacing=spacing)
        # gap(T) = gap(T^n) / n, because the ratio is raised to the n-th power.
        gap = result.spectral_gap_lower / power
        curve.append((power, gap))
        if gap > best_gap:
            best_gap = gap
            best_power = power
            best_ratio = (
                result.subdominant_ratio_upper ** (1.0 / power)
                if math.isfinite(result.subdominant_ratio_upper)
                else float("inf")
            )
    return MultistepGapResult(
        model=transfer.model,
        best_power=best_power,
        subdominant_ratio_upper=best_ratio,
        spectral_gap_lower=best_gap,
        spectral_gap_lower_per_unit=best_gap / spacing,
        lattice_spacing=spacing,
        per_power=tuple(curve),
    )


@dataclass(frozen=True)
class EffectiveMassPoint:
    """One rigorous ``m_eff(tau)`` enclosure."""

    tau: int
    lower: float
    upper: float


@dataclass(frozen=True)
class EffectiveMassCurve:
    """Rigorous **upper** bounds on the mass gap, decreasing in ``tau``.

    The interval twin of the statistical ``effective_mass`` in
    :mod:`omnibias.geometry.gauge.lattice`: same definition
    ``m_eff(tau) = -ln(C(tau+1) / C(tau))``, but on an enclosed correlator built
    from the closed-form spectrum rather than on a Monte-Carlo estimate, so each
    point is a guaranteed enclosure instead of a sample with an error bar.
    """

    model: str
    points: tuple[EffectiveMassPoint, ...]

    @property
    def gap_upper(self) -> float:
        """The tightest certified upper bound on the gap (the last point's ``upper``)."""
        return min((p.upper for p in self.points), default=float("inf"))


def certified_effective_mass_curve(
    transfer: TransferMatrix,
    *,
    taus: Sequence[int] = (1, 2, 4, 8, 16),
) -> EffectiveMassCurve:
    r"""Rigorous ``m_eff(tau)`` enclosures from the closed-form spectrum.

    With unit overlaps onto every excited state the vacuum-subtracted correlator is
    ``C(tau) = sum_{i >= 1} (lambda_i / lambda_0)^tau``, so

    .. math::  m_{\mathrm{eff}}(\tau) = -\ln\frac{C(\tau+1)}{C(\tau)} \ \ge\ m a,

    because ``C(tau+1)/C(tau)`` is a weighted average of ratios all bounded by
    ``lambda_1 / lambda_0``.  Every point is therefore an **upper** bound on the
    gap, falling monotonically toward it -- the natural partner to the certified
    lower bound from :func:`certified_transfer_matrix_gap`, and the instrument that
    makes "how loose is that bound?" a measurable question.

    Requires a matrix whose spectrum is known in closed form; raises otherwise.
    """
    spectrum = transfer.exact_eigenvalues
    if spectrum is None or len(spectrum) < 2:
        raise ValueError(
            "certified_effective_mass_curve needs a closed-form spectrum of at "
            "least two eigenvalues"
        )
    dominant = spectrum[0]
    if dominant.lo <= 0.0:
        raise ValueError("dominant eigenvalue must be certifiably positive")
    ratios = [value / dominant for value in spectrum[1:]]

    def correlator(tau: int) -> Interval:
        total = Interval.point(0.0)
        for ratio in ratios:
            total = total + ratio.pow_int(tau)
        return total

    points: list[EffectiveMassPoint] = []
    for tau in taus:
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        here = correlator(tau)
        nxt = correlator(tau + 1)
        if here.lo <= 0.0 or nxt.lo <= 0.0:
            continue  # underflowed to an interval containing 0; nothing to say
        mass = -ln_iv(nxt / here)
        points.append(EffectiveMassPoint(tau=tau, lower=mass.lo, upper=mass.hi))
    return EffectiveMassCurve(model=transfer.model, points=tuple(points))


@dataclass(frozen=True)
class ScalingPoint:
    """The certified gap at one lattice spacing."""

    lattice_spacing: float
    coupling: float
    spectral_gap_lower: float
    spectral_gap_lower_per_unit: float
    method: str


@dataclass(frozen=True)
class ScalingReport:
    """How the certified gap behaves as the spacing shrinks -- **evidence, not a limit**.

    Each point is an independent fixed-matrix certificate.  A trend across them is
    suggestive and nothing more: no statement here survives the limit ``a -> 0``,
    and ``continuum_claim`` is always ``False``.
    """

    model: str
    points: tuple[ScalingPoint, ...]
    continuum_claim: bool = False
    note: str = (
        "each point is an independent fixed-spacing, finite-dimension certificate; "
        "the trend across them is evidence about a sequence of lattices and is NOT "
        "a continuum-limit or uniform-in-spacing claim"
    )

    @property
    def monotone_per_unit(self) -> bool:
        """Whether the per-unit gap increased at every step (a descriptive fact only)."""
        values = [p.spectral_gap_lower_per_unit for p in self.points]
        return all(b > a for a, b in zip(values, values[1:], strict=False))


def heat_kernel_gap_scaling_report(
    build: object,
    *,
    spacings: Sequence[float],
    couplings: Sequence[float],
    **build_kwargs: object,
) -> ScalingReport:
    r"""Certify the gap at several ``(spacing, coupling)`` pairs and tabulate them.

    ``build`` is one of the constructors in :mod:`.matrices`; it is called once per
    pair as ``build(coupling, lattice_spacing=a, **build_kwargs)``.  Supplying the
    coupling per spacing explicitly (rather than inferring a scaling law) keeps the
    physics input visible instead of hidden in this function.

    The output is a table of independent certificates.  Reading a continuum mass
    off it would be unjustified, which is why :class:`ScalingReport` hard-wires
    ``continuum_claim = False``.
    """
    if len(spacings) != len(couplings):
        raise ValueError(
            f"spacings and couplings must be the same length, got "
            f"{len(spacings)} and {len(couplings)}"
        )
    if not spacings:
        raise ValueError("need at least one (spacing, coupling) pair")
    if not callable(build):
        raise TypeError("build must be a transfer-matrix constructor")
    points: list[ScalingPoint] = []
    model = ""
    for spacing, coupling in zip(spacings, couplings, strict=True):
        transfer = build(coupling, lattice_spacing=spacing, **build_kwargs)
        model = transfer.model
        result = certified_transfer_matrix_gap(transfer, lattice_spacing=spacing)
        points.append(
            ScalingPoint(
                lattice_spacing=float(spacing),
                coupling=float(coupling),
                spectral_gap_lower=result.spectral_gap_lower,
                spectral_gap_lower_per_unit=result.spectral_gap_lower_per_unit,
                method=result.method,
            )
        )
    return ScalingReport(model=model, points=tuple(points))


__all__ = [
    "BIRKHOFF_METHOD",
    "EffectiveMassCurve",
    "EffectiveMassPoint",
    "GapCandidate",
    "MultistepGapResult",
    "SYMMETRIC_METHOD",
    "ScalingPoint",
    "ScalingReport",
    "TransferGapResult",
    "certified_effective_mass_curve",
    "certified_multistep_gap_refinement",
    "certified_transfer_matrix_gap",
    "heat_kernel_gap_scaling_report",
]

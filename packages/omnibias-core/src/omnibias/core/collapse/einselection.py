# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Einselection collapse: a sound coherence enclosure decides a distribution.

Fix a candidate pointer basis of dimension ``d`` and initial amplitudes
``c_i`` with ``sum_i |c_i|^2 = 1``. The exact solution of the pure-dephasing
Lindblad equation (no energy relaxation, ``T1 = inf``) in this basis is::

    rho_ij(t) = c_i * conj(c_j) * exp(-Gamma_ij * t)   (i != j)
    rho_ii(t) = |c_i|^2                                 (i == j)

with a symmetric nonnegative rate matrix ``Gamma_ij = Gamma_ji >= 0``,
``Gamma_ii = 0``. The moving parameter is ``decoherence_rate`` (``Gamma_ij
* t``); its limit is ``inf``. Given a caller-declared sensitivity ``eps >
0``, the adjudicated claim is finite and falsifiable: "no interference
measurement with resolution coarser than ``eps`` can distinguish ``rho(t)``
from the classical mixture ``{|c_i|^2}``." On a *sound* enclosure of the
coherence ``C(t) = max_{i != j} |rho_ij(t)|`` (built from
:class:`~omnibias.core.verified.complex_interval.ComplexInterval` and
:func:`~omnibias.core.verified.transcend.exp_iv`, never a float ``exp``):

* ``C.hi < eps`` -- ``PROVED``; surviving object is the **einselected
  distribution** ``{|c_i|^2}``.
* ``C.lo > eps`` -- ``DISPROVED``; coherence is provably still detectable.
* otherwise -- ``BLOCKED`` (``Inconclusive``, not falsity).

**What this does not claim.** The global state stays pure, unitary, and
entangled throughout. ``rho(t)`` is an *improper* mixture: it has the
numerical form of "the system is in state ``i`` with probability
``|c_i|^2``", but the ignorance interpretation is not licensed, because the
correlations still exist in the (untraced) environment. Einselection
explains the preferred basis and the disappearance of interference. It does
not explain, and this module does not claim to explain, why any single
outcome is realized. This is not a wave-function-collapse claim, not a
resolution of the measurement problem, and not a Born-rule derivation.

Do not conflate this with founding bias collapse (``delta -> 0``),
temperature collapse (``beta -> inf``), or Enclosure Collapse
(``width -> 0`` of a sound enclosure, yielding a point plus a proof).
The surviving object here is a ``d``-vector distribution, not a
derivative, not a 0/1 step, and not a scalar point-plus-proof.

``pointer_basis_verdict`` separately checks whether a *caller-supplied*
candidate system observable commutes with the declared interaction
Hamiltonian on a sound enclosure of the commutator; it does not search for
a pointer basis. ``propose_pointer_basis`` is a float heuristic ranking and
never gates either verdict -- exactly as rank collapse's ``sigma_min`` only
proposes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.transcend import exp_iv

EINSELECTION_SPEC = CollapseSpec(
    name="einselection",
    parameter="decoherence_rate",
    limit="inf",
    surviving_object="einselected_distribution",
    failure="residual_coherence_above_budget",
    home="omnibias.core.collapse.einselection",
    register="measure",
)


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="einselection")
    payload["einselection_collapse"] = True
    payload["wave_function_collapse_claim"] = False
    payload["measurement_problem_resolved"] = False
    payload["single_outcome_claim"] = False
    payload["born_rule_derived"] = False
    payload["continuum_parent_inferred"] = False
    payload["float_residual_is_proof"] = False
    return payload


@dataclass(frozen=True)
class DephasingModel:
    """A pure-dephasing model: amplitudes plus a symmetric decoherence-rate matrix.

    ``rates[i][j]`` is the pairwise dephasing rate ``Gamma_ij`` between
    basis states ``i`` and ``j``. The diagonal must be exactly zero (no
    self-rate) and the matrix must be symmetric and nonnegative. The
    amplitude-norm enclosure ``sum_i |c_i|^2`` must not *exclude* ``1``.
    """

    amplitudes: tuple[ComplexInterval, ...]
    rates: tuple[tuple[Interval, ...], ...]

    def __post_init__(self) -> None:
        amplitudes = tuple(ComplexInterval.from_value(c) for c in self.amplitudes)
        object.__setattr__(self, "amplitudes", amplitudes)
        rates = tuple(tuple(Interval.from_value(x) for x in row) for row in self.rates)
        object.__setattr__(self, "rates", rates)

        dim = len(amplitudes)
        if dim < 1:
            raise ValueError("amplitudes must be non-empty")
        if len(rates) != dim or any(len(row) != dim for row in rates):
            raise ValueError(f"rates must be a {dim}x{dim} matrix matching amplitudes")
        for i in range(dim):
            diagonal = rates[i][i]
            if diagonal.lo != 0.0 or diagonal.hi != 0.0:
                raise ValueError(f"rates[{i}][{i}] must be exactly zero")
            for j in range(dim):
                if rates[i][j].lo < 0.0:
                    raise ValueError(f"rates[{i}][{j}] must be nonnegative")
                other = rates[j][i]
                if rates[i][j].lo != other.lo or rates[i][j].hi != other.hi:
                    raise ValueError(
                        f"rates must be symmetric: rates[{i}][{j}] != rates[{j}][{i}]"
                    )
        total = Interval.point(0.0)
        for c in amplitudes:
            total = total + c.re * c.re + c.im * c.im
        if not total.contains(1.0):
            raise ValueError(
                f"amplitude-norm enclosure [{total.lo}, {total.hi}] excludes 1.0"
            )


def reduced_density_matrix(
    model: DephasingModel, time: IntervalLike
) -> tuple[tuple[ComplexInterval, ...], ...]:
    """Sound enclosure of ``rho(t)`` in the model's declared basis.

    Off-diagonal decay uses :func:`exp_iv`, never a float ``math.exp``.
    Diagonal populations are exact (pure dephasing never moves them).
    """

    time_iv = time if isinstance(time, Interval) else Interval.from_value(time)
    if time_iv.lo < 0.0:
        raise ValueError(f"time must be nonnegative, got lo={time_iv.lo!r}")
    dim = len(model.amplitudes)
    rows: list[tuple[ComplexInterval, ...]] = []
    for i in range(dim):
        c_i = model.amplitudes[i]
        row: list[ComplexInterval] = []
        for j in range(dim):
            if i == j:
                population = c_i.re * c_i.re + c_i.im * c_i.im
                row.append(ComplexInterval.from_parts(population, 0.0))
            else:
                c_j = model.amplitudes[j]
                decay = exp_iv(-(model.rates[i][j] * time_iv))
                row.append((c_i * c_j.conj()) * decay)
        rows.append(tuple(row))
    return tuple(rows)


def coherence_enclosure(rho: Sequence[Sequence[ComplexInterval]]) -> Interval:
    """Sound enclosure of ``max_{i != j} |rho_ij|`` (a max over intervals)."""

    magnitudes: list[Interval] = []
    dim = len(rho)
    for i in range(dim):
        for j in range(dim):
            if i != j:
                magnitudes.append(rho[i][j].modulus())
    if not magnitudes:
        return Interval.point(0.0)
    return Interval(
        max(box.lo for box in magnitudes), max(box.hi for box in magnitudes)
    )


def einselected_distribution(model: DephasingModel) -> tuple[Interval, ...]:
    """Sound enclosure of the pointer-basis populations ``{|c_i|^2}``.

    Pure dephasing never moves the diagonal, so this is time-independent.
    """

    return tuple(c.re * c.re + c.im * c.im for c in model.amplitudes)


_OutcomeStatus = Literal["collapsed", "excluded", "inconclusive"]


def _outcome(status: _OutcomeStatus, residual: Interval, eps: float) -> CollapseOutcome:
    surviving: str | None
    if status == "collapsed":
        surviving = "einselected_distribution"
        detail = (
            f"coherence enclosure [{residual.lo}, {residual.hi}] stays below "
            f"eps={eps}; einselected distribution survives"
        )
    elif status == "excluded":
        surviving = "DISPROVED"
        detail = (
            f"coherence enclosure [{residual.lo}, {residual.hi}] exceeds "
            f"eps={eps}; interference stays detectable"
        )
    else:
        surviving = None
        detail = (
            f"coherence enclosure [{residual.lo}, {residual.hi}] straddles "
            f"eps={eps}; Inconclusive, not falsity"
        )
    return CollapseOutcome(
        status=status,
        spec_name="einselection",
        surviving=surviving,
        residual=residual,
        detail=detail,
        honesty=_honesty(),
    )


def einselection_collapse(
    model: DephasingModel,
    *,
    time: IntervalLike,
    coherence_budget: float,
) -> ObligationVerdict:
    """Decide whether ``rho(t)`` is einselected at sensitivity ``coherence_budget``.

    ``coherence_budget`` (``eps``) is always an explicit caller argument; it
    is never defaulted, so a certificate can never silently hide its
    tolerance. Comparisons are strict: an enclosure that merely touches
    ``eps`` lands in ``BLOCKED``, not a decision.
    """

    if isinstance(coherence_budget, bool) or not isinstance(
        coherence_budget, int | float
    ):
        raise TypeError("coherence_budget must be a real number")
    eps = float(coherence_budget)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"coherence_budget must be a positive finite number, got {eps!r}")

    time_iv = time if isinstance(time, Interval) else Interval.from_value(time)
    if time_iv.lo < 0.0:
        raise ValueError(f"time must be nonnegative, got lo={time_iv.lo!r}")

    rho = reduced_density_matrix(model, time_iv)
    coherence = coherence_enclosure(rho)

    if coherence.hi < eps:
        outcome = _outcome("collapsed", coherence, eps)
        return ObligationVerdict(
            status="PROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    if coherence.lo > eps:
        outcome = _outcome("excluded", coherence, eps)
        return ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    outcome = _outcome("inconclusive", coherence, eps)
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=True,
        evaluated=1,
        complete=False,
        detail=outcome.detail,
    )


def _square_matrix(
    matrix: Sequence[Sequence[ComplexLike]], *, name: str
) -> tuple[tuple[ComplexInterval, ...], ...]:
    rows = [tuple(ComplexInterval.from_value(entry) for entry in row) for row in matrix]
    dim = len(rows)
    if dim < 1 or any(len(row) != dim for row in rows):
        raise ValueError(f"{name} must be a non-empty square matrix")
    return tuple(rows)


def _matmul(
    a: tuple[tuple[ComplexInterval, ...], ...],
    b: tuple[tuple[ComplexInterval, ...], ...],
) -> tuple[tuple[ComplexInterval, ...], ...]:
    dim = len(a)
    rows: list[tuple[ComplexInterval, ...]] = []
    for i in range(dim):
        row: list[ComplexInterval] = []
        for j in range(dim):
            acc = ComplexInterval.zero()
            for k in range(dim):
                acc = acc + a[i][k] * b[k][j]
            row.append(acc)
        rows.append(tuple(row))
    return tuple(rows)


def commutator_enclosure(
    a: Sequence[Sequence[ComplexLike]],
    h: Sequence[Sequence[ComplexLike]],
) -> tuple[tuple[ComplexInterval, ...], ...]:
    """Sound enclosure of the commutator ``[A, H] = A H - H A``."""

    a_boxed = _square_matrix(a, name="a")
    h_boxed = _square_matrix(h, name="h")
    if len(a_boxed) != len(h_boxed):
        raise ValueError("a and h must share the same dimension")
    ah = _matmul(a_boxed, h_boxed)
    ha = _matmul(h_boxed, a_boxed)
    dim = len(a_boxed)
    return tuple(
        tuple(ah[i][j] - ha[i][j] for j in range(dim)) for i in range(dim)
    )


def _is_point(entry: ComplexInterval) -> bool:
    return entry.re.lo == entry.re.hi and entry.im.lo == entry.im.hi


def _all_points(matrix: Sequence[Sequence[ComplexInterval]]) -> bool:
    return all(_is_point(entry) for row in matrix for entry in row)


_ExactComplex = tuple[Fraction, Fraction]


def _exact_matmul(
    a: Sequence[Sequence[_ExactComplex]],
    b: Sequence[Sequence[_ExactComplex]],
    dim: int,
) -> list[list[_ExactComplex]]:
    out: list[list[_ExactComplex]] = []
    for i in range(dim):
        row: list[_ExactComplex] = []
        for j in range(dim):
            re_acc = Fraction(0)
            im_acc = Fraction(0)
            for k in range(dim):
                a_re, a_im = a[i][k]
                b_re, b_im = b[k][j]
                re_acc += a_re * b_re - a_im * b_im
                im_acc += a_re * b_im + a_im * b_re
            row.append((re_acc, im_acc))
        out.append(row)
    return out


def _exact_pointer_basis_verdict(
    a_boxed: tuple[tuple[ComplexInterval, ...], ...],
    h_boxed: tuple[tuple[ComplexInterval, ...], ...],
    dim: int,
) -> ObligationVerdict:
    """Exact ``Q`` commutator of two point matrices; never a fat zero.

    Interval arithmetic chains widen by outward rounding at every step, so
    even an exactly-commuting pair of point matrices would never reach a
    literal ``{0}`` through :func:`commutator_enclosure` alone (the same
    reason :func:`~omnibias.core.collapse.identity.evaluate_difference`
    special-cases the zero polynomial). This mirrors that fix: convert each
    exact double to its exact :class:`~fractions.Fraction` and multiply in
    ``Q``, where a true zero commutator is bit-exact ``0``, not a fat zero.
    """

    a_exact = [
        [(Fraction(entry.re.lo), Fraction(entry.im.lo)) for entry in row]
        for row in a_boxed
    ]
    h_exact = [
        [(Fraction(entry.re.lo), Fraction(entry.im.lo)) for entry in row]
        for row in h_boxed
    ]
    ah = _exact_matmul(a_exact, h_exact, dim)
    ha = _exact_matmul(h_exact, a_exact, dim)
    for i in range(dim):
        for j in range(dim):
            re_diff = ah[i][j][0] - ha[i][j][0]
            im_diff = ah[i][j][1] - ha[i][j][1]
            if re_diff != 0 or im_diff != 0:
                outcome = CollapseOutcome(
                    status="excluded",
                    spec_name="einselection",
                    surviving="DISPROVED",
                    residual=None,
                    detail=(
                        f"exact commutator entry [{i}][{j}] = "
                        f"({re_diff}, {im_diff}) is nonzero over Q"
                    ),
                    honesty=_honesty(),
                )
                return ObligationVerdict(
                    status="DISPROVED",
                    outcome=outcome,
                    existential=True,
                    evaluated=dim * dim,
                    complete=True,
                    detail=outcome.detail,
                )
    outcome = CollapseOutcome(
        status="collapsed",
        spec_name="einselection",
        surviving="commuting_pointer_basis",
        residual=None,
        detail="exact commutator is the zero matrix over Q; candidate commutes",
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="PROVED",
        outcome=outcome,
        existential=True,
        evaluated=dim * dim,
        complete=True,
        detail=outcome.detail,
    )


def pointer_basis_verdict(
    a: Sequence[Sequence[ComplexLike]],
    h: Sequence[Sequence[ComplexLike]],
) -> ObligationVerdict:
    """Decide whether candidate observable ``a`` commutes with ``h``.

    If every entry of ``a`` and ``h`` is an exact point, the commutator is
    decided by exact ``Q`` arithmetic (never a fat zero): the zero matrix
    -- ``PROVED``; any nonzero entry -- ``DISPROVED``. Otherwise this
    checks only for a sound ``DISPROVED`` (some commutator entry's
    rectangle provably excludes the origin); genuinely interval-valued
    inputs can never be forged into a ``PROVED`` and land in ``BLOCKED``
    when no entry is excluded. This checks one caller-supplied candidate;
    it does not search for a pointer basis.
    """

    a_boxed = _square_matrix(a, name="a")
    h_boxed = _square_matrix(h, name="h")
    if len(a_boxed) != len(h_boxed):
        raise ValueError("a and h must share the same dimension")
    dim = len(a_boxed)

    if _all_points(a_boxed) and _all_points(h_boxed):
        return _exact_pointer_basis_verdict(a_boxed, h_boxed, dim)

    ah = _matmul(a_boxed, h_boxed)
    ha = _matmul(h_boxed, a_boxed)
    for i in range(dim):
        for j in range(dim):
            diff = ah[i][j] - ha[i][j]
            if not diff.re.contains_zero() or not diff.im.contains_zero():
                outcome = CollapseOutcome(
                    status="excluded",
                    spec_name="einselection",
                    surviving="DISPROVED",
                    residual=diff.modulus(),
                    detail=f"commutator entry [{i}][{j}] provably excludes 0",
                    honesty=_honesty(),
                )
                return ObligationVerdict(
                    status="DISPROVED",
                    outcome=outcome,
                    existential=True,
                    evaluated=dim * dim,
                    complete=True,
                    detail=outcome.detail,
                )
    outcome = CollapseOutcome(
        status="inconclusive",
        spec_name="einselection",
        surviving=None,
        residual=None,
        detail=(
            "interval-valued inputs cannot certify exact commutativity in "
            "this framework; Inconclusive, not falsity"
        ),
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=True,
        evaluated=dim * dim,
        complete=False,
        detail=outcome.detail,
    )


def propose_pointer_basis(model: DephasingModel) -> tuple[int, ...]:
    """Float predictability-sieve heuristic; a proposer, never an accept.

    Ranks basis indices by descending initial population midpoint. This
    ranking never gates :func:`einselection_collapse` or
    :func:`pointer_basis_verdict` -- exactly as rank collapse's float
    ``sigma_min`` only proposes a candidate kernel vector.
    """

    populations = einselected_distribution(model)
    return tuple(sorted(range(len(populations)), key=lambda i: -populations[i].mid))


def _reseed() -> None:
    try:
        get_collapse("einselection")
    except KeyError:
        register_collapse(EINSELECTION_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "DephasingModel",
    "EINSELECTION_SPEC",
    "coherence_enclosure",
    "commutator_enclosure",
    "einselected_distribution",
    "einselection_collapse",
    "pointer_basis_verdict",
    "propose_pointer_basis",
    "reduced_density_matrix",
]

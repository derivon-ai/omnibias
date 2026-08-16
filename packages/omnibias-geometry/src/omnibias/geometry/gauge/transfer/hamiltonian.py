# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Two-plaquette SU(2) Kogut–Susskind Hamiltonian and a certified gap.

Two spatial plaquettes sharing an edge.  After Gauss-law reduction the
physical basis is the recoupling triples ``|j1, j2, js⟩`` obeying the
triangle inequalities.  The spectrum is **not** known in closed form, so
a holonomy trial space is not secretly the eigenbasis.

Locked operators (this finite matrix, not 4-D Yang-Mills)
---------------------------------------------------------
Spins are stored as ``two_j = 2j`` (non-negative integers).  A triple is
legal when ``|j1-j2| ≤ js ≤ j1+j2``, every spin is at most ``j_max``, and
``j1+j2+js`` is an integer (equivalently ``two_j`` sum even).

**Electric.**  The two-plaquette graph has seven geometric edges: three
unique edges on each plaquette and the shared edge.  In the physics
normalization ``E² |j⟩ = j(j+1) |j⟩``,

    H_E = (g² / 2) [ 3 j1(j1+1) + 3 j2(j2+1) + js(js+1) ].

``j(j+1) = two_j (two_j + 2) / 4`` is an exact ``Fraction``.

**Magnetic.**  ``H_B = -(2/g²) (χ(U_p1) + χ(U_p2))``.  The SU(2)
character identity ``χ_{1/2} χ_j = χ_{j+1/2} + χ_{j-1/2}`` changes a
plaquette flux by ``±1/2``.  Changing only that flux would flip the
``two_j``-sum parity and leave the Gauss-law subspace; the shared edge
must recouple with it.  The locked selection rule is therefore

    M1 : (j1, js) → (j1 ± 1/2, js ± 1/2)   (independent signs)
    M2 : (j2, js) → (j2 ± 1/2, js ± 1/2)

Default ``magnetic="sixj"`` inserts the locked recoupling
``phase × √[(2j+1)…] × 6j`` from :mod:`.sixj`.  ``magnetic="character"``
keeps the older amplitude-1 operator so the two can be compared.

The certified gap is ``λ1 - λ0`` of this finite matrix (not
``-ln(λ1/λ0)``).  Continuum existence and a uniform-in-spacing gap stay
external.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Literal

import numpy as np
from omnibias.core.verified.eig import symmetric_eigenvalue_residual_enclosure
from omnibias.core.verified.eig_operator import certified_spectral_gap, ritz_upper_bound
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import inf_norm_matrix, to_interval_matrix
from omnibias.geometry.gauge.transfer.gap import GapCandidate
from omnibias.geometry.gauge.transfer.matrices import Scalar, decode_scalar, encode_scalar
from omnibias.geometry.gauge.transfer.sixj import magnetic_sixj_amplitude
from omnibias.geometry.gauge.transfer.trial import (
    GRAM_COND_THRESHOLD,
    TrialSpace,
    gram_condition_number,
)

RESIDUAL_EIGENBASIS_METHOD = "residual_complete_eigenbasis"
RESIDUAL_STANDARD_METHOD = "residual_standard_basis"
RESIDUAL_HOLONOMY_METHOD = "residual_holonomy_trial"
LEHMANN_STANDARD_METHOD = "lehmann_standard_basis"
LEHMANN_HOLONOMY_METHOD = "lehmann_maehly_holonomy_trial"

#: Locked small coupling ``g²`` at which the finite Hamiltonian is certified.
COUPLING_LOCK = Fraction(1, 2)

Magnetic = Literal["sixj", "character"]

#: Edge census of the 3-square chain (10 geometric edges).
#: P1 unique 3, P2 unique 2, P3 unique 3, shared s12, shared s23.
THREE_PLAQUETTE_ELECTRIC = (3, 2, 3, 1, 1)


@dataclass(frozen=True)
class GaugeHamiltonian:
    """A finite, Gauss-law-reduced lattice Hamiltonian with enclosed entries."""

    model: str
    coupling: Scalar
    j_max: int
    basis: tuple[tuple[int, ...], ...]
    entries: tuple[tuple[Interval, ...], ...]
    parameters: dict[str, object]
    mode_labels: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def matrix(self) -> tuple[tuple[Interval, ...], ...]:
        return self.entries


@dataclass(frozen=True)
class HamiltonianGapResult:
    """Certified lower bound on ``λ1 - λ0`` of a fixed Hamiltonian."""

    model: str
    dimension: int
    method: str
    spectral_gap_lower: float
    subdominant_ratio_upper: float
    lambda0_upper: float
    lambda1_lower: float
    coupling: float
    j_max: int
    candidates: tuple[GapCandidate, ...] = field(default_factory=tuple)
    trial_gram_condition: float | None = None
    trial_flagged: bool = False
    trial_remainder_width: float = 0.0

    @property
    def certified(self) -> bool:
        return self.spectral_gap_lower > 0.0


def _casimir_jj(two_j: int) -> Fraction:
    """``j(j+1)`` for ``j = two_j / 2``, exact."""
    return Fraction(two_j * (two_j + 2), 4)


def legal_triple(two_j1: int, two_j2: int, two_js: int, *, two_j_max: int) -> bool:
    """Triangle inequalities, range, and integer-sum (even ``two_j`` total)."""
    if min(two_j1, two_j2, two_js) < 0:
        return False
    if max(two_j1, two_j2, two_js) > two_j_max:
        return False
    if (two_j1 + two_j2 + two_js) % 2 != 0:
        return False
    return abs(two_j1 - two_j2) <= two_js <= two_j1 + two_j2


def physical_basis(j_max: int) -> tuple[tuple[int, int, int], ...]:
    """All legal ``(2j1, 2j2, 2js)`` with ``j ≤ j_max``."""
    if j_max < 1:
        raise ValueError(f"j_max must be >= 1, got {j_max}")
    two_j_max = 2 * int(j_max)
    states = [
        (t1, t2, ts)
        for t1, t2, ts in product(range(two_j_max + 1), repeat=3)
        if legal_triple(t1, t2, ts, two_j_max=two_j_max)
    ]
    return tuple(states)


def _magnetic_kind(magnetic: str) -> Magnetic:
    if magnetic not in ("sixj", "character"):
        raise ValueError(f"magnetic must be 'sixj' or 'character', got {magnetic!r}")
    return magnetic  # type: ignore[return-value]


def _directed_magnetic(
    two_j_a: int,
    two_j_s: int,
    two_j_spectator: int,
    two_j_a_prime: int,
    two_j_s_prime: int,
    *,
    magnetic: Magnetic,
) -> Interval:
    if magnetic == "character":
        return Interval.point(1.0)
    return magnetic_sixj_amplitude(
        two_j_a, two_j_s, two_j_spectator, two_j_a_prime, two_j_s_prime
    )


def _symmetrize(raw: list[list[Interval]]) -> tuple[tuple[Interval, ...], ...]:
    n = len(raw)
    half = Interval.from_value(Fraction(1, 2))
    return tuple(
        tuple((raw[i][j] + raw[j][i]) * half for j in range(n)) for i in range(n)
    )


def su2_two_plaquette_hamiltonian(
    coupling: Scalar,
    *,
    j_max: int = 1,
    magnetic: Magnetic = "sixj",
) -> GaugeHamiltonian:
    """The two-plaquette SU(2) Kogut–Susskind Hamiltonian at one ``g²``.

    ``coupling`` is ``g²``.  CI smoke uses ``j_max=1`` (tens of states);
    ``j_max=2`` is the ``--full`` truncation.  This is one finite matrix,
    not 4-D Yang-Mills.
    """
    kind = _magnetic_kind(magnetic)
    if isinstance(coupling, bool) or not isinstance(coupling, int | float | Fraction):
        raise ValueError(f"coupling must be a positive scalar, got {coupling!r}")
    if float(coupling) <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")
    basis = physical_basis(j_max)
    index = {state: i for i, state in enumerate(basis)}
    two_j_max = 2 * int(j_max)
    electric_pre = Interval.from_value(coupling) * Interval.from_value(Fraction(1, 2))
    magnetic_pre = Interval.from_value(-2) / Interval.from_value(coupling)
    n = len(basis)
    raw = [[Interval.point(0.0) for _ in range(n)] for _ in range(n)]
    for state in basis:
        t1, t2, ts = state
        i = index[state]
        casimir = (
            3 * _casimir_jj(t1) + 3 * _casimir_jj(t2) + _casimir_jj(ts)
        )
        raw[i][i] = raw[i][i] + electric_pre * Interval.from_value(casimir)
        for eps, delta in product((-1, 1), repeat=2):
            t1p, tsp = t1 + eps, ts + delta
            if legal_triple(t1p, t2, tsp, two_j_max=two_j_max):
                j = index[(t1p, t2, tsp)]
                amp = _directed_magnetic(t1, ts, t2, t1p, tsp, magnetic=kind)
                raw[i][j] = raw[i][j] + magnetic_pre * amp
            t2p, tsp = t2 + eps, ts + delta
            if legal_triple(t1, t2p, tsp, two_j_max=two_j_max):
                j = index[(t1, t2p, tsp)]
                amp = _directed_magnetic(t2, ts, t1, t2p, tsp, magnetic=kind)
                raw[i][j] = raw[i][j] + magnetic_pre * amp
    entries = _symmetrize(raw)
    labels = tuple(f"|{t1}/2, {t2}/2, {ts}/2>" for t1, t2, ts in basis)
    return GaugeHamiltonian(
        model="su2_two_plaquette",
        coupling=coupling,
        j_max=int(j_max),
        basis=basis,
        entries=entries,
        parameters={
            "builder": "su2_two_plaquette_hamiltonian",
            "coupling": encode_scalar(coupling),
            "j_max": int(j_max),
            "n_plaquettes": 2,
            "magnetic": kind,
        },
        mode_labels=labels,
    )


def legal_chain(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_js12: int,
    two_js23: int,
    *,
    two_j_max: int,
) -> bool:
    """Two triangle inequalities of the 3-plaquette chain."""
    return legal_triple(two_j1, two_j2, two_js12, two_j_max=two_j_max) and legal_triple(
        two_j2, two_j3, two_js23, two_j_max=two_j_max
    )


def three_plaquette_basis(j_max: int) -> tuple[tuple[int, int, int, int, int], ...]:
    """All legal ``(2j1, 2j2, 2j3, 2js12, 2js23)`` with ``j ≤ j_max``."""
    if j_max < 1:
        raise ValueError(f"j_max must be >= 1, got {j_max}")
    two_j_max = 2 * int(j_max)
    states = [
        (t1, t2, t3, s12, s23)
        for t1, t2, t3, s12, s23 in product(range(two_j_max + 1), repeat=5)
        if legal_chain(t1, t2, t3, s12, s23, two_j_max=two_j_max)
    ]
    return tuple(states)


def su2_three_plaquette_hamiltonian(
    coupling: Scalar,
    *,
    j_max: int = 1,
    magnetic: Magnetic = "sixj",
) -> GaugeHamiltonian:
    """The three-plaquette SU(2) Kogut–Susskind Hamiltonian at one ``g²``.

    Basis ``|j1, j2, j3, js12, js23⟩`` with triangles ``(j1, j2, js12)``
    and ``(j2, j3, js23)``.  Electric weights are the locked 3-square-chain
    census :data:`THREE_PLAQUETTE_ELECTRIC`.  CI uses ``j_max=1``;
    ``j_max=2`` is the ``--full`` truncation only.  One finite matrix,
    not 4-D Yang-Mills.
    """
    kind = _magnetic_kind(magnetic)
    if isinstance(coupling, bool) or not isinstance(coupling, int | float | Fraction):
        raise ValueError(f"coupling must be a positive scalar, got {coupling!r}")
    if float(coupling) <= 0.0:
        raise ValueError(f"coupling must be > 0, got {coupling!r}")
    basis = three_plaquette_basis(j_max)
    index = {state: i for i, state in enumerate(basis)}
    two_j_max = 2 * int(j_max)
    electric_pre = Interval.from_value(coupling) * Interval.from_value(Fraction(1, 2))
    magnetic_pre = Interval.from_value(-2) / Interval.from_value(coupling)
    n = len(basis)
    raw = [[Interval.point(0.0) for _ in range(n)] for _ in range(n)]
    w1, w2, w3, ws12, ws23 = THREE_PLAQUETTE_ELECTRIC
    for state in basis:
        t1, t2, t3, s12, s23 = state
        i = index[state]
        casimir = (
            w1 * _casimir_jj(t1)
            + w2 * _casimir_jj(t2)
            + w3 * _casimir_jj(t3)
            + ws12 * _casimir_jj(s12)
            + ws23 * _casimir_jj(s23)
        )
        raw[i][i] = raw[i][i] + electric_pre * Interval.from_value(casimir)
        for eps, delta in product((-1, 1), repeat=2):
            t1p, s12p = t1 + eps, s12 + delta
            if legal_chain(t1p, t2, t3, s12p, s23, two_j_max=two_j_max):
                j = index[(t1p, t2, t3, s12p, s23)]
                amp = _directed_magnetic(t1, s12, t2, t1p, s12p, magnetic=kind)
                raw[i][j] = raw[i][j] + magnetic_pre * amp
            t3p, s23p = t3 + eps, s23 + delta
            if legal_chain(t1, t2, t3p, s12, s23p, two_j_max=two_j_max):
                j = index[(t1, t2, t3p, s12, s23p)]
                amp = _directed_magnetic(t3, s23, t2, t3p, s23p, magnetic=kind)
                raw[i][j] = raw[i][j] + magnetic_pre * amp
        for eps, d12, d23 in product((-1, 1), repeat=3):
            t2p, s12p, s23p = t2 + eps, s12 + d12, s23 + d23
            if legal_chain(t1, t2p, t3, s12p, s23p, two_j_max=two_j_max):
                j = index[(t1, t2p, t3, s12p, s23p)]
                left = _directed_magnetic(t2, s12, t1, t2p, s12p, magnetic=kind)
                right = _directed_magnetic(t2, s23, t3, t2p, s23p, magnetic=kind)
                raw[i][j] = raw[i][j] + magnetic_pre * left * right
    entries = _symmetrize(raw)
    labels = tuple(
        f"|{t1}/2, {t2}/2, {t3}/2, {s12}/2, {s23}/2>" for t1, t2, t3, s12, s23 in basis
    )
    return GaugeHamiltonian(
        model="su2_three_plaquette",
        coupling=coupling,
        j_max=int(j_max),
        basis=basis,
        entries=entries,
        parameters={
            "builder": "su2_three_plaquette_hamiltonian",
            "coupling": encode_scalar(coupling),
            "j_max": int(j_max),
            "n_plaquettes": 3,
            "magnetic": kind,
        },
        mode_labels=labels,
    )


def rebuild_hamiltonian(parameters: Mapping[str, object]) -> GaugeHamiltonian:
    """Replay helper: rebuild from ``(coupling, j_max, n_plaquettes, magnetic)``."""
    spec = dict(parameters)
    raw = spec.get("coupling")
    if raw is None:
        raise ValueError("parameters must carry a 'coupling'")
    coupling: Scalar = decode_scalar(str(raw)) if isinstance(raw, str) else raw  # type: ignore[assignment]
    j_max = spec.get("j_max", 1)
    if not isinstance(j_max, int) or isinstance(j_max, bool) or j_max < 1:
        raise ValueError(f"j_max must be an integer >= 1, got {j_max!r}")
    magnetic = spec.get("magnetic", "sixj")
    if magnetic not in ("sixj", "character"):
        raise ValueError(f"magnetic must be 'sixj' or 'character', got {magnetic!r}")
    n_plaquettes = spec.get("n_plaquettes", 2)
    if n_plaquettes == 3:
        return su2_three_plaquette_hamiltonian(
            coupling, j_max=j_max, magnetic=magnetic
        )
    if n_plaquettes != 2:
        raise ValueError(f"n_plaquettes must be 2 or 3, got {n_plaquettes!r}")
    return su2_two_plaquette_hamiltonian(coupling, j_max=j_max, magnetic=magnetic)


def _orthonormalize(
    vectors: Sequence[Sequence[float]], *, tol: float = 1e-12
) -> tuple[tuple[float, ...], ...]:
    """Modified Gram-Schmidt; drops numerically dependent columns."""
    basis: list[list[float]] = []
    for raw in vectors:
        vec = [float(x) for x in raw]
        for prev in basis:
            dot = math.fsum(a * b for a, b in zip(vec, prev, strict=True))
            vec = [a - dot * b for a, b in zip(vec, prev, strict=True)]
        norm = math.sqrt(math.fsum(a * a for a in vec))
        if norm > tol:
            basis.append([a / norm for a in vec])
    return tuple(tuple(vec) for vec in basis)


def standard_basis_trial_space(hamiltonian: GaugeHamiltonian, *, dim: int | None = None) -> TrialSpace:
    """The first ``dim`` computational-basis vectors (generic, no holonomy)."""
    n = hamiltonian.dimension
    count = n if dim is None else min(int(dim), n)
    vectors = _orthonormalize(
        tuple(tuple(1.0 if i == k else 0.0 for i in range(n)) for k in range(count))
    )
    cond = gram_condition_number(vectors)
    return TrialSpace(
        vectors=vectors,
        gram_condition=cond,
        flagged=cond > GRAM_COND_THRESHOLD or not math.isfinite(cond),
        remainder_width=0.0,
        basis="standard",
    )


def plaquette_holonomy_trial_space(
    hamiltonian: GaugeHamiltonian,
    *,
    dim: int | None = None,
) -> TrialSpace:
    """Plaquette-character trials on the ``|j1, j2, js⟩`` labels.

    Vectors are projectors onto a fixed plaquette (or shared-edge) spin,
    plus the Gauss-law vacuum.  They are not the eigenbasis: the magnetic
    term mixes different ``j``.
    """
    basis = hamiltonian.basis
    n = len(basis)
    n_slots = len(basis[0]) if basis else 0
    family: list[tuple[float, ...]] = []
    vacuum_state = tuple(0 for _ in range(n_slots))
    vacuum = tuple(1.0 if state == vacuum_state else 0.0 for state in basis)
    if any(vacuum):
        family.append(vacuum)
    two_j_max = 2 * hamiltonian.j_max
    for slot in range(n_slots):
        for two_j in range(two_j_max + 1):
            vec = tuple(1.0 if state[slot] == two_j else 0.0 for state in basis)
            if any(vec) and vec not in family:
                family.append(vec)
    count = n if dim is None else min(int(dim), n)
    vectors = _orthonormalize(tuple(family[: max(count, len(family))]))[:count]
    if len(vectors) < 1:
        raise ValueError("holonomy trial space is empty")
    cond = gram_condition_number(vectors)
    return TrialSpace(
        vectors=vectors,
        gram_condition=cond,
        flagged=cond > GRAM_COND_THRESHOLD or not math.isfinite(cond),
        remainder_width=0.0,
        basis="plaquette_character",
    )


def _shifted_ratio(lambda0_upper: float, lambda1_lower: float, shift: float) -> float:
    """``(λ0_up + s) / (λ1_lo + s)`` after a positive shift of the spectrum."""
    if shift <= 0.0:
        raise ValueError("shift must be positive")
    if lambda1_lower <= lambda0_upper:
        return 1.0
    return (lambda0_upper + shift) / (lambda1_lower + shift)


def _residual_gap_from_vectors(
    matrix: Sequence[Sequence[Interval]],
    vectors: Sequence[Sequence[float]],
    *,
    method: str,
    require_complete: bool,
) -> GapCandidate:
    """Residual enclosures of the supplied vectors; gap from the two lowest.

    When ``require_complete`` is true the vector count must equal the
    matrix dimension and the two lowest enclosures must sit strictly
    below the rest, so they can be labelled ``(λ0, λ1)``.  Higher
    degeneracies may overlap.  A short trial list only yields a
    variational pair and is tagged as such.
    """
    if len(vectors) < 2:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail="need at least two vectors",
        )
    try:
        enclosures = [
            symmetric_eigenvalue_residual_enclosure(matrix, list(vec)) for vec in vectors
        ]
    except (ValueError, ZeroDivisionError) as exc:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail=f"not applicable: {exc}",
        )
    ordered = sorted(enclosures, key=lambda iv: 0.5 * (iv.lo + iv.hi))
    if require_complete:
        if len(vectors) != len(matrix):
            return GapCandidate(
                method=method,
                subdominant_ratio_upper=1.0,
                spectral_gap_lower=0.0,
                detail="complete eigenbasis required",
            )
        if ordered[0].hi >= min(item.lo for item in ordered[1:]):
            return GapCandidate(
                method=method,
                subdominant_ratio_upper=1.0,
                spectral_gap_lower=0.0,
                detail="residual enclosures overlap; cannot label λ0",
            )
        if len(ordered) > 2 and ordered[1].hi >= min(item.lo for item in ordered[2:]):
            return GapCandidate(
                method=method,
                subdominant_ratio_upper=1.0,
                spectral_gap_lower=0.0,
                detail="residual enclosures overlap; cannot label λ1",
            )
    else:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail="incomplete residual cover does not label λ0, λ1",
        )
    gap = float(ordered[1].lo - ordered[0].hi)
    if gap <= 0.0:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail="non-positive residual gap",
        )
    shift = inf_norm_matrix(to_interval_matrix(matrix)) + 1.0
    ratio = _shifted_ratio(ordered[0].hi, ordered[1].lo, shift)
    return GapCandidate(
        method=method,
        subdominant_ratio_upper=float(ratio),
        spectral_gap_lower=gap,
        partners_deflated=max(0, len(vectors) - 1),
        detail=f"λ0_hi={ordered[0].hi:.6g} λ1_lo={ordered[1].lo:.6g}",
    )


def _midpoint_eigenvectors(hamiltonian: GaugeHamiltonian) -> list[list[float]]:
    mid = np.array(
        [[0.5 * (entry.lo + entry.hi) for entry in row] for row in hamiltonian.entries],
        dtype=np.float64,
    )
    _values, vectors = np.linalg.eigh(0.5 * (mid + mid.T))
    return [list(map(float, vectors[:, k])) for k in range(vectors.shape[1])]


def _goerisch_grams(
    operator: Sequence[Sequence[Interval]], vectors: Sequence[Sequence[float]]
) -> tuple[list[list[Interval]], list[list[Interval]], list[list[Interval]]]:
    applied = [
        [
            sum(
                (operator[i][j] * Interval.point(vec[j]) for j in range(len(vec))),
                Interval.point(0.0),
            )
            for i in range(len(operator))
        ]
        for vec in vectors
    ]
    size = len(vectors)
    a0 = [[Interval.point(0.0) for _ in range(size)] for _ in range(size)]
    a1 = [[Interval.point(0.0) for _ in range(size)] for _ in range(size)]
    a2 = [[Interval.point(0.0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        left = [Interval.point(x) for x in vectors[i]]
        for j in range(size):
            right = [Interval.point(x) for x in vectors[j]]
            a0[i][j] = sum((left[k] * right[k] for k in range(len(left))), Interval.point(0.0))
            a1[i][j] = sum(
                (applied[i][k] * right[k] for k in range(len(right))), Interval.point(0.0)
            )
            a2[i][j] = sum(
                (applied[i][k] * applied[j][k] for k in range(len(applied[j]))),
                Interval.point(0.0),
            )
    return a0, a1, a2


def _lehmann_gap(
    matrix: Sequence[Sequence[Interval]],
    trial: TrialSpace,
    *,
    method: str,
    rho: float | None = None,
    lambda1_upper: float | None = None,
) -> GapCandidate:
    if trial.flagged:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail=f"trial Gram flagged (cond={trial.gram_condition:.3g})",
        )
    if len(trial.vectors) < 2:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail="need at least two trial vectors",
        )
    try:
        vecs = list(trial.vectors[: min(6, len(trial.vectors))])
        a0, a1, a2 = _goerisch_grams(matrix, vecs)
        lam1_up = (
            float(lambda1_upper)
            if lambda1_upper is not None
            else ritz_upper_bound(matrix, vecs[0]).hi
        )
        if rho is None:
            first = symmetric_eigenvalue_residual_enclosure(matrix, vecs[0])
            second = symmetric_eigenvalue_residual_enclosure(matrix, vecs[1])
            if first.hi >= second.lo:
                raise ValueError("trial residual enclosures overlap; cannot place rho")
            rho = 0.5 * (first.hi + second.lo)
            if rho < lam1_up:
                rho = lam1_up + 0.25 * max(second.lo - lam1_up, 0.0)
        cert = certified_spectral_gap(a0, a1, a2, rho, lam1_up)
        if not cert.certified:
            raise ValueError("Lehmann gap not certified")
        shift = inf_norm_matrix(to_interval_matrix(matrix)) + 1.0
        ratio = _shifted_ratio(cert.lambda1_upper, cert.lambda2_lower, shift)
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=float(ratio),
            spectral_gap_lower=float(cert.gap_lower),
            partners_deflated=max(0, len(vecs) - 1),
            detail=f"gram_cond={trial.gram_condition:.3g} rho={rho:.6g}",
        )
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return GapCandidate(
            method=method,
            subdominant_ratio_upper=1.0,
            spectral_gap_lower=0.0,
            detail=f"not applicable: {exc}",
        )


def certified_hamiltonian_gap(
    hamiltonian: GaugeHamiltonian,
    *,
    trial: TrialSpace | None = None,
) -> HamiltonianGapResult:
    """Certify a lower bound on ``λ1 - λ0`` of one fixed Hamiltonian.

    Always runs a complete residual cover from the midpoint eigensolver
    (sound once the ``n`` enclosures are pairwise disjoint) and a generic
    standard-basis residual.  Optional ``trial=`` adds holonomy residual
    and Lehmann–Maehly candidates.  Existing transfer callers are unchanged.
    """
    matrix = hamiltonian.matrix()
    eigen_vectors = _midpoint_eigenvectors(hamiltonian)
    eigen_candidate = _residual_gap_from_vectors(
        matrix,
        eigen_vectors,
        method=RESIDUAL_EIGENBASIS_METHOD,
        require_complete=True,
    )
    candidates: list[GapCandidate] = [eigen_candidate]
    rho: float | None = None
    lambda1_upper: float | None = None
    if eigen_candidate.spectral_gap_lower > 0.0:
        try:
            covers = [
                symmetric_eigenvalue_residual_enclosure(matrix, vec)
                for vec in eigen_vectors
            ]
            ordered = sorted(covers, key=lambda iv: 0.5 * (iv.lo + iv.hi))
            lambda1_upper = float(ordered[0].hi)
            # n_below=2 wants λ2 <= ρ <= λ3.  A later degeneracy (three-plaquette
            # λ3=λ4) must not block that slot.
            if len(ordered) >= 3 and ordered[1].hi < ordered[2].lo:
                rho = float(ordered[1].hi)
            elif len(ordered) >= 4 and ordered[2].hi < ordered[3].lo:
                rho = float(ordered[2].hi)
        except (ValueError, ZeroDivisionError):
            rho = None
            lambda1_upper = None
    standard = standard_basis_trial_space(hamiltonian, dim=2)
    candidates.append(
        _lehmann_gap(
            matrix,
            standard,
            method=LEHMANN_STANDARD_METHOD,
            rho=rho,
            lambda1_upper=lambda1_upper,
        )
    )
    candidates.append(
        _residual_gap_from_vectors(
            matrix,
            list(standard.vectors),
            method=RESIDUAL_STANDARD_METHOD,
            require_complete=False,
        )
    )
    gram: float | None = None
    flagged = False
    remainder = 0.0
    if trial is not None:
        gram = float(trial.gram_condition)
        flagged = bool(trial.flagged)
        remainder = float(trial.remainder_width)
        candidates.append(
            _residual_gap_from_vectors(
                matrix,
                list(trial.vectors),
                method=RESIDUAL_HOLONOMY_METHOD,
                require_complete=False,
            )
        )
        candidates.append(
            _lehmann_gap(
                matrix,
                trial,
                method=LEHMANN_HOLONOMY_METHOD,
                rho=rho,
                lambda1_upper=lambda1_upper,
            )
        )
    winner = max(candidates, key=lambda item: item.spectral_gap_lower)
    if winner.spectral_gap_lower > 0.0:
        shift = inf_norm_matrix(to_interval_matrix(matrix)) + 1.0
        # Recover λ0_hi, λ1_lo from the winning gap and a Ritz upper on the
        # lowest residual vector we have (midpoint ground state).
        ground = _midpoint_eigenvectors(hamiltonian)[0]
        lam0_up = symmetric_eigenvalue_residual_enclosure(matrix, ground).hi
        lam1_lo = lam0_up + winner.spectral_gap_lower
        ratio = _shifted_ratio(lam0_up, lam1_lo, shift)
    else:
        lam0_up = 0.0
        lam1_lo = 0.0
        ratio = 1.0
    return HamiltonianGapResult(
        model=hamiltonian.model,
        dimension=hamiltonian.dimension,
        method=winner.method,
        spectral_gap_lower=float(winner.spectral_gap_lower),
        subdominant_ratio_upper=float(ratio if winner.spectral_gap_lower > 0.0 else 1.0),
        lambda0_upper=float(lam0_up),
        lambda1_lower=float(lam1_lo),
        coupling=float(hamiltonian.coupling),
        j_max=int(hamiltonian.j_max),
        candidates=tuple(candidates),
        trial_gram_condition=gram,
        trial_flagged=flagged,
        trial_remainder_width=remainder,
    )


def candidate_gap(result: HamiltonianGapResult, method: str) -> float:
    """The certified lower bound recorded for one named candidate (0 if absent)."""
    for item in result.candidates:
        if item.method == method:
            return float(item.spectral_gap_lower)
    return 0.0


__all__ = [
    "COUPLING_LOCK",
    "LEHMANN_HOLONOMY_METHOD",
    "LEHMANN_STANDARD_METHOD",
    "RESIDUAL_EIGENBASIS_METHOD",
    "RESIDUAL_HOLONOMY_METHOD",
    "RESIDUAL_STANDARD_METHOD",
    "THREE_PLAQUETTE_ELECTRIC",
    "GaugeHamiltonian",
    "HamiltonianGapResult",
    "candidate_gap",
    "certified_hamiltonian_gap",
    "legal_chain",
    "legal_triple",
    "physical_basis",
    "plaquette_holonomy_trial_space",
    "rebuild_hamiltonian",
    "standard_basis_trial_space",
    "su2_three_plaquette_hamiltonian",
    "su2_two_plaquette_hamiltonian",
    "three_plaquette_basis",
]

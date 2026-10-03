# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact finite certificates for matrix-valued close-pair sieve research.

This module isolates a finite implication that can sit inside a bounded-prime-gap
argument.  For ordered shifts ``h_i`` and a target distance ``D``, attach a
positive-semidefinite rational matrix ``Q_i`` to each shift.  If

.. math::

   \sum_{i\in E} Q_i \preceq I

for every set ``E`` whose shifts are pairwise farther apart than ``D``, then any
configuration containing no close pair has matrix score at most its mass.  A
separate exact check can establish a strict finite variational crossing from
rational face Gram matrices.

The construction is motivated by matrix-valued Selberg scoring: unlike scalar
window weights, noncommuting matrices can distinguish different face directions.
What is certified here is only the finite rational matrix inequality and the
declared variational arithmetic.  The analytic sieve asymptotics, distribution
estimates, source-support argument, and finite-band passage remain explicit
external premises.  In particular, a crossing returned by this module is not a
proof of any infinite prime-gap statement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Any, Literal

from omnibias.core.proof.certificate import make_certificate

RationalInput = Fraction | int | str
RationalMatrix = tuple[tuple[Fraction, ...], ...]
FiniteStatus = Literal["PROVED", "BLOCKED"]
KnownEstimateKind = Literal["type_i", "type_ii", "type_iii", "remainder"]

ASYMPTOTIC_SIEVE_KIND = "shifted_prime_asymptotic_sieve"
MATRIX_CLOSE_PAIR_KIND = "matrix_close_pair_variational"
BOUNDED_GAP_QUADRATIC_KIND = "bounded_gap_quadratic_witness"
H1_182_SHIFTS: tuple[int, ...] = (
    0,
    2,
    6,
    12,
    20,
    26,
    30,
    32,
    36,
    42,
    48,
    50,
    56,
    60,
    68,
    72,
    78,
    86,
    90,
    92,
    98,
    102,
    110,
    116,
    120,
    126,
    132,
    138,
    140,
    146,
    152,
    156,
    158,
    162,
    168,
    170,
    176,
    180,
    182,
)
H1_186_SHIFTS: tuple[int, ...] = (*H1_182_SHIFTS, 186)
H1_186_PUBLISHED_MARGIN = Fraction(230382667, 10**13)
H1_186_REQUIRED_MARGIN = Fraction(1, 50000)
H1_186_RHO_STAR = Fraction(2624989, 10**7)
FIXED_SHIFT_DETERMINANT_KIND = "fixed_shift_determinant_replay"
FIXED_SHIFT_DETERMINANT_EXTERNAL_PREMISES: tuple[str, ...] = (
    "dualization of the outer absolute value is valid for the declared smooth dyadic form",
    "all determinant boxes are assigned to cited fixed-shift estimates without taking termwise absolute values",
    "zero frequencies and signed main terms cancel with the stated loss budget",
    "the complementary q-range admits a fixed-shift completion with net power saving",
    "the dyadic recombination is uniform in every coefficient of modulus at most one",
)
FIXED_SHIFT_KERNEL_SAVING_FLOOR = Fraction(41, 640)
FIXED_SHIFT_TARGET_SAVING = Fraction(1, 1000)
HEATH_BROWN_SMALL_FACTOR_ROOM = Fraction(11, 630)
ASYMPTOTIC_SIEVE_EXTERNAL_PREMISES: tuple[str, ...] = (
    "the shifted squarefree-prime sequence satisfies the asymptotic-sieve growth axioms",
    "the remainder estimate holds beyond level x^(2/3) in the declared range",
    "the listed Type I/II/III estimates discharge their complete rational cells",
    "the Mobius bilinear estimate holds uniformly on every required cell",
    "the finite partition and smoothing limits are uniform for all sufficiently large x",
)
FI_TERMINAL_ATLAS_EXTERNAL_PREMISES: tuple[str, ...] = (
    "the complete Friedlander-Iwaniec Vaughan and dyadic reduction is replayed",
    "Mirsky's shifted-squarefree-prime asymptotic supplies A(x) asymptotic to G*x",
    "the seventh divisor-moment estimate has a uniform effective constant",
    "the level-3/4 weighted remainder estimate R-prime holds",
    "the weighted Mobius bilinear estimate B-prime holds on the terminal product shell",
    "all log-scale partitions and limiting estimates are uniform for sufficiently large x",
)
TWIN_PRIME_EXTERNAL_PREMISES: tuple[str, ...] = (
    "the face Gram matrices arise from one source-valid Selberg trial",
    "the required prime-distribution estimates hold uniformly on every supported modulus",
    "the signed prime-minorant comparison preserves the declared matrix score",
    "the finite-band approximation and asymptotic passage preserve the strict surplus",
)
H1_182_EXTERNAL_PREMISES: tuple[str, ...] = (
    "the numerator and denominator matrices come from one parameterized k=39 Selberg trial",
    "all cap and source losses are outward-rounded and exhaustively inventoried",
    "the prime-minorant support inequalities hold as exact strict rational comparisons",
    "the dense-divisibility and finite-field distribution inputs apply to every supported modulus",
    "the finite trial is transferred to DHL[39,2] by a complete analytic instantiation",
)


def _as_fraction(value: RationalInput) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("matrix entries must be exact rationals, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int | str):
        return Fraction(value)
    raise TypeError("matrix entries must be Fraction, int, or rational strings")


def _as_matrix(raw: Sequence[Sequence[RationalInput]]) -> RationalMatrix:
    matrix = tuple(tuple(_as_fraction(value) for value in row) for row in raw)
    if not matrix:
        raise ValueError("matrix must be non-empty")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(i)):
        raise ValueError("matrix must be symmetric")
    return matrix


def _identity(n: int) -> RationalMatrix:
    return tuple(
        tuple(Fraction(int(i == j)) for j in range(n))
        for i in range(n)
    )


def _matrix_sum(matrices: Sequence[RationalMatrix], n: int) -> RationalMatrix:
    return tuple(
        tuple(sum((matrix[i][j] for matrix in matrices), Fraction(0)) for j in range(n))
        for i in range(n)
    )


def _matrix_sub(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(left[i][j] - right[i][j] for j in range(len(left)))
        for i in range(len(left))
    )


def _swap_symmetric(matrix: RationalMatrix, pivot: int) -> RationalMatrix:
    order = list(range(len(matrix)))
    order[0], order[pivot] = order[pivot], order[0]
    return tuple(tuple(matrix[i][j] for j in order) for i in order)


def _psd_pivots(matrix: RationalMatrix) -> tuple[Fraction, ...] | None:
    """Exact symmetric elimination; ``None`` means a negative direction exists."""

    n = len(matrix)
    if n == 0:
        return ()
    diagonal = tuple(matrix[i][i] for i in range(n))
    if all(value == 0 for value in diagonal):
        if all(value == 0 for row in matrix for value in row):
            return ()
        return None
    pivot_index = next(i for i, value in enumerate(diagonal) if value != 0)
    permuted = _swap_symmetric(matrix, pivot_index)
    pivot = permuted[0][0]
    if pivot < 0:
        return None
    schur = tuple(
        tuple(
            permuted[i][j] - permuted[i][0] * permuted[0][j] / pivot
            for j in range(1, n)
        )
        for i in range(1, n)
    )
    rest = _psd_pivots(schur)
    if rest is None:
        return None
    return (pivot, *rest)


@dataclass(frozen=True)
class PSDReport:
    """Exact positive-semidefiniteness report over the rationals."""

    psd: bool
    rank: int
    positive_pivots: tuple[Fraction, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "psd": self.psd,
            "rank": self.rank,
            "positive_pivots": [_fraction_payload(value) for value in self.positive_pivots],
        }


def psd_report(matrix: Sequence[Sequence[RationalInput]]) -> PSDReport:
    """Certify or refute positive semidefiniteness by exact congruence elimination."""

    exact = _as_matrix(matrix)
    pivots = _psd_pivots(exact)
    if pivots is None:
        return PSDReport(False, 0, ())
    return PSDReport(True, len(pivots), pivots)


def _primes_up_to(n: int) -> tuple[int, ...]:
    primes: list[int] = []
    for candidate in range(2, n + 1):
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
    return tuple(primes)


def _require_prime(prime: int) -> None:
    if prime < 2 or any(prime % divisor == 0 for divisor in range(2, isqrt(prime) + 1)):
        raise ValueError(f"{prime} is not prime")


@dataclass(frozen=True)
class LocalFactorIdentity:
    """One exact local identity in ``H * G = 2 * C_2``."""

    prime: int
    shifted_squarefree_factor: Fraction
    divisor_density: Fraction
    sieve_factor: Fraction
    combined_factor: Fraction
    twin_factor: Fraction
    verified: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "prime": self.prime,
            "shifted_squarefree_factor": _fraction_payload(
                self.shifted_squarefree_factor
            ),
            "divisor_density": _fraction_payload(self.divisor_density),
            "sieve_factor": _fraction_payload(self.sieve_factor),
            "combined_factor": _fraction_payload(self.combined_factor),
            "twin_factor": _fraction_payload(self.twin_factor),
            "verified": self.verified,
        }


def asymptotic_sieve_local_factor(prime: int) -> LocalFactorIdentity:
    """Return the exact local shifted-squarefree/asymptotic-sieve factor identity."""

    _require_prime(prime)
    if prime == 2:
        shifted = Fraction(1)
        density = Fraction(0)
        sieve = Fraction(2)
        twin = Fraction(2)
    else:
        shifted = Fraction(prime * prime - prime - 1, prime * (prime - 1))
        density = Fraction(prime - 1, prime * prime - prime - 1)
        sieve = (1 - density) / (1 - Fraction(1, prime))
        twin = 1 - Fraction(1, (prime - 1) ** 2)
    combined = shifted * sieve
    return LocalFactorIdentity(
        prime=prime,
        shifted_squarefree_factor=shifted,
        divisor_density=density,
        sieve_factor=sieve,
        combined_factor=combined,
        twin_factor=twin,
        verified=combined == twin,
    )


@dataclass(frozen=True)
class LocalProductCertificate:
    """Finite Euler-product prefix; no infinite-product tail is inferred."""

    prime_limit: int
    factors: tuple[LocalFactorIdentity, ...]
    shifted_squarefree_product: Fraction
    sieve_product: Fraction
    combined_product: Fraction
    twin_product: Fraction
    proved: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "prime_limit": self.prime_limit,
            "factors": [factor.to_payload() for factor in self.factors],
            "shifted_squarefree_product": _fraction_payload(
                self.shifted_squarefree_product
            ),
            "sieve_product": _fraction_payload(self.sieve_product),
            "combined_product": _fraction_payload(self.combined_product),
            "twin_product": _fraction_payload(self.twin_product),
            "proved": self.proved,
            "infinite_product_inferred": False,
        }


def certify_asymptotic_sieve_local_product(prime_limit: int) -> LocalProductCertificate:
    """Certify ``prod_{p<=P} G_p H_p = 2 prod_{2<p<=P}(1-1/(p-1)^2)``."""

    if prime_limit < 2:
        raise ValueError("prime_limit must be at least 2")
    factors = tuple(
        asymptotic_sieve_local_factor(prime)
        for prime in _primes_up_to(prime_limit)
    )
    shifted = _product(factor.shifted_squarefree_factor for factor in factors)
    sieve = _product(factor.sieve_factor for factor in factors)
    combined = _product(factor.combined_factor for factor in factors)
    twin = _product(factor.twin_factor for factor in factors)
    return LocalProductCertificate(
        prime_limit=prime_limit,
        factors=factors,
        shifted_squarefree_product=shifted,
        sieve_product=sieve,
        combined_product=combined,
        twin_product=twin,
        proved=all(factor.verified for factor in factors) and shifted * sieve == twin,
    )


def seal_asymptotic_sieve_local_product(
    report: LocalProductCertificate,
) -> dict[str, Any]:
    """Seal a finite local-factor prefix without asserting the infinite product."""

    payload = report.to_payload()
    payload["type"] = ASYMPTOTIC_SIEVE_KIND
    payload["subtype"] = "local_factor_prefix"
    return make_certificate(
        claim=ASYMPTOTIC_SIEVE_KIND,
        payload=payload,
        honesty=asymptotic_sieve_honesty(
            finite_reduction=False,
            finite_local_factors=report.proved,
        ),
    )


def local_factor_obligation_certificates(
    report: LocalProductCertificate,
) -> tuple[dict[str, Any], ...]:
    """Emit finite ``rational_identity`` obligations for independent kernel replay."""

    if not report.proved:
        raise ValueError("local-factor report must be proved before obligation emission")
    honesty = asymptotic_sieve_honesty(
        finite_reduction=False,
        finite_local_factors=True,
    )
    identities = [
        (
            f"shifted-prime local factor at p={factor.prime}",
            factor.combined_factor,
            factor.twin_factor,
            {"prime": factor.prime, "scope": "finite_local_factor"},
        )
        for factor in report.factors
    ]
    identities.append(
        (
            f"shifted-prime local product through {report.prime_limit}",
            report.combined_product,
            report.twin_product,
            {"prime_limit": report.prime_limit, "scope": "finite_product_prefix"},
        )
    )
    certificates: list[dict[str, Any]] = []
    for claim, computed, expected, meta in identities:
        pc, qc = computed.numerator, computed.denominator
        pe, qe = expected.numerator, expected.denominator
        certificates.append(
            make_certificate(
                claim=claim,
                payload={
                    "type": "rational_identity",
                    "lhs_terms": [[qe, pc], [-qc, pe]],
                    "rhs": 0,
                },
                honesty=dict(honesty),
                meta=meta,
            )
        )
    return tuple(certificates)


@dataclass(frozen=True)
class RationalCell:
    """A half-open rational cell in log-factor coordinates ``(log r, log s)/log x``."""

    r_lo: Fraction
    r_hi: Fraction
    s_lo: Fraction
    s_hi: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_lo", _as_fraction(self.r_lo))
        object.__setattr__(self, "r_hi", _as_fraction(self.r_hi))
        object.__setattr__(self, "s_lo", _as_fraction(self.s_lo))
        object.__setattr__(self, "s_hi", _as_fraction(self.s_hi))
        if self.r_lo >= self.r_hi or self.s_lo >= self.s_hi:
            raise ValueError("cell bounds must have positive width")

    @property
    def area(self) -> Fraction:
        return (self.r_hi - self.r_lo) * (self.s_hi - self.s_lo)

    def contains(self, r_value: Fraction, s_value: Fraction) -> bool:
        return (
            self.r_lo < r_value < self.r_hi
            and self.s_lo < s_value < self.s_hi
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "r": [_fraction_payload(self.r_lo), _fraction_payload(self.r_hi)],
            "s": [_fraction_payload(self.s_lo), _fraction_payload(self.s_hi)],
        }


@dataclass(frozen=True)
class KnownEstimate:
    """A named non-parity estimate assigned to one discharged factor cell."""

    name: str
    kind: KnownEstimateKind

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("estimate name must be non-empty")
        if self.kind not in {"type_i", "type_ii", "type_iii", "remainder"}:
            raise ValueError("estimate kind must be Type I/II/III or remainder")

    def to_payload(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


def _union_area(cells: Sequence[RationalCell]) -> Fraction:
    if not cells:
        return Fraction(0)
    r_points = sorted({point for cell in cells for point in (cell.r_lo, cell.r_hi)})
    s_points = sorted({point for cell in cells for point in (cell.s_lo, cell.s_hi)})
    area = Fraction(0)
    for r_lo, r_hi in zip(r_points, r_points[1:], strict=False):
        r_mid = (r_lo + r_hi) / 2
        for s_lo, s_hi in zip(s_points, s_points[1:], strict=False):
            s_mid = (s_lo + s_hi) / 2
            if any(cell.contains(r_mid, s_mid) for cell in cells):
                area += (r_hi - r_lo) * (s_hi - s_lo)
    return area


def _partition_checks(
    classical: Sequence[RationalCell],
    discharged: Sequence[RationalCell],
    required: Sequence[RationalCell],
) -> tuple[bool, bool, str]:
    all_cells = (*classical, *discharged, *required)
    r_points = sorted({point for cell in all_cells for point in (cell.r_lo, cell.r_hi)})
    s_points = sorted({point for cell in all_cells for point in (cell.s_lo, cell.s_hi)})
    exact_partition = True
    internally_disjoint = True
    detail = "exact rational partition"
    for r_lo, r_hi in zip(r_points, r_points[1:], strict=False):
        r_mid = (r_lo + r_hi) / 2
        for s_lo, s_hi in zip(s_points, s_points[1:], strict=False):
            s_mid = (s_lo + s_hi) / 2
            memberships = tuple(
                sum(cell.contains(r_mid, s_mid) for cell in group)
                for group in (classical, discharged, required)
            )
            in_classical, in_discharged, in_required = (
                count > 0 for count in memberships
            )
            if any(count > 1 for count in memberships):
                internally_disjoint = False
                detail = "at least one region contains overlapping positive-area cells"
            if (
                (in_discharged or in_required) != in_classical
                or (in_discharged and in_required)
            ):
                exact_partition = False
                detail = "discharged and required cells do not partition the classical region"
    return exact_partition, internally_disjoint, detail


@dataclass(frozen=True)
class MobiusCellReductionCertificate:
    """Exact cell atlas for the still-required Möbius bilinear estimate."""

    nu: Fraction
    classical_region: tuple[RationalCell, ...]
    discharged_region: tuple[RationalCell, ...]
    required_region: tuple[RationalCell, ...]
    discharged_by: tuple[KnownEstimate, ...]
    classical_area: Fraction
    required_area: Fraction
    exact_partition: bool
    internally_disjoint: bool
    strict_reduction: bool
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": ASYMPTOTIC_SIEVE_KIND,
            "nu": _fraction_payload(self.nu),
            "classical_region": [cell.to_payload() for cell in self.classical_region],
            "discharged_region": [cell.to_payload() for cell in self.discharged_region],
            "required_mobius_region": [cell.to_payload() for cell in self.required_region],
            "discharged_by": [estimate.to_payload() for estimate in self.discharged_by],
            "classical_area": _fraction_payload(self.classical_area),
            "required_area": _fraction_payload(self.required_area),
            "exact_partition": self.exact_partition,
            "internally_disjoint": self.internally_disjoint,
            "strict_reduction": self.strict_reduction,
            "status": self.status,
            "external_premises": list(self.external_premises),
        }


def asymptotic_sieve_honesty(
    *,
    finite_reduction: bool,
    finite_local_factors: bool = False,
) -> dict[str, bool]:
    """Honesty payload separating a finite atlas from every analytic hypothesis."""

    return {
        "unproven_claim": False,
        "finite_local_factor_check": finite_local_factors,
        "finite_mobius_cell_partition_check": finite_reduction,
        "friedlander_iwaniec_reduction_replayed": False,
        "remainder_estimate_proved": False,
        "mobius_bilinear_estimate_proved": False,
        "uniform_asymptotic_passage_proved": False,
        "twin_prime_conjecture_proof_claim": False,
        "hardy_littlewood_asymptotic_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def certify_mobius_cell_reduction(
    *,
    nu: RationalInput,
    classical_region: Sequence[RationalCell],
    discharged_region: Sequence[RationalCell],
    required_region: Sequence[RationalCell],
    discharged_by: Sequence[KnownEstimate],
) -> MobiusCellReductionCertificate:
    """Certify a strict rational reduction of the required parity-sensitive region."""

    exact_nu = _as_fraction(nu)
    if exact_nu <= Fraction(2, 3):
        raise ValueError("the asymptotic-sieve level exponent nu must exceed 2/3")
    classical = tuple(classical_region)
    discharged = tuple(discharged_region)
    required = tuple(required_region)
    estimates = tuple(discharged_by)
    if not classical or not discharged or not required:
        raise ValueError("classical, discharged, and required regions must be non-empty")
    if len(estimates) != len(discharged):
        raise ValueError("one named estimate is required per discharged cell")
    exact_partition, disjoint, _ = _partition_checks(
        classical,
        discharged,
        required,
    )
    classical_area = _union_area(classical)
    required_area = _union_area(required)
    strict = required_area < classical_area
    finite_reduction = exact_partition and disjoint and strict
    status: FiniteStatus = "PROVED" if finite_reduction else "BLOCKED"
    return MobiusCellReductionCertificate(
        nu=exact_nu,
        classical_region=classical,
        discharged_region=discharged,
        required_region=required,
        discharged_by=estimates,
        classical_area=classical_area,
        required_area=required_area,
        exact_partition=exact_partition,
        internally_disjoint=disjoint,
        strict_reduction=strict,
        status=status,
        external_premises=ASYMPTOTIC_SIEVE_EXTERNAL_PREMISES,
        honesty=asymptotic_sieve_honesty(finite_reduction=finite_reduction),
    )


def seal_mobius_cell_reduction(
    report: MobiusCellReductionCertificate,
) -> dict[str, Any]:
    """Seal only the finite cell claim; all asymptotic premises remain external."""

    return make_certificate(
        claim=ASYMPTOTIC_SIEVE_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


@dataclass(frozen=True)
class LogAffineBound:
    """One exact halfspace in normalized log coordinates ``(u, v, w, ell)``."""

    u: Fraction = Fraction(0)
    v: Fraction = Fraction(0)
    w: Fraction = Fraction(0)
    ell: Fraction = Fraction(0)
    rhs: Fraction = Fraction(0)
    strict: bool = False

    def __post_init__(self) -> None:
        for name in ("u", "v", "w", "ell", "rhs"):
            object.__setattr__(self, name, _as_fraction(getattr(self, name)))

    def holds(
        self,
        *,
        u: RationalInput,
        v: RationalInput,
        w: RationalInput,
        ell: RationalInput = 0,
    ) -> bool:
        value = (
            self.u * _as_fraction(u)
            + self.v * _as_fraction(v)
            + self.w * _as_fraction(w)
            + self.ell * _as_fraction(ell)
        )
        return value < self.rhs if self.strict else value <= self.rhs

    def to_payload(self) -> dict[str, object]:
        return {
            "coefficients": {
                "u": _fraction_payload(self.u),
                "v": _fraction_payload(self.v),
                "w": _fraction_payload(self.w),
                "ell": _fraction_payload(self.ell),
            },
            "rhs": _fraction_payload(self.rhs),
            "strict": self.strict,
        }


@dataclass(frozen=True)
class LogPolyhedralCell:
    """A named finite intersection of exact normalized-log halfspaces."""

    name: str
    bounds: tuple[LogAffineBound, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("cell name must be non-empty")
        if not self.bounds:
            raise ValueError("polyhedral cell needs at least one bound")

    def contains(
        self,
        *,
        u: RationalInput,
        v: RationalInput,
        w: RationalInput,
        ell: RationalInput = 0,
    ) -> bool:
        return all(bound.holds(u=u, v=v, w=w, ell=ell) for bound in self.bounds)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bounds": [bound.to_payload() for bound in self.bounds],
        }


@dataclass(frozen=True)
class FIAtlasParameters:
    """Rational fallback parameters for the FI terminal-product atlas."""

    nu: Fraction = Fraction(3, 4)
    large_delta_exponent: Fraction = Fraction(1, 16)
    small_delta_log_power: int = 1
    lower_n_exponent: Fraction = Fraction(5, 16)
    upper_n_exponent: Fraction = Fraction(1, 2)
    c_exponent: Fraction = Fraction(1, 4)
    terminal_cutoff: Fraction = Fraction(63, 64)
    moving_log_power: int = 131
    divisor_moment_degree: int = 127
    mangoldt_log_degree: int = 1

    def __post_init__(self) -> None:
        for name in (
            "nu",
            "large_delta_exponent",
            "lower_n_exponent",
            "upper_n_exponent",
            "c_exponent",
            "terminal_cutoff",
        ):
            object.__setattr__(self, name, _as_fraction(getattr(self, name)))
        if self.nu <= Fraction(2, 3):
            raise ValueError("FI level exponent nu must exceed 2/3")
        if self.lower_n_exponent != self.nu / 2 - self.large_delta_exponent:
            raise ValueError("lower N exponent must equal nu/2 - large_delta_exponent")
        if self.small_delta_log_power < 1:
            raise ValueError("small-delta logarithmic power must be positive")
        if self.c_exponent != 1 - self.nu:
            raise ValueError("C exponent must equal 1 - nu")
        if not (
            0 < self.lower_n_exponent < self.upper_n_exponent < self.terminal_cutoff < 1
        ):
            raise ValueError("FI rational fallback exponents are inconsistent")
        if self.moving_log_power <= (
            self.divisor_moment_degree + self.mangoldt_log_degree
        ):
            raise ValueError("moving shell must leave a positive logarithmic saving")


def _fi_base_bounds(
    *,
    moving: bool,
    small_delta_log_power: int = 1,
) -> tuple[LogAffineBound, ...]:
    upper_v = LogAffineBound(
        v=Fraction(1),
        ell=Fraction(small_delta_log_power if moving else 0),
        rhs=Fraction(1, 2),
        strict=True,
    )
    return (
        LogAffineBound(u=Fraction(-1), rhs=Fraction(0)),
        LogAffineBound(v=Fraction(-1), rhs=Fraction(-5, 16), strict=True),
        upper_v,
        LogAffineBound(w=Fraction(-1), rhs=Fraction(0)),
        LogAffineBound(w=Fraction(1), rhs=Fraction(1, 4)),
        LogAffineBound(u=Fraction(1), v=Fraction(1), rhs=Fraction(1)),
    )


@dataclass(frozen=True)
class FITerminalAtlasCertificate:
    """Exact geometry of a terminal-shell reduction of the FI bilinear region."""

    parameters: FIAtlasParameters
    classical: LogPolyhedralCell
    discharged: LogPolyhedralCell
    required: LogPolyhedralCell
    moving_classical: LogPolyhedralCell
    moving_discharged: LogPolyhedralCell
    moving_required: LogPolyhedralCell
    classical_volume: Fraction
    discharged_volume: Fraction
    required_volume: Fraction
    unresolved_fraction: Fraction
    far_log_saving: int
    partition_proved: bool
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": ASYMPTOTIC_SIEVE_KIND,
            "subtype": "fi_terminal_log_atlas",
            "parameters": {
                "nu": _fraction_payload(self.parameters.nu),
                "large_delta_exponent": _fraction_payload(
                    self.parameters.large_delta_exponent
                ),
                "small_delta_log_power": self.parameters.small_delta_log_power,
                "lower_n_exponent": _fraction_payload(
                    self.parameters.lower_n_exponent
                ),
                "upper_n_exponent": _fraction_payload(
                    self.parameters.upper_n_exponent
                ),
                "c_exponent": _fraction_payload(self.parameters.c_exponent),
                "terminal_cutoff": _fraction_payload(
                    self.parameters.terminal_cutoff
                ),
                "moving_log_power": self.parameters.moving_log_power,
                "divisor_moment_degree": self.parameters.divisor_moment_degree,
                "mangoldt_log_degree": self.parameters.mangoldt_log_degree,
            },
            "classical": self.classical.to_payload(),
            "discharged": self.discharged.to_payload(),
            "required": self.required.to_payload(),
            "moving_classical": self.moving_classical.to_payload(),
            "moving_discharged": self.moving_discharged.to_payload(),
            "moving_required": self.moving_required.to_payload(),
            "classical_volume": _fraction_payload(self.classical_volume),
            "discharged_volume": _fraction_payload(self.discharged_volume),
            "required_volume": _fraction_payload(self.required_volume),
            "unresolved_fraction": _fraction_payload(self.unresolved_fraction),
            "far_log_saving": self.far_log_saving,
            "partition_proved": self.partition_proved,
            "status": self.status,
            "external_premises": list(self.external_premises),
        }


def fi_terminal_atlas_honesty(*, partition_proved: bool) -> dict[str, bool]:
    """Honesty payload for the finite atlas and conditional far-region estimate."""

    return {
        "unproven_claim": False,
        "finite_log_polyhedral_partition_check": partition_proved,
        "finite_volume_ratio_check": partition_proved,
        "friedlander_iwaniec_reduction_replayed": False,
        "divisor_moment_uniform_constant_proved": False,
        "remainder_estimate_proved": False,
        "mobius_bilinear_estimate_proved": False,
        "uniform_asymptotic_passage_proved": False,
        "twin_prime_conjecture_proof_claim": False,
        "hardy_littlewood_asymptotic_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def certify_fi_terminal_atlas(
    parameters: FIAtlasParameters | None = None,
) -> FITerminalAtlasCertificate:
    """Certify the fixed ``63/64`` atlas and encode the moving log-shell atlas."""

    params = parameters or FIAtlasParameters()
    base = _fi_base_bounds(
        moving=False,
        small_delta_log_power=params.small_delta_log_power,
    )
    classical = LogPolyhedralCell("fi_classical_rational", base)
    discharged = LogPolyhedralCell(
        "fi_far_product_rational",
        (
            *base,
            LogAffineBound(
                u=Fraction(1),
                v=Fraction(1),
                rhs=params.terminal_cutoff,
            ),
        ),
    )
    required = LogPolyhedralCell(
        "fi_terminal_product_rational",
        (
            *base,
            LogAffineBound(
                u=Fraction(-1),
                v=Fraction(-1),
                rhs=-params.terminal_cutoff,
                strict=True,
            ),
        ),
    )
    moving_base = _fi_base_bounds(
        moving=True,
        small_delta_log_power=params.small_delta_log_power,
    )
    moving_classical = LogPolyhedralCell("fi_classical_moving", moving_base)
    moving_discharged = LogPolyhedralCell(
        "fi_far_product_moving",
        (
            *moving_base,
            LogAffineBound(
                u=Fraction(1),
                v=Fraction(1),
                ell=Fraction(params.moving_log_power),
                rhs=Fraction(1),
            ),
        ),
    )
    moving_required = LogPolyhedralCell(
        "fi_terminal_product_moving",
        (
            *moving_base,
            LogAffineBound(
                u=Fraction(-1),
                v=Fraction(-1),
                ell=Fraction(-params.moving_log_power),
                rhs=Fraction(-1),
                strict=True,
            ),
        ),
    )
    v_width = params.upper_n_exponent - params.lower_n_exponent
    classical_volume = params.c_exponent * (
        params.upper_n_exponent
        - params.upper_n_exponent**2 / 2
        - params.lower_n_exponent
        + params.lower_n_exponent**2 / 2
    )
    required_volume = (
        params.c_exponent * v_width * (1 - params.terminal_cutoff)
    )
    discharged_volume = classical_volume - required_volume
    unresolved_fraction = required_volume / classical_volume
    far_log_saving = params.moving_log_power - (
        params.divisor_moment_degree + params.mangoldt_log_degree
    )
    partition_proved = (
        0 < required_volume < classical_volume
        and discharged_volume + required_volume == classical_volume
        and unresolved_fraction == required_volume / classical_volume
        and far_log_saving > 0
    )
    status: FiniteStatus = "PROVED" if partition_proved else "BLOCKED"
    return FITerminalAtlasCertificate(
        parameters=params,
        classical=classical,
        discharged=discharged,
        required=required,
        moving_classical=moving_classical,
        moving_discharged=moving_discharged,
        moving_required=moving_required,
        classical_volume=classical_volume,
        discharged_volume=discharged_volume,
        required_volume=required_volume,
        unresolved_fraction=unresolved_fraction,
        far_log_saving=far_log_saving,
        partition_proved=partition_proved,
        status=status,
        external_premises=FI_TERMINAL_ATLAS_EXTERNAL_PREMISES,
        honesty=fi_terminal_atlas_honesty(partition_proved=partition_proved),
    )


def fi_rational_terminal_family(
    epsilons: Sequence[RationalInput],
) -> tuple[FITerminalAtlasCertificate, ...]:
    """Certify fixed shells ``1-epsilon < u+v <= 1`` for rational epsilons.

    Every fixed positive epsilon gives a power-saving far region.  The returned
    family makes explicit that normalized-log volume can tend to zero while the
    fixed-shift parity estimate remains wholly unresolved.
    """

    reports: list[FITerminalAtlasCertificate] = []
    for raw_epsilon in epsilons:
        epsilon = _as_fraction(raw_epsilon)
        if not 0 < epsilon < Fraction(1, 2):
            raise ValueError("terminal-shell epsilon must lie strictly between 0 and 1/2")
        reports.append(
            certify_fi_terminal_atlas(
                FIAtlasParameters(terminal_cutoff=1 - epsilon)
            )
        )
    return tuple(reports)


def fi_atlas_obligation_certificates(
    report: FITerminalAtlasCertificate,
) -> tuple[dict[str, Any], ...]:
    """Emit Lean-ready integer identities for the finite atlas arithmetic."""

    if not report.partition_proved:
        raise ValueError("FI terminal atlas must be proved before obligation emission")
    honesty = dict(report.honesty)
    pairs = (
        (
            "FI rational atlas volume partition",
            report.discharged_volume + report.required_volume,
            report.classical_volume,
        ),
        (
            "FI rational atlas unresolved ratio",
            report.classical_volume,
            38 * report.required_volume,
        ),
        (
            "FI moving-shell logarithmic saving",
            Fraction(report.parameters.moving_log_power),
            Fraction(
                report.parameters.divisor_moment_degree
                + report.parameters.mangoldt_log_degree
                + report.far_log_saving
            ),
        ),
    )
    certificates: list[dict[str, Any]] = []
    for claim, computed, expected in pairs:
        pc, qc = computed.numerator, computed.denominator
        pe, qe = expected.numerator, expected.denominator
        certificates.append(
            make_certificate(
                claim=claim,
                payload={
                    "type": "rational_identity",
                    "lhs_terms": [[qe, pc], [-qc, pe]],
                    "rhs": 0,
                },
                honesty=honesty,
                meta={"scope": "finite_fi_terminal_atlas"},
            )
        )
    return tuple(certificates)


def seal_fi_terminal_atlas(
    report: FITerminalAtlasCertificate,
) -> dict[str, Any]:
    """Seal the finite atlas; analytic estimates remain explicit external premises."""

    return make_certificate(
        claim=ASYMPTOTIC_SIEVE_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


def _factor_integer(value: int) -> tuple[tuple[int, int], ...]:
    if value < 1:
        raise ValueError("integer arithmetic is defined for positive values")
    factors: list[tuple[int, int]] = []
    remainder = value
    divisor = 2
    while divisor * divisor <= remainder:
        exponent = 0
        while remainder % divisor == 0:
            remainder //= divisor
            exponent += 1
        if exponent:
            factors.append((divisor, exponent))
        divisor += 1
    if remainder > 1:
        factors.append((remainder, 1))
    return tuple(factors)


def mobius_integer(value: int) -> int:
    """Return the exact Möbius value of one positive integer."""

    factors = _factor_integer(value)
    if any(exponent > 1 for _prime, exponent in factors):
        return 0
    return -1 if len(factors) % 2 else 1


def _divisors(value: int) -> tuple[int, ...]:
    divisors = [1]
    for prime, exponent in _factor_integer(value):
        previous = tuple(divisors)
        power = 1
        for _ in range(exponent):
            power *= prime
            divisors.extend(item * power for item in previous)
    return tuple(sorted(divisors))


def gamma_truncation(value: int, cutoff: int) -> int:
    """Return ``sum_{d|value, d<=cutoff} mu(d)`` exactly."""

    if cutoff < 1:
        raise ValueError("gamma cutoff must be positive")
    return sum(
        mobius_integer(divisor)
        for divisor in _divisors(value)
        if divisor <= cutoff
    )


def verify_gamma_mobius_identity(value: int, cutoff: int) -> bool:
    """Verify FI's truncated-gamma identity on one squarefree integer."""

    if mobius_integer(value) == 0:
        raise ValueError("gamma-Mobius identity is scoped to squarefree integers")
    left = mobius_integer(value) * gamma_truncation(value, cutoff)
    right = sum(
        mobius_integer(value // divisor)
        for divisor in _divisors(value)
        if divisor <= cutoff
    )
    return left == right


CoefficientVector = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VaughanCoefficientCase:
    """Formal coefficients of FI equation (3.1) for one integer ``n``."""

    n: int
    left: CoefficientVector
    first: CoefficientVector
    second: CoefficientVector
    f1: CoefficientVector
    f2: CoefficientVector
    f3: CoefficientVector
    right: CoefficientVector
    terminal_pairs: tuple[tuple[int, int], ...]
    assigned_pairs: tuple[tuple[str, int, int], ...]
    identity_proved: bool
    terminal_partition_proved: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "n": self.n,
            "left": [list(item) for item in self.left],
            "first": [list(item) for item in self.first],
            "second": [list(item) for item in self.second],
            "f1": [list(item) for item in self.f1],
            "f2": [list(item) for item in self.f2],
            "f3": [list(item) for item in self.f3],
            "right": [list(item) for item in self.right],
            "terminal_pairs": [list(item) for item in self.terminal_pairs],
            "assigned_pairs": [list(item) for item in self.assigned_pairs],
            "identity_proved": self.identity_proved,
            "terminal_partition_proved": self.terminal_partition_proved,
        }


def _coefficient_vector(
    coefficients: Mapping[int, int],
    divisors: Sequence[int],
) -> CoefficientVector:
    return tuple((divisor, int(coefficients.get(divisor, 0))) for divisor in divisors)


def _vaughan_terminal_region(
    *,
    b: int,
    c: int,
    y: Fraction,
    z: Fraction,
    s: int,
) -> str | None:
    if b > s * y and c > s * z:
        return "f1"
    if y < b <= s * y and c > z:
        return "f2"
    if b > s * y and z < c <= s * z:
        return "f3"
    return None


def fi_vaughan_coefficient_case(
    n: int,
    *,
    y: RationalInput,
    z: RationalInput,
    s: int,
) -> VaughanCoefficientCase:
    r"""Replay FI (3.1) coefficientwise, hence for every arithmetic function.

    The exact endpoint convention is
    ``b <= y``, ``c <= z``, ``b > s*y``, and ``c > s*z``.  The three terminal
    regions are pairwise disjoint and cover ``b > y, c > z``.
    """

    if n < 1:
        raise ValueError("n must be positive")
    exact_y = _as_fraction(y)
    exact_z = _as_fraction(z)
    if exact_y < 1 or exact_z < 1:
        raise ValueError("Vaughan thresholds y and z must be at least one")
    if s <= 1:
        raise ValueError("Vaughan dilation s must exceed one")
    divisors = _divisors(n)
    left = {divisor: int(divisor == n and n > exact_z) for divisor in divisors}
    first = {divisor: 0 for divisor in divisors}
    second = {divisor: 0 for divisor in divisors}
    pieces = {
        "f1": {divisor: 0 for divisor in divisors},
        "f2": {divisor: 0 for divisor in divisors},
        "f3": {divisor: 0 for divisor in divisors},
    }
    terminal_pairs: list[tuple[int, int]] = []
    assigned_pairs: list[tuple[str, int, int]] = []
    for b in divisors:
        mu_b = mobius_integer(b)
        if b <= exact_y:
            for c in _divisors(n // b):
                first[c] += mu_b
                if c <= exact_z:
                    second[c] += mu_b
        for c in _divisors(n // b):
            if b <= exact_y or c <= exact_z:
                continue
            terminal_pairs.append((b, c))
            region = _vaughan_terminal_region(
                b=b,
                c=c,
                y=exact_y,
                z=exact_z,
                s=s,
            )
            if region is not None:
                pieces[region][c] += mu_b
                assigned_pairs.append((region, b, c))
    right = {
        divisor: (
            first[divisor]
            - second[divisor]
            + pieces["f1"][divisor]
            + pieces["f2"][divisor]
            + pieces["f3"][divisor]
        )
        for divisor in divisors
    }
    assigned_plain = tuple((b, c) for _region, b, c in assigned_pairs)
    terminal_partition_proved = (
        len(assigned_plain) == len(set(assigned_plain))
        and set(assigned_plain) == set(terminal_pairs)
    )
    return VaughanCoefficientCase(
        n=n,
        left=_coefficient_vector(left, divisors),
        first=_coefficient_vector(first, divisors),
        second=_coefficient_vector(second, divisors),
        f1=_coefficient_vector(pieces["f1"], divisors),
        f2=_coefficient_vector(pieces["f2"], divisors),
        f3=_coefficient_vector(pieces["f3"], divisors),
        right=_coefficient_vector(right, divisors),
        terminal_pairs=tuple(terminal_pairs),
        assigned_pairs=tuple(assigned_pairs),
        identity_proved=left == right,
        terminal_partition_proved=terminal_partition_proved,
    )


FI_COMBINATORIAL_EXTERNAL_PREMISES: tuple[str, ...] = (
    "the logarithmic smoothing operator is justified and preserves equation (3.2)",
    "the upper-bound sieve rho has the support and positivity properties used in FI",
    "the T, S1, S2, and S3 analytic estimates hold uniformly in the FI ranges",
    "the weighted remainder estimate R-prime holds beyond level x^(2/3)",
    "the fixed-shift weighted Mobius estimate B-prime holds in every required cell",
    "the asymptotic passage from S(x,Z) to S(x) is uniform",
)


@dataclass(frozen=True)
class OpenClosedInterval:
    """Exact interval ``(lo, hi]`` used by FI's dyadic decomposition."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", _as_fraction(self.lo))
        object.__setattr__(self, "hi", _as_fraction(self.hi))
        if self.lo >= self.hi:
            raise ValueError("open-closed interval must have lo < hi")

    def contains(self, value: RationalInput) -> bool:
        exact = _as_fraction(value)
        return self.lo < exact <= self.hi

    def to_payload(self) -> list[list[str]]:
        return [_fraction_payload(self.lo), _fraction_payload(self.hi)]


def fi_dyadic_intervals(
    y: RationalInput,
    s: int,
) -> tuple[OpenClosedInterval, ...]:
    """Return the exact disjoint cover of ``(y, s*y]`` for dyadic ``s``."""

    exact_y = _as_fraction(y)
    if exact_y <= 0:
        raise ValueError("dyadic base y must be positive")
    if s <= 1 or s & (s - 1):
        raise ValueError("FI dyadic replay requires s to be a power of two")
    intervals: list[OpenClosedInterval] = []
    current = exact_y
    dilation = 1
    while dilation < s:
        following = 2 * current
        intervals.append(OpenClosedInterval(current, following))
        current = following
        dilation *= 2
    return tuple(intervals)


def verify_dyadic_membership(
    value: RationalInput,
    *,
    y: RationalInput,
    s: int,
) -> bool:
    """Check exact pointwise ownership in FI's open-closed dyadic cover."""

    exact_value = _as_fraction(value)
    exact_y = _as_fraction(y)
    intervals = fi_dyadic_intervals(exact_y, s)
    owners = sum(interval.contains(exact_value) for interval in intervals)
    return owners == int(exact_y < exact_value <= s * exact_y)


@dataclass(frozen=True)
class DyadicSelectionCertificate:
    """Exact squared comparison for FI's unique power-of-two dilation."""

    a_squared: Fraction
    s: int
    lower_open: bool
    upper_closed: bool
    unique: bool
    status: FiniteStatus

    def to_payload(self) -> dict[str, object]:
        return {
            "a_squared": _fraction_payload(self.a_squared),
            "s": self.s,
            "lower_open": self.lower_open,
            "upper_closed": self.upper_closed,
            "unique": self.unique,
            "status": self.status,
        }


def select_fi_dyadic_s(a_squared: RationalInput) -> DyadicSelectionCertificate:
    r"""Select the unique power of two in ``(a, 2a]`` without square roots."""

    exact = _as_fraction(a_squared)
    if exact <= 0:
        raise ValueError("a_squared must be positive")
    s = 1
    while Fraction(s * s) <= exact:
        s *= 2
    lower_open = exact < s * s
    upper_closed = s * s <= 4 * exact
    previous_is_out = s == 1 or (s // 2) ** 2 <= exact
    next_is_out = (2 * s) ** 2 > 4 * exact
    unique = lower_open and upper_closed and previous_is_out and next_is_out
    return DyadicSelectionCertificate(
        a_squared=exact,
        s=s,
        lower_open=lower_open,
        upper_closed=upper_closed,
        unique=unique,
        status="PROVED" if unique else "BLOCKED",
    )


@dataclass(frozen=True)
class FICombinatorialReplayCertificate:
    """Finite universal-coefficient replay of FI's Vaughan identity."""

    n_max: int
    y: Fraction
    z: Fraction
    s: int
    dyadic_intervals: tuple[OpenClosedInterval, ...]
    cases: tuple[VaughanCoefficientCase, ...]
    identity_proved: bool
    terminal_partition_proved: bool
    dyadic_partition_proved: bool
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": ASYMPTOTIC_SIEVE_KIND,
            "subtype": "fi_combinatorial_replay",
            "n_max": self.n_max,
            "y": _fraction_payload(self.y),
            "z": _fraction_payload(self.z),
            "s": self.s,
            "dyadic_intervals": [
                interval.to_payload() for interval in self.dyadic_intervals
            ],
            "cases": [case.to_payload() for case in self.cases],
            "identity_proved": self.identity_proved,
            "terminal_partition_proved": self.terminal_partition_proved,
            "dyadic_partition_proved": self.dyadic_partition_proved,
            "status": self.status,
            "external_premises": list(self.external_premises),
        }


def fi_combinatorial_honesty(
    *,
    identity_proved: bool,
    terminal_partition_proved: bool,
    dyadic_partition_proved: bool,
) -> dict[str, bool]:
    """Separate the exact FI combinatorics from every analytic estimate."""

    finite = identity_proved and terminal_partition_proved and dyadic_partition_proved
    return {
        "unproven_claim": False,
        "finite_fi_vaughan_identity_check": identity_proved,
        "finite_fi_terminal_partition_check": terminal_partition_proved,
        "finite_fi_dyadic_partition_check": dyadic_partition_proved,
        "friedlander_iwaniec_combinatorial_replay": finite,
        "friedlander_iwaniec_reduction_replayed": False,
        "remainder_estimate_proved": False,
        "mobius_bilinear_estimate_proved": False,
        "uniform_asymptotic_passage_proved": False,
        "twin_prime_conjecture_proof_claim": False,
        "hardy_littlewood_asymptotic_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def certify_fi_combinatorial_replay(
    *,
    n_max: int,
    y: RationalInput,
    z: RationalInput,
    s: int,
) -> FICombinatorialReplayCertificate:
    """Certify FI (3.1) and its terminal/dyadic partitions over ``1..n_max``."""

    if n_max < 1:
        raise ValueError("n_max must be positive")
    if s <= 1 or s & (s - 1):
        raise ValueError("FI dyadic replay requires s to be a power of two")
    exact_y = _as_fraction(y)
    exact_z = _as_fraction(z)
    cases = tuple(
        fi_vaughan_coefficient_case(n, y=exact_y, z=exact_z, s=s)
        for n in range(1, n_max + 1)
    )
    intervals = fi_dyadic_intervals(exact_y, s)
    dyadic_partition_proved = (
        bool(intervals)
        and intervals[0].lo == exact_y
        and intervals[-1].hi == s * exact_y
        and all(
            left.hi == right.lo
            for left, right in zip(intervals, intervals[1:], strict=False)
        )
    )
    identity_proved = all(case.identity_proved for case in cases)
    terminal_partition_proved = all(
        case.terminal_partition_proved for case in cases
    )
    status: FiniteStatus = (
        "PROVED"
        if identity_proved and terminal_partition_proved and dyadic_partition_proved
        else "BLOCKED"
    )
    honesty = fi_combinatorial_honesty(
        identity_proved=identity_proved,
        terminal_partition_proved=terminal_partition_proved,
        dyadic_partition_proved=dyadic_partition_proved,
    )
    return FICombinatorialReplayCertificate(
        n_max=n_max,
        y=exact_y,
        z=exact_z,
        s=s,
        dyadic_intervals=intervals,
        cases=cases,
        identity_proved=identity_proved,
        terminal_partition_proved=terminal_partition_proved,
        dyadic_partition_proved=dyadic_partition_proved,
        status=status,
        external_premises=FI_COMBINATORIAL_EXTERNAL_PREMISES,
        honesty=honesty,
    )


def fi_combinatorial_obligation_certificates(
    report: FICombinatorialReplayCertificate,
) -> tuple[dict[str, Any], ...]:
    """Emit Lean-ready identities for coefficient, assignment, and dyadic checks."""

    if report.status != "PROVED":
        raise ValueError("FI combinatorial replay must be proved before sealing")
    coefficient_differences = [
        right - left
        for case in report.cases
        for (_index, left), (_same_index, right) in zip(
            case.left,
            case.right,
            strict=True,
        )
    ]
    assignment_violations = [
        int(not case.terminal_partition_proved) for case in report.cases
    ]
    interval_width = sum(
        (interval.hi - interval.lo for interval in report.dyadic_intervals),
        Fraction(0),
    )
    identities = (
        (
            "FI Vaughan coefficient-vector replay",
            sum((value * value for value in coefficient_differences), 0),
            0,
        ),
        (
            "FI terminal-pair assignment replay",
            sum(assignment_violations),
            0,
        ),
        (
            "FI dyadic interval-width replay",
            interval_width,
            Fraction(report.s - 1) * report.y,
        ),
    )
    certificates: list[dict[str, Any]] = []
    for claim, computed_raw, expected_raw in identities:
        computed = _as_fraction(computed_raw)
        expected = _as_fraction(expected_raw)
        pc, qc = computed.numerator, computed.denominator
        pe, qe = expected.numerator, expected.denominator
        certificates.append(
            make_certificate(
                claim=claim,
                payload={
                    "type": "rational_identity",
                    "lhs_terms": [[qe, pc], [-qc, pe]],
                    "rhs": 0,
                },
                honesty=dict(report.honesty),
                meta={"scope": "finite_fi_combinatorial_replay"},
            )
        )
    return tuple(certificates)


def seal_fi_combinatorial_replay(
    report: FICombinatorialReplayCertificate,
) -> dict[str, Any]:
    """Seal the finite FI combinatorics without asserting an analytic estimate."""

    return make_certificate(
        claim=ASYMPTOTIC_SIEVE_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


@dataclass(frozen=True)
class RhoInsertionCertificate:
    """Finite replay of ``rho_n Lambda(n>Z) = Lambda(n>Z)`` on squarefree support."""

    n_max: int
    delta: Fraction
    z: Fraction
    lambdas: tuple[tuple[int, Fraction], ...]
    support_values: tuple[tuple[int, Fraction], ...]
    coefficient_guards_proved: bool
    finite_nonnegative: bool
    identity_proved: bool
    status: FiniteStatus

    def to_payload(self) -> dict[str, object]:
        return {
            "n_max": self.n_max,
            "delta": _fraction_payload(self.delta),
            "z": _fraction_payload(self.z),
            "lambdas": [
                [divisor, _fraction_payload(value)] for divisor, value in self.lambdas
            ],
            "support_values": [
                [n, _fraction_payload(value)] for n, value in self.support_values
            ],
            "coefficient_guards_proved": self.coefficient_guards_proved,
            "finite_nonnegative": self.finite_nonnegative,
            "identity_proved": self.identity_proved,
            "status": self.status,
        }


def _is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, isqrt(value) + 1)
    )


def certify_rho_insertion(
    *,
    n_max: int,
    delta: RationalInput,
    z: RationalInput,
    lambdas: Mapping[int, RationalInput],
) -> RhoInsertionCertificate:
    """Check FI's upper-sieve insertion on one finite squarefree support."""

    if n_max < 1:
        raise ValueError("n_max must be positive")
    exact_delta = _as_fraction(delta)
    exact_z = _as_fraction(z)
    if exact_delta < 1 or exact_z <= exact_delta:
        raise ValueError("rho insertion requires 1 <= delta < z")
    if any(divisor < 1 for divisor in lambdas):
        raise ValueError("sieve divisors must be positive")
    exact_lambdas = tuple(
        sorted((divisor, _as_fraction(value)) for divisor, value in lambdas.items())
    )
    coefficient_guards_proved = (
        dict(exact_lambdas).get(1) == 1
        and all(divisor <= exact_delta for divisor, _value in exact_lambdas)
        and all(abs(value) <= 1 for _divisor, value in exact_lambdas)
    )
    support_values: list[tuple[int, Fraction]] = []
    finite_nonnegative = True
    identity_proved = coefficient_guards_proved
    lambda_map = dict(exact_lambdas)
    for n in range(1, n_max + 1):
        if mobius_integer(n) == 0:
            continue
        rho = sum(
            (
                value
                for divisor, value in lambda_map.items()
                if n % divisor == 0
            ),
            Fraction(0),
        )
        finite_nonnegative = finite_nonnegative and rho >= 0
        if _is_prime(n) and n > exact_z:
            support_values.append((n, rho))
            identity_proved = identity_proved and rho == 1
    proved = coefficient_guards_proved and finite_nonnegative and identity_proved
    return RhoInsertionCertificate(
        n_max=n_max,
        delta=exact_delta,
        z=exact_z,
        lambdas=exact_lambdas,
        support_values=tuple(support_values),
        coefficient_guards_proved=coefficient_guards_proved,
        finite_nonnegative=finite_nonnegative,
        identity_proved=identity_proved,
        status="PROVED" if proved else "BLOCKED",
    )


FormalLogVector = tuple[tuple[int, int], ...]


def integer_log_vector(value: int) -> FormalLogVector:
    """Encode ``log(value)`` in the free basis ``{log(p): p prime}``."""

    return _factor_integer(value)


def _combine_log_vectors(
    terms: Iterable[tuple[int, FormalLogVector]],
) -> FormalLogVector:
    coefficients: dict[int, int] = {}
    for scalar, vector in terms:
        for prime, exponent in vector:
            coefficients[prime] = coefficients.get(prime, 0) + scalar * exponent
    return tuple(
        (prime, exponent)
        for prime, exponent in sorted(coefficients.items())
        if exponent
    )


def oriented_lambda_log_vector(value: int) -> FormalLogVector:
    """Replay ``Lambda(n)=sum_{qk=n} mu(q) log(k)`` coefficientwise."""

    if value < 1:
        raise ValueError("oriented Lambda identity requires a positive integer")
    return _combine_log_vectors(
        (
            mobius_integer(divisor),
            integer_log_vector(value // divisor),
        )
        for divisor in _divisors(value)
    )


def unsigned_oriented_lambda_log_vector(value: int) -> FormalLogVector:
    """Negative control obtained by erasing the essential Möbius signs."""

    if value < 1:
        raise ValueError("oriented Lambda identity requires a positive integer")
    return _combine_log_vectors(
        (
            abs(mobius_integer(divisor)),
            integer_log_vector(value // divisor),
        )
        for divisor in _divisors(value)
    )


def von_mangoldt_log_vector(value: int) -> FormalLogVector:
    """Encode ``Lambda(value)`` exactly in the formal logarithm basis."""

    if value < 1:
        raise ValueError("von Mangoldt identity requires a positive integer")
    factors = _factor_integer(value)
    return ((factors[0][0], 1),) if len(factors) == 1 else ()


@dataclass(frozen=True)
class FixedShiftDeterminantCase:
    """One coefficientwise fixed-shift signed determinant identity."""

    r: int
    s: int
    cutoff: int
    shift: int
    shifted_value: int
    left: FormalLogVector
    right: FormalLogVector
    q_one: FormalLogVector
    k_one_zero: bool
    power_of_two_exception: bool
    remaining_support_has_only_odd_divisors: bool
    proved: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "r": self.r,
            "s": self.s,
            "cutoff": self.cutoff,
            "shift": self.shift,
            "shifted_value": self.shifted_value,
            "left": [list(item) for item in self.left],
            "right": [list(item) for item in self.right],
            "q_one": [list(item) for item in self.q_one],
            "k_one_zero": self.k_one_zero,
            "power_of_two_exception": self.power_of_two_exception,
            "remaining_support_has_only_odd_divisors": (
                self.remaining_support_has_only_odd_divisors
            ),
            "proved": self.proved,
        }


def fixed_shift_determinant_case(
    *,
    r: int,
    s: int,
    cutoff: int,
    shift: int = 2,
) -> FixedShiftDeterminantCase:
    r"""Replay the signed ``rdt-qk=shift`` transform on squarefree support."""

    if min(r, s, cutoff, shift) < 1 or r * s <= shift:
        raise ValueError("require positive inputs with r*s greater than shift")
    if mobius_integer(r) == 0 or mobius_integer(s) == 0 or gcd(r, s) != 1:
        raise ValueError("determinant replay requires coprime squarefree r and s")
    shifted_value = r * s - shift
    oriented = oriented_lambda_log_vector(shifted_value)
    direct = von_mangoldt_log_vector(shifted_value)
    left_scalar = gamma_truncation(s, cutoff) * mobius_integer(r * s)
    left = _combine_log_vectors(((left_scalar, direct),))
    right = _combine_log_vectors(
        (
            mobius_integer(r) * mobius_integer(s // divisor),
            oriented,
        )
        for divisor in _divisors(s)
        if divisor <= cutoff
    )
    q_one = integer_log_vector(shifted_value)
    k_one_zero = integer_log_vector(1) == ()
    power_of_two_exception = bool(
        direct and direct[0][0] == 2
    )
    remaining_support_has_only_odd_divisors = (
        not direct
        or power_of_two_exception
        or all(divisor % 2 for divisor in _divisors(shifted_value))
    )
    proved = (
        oriented == direct
        and left == right
        and q_one == integer_log_vector(shifted_value)
        and k_one_zero
        and remaining_support_has_only_odd_divisors
    )
    return FixedShiftDeterminantCase(
        r=r,
        s=s,
        cutoff=cutoff,
        shift=shift,
        shifted_value=shifted_value,
        left=left,
        right=right,
        q_one=q_one,
        k_one_zero=k_one_zero,
        power_of_two_exception=power_of_two_exception,
        remaining_support_has_only_odd_divisors=(
            remaining_support_has_only_odd_divisors
        ),
        proved=proved,
    )


@dataclass(frozen=True)
class FixedShiftDeterminantCertificate:
    """Finite universal replay of the proposed fixed-shift signed transform."""

    r_max: int
    s_max: int
    cutoff: int
    shift: int
    cases: tuple[FixedShiftDeterminantCase, ...]
    oriented_lambda_proved: bool
    determinant_identity_proved: bool
    no_shift_average: bool
    no_termwise_absolute_value: bool
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": FIXED_SHIFT_DETERMINANT_KIND,
            "r_max": self.r_max,
            "s_max": self.s_max,
            "cutoff": self.cutoff,
            "shift": self.shift,
            "cases": [case.to_payload() for case in self.cases],
            "oriented_lambda_proved": self.oriented_lambda_proved,
            "determinant_identity_proved": self.determinant_identity_proved,
            "no_shift_average": self.no_shift_average,
            "no_termwise_absolute_value": self.no_termwise_absolute_value,
            "status": self.status,
            "external_premises": list(self.external_premises),
        }


def fixed_shift_determinant_honesty(
    *,
    finite_replay: bool,
) -> dict[str, bool]:
    """Separate exact signed algebra from the missing completion estimate."""

    return {
        "unproven_claim": False,
        "finite_fixed_shift_determinant_replay": finite_replay,
        "fixed_shift_2_completion_lemma_proved": False,
        "mobius_bilinear_estimate_proved": False,
        "shift_average_used": False,
        "termwise_absolute_value_used": False,
        "twin_prime_conjecture_proof_claim": False,
        "continuum_parent_inferred": False,
    }


def certify_fixed_shift_determinant_replay(
    *,
    r_max: int,
    s_max: int,
    cutoff: int,
    shift: int = 2,
    external_premises: Sequence[str] = FIXED_SHIFT_DETERMINANT_EXTERNAL_PREMISES,
) -> FixedShiftDeterminantCertificate:
    """Replay every eligible finite ``(r,s)`` determinant coefficient."""

    if min(r_max, s_max, cutoff, shift) < 1:
        raise ValueError("finite determinant bounds must be positive")
    premises = tuple(str(premise).strip() for premise in external_premises)
    if not premises or any(not premise for premise in premises):
        raise ValueError("determinant external premises must be explicit and non-empty")
    cases = tuple(
        fixed_shift_determinant_case(
            r=r,
            s=s,
            cutoff=cutoff,
            shift=shift,
        )
        for r in range(1, r_max + 1)
        for s in range(1, s_max + 1)
        if r * s > shift
        and mobius_integer(r) != 0
        and mobius_integer(s) != 0
        and gcd(r, s) == 1
    )
    oriented_lambda_proved = all(
        oriented_lambda_log_vector(value) == von_mangoldt_log_vector(value)
        for value in range(1, r_max * s_max - shift + 1)
    )
    determinant_identity_proved = bool(cases) and all(case.proved for case in cases)
    proved = oriented_lambda_proved and determinant_identity_proved
    honesty = fixed_shift_determinant_honesty(finite_replay=proved)
    return FixedShiftDeterminantCertificate(
        r_max=r_max,
        s_max=s_max,
        cutoff=cutoff,
        shift=shift,
        cases=cases,
        oriented_lambda_proved=oriented_lambda_proved,
        determinant_identity_proved=determinant_identity_proved,
        no_shift_average=True,
        no_termwise_absolute_value=True,
        status="PROVED" if proved else "BLOCKED",
        external_premises=premises,
        honesty=honesty,
    )


def seal_fixed_shift_determinant_replay(
    report: FixedShiftDeterminantCertificate,
) -> dict[str, Any]:
    """Seal finite determinant algebra without asserting cancellation."""

    return make_certificate(
        claim=FIXED_SHIFT_DETERMINANT_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


@dataclass(frozen=True)
class WrightRangeReport:
    """Exact exponent check for the two relevant Corollary 2.2 ranges."""

    short_exponent: Fraction
    modulus_exponent: Fraction
    epsilon: Fraction
    general_range: bool
    fixed_small_shift_range: bool
    fixed_shift_size_condition_external: bool

    @property
    def exponent_range_covered(self) -> bool:
        return self.general_range or self.fixed_small_shift_range

    def to_payload(self) -> dict[str, object]:
        return {
            "short_exponent": _fraction_payload(self.short_exponent),
            "modulus_exponent": _fraction_payload(self.modulus_exponent),
            "epsilon": _fraction_payload(self.epsilon),
            "general_range": self.general_range,
            "fixed_small_shift_range": self.fixed_small_shift_range,
            "fixed_shift_size_condition_external": (
                self.fixed_shift_size_condition_external
            ),
            "exponent_range_covered": self.exponent_range_covered,
        }


def wright_fixed_shift_range(
    *,
    short_exponent: RationalInput,
    modulus_exponent: RationalInput,
    epsilon: RationalInput,
) -> WrightRangeReport:
    """Check only the rational exponent inequalities in Wright Corollary 2.2."""

    b = _as_fraction(short_exponent)
    rho = _as_fraction(modulus_exponent)
    eps = _as_fraction(epsilon)
    if b < 0 or rho < 0 or eps <= 0:
        raise ValueError("range exponents must be nonnegative and epsilon positive")
    general_range = b <= Fraction(17, 28) - Fraction(33, 28) * rho - eps
    fixed_small_shift_range = (
        b <= Fraction(101, 630) - eps
        and rho <= Fraction(45, 89) - eps
    )
    return WrightRangeReport(
        short_exponent=b,
        modulus_exponent=rho,
        epsilon=eps,
        general_range=general_range,
        fixed_small_shift_range=fixed_small_shift_range,
        fixed_shift_size_condition_external=True,
    )


@dataclass(frozen=True)
class FixedShiftKernelCellCertificate:
    """Exact geometry and unspent saving for the proposed determinant cell."""

    terminal_sum_lower: Fraction
    v_lower: Fraction
    v_upper: Fraction
    d_lower: Fraction
    d_upper: Fraction
    kernel_saving_floor: Fraction
    target_saving: Fraction
    completion_loss_budget: Fraction
    geometry_proved: bool
    status: FiniteStatus
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "terminal_sum_lower": _fraction_payload(self.terminal_sum_lower),
            "v_lower": _fraction_payload(self.v_lower),
            "v_upper": _fraction_payload(self.v_upper),
            "d_lower": _fraction_payload(self.d_lower),
            "d_upper": _fraction_payload(self.d_upper),
            "kernel_saving_floor": _fraction_payload(self.kernel_saving_floor),
            "target_saving": _fraction_payload(self.target_saving),
            "completion_loss_budget": _fraction_payload(
                self.completion_loss_budget
            ),
            "geometry_proved": self.geometry_proved,
            "status": self.status,
        }


def certify_fixed_shift_kernel_cell(
    *,
    terminal_sum_lower: RationalInput = Fraction(63, 64),
    v_lower: RationalInput = Fraction(39, 100),
    v_upper: RationalInput = Fraction(2, 5),
    d_lower: RationalInput = Fraction(0),
    d_upper: RationalInput = Fraction(1, 4),
    target_saving: RationalInput = FIXED_SHIFT_TARGET_SAVING,
) -> FixedShiftKernelCellCertificate:
    """Certify kernel credit only; no completion theorem is inferred."""

    sum_lower = _as_fraction(terminal_sum_lower)
    exact_v_lower = _as_fraction(v_lower)
    exact_v_upper = _as_fraction(v_upper)
    exact_d_lower = _as_fraction(d_lower)
    exact_d_upper = _as_fraction(d_upper)
    exact_target = _as_fraction(target_saving)
    geometry_proved = (
        0 <= exact_v_lower <= exact_v_upper < Fraction(1, 2)
        and 0 <= exact_d_lower <= exact_d_upper <= Fraction(1, 4)
        and exact_d_upper <= exact_v_lower
        and exact_v_upper < sum_lower <= 1
        and exact_target > 0
    )
    kernel_saving_floor = (
        (sum_lower - exact_v_upper) / 6
        - exact_v_upper / 12
        + exact_d_lower / 4
    )
    completion_loss_budget = kernel_saving_floor - exact_target
    proved = geometry_proved and completion_loss_budget > 0
    honesty = {
        "unproven_claim": False,
        "finite_kernel_cell_geometry": proved,
        "dong_robles_zeindler_theorem_reproved": False,
        "fixed_shift_2_completion_lemma_proved": False,
        "mobius_bilinear_estimate_proved": False,
        "twin_prime_conjecture_proof_claim": False,
    }
    return FixedShiftKernelCellCertificate(
        terminal_sum_lower=sum_lower,
        v_lower=exact_v_lower,
        v_upper=exact_v_upper,
        d_lower=exact_d_lower,
        d_upper=exact_d_upper,
        kernel_saving_floor=kernel_saving_floor,
        target_saving=exact_target,
        completion_loss_budget=completion_loss_budget,
        geometry_proved=geometry_proved,
        status="PROVED" if proved else "BLOCKED",
        honesty=honesty,
    )


FIUseMechanism = Literal[
    "growth",
    "mobius_density",
    "r_prime",
    "b_prime",
    "upper_sieve",
]
FIGammaMode = Literal["none", "one", "truncated_mobius"]
FIShiftMode = Literal["none", "fixed_2", "averaged"]


@dataclass(frozen=True)
class FIAnalyticUse:
    """One declared leaf in the source-faithful FI analytic routing ledger."""

    term_id: str
    mechanism: FIUseMechanism
    source_ref: str
    gamma_mode: FIGammaMode = "none"
    shift_mode: FIShiftMode = "none"
    q_one_implies_twin_target: bool = False

    def __post_init__(self) -> None:
        if not self.term_id or not self.source_ref:
            raise ValueError("FI analytic use needs a term id and source reference")

    def to_payload(self) -> dict[str, object]:
        return {
            "term_id": self.term_id,
            "mechanism": self.mechanism,
            "source_ref": self.source_ref,
            "gamma_mode": self.gamma_mode,
            "shift_mode": self.shift_mode,
            "q_one_implies_twin_target": self.q_one_implies_twin_target,
        }


_FI_CANONICAL_ROUTES: tuple[FIAnalyticUse, ...] = (
    FIAnalyticUse("small_prime_tail", "growth", "FI (3.5)"),
    FIAnalyticUse("T.T1.main", "mobius_density", "FI (2.4), Section 4"),
    FIAnalyticUse("T.T1.remainder", "r_prime", "FI (4.5)"),
    FIAnalyticUse("T.T2.main", "mobius_density", "FI (2.4), Section 4"),
    FIAnalyticUse("T.T2.remainder", "r_prime", "FI (4.5)"),
    FIAnalyticUse("T_YZ.main", "mobius_density", "FI (1.8), (2.4), (5.1)"),
    FIAnalyticUse("T_YZ.remainder", "r_prime", "FI (5.1)"),
    FIAnalyticUse("S1.main", "upper_sieve", "FI (1.8), (1.9), (6.6)"),
    FIAnalyticUse("S1.remainder", "r_prime", "FI (6.3)"),
    FIAnalyticUse(
        "S2",
        "b_prime",
        "FI (7.1), (7.2)",
        gamma_mode="one",
        shift_mode="fixed_2",
    ),
    FIAnalyticUse(
        "S3.lambda_minus",
        "b_prime",
        "FI (8.1)",
        gamma_mode="truncated_mobius",
        shift_mode="fixed_2",
    ),
    FIAnalyticUse(
        "S3.lambda_plus.main",
        "mobius_density",
        "FI (2.4), (8.3)",
    ),
    FIAnalyticUse("S3.lambda_plus.remainder", "r_prime", "FI (8.4), (8.5)"),
)


def canonical_fi_term_ledger() -> tuple[FIAnalyticUse, ...]:
    """Return the source-faithful, single-owner FI analytic leaf inventory."""

    return _FI_CANONICAL_ROUTES


@dataclass(frozen=True)
class FITermLedgerReport:
    """Validation report for complete, non-circular FI term routing."""

    uses: tuple[FIAnalyticUse, ...]
    missing: tuple[str, ...]
    duplicates: tuple[str, ...]
    wrong_routes: tuple[str, ...]
    circular: tuple[str, ...]
    shift_mismatches: tuple[str, ...]
    complete: bool
    noncircular: bool
    status: FiniteStatus

    def to_payload(self) -> dict[str, object]:
        return {
            "uses": [use.to_payload() for use in self.uses],
            "missing": list(self.missing),
            "duplicates": list(self.duplicates),
            "wrong_routes": list(self.wrong_routes),
            "circular": list(self.circular),
            "shift_mismatches": list(self.shift_mismatches),
            "complete": self.complete,
            "noncircular": self.noncircular,
            "status": self.status,
        }


def validate_fi_term_ledger(
    uses: Sequence[FIAnalyticUse],
) -> FITermLedgerReport:
    """Reject missing, duplicate, misrouted, averaged-shift, or circular leaves."""

    declared = tuple(uses)
    expected = {use.term_id: use for use in _FI_CANONICAL_ROUTES}
    counts = {
        term_id: sum(use.term_id == term_id for use in declared)
        for term_id in expected
    }
    missing = tuple(term_id for term_id, count in counts.items() if count == 0)
    duplicates = tuple(term_id for term_id, count in counts.items() if count > 1)
    wrong_routes: list[str] = []
    shift_mismatches: list[str] = []
    circular: list[str] = []
    for use in declared:
        reference = expected.get(use.term_id)
        if reference is None:
            wrong_routes.append(use.term_id)
            continue
        if (
            use.mechanism != reference.mechanism
            or use.gamma_mode != reference.gamma_mode
            or use.source_ref != reference.source_ref
        ):
            wrong_routes.append(use.term_id)
        if use.shift_mode != reference.shift_mode or use.shift_mode == "averaged":
            shift_mismatches.append(use.term_id)
        if use.q_one_implies_twin_target:
            circular.append(use.term_id)
    complete = not missing and not duplicates and not wrong_routes
    noncircular = not circular and not shift_mismatches
    return FITermLedgerReport(
        uses=declared,
        missing=missing,
        duplicates=duplicates,
        wrong_routes=tuple(wrong_routes),
        circular=tuple(circular),
        shift_mismatches=tuple(shift_mismatches),
        complete=complete,
        noncircular=noncircular,
        status="PROVED" if complete and noncircular else "BLOCKED",
    )


@dataclass(frozen=True)
class AdmissibilityReport:
    """Finite residue-class certificate for an admissible shift tuple."""

    shifts: tuple[int, ...]
    admissible: bool
    missing_residues: tuple[tuple[int, tuple[int, ...]], ...]
    obstruction_prime: int | None

    def to_payload(self) -> dict[str, object]:
        return {
            "shifts": list(self.shifts),
            "admissible": self.admissible,
            "missing_residues": {
                str(prime): list(residues)
                for prime, residues in self.missing_residues
            },
            "obstruction_prime": self.obstruction_prime,
        }


def admissibility_report(shifts: Sequence[int]) -> AdmissibilityReport:
    """Check admissibility; primes larger than the tuple cardinality are automatic."""

    ordered = tuple(sorted(int(shift) for shift in shifts))
    if len(ordered) < 2:
        raise ValueError("an admissible sieve tuple requires at least two shifts")
    if len(set(ordered)) != len(ordered):
        raise ValueError("shifts must be distinct")
    missing: list[tuple[int, tuple[int, ...]]] = []
    obstruction: int | None = None
    for prime in _primes_up_to(len(ordered)):
        occupied = {shift % prime for shift in ordered}
        absent = tuple(residue for residue in range(prime) if residue not in occupied)
        missing.append((prime, absent))
        if not absent and obstruction is None:
            obstruction = prime
    return AdmissibilityReport(
        shifts=ordered,
        admissible=obstruction is None,
        missing_residues=tuple(missing),
        obstruction_prime=obstruction,
    )


MINIMAL_ADMISSIBLE_DIAMETER_KIND = "minimal_admissible_diameter"
# OEIS A008407 / Engelsma diameters H(k) for 2 <= k <= 56. Entries with
# k <= PROVED_MINIMAL_ADMISSIBLE_CARDINALITY are independently re-proved by
# residue-pattern exhaustion. Larger entries are a named lookup, not a
# minimality theorem, and do not license a narrower k=39 target than 182.
PUBLISHED_ADMISSIBLE_DIAMETERS: tuple[int, ...] = (
    0,
    0,
    2,
    6,
    8,
    12,
    16,
    20,
    26,
    30,
    32,
    36,
    42,
    48,
    50,
    56,
    60,
    66,
    70,
    76,
    80,
    84,
    90,
    94,
    100,
    110,
    114,
    120,
    126,
    130,
    136,
    140,
    146,
    152,
    156,
    158,
    162,
    168,
    176,
    182,
    186,
    188,
    196,
    200,
    210,
    212,
    216,
    226,
    236,
    240,
    246,
    252,
    254,
    264,
    270,
    272,
    278,
)
PROVED_MINIMAL_ADMISSIBLE_CARDINALITY = 12


def published_admissible_diameter(k: int) -> int:
    """Return the named published H(k) lookup; this is not a minimality proof."""

    if k < 2 or k >= len(PUBLISHED_ADMISSIBLE_DIAMETERS):
        raise ValueError("published admissible diameters cover 2 <= k <= 56")
    return PUBLISHED_ADMISSIBLE_DIAMETERS[k]


def scalar_gap_cardinality_ladder(
    *,
    start: int = 40,
    stop: int = 36,
) -> tuple[tuple[int, int], ...]:
    """Published ``(k, H(k))`` descending from the named ``k=40`` baseline."""

    if start < 2 or stop < 2 or start >= len(PUBLISHED_ADMISSIBLE_DIAMETERS):
        raise ValueError("ladder cardinalities must lie in the published table")
    if stop > start:
        raise ValueError("ladder stop must not exceed start")
    return tuple(
        (cardinality, published_admissible_diameter(cardinality))
        for cardinality in range(start, stop - 1, -1)
    )


def _largest_survivor_set(
    diameter: int,
    primes: tuple[int, ...],
) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    assignments = 1
    for prime in primes:
        assignments *= prime
    for code in range(assignments):
        forbidden: list[int] = []
        rest = code
        for prime in primes:
            forbidden.append(rest % prime)
            rest //= prime
        survivors = tuple(
            point
            for point in range(diameter + 1)
            if all(
                point % prime != residue
                for prime, residue in zip(primes, forbidden, strict=True)
            )
        )
        if len(survivors) > len(best):
            best = survivors
            if len(best) == diameter + 1:
                break
    return best


def _witness_from_survivors(
    survivors: tuple[int, ...],
    cardinality: int,
) -> tuple[int, ...]:
    if len(survivors) < cardinality:
        raise ArithmeticError("survivor set is smaller than the requested cardinality")
    origin = survivors[0]
    shifted = tuple(point - origin for point in survivors)
    chosen = (shifted[0], *shifted[1 : cardinality - 1], shifted[-1])
    if len(set(chosen)) != cardinality:
        raise ArithmeticError("witness collapsed duplicate shifts")
    return chosen


@dataclass(frozen=True)
class MinimalAdmissibleDiameterCertificate:
    """Finite residue-pattern exhaustion, or a named published lookup."""

    cardinality: int
    diameter: int
    witness: tuple[int, ...]
    published_diameter: int | None
    status: FiniteStatus
    honesty: Mapping[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": MINIMAL_ADMISSIBLE_DIAMETER_KIND,
            "cardinality": self.cardinality,
            "diameter": self.diameter,
            "witness": list(self.witness),
            "published_diameter": self.published_diameter,
            "status": self.status,
        }


def minimal_admissible_diameter_honesty(
    *,
    finite_exhaustion: bool,
    published_lookup: bool,
) -> dict[str, bool]:
    """Keep a finite H(k) proof separate from any gap or twin-prime claim."""

    return {
        "unproven_claim": False,
        "finite_residue_exhaustion_proved": finite_exhaustion,
        "published_diameter_lookup": published_lookup,
        "narrower_than_182_k39_possible": False,
        "dhl_claim": False,
        "h1_182_claim": False,
        "twin_prime_conjecture_proof_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def prove_minimal_admissible_diameter(
    k: int,
    *,
    search_limit: int = PROVED_MINIMAL_ADMISSIBLE_CARDINALITY,
) -> MinimalAdmissibleDiameterCertificate:
    """Prove H(k) by residue-pattern exhaustion, or refuse above the search limit."""

    if k < 2:
        raise ValueError("an admissible tuple requires at least two shifts")
    if search_limit < 2:
        raise ValueError("search_limit must be at least 2")
    published = (
        published_admissible_diameter(k)
        if k < len(PUBLISHED_ADMISSIBLE_DIAMETERS)
        else None
    )
    if k > search_limit:
        return MinimalAdmissibleDiameterCertificate(
            cardinality=k,
            diameter=published if published is not None else 0,
            witness=(),
            published_diameter=published,
            status="BLOCKED",
            honesty=minimal_admissible_diameter_honesty(
                finite_exhaustion=False,
                published_lookup=published is not None,
            ),
        )
    primes = _primes_up_to(k)
    diameter = k - 1
    survivors = _largest_survivor_set(diameter, primes)
    while len(survivors) < k:
        diameter += 1
        if diameter > 6 * k:
            raise ArithmeticError("admissible-diameter search exceeded its budget")
        survivors = _largest_survivor_set(diameter, primes)
    witness = _witness_from_survivors(survivors, k)
    report = admissibility_report(witness)
    if not report.admissible or witness[-1] - witness[0] != diameter:
        raise ArithmeticError("residue exhaustion produced an invalid witness")
    if published is not None and diameter != published:
        raise ArithmeticError(
            "proved H(k) disagrees with the published diameter table"
        )
    return MinimalAdmissibleDiameterCertificate(
        cardinality=k,
        diameter=diameter,
        witness=witness,
        published_diameter=published,
        status="PROVED",
        honesty=minimal_admissible_diameter_honesty(
            finite_exhaustion=True,
            published_lookup=published is not None,
        ),
    )


def maximal_far_sets(
    shifts: Sequence[int],
    max_gap: int,
    *,
    search_budget: int = 100_000,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate maximal index sets with every pair farther apart than ``max_gap``."""

    ordered = tuple(int(shift) for shift in shifts)
    if tuple(sorted(ordered)) != ordered or len(set(ordered)) != len(ordered):
        raise ValueError("shifts must be strictly increasing")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if search_budget < 1:
        raise ValueError("search_budget must be positive")
    far_sets: list[tuple[int, ...]] = [()]
    for index, shift in enumerate(ordered):
        additions = [
            chosen + (index,)
            for chosen in far_sets
            if all(abs(shift - ordered[prior]) > max_gap for prior in chosen)
        ]
        far_sets.extend(additions)
        if len(far_sets) > search_budget:
            raise ValueError(
                "far-set enumeration exceeded search_budget; "
                "tighten the host tuple or raise the explicit budget"
            )
    maximal = [
        chosen
        for chosen in far_sets
        if chosen
        and not any(
            index not in chosen
            and all(abs(ordered[index] - ordered[prior]) > max_gap for prior in chosen)
            for index in range(len(ordered))
        )
    ]
    return tuple(sorted(maximal))


@dataclass(frozen=True)
class MatrixClosePairCertificate:
    """Finite certificate for the matrix score of every no-close-pair set."""

    shifts: tuple[int, ...]
    max_gap: int
    scoring_matrices: tuple[RationalMatrix, ...]
    admissibility: AdmissibilityReport
    matrix_reports: tuple[PSDReport, ...]
    maximal_far_sets: tuple[tuple[int, ...], ...]
    slack_reports: tuple[PSDReport, ...]
    proved: bool
    detail: str

    @property
    def dimension(self) -> int:
        return len(self.scoring_matrices[0])

    def to_payload(self) -> dict[str, object]:
        return {
            "shifts": list(self.shifts),
            "max_gap": self.max_gap,
            "dimension": self.dimension,
            "scoring_matrices": [_matrix_payload(matrix) for matrix in self.scoring_matrices],
            "admissibility": self.admissibility.to_payload(),
            "matrix_reports": [report.to_payload() for report in self.matrix_reports],
            "maximal_far_sets": [list(indices) for indices in self.maximal_far_sets],
            "slack_reports": [report.to_payload() for report in self.slack_reports],
            "proved": self.proved,
            "detail": self.detail,
        }


def certify_matrix_close_pair(
    shifts: Sequence[int],
    max_gap: int,
    scoring_matrices: Sequence[Sequence[Sequence[RationalInput]]],
    *,
    search_budget: int = 100_000,
) -> MatrixClosePairCertificate:
    """Check admissibility, ``Q_i >= 0``, and every maximal far-set constraint."""

    raw_shifts = tuple(int(shift) for shift in shifts)
    if raw_shifts != tuple(sorted(raw_shifts)):
        raise ValueError("shifts must be strictly increasing to preserve matrix alignment")
    admissibility = admissibility_report(shifts)
    ordered = admissibility.shifts
    matrices = tuple(_as_matrix(matrix) for matrix in scoring_matrices)
    if len(matrices) != len(ordered):
        raise ValueError("one scoring matrix is required per shift")
    dimension = len(matrices[0])
    if any(len(matrix) != dimension for matrix in matrices):
        raise ValueError("all scoring matrices must have the same dimension")
    far_sets = maximal_far_sets(ordered, max_gap, search_budget=search_budget)
    matrix_reports = tuple(psd_report(matrix) for matrix in matrices)
    identity = _identity(dimension)
    slack_reports: list[PSDReport] = []
    for indices in far_sets:
        total = _matrix_sum(tuple(matrices[index] for index in indices), dimension)
        slack_reports.append(psd_report(_matrix_sub(identity, total)))
    proved = (
        admissibility.admissible
        and all(report.psd for report in matrix_reports)
        and all(report.psd for report in slack_reports)
    )
    if not admissibility.admissible:
        detail = f"shift tuple is inadmissible modulo {admissibility.obstruction_prime}"
    elif not all(report.psd for report in matrix_reports):
        detail = "at least one scoring matrix is not positive semidefinite"
    elif not all(report.psd for report in slack_reports):
        detail = "a maximal no-close-pair set has matrix score exceeding the identity"
    else:
        detail = (
            "every matrix and maximal no-close-pair slack is positive semidefinite "
            "over the rationals"
        )
    return MatrixClosePairCertificate(
        shifts=ordered,
        max_gap=int(max_gap),
        scoring_matrices=matrices,
        admissibility=admissibility,
        matrix_reports=matrix_reports,
        maximal_far_sets=far_sets,
        slack_reports=tuple(slack_reports),
        proved=proved,
        detail=detail,
    )


def matrix_trace_product(left: RationalMatrix, right: RationalMatrix) -> Fraction:
    """Return ``trace(left @ right)`` exactly."""

    if len(left) != len(right):
        raise ValueError("matrix dimensions must match")
    n = len(left)
    return sum(
        (left[i][j] * right[j][i] for i in range(n) for j in range(n)),
        Fraction(0),
    )


@dataclass(frozen=True)
class MatrixVariationalCertificate:
    """Exact finite crossing, conditional on separately listed analytic premises."""

    close_pair: MatrixClosePairCertificate
    face_grams: tuple[RationalMatrix, ...]
    face_reports: tuple[PSDReport, ...]
    mass: Fraction
    score: Fraction
    surplus: Fraction
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    @property
    def finite_crossing(self) -> bool:
        return self.status == "PROVED"

    def to_payload(self) -> dict[str, object]:
        return {
            "type": MATRIX_CLOSE_PAIR_KIND,
            "status": self.status,
            "close_pair": self.close_pair.to_payload(),
            "face_grams": [_matrix_payload(matrix) for matrix in self.face_grams],
            "face_reports": [report.to_payload() for report in self.face_reports],
            "mass": _fraction_payload(self.mass),
            "score": _fraction_payload(self.score),
            "surplus": _fraction_payload(self.surplus),
            "external_premises": list(self.external_premises),
        }


def prime_gap_honesty(
    *,
    finite_crossing: bool,
    finite_admissibility: bool = False,
    finite_matrix_inequality: bool = False,
) -> dict[str, bool]:
    """Honesty payload; no finite witness may assert the infinite parent."""

    return {
        "unproven_claim": False,
        "finite_admissibility_check": finite_admissibility,
        "finite_matrix_inequality_check": finite_matrix_inequality,
        "exact_rational_variational_check": finite_crossing,
        "analytic_sieve_asymptotics_proved": False,
        "prime_distribution_hypotheses_discharged": False,
        "twin_prime_conjecture_proof_claim": False,
        "hardy_littlewood_asymptotic_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def certify_matrix_variational_witness(
    close_pair: MatrixClosePairCertificate,
    face_grams: Sequence[Sequence[Sequence[RationalInput]]],
    mass: RationalInput,
    *,
    external_premises: Sequence[str] = TWIN_PRIME_EXTERNAL_PREMISES,
) -> MatrixVariationalCertificate:
    """Check an exact matrix-weighted face score against an exact positive mass."""

    grams = tuple(_as_matrix(matrix) for matrix in face_grams)
    if len(grams) != len(close_pair.shifts):
        raise ValueError("one face Gram matrix is required per shift")
    if any(len(matrix) != close_pair.dimension for matrix in grams):
        raise ValueError("face Gram and scoring-matrix dimensions must match")
    exact_mass = _as_fraction(mass)
    if exact_mass <= 0:
        raise ValueError("mass must be positive")
    face_reports = tuple(psd_report(matrix) for matrix in grams)
    score = sum(
        (
            matrix_trace_product(scoring, gram)
            for scoring, gram in zip(
                close_pair.scoring_matrices,
                grams,
                strict=True,
            )
        ),
        Fraction(0),
    )
    surplus = score - exact_mass
    finite_crossing = (
        close_pair.proved
        and all(report.psd for report in face_reports)
        and surplus > 0
    )
    status: FiniteStatus = "PROVED" if finite_crossing else "BLOCKED"
    premises = tuple(str(premise).strip() for premise in external_premises)
    if not premises or any(not premise for premise in premises):
        raise ValueError("analytic external_premises must be explicit and non-empty")
    return MatrixVariationalCertificate(
        close_pair=close_pair,
        face_grams=grams,
        face_reports=face_reports,
        mass=exact_mass,
        score=score,
        surplus=surplus,
        status=status,
        external_premises=premises,
        honesty=prime_gap_honesty(
            finite_crossing=finite_crossing,
            finite_admissibility=close_pair.admissibility.admissible,
            finite_matrix_inequality=(
                all(report.psd for report in close_pair.matrix_reports)
                and all(report.psd for report in close_pair.slack_reports)
            ),
        ),
    )


def seal_matrix_variational_certificate(
    report: MatrixVariationalCertificate,
) -> dict[str, Any]:
    """Seal the finite result; the certificate cannot discharge its external premises."""

    return make_certificate(
        claim=MATRIX_CLOSE_PAIR_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


@dataclass(frozen=True)
class BoundedGapTargetSpec:
    """Exact tuple and quotient threshold for a finite bounded-gap target."""

    name: str
    shifts: tuple[int, ...]
    gap: int
    rho_star: Fraction
    required_margin: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_star", _as_fraction(self.rho_star))
        object.__setattr__(
            self,
            "required_margin",
            _as_fraction(self.required_margin),
        )
        if not self.name:
            raise ValueError("bounded-gap target name must be non-empty")
        if len(self.shifts) < 2 or tuple(sorted(set(self.shifts))) != self.shifts:
            raise ValueError("bounded-gap shifts must be distinct and strictly increasing")
        if self.gap != self.shifts[-1] - self.shifts[0]:
            raise ValueError("target gap must equal the tuple diameter")
        if self.rho_star <= 0 or self.required_margin < 0:
            raise ValueError("rho_star must be positive and required_margin nonnegative")

    @property
    def k(self) -> int:
        return len(self.shifts)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "k": self.k,
            "shifts": list(self.shifts),
            "gap": self.gap,
            "rho_star": _fraction_payload(self.rho_star),
            "required_margin": _fraction_payload(self.required_margin),
        }


@dataclass(frozen=True)
class BoundedGapBaseline:
    """Named published finite-certificate baseline for comparison."""

    name: str
    target: BoundedGapTargetSpec
    basis_size: int
    grid_size: int
    source_components: int
    raw_source_forms: int
    reported_margin: Fraction

    @property
    def passed_reported_threshold(self) -> bool:
        return self.reported_margin > self.target.required_margin

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target": self.target.to_payload(),
            "basis_size": self.basis_size,
            "grid_size": self.grid_size,
            "source_components": self.source_components,
            "raw_source_forms": self.raw_source_forms,
            "reported_margin": _fraction_payload(self.reported_margin),
            "passed_reported_threshold": self.passed_reported_threshold,
        }


def h1_186_baseline() -> BoundedGapBaseline:
    """Return the exact public PrimeGaps186 finite-certificate metadata."""

    return BoundedGapBaseline(
        name="OpenAI Improved Short Gaps Between Primes",
        target=BoundedGapTargetSpec(
            name="reported_H1_186_DHL_40_2",
            shifts=H1_186_SHIFTS,
            gap=186,
            rho_star=H1_186_RHO_STAR,
            required_margin=H1_186_REQUIRED_MARGIN,
        ),
        basis_size=77,
        grid_size=98304,
        source_components=97,
        raw_source_forms=149,
        reported_margin=H1_186_PUBLISHED_MARGIN,
    )


def h1_182_target() -> BoundedGapTargetSpec:
    """Return the exact next scalar-sieve target: the diameter-182 39-tuple."""

    return BoundedGapTargetSpec(
        name="target_H1_182_DHL_39_2",
        shifts=H1_182_SHIFTS,
        gap=182,
        rho_star=H1_186_RHO_STAR,
        required_margin=H1_186_REQUIRED_MARGIN,
    )


def _as_vector(raw: Sequence[RationalInput]) -> tuple[Fraction, ...]:
    vector = tuple(_as_fraction(value) for value in raw)
    if not vector:
        raise ValueError("coefficient vector must be non-empty")
    if all(value == 0 for value in vector):
        raise ValueError("coefficient vector must be nonzero")
    return vector


def _quadratic_form(
    matrix: RationalMatrix,
    vector: tuple[Fraction, ...],
) -> Fraction:
    if len(matrix) != len(vector):
        raise ValueError("matrix and coefficient dimensions must match")
    return sum(
        (
            vector[i] * matrix[i][j] * vector[j]
            for i in range(len(vector))
            for j in range(len(vector))
        ),
        Fraction(0),
    )


@dataclass(frozen=True)
class BoundedGapQuadraticCertificate:
    """Exact generalized-Rayleigh crossing for declared rational forms."""

    target: BoundedGapTargetSpec
    admissibility: AdmissibilityReport
    numerator_matrix: RationalMatrix
    denominator_matrix: RationalMatrix
    denominator_report: PSDReport
    coefficients: tuple[Fraction, ...]
    numerator: Fraction
    denominator: Fraction
    quotient_minus_one: Fraction
    threshold_surplus: Fraction
    status: FiniteStatus
    external_premises: tuple[str, ...]
    honesty: Mapping[str, bool]

    @property
    def finite_crossing(self) -> bool:
        return self.status == "PROVED"

    def to_payload(self) -> dict[str, object]:
        return {
            "type": BOUNDED_GAP_QUADRATIC_KIND,
            "target": self.target.to_payload(),
            "admissibility": self.admissibility.to_payload(),
            "numerator_matrix": _matrix_payload(self.numerator_matrix),
            "denominator_matrix": _matrix_payload(self.denominator_matrix),
            "denominator_report": self.denominator_report.to_payload(),
            "coefficients": [
                _fraction_payload(value) for value in self.coefficients
            ],
            "numerator": _fraction_payload(self.numerator),
            "denominator": _fraction_payload(self.denominator),
            "quotient_minus_one": _fraction_payload(self.quotient_minus_one),
            "threshold_surplus": _fraction_payload(self.threshold_surplus),
            "status": self.status,
            "external_premises": list(self.external_premises),
        }


def bounded_gap_quadratic_honesty(
    *,
    finite_crossing: bool,
    finite_admissibility: bool,
) -> dict[str, bool]:
    """Keep a rational form crossing separate from DHL and gap conclusions."""

    return {
        "unproven_claim": False,
        "finite_admissibility_check": finite_admissibility,
        "exact_rational_quadratic_check": finite_crossing,
        "source_forms_outward_certified": False,
        "support_inequalities_discharged": False,
        "bounded_gap_analytic_instantiation_proved": False,
        "dhl_39_2_claim": False,
        "h1_182_claim": False,
        "twin_prime_conjecture_proof_claim": False,
        "float_residual_is_proof": False,
        "continuum_parent_inferred": False,
    }


def certify_bounded_gap_quadratic_witness(
    *,
    target: BoundedGapTargetSpec,
    numerator_matrix: Sequence[Sequence[RationalInput]],
    denominator_matrix: Sequence[Sequence[RationalInput]],
    coefficients: Sequence[RationalInput],
    external_premises: Sequence[str] = H1_182_EXTERNAL_PREMISES,
) -> BoundedGapQuadraticCertificate:
    """Certify one exact rational quotient; source validity remains external."""

    numerator_form = _as_matrix(numerator_matrix)
    denominator_form = _as_matrix(denominator_matrix)
    vector = _as_vector(coefficients)
    if len(numerator_form) != len(denominator_form):
        raise ValueError("numerator and denominator dimensions must match")
    if len(vector) != len(numerator_form):
        raise ValueError("coefficient dimension must match the quadratic forms")
    premises = tuple(str(premise).strip() for premise in external_premises)
    if not premises or any(not premise for premise in premises):
        raise ValueError("bounded-gap external premises must be explicit and non-empty")
    admissibility = admissibility_report(target.shifts)
    denominator_report = psd_report(denominator_form)
    numerator = _quadratic_form(numerator_form, vector)
    denominator = _quadratic_form(denominator_form, vector)
    quotient_minus_one = (
        target.rho_star * numerator / denominator - 1
        if denominator > 0
        else Fraction(0)
    )
    threshold_surplus = (
        target.rho_star * numerator
        - (1 + target.required_margin) * denominator
    )
    finite_crossing = (
        admissibility.admissible
        and denominator_report.psd
        and denominator > 0
        and threshold_surplus > 0
    )
    honesty = bounded_gap_quadratic_honesty(
        finite_crossing=finite_crossing,
        finite_admissibility=admissibility.admissible,
    )
    return BoundedGapQuadraticCertificate(
        target=target,
        admissibility=admissibility,
        numerator_matrix=numerator_form,
        denominator_matrix=denominator_form,
        denominator_report=denominator_report,
        coefficients=vector,
        numerator=numerator,
        denominator=denominator,
        quotient_minus_one=quotient_minus_one,
        threshold_surplus=threshold_surplus,
        status="PROVED" if finite_crossing else "BLOCKED",
        external_premises=premises,
        honesty=honesty,
    )


def seal_bounded_gap_quadratic_witness(
    report: BoundedGapQuadraticCertificate,
) -> dict[str, Any]:
    """Seal exact quotient arithmetic without asserting source validity."""

    return make_certificate(
        claim=BOUNDED_GAP_QUADRATIC_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
    )


def _fraction_payload(value: Fraction) -> list[str]:
    return [str(value.numerator), str(value.denominator)]


def _product(values: Iterable[Fraction]) -> Fraction:
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def _matrix_payload(matrix: RationalMatrix) -> list[list[list[str]]]:
    return [
        [_fraction_payload(value) for value in row]
        for row in matrix
    ]


__all__ = [
    "ASYMPTOTIC_SIEVE_EXTERNAL_PREMISES",
    "ASYMPTOTIC_SIEVE_KIND",
    "AdmissibilityReport",
    "BOUNDED_GAP_QUADRATIC_KIND",
    "BoundedGapBaseline",
    "BoundedGapQuadraticCertificate",
    "BoundedGapTargetSpec",
    "FIAnalyticUse",
    "FIAtlasParameters",
    "FICombinatorialReplayCertificate",
    "FITermLedgerReport",
    "FITerminalAtlasCertificate",
    "FIXED_SHIFT_DETERMINANT_EXTERNAL_PREMISES",
    "FIXED_SHIFT_DETERMINANT_KIND",
    "FIXED_SHIFT_KERNEL_SAVING_FLOOR",
    "FIXED_SHIFT_TARGET_SAVING",
    "FI_COMBINATORIAL_EXTERNAL_PREMISES",
    "FI_TERMINAL_ATLAS_EXTERNAL_PREMISES",
    "FiniteStatus",
    "FixedShiftDeterminantCase",
    "FixedShiftDeterminantCertificate",
    "FixedShiftKernelCellCertificate",
    "FormalLogVector",
    "H1_182_EXTERNAL_PREMISES",
    "H1_182_SHIFTS",
    "H1_186_PUBLISHED_MARGIN",
    "H1_186_REQUIRED_MARGIN",
    "H1_186_RHO_STAR",
    "H1_186_SHIFTS",
    "HEATH_BROWN_SMALL_FACTOR_ROOM",
    "KnownEstimate",
    "KnownEstimateKind",
    "LocalFactorIdentity",
    "LocalProductCertificate",
    "LogAffineBound",
    "LogPolyhedralCell",
    "MATRIX_CLOSE_PAIR_KIND",
    "MatrixClosePairCertificate",
    "MatrixVariationalCertificate",
    "MobiusCellReductionCertificate",
    "OpenClosedInterval",
    "PSDReport",
    "RationalCell",
    "RationalInput",
    "RationalMatrix",
    "RhoInsertionCertificate",
    "TWIN_PRIME_EXTERNAL_PREMISES",
    "VaughanCoefficientCase",
    "WrightRangeReport",
    "admissibility_report",
    "asymptotic_sieve_honesty",
    "asymptotic_sieve_local_factor",
    "bounded_gap_quadratic_honesty",
    "canonical_fi_term_ledger",
    "certify_asymptotic_sieve_local_product",
    "certify_bounded_gap_quadratic_witness",
    "certify_fi_combinatorial_replay",
    "certify_fi_terminal_atlas",
    "certify_fixed_shift_determinant_replay",
    "certify_fixed_shift_kernel_cell",
    "certify_matrix_close_pair",
    "certify_matrix_variational_witness",
    "certify_mobius_cell_reduction",
    "certify_rho_insertion",
    "fi_atlas_obligation_certificates",
    "fi_combinatorial_honesty",
    "fi_combinatorial_obligation_certificates",
    "fi_rational_terminal_family",
    "fi_terminal_atlas_honesty",
    "fi_vaughan_coefficient_case",
    "fixed_shift_determinant_case",
    "fixed_shift_determinant_honesty",
    "gamma_truncation",
    "h1_182_target",
    "h1_186_baseline",
    "integer_log_vector",
    "local_factor_obligation_certificates",
    "matrix_trace_product",
    "maximal_far_sets",
    "mobius_integer",
    "oriented_lambda_log_vector",
    "prime_gap_honesty",
    "psd_report",
    "seal_asymptotic_sieve_local_product",
    "seal_bounded_gap_quadratic_witness",
    "seal_fi_combinatorial_replay",
    "seal_fi_terminal_atlas",
    "seal_fixed_shift_determinant_replay",
    "seal_matrix_variational_certificate",
    "seal_mobius_cell_reduction",
    "select_fi_dyadic_s",
    "unsigned_oriented_lambda_log_vector",
    "validate_fi_term_ledger",
    "verify_dyadic_membership",
    "verify_gamma_mobius_identity",
    "von_mangoldt_log_vector",
    "wright_fixed_shift_range",
]

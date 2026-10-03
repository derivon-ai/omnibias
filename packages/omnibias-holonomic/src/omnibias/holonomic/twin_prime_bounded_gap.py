# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite replay tools for the scalar bounded-gap sieve track.

This module parameterizes the pure rational input-generation portion of the
pinned PrimeGaps186 evaluator and independently checks completed numerical and
quadratic receipts.  A valid manifest or structurally valid matrix receipt is
not by itself a numerical crossing and does not establish a bounded-prime-gap
theorem.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Literal

from omnibias.core.proof.certificate import make_certificate

RationalInput = Fraction | int | str
SourceRole = Literal["outer", "old_inner", "new_inner"]
TaskKind = Literal["low", "rank_two", "high"]

PRIME_GAPS_186_COMMIT = "61340d0b74163003b32756bb16e91d9209a5e330"
PRIME_GAPS_186_SOURCE_URL = (
    "https://raw.githubusercontent.com/openai/PrimeGaps186/"
    f"{PRIME_GAPS_186_COMMIT}/prime_gap_186_certificate.py"
)
PRIME_GAPS_186_SOURCE_SHA256 = (
    "7f71bdefcfe3bb5ca76a143929b3cb3f4156c21dc483253cda3077420f1e5de4"
)
PRIME_GAPS_INPUT_MANIFEST_KIND = "prime_gaps_input_manifest_replay"
PRIME_GAPS_NUMERICAL_RECEIPT_KIND = "prime_gaps_numerical_receipt_replay"
PRIME_GAPS_QUADRATIC_RECEIPT_KIND = "prime_gaps_quadratic_receipt_replay"
PRIME_GAP_QUADRATIC_MATRIX_NAMES = (
    "denominator",
    "J0",
    "Jplus",
    "Jtail",
    "source_loss",
)
PRIME_GAP_COEFFICIENT_SIGNATURES = (
    (),
    (2,),
    (3,),
    (4,),
    (5,),
    (6,),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 2, 2),
)

_YOUNG_Q: tuple[tuple[int, ...], ...] = (
    (
        961904,
        502424,
        483341,
        547373,
        563915,
        583181,
        604629,
        620671,
        629321,
        635211,
        616326,
        593862,
        573977,
        553178,
        531463,
        508862,
        459016,
    ),
    (
        7266522,
        1241454497,
        1208324400,
        1152630107,
        1126190783,
        1096246679,
        1058983690,
        967816560,
        867471653,
        603785822,
        32188902,
        1308239,
        386321,
        373849,
        377891,
        385136,
        395013,
        405835,
        419505,
        432001,
        445139,
        457321,
        525975,
        518733,
        515168,
        512357,
        509770,
        507320,
        504951,
        502604,
        503256,
        498048,
        492222,
        485810,
        433769,
    ),
)


def _fraction(value: RationalInput) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("expected an exact rational, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int | str):
        return Fraction(value)
    raise TypeError("expected Fraction, int, or rational string")


def _fraction_payload(value: Fraction) -> list[str]:
    return [str(value.numerator), str(value.denominator)]


@dataclass(frozen=True)
class PrimeGapLadderRow:
    ladder: str
    index: int
    source_order: int
    previous: Fraction
    omega: Fraction
    delta: Fraction
    upper_b: Fraction
    xi: Fraction
    a: Fraction
    b: Fraction
    outer_core: Fraction
    inner_core: Fraction
    owner_model: str
    owner_plateau: Fraction | None


def _ladder(
    *,
    name: str,
    inner_radius: Fraction,
    epsilon: Fraction,
    limit: Fraction,
    constants: tuple[tuple[Fraction, Fraction], ...],
    outer_radius: Fraction,
    rho: Fraction,
    gap: Fraction,
) -> tuple[PrimeGapLadderRow, ...]:
    previous = Fraction(0)
    exponent_offset = rho * (outer_radius + inner_radius) - Fraction(1, 2)
    rows: list[PrimeGapLadderRow] = []
    for index in range(100):
        order = min(index // 12 + 1, 3)
        constant, slope = constants[order - 1]
        omega = min(
            limit,
            (constant - epsilon - exponent_offset + 2 * previous - gap) / slope,
        )
        delta = constant - slope * omega - epsilon
        upper_b = (Fraction(1, 2) + 2 * previous) / rho
        xi = delta / rho
        a = upper_b - inner_radius
        b = upper_b - outer_radius
        eta = (
            xi
            if order < 3
            else (xi + outer_radius + inner_radius - upper_b) / 2
        )
        if not previous < omega <= limit or delta <= 0:
            raise ArithmeticError("source ladder lost its strict rational guards")
        rows.append(
            PrimeGapLadderRow(
                ladder=name,
                index=index,
                source_order=order,
                previous=previous,
                omega=omega,
                delta=delta,
                upper_b=upper_b,
                xi=xi,
                a=a,
                b=b,
                outer_core=a + eta,
                inner_core=b + eta,
                owner_model="linear" if order < 3 else "capped_outer_linear",
                owner_plateau=None if order < 3 else 23 * (b + eta) / 40,
            )
        )
        if omega == limit:
            return tuple(rows)
        previous = omega
    raise ArithmeticError("source ladder did not reach its prescribed limit")


@dataclass(frozen=True)
class PrimeGapShell:
    lower: Fraction
    upper: Fraction
    ceiling: Fraction


def _shells(
    pairs: Iterable[tuple[Fraction, Fraction]],
) -> tuple[PrimeGapShell, ...]:
    lower = Fraction(0)
    result: list[PrimeGapShell] = []
    for upper, ceiling in pairs:
        result.append(PrimeGapShell(lower, upper, ceiling))
        lower = upper
    return tuple(result)


@dataclass(frozen=True)
class PrimeGapCell:
    shell_index: int
    first_index: int
    last_index: int
    ceiling: Fraction
    assigned_lower: Fraction
    assigned_upper: Fraction
    physical_lower: Fraction
    physical_upper: Fraction


def _cells(
    shells: tuple[PrimeGapShell, ...],
    *,
    dimension: int,
    convolution_length: int,
    step: Fraction,
) -> tuple[PrimeGapCell, ...]:
    result: list[PrimeGapCell] = []
    for index, shell in enumerate(shells):
        first = max(0, shell.lower // step - dimension + 1)
        last = min(
            convolution_length - 1,
            shell.upper // step - dimension,
        )
        if first <= last:
            result.append(
                PrimeGapCell(
                    shell_index=index,
                    first_index=first,
                    last_index=last,
                    ceiling=(shell.ceiling // step) * step,
                    assigned_lower=shell.lower,
                    assigned_upper=shell.upper,
                    physical_lower=first * step,
                    physical_upper=(last + dimension) * step,
                )
            )
    return tuple(result)


def _event_cells(
    cells: tuple[PrimeGapCell, ...],
    *,
    dimension: int,
    step: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> tuple[PrimeGapCell, ...]:
    result: list[PrimeGapCell] = []
    for cell in cells:
        first = max(cell.first_index, lower // step - dimension + 1)
        last = min(cell.last_index, upper // step)
        if first <= last:
            result.append(
                PrimeGapCell(
                    shell_index=cell.shell_index,
                    first_index=first,
                    last_index=last,
                    ceiling=cell.ceiling,
                    assigned_lower=cell.assigned_lower,
                    assigned_upper=cell.assigned_upper,
                    physical_lower=first * step,
                    physical_upper=(last + dimension) * step,
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class PrimeGapSourceGroup:
    identifier: str
    role: SourceRole
    dimension: int
    order: Fraction
    ceiling: Fraction
    activation: Fraction
    radial_lower: Fraction
    radial_upper: Fraction
    hard_cap: Fraction
    split: Fraction
    rank_lower: Fraction
    source_rows: tuple[tuple[str, int], ...]
    cells: tuple[PrimeGapCell, ...]


def _group(
    *,
    identifier: str,
    role: SourceRole,
    dimension: int,
    order: Fraction,
    rows: tuple[PrimeGapLadderRow, ...],
    outer: bool,
    lower: Fraction,
    upper: Fraction,
    cells: tuple[PrimeGapCell, ...],
    step: Fraction,
) -> PrimeGapSourceGroup:
    ceiling = min(
        row.outer_core if outer else row.inner_core
        for row in rows
    )
    activation = min(row.xi for row in rows)
    pieces = _event_cells(
        cells,
        dimension=dimension,
        step=step,
        lower=lower,
        upper=upper,
    )
    if not pieces:
        raise ArithmeticError(f"source group {identifier} has no physical cells")
    hard_cap = max(piece.ceiling for piece in pieces)
    low_index = activation // step
    maximum_split_index = min(
        hard_cap // step,
        (ceiling / order) // step,
    )
    split_index = min(
        max(low_index + 1, (ceiling / (2 * order)) // step),
        maximum_split_index,
    )
    if not 2 <= low_index < split_index <= maximum_split_index:
        raise ArithmeticError("source split lost its exact grid guards")
    split = split_index * step
    rank_lower = max(activation, split, ceiling / (order + 1))
    return PrimeGapSourceGroup(
        identifier=identifier,
        role=role,
        dimension=dimension,
        order=order,
        ceiling=ceiling,
        activation=activation,
        radial_lower=lower,
        radial_upper=upper,
        hard_cap=hard_cap,
        split=split,
        rank_lower=rank_lower,
        source_rows=tuple((row.ladder, row.index) for row in rows),
        cells=pieces,
    )


@dataclass(frozen=True)
class PrimeGapSourceTask:
    group: str
    kind: TaskKind
    index: int
    lower: Fraction | None
    upper: Fraction | None
    slope: int | None
    young_q: int | None
    restoration: Fraction | None

    @property
    def key(self) -> tuple[str, str, int]:
        return self.group, self.kind, self.index


def _schedule(
    groups: tuple[PrimeGapSourceGroup, ...],
    *,
    restoration_old: Fraction,
    restoration_new: Fraction,
) -> tuple[PrimeGapSourceTask, ...]:
    a = tuple(Fraction(1, 20) * Fraction(6, 5) ** j for j in range(10))
    x = tuple(group.activation for group in groups)
    p = tuple(group.split for group in groups)
    low_sequences = (
        (
            x[0],
            3 * x[0] / 2,
            a[0],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            (a[8] + a[9]) / 2,
            a[9],
            p[0],
        ),
        tuple(x[1] * 2**j for j in range(9))
        + (Fraction(1, 100), Fraction(3, 200), Fraction(9, 400), Fraction(27, 800))
        + a[:8]
        + ((a[7] + p[1]) / 2, p[1]),
        (x[2], a[0], a[4], a[6], a[8], p[2]),
        (x[3], 2 * x[3], Fraction(1, 100), Fraction(27, 800), a[3], a[5], a[6], p[3]),
        (x[4], a[1], a[4], a[5], a[7], a[8], p[4]),
        (
            x[5],
            2 * x[5],
            16 * x[5],
            64 * x[5],
            256 * x[5],
            Fraction(9, 400),
            a[1],
            a[3],
            a[5],
            a[6],
            a[7],
            p[5],
        ),
    )
    rank_fractions = (
        tuple(Fraction(j, 6) for j in range(7)),
        tuple(Fraction(j, 16) for j in range(9))
        + (Fraction(5, 8), Fraction(3, 4), Fraction(7, 8), Fraction(1)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(1, 2), Fraction(1)),
        (Fraction(0), Fraction(1, 6), Fraction(1, 2), Fraction(2, 3), Fraction(1)),
        (
            Fraction(0),
            Fraction(1, 8),
            Fraction(3, 8),
            Fraction(1, 2),
            Fraction(3, 4),
            Fraction(1),
        ),
    )
    tasks: list[PrimeGapSourceTask] = []
    for group_index, (group, low, rank) in enumerate(
        zip(groups, low_sequences, rank_fractions, strict=True)
    ):
        if (
            low[0] != group.activation
            or low[-1] != group.split
            or any(left >= right for left, right in zip(low, low[1:], strict=False))
        ):
            raise ArithmeticError("low source bins do not tile their group")
        local: list[PrimeGapSourceTask] = []
        for index, (lower, upper) in enumerate(
            zip(low, low[1:], strict=False)
        ):
            slope = (
                120
                if group_index == 0 and index == 2
                else ceil(Fraction(9 if group_index == 5 else 7, 1) / upper)
            )
            local.append(
                PrimeGapSourceTask(
                    group=group.identifier,
                    kind="low",
                    index=index,
                    lower=lower,
                    upper=upper,
                    slope=slope,
                    young_q=None,
                    restoration=None,
                )
            )
        rank_origin = group.ceiling / (group.order + 1)
        for index, (lower, upper) in enumerate(
            zip(rank, rank[1:], strict=False)
        ):
            local.append(
                PrimeGapSourceTask(
                    group=group.identifier,
                    kind="rank_two",
                    index=index,
                    lower=rank_origin + lower * (group.hard_cap - rank_origin),
                    upper=rank_origin + upper * (group.hard_cap - rank_origin),
                    slope=None,
                    young_q=None,
                    restoration=None,
                )
            )
        local.append(
            PrimeGapSourceTask(
                group=group.identifier,
                kind="high",
                index=0,
                lower=None,
                upper=None,
                slope=None,
                young_q=None,
                restoration=None,
            )
        )
        for local_index, task in enumerate(local):
            if group_index < 2:
                local[local_index] = PrimeGapSourceTask(
                    group=task.group,
                    kind=task.kind,
                    index=task.index,
                    lower=task.lower,
                    upper=task.upper,
                    slope=task.slope,
                    young_q=_YOUNG_Q[group_index][local_index],
                    restoration=None,
                )
            else:
                local[local_index] = PrimeGapSourceTask(
                    group=task.group,
                    kind=task.kind,
                    index=task.index,
                    lower=task.lower,
                    upper=task.upper,
                    slope=task.slope,
                    young_q=None,
                    restoration=(
                        restoration_old if group_index < 4 else restoration_new
                    ),
                )
        tasks.extend(local)
    return tuple(tasks)


@dataclass(frozen=True)
class PrimeGapInputManifest:
    dimension: int
    face_dimension: int
    intervals: int
    convolution_length: int
    basis_size: int
    old_ladder_length: int
    new_ladder_length: int
    selected_old_length: int
    selected_new_length: int
    groups: tuple[PrimeGapSourceGroup, ...]
    tasks: tuple[PrimeGapSourceTask, ...]
    outer_task_count: int
    inner_task_count: int
    raw_form_count: int
    grid_step: Fraction
    maxima: tuple[tuple[str, Fraction], ...]
    inventory_proved: bool
    upstream_commit: str
    status: Literal["PROVED", "BLOCKED"]
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "type": PRIME_GAPS_INPUT_MANIFEST_KIND,
            "dimension": self.dimension,
            "face_dimension": self.face_dimension,
            "intervals": self.intervals,
            "convolution_length": self.convolution_length,
            "basis_size": self.basis_size,
            "old_ladder_length": self.old_ladder_length,
            "new_ladder_length": self.new_ladder_length,
            "selected_old_length": self.selected_old_length,
            "selected_new_length": self.selected_new_length,
            "source_group_dimensions": [
                [group.identifier, group.dimension] for group in self.groups
            ],
            "task_count": len(self.tasks),
            "outer_task_count": self.outer_task_count,
            "inner_task_count": self.inner_task_count,
            "raw_form_count": self.raw_form_count,
            "grid_step": _fraction_payload(self.grid_step),
            "maxima": [
                [name, _fraction_payload(value)] for name, value in self.maxima
            ],
            "inventory_proved": self.inventory_proved,
            "upstream_commit": self.upstream_commit,
            "upstream_source": PRIME_GAPS_186_SOURCE_URL,
            "status": self.status,
        }


@dataclass(frozen=True)
class PrimeGapEngineLayout:
    """Every dimension-sensitive numerical-engine quantity."""

    dimension: int
    face_dimension: int
    intervals: int
    convolution_length: int
    outer_mask_count: int
    inner_mask_count: int
    denominator_moment_count: int
    face_moment_count: int
    normalization_multiplier: int
    normalization_power: int
    midpoint_offset: Fraction
    first_radial_point: Fraction
    last_radial_point: Fraction

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "face_dimension": self.face_dimension,
            "intervals": self.intervals,
            "convolution_length": self.convolution_length,
            "outer_mask_count": self.outer_mask_count,
            "inner_mask_count": self.inner_mask_count,
            "denominator_moment_count": self.denominator_moment_count,
            "face_moment_count": self.face_moment_count,
            "normalization_multiplier": self.normalization_multiplier,
            "normalization_power": self.normalization_power,
            "midpoint_offset": _fraction_payload(self.midpoint_offset),
            "first_radial_point": _fraction_payload(self.first_radial_point),
            "last_radial_point": _fraction_payload(self.last_radial_point),
        }


def prime_gap_engine_layout(
    *,
    dimension: int,
    intervals: int = 98304,
) -> PrimeGapEngineLayout:
    """Derive the cap/source engine layout without hidden dimension literals."""

    if dimension < 3 or intervals <= dimension + 2:
        raise ValueError("invalid sieve dimension or interval count")
    rho_star = Fraction(2624989, 10**7)
    outer_radius = Fraction(2742997, 10**7) / rho_star
    step = outer_radius / intervals
    convolution_length = intervals - dimension
    midpoint_offset = Fraction(dimension, 2)
    center = Fraction(9, 10)
    return PrimeGapEngineLayout(
        dimension=dimension,
        face_dimension=dimension - 1,
        intervals=intervals,
        convolution_length=convolution_length,
        outer_mask_count=dimension,
        inner_mask_count=dimension - 1,
        denominator_moment_count=dimension,
        face_moment_count=dimension - 1,
        normalization_multiplier=dimension,
        normalization_power=dimension,
        midpoint_offset=midpoint_offset,
        first_radial_point=midpoint_offset * step - center,
        last_radial_point=(
            (convolution_length - 1 + midpoint_offset) * step - center
        ),
    )


@dataclass(frozen=True)
class ExactCoefficientInterval:
    """Closed rational interval used to replay convolution enclosures."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    def to_payload(self) -> list[str]:
        return [str(self.lower), str(self.upper)]


@dataclass(frozen=True)
class PrimeGapCoefficientDescriptor:
    """One coefficient in the pinned signature-major, degree-minor order."""

    index: int
    signature: tuple[int, ...]
    radial_degree: int

    def to_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "signature": list(self.signature),
            "radial_degree": self.radial_degree,
        }


def prime_gap_coefficient_descriptors() -> tuple[PrimeGapCoefficientDescriptor, ...]:
    """Return the authenticated 77-variable quadratic-form ordering."""

    descriptors = tuple(
        PrimeGapCoefficientDescriptor(
            index=7 * signature_index + degree,
            signature=signature,
            radial_degree=degree,
        )
        for signature_index, signature in enumerate(
            PRIME_GAP_COEFFICIENT_SIGNATURES
        )
        for degree in range(7)
    )
    if len(descriptors) != 77 or any(
        descriptor.index != index
        for index, descriptor in enumerate(descriptors)
    ):
        raise ArithmeticError("invalid PrimeGaps coefficient descriptor ordering")
    return descriptors


@dataclass(frozen=True)
class ExactSymmetricIntervalMatrix:
    """Upper-triangular exact interval matrix with sign-aware contraction."""

    size: int
    upper_triangle: tuple[ExactCoefficientInterval, ...]

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("matrix size must be positive")
        expected = self.size * (self.size + 1) // 2
        if len(self.upper_triangle) != expected:
            raise ValueError(
                f"upper triangle has {len(self.upper_triangle)} entries, "
                f"expected {expected}"
            )

    def entry(self, row: int, column: int) -> ExactCoefficientInterval:
        if not 0 <= row < self.size or not 0 <= column < self.size:
            raise IndexError("matrix index out of range")
        if row > column:
            row, column = column, row
        offset = row * self.size - row * (row - 1) // 2
        return self.upper_triangle[offset + column - row]

    def contract(
        self,
        coefficients: tuple[Fraction, ...],
    ) -> ExactCoefficientInterval:
        if len(coefficients) != self.size:
            raise ValueError("coefficient vector has the wrong length")
        lower = Fraction(0)
        upper = Fraction(0)
        for row, left in enumerate(coefficients):
            for column in range(row, self.size):
                multiplier = left * coefficients[column]
                if row != column:
                    multiplier *= 2
                interval = self.entry(row, column)
                if multiplier >= 0:
                    lower += multiplier * interval.lower
                    upper += multiplier * interval.upper
                else:
                    lower += multiplier * interval.upper
                    upper += multiplier * interval.lower
        return ExactCoefficientInterval(lower, upper)

    def to_payload(self) -> dict[str, object]:
        return {
            "size": self.size,
            "upper_triangle": [
                interval.to_payload() for interval in self.upper_triangle
            ],
        }


@dataclass(frozen=True)
class PrimeGapRoundingReserveCertificate:
    """Global reserve for raw and component decimal ceilings."""

    dimension: int
    outer_task_count: int
    inner_task_count: int
    raw_relative_decimal_scale: int
    component_relative_decimal_scale: int
    relative_reserve: Fraction
    status: str
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "outer_task_count": self.outer_task_count,
            "inner_task_count": self.inner_task_count,
            "raw_relative_decimal_scale": self.raw_relative_decimal_scale,
            "component_relative_decimal_scale": (
                self.component_relative_decimal_scale
            ),
            "relative_reserve": str(self.relative_reserve),
            "status": self.status,
            "honesty": dict(self.honesty),
        }


def certify_prime_gap_rounding_reserve(
    *,
    dimension: int,
    intervals: int = 98304,
    raw_relative_decimal_scale: int = 10**18,
    component_relative_decimal_scale: int = 10**12,
) -> PrimeGapRoundingReserveCertificate:
    """Bound both source-rounding stages for the complete task inventory."""

    if raw_relative_decimal_scale < 1 or component_relative_decimal_scale < 1:
        raise ValueError("decimal scales must be positive")
    manifest = build_prime_gap_input_manifest(
        dimension=dimension,
        intervals=intervals,
    )
    reserve = Fraction(0)
    outer_count = 0
    inner_count = 0
    for task in manifest.tasks:
        reserve += Fraction(1, component_relative_decimal_scale)
        if task.young_q is not None:
            young = Fraction(task.young_q, 10**6)
            if young <= 0:
                raise ArithmeticError("Young parameter must be positive")
            reserve += (young + 1 / young) / raw_relative_decimal_scale
            outer_count += 1
        else:
            if task.restoration is None or task.restoration <= 0:
                raise ArithmeticError("restoration coefficient must be positive")
            reserve += task.restoration / raw_relative_decimal_scale
            inner_count += 1
    if (
        outer_count != manifest.outer_task_count
        or inner_count != manifest.inner_task_count
        or outer_count + inner_count != len(manifest.tasks)
    ):
        raise ArithmeticError("rounding reserve omitted source tasks")
    return PrimeGapRoundingReserveCertificate(
        dimension=dimension,
        outer_task_count=outer_count,
        inner_task_count=inner_count,
        raw_relative_decimal_scale=raw_relative_decimal_scale,
        component_relative_decimal_scale=component_relative_decimal_scale,
        relative_reserve=reserve,
        status="PROVED",
        honesty={
            "finite_rounding_reserve_proved": True,
            "quadratic_kernels_extracted": False,
            "k39_crossing_proved": False,
            "analytic_distribution_inputs_proved": False,
            "h1_182_claim": False,
        },
    )


@dataclass(frozen=True)
class PrimeGapQuadraticKernelSpec:
    """Acceptance manifest for coefficient-independent interval matrices."""

    dimension: int
    intervals: int
    source_sha256: str
    arb_precision_bits: int
    cap_fractional_bits: int
    source_fractional_bits: int
    signed_convolution_strategy: str
    variable_count: int
    upper_triangle_entry_count: int
    descriptors: tuple[PrimeGapCoefficientDescriptor, ...]
    cap_forms: tuple[str, ...]
    source_task_count: int
    raw_source_form_count: int
    source_task_keys: tuple[tuple[str, str, int], ...]
    rounding_reserve: PrimeGapRoundingReserveCertificate

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "intervals": self.intervals,
            "source_sha256": self.source_sha256,
            "arb_precision_bits": self.arb_precision_bits,
            "cap_fractional_bits": self.cap_fractional_bits,
            "source_fractional_bits": self.source_fractional_bits,
            "signed_convolution_strategy": self.signed_convolution_strategy,
            "variable_count": self.variable_count,
            "upper_triangle_entry_count": self.upper_triangle_entry_count,
            "descriptors": [
                descriptor.to_payload() for descriptor in self.descriptors
            ],
            "cap_forms": list(self.cap_forms),
            "source_task_count": self.source_task_count,
            "raw_source_form_count": self.raw_source_form_count,
            "source_task_keys": [list(key) for key in self.source_task_keys],
            "rounding_reserve": self.rounding_reserve.to_payload(),
        }


def prime_gap_quadratic_kernel_spec(
    *,
    dimension: int,
    intervals: int = 98304,
) -> PrimeGapQuadraticKernelSpec:
    """Define the complete matrix artifact required before optimization."""

    manifest = build_prime_gap_input_manifest(
        dimension=dimension,
        intervals=intervals,
    )
    descriptors = prime_gap_coefficient_descriptors()
    return PrimeGapQuadraticKernelSpec(
        dimension=dimension,
        intervals=intervals,
        source_sha256=PRIME_GAPS_186_SOURCE_SHA256,
        arb_precision_bits=160,
        cap_fractional_bits=224,
        source_fractional_bits=192,
        signed_convolution_strategy="nonnegative_part_split",
        variable_count=len(descriptors),
        upper_triangle_entry_count=len(descriptors) * (len(descriptors) + 1) // 2,
        descriptors=descriptors,
        cap_forms=("denominator", "J0", "Jplus", "Jtail"),
        source_task_count=len(manifest.tasks),
        raw_source_form_count=manifest.raw_form_count,
        source_task_keys=tuple(
            (task.group, task.kind, task.index) for task in manifest.tasks
        ),
        rounding_reserve=certify_prime_gap_rounding_reserve(
            dimension=dimension,
            intervals=intervals,
        ),
    )


@dataclass(frozen=True)
class SignedConvolutionSplitCertificate:
    """Exact replay of a signed-times-nonnegative interval convolution."""

    signed_length: int
    nonnegative_length: int
    direct: tuple[ExactCoefficientInterval, ...]
    split: tuple[ExactCoefficientInterval, ...]
    exact_match: bool
    status: str
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "signed_length": self.signed_length,
            "nonnegative_length": self.nonnegative_length,
            "direct": [interval.to_payload() for interval in self.direct],
            "split": [interval.to_payload() for interval in self.split],
            "exact_match": self.exact_match,
            "status": self.status,
            "honesty": dict(self.honesty),
        }


def _add_exact_intervals(
    left: ExactCoefficientInterval,
    right: ExactCoefficientInterval,
) -> ExactCoefficientInterval:
    return ExactCoefficientInterval(
        left.lower + right.lower,
        left.upper + right.upper,
    )


def _subtract_exact_intervals(
    left: ExactCoefficientInterval,
    right: ExactCoefficientInterval,
) -> ExactCoefficientInterval:
    return ExactCoefficientInterval(
        left.lower - right.upper,
        left.upper - right.lower,
    )


def _multiply_by_nonnegative_interval(
    signed: ExactCoefficientInterval,
    nonnegative: ExactCoefficientInterval,
) -> ExactCoefficientInterval:
    if nonnegative.lower < 0:
        raise ValueError("kernel coefficient interval is not nonnegative")
    products = (
        signed.lower * nonnegative.lower,
        signed.lower * nonnegative.upper,
        signed.upper * nonnegative.lower,
        signed.upper * nonnegative.upper,
    )
    return ExactCoefficientInterval(min(products), max(products))


def _convolve_exact_intervals(
    left: tuple[ExactCoefficientInterval, ...],
    right: tuple[ExactCoefficientInterval, ...],
) -> tuple[ExactCoefficientInterval, ...]:
    if not left or not right:
        return ()
    zero = ExactCoefficientInterval(Fraction(0), Fraction(0))
    output = [zero] * (len(left) + len(right) - 1)
    for i, left_interval in enumerate(left):
        for j, right_interval in enumerate(right):
            product = _multiply_by_nonnegative_interval(
                left_interval,
                right_interval,
            )
            output[i + j] = _add_exact_intervals(output[i + j], product)
    return tuple(output)


def certify_signed_convolution_split(
    signed: tuple[ExactCoefficientInterval, ...],
    nonnegative: tuple[ExactCoefficientInterval, ...],
) -> SignedConvolutionSplitCertificate:
    """Prove ``P*Q = P_+*Q - P_-*Q`` in exact interval arithmetic."""

    if any(interval.lower < 0 for interval in nonnegative):
        raise ValueError("kernel coefficient intervals must be nonnegative")
    direct = _convolve_exact_intervals(signed, nonnegative)
    positive = tuple(
        ExactCoefficientInterval(
            max(Fraction(0), interval.lower),
            max(Fraction(0), interval.upper),
        )
        for interval in signed
    )
    negative = tuple(
        ExactCoefficientInterval(
            max(Fraction(0), -interval.upper),
            max(Fraction(0), -interval.lower),
        )
        for interval in signed
    )
    positive_convolution = _convolve_exact_intervals(positive, nonnegative)
    negative_convolution = _convolve_exact_intervals(negative, nonnegative)
    split = tuple(
        _subtract_exact_intervals(plus, minus)
        for plus, minus in zip(
            positive_convolution,
            negative_convolution,
            strict=True,
        )
    )
    exact_match = direct == split
    if not exact_match:
        raise ArithmeticError("signed convolution split replay failed")
    return SignedConvolutionSplitCertificate(
        signed_length=len(signed),
        nonnegative_length=len(nonnegative),
        direct=direct,
        split=split,
        exact_match=True,
        status="PROVED",
        honesty={
            "finite_exact_interval_identity_proved": True,
            "arb_implementation_equivalence_proved": False,
            "k40_numerical_equivalence_proved": False,
            "h1_182_claim": False,
        },
    )


@dataclass(frozen=True)
class PrimeGapNumericalReceiptCertificate:
    """Independent exact replay of a generated cap/source receipt."""

    dimension: int
    component_count: int
    raw_form_count: int
    source_total_relative_units: int
    final_margin: Fraction
    required_margin: Fraction
    published_baseline_floor_met: bool | None
    receipt_sha256: str
    status: str
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "component_count": self.component_count,
            "raw_form_count": self.raw_form_count,
            "source_total_relative_units": self.source_total_relative_units,
            "final_margin": str(self.final_margin),
            "required_margin": str(self.required_margin),
            "published_baseline_floor_met": self.published_baseline_floor_met,
            "receipt_sha256": self.receipt_sha256,
            "status": self.status,
            "honesty": dict(self.honesty),
        }


def _receipt_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _receipt_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a sequence")
    return value


def _receipt_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(result) != value:
        raise ValueError(f"{label} is not a canonical integer")
    return result


def _receipt_fraction(value: object, label: str) -> Fraction:
    if not isinstance(value, str | int):
        raise ValueError(f"{label} must be an exact rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be an exact rational string") from exc


@dataclass(frozen=True)
class PrimeGapQuadraticReceiptCertificate:
    """Structural replay of a coefficient-independent interval-Gram receipt."""

    dimension: int
    transformed_source_sha256: str
    matrices: dict[str, ExactSymmetricIntervalMatrix]
    receipt_sha256: str
    status: str
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "transformed_source_sha256": self.transformed_source_sha256,
            "matrix_names": sorted(self.matrices),
            "matrix_size": next(iter(self.matrices.values())).size,
            "matrix_entry_count": sum(
                len(matrix.upper_triangle) for matrix in self.matrices.values()
            ),
            "receipt_sha256": self.receipt_sha256,
            "status": self.status,
            "honesty": dict(self.honesty),
        }


def _quadratic_receipt_fraction(value: object, label: str) -> Fraction:
    result = _receipt_fraction(value, label)
    if isinstance(value, str) and str(result) != value:
        raise ValueError(f"{label} is not a canonical rational")
    return result


def _is_dyadic(value: Fraction) -> bool:
    denominator = value.denominator
    return denominator & (denominator - 1) == 0


def _quadratic_receipt_matrix(
    value: object,
    *,
    name: str,
    size: int,
) -> ExactSymmetricIntervalMatrix:
    payload = _receipt_mapping(value, f"matrices.{name}")
    if set(payload) != {"size", "upper_triangle"}:
        raise ValueError(f"matrices.{name} has unexpected fields")
    if _receipt_integer(payload.get("size"), f"matrices.{name}.size") != size:
        raise ArithmeticError(f"matrices.{name} has the wrong dimension")
    rows = _receipt_sequence(
        payload.get("upper_triangle"),
        f"matrices.{name}.upper_triangle",
    )
    expected = size * (size + 1) // 2
    if len(rows) != expected:
        raise ArithmeticError(f"matrices.{name} is missing upper-triangle entries")
    intervals: list[ExactCoefficientInterval] = []
    for index, raw in enumerate(rows):
        endpoints = _receipt_sequence(raw, f"matrices.{name}[{index}]")
        if len(endpoints) != 2:
            raise ValueError(f"matrices.{name}[{index}] needs two endpoints")
        lower = _quadratic_receipt_fraction(
            endpoints[0],
            f"matrices.{name}[{index}].lower",
        )
        upper = _quadratic_receipt_fraction(
            endpoints[1],
            f"matrices.{name}[{index}].upper",
        )
        if not _is_dyadic(lower) or not _is_dyadic(upper):
            raise ArithmeticError(
                f"matrices.{name}[{index}] endpoints are not exact dyadics"
            )
        intervals.append(ExactCoefficientInterval(lower, upper))
    return ExactSymmetricIntervalMatrix(size=size, upper_triangle=tuple(intervals))


def certify_prime_gap_quadratic_receipt(
    receipt: Mapping[str, object],
    *,
    dimension: int,
    intervals: int = 98304,
) -> PrimeGapQuadraticReceiptCertificate:
    """Replay the complete schema of an extracted interval-Gram artifact."""

    spec = prime_gap_quadratic_kernel_spec(
        dimension=dimension,
        intervals=intervals,
    )
    expected_scalars: dict[str, object] = {
        "schema": "omnibias.prime_gap_quadratic.v1",
        "algorithm": "coefficient_independent_interval_gram_v1",
        "source_sha256": spec.source_sha256,
        "dimension": dimension,
        "intervals": intervals,
        "arb_precision_bits": spec.arb_precision_bits,
        "cap_fractional_bits": spec.cap_fractional_bits,
        "source_fractional_bits": spec.source_fractional_bits,
        "signed_convolution_strategy": spec.signed_convolution_strategy,
    }
    expected_fields = set(expected_scalars) | {
        "transformed_source_sha256",
        "descriptors",
        "source_task_keys",
        "matrices",
    }
    if set(receipt) != expected_fields:
        raise ValueError("quadratic receipt has unexpected or missing fields")
    for key, expected in expected_scalars.items():
        if receipt.get(key) != expected:
            raise ArithmeticError(f"quadratic receipt {key} does not match the spec")
    transformed_sha = receipt.get("transformed_source_sha256")
    if (
        not isinstance(transformed_sha, str)
        or len(transformed_sha) != 64
        or any(character not in "0123456789abcdef" for character in transformed_sha)
    ):
        raise ValueError("invalid transformed source SHA-256")
    expected_descriptors = [
        descriptor.to_payload() for descriptor in spec.descriptors
    ]
    if receipt.get("descriptors") != expected_descriptors:
        raise ArithmeticError("quadratic receipt coefficient order changed")
    expected_task_keys = [list(key) for key in spec.source_task_keys]
    if receipt.get("source_task_keys") != expected_task_keys:
        raise ArithmeticError("quadratic receipt source task inventory changed")
    raw_matrices = _receipt_mapping(receipt.get("matrices"), "matrices")
    if set(raw_matrices) != set(PRIME_GAP_QUADRATIC_MATRIX_NAMES):
        raise ArithmeticError("quadratic receipt matrix inventory is incomplete")
    matrices = {
        name: _quadratic_receipt_matrix(
            raw_matrices[name],
            name=name,
            size=spec.variable_count,
        )
        for name in PRIME_GAP_QUADRATIC_MATRIX_NAMES
    }
    receipt_sha256 = hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return PrimeGapQuadraticReceiptCertificate(
        dimension=dimension,
        transformed_source_sha256=transformed_sha,
        matrices=matrices,
        receipt_sha256=receipt_sha256,
        status="PROVED",
        honesty={
            "finite_matrix_schema_replayed": True,
            "exact_dyadic_entries_replayed": True,
            "coefficient_order_replayed": True,
            "source_task_inventory_replayed": True,
            "quadratic_kernels_extracted": True,
            "matrix_values_independently_recomputed": False,
            "finite_k39_crossing_found": False,
            "source_valid_k39_crossing_proved": False,
            "h1_182_claim": False,
            "dhl_claim": False,
            "twin_prime_claim": False,
        },
    )


def _scale_exact_interval(
    interval: ExactCoefficientInterval,
    scalar: Fraction,
) -> ExactCoefficientInterval:
    if scalar >= 0:
        return ExactCoefficientInterval(
            scalar * interval.lower,
            scalar * interval.upper,
        )
    return ExactCoefficientInterval(
        scalar * interval.upper,
        scalar * interval.lower,
    )


def _sum_exact_intervals(
    *intervals: ExactCoefficientInterval,
) -> ExactCoefficientInterval:
    return ExactCoefficientInterval(
        sum((interval.lower for interval in intervals), Fraction(0)),
        sum((interval.upper for interval in intervals), Fraction(0)),
    )


def _fraction_floor(value: Fraction, scale: int) -> Fraction:
    scaled = value * scale
    return Fraction(scaled.numerator // scaled.denominator, scale)


def _fraction_ceil(value: Fraction, scale: int) -> Fraction:
    scaled = value * scale
    return Fraction(-((-scaled.numerator) // scaled.denominator), scale)


@dataclass(frozen=True)
class PrimeGapQuadraticCandidateCertificate:
    """Exact screening result for one rational coefficient vector."""

    dimension: int
    denominator: ExactCoefficientInterval
    cap_numerator: ExactCoefficientInterval
    source_loss: ExactCoefficientInterval
    rounded_cap_numerator_lower: Fraction
    rounded_denominator_upper: Fraction
    rounded_source_loss_upper: Fraction
    residual_lower: Fraction
    residual_upper: Fraction
    status: str
    honesty: dict[str, bool]

    def to_payload(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "denominator": self.denominator.to_payload(),
            "cap_numerator": self.cap_numerator.to_payload(),
            "source_loss": self.source_loss.to_payload(),
            "rounded_cap_numerator_lower": str(
                self.rounded_cap_numerator_lower
            ),
            "rounded_denominator_upper": str(self.rounded_denominator_upper),
            "rounded_source_loss_upper": str(self.rounded_source_loss_upper),
            "residual_lower": str(self.residual_lower),
            "residual_upper": str(self.residual_upper),
            "status": self.status,
            "honesty": dict(self.honesty),
        }


def certify_prime_gap_quadratic_candidate(
    receipt: PrimeGapQuadraticReceiptCertificate,
    coefficients: tuple[Fraction, ...],
) -> PrimeGapQuadraticCandidateCertificate:
    """Screen a rational vector with exact matrix contraction and reserves."""

    spec = prime_gap_quadratic_kernel_spec(dimension=receipt.dimension)
    if receipt.status != "PROVED" or set(receipt.matrices) != set(
        PRIME_GAP_QUADRATIC_MATRIX_NAMES
    ):
        raise ArithmeticError("quadratic matrix receipt is not structurally proved")
    if len(coefficients) != spec.variable_count:
        raise ValueError("candidate must provide all 77 coefficients")
    forms = {
        name: matrix.contract(coefficients)
        for name, matrix in receipt.matrices.items()
    }
    denominator = forms["denominator"]
    if denominator.lower <= 0:
        raise ArithmeticError("candidate denominator is not proved positive")
    pair_constant = Fraction(2479900401, 2500000000)
    tail_constant = Fraction(-843183, 10**9)
    cap_numerator = _sum_exact_intervals(
        forms["J0"],
        _scale_exact_interval(
            forms["Jplus"],
            pair_constant + tail_constant,
        ),
        _scale_exact_interval(forms["Jtail"], tail_constant),
    )
    source_loss = forms["source_loss"]
    decimal_scale = 10**24
    rounded_cap_lower = _fraction_floor(cap_numerator.lower, decimal_scale)
    rounded_denominator_upper = _fraction_ceil(
        denominator.upper,
        decimal_scale,
    )
    rounded_source_upper = (
        source_loss.upper
        + spec.rounding_reserve.relative_reserve * denominator.upper
    )
    rho_star = Fraction(2624989, 10**7)
    required_margin = Fraction(1, 50000)
    residual_lower = (
        rho_star * (rounded_cap_lower - rounded_source_upper)
        - (1 + required_margin) * rounded_denominator_upper
    )
    residual_upper = (
        rho_star * (cap_numerator.upper - max(Fraction(0), source_loss.lower))
        - (1 + required_margin) * denominator.lower
    )
    status = (
        "PROVED"
        if residual_lower > 0
        else "DISPROVED"
        if residual_upper < 0
        else "BLOCKED"
    )
    return PrimeGapQuadraticCandidateCertificate(
        dimension=receipt.dimension,
        denominator=denominator,
        cap_numerator=cap_numerator,
        source_loss=source_loss,
        rounded_cap_numerator_lower=rounded_cap_lower,
        rounded_denominator_upper=rounded_denominator_upper,
        rounded_source_loss_upper=rounded_source_upper,
        residual_lower=residual_lower,
        residual_upper=residual_upper,
        status=status,
        honesty={
            "finite_interval_matrix_screen_proved": status == "PROVED",
            "matrix_values_independently_recomputed": False,
            "complete_direct_receipt_replayed": False,
            "quadratic_candidate_is_screen": True,
            "finite_k39_crossing_found": False,
            "source_valid_k39_crossing_proved": False,
            "h1_182_claim": False,
            "dhl_claim": False,
            "twin_prime_claim": False,
        },
    )


def certify_prime_gap_numerical_receipt(
    receipt: Mapping[str, object],
    *,
    dimension: int,
    intervals: int = 98304,
) -> PrimeGapNumericalReceiptCertificate:
    """Check inventory and exact final arithmetic of a numerical receipt."""

    manifest = build_prime_gap_input_manifest(
        dimension=dimension,
        intervals=intervals,
    )
    layout = prime_gap_engine_layout(
        dimension=dimension,
        intervals=intervals,
    )
    settings = _receipt_mapping(receipt.get("settings"), "settings")
    if (
        _receipt_integer(settings.get("intervals"), "settings.intervals")
        != intervals
        or _receipt_integer(
            settings.get("convolution_length"),
            "settings.convolution_length",
        )
        != layout.convolution_length
    ):
        raise ArithmeticError("receipt engine layout does not match its dimension")
    expected_normalization = f"physical form / (h*sum(g_j^2))^{dimension}"
    if receipt.get("normalization") != expected_normalization:
        raise ArithmeticError("receipt normalization power is inconsistent")
    cap = _receipt_mapping(receipt.get("cap"), "cap")
    units = _receipt_mapping(cap.get("rounded_units"), "cap.rounded_units")
    cap_scale = _receipt_integer(
        settings.get("cap_decimal_scale"),
        "settings.cap_decimal_scale",
    )
    raw_scale = _receipt_integer(
        settings.get("raw_relative_decimal_scale"),
        "settings.raw_relative_decimal_scale",
    )
    component_scale = _receipt_integer(
        settings.get("component_relative_decimal_scale"),
        "settings.component_relative_decimal_scale",
    )
    young_denominator = _receipt_integer(
        settings.get("young_denominator"),
        "settings.young_denominator",
    )
    normalized_forms = _receipt_mapping(
        cap.get("normalized_forms"),
        "cap.normalized_forms",
    )
    denominator_bounds = _receipt_mapping(
        normalized_forms.get("denominator"),
        "cap.normalized_forms.denominator",
    )
    denominator_lower = _receipt_fraction(
        denominator_bounds.get("lower"),
        "cap.normalized_forms.denominator.lower",
    )
    denominator_upper = _receipt_fraction(
        denominator_bounds.get("upper"),
        "cap.normalized_forms.denominator.upper",
    )
    numerator_bounds = _receipt_mapping(
        cap.get("hybrid_numerator"),
        "cap.hybrid_numerator",
    )
    numerator_lower = _receipt_fraction(
        numerator_bounds.get("lower"),
        "cap.hybrid_numerator.lower",
    )
    numerator_upper = _receipt_fraction(
        numerator_bounds.get("upper"),
        "cap.hybrid_numerator.upper",
    )
    if (
        denominator_lower <= 0
        or denominator_lower > denominator_upper
        or numerator_lower > numerator_upper
    ):
        raise ArithmeticError("receipt cap intervals are inconsistent")
    if (
        _receipt_integer(units.get("I_lower"), "cap.I_lower")
        != (denominator_lower * cap_scale).numerator
        // (denominator_lower * cap_scale).denominator
        or _receipt_integer(units.get("I_upper"), "cap.I_upper")
        != ceil(denominator_upper * cap_scale)
        or _receipt_integer(units.get("J_lower"), "cap.J_lower")
        != (numerator_lower * cap_scale).numerator
        // (numerator_lower * cap_scale).denominator
    ):
        raise ArithmeticError("receipt cap rounding units are inconsistent")
    denominator = Fraction(
        _receipt_integer(units.get("I_lower"), "cap.I_lower"),
        cap_scale,
    )
    upper = Fraction(
        _receipt_integer(units.get("I_upper"), "cap.I_upper"),
        cap_scale,
    )
    numerator = Fraction(
        _receipt_integer(units.get("J_lower"), "cap.J_lower"),
        cap_scale,
    )
    if denominator <= 0 or upper < denominator:
        raise ArithmeticError("receipt cap denominator enclosure is invalid")
    if denominator != _receipt_fraction(
        receipt.get("source_normalization_denominator"),
        "source_normalization_denominator",
    ):
        raise ArithmeticError("receipt source denominator is inconsistent")

    expected_tasks: dict[tuple[str, str, int], dict[str, Fraction]] = {}
    expected_task_specs: dict[tuple[str, str, int], PrimeGapSourceTask] = {}
    for expected_task in manifest.tasks:
        if expected_task.kind == "low":
            assert expected_task.lower is not None
            assert expected_task.upper is not None
            assert expected_task.slope is not None
            expected_parameters = {
                "low": expected_task.lower,
                "high": expected_task.upper,
                "slope": Fraction(expected_task.slope),
            }
        elif expected_task.kind == "rank_two":
            assert expected_task.lower is not None
            assert expected_task.upper is not None
            expected_parameters = {
                "q_low": expected_task.lower,
                "q_high": expected_task.upper,
            }
        else:
            expected_parameters = {}
        expected_key = (
            expected_task.group,
            expected_task.kind,
            expected_task.index,
        )
        expected_tasks[expected_key] = expected_parameters
        expected_task_specs[expected_key] = expected_task
    components = _receipt_sequence(receipt.get("components"), "components")
    seen: set[tuple[str, str, int]] = set()
    total_units = 0
    raw_form_count = 0
    for component_value in components:
        component = _receipt_mapping(component_value, "component")
        task_mapping = _receipt_mapping(component.get("task"), "component.task")
        group = task_mapping.get("group")
        kind = task_mapping.get("kind")
        if not isinstance(group, str) or not isinstance(kind, str):
            raise ValueError("component task group and kind must be strings")
        index = _receipt_integer(
            task_mapping.get("index"),
            "component.task.index",
        )
        key = (group, kind, index)
        if key not in expected_tasks or key in seen:
            raise ArithmeticError("receipt has an unknown or duplicate task")
        parameter_mapping = _receipt_mapping(
            task_mapping.get("parameters"),
            "component.task.parameters",
        )
        actual_parameters = {
            name: _receipt_fraction(value, f"task parameter {name}")
            for name, value in parameter_mapping.items()
        }
        if actual_parameters != expected_tasks[key]:
            raise ArithmeticError("receipt task parameters do not match the manifest")
        expected_task = expected_task_specs[key]
        if expected_task.young_q is not None:
            if (
                _receipt_integer(task_mapping.get("young_q"), "task.young_q")
                != expected_task.young_q
                or _receipt_integer(
                    task_mapping.get("young_denominator"),
                    "task.young_denominator",
                )
                != young_denominator
                or young_denominator != 10**6
                or _receipt_fraction(task_mapping.get("young"), "task.young")
                != Fraction(expected_task.young_q, young_denominator)
            ):
                raise ArithmeticError("receipt Young parameter does not match")
        elif expected_task.restoration is not None:
            if _receipt_fraction(
                task_mapping.get("restoration_coefficient"),
                "task.restoration_coefficient",
            ) != expected_task.restoration:
                raise ArithmeticError(
                    "receipt restoration coefficient does not match"
                )
        raw_forms = _receipt_mapping(
            component.get("raw_forms"),
            "component.raw_forms",
        )
        expected_raw_forms = (
            {"root_square", "outer_face_square"}
            if group.startswith("outer_")
            else {"inner_face"}
        )
        if set(raw_forms) != expected_raw_forms:
            raise ArithmeticError("receipt task has the wrong raw-form count")
        relative_units: dict[str, int] = {}
        for raw_name, raw_value in raw_forms.items():
            bounds = _receipt_mapping(
                raw_value,
                f"component.raw_forms.{raw_name}",
            )
            lower = _receipt_fraction(
                bounds.get("lower"),
                f"component.raw_forms.{raw_name}.lower",
            )
            raw_upper = _receipt_fraction(
                bounds.get("upper"),
                f"component.raw_forms.{raw_name}.upper",
            )
            if not 0 <= lower <= raw_upper:
                raise ArithmeticError("receipt has invalid positive raw-form bounds")
            relative_units[raw_name] = ceil(
                raw_upper * raw_scale / denominator
            )
        recorded_relative_units = _receipt_mapping(
            component.get("raw_relative_units"),
            "component.raw_relative_units",
        )
        if {
            name: _receipt_integer(value, f"raw_relative_units.{name}")
            for name, value in recorded_relative_units.items()
        } != relative_units:
            raise ArithmeticError("receipt raw relative units are inconsistent")
        if expected_task.young_q is not None:
            young = Fraction(expected_task.young_q, young_denominator)
            component_cost = (
                young * Fraction(relative_units["root_square"], raw_scale)
                + Fraction(
                    relative_units["outer_face_square"],
                    raw_scale,
                )
                / young
            )
        else:
            assert expected_task.restoration is not None
            component_cost = expected_task.restoration * Fraction(
                relative_units["inner_face"],
                raw_scale,
            )
        calculated_component_units = ceil(component_cost * component_scale)
        recorded_component_units = _receipt_integer(
            component.get("component_relative_units"),
            "component.component_relative_units",
        )
        if recorded_component_units != calculated_component_units:
            raise ArithmeticError("receipt component units are inconsistent")
        component_loss = denominator * Fraction(
            recorded_component_units,
            component_scale,
        )
        if component_loss != _receipt_fraction(
            component.get("normalized_loss_upper"),
            "component.normalized_loss_upper",
        ):
            raise ArithmeticError("receipt component loss is inconsistent")
        raw_form_count += len(raw_forms)
        total_units += recorded_component_units
        seen.add(key)
    if seen != set(expected_tasks):
        raise ArithmeticError("receipt does not cover the complete source inventory")
    if (
        len(components) != len(manifest.tasks)
        or raw_form_count != manifest.raw_form_count
        or _receipt_integer(receipt.get("component_count"), "component_count")
        != len(components)
        or _receipt_integer(receipt.get("raw_form_count"), "raw_form_count")
        != raw_form_count
    ):
        raise ArithmeticError("receipt source inventory counts are inconsistent")
    recorded_total = _receipt_integer(
        receipt.get("source_total_relative_units"),
        "source_total_relative_units",
    )
    if total_units != recorded_total:
        raise ArithmeticError("receipt source-unit total is inconsistent")

    loss = denominator * Fraction(total_units, component_scale)
    if loss != _receipt_fraction(
        receipt.get("normalized_source_loss_upper"),
        "normalized_source_loss_upper",
    ):
        raise ArithmeticError("receipt source loss is inconsistent")
    margin = Fraction(2624989, 10**7) * (numerator - loss) / upper - 1
    if margin != _receipt_fraction(
        receipt.get("final_margin_lower"),
        "final_margin_lower",
    ):
        raise ArithmeticError("receipt final margin is inconsistent")
    if 1 + margin != _receipt_fraction(
        receipt.get("final_quotient_lower"),
        "final_quotient_lower",
    ):
        raise ArithmeticError("receipt final quotient is inconsistent")
    required_margin = _receipt_fraction(
        receipt.get("required_margin"),
        "required_margin",
    )
    passed = margin > required_margin
    if receipt.get("passed") is not passed:
        raise ArithmeticError("receipt pass flag is inconsistent")
    expected_status = (
        "PASS_FRESH_NUMERICAL_CERTIFICATE"
        if passed
        else "FAIL_NUMERICAL_MARGIN"
    )
    if receipt.get("status") != expected_status:
        raise ArithmeticError("receipt status is inconsistent")
    published_baseline_floor_met = (
        margin >= Fraction(230382667, 10**13) if dimension == 40 else None
    )
    receipt_sha256 = hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return PrimeGapNumericalReceiptCertificate(
        dimension=dimension,
        component_count=len(components),
        raw_form_count=raw_form_count,
        source_total_relative_units=total_units,
        final_margin=margin,
        required_margin=required_margin,
        published_baseline_floor_met=published_baseline_floor_met,
        receipt_sha256=receipt_sha256,
        status="PROVED" if passed else "DISPROVED",
        honesty={
            "finite_receipt_arithmetic_replayed": True,
            "exact_task_inventory_replayed": True,
            "exact_component_rounding_replayed": True,
            "finite_k39_crossing_found": dimension == 39 and passed,
            "source_valid_k39_crossing_proved": False,
            "h1_182_claim": False,
            "arb_interval_algorithms_formally_verified": False,
            "analytic_distribution_inputs_proved": False,
            "dhl_claim": False,
            "twin_prime_claim": False,
        },
    )


def seal_prime_gap_numerical_receipt(
    certificate: PrimeGapNumericalReceiptCertificate,
) -> dict[str, object]:
    """Seal the exact receipt replay without upgrading analytic claims."""

    return make_certificate(
        claim=PRIME_GAPS_NUMERICAL_RECEIPT_KIND,
        payload=certificate.to_payload(),
        honesty=dict(certificate.honesty),
        meta={
            "method": "exact_fraction_receipt_replay",
            "status": certificate.status,
            "source_receipt_sha256": certificate.receipt_sha256,
            "scope": "finite numerical receipt only",
        },
    )


def seal_prime_gap_quadratic_receipt(
    certificate: PrimeGapQuadraticReceiptCertificate,
) -> dict[str, object]:
    """Seal matrix structure and provenance without asserting a crossing."""

    return make_certificate(
        claim=PRIME_GAPS_QUADRATIC_RECEIPT_KIND,
        payload=certificate.to_payload(),
        honesty=dict(certificate.honesty),
        meta={
            "method": "exact_interval_matrix_receipt_replay",
            "status": certificate.status,
            "source_receipt_sha256": certificate.receipt_sha256,
            "scope": "finite quadratic receipt structure only",
        },
    )


def _replace_exact(
    source: str,
    old: str,
    new: str,
    *,
    expected: int = 1,
) -> str:
    count = source.count(old)
    if count != expected:
        raise ValueError(
            f"pinned evaluator drift: expected {expected} occurrence(s) of {old!r}, "
            f"found {count}"
        )
    return source.replace(old, new)


def _prime_gap_evaluator_rewrites(
    *,
    dimension: int,
    intervals: int = 98304,
) -> tuple[tuple[str, str], ...]:
    """Return the complete fail-closed pinned-source rewrite inventory."""

    layout = prime_gap_engine_layout(
        dimension=dimension,
        intervals=intervals,
    )
    rewrites = (
        (
            "import multiprocessing as mp\nimport sys",
            "import multiprocessing as mp\nimport os\nimport sys",
        ),
        (
            '"convolution_length": 98264,',
            f'"convolution_length": {layout.convolution_length},',
        ),
        ("k, N = 40, 98304", f"k, N = {dimension}, {intervals}"),
        (
            "def __init__(self, intervals=98304, precision=160, "
            "fixed_bits=224, arb_threads=1):\n"
            "        _cap_check_environment()\n"
            "        if intervals <= 42",
            "def __init__(self, intervals=98304, precision=160, "
            "fixed_bits=224, arb_threads=1, dimension=40):\n"
            "        dimension = int(dimension)\n"
            "        _cap_check_environment()\n"
            "        if intervals <= dimension + 2",
        ),
        (
            "self.k, self.intervals, self.n = 40, int(intervals), "
            "int(intervals) - 40",
            "self.k, self.intervals, self.n = dimension, int(intervals), "
            "int(intervals) - dimension",
        ),
        (
            "self.shell_mask(s, 40) for s in self.outer_shells",
            "self.shell_mask(s, self.k) for s in self.outer_shells",
        ),
        (
            "self.shell_mask(shell, 39)",
            "self.shell_mask(shell, self.k - 1)",
        ),
        (
            "arb(40) * self.h / self.Z, (self.h * self.Z) ** 40",
            "arb(self.k) * self.h / self.Z, (self.h * self.Z) ** self.k",
        ),
        (
            "rational((F(j) + 20) * self.hq - self.center)",
            "rational((F(j) + F(self.k, 2)) * self.hq - self.center)",
        ),
        (
            "self.moment_interval(cap, 39, eta)",
            "self.moment_interval(cap, self.k - 1, eta)",
        ),
        (
            "self.moment_interval(self.caps[layer - 1], 39, eta)",
            "self.moment_interval(self.caps[layer - 1], self.k - 1, eta)",
        ),
        (
            "engine = CapEngine(\n"
            '        intervals=POLICY["intervals"],',
            "engine = CapEngine(\n"
            '        intervals=POLICY["intervals"],\n'
            '        dimension=int(TRIAL["dimension"]),',
        ),
        (
            "engine.k != 40",
            'engine.k != int(TRIAL["dimension"])',
        ),
        (
            'normalization="physical form / (h*sum(g_j^2))^40",',
            "normalization=f\"physical form / (h*sum(g_j^2))^"
            "{int(TRIAL['dimension'])}\",",
        ),
        (
            "def _driver_emit(event, **fields):\n"
            "    print(json.dumps(dict(event=event, **fields), "
            "sort_keys=True, allow_nan=False), flush=True)",
            "def _driver_emit(event, **fields):\n"
            "    print(json.dumps(dict(event=event, **fields), "
            "sort_keys=True, allow_nan=False), flush=True)\n"
            "\n"
            "\n"
            "def _driver_checkpoint(path, event, **fields):\n"
            "    if path is None:\n"
            "        return\n"
            "    with path.open(\"a\", encoding=\"utf-8\") as output:\n"
            "        output.write(json.dumps(dict(event=event, **fields), "
            "sort_keys=True, allow_nan=False) + \"\\n\")\n"
            "        output.flush()\n"
            "        os.fsync(output.fileno())\n"
            "\n"
            "\n"
            "def _driver_load_resume(paths):\n"
            "    expected, cap, rows = _driver_inventory(), None, {}\n"
            "    for path in paths:\n"
            "        with path.open(\"r\", encoding=\"utf-8\") as source:\n"
            "            for line in source:\n"
            "                try:\n"
            "                    event = json.loads(line)\n"
            "                except (json.JSONDecodeError, TypeError):\n"
            "                    continue\n"
            "                if not isinstance(event, dict):\n"
            "                    continue\n"
            "                if event.get(\"event\") == \"CAP_COMPLETE\":\n"
            "                    candidate = event.get(\"cap\")\n"
            "                    if not isinstance(candidate, dict):\n"
            "                        raise ArithmeticError(\"invalid resumed cap\")\n"
            "                    if cap is not None and candidate != cap:\n"
            "                        raise ArithmeticError(\"conflicting resumed caps\")\n"
            "                    cap = candidate\n"
            "                elif event.get(\"event\") == \"SOURCE_COMPLETE\":\n"
            "                    row = event.get(\"component\")\n"
            "                    if not isinstance(row, dict) or "
            "not isinstance(row.get(\"task\"), dict):\n"
            "                        raise ArithmeticError(\"invalid resumed source row\")\n"
            "                    key = _driver_key(row[\"task\"])\n"
            "                    if key not in expected or row[\"task\"] != expected[key]:\n"
            "                        raise ArithmeticError(\"resumed source task mismatch\")\n"
            "                    if key in rows and rows[key] != row:\n"
            "                        raise ArithmeticError(\"conflicting resumed source rows\")\n"
            "                    rows[key] = row\n"
            "    if rows and cap is None:\n"
            "        raise ArithmeticError(\"resumed source rows lack a cap receipt\")\n"
            "    return cap, list(rows.values())",
        ),
        (
            "    parser.add_argument(\n"
            "        \"--output\",\n"
            "        type=Path,\n"
            "        default=Path(\"prime_gap_186_fresh.json\"),\n"
            "        help=\"new JSON receipt; an existing file is never overwritten\",\n"
            "    )\n"
            "    args = parser.parse_args()",
            "    parser.add_argument(\n"
            "        \"--output\",\n"
            "        type=Path,\n"
            "        default=Path(\"prime_gap_186_fresh.json\"),\n"
            "        help=\"new JSON receipt; an existing file is never overwritten\",\n"
            "    )\n"
            "    parser.add_argument(\"--resume-log\", type=Path, action=\"append\", "
            "default=[])\n"
            "    parser.add_argument(\"--checkpoint\", type=Path)\n"
            "    args = parser.parse_args()",
        ),
        (
            "    if args.output.exists():\n"
            "        raise FileExistsError(\"refusing to replace an existing receipt\")\n"
            "    _driver_inventory()\n"
            "    started = monotonic()\n"
            "    _driver_emit(\"CAP_START\")\n"
            "    cap = compute_fresh_cap()\n"
            "    gc.collect()  # The serial cap engine is gone before any source "
            "process starts.\n"
            "    denominator = F(cap[\"rounded_units\"][\"I_lower\"], "
            "POLICY[\"cap_decimal_scale\"])\n"
            "    _driver_emit(\"CAP_COMPLETE\", cap=cap, "
            "source_normalization_denominator=str(denominator))\n"
            "    rows = []\n"
            "    # Each process imports this same script normally and starts with "
            "fresh FLINT\n"
            "    # settings. Pool termination also stops remaining work if any task "
            "fails.\n"
            "    with mp.get_context(\"spawn\").Pool(args.workers) as pool:\n"
            "        for record in pool.imap_unordered(compute_source_task, TASKS, "
            "chunksize=1):\n"
            "            row = round_source_component(record, denominator)\n"
            "            rows.append(row)\n"
            "            _driver_emit(\"SOURCE_COMPLETE\", completed=len(rows), "
            "required=len(TASKS), component=row)",
            "    if args.output.exists():\n"
            "        raise FileExistsError(\"refusing to replace an existing receipt\")\n"
            "    if args.checkpoint is not None and args.checkpoint.exists():\n"
            "        raise FileExistsError(\"refusing to replace an existing checkpoint\")\n"
            "    expected = _driver_inventory()\n"
            "    started = monotonic()\n"
            "    cap, rows = _driver_load_resume(args.resume_log)\n"
            "    if cap is None:\n"
            "        _driver_emit(\"CAP_START\")\n"
            "        cap = compute_fresh_cap()\n"
            "        gc.collect()\n"
            "        denominator = F(cap[\"rounded_units\"][\"I_lower\"], "
            "POLICY[\"cap_decimal_scale\"])\n"
            "        _driver_emit(\"CAP_COMPLETE\", cap=cap, "
            "source_normalization_denominator=str(denominator))\n"
            "        _driver_checkpoint(args.checkpoint, \"CAP_COMPLETE\", cap=cap, "
            "source_normalization_denominator=str(denominator))\n"
            "    else:\n"
            "        denominator = F(cap[\"rounded_units\"][\"I_lower\"], "
            "POLICY[\"cap_decimal_scale\"])\n"
            "        _driver_emit(\"CAP_REUSED\", "
            "source_normalization_denominator=str(denominator))\n"
            "        _driver_checkpoint(args.checkpoint, \"CAP_COMPLETE\", cap=cap, "
            "source_normalization_denominator=str(denominator))\n"
            "    completed = {_driver_key(row[\"task\"]) for row in rows}\n"
            "    pending = [task for task in TASKS if _driver_key(task) not in completed]\n"
            "    _driver_emit(\"RESUME_STATE\", completed=len(rows), "
            "remaining=len(pending))\n"
            "    if pending:\n"
            "        with mp.get_context(\"spawn\").Pool(args.workers) as pool:\n"
            "            for record in pool.imap_unordered(compute_source_task, "
            "pending, chunksize=1):\n"
            "                row = round_source_component(record, denominator)\n"
            "                key = _driver_key(row[\"task\"])\n"
            "                if key not in expected or key in completed:\n"
            "                    raise ArithmeticError(\"new source row is invalid\")\n"
            "                rows.append(row)\n"
            "                completed.add(key)\n"
            "                _driver_emit(\"SOURCE_COMPLETE\", completed=len(rows), "
            "required=len(TASKS), component=row)\n"
            "                _driver_checkpoint(args.checkpoint, \"SOURCE_COMPLETE\", "
            "completed=len(rows), required=len(TASKS), component=row)",
        ),
        (
            "def check_flint_signed_fft():\n"
            '    """Reject the known signed-FFT regression; never modify the library."""\n'
            "    a, b = (1 << 509) - 1, (1 << 510) - 1\n"
            "    p, q = fmpz_poly([a] * 16), fmpz_poly([-b] * 16)\n"
            "    expected = [-min(j + 1, 31 - j, 16) * a * b for j in range(31)]\n"
            "    if p * q != fmpz_poly(expected) or p.mul_low(q, 16) != "
            "fmpz_poly(expected[:16]):\n"
            "        raise RuntimeError(\n"
            '            "FLINT failed its signed-FFT regression; install a build '
            'with corrected signed convolution."\n'
            "        )",
            "def signed_arb_poly_times_nonnegative(signed, positive):\n"
            '    """Multiply through nonnegative parts, avoiding defective signed FFT."""\n'
            "    zero = arb(0)\n"
            "    if any(value.lower() < 0 for value in positive.coeffs()):\n"
            '        raise ArithmeticError("second polynomial is not nonnegative")\n'
            "    plus, minus = [], []\n"
            "    for value in signed.coeffs():\n"
            "        low, high = value.lower(), value.upper()\n"
            "        if not value.is_finite() or low > high:\n"
            '            raise ArithmeticError("invalid signed polynomial coefficient")\n'
            "        plus.append(max(zero, low).union(max(zero, high)))\n"
            "        minus.append(max(zero, -high).union(max(zero, -low)))\n"
            "    return arb_poly(plus) * positive - arb_poly(minus) * positive\n"
            "\n"
            "\n"
            "def check_flint_signed_fft():\n"
            '    """Require exact positive convolution and validate the signed split."""\n'
            "    a, b = (1 << 509) - 1, (1 << 510) - 1\n"
            "    p, q = fmpz_poly([a] * 16), fmpz_poly([b] * 16)\n"
            "    positive = [min(j + 1, 31 - j, 16) * a * b for j in range(31)]\n"
            "    if p * q != fmpz_poly(positive) or p.mul_low(q, 16) != "
            "fmpz_poly(positive[:16]):\n"
            '        raise RuntimeError("FLINT failed its nonnegative convolution regression")\n'
            "    signed = signed_arb_poly_times_nonnegative(\n"
            "        arb_poly([arb(a)] * 16), arb_poly([arb(b)] * 16)\n"
            "    )\n"
            "    if len(signed) != len(positive) or any(\n"
            "        not signed[j].contains(arb(value)) "
            "for j, value in enumerate(positive)\n"
            "    ):\n"
            '        raise RuntimeError("nonnegative Arb convolution regression failed")\n'
            "    negative = signed_arb_poly_times_nonnegative(\n"
            "        arb_poly([arb(-a)] * 16), arb_poly([arb(b)] * 16)\n"
            "    )\n"
            "    if len(negative) != len(positive) or any(\n"
            "        not negative[j].contains(arb(-value)) "
            "for j, value in enumerate(positive)\n"
            "    ):\n"
            '        raise RuntimeError("signed split-convolution regression failed")',
        ),
        (
            "correlation = radial * arb_poly(list(reversed(fiber)))",
            "correlation = signed_arb_poly_times_nonnegative(\n"
            "                    radial, arb_poly(list(reversed(fiber)))\n"
            "                )",
        ),
    )
    return rewrites


def parameterize_prime_gap_evaluator_source(
    source: str,
    *,
    dimension: int,
    intervals: int = 98304,
) -> str:
    """Fail-closed rewrite of the pinned evaluator's dimension dependencies."""

    transformed = source
    rewrites = _prime_gap_evaluator_rewrites(
        dimension=dimension,
        intervals=intervals,
    )
    for old, new in rewrites:
        transformed = _replace_exact(transformed, old, new)
    return transformed


def _inventory_proved(
    groups: tuple[PrimeGapSourceGroup, ...],
    tasks: tuple[PrimeGapSourceTask, ...],
) -> bool:
    group_by_id = {group.identifier: group for group in groups}
    if len(groups) != 6 or len(group_by_id) != 6:
        return False
    keys = tuple(task.key for task in tasks)
    if len(keys) != len(set(keys)):
        return False
    for group in groups:
        selected = tuple(task for task in tasks if task.group == group.identifier)
        low = sorted(
            (task for task in selected if task.kind == "low"),
            key=lambda task: task.index,
        )
        rank = sorted(
            (task for task in selected if task.kind == "rank_two"),
            key=lambda task: task.index,
        )
        high = tuple(task for task in selected if task.kind == "high")
        if (
            not low
            or low[0].lower != group.activation
            or low[-1].upper != group.split
            or any(left.upper != right.lower for left, right in zip(low, low[1:], strict=False))
            or not rank
            or rank[0].lower != group.rank_lower
            or rank[-1].upper != group.hard_cap
            or any(left.upper != right.lower for left, right in zip(rank, rank[1:], strict=False))
            or len(high) != 1
            or high[0].index != 0
        ):
            return False
    return True


def build_prime_gap_input_manifest(
    *,
    dimension: int,
    intervals: int = 98304,
) -> PrimeGapInputManifest:
    """Replay the exact source schedule for a chosen sieve dimension."""

    if dimension < 3 or intervals <= dimension + 2:
        raise ValueError("invalid sieve dimension or interval count")
    gap = Fraction(1, 10**7)
    tau = Fraction(1, 10**10)
    rho = Fraction(1, 4) + Fraction(12499, 10**6)
    rho_star = rho - gap
    outer_radius = Fraction(2742997, 10**7) / rho_star
    inner_new = Fraction(251, 1000) / rho_star
    inner_old = 2 - Fraction(3, 1000) - outer_radius
    step = outer_radius / intervals
    e = gap / rho
    global_cap = Fraction(19037, 100000) / rho
    sigma0 = Fraction(100001, 10**6)
    sigma_m = Fraction(1, 2) - Fraction(40481, 100000) + tau
    old_constants = (
        ((1 - 5 * sigma0) / 15, Fraction(18, 5)),
        ((1 - 4 * sigma0) / 16, Fraction(7, 2)),
        (Fraction(3, 80), Fraction(3)),
    )
    new_constants = (
        ((1 - 5 * sigma_m) / 15, Fraction(18, 5)),
        ((1 - 4 * sigma_m) / 16, Fraction(7, 2)),
        ((1 - 2 * sigma_m) / 20, Fraction(16, 5)),
    )
    old_all = _ladder(
        name="old",
        inner_radius=inner_old,
        epsilon=Fraction(1, 10**6),
        limit=Fraction(12499, 10**6),
        constants=old_constants,
        outer_radius=outer_radius,
        rho=rho,
        gap=gap,
    )
    new_all = _ladder(
        name="new",
        inner_radius=inner_new,
        epsilon=Fraction(1, 10**7),
        limit=Fraction(253, 20000),
        constants=new_constants,
        outer_radius=outer_radius,
        rho=rho,
        gap=gap,
    )
    shells = {
        "outer": _shells(
            (
                (new_all[0].a, global_cap),
                (new_all[24].a, (outer_radius + e) / 2),
                (
                    outer_radius,
                    outer_radius
                    + e / 2
                    - 23 * (inner_new + e / 2) / 40,
                ),
            )
        ),
        "base": _shells(
            (
                (old_all[12].b, global_cap),
                (old_all[24].b, (inner_old + e) / 2),
                (inner_old, 63 * (inner_old + e / 2) / 160),
            )
        ),
        "enlarged": _shells(
            (
                (new_all[12].b, global_cap),
                (new_all[24].b, (inner_new + e) / 2),
                (inner_new, 63 * (inner_new + e / 2) / 160),
            )
        ),
        "full": _shells(((outer_radius, global_cap),)),
    }
    convolution_length = intervals - dimension
    cells = {
        name: _cells(
            layers,
            dimension=dimension if name == "outer" else dimension - 1,
            convolution_length=convolution_length,
            step=step,
        )
        for name, layers in shells.items()
    }
    maxima = {
        name: max(cell.physical_upper for cell in rows)
        for name, rows in cells.items()
    }
    old = tuple(
        row
        for row in old_all
        if row.upper_b < maxima["outer"] + maxima["base"]
    )
    new = tuple(
        row
        for row in new_all
        if row.upper_b < maxima["outer"] + maxima["enlarged"]
    )
    groups: list[PrimeGapSourceGroup] = []
    source_specs: tuple[
        tuple[SourceRole, tuple[PrimeGapLadderRow, ...], bool, str],
        ...,
    ] = (
        ("outer", old + new, True, "outer"),
        ("old_inner", old, False, "base"),
        ("new_inner", new, False, "enlarged"),
    )
    for role, rows, outer, cell_name in source_specs:
        h2 = tuple(
            row
            for row in rows
            if row.source_order < 3 and (outer or row.source_order == 2)
        )
        h25 = tuple(row for row in rows if row.source_order == 3)
        c2 = min(row.a for row in h2) if outer else min(row.b for row in h2)
        c25 = min(row.a for row in h25) if outer else min(row.b for row in h25)
        group_dimension = dimension if outer else dimension - 1
        groups.append(
            _group(
                identifier=f"{role}_h2",
                role=role,
                dimension=group_dimension,
                order=Fraction(2),
                rows=h2,
                outer=outer,
                lower=c2,
                upper=c25,
                cells=cells[cell_name],
                step=step,
            )
        )
        groups.append(
            _group(
                identifier=f"{role}_h25",
                role=role,
                dimension=group_dimension,
                order=Fraction(5, 2),
                rows=h25,
                outer=outer,
                lower=c25,
                upper=maxima[cell_name],
                cells=cells[cell_name],
                step=step,
            )
        )
    mass = Fraction(49999, 50000)
    pair_constant = Fraction(17, 50)
    hybrid_lambda = Fraction(1, 125)
    hybrid_a = mass * mass - mass * hybrid_lambda
    hybrid_b = (1 - mass / hybrid_lambda) * (1 - mass) * pair_constant
    tasks = _schedule(
        tuple(groups),
        restoration_old=1 - hybrid_a - hybrid_b,
        restoration_new=1 - hybrid_b,
    )
    group_by_id = {group.identifier: group for group in groups}
    outer_task_count = sum(
        group_by_id[task.group].role == "outer"
        for task in tasks
    )
    inner_task_count = len(tasks) - outer_task_count
    raw_form_count = 2 * outer_task_count + inner_task_count
    inventory_proved = _inventory_proved(tuple(groups), tasks)
    honesty = {
        "unproven_claim": False,
        "exact_input_manifest_replay": inventory_proved,
        "arb_flint_integrals_recomputed": False,
        "source_loss_enclosures_recomputed": False,
        "finite_numerical_crossing_proved": False,
        "bounded_gap_analytic_instantiation_proved": False,
        "h1_182_claim": False,
        "twin_prime_conjecture_proof_claim": False,
    }
    return PrimeGapInputManifest(
        dimension=dimension,
        face_dimension=dimension - 1,
        intervals=intervals,
        convolution_length=convolution_length,
        basis_size=77,
        old_ladder_length=len(old_all),
        new_ladder_length=len(new_all),
        selected_old_length=len(old),
        selected_new_length=len(new),
        groups=tuple(groups),
        tasks=tasks,
        outer_task_count=outer_task_count,
        inner_task_count=inner_task_count,
        raw_form_count=raw_form_count,
        grid_step=step,
        maxima=tuple(sorted(maxima.items())),
        inventory_proved=inventory_proved,
        upstream_commit=PRIME_GAPS_186_COMMIT,
        status="PROVED" if inventory_proved else "BLOCKED",
        honesty=honesty,
    )


def seal_prime_gap_input_manifest(
    report: PrimeGapInputManifest,
) -> dict[str, object]:
    """Seal the finite rational manifest, never the numerical crossing."""

    return make_certificate(
        claim=PRIME_GAPS_INPUT_MANIFEST_KIND,
        payload=report.to_payload(),
        honesty=dict(report.honesty),
        meta={"upstream_commit": report.upstream_commit},
    )


__all__ = [
    "ExactCoefficientInterval",
    "ExactSymmetricIntervalMatrix",
    "PRIME_GAPS_186_COMMIT",
    "PRIME_GAPS_186_SOURCE_SHA256",
    "PRIME_GAPS_186_SOURCE_URL",
    "PRIME_GAPS_INPUT_MANIFEST_KIND",
    "PRIME_GAPS_NUMERICAL_RECEIPT_KIND",
    "PRIME_GAPS_QUADRATIC_RECEIPT_KIND",
    "PRIME_GAP_COEFFICIENT_SIGNATURES",
    "PRIME_GAP_QUADRATIC_MATRIX_NAMES",
    "PrimeGapCell",
    "PrimeGapCoefficientDescriptor",
    "PrimeGapEngineLayout",
    "PrimeGapInputManifest",
    "PrimeGapLadderRow",
    "PrimeGapNumericalReceiptCertificate",
    "PrimeGapQuadraticCandidateCertificate",
    "PrimeGapQuadraticKernelSpec",
    "PrimeGapQuadraticReceiptCertificate",
    "PrimeGapRoundingReserveCertificate",
    "PrimeGapShell",
    "PrimeGapSourceGroup",
    "PrimeGapSourceTask",
    "SignedConvolutionSplitCertificate",
    "build_prime_gap_input_manifest",
    "certify_prime_gap_numerical_receipt",
    "certify_prime_gap_quadratic_candidate",
    "certify_prime_gap_quadratic_receipt",
    "certify_prime_gap_rounding_reserve",
    "certify_signed_convolution_split",
    "parameterize_prime_gap_evaluator_source",
    "prime_gap_coefficient_descriptors",
    "prime_gap_engine_layout",
    "prime_gap_quadratic_kernel_spec",
    "seal_prime_gap_input_manifest",
    "seal_prime_gap_numerical_receipt",
    "seal_prime_gap_quadratic_receipt",
]

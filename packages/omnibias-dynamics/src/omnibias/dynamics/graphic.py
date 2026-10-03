# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Named local graphic models and honest Dulac-model cyclicity verdicts.

The rational, resonant, and interval-ratio targets bind an exact quadratic
field, a checked hyperbolic saddle, a transversal source equation, and a
declared finite Dulac model. Their proved status applies only to that finite
model. The saddle-node-at-infinity target returns ``BLOCKED`` with the existing
coalescing-root and central-capture obstructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.bautin import BautinBasis
from omnibias.dynamics.compactify import (
    PlanarPolynomialField,
    PoincareCompactificationCertificate,
    certify_poincare_compactification,
    verify_poincare_compactification,
)
from omnibias.dynamics.dulac import (
    DulacCyclicityCertificate,
    DulacExpansion,
    DulacTerm,
    DulacUniformCoverCertificate,
    RationalInterval,
    certify_dulac_cyclicity,
    certify_dulac_uniform_cover,
    displacement_expansion,
    verify_dulac_cyclicity,
    verify_dulac_uniform_cover,
)
from omnibias.dynamics.focal import FocalCertificate, verify_focal_values

# omnibias.dynamics.saddle_normal_form imports HyperbolicSaddle from this
# module, so a top-level import back here would be circular; the types are
# only needed for annotations (resolved lazily under
# ``from __future__ import annotations``) and the verify call is deferred
# into the one function that needs it.
if TYPE_CHECKING:
    from omnibias.dynamics.saddle_normal_form import (
        DerivedDulacExpansion,
        ResonantNormalFormCertificate,
    )

GraphicKind = Literal[
    "hyperbolic_rational",
    "hyperbolic_resonant",
    "hyperbolic_irrational_enclosure",
    "saddle_node_infinity_open",
]
GraphicStatus = Literal["PROVED_MODEL", "BLOCKED"]


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True)
class HyperbolicSaddle:
    """An exact equilibrium with rigorously enclosed real eigenvalues."""

    point: tuple[Fraction, Fraction]
    stable: RationalInterval
    unstable: RationalInterval
    ratio: RationalInterval

    def verifies(self, field: PlanarPolynomialField) -> bool:
        x0, y0 = self.point
        if field.p.evaluate((x0, y0)) or field.q.evaluate((x0, y0)):
            return False
        p_x = field.p.derivative(0).evaluate((x0, y0))
        p_y = field.p.derivative(1).evaluate((x0, y0))
        q_x = field.q.derivative(0).evaluate((x0, y0))
        q_y = field.q.derivative(1).evaluate((x0, y0))
        trace = p_x + q_y
        determinant = p_x * q_y - p_y * q_x
        stable = self.stable.to_interval()
        unstable = self.unstable.to_interval()
        trace_iv = Interval.from_rational(trace)
        determinant_iv = Interval.from_rational(determinant)

        def characteristic(value: Interval) -> Interval:
            return value * value - trace_iv * value + determinant_iv

        if (
            self.stable.hi >= 0
            or self.unstable.lo <= 0
            or not characteristic(stable).contains_zero()
            or not characteristic(unstable).contains_zero()
            or (2 * stable - trace_iv).contains_zero()
            or (2 * unstable - trace_iv).contains_zero()
        ):
            return False
        ratio = (-stable) / unstable
        return (
            max(ratio.lo, self.ratio.to_interval().lo)
            <= min(ratio.hi, self.ratio.to_interval().hi)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "point": [_q(value) for value in self.point],
            "stable": self.stable.to_payload(),
            "unstable": self.unstable.to_payload(),
            "ratio": self.ratio.to_payload(),
        }


CollarStatus = Literal["PROVED_COLLAR", "BLOCKED"]


@dataclass(frozen=True)
class GraphicTarget:
    """A declared local target, section, itinerary, and finite return model.

    Five optional fields let a caller attach genuinely computed evidence
    from the newer engines without changing the declared ``return_map``
    itself: ``focal`` (a verified
    :class:`~omnibias.dynamics.focal.FocalCertificate` for a monodromic
    point relevant to this graphic), ``bautin`` (a computed
    :class:`~omnibias.dynamics.bautin.BautinBasis`), ``resonant_normal_form``
    (a verified :class:`~omnibias.dynamics.saddle_normal_form.ResonantNormalFormCertificate`
    for *this* field and saddle -- cross-checked, not merely accepted),
    ``derived_corner`` (the matching
    :class:`~omnibias.dynamics.saddle_normal_form.DerivedDulacExpansion`),
    and ``collar_status`` (the outcome of a
    :func:`~omnibias.dynamics.membership.certify_collar_membership` run,
    stored as a bare status rather than the full certificate to avoid a
    self-referential ``target`` field). Attaching any of these flips the
    matching honesty flag in :func:`certify_graphic_cyclicity`; none of them
    change ``status`` or ``upper_bound``, which remain governed solely by
    the declared ``return_map``.
    """

    name: str
    kind: GraphicKind
    field: PlanarPolynomialField
    saddle: HyperbolicSaddle | None
    section: SparsePolynomial
    itinerary: tuple[str, ...]
    return_map: DulacExpansion
    obstruction: tuple[str, ...] = ()
    focal: FocalCertificate | None = None
    bautin: BautinBasis | None = None
    resonant_normal_form: ResonantNormalFormCertificate | None = None
    derived_corner: DerivedDulacExpansion | None = None
    collar_status: CollarStatus | None = None

    def __post_init__(self) -> None:
        if self.section.nvars != 2:
            raise ValueError("the transversal section must be bivariate")
        if not self.name or not self.itinerary:
            raise ValueError("a graphic target needs a name and itinerary")
        if self.kind == "saddle_node_infinity_open":
            if self.saddle is not None or not self.obstruction:
                raise ValueError("the open infinity target needs obstructions and no hyperbolic saddle")
        elif self.saddle is None or not self.saddle.verifies(self.field):
            raise ValueError("the declared hyperbolic saddle did not verify")
        if self.focal is not None and not verify_focal_values(self.focal):
            raise ValueError("the attached focal certificate did not verify")
        if self.resonant_normal_form is not None:
            from omnibias.dynamics.saddle_normal_form import verify_resonant_normal_form

            if not verify_resonant_normal_form(self.resonant_normal_form):
                raise ValueError("the attached resonant normal form certificate did not verify")
            if self.resonant_normal_form.field.digest != self.field.digest:
                raise ValueError("the attached resonant normal form is for a different field")
            if self.saddle is None or self.resonant_normal_form.saddle != self.saddle:
                raise ValueError("the attached resonant normal form is for a different saddle")
        if self.derived_corner is not None:
            if self.resonant_normal_form is None:
                raise ValueError("derived_corner requires a matching resonant_normal_form")
            if self.derived_corner.normal_form != self.resonant_normal_form.normal_form:
                raise ValueError("derived_corner does not match the attached resonant normal form")
        if self.collar_status is not None and self.collar_status not in ("PROVED_COLLAR", "BLOCKED"):
            raise ValueError("collar_status must be PROVED_COLLAR or BLOCKED")

    @property
    def digest(self) -> str:
        return source_digest(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "field": self.field.to_payload(),
            "saddle": None if self.saddle is None else self.saddle.to_payload(),
            "section": self.section.to_payload(),
            "itinerary": list(self.itinerary),
            "return_map": self.return_map.to_payload(),
            "obstruction": list(self.obstruction),
            "focal_attached": self.focal is not None,
            "bautin_attached": self.bautin is not None,
            "resonant_normal_form_attached": self.resonant_normal_form is not None,
            "derived_corner_attached": self.derived_corner is not None,
            "collar_status": self.collar_status,
        }


def _variables() -> tuple[SparsePolynomial, SparsePolynomial]:
    return SparsePolynomial.variable(2, 0), SparsePolynomial.variable(2, 1)


def named_rational_hyperbolic_graphic() -> GraphicTarget:
    """A checked rational-ratio saddle with a declared three-term model."""
    x, y = _variables()
    field = PlanarPolynomialField(-2 * x + x**2, y + y**2, 2)
    saddle = HyperbolicSaddle(
        (Fraction(0), Fraction(0)),
        RationalInterval.create(-2),
        RationalInterval.create(1),
        RationalInterval.create(2),
    )
    return_map = DulacExpansion.create(
        (
            (1, 0, 1),
            (2, 0, 2),
            (Fraction(5, 2), 0, 3),
            (3, 0, -1),
        ),
        truncation_order=3,
        remainder_bound=Fraction(1, 1000),
    )
    return GraphicTarget(
        "rational-hyperbolic-local-model",
        "hyperbolic_rational",
        field,
        saddle,
        x - Fraction(1, 10),
        ("incoming_section", "hyperbolic_saddle_chart", "outgoing_section"),
        return_map,
    )


def named_resonant_homoclinic_graphic() -> GraphicTarget:
    """The exact Duffing homoclinic saddle with a resonant declared model."""
    x, y = _variables()
    field = PlanarPolynomialField(y, x - x**2, 2)
    saddle = HyperbolicSaddle(
        (Fraction(0), Fraction(0)),
        RationalInterval.create(-1),
        RationalInterval.create(1),
        RationalInterval.create(1),
    )
    return_map = DulacExpansion.create(
        (
            (1, 0, 1),
            (2, 1, 3),
            (2, 0, 2),
            (3, 0, Fraction(1, 2)),
        ),
        truncation_order=3,
        remainder_bound=Fraction(1, 1000),
    )
    return GraphicTarget(
        "resonant-duffing-homoclinic-model",
        "hyperbolic_resonant",
        field,
        saddle,
        x - Fraction(1, 10),
        ("incoming_section", "duffing_saddle", "homoclinic_arc", "outgoing_section"),
        return_map,
    )


def named_irrational_hyperbolic_graphic() -> GraphicTarget:
    """A rational field whose irrational eigenvalue ratio is enclosed exactly."""
    x, y = _variables()
    field = PlanarPolynomialField(y + x**2, x + y + y**2, 2)
    saddle = HyperbolicSaddle(
        (Fraction(0), Fraction(0)),
        RationalInterval.create((Fraction(-13, 20), Fraction(-3, 5))),
        RationalInterval.create((Fraction(8, 5), Fraction(33, 20))),
        RationalInterval.create((Fraction(3, 8), Fraction(2, 5))),
    )
    return_map = DulacExpansion.create(
        (
            (1, 0, 1),
            ((Fraction(3, 8), Fraction(2, 5)), 0, (1, 2)),
            ((Fraction(11, 8), Fraction(7, 5)), 1, (-3, -2)),
            ((Fraction(19, 8), Fraction(12, 5)), 0, (1, 2)),
        ),
        truncation_order=3,
        remainder_bound=Fraction(1, 1000),
    )
    return GraphicTarget(
        "irrational-ratio-hyperbolic-local-model",
        "hyperbolic_irrational_enclosure",
        field,
        saddle,
        x - Fraction(1, 10),
        ("incoming_section", "interval_ratio_saddle_chart", "outgoing_section"),
        return_map,
    )


def named_open_saddle_node_infinity_graphic() -> GraphicTarget:
    """The open coalescing-root/central-capture target from the H16 program."""
    x, y = _variables()
    field = PlanarPolynomialField(x - y + x**2, 2 * x + x**2 + x * y, 2)
    return_map = DulacExpansion.create(
        ((1, 0, 1), (2, 0, 1), (3, 1, -1)),
        truncation_order=3,
        remainder_bound=1,
    )
    return GraphicTarget(
        "open-saddle-node-at-infinity",
        "saddle_node_infinity_open",
        field,
        None,
        x - Fraction(1, 10),
        (
            "incoming_outer_chart",
            "coalescing_root_chart",
            "central_capture",
            "outgoing_outer_chart",
        ),
        return_map,
        (
            "G1 remains open after the lambda=0 Cauchy majorant; fold (L, lambda1) compact, sep=exp kill continuation, and outgoing first-hit stay open",
            "the L=1/n shrinking-root section loses a positive transversality radius",
            "G4 complete physical itinerary capture has not been opened",
        ),
    )


@dataclass(frozen=True)
class GraphicCyclicityCertificate:
    """A model-level bound or an explicit unresolved physical obstruction."""

    target: GraphicTarget
    compactification: PoincareCompactificationCertificate
    status: GraphicStatus
    upper_bound: int | None
    leading_terms: tuple[DulacTerm, ...]
    rational: DulacCyclicityCertificate | None
    uniform: DulacUniformCoverCertificate | None
    obstruction: tuple[str, ...]
    source_digest: str
    seal: Cert


def certify_graphic_cyclicity(
    target: GraphicTarget,
) -> GraphicCyclicityCertificate:
    """Certify the declared finite model or preserve the open obstruction."""
    compactification = certify_poincare_compactification(target.field)
    displacement = displacement_expansion(target.return_map)
    rational_certificate: DulacCyclicityCertificate | None = None
    uniform_certificate: DulacUniformCoverCertificate | None = None
    status: GraphicStatus
    upper_bound: int | None
    obstruction = target.obstruction
    if target.kind == "saddle_node_infinity_open":
        status, upper_bound = "BLOCKED", None
    elif displacement.exact:
        rational_certificate = certify_dulac_cyclicity(displacement)
        status, upper_bound = "PROVED_MODEL", rational_certificate.upper_bound
    else:
        uniform_certificate = certify_dulac_uniform_cover(displacement)
        if uniform_certificate.status == "PROVED":
            status, upper_bound = "PROVED_MODEL", uniform_certificate.uniform_bound
        else:
            status, upper_bound = "BLOCKED", None
            obstruction = (
                *obstruction,
                "interval-ratio derivative-sign cover remained blocked",
            )
    seal = make_certificate(
        claim=(
            "Finite declared Dulac-model zero bound."
            if status == "PROVED_MODEL"
            else "Explicit unresolved graphic obstruction record."
        ),
        payload={
            "type": "dulac_graphic_model",
            "source_digest": target.digest,
            "status": status,
            "upper_bound": upper_bound,
            "leading_terms": [term.to_payload() for term in displacement.leading_terms],
            "obstruction": list(obstruction),
        },
        honesty={
            "poincare_compactification_exact_q": True,
            "dulac_truncated_model_only": True,
            "physical_return_membership_proved": False,
            "uniform_remainder_proved": False,
            "focal_values_computed": target.focal is not None,
            "bautin_ideal_stabilization_proved": (
                target.bautin is not None and target.bautin.bautin_ideal_stabilization_proved
            ),
            "collar_return_membership_proved": target.collar_status == "PROVED_COLLAR",
            "graphic_presence_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "drr_case_closed": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return GraphicCyclicityCertificate(
        target,
        compactification,
        status,
        upper_bound,
        displacement.leading_terms,
        rational_certificate,
        uniform_certificate,
        obstruction,
        target.digest,
        seal,
    )


def verify_graphic_cyclicity(certificate: GraphicCyclicityCertificate) -> bool:
    """Replay compactification, finite model, and the honesty boundary."""
    if (
        certificate.source_digest != certificate.target.digest
        or not verify_certificate_digest(certificate.seal)
        or not verify_poincare_compactification(certificate.compactification)
    ):
        return False
    if certificate.rational is not None and not verify_dulac_cyclicity(
        certificate.rational
    ):
        return False
    if certificate.uniform is not None and not verify_dulac_uniform_cover(
        certificate.uniform
    ):
        return False
    try:
        expected = certify_graphic_cyclicity(certificate.target)
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return False
    return expected == certificate


__all__ = [
    "GraphicCyclicityCertificate",
    "GraphicKind",
    "GraphicStatus",
    "GraphicTarget",
    "HyperbolicSaddle",
    "certify_graphic_cyclicity",
    "named_irrational_hyperbolic_graphic",
    "named_open_saddle_node_infinity_graphic",
    "named_rational_hyperbolic_graphic",
    "named_resonant_homoclinic_graphic",
    "verify_graphic_cyclicity",
]

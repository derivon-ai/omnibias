# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Named graphic-model verdict and honesty regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.dynamics.bautin import bautin_basis, nonzero_focus_example
from omnibias.dynamics.compactify import PlanarPolynomialField
from omnibias.dynamics.focal import certify_focal_values, lyapunov_quantities
from omnibias.dynamics.graphic import (
    certify_graphic_cyclicity,
    named_irrational_hyperbolic_graphic,
    named_open_saddle_node_infinity_graphic,
    named_rational_hyperbolic_graphic,
    named_resonant_homoclinic_graphic,
    verify_graphic_cyclicity,
)
from omnibias.dynamics.saddle_normal_form import (
    certify_resonant_normal_form,
    diagonalize_saddle,
    dulac_corner_expansion,
    resonant_normal_form,
)


@pytest.mark.parametrize(
    ("factory", "bound"),
    (
        (named_rational_hyperbolic_graphic, 2),
        (named_resonant_homoclinic_graphic, 2),
        (named_irrational_hyperbolic_graphic, 0),
    ),
)
def test_named_finite_models_have_replayable_bounds(factory, bound: int) -> None:
    target = factory()
    assert target.saddle is not None and target.saddle.verifies(target.field)
    certificate = certify_graphic_cyclicity(target)
    assert certificate.status == "PROVED_MODEL"
    assert certificate.upper_bound == bound
    assert len(certificate.leading_terms) == 3
    assert verify_graphic_cyclicity(certificate)
    honesty = certificate.seal["honesty"]
    assert honesty["dulac_truncated_model_only"]
    assert honesty["physical_return_membership_proved"] is False
    assert honesty["uniform_remainder_proved"] is False
    assert honesty["graphic_finite_cyclicity_proved"] is False
    assert honesty["drr_case_closed"] is False
    assert honesty["full_hilbert16_solved"] is False


def test_resonant_named_field_has_the_exact_duffing_homoclinic() -> None:
    target = named_resonant_homoclinic_graphic()
    x, y = (P.variable(2, axis) for axis in range(2))
    hamiltonian = y**2 * Q(1, 2) - x**2 * Q(1, 2) + x**3 * Q(1, 3)
    derivative = (
        hamiltonian.derivative(0) * target.field.p
        + hamiltonian.derivative(1) * target.field.q
    )
    assert not derivative.terms
    # H=0 gives y^2=x^2-(2/3)x^3, a loop from the saddle to x=3/2.
    assert hamiltonian.evaluate((Q(3, 2), 0)) == 0
    certificate = certify_graphic_cyclicity(target)
    assert certificate.seal["honesty"]["graphic_presence_proved"] is False


def test_open_saddle_node_target_stays_blocked_with_named_obstructions() -> None:
    target = named_open_saddle_node_infinity_graphic()
    certificate = certify_graphic_cyclicity(target)
    assert certificate.status == "BLOCKED"
    assert certificate.upper_bound is None
    assert len(certificate.obstruction) == 3
    assert any("sep=exp" in reason for reason in certificate.obstruction)
    assert any("G4" in reason for reason in certificate.obstruction)
    assert certificate.seal["honesty"]["graphic_finite_cyclicity_proved"] is False
    assert verify_graphic_cyclicity(certificate)


def test_graphic_tamper_and_invalid_saddle_are_rejected() -> None:
    certificate = certify_graphic_cyclicity(named_rational_hyperbolic_graphic())
    assert not verify_graphic_cyclicity(
        replace(certificate, source_digest="tampered")
    )
    altered = deepcopy(certificate.seal)
    altered["payload"]["upper_bound"] = 0
    altered = seal_certificate(altered)
    assert not verify_graphic_cyclicity(replace(certificate, seal=altered))
    target = named_rational_hyperbolic_graphic()
    assert target.saddle is not None
    with pytest.raises(ValueError, match="saddle"):
        replace(
            target,
            saddle=replace(
                target.saddle,
                point=(Q(1), Q(0)),
            ),
        )


def _numeric_focal_certificate():
    x, y = (P.variable(2, axis) for axis in range(2))
    p_pert, q_pert = nonzero_focus_example()
    p = -y + P(2, dict(p_pert.terms))
    q = x + P(2, dict(q_pert.terms))
    field = PlanarPolynomialField(p, q, 3)
    return certify_focal_values(field, order=4)


def test_attaching_a_focal_certificate_flips_focal_values_computed() -> None:
    target = named_rational_hyperbolic_graphic()
    focal_cert = _numeric_focal_certificate()
    with_focal = replace(target, focal=focal_cert)
    certificate = certify_graphic_cyclicity(with_focal)
    assert certificate.seal["honesty"]["focal_values_computed"] is True
    assert verify_graphic_cyclicity(certificate)
    # An unattached target still reports honestly False.
    baseline_certificate = certify_graphic_cyclicity(target)
    assert baseline_certificate.seal["honesty"]["focal_values_computed"] is False


def test_attaching_an_unverifiable_focal_certificate_is_rejected() -> None:
    target = named_rational_hyperbolic_graphic()
    focal_cert = _numeric_focal_certificate()
    tampered = replace(focal_cert, source_digest="tampered")
    with pytest.raises(ValueError, match="focal certificate"):
        replace(target, focal=tampered)


def test_attaching_bautin_basis_reports_stabilization_status_honestly() -> None:
    from omnibias.dynamics.bautin import bautin_quadratic_family

    target = named_rational_hyperbolic_graphic()
    p, q = bautin_quadratic_family()
    quantities = lyapunov_quantities(p, q, order=6)
    basis = bautin_basis(quantities)
    with_bautin = replace(target, bautin=basis)
    certificate = certify_graphic_cyclicity(with_bautin)
    # bautin_ideal_stabilization_proved is always False by design (see
    # omnibias.dynamics.bautin's honesty note): stabilization is never
    # provable from a Groebner basis of a truncated generator list alone.
    assert certificate.seal["honesty"]["bautin_ideal_stabilization_proved"] is False
    assert verify_graphic_cyclicity(certificate)


def test_attaching_a_matching_resonant_normal_form_and_corner_wires_through() -> None:
    target = named_rational_hyperbolic_graphic()
    nf_cert = certify_resonant_normal_form(target.field, target.saddle, order=3)
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)
    wired = replace(target, resonant_normal_form=nf_cert, derived_corner=derived)
    certificate = certify_graphic_cyclicity(wired)
    assert verify_graphic_cyclicity(certificate)


def test_attaching_a_mismatched_resonant_normal_form_is_rejected() -> None:
    target = named_rational_hyperbolic_graphic()
    other = named_resonant_homoclinic_graphic()
    nf_cert_for_other = certify_resonant_normal_form(other.field, other.saddle, order=3)
    with pytest.raises(ValueError, match="different field"):
        replace(target, resonant_normal_form=nf_cert_for_other)


def test_attaching_derived_corner_without_a_matching_normal_form_is_rejected() -> None:
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)
    with pytest.raises(ValueError, match="resonant_normal_form"):
        replace(target, derived_corner=derived)


def test_attaching_a_proved_collar_status_flips_the_honesty_flag() -> None:
    target = named_rational_hyperbolic_graphic()
    with_collar = replace(target, collar_status="PROVED_COLLAR")
    certificate = certify_graphic_cyclicity(with_collar)
    assert certificate.seal["honesty"]["collar_return_membership_proved"] is True
    assert verify_graphic_cyclicity(certificate)
    blocked = replace(target, collar_status="BLOCKED")
    blocked_certificate = certify_graphic_cyclicity(blocked)
    assert blocked_certificate.seal["honesty"]["collar_return_membership_proved"] is False


def test_invalid_collar_status_is_rejected() -> None:
    target = named_rational_hyperbolic_graphic()
    with pytest.raises(ValueError, match="collar_status"):
        replace(target, collar_status="MAYBE")  # type: ignore[arg-type]

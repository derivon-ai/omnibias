# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The certified gap must never exceed the true gap, and should sandwich it.

Soundness first: ``certified_transfer_matrix_gap`` returns a **lower** bound, so
the contract is ``certified <= true`` on every construction whose spectrum is known
in closed form -- and all of them are.  ``certified_effective_mass_curve`` returns
**upper** bounds, so the pair must bracket the truth.

Beyond soundness these tests pin down *how tight* the bound is, because "sound"
alone permits a useless answer.  With the closed-form eigenvectors each
construction supplies the sandwich closes to about ``1e-13``; strip the partner
chain and the ``SU(2)`` Wilson bound at ``beta = 8`` collapses to vacuous, which is
exactly the polluting tail the deflation exists to remove, and which powering
``T -> T^n`` then recovers.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.geometry.gauge.transfer.gap import (
    BIRKHOFF_METHOD,
    SYMMETRIC_METHOD,
    certified_effective_mass_curve,
    certified_multistep_gap_refinement,
    certified_transfer_matrix_gap,
    heat_kernel_gap_scaling_report,
)
from omnibias.geometry.gauge.transfer.matrices import (
    TransferMatrix,
    su2_heat_kernel_transfer,
    su2_wilson_transfer,
    su3_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)

COUPLING = 0.8


def _exact_gap(transfer: TransferMatrix) -> float:
    ratio = transfer.exact_subdominant_ratio()
    assert ratio is not None
    return -math.log(0.5 * (ratio.lo + ratio.hi))


def _cases() -> list[TransferMatrix]:
    return [
        u1_heat_kernel_transfer(COUPLING, n_max=3),
        u1_heat_kernel_transfer(COUPLING, n_max=3, basis="angle"),
        su2_heat_kernel_transfer(COUPLING, max_dynkin=4),
        su3_heat_kernel_transfer(COUPLING, max_dynkin=2),
        su2_wilson_transfer(2.0, n_modes=6),
        su2_wilson_transfer(5.0, n_modes=8),
    ]


def _without_partners(transfer: TransferMatrix) -> TransferMatrix:
    return TransferMatrix(
        model=transfer.model,
        basis=transfer.basis,
        entries=transfer.entries,
        mode_labels=transfer.mode_labels,
        exact_eigenvalues=transfer.exact_eigenvalues,
        parameters=transfer.parameters,
        perron_vector=transfer.perron_vector,
        subdominant_vectors=(),
        symmetric=transfer.symmetric,
    )


def _jitter(transfer: TransferMatrix, sigma: float, seed: int = 0) -> TransferMatrix:
    rng = random.Random(seed)

    def shake(vector: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(x + rng.gauss(0.0, sigma) for x in vector)

    return TransferMatrix(
        model=transfer.model,
        basis=transfer.basis,
        entries=transfer.entries,
        mode_labels=transfer.mode_labels,
        exact_eigenvalues=transfer.exact_eigenvalues,
        parameters=transfer.parameters,
        perron_vector=shake(transfer.perron_vector),
        subdominant_vectors=tuple(shake(v) for v in transfer.subdominant_vectors),
        symmetric=transfer.symmetric,
    )


# --------------------------------------------------------------------------- #
# soundness: the certified bound never exceeds the truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_certified_bound_never_exceeds_the_true_gap(
    transfer: TransferMatrix,
) -> None:
    result = certified_transfer_matrix_gap(transfer)
    assert result.certified
    assert result.spectral_gap_lower <= _exact_gap(transfer) + 1e-12


@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_every_candidate_is_independently_a_valid_lower_bound(
    transfer: TransferMatrix,
) -> None:
    """Taking the max over candidates is only sound if each one is sound alone."""
    truth = _exact_gap(transfer)
    result = certified_transfer_matrix_gap(transfer)
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.spectral_gap_lower <= truth + 1e-12


@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_a_perturbed_test_vector_stays_sound(transfer: TransferMatrix) -> None:
    """Soundness must not depend on the quality of the test vectors, only tightness."""
    truth = _exact_gap(transfer)
    for sigma in (1e-3, 1e-1, 1.0):
        result = certified_transfer_matrix_gap(_jitter(transfer, sigma))
        assert result.spectral_gap_lower <= truth + 1e-12


@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_dropping_the_partner_chain_stays_sound(transfer: TransferMatrix) -> None:
    result = certified_transfer_matrix_gap(transfer, deflate=False)
    assert result.spectral_gap_lower <= _exact_gap(transfer) + 1e-12


# --------------------------------------------------------------------------- #
# tightness: the sandwich, and what happens without deflation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_lower_and_upper_bounds_sandwich_the_true_gap(
    transfer: TransferMatrix,
) -> None:
    lower = certified_transfer_matrix_gap(transfer).spectral_gap_lower
    upper = certified_effective_mass_curve(transfer, taus=(32, 64, 128)).gap_upper
    truth = _exact_gap(transfer)
    assert lower <= truth <= upper + 1e-12
    assert upper - lower < 1e-9, "the certified sandwich should close to ~1e-13"


@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_supplied_eigenvectors_make_the_bound_essentially_exact(
    transfer: TransferMatrix,
) -> None:
    """Each construction hands the engine its exact eigenvectors, so the chain closes.

    Worth stating explicitly: the tightness below is a property of these
    *constructions*, not of the engine in general. Feed it worse vectors and the
    bound loosens (while staying sound) -- which the perturbation tests cover.
    """
    result = certified_transfer_matrix_gap(transfer)
    assert result.spectral_gap_lower == pytest.approx(_exact_gap(transfer), rel=1e-9)


def test_without_deflation_a_heavy_bessel_tail_makes_the_bound_vacuous() -> None:
    """The concrete case the partner chain exists for, measured rather than asserted.

    At ``beta = 8`` the tail behind the subdominant Wilson mode is heavy enough that
    an undeflated power-sum bound certifies *nothing*, while the same matrix with
    the chain recovers the exact gap.
    """
    transfer = su2_wilson_transfer(8.0, n_modes=10)
    bare = certified_transfer_matrix_gap(_without_partners(transfer))
    deflated = certified_transfer_matrix_gap(transfer)
    assert bare.spectral_gap_lower == 0.0
    assert deflated.spectral_gap_lower == pytest.approx(_exact_gap(transfer), rel=1e-9)
    assert deflated.partners_deflated >= 8


def test_birkhoff_hopf_is_sound_but_visibly_conservative() -> None:
    """It needs no spectral input at all, and pays for that in tightness."""
    transfer = u1_heat_kernel_transfer(COUPLING, n_max=3, basis="angle")
    result = certified_transfer_matrix_gap(transfer)
    birkhoff = next(c for c in result.candidates if c.method == BIRKHOFF_METHOD)
    truth = _exact_gap(transfer)
    assert 0.0 < birkhoff.spectral_gap_lower <= truth
    assert birkhoff.spectral_gap_lower < 0.5 * truth  # ~30% of truth at t = 0.8
    # The symmetric engine wins, and the dispatcher picks it.
    assert result.method == SYMMETRIC_METHOD


def test_birkhoff_hopf_only_runs_where_it_applies() -> None:
    """A diagonal matrix has zero entries, so the positivity precondition fails."""
    result = certified_transfer_matrix_gap(u1_heat_kernel_transfer(COUPLING, n_max=3))
    assert all(c.method != BIRKHOFF_METHOD for c in result.candidates)


# --------------------------------------------------------------------------- #
# multistep refinement
# --------------------------------------------------------------------------- #
def test_powering_recovers_a_bound_the_bare_engine_could_not_reach() -> None:
    r"""``T^n`` shrinks the tail's relative pollution geometrically."""
    transfer = _without_partners(su2_wilson_transfer(5.0, n_modes=8))
    truth = _exact_gap(su2_wilson_transfer(5.0, n_modes=8))
    single = certified_transfer_matrix_gap(transfer).spectral_gap_lower
    refined = certified_multistep_gap_refinement(transfer, max_power=8)
    assert single < 0.6 * truth
    assert refined.spectral_gap_lower > 0.99 * truth
    assert refined.best_power > 1
    assert refined.spectral_gap_lower <= truth + 1e-12


def test_the_multistep_curve_is_reported_per_power() -> None:
    transfer = _without_partners(su2_wilson_transfer(5.0, n_modes=8))
    refined = certified_multistep_gap_refinement(transfer, max_power=4)
    powers = [p for p, _ in refined.per_power]
    assert powers == [1, 2, 3, 4]
    truth = _exact_gap(su2_wilson_transfer(5.0, n_modes=8))
    assert all(gap <= truth + 1e-12 for _, gap in refined.per_power)


def test_multistep_rejects_a_meaningless_power() -> None:
    with pytest.raises(ValueError, match="max_power must be >= 1"):
        certified_multistep_gap_refinement(su2_wilson_transfer(2.0), max_power=0)


# --------------------------------------------------------------------------- #
# effective-mass curve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transfer", _cases(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_effective_mass_falls_monotonically_toward_the_gap(
    transfer: TransferMatrix,
) -> None:
    curve = certified_effective_mass_curve(transfer, taus=(1, 2, 4, 8, 16, 32))
    uppers = [p.upper for p in curve.points]
    assert len(uppers) >= 4
    assert all(b <= a + 1e-12 for a, b in zip(uppers, uppers[1:], strict=False))
    truth = _exact_gap(transfer)
    # Each interval encloses m_eff(tau), which *exceeds* the gap rather than
    # equalling it, so the guarantee is on the upper endpoint only.
    assert all(p.upper >= truth - 1e-12 for p in curve.points)
    assert curve.points[-1].upper == pytest.approx(truth, rel=1e-6)


def test_the_effective_mass_needs_a_closed_form_spectrum() -> None:
    transfer = su2_wilson_transfer(2.0, n_modes=4)
    blind = TransferMatrix(
        model=transfer.model,
        basis=transfer.basis,
        entries=transfer.entries,
        mode_labels=transfer.mode_labels,
        exact_eigenvalues=None,
        parameters=transfer.parameters,
        perron_vector=transfer.perron_vector,
        subdominant_vectors=transfer.subdominant_vectors,
    )
    with pytest.raises(ValueError, match="closed-form spectrum"):
        certified_effective_mass_curve(blind)


def test_a_negative_tau_is_refused() -> None:
    with pytest.raises(ValueError, match="tau must be >= 0"):
        certified_effective_mass_curve(su2_wilson_transfer(2.0), taus=(-1,))


# --------------------------------------------------------------------------- #
# the scaling report is evidence, never a limit
# --------------------------------------------------------------------------- #
def test_the_scaling_report_never_claims_a_continuum_limit() -> None:
    report = heat_kernel_gap_scaling_report(
        su2_heat_kernel_transfer,
        spacings=[1.0, 0.5, 0.25],
        couplings=[0.8, 0.4, 0.2],
        max_dynkin=4,
    )
    assert report.continuum_claim is False
    assert "NOT a continuum-limit" in report.note
    assert len(report.points) == 3
    assert report.model == "su2_heat_kernel"


def test_the_scaling_report_holds_the_physical_gap_fixed_when_the_coupling_tracks_the_spacing() -> None:
    """With ``t`` proportional to ``a`` the per-unit gap is constant -- a fact, not a limit."""
    report = heat_kernel_gap_scaling_report(
        su2_heat_kernel_transfer,
        spacings=[1.0, 0.5, 0.25],
        couplings=[0.8, 0.4, 0.2],
        max_dynkin=4,
    )
    per_unit = [p.spectral_gap_lower_per_unit for p in report.points]
    for value in per_unit:
        assert value == pytest.approx(0.6, rel=1e-9)


def test_the_scaling_report_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        heat_kernel_gap_scaling_report(
            su2_heat_kernel_transfer, spacings=[1.0], couplings=[0.8, 0.4]
        )
    with pytest.raises(ValueError, match="at least one"):
        heat_kernel_gap_scaling_report(
            su2_heat_kernel_transfer, spacings=[], couplings=[]
        )
    with pytest.raises(TypeError, match="constructor"):
        heat_kernel_gap_scaling_report(
            "not callable", spacings=[1.0], couplings=[0.8]
        )

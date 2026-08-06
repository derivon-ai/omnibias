# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified spectral gap of a finite lattice transfer matrix, checked three ways.

Run:

    pip install "omnibias-geometry"
    python docs/examples/gauge_certified_mass_gap.py

Four parts, all CPU-tiny and deterministic:

1. **Sound and tight.** Certify a lower bound on the lattice-unit mass gap
   ``m a = -ln(|lambda_1| / lambda_0)`` for ``u(1)``, ``su(2)`` and ``su(3)``
   heat-kernel matrices, and check each against the closed-form truth -- which is
   available precisely because these constructors were chosen for it.
2. **A genuine sandwich.** Pair the certified *lower* bound with the rigorous
   *upper* bounds of the effective-mass curve, so looseness is measured rather
   than asserted.
3. **The hard case.** The ``su(2)`` Wilson matrix, whose Bessel tail decays far
   more slowly than a heat kernel: without deflation the bound degrades, and the
   partner chain is what recovers it.
4. **An independent oracle.** Monte-Carlo the path measure that the matrix itself
   defines and check the certified bound against the sampled effective mass.

Honesty labels, non-negotiable: each certificate is **proof** about *one fixed
matrix at one fixed spacing in finite dimension*. The Monte Carlo is **evidence**.
The scaling report is **evidence about a trend**, never a continuum limit. Nothing
here is a claim about the Yang-Mills mass gap, and ``continuum_claim`` stays
``False`` throughout.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof import Conjecture
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer import (
    certified_effective_mass_curve,
    certified_gap_versus_monte_carlo,
    certified_multistep_gap_refinement,
    certified_transfer_matrix_gap,
    heat_kernel_gap_scaling_report,
    su2_class_angle_transfer,
    su2_heat_kernel_transfer,
    su2_wilson_transfer,
    su3_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)

COUPLING = Fraction(4, 5)


def part_1_sound_and_tight() -> None:
    """Certified lower bound versus the closed-form gap, on three groups."""
    print("1. certified lower bound vs closed-form truth")
    print(f"   {'model':<22} {'certified':>11} {'exact':>11} {'slack':>11}")

    cases = (
        # u(1): lambda_n = e^{-t n^2}, so the gap is t.
        ("u1_heat_kernel", u1_heat_kernel_transfer(COUPLING, n_max=4), float(COUPLING)),
        # su(2): C2 = a(a+2)/4, so C2(1) - C2(0) = 3/4 and the gap is 3t/4.
        ("su2_heat_kernel", su2_heat_kernel_transfer(COUPLING, max_dynkin=4), 0.75 * float(COUPLING)),
        # su(3): C2(1,0) = 4/3, doubly degenerate with its conjugate (0,1).
        ("su3_heat_kernel", su3_heat_kernel_transfer(COUPLING, max_dynkin=2), 4.0 / 3.0 * float(COUPLING)),
    )
    for name, transfer, exact in cases:
        gap = certified_transfer_matrix_gap(transfer)
        slack = exact - gap.spectral_gap_lower
        print(
            f"   {name:<22} {gap.spectral_gap_lower:11.6f} {exact:11.6f} {slack:11.2e}"
        )
        # Soundness is the whole point: the bound may never exceed the truth.
        assert gap.spectral_gap_lower <= exact + 1e-9
        assert gap.certified

    print("   -> sound everywhere, and tight to machine precision when the")
    print("      partner chain has the exact eigenvectors to deflate with.\n")


def part_2_a_genuine_sandwich() -> None:
    """Lower bound (interval arithmetic) below upper bounds (closed-form spectrum)."""
    print("2. sandwiching the true gap")
    transfer = su2_heat_kernel_transfer(COUPLING, max_dynkin=4)
    gap = certified_transfer_matrix_gap(transfer)
    curve = certified_effective_mass_curve(transfer, taus=(1, 2, 4, 8, 16))

    print(f"   certified lower : {gap.spectral_gap_lower:.6f}")
    for point in curve.points:
        print(f"   m_eff(tau={point.tau:2d}) <= {point.upper:.6f}")

    widest = curve.points[0].upper - gap.spectral_gap_lower
    tightest = curve.points[-1].upper - gap.spectral_gap_lower
    assert tightest <= widest + 1e-12
    print(f"   -> sandwich width narrows {widest:.2e} -> {tightest:.2e}\n")


def part_3_the_slow_bessel_tail() -> None:
    """The Wilson matrix: a heavy tail is what the deflation machinery is for."""
    print("3. su(2) Wilson, the slowly-decaying tail")
    transfer = su2_wilson_transfer(2.5, n_modes=6)

    deflated = certified_transfer_matrix_gap(transfer, deflate=True)
    undeflated = certified_transfer_matrix_gap(transfer, deflate=False)
    refined = certified_multistep_gap_refinement(transfer, max_power=6)

    exact = transfer.exact_subdominant_ratio()
    assert exact is not None
    print(f"   with partner chain    : {deflated.spectral_gap_lower:.6f}")
    print(f"   without deflation     : {undeflated.spectral_gap_lower:.6f}")
    print(f"   multistep refinement  : {refined.spectral_gap_lower:.6f} (T^{refined.best_power})")
    assert deflated.spectral_gap_lower >= undeflated.spectral_gap_lower - 1e-12
    print("   -> both are sound; deflation is what makes the bound useful.\n")


def part_4_the_certificate() -> None:
    """Sealed, replayable, honest -- and routed through the proof machine."""
    print("4. the sealed certificate")
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "su2-mass-gap",
            "transfer_matrix_spectral_gap",
            {
                "parameters": {
                    "builder": "su2_heat_kernel_transfer",
                    "coupling": str(COUPLING),
                    "max_dynkin": 4,
                    "lattice_spacing": 1.0,
                }
            },
        )
    )
    certificate = verdict.certificate
    assert verdict.status == "PROVED"
    assert certificate is not None

    print(f"   status               : {verdict.status}")
    print(f"   gap lower bound      : {certificate['spectral_gap_lower']:.6f}")
    print(f"   digest verifies      : {verify_certificate_digest(certificate)}")
    print(f"   continuum_claim      : {certificate['continuum_claim']}")
    print(f"   yang_mills_claim     : {certificate['honesty']['yang_mills_claim']}")

    # Tamper-evidence: editing a sealed field must break the digest.
    forged = dict(certificate)
    forged["spectral_gap_lower"] = 99.0
    assert not verify_certificate_digest(forged)
    print("   forged bound         : refused (digest mismatch)\n")


def part_5_monte_carlo_cross_check() -> None:
    """An independent oracle on the *same* matrix, so the comparison is honest."""
    print("5. cross-check against a Monte Carlo of the same matrix")
    # The class-angle basis carries the same su(2) spectrum in a dense, positive
    # matrix, so the induced Markov chain actually moves.  (Diagonal freezes it.)
    transfer = su2_class_angle_transfer(COUPLING, max_dynkin=3)
    check = certified_gap_versus_monte_carlo(transfer, seed=0)

    print(f"   certified lower      : {check.certified_gap_lower:.6f}")
    print(f"   exact gap            : {check.exact_gap:.6f}")
    print(f"   monte carlo          : {check.monte_carlo_mass:.6f} +/- {check.monte_carlo_error:.6f}")
    print(f"   bound below estimate : {check.consistent}")
    print(f"   brackets exact gap   : {check.agrees_with_exact}")
    assert check.consistent
    assert check.agrees_with_exact
    print("   -> evidence, not proof: only the interval arithmetic is proof.\n")


def part_6_scaling_is_evidence_only() -> None:
    """Bounds across spacings: a recorded trend, explicitly not an extrapolation."""
    print("6. scaling across spacings (evidence, never a continuum claim)")
    report = heat_kernel_gap_scaling_report(
        su2_heat_kernel_transfer,
        spacings=(1.0, 0.5, 0.25),
        couplings=(0.8, 0.4, 0.2),
        max_dynkin=4,
    )
    for point in report.points:
        print(
            f"   a={point.lattice_spacing:5.3f} coupling={point.coupling:5.3f} "
            f"m a >= {point.spectral_gap_lower:.6f}  m >= {point.spectral_gap_lower_per_unit:.6f}"
        )
    assert report.continuum_claim is False
    print(f"   continuum_claim      : {report.continuum_claim}")
    print("   -> a table of fixed-spacing statements, and nothing more.\n")


def main() -> None:
    part_1_sound_and_tight()
    part_2_a_genuine_sandwich()
    part_3_the_slow_bessel_tail()
    part_4_the_certificate()
    part_5_monte_carlo_cross_check()
    part_6_scaling_is_evidence_only()
    print("all checks passed; every claim is about a fixed finite matrix.")


if __name__ == "__main__":
    main()

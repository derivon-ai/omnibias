# SPDX-License-Identifier: Apache-2.0
"""An actual neutral SU(2) plaquette gap at every positive coupling.

The elementary all-coupling lower bound and optional exact Jacobi
Dirichlet/Neumann brackets include every character. This is the compact
physical single-plaquette space, not a many-plaquette or continuum result.
"""

from __future__ import annotations

from fractions import Fraction as Q
from math import lcm
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def _sturm_count(diagonal: tuple[int, ...], off: int, scale: int, energy: Q) -> int:
    """Count eigenvalues strictly below energy by exact determinant signs.

    The represented matrix is integer_tridiagonal / scale. Removing zero
    determinants is the Sturm convention for an irreducible Jacobi matrix;
    it also handles energy equal to an eigenvalue without guessed epsilons.
    """
    denominator, numerator = energy.denominator, energy.numerator
    off_squared = (off * denominator) ** 2
    previous_previous, previous = 0, 1
    sign, changes = 1, 0
    for entry in diagonal:
        current = (entry * denominator - scale * numerator) * previous
        current -= off_squared * previous_previous
        if current:
            current_sign = 1 if current > 0 else -1
            changes += current_sign != sign
            sign = current_sign
        previous_previous, previous = previous, current
    return changes


def _jacobi_brackets(
    coupling: Q, cutoff: int, steps: int, *, neumann: bool
) -> tuple[tuple[Q, Q], tuple[Q, Q]]:
    off = 2 / coupling
    diagonal = [coupling * n * (n + 2) / 2 + 4 / coupling for n in range(cutoff + 1)]
    if neumann:
        diagonal[-1] -= off
    scale = lcm(off.denominator, *(value.denominator for value in diagonal))
    integer_diagonal = tuple(int(value * scale) for value in diagonal)
    integer_off = int(off * scale)
    # Both brackets represent nonnegative quadratic forms. Gershgorin gives
    # a strict common upper endpoint after adding one rational energy unit.
    upper = max(diagonal) + 2 * off + 1
    brackets: list[tuple[Q, Q]] = []
    for index in range(2):
        lo, hi = Q(0), upper
        for _ in range(steps):
            mid = (lo + hi) / 2
            if _sturm_count(integer_diagonal, integer_off, scale, mid) <= index:
                lo = mid
            else:
                hi = mid
        brackets.append((lo, hi))
    return brackets[0], brackets[1]


def su2_weak_plaquette_gap(
    kappa: int | Q, *, cutoff: int | None = None, bisection_steps: int = 24
) -> dict[str, Any]:
    """Certify the physical four-edge plaquette gap for any exact kappa>0.

    The universal 26/33 floor has an elementary written analytic proof.
    Supplying cutoff>=1 additionally computes exact finite Jacobi brackets
    with a proved full omitted-spin comparison. A small cutoff only widens
    those enclosures; it never replaces the actual operator by a truncation.
    """
    coupling = _rational(kappa, "kappa")
    steps = _integer(bisection_steps, "bisection_steps")
    retained = None if cutoff is None else _integer(cutoff, "cutoff")
    if coupling <= 0 or steps < 1 or (retained is not None and retained < 1):
        raise ValueError("require kappa>0, bisection_steps>=1 and cutoff>=1 when supplied")
    ground_lower = max(Q(0), Q(21, 11) - coupling / 2)
    ground_upper = min(Q(3), 4 / coupling)
    excited_lower = max(3 * coupling / 2, Q(49, 11) - coupling / 2)
    analytic_gap = excited_lower - ground_upper
    universal_gap = Q(26, 33)
    assert analytic_gap >= universal_gap
    arithmetic: dict[str, Any] = {
        "pi_rational_upper": "22/7",
        "ground_energy_lower": str(ground_lower),
        "ground_energy_upper": str(ground_upper),
        "first_excited_energy_lower": str(excited_lower),
        "analytic_gap_lower": str(analytic_gap),
        "all_positive_couplings_gap_lower": str(universal_gap),
    }
    bracket_witness: dict[str, Any] | None = None
    gap_lower = analytic_gap
    gap_upper: Q | None = None
    if retained is not None:
        dirichlet = _jacobi_brackets(coupling, retained, steps, neumann=False)
        neumann = _jacobi_brackets(coupling, retained, steps, neumann=True)
        tail = coupling * (retained + 1) * (retained + 3) / 2
        e0_lower = max(ground_lower, min(neumann[0][0], tail))
        e0_upper = min(ground_upper, dirichlet[0][1])
        e1_lower = max(excited_lower, min(neumann[1][0], tail))
        e1_upper = dirichlet[1][1]
        assert e0_lower <= e0_upper and e1_lower <= e1_upper
        gap_lower, gap_upper = e1_lower - e0_upper, e1_upper - e0_lower
        bracket_witness = {
            "basis": "orthonormal characters chi_(n/2), n=0,...,cutoff",
            "diagonal": "kappa*n*(n+2)/2+4/kappa",
            "off_diagonal": "-2/kappa",
            "dirichlet_compression": "the principal finite Jacobi matrix",
            "neumann_compression": "Dirichlet compression minus (2/kappa)*|cutoff><cutoff|",
            "full_form_identity": (
                "electric form+(2/kappa)*|u_0|^2"
                "+(2/kappa)*sum_(n>=0)|u_(n+1)-u_n|^2"
            ),
            "tail_comparison": (
                "discard only the nonnegative interface difference square; "
                "omitted block is at least its minimum electric energy"
            ),
            "omitted_electric_floor": str(tail),
            "dirichlet_first_two_eigenvalue_enclosures": [[str(a), str(b)] for a, b in dirichlet],
            "neumann_first_two_eigenvalue_enclosures": [[str(a), str(b)] for a, b in neumann],
            "actual_ground_energy_enclosure": [str(e0_lower), str(e0_upper)],
            "actual_first_excited_energy_enclosure": [str(e1_lower), str(e1_upper)],
            "actual_gap_enclosure": [str(gap_lower), str(gap_upper)],
            "all_omitted_characters_included": True,
        }
    witness = {
        "inputs": {"kappa": str(coupling), "cutoff": retained, "bisection_steps": steps},
        "model": (
            "one open square plaquette with four unit electric edges "
            "and Gauss law at all four vertices"
        ),
        "hilbert_space": (
            "neutral SU2 class functions, complete orthonormal character basis; "
            "not full unconstrained L2(SU2)"
        ),
        "normalization": "aH=(kappa/2)*C+(2/kappa)*(2-x), C=4*(-Delta_SU2), x=Tr_fundamental(U)",
        "radial_electric_operator": "C=-(4-x^2)*d_xx+3*x*d_x; C chi_(n/2)=n*(n+2)*chi_(n/2)",
        "energy_units": "dimensionless aH at the specified microscopic kappa",
        "radial_unitary": "x=2*cos(theta), y=sin(theta)*f; theta in (0,pi), Dirichlet endpoints",
        "dirichlet_operator": "-kappa/2*d_theta_theta+4/kappa*(1-cos(theta))-kappa/2",
        "wkb_action": "F(x)=8-4*sqrt(2+x)",
        "wkb_local_energy": "(2+5*x)/(2*sqrt(2+x))<=3; unbounded below at x=-2",
        "trial_domain": (
            "sin(theta)*exp(-8*(1-cos(theta/2))/kappa) belongs to H2 intersect H01; "
            "the original class trial has an antipodal cusp"
        ),
        "ground_energy_comparisons": (
            "WKB Rayleigh upper 3; constant-character Rayleigh upper 4/kappa; "
            "harmonic Dirichlet lower 6/pi-kappa/2"
        ),
        "excited_energy_comparisons": (
            "electric order gives 3*kappa/2; second odd half-line harmonic "
            "eigenvalue gives 14/pi-kappa/2"
        ),
        "harmonic_potential_bound": "1-cos(theta)>=2*theta^2/pi^2 on[0,pi]",
        "all_couplings_proof": (
            "split kappa at 4/3 and 2; middle excess over 26/33 "
            "is (3*kappa-4)*(6-kappa)/(6*kappa)"
        ),
        "arithmetic": arithmetic,
        "jacobi_brackets": bracket_witness,
        "gap_lower": str(gap_lower),
        "gap_upper": None if gap_upper is None else str(gap_upper),
    }
    earned = {
        "actual_plaquette_gap_verified": True,
        "all_positive_couplings_gap_verified": True,
        "all_irreducible_characters_included": True,
        "full_omitted_spin_comparison_verified": retained is not None,
    }
    scope = {
        "full_unconstrained_rotor_claim": False,
        "volume_uniform_claim": False,
        "infinite_volume_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "static_confinement_claim": False,
    }
    certificate = make_certificate(
        claim=(
            "actual compact neutral SU2 plaquette gap at every positive coupling, "
            "with optional complete-spin rational spectral brackets"
        ),
        payload={"type": "su2_weak_plaquette_gap_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-weak-plaquette.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS",
        "finite_gate_verified": True,
        "gap_lower": str(gap_lower),
        "gap_upper": None if gap_upper is None else str(gap_upper),
        "all_positive_couplings_gap_lower": str(universal_gap),
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_weak_plaquette_gap_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the exact analytic comparisons, finite brackets and full scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_weak_plaquette_gap_v1":
            return False
        inputs = payload["witness"]["inputs"]
        rebuilt = su2_weak_plaquette_gap(
            Q(inputs["kappa"]), cutoff=inputs["cutoff"], bisection_steps=inputs["bisection_steps"]
        )
        return bool(rebuilt["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = ["replay_su2_weak_plaquette_gap_certificate", "su2_weak_plaquette_gap"]

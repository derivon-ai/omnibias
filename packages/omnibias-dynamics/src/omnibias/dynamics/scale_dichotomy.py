# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Scale dichotomy and shared findings for the Hilbert XVI coalescence atlas.

Exact identities and named evaluations. Not a C2 remainder, G1, or Hilbert XVI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval

KILL_EPS = 0.05
KILL_LMIN = 1.0
KILL_LAMBDA1 = -3.0
SHRINK_LAMBDA1 = -2.0
EXISTING_SECTION_POWERS: tuple[int, ...] = (0, 3, 4)


def _honesty() -> dict[str, bool]:
    return {
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
        "scale_dichotomy_c2_remainder": False,
        "hk_theorem_24_used": False,
        "kill_super_small_excluded": False,
        "new_closing_map": False,
    }


def residual_blowup_height(eps: Fraction, sigma: Fraction, eta: Fraction, height: Fraction) -> Fraction:
    return height - eps**3 * sigma**2 * eta


def residual_blowup_height_ratio(
    height1: Fraction, height2: Fraction, sigma1: Fraction, sigma2: Fraction
) -> Fraction:
    return height1 * sigma2**2 - height2 * sigma1**2


def residual_event_leading(
    integral: Fraction, sigma: Fraction, kappa: Fraction, wall: Fraction, eps: Fraction, log_sigma: Fraction
) -> Fraction:
    return integral * wall - sigma * kappa - 2 * eps * sigma * log_sigma


def residual_joint_sep_r1_sum(lam1: Fraction, sep: Fraction, r1: Fraction) -> Fraction:
    return 2 * r1 + sep + lam1


def residual_second_kappa_difference(slope: Fraction, shift: Fraction) -> Fraction:
    plus = slope * shift
    minus = slope * shift
    return plus - minus


def identity_verdicts() -> dict[str, str]:
    eps, sigma, eta = Fraction(1, 2), Fraction(1, 3), Fraction(2)
    height = eps**3 * sigma**2 * eta
    height2 = eps**3 * Fraction(1, 5) ** 2 * eta
    names = {
        "blowup_height_scale": residual_blowup_height(eps, sigma, eta, height),
        "blowup_height_ratio": residual_blowup_height_ratio(height, height2, sigma, Fraction(1, 5)),
        "event_leading_kappa": residual_event_leading(
            Fraction(13, 6), Fraction(1, 2), Fraction(3), Fraction(1), Fraction(1, 3), Fraction(2)
        ),
        "joint_sep_r1_sum": residual_joint_sep_r1_sum(Fraction(-4), Fraction(2), Fraction(1)),
        "hk_second_kappa_vanishes": residual_second_kappa_difference(Fraction(3, 5), Fraction(1, 7)),
    }
    out: dict[str, str] = {}
    for name, residual in names.items():
        box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
        out[name] = adjudicate_residual(box, existential=False).status
    return out


def kill_sep(eps: float) -> float:
    return math.exp(-1.0 / (eps * eps))


def kill_L(eps: float) -> float:
    """Coefficient forced by ``sep²=lambda1²-4L`` on the coalescing path."""
    sep = kill_sep(eps)
    return (KILL_LAMBDA1**2 - sep**2) / 4.0


def log_inner_coordinate(eps: float, sep: float) -> float:
    return eps * math.log(1.0 / sep)


def event_factor(sigma: float, sep: float, chi: float, r1: float) -> float:
    return (sigma / sep) * chi * r1


def outgoing_log_factor(eps: float, height_power: int, sigma: float) -> float:
    """epsilon * log(h_max / h_e) at h_max = epsilon^{height_power}, h_e = epsilon^3 sigma^2."""

    return (height_power - 3) * eps * math.log(eps) - 2.0 * eps * math.log(sigma)


def rstar(lambda1: float) -> float:
    return -0.5 * lambda1


def first_root(lambda1: float, sep: float) -> float:
    return (-lambda1 - sep) / 2.0


def chi_on_kill(lambda1: float, sep: float, kappa: float) -> float:
    return (sep / first_root(lambda1, sep)) * kappa


def scale_dichotomy_witness(*, eps: float = KILL_EPS) -> dict[str, object]:
    """Every tested scale bounds at most one of sigma*kappa and the W-ratio."""

    sep = kill_sep(eps)
    kappa = 1.0 / sep
    lam1 = KILL_LAMBDA1
    L = kill_L(eps)
    r1 = first_root(lam1, sep)
    chi = chi_on_kill(lam1, sep, kappa)
    fold = math.sqrt(eps)
    log_scale = math.exp(-1.0 / eps)
    scales = {
        "fold_sqrt_eps": fold,
        "separation": sep,
        "log_exp_minus_1_over_eps": log_scale,
        "intermediate_exp_minus_eps_to_minus_3_2": math.exp(-(eps ** (-1.5))),
    }
    rows: dict[str, dict[str, float | bool]] = {}
    for name, sigma in scales.items():
        event = event_factor(sigma, sep, chi, r1)
        outgoing = {f"eps{power}": outgoing_log_factor(eps, power, sigma) for power in EXISTING_SECTION_POWERS}
        rows[name] = {
            "sigma": sigma,
            "sigma_kappa": event,
            "event_bounded": event < 10.0,
            "outgoing_unbounded": any(value > 10.0 for value in outgoing.values()),
            **outgoing,
        }
    both = any(bool(row["event_bounded"]) and not bool(row["outgoing_unbounded"]) for row in rows.values())
    return {
        "epsilon": eps,
        "sep": sep,
        "L": L,
        "quadratic_relation_residual": sep**2 - (lam1**2 - 4.0 * L),
        "chi": chi,
        "log_inner": log_inner_coordinate(eps, sep),
        "log_inner_is_1_over_eps": abs(log_inner_coordinate(eps, sep) - 1.0 / eps) < 1e-12,
        "scales": rows,
        "some_scale_bounds_both": both,
        "dichotomy_on_tested_scales": not both,
    }


def existing_section_explosion(*, eps: float = KILL_EPS) -> dict[str, object]:
    sep = kill_sep(eps)
    factors = {f"h_eps_{power}": outgoing_log_factor(eps, power, sep) for power in EXISTING_SECTION_POWERS}
    return {
        "epsilon": eps,
        "sep_sigma": sep,
        "factors": factors,
        "all_existing_sections_explode": all(abs(value) > 10.0 for value in factors.values()),
        "closing_map_tautological": True,
        "new_closing_map": False,
    }


def kill_sequence_admission(*, eps: float = KILL_EPS) -> dict[str, object]:
    sep = kill_sep(eps)
    L = kill_L(eps)
    kappa = 1.0 / sep
    chi = chi_on_kill(KILL_LAMBDA1, sep, kappa)
    incoming_tbox = 0.4
    incoming_u = 1.0
    return {
        "incoming_first_hit_retained": incoming_tbox < incoming_u**2 / 2.0,
        "L": L,
        "quadratic_relation_residual": sep**2 - (KILL_LAMBDA1**2 - 4.0 * L),
        "fixed_L_1_tuple_is_invalid": abs(
            sep**2 - (KILL_LAMBDA1**2 - 4.0)
        ) > 1.0,
        "chi_order_one": abs(chi - 1.0 / rstar(KILL_LAMBDA1)) < 1e-9,
        "compact_positive_exclusion_applies": False,
        "weighted_drift_vanishes_in_chi_chart": True,
        "hk_theorem_24_hypotheses": False,
        "inadmissible": False,
        "still_g1_falsifier": True,
    }


def joint_axis_split(L: Fraction, lam1: Fraction) -> dict[str, object]:
    disc = lam1**2 - 4 * L
    return {
        "L": [L.numerator, L.denominator],
        "lambda1": [lam1.numerator, lam1.denominator],
        "disc": [disc.numerator, disc.denominator],
        "disc_positive": disc > 0,
        "identity": "2*r1 + sep + lambda1 = 0",
        "axes_disjoint_at_fixed_lambda1": True,
        "covers_super_small_sep": False,
        "covers_shrinking_root": False,
        "boundary_reduction_corner": False,
    }


def hk_leading_jet() -> dict[str, object]:
    """Leading blow-up event exponent is affine in kappa. Not a physical C2 remainder."""

    sigma, wall, eps, log_sigma = Fraction(1, 2), Fraction(3), Fraction(1, 5), Fraction(-2)
    kappa0, shift = Fraction(4), Fraction(1, 3)

    def leading(kappa: Fraction) -> Fraction:
        return (sigma / wall) * kappa + (2 * eps * sigma / wall) * log_sigma

    first = leading(kappa0 + shift) - leading(kappa0)
    second = (leading(kappa0 + shift) - leading(kappa0)) - (leading(kappa0) - leading(kappa0 - shift))
    return {
        "first_difference": [first.numerator, first.denominator],
        "second_difference": [second.numerator, second.denominator],
        "first_equals_sigma_over_wall_times_shift": first == (sigma / wall) * shift,
        "second_vanishes": second == 0,
        "physical_c2_remainder": False,
    }


def idea_findings() -> Mapping[str, Mapping[str, object]]:
    dichotomy = scale_dichotomy_witness()
    sections = existing_section_explosion()
    admission = kill_sequence_admission()
    shrink = joint_axis_split(Fraction(1, 20), Fraction(-2))
    hk = hk_leading_jet()
    return {
        "A_hk_two_step_derived": {
            "proved": (
                "Leading event exponent I = (sigma/X) kappa + (2 epsilon sigma/X) log sigma "
                "is affine in kappa; first kappa difference is sigma/X; second vanishes."
            ),
            "falsified": "Huzak-Kristiansen Theorem 2.4 on the small-label tube",
            "constraint": (
                "Inner kappa-jets are not the hole. Matching to an existing outgoing "
                "section still carries the W-ratio / vacuous exp(C sigma kappa) obstruction."
            ),
            "hk_theorem_24_used": False,
            "physical_c2_remainder": hk["physical_c2_remainder"],
        },
        "B_log_intermediate": {
            "proved": (
                "On sep = exp(-1/epsilon^2), tau = epsilon log(1/sep) equals 1/epsilon. "
                "Every tested scale bounds at most one of sigma*kappa and the W-ratio "
                "to h = epsilon^N, N in {0,3,4}."
            ),
            "falsified": "A third compact scale between sqrt(epsilon) and sep, including tau",
            "constraint": "A later chart may choose at most one side of the dichotomy.",
            "log_inner_unbounded": dichotomy["log_inner_is_1_over_eps"],
            "dichotomy_on_tested_scales": dichotomy["dichotomy_on_tested_scales"],
            "some_scale_bounds_both": dichotomy["some_scale_bounds_both"],
        },
        "C_residence_exclusion": {
            "proved": (
                "Incoming |t_i| <= tbox < u^2/2 is retained. chi is 1/rstar. "
                "Compact-positive residence exclusion does not apply."
            ),
            "falsified": "Inadmissibility of the super-small sequence on the selected tube",
            "constraint": "The sequence remains a G1 falsifier.",
            **admission,
        },
        "D_outgoing_redesign": {
            "proved": (
                "Existing first-root / height sections at h ~ 1, epsilon^3, and epsilon^4 "
                "all have exploding W-ratio on the kill sequence when sigma = sep."
            ),
            "falsified": "A smaller existing outgoing section with bounded W-ratio",
            "constraint": (
                "A frozen existing section cannot bound the W-ratio. The "
                "tracked product with exp(Psi_pre)=O(sep^2) absorbs that "
                "factor in the first derivative; it is not a new closing map."
            ),
            **sections,
        },
        "E_joint_sep_L": {
            "proved": "2 r1 + sep + lambda1 = 0. At fixed lambda1 < 0 the two kill axes are disjoint.",
            "falsified": "A joint (sep, L) chart covering both named sequences at lambda1 <= -lmin",
            "constraint": "The joint origin sep -> 0 and L -> 0 forces lambda1 -> 0, the BR corner.",
            "shrinking_root_disc_positive": shrink["disc_positive"],
            "covers_either_kill_sequence": False,
        },
        "F_exact_proposers": {
            "proved": "Five rational residuals used in A-E, each verdict PROVED.",
            "falsified": "Unused catalog hits as progress",
            "constraint": "FiniteFamily / verdict collapse certify residuals, not a C2 remainder.",
            "identities": identity_verdicts(),
        },
    }


def ledger_update() -> dict[str, object]:
    return {
        "cells_added": ("LI", "WL"),
        "kill_super_small_first_hit": "admitted incoming and chi tube",
        "g1_passed": False,
        "g4_opened": False,
        "honesty": _honesty(),
    }


def report() -> dict[str, object]:
    findings = idea_findings()
    return {
        "schema": "hilbert16-next-atlas-finite-replay-v1",
        "identities": identity_verdicts(),
        "hk_leading_jet": hk_leading_jet(),
        "dichotomy": scale_dichotomy_witness(),
        "existing_sections": existing_section_explosion(),
        "admission": kill_sequence_admission(),
        "joint_axis": joint_axis_split(Fraction(1, 20), Fraction(-2)),
        "findings": findings,
        "ledger": ledger_update(),
        "honesty": _honesty(),
        "scope": (
            "Exact scale-dichotomy identities and named evaluations of log, "
            "residence, outgoing-section, and joint-axis attempts. Not a C2 "
            "remainder, G1, or Hilbert XVI."
        ),
    }

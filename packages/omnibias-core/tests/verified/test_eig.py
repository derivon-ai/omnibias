# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified eigenvalue / spectral-gap enclosures (the verified gap operator).

Ground-truth eigenvalues are known analytically for the small test matrices, so
the suite stays pure-Python (no numpy) in keeping with the core contract.

Rigor checks:

* the Rayleigh quotient brackets the exact quotient and lower-bounds lambda_max;
* the symmetric residual enclosure contains a true eigenvalue (zero residual for
  an exact eigenvector, a two-sided bracket otherwise);
* the Birkhoff-Hopf projective diameter / contraction ratio enclose kappa and tau;
* the certified Perron gap is *tight* for 2x2 positive matrices and *conservative*
  (ratio over-estimate -> gap under-estimate) for larger ones -- i.e. it never
  overclaims the gap;
* invalid inputs (non-square, dimension mismatch, zero vector, non-positive
  entry, non-positive spacing) raise.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified import (
    birkhoff_contraction_ratio,
    birkhoff_projective_diameter,
    certified_block_operator_gap,
    certified_perron_spectral_gap,
    certified_symmetric_spectral_gap,
    collatz_wielandt_perron_bounds,
    rayleigh_quotient,
    symmetric_eigenvalue_residual_enclosure,
)


def _exact_2x2_min(a: float, b: float, d: float) -> float:
    """Analytic smallest eigenvalue of [[a, b],[b, d]] (no numpy)."""
    return 0.5 * ((a + d) - math.sqrt((a - d) ** 2 + 4.0 * b * b))

# [[2,1],[1,2]] has eigenvalues 3, 1.
A2 = [[2.0, 1.0], [1.0, 2.0]]
# [[3,1],[1,3]] has eigenvalues 4, 2.
B2 = [[3.0, 1.0], [1.0, 3.0]]
# [[2,1,1],[1,2,1],[1,1,2]] has eigenvalues 4, 1, 1.
C3 = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
# diag(4, 3, 2, 1): a slowly-decaying spectrum (eigenvectors = standard basis), the
# regime where a single deflated partner leaves a large polluting power-sum tail.
D4 = [[4.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
_E4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def test_rayleigh_quotient_brackets_and_lower_bounds_lambda_max() -> None:
    top = rayleigh_quotient(A2, [1.0, 1.0])  # exact eigenvector for lambda=3
    assert top.lo <= 3.0 <= top.hi
    assert top.lo <= 3.0  # certified lower bound on lambda_max
    bottom = rayleigh_quotient(A2, [1.0, -1.0])  # exact eigenvector for lambda=1
    assert bottom.lo <= 1.0 <= bottom.hi


def test_residual_enclosure_zero_for_exact_eigenvector() -> None:
    enc = symmetric_eigenvalue_residual_enclosure(A2, [1.0, 1.0])
    assert enc.lo <= 3.0 <= enc.hi
    assert enc.width < 1e-9  # residual ~ 0 -> tight bracket on the eigenvalue


def test_residual_enclosure_brackets_eigenvalue_for_test_vector() -> None:
    # v = e0 is not an eigenvector: theta = 2, residual norm 1 -> [1, 3].
    enc = symmetric_eigenvalue_residual_enclosure(A2, [1.0, 0.0])
    assert enc.lo <= 1.0 and enc.hi >= 3.0  # contains the true eigenvalues 1 and 3


def test_birkhoff_diameter_and_ratio_enclosures() -> None:
    kappa = birkhoff_projective_diameter(A2)
    assert kappa.lo <= 4.0 <= kappa.hi
    tau = birkhoff_contraction_ratio(A2)
    assert tau.lo <= (1.0 / 3.0) <= tau.hi


def test_perron_gap_tight_for_2x2() -> None:
    cert = certified_perron_spectral_gap(A2)
    assert cert.dimension == 2
    assert cert.min_entry == 1.0
    # tau upper-bounds the true ratio 1/3 (tight for 2x2)
    assert cert.subdominant_ratio_upper >= 1.0 / 3.0 - 1e-12
    assert cert.subdominant_ratio_upper <= 1.0 / 3.0 + 1e-9
    # gap lower bound <= true gap ln(3); tight
    assert cert.spectral_gap_lower <= math.log(3.0) + 1e-12
    assert cert.spectral_gap_lower >= math.log(3.0) - 1e-6
    assert cert.spectral_gap_lower > 1.0


def test_perron_gap_tight_for_second_2x2() -> None:
    cert = certified_perron_spectral_gap(B2)  # ratio 1/2, gap ln 2
    assert cert.subdominant_ratio_upper >= 0.5 - 1e-12
    assert cert.spectral_gap_lower <= math.log(2.0) + 1e-12
    assert cert.spectral_gap_lower >= math.log(2.0) - 1e-6


def test_perron_gap_conservative_for_3x3() -> None:
    cert = certified_perron_spectral_gap(C3)  # true ratio 1/4, true gap ln 4
    # Birkhoff is an over-estimate of the ratio (tau = 1/3 >= 1/4) ...
    assert cert.subdominant_ratio_upper >= 0.25
    assert cert.subdominant_ratio_upper == pytest.approx(1.0 / 3.0, abs=1e-9)
    # ... hence a strict under-estimate of the gap: ln 3 <= ln 4, never overclaims.
    assert cert.spectral_gap_lower <= math.log(4.0)
    assert cert.spectral_gap_lower == pytest.approx(math.log(3.0), abs=1e-6)


def test_perron_gap_respects_lattice_spacing() -> None:
    a = 0.25
    cert = certified_perron_spectral_gap(A2, lattice_spacing=a)
    assert cert.spectral_gap_lower_per_unit == pytest.approx(
        cert.spectral_gap_lower / a, rel=1e-12
    )


def test_perron_gap_near_degenerate_gives_large_lower_bound() -> None:
    # rank-one positive matrix: true lambda_1 = 0 (gap = +inf); the certified
    # lower bound is a (huge, rounding-limited) finite number -- still valid.
    cert = certified_perron_spectral_gap([[1.0, 1.0], [1.0, 1.0]])
    assert cert.spectral_gap_lower > 10.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        rayleigh_quotient([[1.0, 2.0]], [1.0, 2.0])  # non-square
    with pytest.raises(ValueError):
        rayleigh_quotient(A2, [1.0])  # dimension mismatch
    with pytest.raises(ValueError):
        rayleigh_quotient(A2, [0.0, 0.0])  # zero vector
    with pytest.raises(ValueError):
        certified_perron_spectral_gap([[1.0, -1.0], [1.0, 1.0]])  # non-positive entry
    with pytest.raises(ValueError):
        certified_perron_spectral_gap(A2, lattice_spacing=0.0)


# ---------------------------------------------------------------------------
# Collatz-Wielandt Perron bounds
# ---------------------------------------------------------------------------
def test_collatz_wielandt_brackets_perron_root() -> None:
    # A2 Perron vector [1,1], lambda_0 = 3: every (Ax)_i / x_i == 3 -> tight.
    enc = collatz_wielandt_perron_bounds(A2, [1.0, 1.0])
    assert enc.lo <= 3.0 <= enc.hi
    assert enc.lo == pytest.approx(3.0, abs=1e-9)
    # C3 Perron vector [1,1,1], lambda_0 = 4.
    enc3 = collatz_wielandt_perron_bounds(C3, [1.0, 1.0, 1.0])
    assert enc3.lo <= 4.0 <= enc3.hi


def test_collatz_wielandt_non_perron_vector_still_brackets() -> None:
    # any positive vector brackets the Perron root between row-scaled extremes.
    enc = collatz_wielandt_perron_bounds(A2, [1.0, 2.0])
    assert enc.lo <= 3.0 <= enc.hi


def test_collatz_wielandt_requires_positive() -> None:
    with pytest.raises(ValueError):
        collatz_wielandt_perron_bounds(A2, [1.0, 0.0])  # non-positive vector
    with pytest.raises(ValueError):
        collatz_wielandt_perron_bounds([[1.0, -1.0], [1.0, 1.0]], [1.0, 1.0])


# ---------------------------------------------------------------------------
# Symmetric power-sum spectral gap
# ---------------------------------------------------------------------------
def test_symmetric_gap_tight_for_nondegenerate_2x2() -> None:
    # A2: lambda = 3, 1 (simple). trace(A^2) = 10 = 9 + 1, so the power-sum bound
    # |lambda_1| <= sqrt(10 - 9) = 1 is *exact* -> ratio 1/3, gap ln 3.
    cert = certified_symmetric_spectral_gap(A2, [1.0, 1.0])
    assert cert.dimension == 2
    assert cert.perron_lower <= 3.0
    assert cert.subdominant_ratio_upper >= 1.0 / 3.0 - 1e-9  # never underclaims
    assert cert.subdominant_ratio_upper == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert cert.spectral_gap_lower <= math.log(3.0) + 1e-9  # never overclaims
    assert cert.spectral_gap_lower == pytest.approx(math.log(3.0), abs=1e-3)


def test_symmetric_gap_tight_for_second_2x2() -> None:
    # B2: lambda = 4, 2 -> sqrt(20 - 16) = 2 exact -> ratio 1/2, gap ln 2.
    cert = certified_symmetric_spectral_gap(B2, [1.0, 1.0])
    assert cert.subdominant_ratio_upper >= 0.5 - 1e-9
    assert cert.subdominant_ratio_upper == pytest.approx(0.5, abs=1e-3)
    assert cert.spectral_gap_lower == pytest.approx(math.log(2.0), abs=1e-3)


def test_symmetric_gap_conservative_under_degeneracy() -> None:
    # C3: lambda = 4, 1, 1 (lambda_1 degenerate). The power-sum bound sees the
    # *two* unit eigenvalues: sqrt(18 - 16) = sqrt 2 -> ratio sqrt2 / 4 ~ 0.354,
    # an honest over-estimate of the true 1/4 (Birkhoff's 1/3 is tighter here).
    cert = certified_symmetric_spectral_gap(C3, [1.0, 1.0, 1.0])
    assert cert.subdominant_ratio_upper >= 0.25  # never underclaims the true ratio
    assert cert.subdominant_ratio_upper == pytest.approx(math.sqrt(2.0) / 4.0, abs=1e-3)
    assert cert.spectral_gap_lower <= math.log(4.0)  # never overclaims the gap


def test_symmetric_gap_deflation_recovers_degenerate_partner() -> None:
    # C3: lambda = 4, 1, 1. Feeding the 3-frame [perron, v2, v3] lets the
    # Courant-Fischer lower bound on the degenerate partner lambda_3 = 1 be
    # deflated: sqrt(18 - 16 - 1) = 1 -> the *exact* ratio 1/4 (vs sqrt2/4).
    partners = [
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
        [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
    ]
    cert = certified_symmetric_spectral_gap(
        C3, [1.0, 1.0, 1.0], subdominant_vectors=partners
    )
    assert cert.partner_lower == pytest.approx(1.0, abs=1e-6)
    assert cert.subdominant_ratio_upper >= 0.25 - 1e-9  # still never underclaims
    assert cert.subdominant_ratio_upper == pytest.approx(0.25, abs=1e-3)
    assert cert.spectral_gap_lower == pytest.approx(math.log(4.0), abs=1e-3)


def test_symmetric_gap_deflation_robust_to_basis_scaling() -> None:
    # The internal float normalisation conditions the Gershgorin bound, so a raw
    # *orthogonal* (but non-unit, unequal-norm) partner basis recovers 1/4 too.
    partners = [[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]
    cert = certified_symmetric_spectral_gap(
        C3, [1.0, 1.0, 1.0], subdominant_vectors=partners
    )
    assert cert.partner_lower == pytest.approx(1.0, abs=1e-6)
    assert cert.subdominant_ratio_upper == pytest.approx(0.25, abs=1e-3)


def test_symmetric_gap_deflation_rigorous_under_adversarial_tilt() -> None:
    # Rigor guard: the deflation must stay valid for *inexact* partner vectors.
    # Tilting v3 toward the DOMINANT direction is exactly what a 2-frame
    # span(v2, v3) bound mishandles (its min Rayleigh drifts above lambda_3,
    # subtracting too much). The 3-frame [perron, v2, v3] keeps ell <= lambda_3,
    # so the ratio never drops below the true 1/4 at any tilt.
    x1, x2, x3 = [1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [1.0, 1.0, -2.0]
    for eps in (0.0, 0.1, 0.3, 0.6, 1.0):
        tilted = [x3[i] + eps * x1[i] for i in range(3)]
        cert = certified_symmetric_spectral_gap(
            C3, [1.0, 1.0, 1.0], subdominant_vectors=[x2, tilted]
        )
        assert cert.partner_lower <= 1.0 + 1e-9  # never over-estimates lambda_3
        assert cert.subdominant_ratio_upper >= 0.25 - 1e-9  # never overclaims gap


def test_symmetric_gap_deflation_noop_with_single_vector() -> None:
    # Fewer than two partner vectors -> no deflation (identical to plain power sum).
    plain = certified_symmetric_spectral_gap(C3, [1.0, 1.0, 1.0])
    one = certified_symmetric_spectral_gap(
        C3, [1.0, 1.0, 1.0], subdominant_vectors=[[1.0, -1.0, 0.0]]
    )
    assert one.subdominant_ratio_upper == pytest.approx(plain.subdominant_ratio_upper)
    assert one.partner_lower == 0.0


def test_symmetric_gap_chain_deflation_tightens_slow_spectrum() -> None:
    # diag(4, 3, 2, 1): lambda_1 / lambda_0 = 3/4. Deflating only one partner
    # (lambda_2 = 2) leaves the lambda_3 = 1 mode polluting the tail:
    # sqrt(30 - 16 - 4) / 4 = sqrt(10)/4 ~ 0.79 (loose).  The full chain deflates
    # lambda_2 AND lambda_3 -> sqrt(30 - 16 - 4 - 1)/4 = 3/4, the *exact* ratio.
    perron = _E4[0]
    one = certified_symmetric_spectral_gap(
        D4, perron, subdominant_vectors=[_E4[1], _E4[2]]
    )
    chain = certified_symmetric_spectral_gap(
        D4, perron, subdominant_vectors=[_E4[1], _E4[2], _E4[3]]
    )
    assert one.partners_deflated == 1
    assert chain.partners_deflated == 2
    assert one.subdominant_ratio_upper == pytest.approx(math.sqrt(10.0) / 4.0, abs=1e-3)
    # the chain collapses onto the exact 3/4 and is strictly tighter than one partner
    assert chain.subdominant_ratio_upper == pytest.approx(0.75, abs=1e-6)
    assert chain.subdominant_ratio_upper < one.subdominant_ratio_upper - 1e-3
    assert chain.subdominant_ratio_upper >= 0.75 - 1e-9  # never underclaims the truth
    assert chain.spectral_gap_lower == pytest.approx(math.log(4.0 / 3.0), abs=1e-6)


def test_symmetric_gap_chain_deflation_rank_deficient_is_safe() -> None:
    # A duplicate (linearly dependent) partner makes the frame's Gram matrix
    # singular; the g_min positive-definite guard returns ell = 0 for that frame, so
    # the tail is never over-deflated and the ratio stays a rigorous over-estimate.
    cert = certified_symmetric_spectral_gap(
        D4, _E4[0], subdominant_vectors=[_E4[1], _E4[1]]
    )
    assert cert.partners_deflated == 0  # the singular frame deflated nothing
    assert cert.partner_lower == 0.0
    assert cert.subdominant_ratio_upper >= 0.75 - 1e-9  # still rigorous (just looser)


def test_symmetric_gap_two_sided_bracket_contains_truth() -> None:
    # A2: lambda = 3, 1; true gap = ln 3. Given the subdominant eigenvector the
    # certificate brackets m in [gap_lower, gap_upper] around the exact gap, with a
    # rigorous lambda_0 upper bound and lambda_1 lower (Ritz) bound.
    cert = certified_symmetric_spectral_gap(
        A2, [1.0, 1.0], subdominant_vectors=[[1.0, -1.0]]
    )
    assert cert.spectral_gap_upper < math.inf
    assert cert.spectral_gap_lower <= math.log(3.0) + 1e-9  # lower never overclaims
    assert cert.spectral_gap_upper >= math.log(3.0) - 1e-9  # upper never underclaims
    assert cert.spectral_gap_lower <= cert.spectral_gap_upper + 1e-12  # ordered bracket
    assert cert.perron_upper >= 3.0 - 1e-9  # rigorous upper bound on lambda_0 = 3
    assert cert.subdominant_lower <= 1.0 + 1e-9  # rigorous lower bound on lambda_1 = 1
    assert cert.subdominant_lower == pytest.approx(1.0, abs=1e-3)


def test_symmetric_gap_no_upper_bracket_without_vectors() -> None:
    # Without subdominant vectors there is no certified lambda_1 lower bound, so the
    # upper gap is +inf (one-sided lower bound only) -- backward compatible.
    cert = certified_symmetric_spectral_gap(A2, [1.0, 1.0])
    assert cert.spectral_gap_upper == math.inf
    assert cert.spectral_gap_upper_per_unit == math.inf
    assert cert.subdominant_lower == 0.0
    assert cert.perron_upper >= 3.0 - 1e-9  # lambda_0 upper needs no eigenvector hint


def test_symmetric_gap_upper_bracket_respects_spacing() -> None:
    a = 0.5
    cert = certified_symmetric_spectral_gap(
        A2, [1.0, 1.0], subdominant_vectors=[[1.0, -1.0]], lattice_spacing=a
    )
    assert cert.spectral_gap_upper_per_unit == pytest.approx(
        cert.spectral_gap_upper / a, rel=1e-12
    )


def test_symmetric_gap_respects_lattice_spacing() -> None:
    a = 0.25
    cert = certified_symmetric_spectral_gap(A2, [1.0, 1.0], lattice_spacing=a)
    assert cert.spectral_gap_lower_per_unit == pytest.approx(
        cert.spectral_gap_lower / a, rel=1e-12
    )


def test_symmetric_gap_invalid_spacing_raises() -> None:
    with pytest.raises(ValueError):
        certified_symmetric_spectral_gap(A2, [1.0, 1.0], lattice_spacing=0.0)


# --------------------------------------------------------------------------- #
# Block-operator (finite-section + tail) conditional coercivity gap            #
# --------------------------------------------------------------------------- #
def test_block_gap_matches_exact_2x2_min_and_is_a_lower_bound() -> None:
    # A 1x1 finite block a, coupling b, tail d gives exactly lambda_min([[a,b],[b,d]]).
    for a, b, d in [(2.0, 0.5, 2.0), (1.5, 0.3, 3.0), (0.8, 0.1, 0.8)]:
        cert = certified_block_operator_gap([[a]], coupling_norm_upper=b, tail_gap_lower=d)
        truth = _exact_2x2_min(a, b, d)
        assert cert.gap_lower is not None
        assert cert.gap_lower <= truth + 1e-12  # rigorous lower bound
        assert cert.gap_lower == pytest.approx(truth, abs=1e-9)  # and tight for 1x1
        assert cert.coercive is (truth > 0.0)


def test_block_gap_threshold_is_b_squared_over_a() -> None:
    cert = certified_block_operator_gap([[2.0]], coupling_norm_upper=1.0, tail_gap_lower=2.0)
    # coercive iff tail gap d > b^2/a = 0.5; here d = 2 > 0.5.
    assert cert.threshold_tail_gap == pytest.approx(0.5, abs=1e-12)
    assert cert.coercive is True
    # exactly at/above the threshold separates coercive from not.
    below = certified_block_operator_gap([[2.0]], coupling_norm_upper=1.0, tail_gap_lower=0.4)
    assert below.coercive is False  # d = 0.4 < 0.5 threshold


def test_block_gap_threshold_only_when_no_tail_hypothesis() -> None:
    cert = certified_block_operator_gap([[3.0]], coupling_norm_upper=1.2)
    assert cert.gap_lower is None
    assert cert.tail_gap_lower is None
    assert cert.tail_is_hypothesis is True  # never claims to prove the tail
    assert cert.threshold_tail_gap == pytest.approx(1.2 * 1.2 / 3.0, abs=1e-12)


def test_block_gap_non_coercive_finite_block_has_infinite_threshold() -> None:
    # If the finite block itself is not positive (Gershgorin a <= 0) no tail saves it.
    cert = certified_block_operator_gap(
        [[0.1, 0.5], [0.5, 0.1]], coupling_norm_upper=0.2, tail_gap_lower=10.0
    )
    assert cert.finite_gap_lower <= 0.0  # Gershgorin: 0.1 - 0.5 < 0
    assert cert.threshold_tail_gap == math.inf
    assert cert.coercive is False


def test_block_gap_multidim_finite_block_uses_gershgorin() -> None:
    cert = certified_block_operator_gap(
        [[3.0, 0.2], [0.2, 2.5]], coupling_norm_upper=0.4, tail_gap_lower=2.0
    )
    assert cert.finite_gap_lower == pytest.approx(2.3, abs=1e-9)  # 2.5 - 0.2
    assert cert.coercive is True
    assert cert.gap_lower is not None and cert.gap_lower > 0.0


def test_block_gap_rejects_negative_coupling() -> None:
    with pytest.raises(ValueError):
        certified_block_operator_gap([[1.0]], coupling_norm_upper=-0.1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The transfer-matrix constructors must enclose the spectrum they claim.

Each construction has a *closed-form* spectrum -- ``e^{-t n^2}`` for ``U(1)``,
``e^{-t C2(R)}`` from an exact rational Casimir for ``su(N)``, ``I_m(beta)`` for the
``SU(2)`` Wilson character expansion -- so these tests compare the enclosures
against numerically diagonalising the matrix, and the exact gaps against their
closed forms (``t``, ``3t/4``, ``4t/3``).  Getting a Casimir normalisation or a
Bessel order wrong shows up immediately.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from omnibias.geometry.gauge.transfer.matrices import (
    TransferMatrix,
    decode_scalar,
    encode_scalar,
    rebuild,
    su2_heat_kernel_transfer,
    su2_wilson_transfer,
    su3_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)

mpmath = pytest.importorskip("mpmath")

COUPLING = 0.8


def _midpoints(transfer: TransferMatrix) -> np.ndarray:
    return np.array([[0.5 * (c.lo + c.hi) for c in row] for row in transfer.entries])


def _numeric_spectrum(transfer: TransferMatrix) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(_midpoints(transfer)))[::-1]


def _all_transfers() -> list[TransferMatrix]:
    return [
        u1_heat_kernel_transfer(COUPLING, n_max=3),
        u1_heat_kernel_transfer(COUPLING, n_max=3, basis="angle"),
        su2_heat_kernel_transfer(COUPLING, max_dynkin=4),
        su3_heat_kernel_transfer(COUPLING, max_dynkin=2),
        su2_wilson_transfer(2.0, n_modes=6),
    ]


# --------------------------------------------------------------------------- #
# the declared spectrum is the actual spectrum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transfer", _all_transfers(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_declared_spectrum_matches_a_numerical_diagonalisation(
    transfer: TransferMatrix,
) -> None:
    assert transfer.exact_eigenvalues is not None
    claimed = sorted((0.5 * (e.lo + e.hi) for e in transfer.exact_eigenvalues), reverse=True)
    np.testing.assert_allclose(_numeric_spectrum(transfer), claimed, atol=1e-12)


@pytest.mark.parametrize("transfer", _all_transfers(), ids=lambda t: f"{t.model}-{t.basis}")
def test_every_eigenvalue_enclosure_contains_its_true_value(
    transfer: TransferMatrix,
) -> None:
    """Containment, up to the *eigensolver's* own error rather than ours.

    The enclosures are rigorous for the exact matrix and only a few ulp wide, while
    LAPACK is run on the enclosure midpoints and carries its own rounding, so it can
    land an ulp or two outside a correct enclosure. The tolerance below is a
    statement about ``eigvalsh``, not slack in the enclosure.
    """
    assert transfer.exact_eigenvalues is not None
    solver_error = 1e-14
    for value, truth in zip(
        transfer.exact_eigenvalues, _numeric_spectrum(transfer), strict=True
    ):
        assert value.lo - solver_error <= truth <= value.hi + solver_error


@pytest.mark.parametrize("transfer", _all_transfers(), ids=lambda t: f"{t.model}-{t.basis}")
def test_the_matrix_is_symmetric_and_the_spectrum_is_ordered(
    transfer: TransferMatrix,
) -> None:
    matrix = _midpoints(transfer)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
    assert transfer.exact_eigenvalues is not None
    values = [0.5 * (e.lo + e.hi) for e in transfer.exact_eigenvalues]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------- #
# the closed-form gaps
# --------------------------------------------------------------------------- #
def _exact_gap(transfer: TransferMatrix) -> float:
    ratio = transfer.exact_subdominant_ratio()
    assert ratio is not None
    return -math.log(0.5 * (ratio.lo + ratio.hi))


def test_the_u1_gap_is_the_coupling() -> None:
    """``lambda_n = e^{-t n^2}``, so ``m a = -ln(e^{-t} / 1) = t`` exactly."""
    assert _exact_gap(u1_heat_kernel_transfer(COUPLING, n_max=4)) == pytest.approx(
        COUPLING, rel=1e-12
    )


def test_the_su2_gap_is_three_quarters_of_the_coupling() -> None:
    """``C2`` of the ``su(2)`` fundamental is ``3/4`` (spin ``j(j+1)`` at ``j = 1/2``)."""
    assert _exact_gap(
        su2_heat_kernel_transfer(COUPLING, max_dynkin=4)
    ) == pytest.approx(0.75 * COUPLING, rel=1e-12)


def test_the_su3_gap_is_four_thirds_of_the_coupling() -> None:
    """``C2`` of the ``su(3)`` fundamental is ``4/3``."""
    assert _exact_gap(
        su3_heat_kernel_transfer(COUPLING, max_dynkin=2)
    ) == pytest.approx(4.0 * COUPLING / 3.0, rel=1e-12)


def test_the_su3_subdominant_mode_is_doubly_degenerate() -> None:
    """``(1,0)`` and ``(0,1)`` are conjugates sharing ``C2 = 4/3`` exactly.

    This is the degeneracy that costs an undeflated power-sum bound a ``sqrt(2)``,
    and the reason the gap engine takes a partner chain at all.
    """
    transfer = su3_heat_kernel_transfer(COUPLING, max_dynkin=2)
    assert transfer.exact_eigenvalues is not None
    first, second = transfer.exact_eigenvalues[1], transfer.exact_eigenvalues[2]
    assert first.lo <= second.hi and second.lo <= first.hi
    assert "C2 4/3" in transfer.mode_labels[1]
    assert "C2 4/3" in transfer.mode_labels[2]


def test_the_su2_spectrum_is_not_degenerate() -> None:
    """The contrast case: ``su(2)`` irreps are self-conjugate, so no pairing."""
    transfer = su2_heat_kernel_transfer(COUPLING, max_dynkin=4)
    assert transfer.exact_eigenvalues is not None
    values = [0.5 * (e.lo + e.hi) for e in transfer.exact_eigenvalues]
    assert all(b < a for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("beta", [0.5, 2.0, 5.0])
def test_the_wilson_entries_are_the_modified_bessel_functions(beta: float) -> None:
    transfer = su2_wilson_transfer(beta, n_modes=5)
    assert transfer.exact_eigenvalues is not None
    for m, value in enumerate(transfer.exact_eigenvalues, start=1):
        with mpmath.workdps(50):
            truth = float(mpmath.besseli(m, beta))
        assert value.lo <= truth <= value.hi


@pytest.mark.parametrize("beta", [0.7, 2.0, 5.0])
@pytest.mark.parametrize("theta", [0.3, 1.1, 2.7])
def test_the_wilson_character_expansion_reproduces_the_weight(
    beta: float, theta: float
) -> None:
    r"""``e^{beta cos t} = (2/beta) sum_m m I_m(beta) sin(m t)/sin t``.

    The identity the Wilson constructor rests on. If it did not hold, the entries
    would not be the character coefficients they are labelled as.
    """
    with mpmath.workdps(50):
        series = sum(
            (2.0 / beta) * m * float(mpmath.besseli(m, beta)) * math.sin(m * theta) / math.sin(theta)
            for m in range(1, 120)
        )
    assert series == pytest.approx(math.exp(beta * math.cos(theta)), rel=1e-10)


def test_the_wilson_tail_is_no_lighter_than_a_heat_kernel_at_the_same_gap() -> None:
    r"""Compared where the comparison is meaningful: same dimension, same leading gap.

    The quantity that matters to a power-sum bound is the tail behind the
    subdominant mode, ``sum_{i >= 2} r_i^2 / r_1^2``. Measured at ``beta = 5`` and
    the ``su(2)`` heat-kernel coupling that reproduces the same gap, Wilson's tail
    is heavier -- but only modestly (about ``0.45`` against ``0.42``), so this
    asserts what is true rather than the dramatic separation one might expect.
    Comparing at *different* gaps would inflate the difference meaninglessly.
    """
    wilson = su2_wilson_transfer(5.0, n_modes=6)
    gap = _exact_gap(wilson)
    heat = su2_heat_kernel_transfer(gap / 0.75, max_dynkin=5)  # su(2) gap is 3t/4
    assert _exact_gap(heat) == pytest.approx(gap, rel=1e-9)
    assert wilson.dimension == heat.dimension

    def tail_weight(transfer: TransferMatrix) -> float:
        values = [0.5 * (e.lo + e.hi) for e in transfer.exact_eigenvalues or ()]
        ratios = [v / values[0] for v in values]
        return sum(r * r for r in ratios[2:]) / ratios[1] ** 2

    assert tail_weight(wilson) > tail_weight(heat)


# --------------------------------------------------------------------------- #
# the angle basis
# --------------------------------------------------------------------------- #
def test_the_angle_basis_is_dense_positive_and_isospectral() -> None:
    """The Birkhoff-Hopf anchor: a non-diagonal positive matrix with known truth."""
    character = u1_heat_kernel_transfer(COUPLING, n_max=3)
    angle = u1_heat_kernel_transfer(COUPLING, n_max=3, basis="angle")
    assert angle.entrywise_positive
    assert not character.entrywise_positive  # diagonal, so it has exact zeros
    off_diagonal = [
        abs(0.5 * (angle.entries[i][j].lo + angle.entries[i][j].hi))
        for i in range(angle.dimension)
        for j in range(angle.dimension)
        if i != j
    ]
    assert min(off_diagonal) > 0.0  # genuinely dense
    np.testing.assert_allclose(
        _numeric_spectrum(angle), _numeric_spectrum(character), atol=1e-12
    )


def test_a_weakly_coupled_angle_basis_loses_positivity_honestly() -> None:
    """At small ``t`` the circulant is no longer entrywise positive, and says so."""
    angle = u1_heat_kernel_transfer(0.05, n_max=3, basis="angle")
    assert not angle.entrywise_positive


# --------------------------------------------------------------------------- #
# inputs, replay and validation
# --------------------------------------------------------------------------- #
def test_a_rational_coupling_stays_exact_through_a_round_trip() -> None:
    assert decode_scalar(encode_scalar(Fraction(4, 5))) == Fraction(4, 5)
    assert decode_scalar(encode_scalar(0.8)) == 0.8
    assert isinstance(decode_scalar(encode_scalar(Fraction(4, 5))), Fraction)


@pytest.mark.parametrize("transfer", _all_transfers(), ids=lambda t: f"{t.model}-{t.basis}")
def test_a_matrix_can_be_rebuilt_from_its_recorded_parameters(
    transfer: TransferMatrix,
) -> None:
    """The replay path: the parameters alone must reproduce the matrix bit for bit."""
    again = rebuild(transfer.parameters)
    assert again.model == transfer.model
    assert again.basis == transfer.basis
    assert again.entries == transfer.entries


def test_rebuilding_from_an_unknown_builder_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown transfer-matrix builder"):
        rebuild({"builder": "not_a_builder", "coupling": "1.0"})


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: u1_heat_kernel_transfer(0.0, n_max=2), "coupling must be > 0"),
        (lambda: u1_heat_kernel_transfer(0.5, n_max=0), "n_max must be >= 1"),
        (lambda: u1_heat_kernel_transfer(0.5, basis="fourier"), "basis must be"),  # type: ignore[arg-type]
        (lambda: su2_heat_kernel_transfer(-1.0), "coupling must be > 0"),
        (lambda: su3_heat_kernel_transfer(0.5, max_dynkin=0), "max_dynkin must be >= 1"),
        (lambda: su2_wilson_transfer(0.0), "beta must be > 0"),
        (lambda: su2_wilson_transfer(2.0, n_modes=1), "n_modes must be >= 2"),
    ],
)
def test_invalid_inputs_are_refused(call: object, match: str) -> None:
    assert callable(call)
    with pytest.raises(ValueError, match=match):
        call()

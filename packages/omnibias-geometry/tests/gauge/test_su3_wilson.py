# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(3) Wilson Haar transfer: sample containment, gap soundness, honesty."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.geometry.gauge._core.representation import Irrep, character
from omnibias.geometry.gauge.transfer.certificates import (
    seal_transfer_gap_certificate,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import rebuild, su3_wilson_transfer
from omnibias.geometry.gauge.transfer.su3_wilson import (
    HAAR_VOLUME,
    su3_wilson_haar_coefficient,
)

BETA = 1.0
N_CELLS = 8
DYNKIN = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2))


def _haar_density(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    s1 = np.sin((theta - phi) / 2.0)
    s2 = np.sin((2.0 * theta + phi) / 2.0)
    s3 = np.sin((theta + 2.0 * phi) / 2.0)
    return 64.0 * s1**2 * s2**2 * s3**2


def _re_fund(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.cos(theta) + np.cos(phi) + np.cos(theta + phi)


def _im_fund(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.sin(theta) + np.sin(phi) - np.sin(theta + phi)


def _chi(dynkin: tuple[int, int], theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    re_chi = _re_fund(theta, phi)
    imag = _im_fund(theta, phi)
    if dynkin == (0, 0):
        return np.ones_like(theta, dtype=np.float64)
    if dynkin in ((1, 0), (0, 1)):
        return re_chi
    if dynkin == (1, 1):
        return re_chi**2 + imag**2 - 1.0
    if dynkin in ((2, 0), (0, 2)):
        return re_chi**2 - imag**2 - re_chi
    if dynkin in ((2, 1), (1, 2)):
        amp = re_chi**2 - imag**2 - re_chi
        bim = 2.0 * re_chi * imag + imag
        return amp * re_chi + bim * imag - re_chi
    if dynkin == (2, 2):
        amp = re_chi**2 - imag**2 - re_chi
        bim = 2.0 * re_chi * imag + imag
        return amp**2 + bim**2 - re_chi**2 - imag**2
    raise ValueError(dynkin)


def _numerical_haar_coefficient(
    dynkin: tuple[int, int],
    beta: float,
    *,
    n_grid: int = 256,
) -> float:
    """Independent torus quadrature of the unnormalized coefficient ``∫ χ w ρ``."""
    axis = (2.0 * math.pi) * (np.arange(n_grid) + 0.5) / n_grid
    theta, phi = np.meshgrid(axis, axis, indexing="ij")
    weight = np.exp((beta / 3.0) * _re_fund(theta, phi))
    area = (2.0 * math.pi / n_grid) ** 2
    return float(np.sum(_chi(dynkin, theta, phi) * weight * _haar_density(theta, phi)) * area)


def test_haar_volume_is_the_locked_weyl_normalisation() -> None:
    assert HAAR_VOLUME.contains(24.0 * math.pi**2)


@pytest.mark.parametrize("dynkin", DYNKIN)
def test_enclosure_contains_a_numerical_haar_sample(dynkin: tuple[int, int]) -> None:
    enclosure = su3_wilson_haar_coefficient(dynkin, BETA, n_cells=N_CELLS)
    sample = _numerical_haar_coefficient(dynkin, BETA)
    assert enclosure.contains(sample), (dynkin, enclosure, sample)


def test_locked_characters_match_weyl_bialternant_away_from_walls() -> None:
    theta, phi = 0.7, 0.4
    torus = np.array(
        [
            np.exp(1j * theta),
            np.exp(1j * phi),
            np.exp(-1j * (theta + phi)),
        ],
        dtype=np.complex128,
    )
    re_chi = _re_fund(np.array(theta), np.array(phi))
    imag = _im_fund(np.array(theta), np.array(phi))
    for dynkin in DYNKIN:
        if dynkin == (0, 0):
            continue
        got = character(Irrep(n=3, dynkin=dynkin), torus)
        pred = float(_chi(dynkin, np.array(theta), np.array(phi)))
        assert got.real == pytest.approx(pred, abs=1e-12)
    fund = character(Irrep(n=3, dynkin=(1, 0)), torus)
    assert fund.real == pytest.approx(float(re_chi), abs=1e-12)
    assert fund.imag == pytest.approx(float(imag), abs=1e-12)


def test_transfer_is_diagonal_character_truncation() -> None:
    transfer = su3_wilson_transfer(BETA, n_cells=N_CELLS)
    assert transfer.model == "su3_wilson"
    assert transfer.basis == "character"
    assert transfer.exact_eigenvalues is not None
    assert len(transfer.exact_eigenvalues) == 4
    for i, row in enumerate(transfer.entries):
        for j, entry in enumerate(row):
            if i != j:
                assert entry.lo == 0.0 and entry.hi == 0.0


def test_certified_gap_never_exceeds_midpoint_eigengap() -> None:
    transfer = su3_wilson_transfer(BETA, n_cells=N_CELLS)
    mid = np.array([[0.5 * (c.lo + c.hi) for c in row] for row in transfer.entries])
    values = np.sort(np.linalg.eigvalsh(mid))[::-1]
    numerical_gap = -math.log(abs(values[1]) / values[0]) if values[0] > 0.0 else 0.0
    result = certified_transfer_matrix_gap(transfer)
    assert result.spectral_gap_lower <= numerical_gap + 1e-12


def test_honesty_flags_are_hard_wired_false() -> None:
    transfer = su3_wilson_transfer(BETA, n_cells=N_CELLS)
    result = certified_transfer_matrix_gap(transfer)
    sealed = dict(seal_transfer_gap_certificate(result, transfer, claim="su3 wilson"))
    assert sealed["continuum_claim"] is False
    assert sealed["honesty"]["continuum_claim"] is False
    assert sealed["honesty"]["yang_mills_claim"] is False
    assert sealed["honesty"]["fixed_matrix"] is True
    forged = dict(sealed)
    forged["continuum_claim"] = True
    forged["honesty"] = dict(sealed["honesty"], continuum_claim=True, yang_mills_claim=True)
    errors = transfer_gap_schema_errors(forged)
    assert any("continuum_claim" in err for err in errors)
    assert any("yang_mills_claim" in err for err in errors)


def test_rebuild_round_trips() -> None:
    transfer = su3_wilson_transfer(BETA, n_cells=N_CELLS)
    again = rebuild(transfer.parameters)
    assert again.model == transfer.model
    assert again.entries == transfer.entries


def test_max_dynkin_two_is_unlocked() -> None:
    transfer = su3_wilson_transfer(BETA, max_dynkin=2, n_cells=N_CELLS)
    assert transfer.exact_eigenvalues is not None
    assert len(transfer.exact_eigenvalues) == 9


def test_max_dynkin_and_beta_are_locked() -> None:
    with pytest.raises(ValueError, match="max_dynkin"):
        su3_wilson_transfer(BETA, max_dynkin=3)
    with pytest.raises(ValueError, match="beta must be > 0"):
        su3_wilson_transfer(0.0)
    with pytest.raises(ValueError, match="p,q <= 2"):
        su3_wilson_haar_coefficient((3, 0), BETA, n_cells=N_CELLS)

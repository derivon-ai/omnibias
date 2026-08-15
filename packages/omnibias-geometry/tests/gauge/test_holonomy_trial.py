# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomy trial spaces: soundness, gauge invariance, honesty, kernel flag."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from omnibias.core.proof import Conjecture
from omnibias.core.proof.lean_check import check_certificate, generate_obligation
from omnibias.core.verified.interval import Interval
from omnibias.geometry.gauge.band._core import su2_transverse_constant
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    TRANSFER_GAP_KIND,
    replay_transfer_matrix_gap,
    seal_transfer_gap_certificate,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import (
    TransferMatrix,
    su2_class_angle_transfer,
    su2_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)
from omnibias.geometry.gauge.transfer.trial import (
    GRAM_COND_THRESHOLD,
    Loop,
    holonomy_trial_space,
    su2_holonomy_trace,
)


def _true_gap(transfer: TransferMatrix) -> float:
    ratio = transfer.exact_subdominant_ratio()
    assert ratio is not None
    return -math.log(0.5 * (ratio.lo + ratio.hi))


def _dense_suite() -> list[TransferMatrix]:
    suite: list[TransferMatrix] = []
    for n_max in (2, 3, 4):
        for coupling in (0.4, 0.6, 0.8, 1.0):
            suite.append(
                u1_heat_kernel_transfer(coupling, n_max=n_max, basis="angle")
            )
    for max_dynkin in (3, 4, 5):
        for coupling in (0.5, 0.8, 1.0):
            suite.append(su2_class_angle_transfer(coupling, max_dynkin=max_dynkin))
    return suite


def _synthetic_spd(evals: tuple[float, ...], seed: int) -> tuple[TransferMatrix, float]:
    n = len(evals)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    q, _ = np.linalg.qr(raw)
    mid = q @ np.diag(evals) @ q.T
    mid = 0.5 * (mid + mid.T)
    entries = tuple(tuple(Interval.point(float(mid[i, j])) for j in range(n)) for i in range(n))
    transfer = TransferMatrix(
        model="synthetic_spd",
        basis="angle",
        entries=entries,
        mode_labels=tuple(f"e{i}" for i in range(n)),
        exact_eigenvalues=None,
        parameters={"builder": "synthetic", "coupling": "1"},
        perron_vector=tuple(float(q[i, 0]) for i in range(n)),
        subdominant_vectors=tuple(
            tuple(float(q[i, j]) for i in range(n)) for j in range(1, n)
        ),
        symmetric=True,
    )
    ordered = sorted((float(v) for v in evals), reverse=True)
    true_gap = -math.log(ordered[1] / ordered[0])
    return transfer, true_gap


def test_holonomy_trial_space_reports_gram_condition() -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=4)
    trial = holonomy_trial_space(transfer)
    assert trial.dim == transfer.dimension
    assert trial.gram_condition < GRAM_COND_THRESHOLD
    assert trial.flagged is False
    assert trial.remainder_width == 0.0


def test_u1_angle_fourier_family_is_well_conditioned() -> None:
    transfer = u1_heat_kernel_transfer(0.8, n_max=3, basis="angle")
    trial = holonomy_trial_space(transfer)
    assert trial.flagged is False
    assert trial.gram_condition < GRAM_COND_THRESHOLD


def test_g2_trial_bound_never_exceeds_true_gap() -> None:
    cases = [
        u1_heat_kernel_transfer(0.8, n_max=3, basis="angle"),
        su2_class_angle_transfer(0.8, max_dynkin=4),
        su2_heat_kernel_transfer(0.8, max_dynkin=4),
    ]
    for transfer in cases:
        trial = holonomy_trial_space(transfer)
        result = certified_transfer_matrix_gap(transfer, trial=trial)
        assert result.spectral_gap_lower <= _true_gap(transfer) + 1e-9
        assert result.trial_gram_condition is not None
        assert result.trial_flagged is False


def test_g2_synthetic_spd_never_exceeds_true_gap() -> None:
    for seed in range(32):
        transfer, true_gap = _synthetic_spd((1.0, 0.55, 0.3, 0.12), seed)
        trial = holonomy_trial_space(transfer)
        result = certified_transfer_matrix_gap(transfer, trial=trial)
        assert result.spectral_gap_lower <= true_gap + 1e-8
        if result.certified:
            assert 0.0 <= result.subdominant_ratio_upper < 1.0


def test_g3_su2_trace_is_gauge_invariant() -> None:
    components = (0.3, -0.2, 0.5)
    bare = su2_holonomy_trace(components, length=1.0, coupling=1.0)
    u00, u01, u10, u11 = su2_transverse_constant(
        components, length=1.0, coupling=1.0
    )
    rng = random.Random(1)
    for _ in range(8):
        axis = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        g00, g01, g10, g11 = su2_transverse_constant(axis, length=1.2, coupling=0.8)
        gd00, gd01, gd10, gd11 = (
            g00.conjugate(),
            g10.conjugate(),
            g01.conjugate(),
            g11.conjugate(),
        )
        w00 = gd00 * u00 + gd01 * u10
        w01 = gd00 * u01 + gd01 * u11
        w10 = gd10 * u00 + gd11 * u10
        w11 = gd10 * u01 + gd11 * u11
        t00 = w00 * g00 + w01 * g10
        t11 = w10 * g01 + w11 * g11
        transformed = float((t00 + t11).real)
        rel = abs(transformed - bare) / max(abs(bare), 1e-15)
        assert rel < 1e-14


def test_g4_duplicate_loops_are_flagged() -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    loops = (Loop(winding=0), Loop(winding=0), Loop(winding=1))
    trial = holonomy_trial_space(transfer, loops)
    assert trial.flagged is True
    result = certified_transfer_matrix_gap(transfer, trial=trial)
    assert result.trial_flagged is True
    assert result.trial_gram_condition is not None
    assert result.spectral_gap_lower <= _true_gap(transfer) + 1e-9


def test_g4_magnus_remainder_enters_the_result_width() -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    loops = tuple(Loop(winding=index, regime="magnus") for index in range(transfer.dimension))
    trial = holonomy_trial_space(transfer, loops)
    assert trial.remainder_width > 0.0
    result = certified_transfer_matrix_gap(transfer, trial=trial)
    assert result.trial_remainder_width == pytest.approx(trial.remainder_width)
    assert result.spectral_gap_lower <= _true_gap(transfer) + 1e-9


def test_g5_honesty_flags_stay_false_with_trial() -> None:
    transfer = su2_class_angle_transfer(0.8, max_dynkin=3)
    trial = holonomy_trial_space(transfer)
    result = certified_transfer_matrix_gap(transfer, trial=trial)
    cert = seal_transfer_gap_certificate(result, transfer)
    assert transfer_gap_schema_errors(cert) == []
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["trial_flagged"] is False
    assert cert["trial_gram_condition"] == pytest.approx(trial.gram_condition)
    assert replay_transfer_matrix_gap(cert) is True


def test_g6_kernel_flag_cannot_be_forged() -> None:
    transfer = su2_heat_kernel_transfer(0.8, max_dynkin=3)
    trial = holonomy_trial_space(transfer)
    result = certified_transfer_matrix_gap(transfer, trial=trial)
    cert = seal_transfer_gap_certificate(result, transfer)
    source = generate_obligation(cert)
    assert source is not None
    assert "spectral_gap_pos" in source
    checked = check_certificate(cert)
    if checked.available:
        assert checked.verified is True
    else:
        assert checked.verified is False
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "forged formal",
            TRANSFER_GAP_KIND,
            {"parameters": dict(transfer.parameters)},
            claims={"theorem_prover_verified": True},
        ),
        lean_check=True,
    )
    if not checked.available:
        assert verdict.status == "BLOCKED"
        assert verdict.theorem_prover_verified is False
    else:
        assert verdict.theorem_prover_verified is True


def test_g1_holonomy_bound_is_at_least_generic_on_dense_bases() -> None:
    suite = _dense_suite()
    assert len(suite) >= 20
    factors: list[float] = []
    fractions: list[float] = []
    for transfer in suite:
        generic = certified_transfer_matrix_gap(transfer)
        trial = holonomy_trial_space(transfer)
        holonomy = certified_transfer_matrix_gap(transfer, trial=trial)
        assert holonomy.spectral_gap_lower + 1e-12 >= generic.spectral_gap_lower
        assert holonomy.spectral_gap_lower <= _true_gap(transfer) + 1e-9
        assert holonomy.trial_flagged is False
        if generic.spectral_gap_lower > 0.0:
            factors.append(holonomy.spectral_gap_lower / generic.spectral_gap_lower)
        true = _true_gap(transfer)
        if true > 0.0:
            fractions.append(holonomy.spectral_gap_lower / true)
    assert factors
    assert min(factors) >= 1.0 - 1e-12
    assert min(fractions) > 0.0

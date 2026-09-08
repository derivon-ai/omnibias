# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-32: open-system Lindblad dynamics, float gates G1/G5."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from omnibias.core.collapse.einselection import DephasingModel, reduced_density_matrix
from omnibias.core.lindblad import (
    LindbladModel,
    density_matrix,
    honesty_payload,
    qubit_bloch_solution,
    qubit_pure_dephasing,
    qubit_thermal,
    steady_state,
    thermal_steady_population,
    time_derivative_tower,
)
from omnibias.core.occupancy import FermiModel, occupancy
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

mpmath = pytest.importorskip("mpmath")

_EQUAL = 1.0 / math.sqrt(2.0)
_RHO_PLUS = ((0.5 + 0.0j, 0.5 + 0.0j), (0.5 + 0.0j, 0.5 + 0.0j))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def test_g1_pure_dephasing_matches_einselection_to_1e_14() -> None:
    model = qubit_pure_dephasing(rate=1.0)
    dephasing = DephasingModel(
        amplitudes=(ComplexInterval.point(_EQUAL), ComplexInterval.point(_EQUAL)),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    for time in (0.0, 0.5, 1.0, 2.0, 5.0):
        got = density_matrix(model, _RHO_PLUS, time)
        enclosed = reduced_density_matrix(dephasing, time)
        for i in range(2):
            for j in range(2):
                mid = complex(enclosed[i][j].re.mid, enclosed[i][j].im.mid)
                assert abs(got[i, j] - mid) < 1e-14


def test_g1_amplitude_damping_matches_qubit_bloch_closed_form() -> None:
    omega, gamma_down, beta = 1.25, 0.4, 0.8
    model = qubit_thermal(omega=omega, beta=beta, gamma_down=gamma_down, gamma_phi=0.1)
    gamma_up = gamma_down * math.exp(-beta * omega)
    rho0 = np.array([[0.2, 0.1 - 0.05j], [0.1 + 0.05j, 0.8]], dtype=np.complex128)
    for time in (0.0, 0.25, 1.0, 3.0):
        got = density_matrix(model, rho0, time)
        closed = qubit_bloch_solution(
            omega=omega,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            gamma_phi=0.1,
            rho0=rho0,
            time=time,
        )
        assert _max_abs(got, closed) < 1e-12


def test_g1_time_derivative_tower_matches_mpmath_diff() -> None:
    mpmath.mp.dps = 40
    omega, gamma_down, beta, gamma_phi = 1.0, 0.5, 1.0, 0.0
    model = qubit_thermal(omega=omega, beta=beta, gamma_down=gamma_down)
    gamma_up = model.rates[1]
    rho0 = np.array(_RHO_PLUS, dtype=np.complex128)
    time = 0.35
    tower = time_derivative_tower(model, rho0, time, order=6)
    t1_inv = gamma_down + gamma_up
    t2_inv = 0.5 * t1_inv + 2.0 * gamma_phi
    c01_0 = mpmath.mpc(0.5, 0.0)

    def _entry(t: object) -> object:
        tt = mpmath.mpf(t)
        c01 = c01_0 * mpmath.exp(mpmath.j * omega * tt) * mpmath.exp(-tt * t2_inv)
        return c01

    for order in range(7):
        truth = mpmath.diff(_entry, time, order)
        got = tower[order][0, 1]
        err = abs(complex(got) - complex(truth))
        assert err < 1e-8, (order, err)

    # Kronecker superoperator vs the GKSL formula.
    from omnibias.core.lindblad import apply_lindblad

    current = density_matrix(model, rho0, time)
    for order in range(1, 7):
        current = np.array(apply_lindblad(model, current))
        assert _max_abs(np.array(tower[order]), current) < 1e-10


def test_g5_thermal_population_matches_occupancy_to_1e_15() -> None:
    omega, beta = 1.3, 0.7
    got = thermal_steady_population(omega=omega, beta=beta)
    ref = occupancy(FermiModel(beta=beta, mu=0.0), omega)
    assert abs(got - ref) < 1e-15
    model = qubit_thermal(omega=omega, beta=beta, gamma_down=0.9)
    ss = steady_state(model)
    assert abs(ss[1, 1].real - got) < 1e-12
    gamma_up = model.rates[1]
    gamma_down = model.rates[0]
    assert abs(gamma_up / gamma_down - math.exp(-beta * omega)) < 1e-15


def test_model_validation_rejects_non_hermitian_and_negative_rates() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        LindbladModel(
            hamiltonian=((0.0, 1.0), (0.0, 0.0)),
            jumps=(),
            rates=(),
        )
    with pytest.raises(ValueError, match=">= 0"):
        qubit_thermal(omega=1.0, beta=1.0, gamma_down=-0.1)


def test_honesty_payload_keys() -> None:
    payload = honesty_payload()
    assert payload["requests_new_collapse_registry_slot"] is True
    assert payload["markovian_model_declared_not_derived"] is True
    for key in (
        "wave_function_collapse_claim",
        "measurement_problem_resolved",
        "single_outcome_claim",
        "born_rule_derived",
        "non_markovian_claim",
        "general_closed_form_claim",
        "thermodynamic_limit_taken",
        "continuum_limit_taken",
        "quantum_advantage_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "float_residual_is_proof",
    ):
        assert payload[key] is False, key


def test_honesty_keys_never_set_true_in_source() -> None:
    import re

    forbidden = (
        "wave_function_collapse_claim",
        "measurement_problem_resolved",
        "single_outcome_claim",
        "born_rule_derived",
        "non_markovian_claim",
        "general_closed_form_claim",
        "thermodynamic_limit_taken",
        "continuum_limit_taken",
        "quantum_advantage_claim",
    )
    repo = Path(__file__).resolve().parents[3]
    sources = [
        repo / "packages/omnibias-core/src/omnibias/core/lindblad.py",
        repo / "packages/omnibias-core/src/omnibias/core/verified/lindblad.py",
        repo / "packages/omnibias-core/src/omnibias/core/collapse/relaxation.py",
    ]
    pattern = re.compile(r'"(' + "|".join(forbidden) + r')"\s*:\s*True')
    for path in sources:
        matches = pattern.findall(path.read_text(encoding="utf-8"))
        assert matches == [], (path, matches)

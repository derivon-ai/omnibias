# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-32: certified Lindblad enclosures, gates G2/G3/G4/G7."""

from __future__ import annotations

import math
import random

from omnibias.core.collapse.einselection import DephasingModel, reduced_density_matrix
from omnibias.core.lindblad import (
    density_matrix,
    dissipative_gap,
    qubit_pure_dephasing,
    qubit_thermal,
    thermal_steady_population,
)
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.lindblad import (
    certified_relaxation_time,
    density_matrix_enclosure,
    hermiticity_residual_enclosure,
    positivity_verdict,
    trace_enclosure,
    trajectory_enclosure,
)

_RNG = random.Random(20260909)
_EQUAL = 1.0 / math.sqrt(2.0)
_RHO_PLUS = ((0.5 + 0.0j, 0.5 + 0.0j), (0.5 + 0.0j, 0.5 + 0.0j))
_EXCITED = ((0.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j))


def _contains_complex(entry: ComplexInterval, value: complex) -> None:
    assert entry.re.contains(value.real), (entry.re, value.real)
    assert entry.im.contains(value.imag), (entry.im, value.imag)


def test_g2_density_matrix_enclosure_contains_grid_and_random() -> None:
    model = qubit_thermal(omega=1.0, beta=0.8, gamma_down=0.3)
    times = (0.0, 0.25, 0.5, 1.0, 2.0)
    for time in times:
        boxed = density_matrix_enclosure(model, _RHO_PLUS, time)
        assert boxed is not None
        truth = density_matrix(model, _RHO_PLUS, time)
        for i in range(2):
            for j in range(2):
                _contains_complex(boxed[i][j], complex(truth[i, j]))
    for _ in range(16):
        t = _RNG.random() * 1.5
        p = 0.1 + 0.8 * _RNG.random()
        rho0 = ((complex(1.0 - p), 0.0j), (0.0j, complex(p)))
        boxed = density_matrix_enclosure(model, rho0, t)
        assert boxed is not None
        truth = density_matrix(model, rho0, t)
        for i in range(2):
            for j in range(2):
                _contains_complex(boxed[i][j], complex(truth[i, j]))


def test_g1_certified_propagator_contains_einselection() -> None:
    model = qubit_pure_dephasing(rate=1.0)
    dephasing = DephasingModel(
        amplitudes=(ComplexInterval.point(_EQUAL), ComplexInterval.point(_EQUAL)),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    boxed = density_matrix_enclosure(model, _RHO_PLUS, 1.25)
    assert boxed is not None
    enclosed = reduced_density_matrix(dephasing, 1.25)
    for i in range(2):
        for j in range(2):
            mid = complex(enclosed[i][j].re.mid, enclosed[i][j].im.mid)
            _contains_complex(boxed[i][j], mid)


def test_g3_trace_hermiticity_and_positivity() -> None:
    model = qubit_thermal(omega=1.0, beta=1.0, gamma_down=0.8)
    mixed = density_matrix(model, _EXCITED, 4.0)
    boxed = density_matrix_enclosure(model, _EXCITED, 4.0)
    assert boxed is not None
    assert trace_enclosure(boxed).contains(1.0)
    assert hermiticity_residual_enclosure(boxed).contains_zero()
    mixed_list = tuple(tuple(complex(mixed[i, j]) for j in range(2)) for i in range(2))
    proved = positivity_verdict(mixed_list)
    assert proved.status == "PROVED"
    blocked = positivity_verdict(_EXCITED)
    assert blocked.status == "BLOCKED"


def test_g4_certified_relaxation_time_is_conservative() -> None:
    model = qubit_thermal(omega=1.0, beta=0.5, gamma_down=2.0)
    eps = 0.2
    gap = dissipative_gap(model)
    assert gap < 0.0
    t_float = math.log(1.0 / eps) / abs(gap)
    report = certified_relaxation_time(model, distance_budget=eps, t_max=20.0)
    assert report.reason == "certified"
    assert report.time is not None
    assert report.time.lo >= t_float - 1e-12
    dephasing = qubit_pure_dephasing(rate=1.0)
    halt = certified_relaxation_time(dephasing, distance_budget=0.1, t_max=5.0)
    assert halt.time is None
    assert halt.reason == "not_unique"


def test_g5_certified_thermal_population_encloses_occupancy() -> None:
    omega, beta = 1.1, 0.9
    model = qubit_thermal(omega=omega, beta=beta, gamma_down=1.5)
    boxed = density_matrix_enclosure(model, _EXCITED, 8.0)
    assert boxed is not None
    occ = thermal_steady_population(omega=omega, beta=beta)
    assert boxed[1][1].re.contains(occ)


def test_g7_lohner_beats_naive_wrapping() -> None:
    model = qubit_thermal(omega=2.0, beta=0.4, gamma_down=0.15)
    lohner = trajectory_enclosure(
        model, _RHO_PLUS, h=0.05, n_steps=24, method="lohner"
    )
    naive = trajectory_enclosure(
        model, _RHO_PLUS, h=0.05, n_steps=24, method="naive"
    )
    assert lohner.width < naive.width

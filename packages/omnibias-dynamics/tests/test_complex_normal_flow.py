# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Complex fixed-time normal-flow precursor for H3."""

from __future__ import annotations

import random
from fractions import Fraction

import mpmath as mp
import pytest
from omnibias.dynamics.complex_normal_flow import (
    complex_epsilon_cell,
    enclose_complex_normal_flow,
    report,
)
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


def _truth(epsilon: mp.mpc) -> tuple[mp.mpc, mp.mpc]:
    ell = mp.mpf(9) / 25

    def field(
        _time: mp.mpf,
        state: list[mp.mpc],
    ) -> list[mp.mpc]:
        v_coord, height = state
        field_f = (
            -ell * epsilon**3
            - 2 * epsilon**2 * v_coord
            - epsilon * v_coord**2
            + epsilon * v_coord**3 / 3
        )
        field_g = -1 + epsilon * (v_coord - 1)
        return [field_f + height * field_g, -v_coord * height]

    solution = mp.odefun(
        field,
        0,
        [-epsilon, 4 * epsilon**3],
        tol=mp.mpf("1e-40"),
        degree=30,
    )
    final = solution(mp.mpf("0.5"))
    return final[0], final[1]


def test_complex_flow_contains_grid_and_random_high_precision_truth() -> None:
    epsilon = complex_epsilon_cell()
    v_box, h_box, epsilon_out = enclose_complex_normal_flow(epsilon=epsilon)
    real_values = (epsilon.re.lo, epsilon.re.mid, epsilon.re.hi)
    imag_values = (epsilon.im.lo, epsilon.im.mid, epsilon.im.hi)
    samples = [complex(real, imag) for real in real_values for imag in imag_values]
    rng = random.Random(17003)
    samples.extend(
        complex(
            rng.uniform(epsilon.re.lo, epsilon.re.hi),
            rng.uniform(epsilon.im.lo, epsilon.im.hi),
        )
        for _ in range(4)
    )
    with mp.workdps(70):
        for sample in samples:
            epsilon_mp = mp.mpc(sample.real, sample.imag)
            v_true, h_true = _truth(epsilon_mp)
            assert v_box.re.lo <= float(mp.re(v_true)) <= v_box.re.hi
            assert v_box.im.lo <= float(mp.im(v_true)) <= v_box.im.hi
            assert h_box.re.lo <= float(mp.re(h_true)) <= h_box.re.hi
            assert h_box.im.lo <= float(mp.im(h_true)) <= h_box.im.hi
    assert epsilon_out.re.lo <= epsilon.re.lo <= epsilon_out.re.hi
    assert epsilon_out.re.lo <= epsilon.re.hi <= epsilon_out.re.hi
    assert epsilon_out.im.lo <= epsilon.im.lo <= epsilon_out.im.hi
    assert epsilon_out.im.lo <= epsilon.im.hi <= epsilon_out.im.hi


def test_complex_normal_flow_report_keeps_h3_refusals() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-complex-normal-flow-v1"
    assert payload["fixed_time_complex_flow_enclosed"] is True
    assert payload["flow_sup_upper"] > 0
    assert payload["complex_first_hit_holomorphic"] is False
    assert payload["actual_return_ln_membership_proved"] is False
    assert payload["g3_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    honesty = payload["honesty"]
    assert honesty["fixed_time_complex_flow_enclosed"] is True
    assert honesty["complex_first_hit_holomorphic"] is False
    assert (
        honesty["_honesty_reasons"]["complex_first_hit_holomorphic"]["reason"]
        == "unimplemented"
    )
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_complex_normal_flow")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False


def test_complex_normal_flow_rejects_invalid_cells_and_time() -> None:
    with pytest.raises(ValueError, match="radii must be positive"):
        complex_epsilon_cell(real_radius=Fraction(0))
    with pytest.raises(ValueError, match="Re\\(epsilon\\) > 0"):
        complex_epsilon_cell(
            center=Fraction(1, 20_000),
            real_radius=Fraction(1, 10_000),
        )
    with pytest.raises(ValueError, match="final_time must be positive"):
        enclose_complex_normal_flow(final_time=Fraction(0))

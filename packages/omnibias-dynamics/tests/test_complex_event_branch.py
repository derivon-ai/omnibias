# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Complex event-time branch on one normal-form cell."""

from __future__ import annotations

import random

import mpmath as mp
from omnibias.dynamics.complex_event_branch import (
    certify_complex_event_branch,
    event_epsilon_cell,
    report,
)
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


def _event_time(epsilon: mp.mpc) -> mp.mpc:
    ell = mp.mpf(9) / 25
    target = -mp.mpf(629_534) / 10_000_000

    def field(state: tuple[mp.mpc, mp.mpc]) -> tuple[mp.mpc, mp.mpc]:
        v_coord, height = state
        field_f = (
            -ell * epsilon**3
            - 2 * epsilon**2 * v_coord
            - epsilon * v_coord**2
            + epsilon * v_coord**3 / 3
        )
        field_g = -1 + epsilon * (v_coord - 1)
        return field_f + height * field_g, -v_coord * height

    def endpoint(time: mp.mpc) -> tuple[mp.mpc, mp.mpc]:
        state = (-epsilon, 4 * epsilon**3)
        step = time / 512
        for _ in range(512):
            k1 = field(state)
            k2 = field(
                (state[0] + step * k1[0] / 2, state[1] + step * k1[1] / 2)
            )
            k3 = field(
                (state[0] + step * k2[0] / 2, state[1] + step * k2[1] / 2)
            )
            k4 = field((state[0] + step * k3[0], state[1] + step * k3[1]))
            state = (
                state[0] + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
                state[1] + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
            )
        return state

    time = mp.mpc("0.5")
    for _ in range(6):
        state = endpoint(time)
        derivative = field(state)[0]
        time -= (state[0] - target) / derivative
    return time


def test_complex_branch_contains_grid_and_random_high_precision_roots() -> None:
    epsilon = event_epsilon_cell()
    branch = certify_complex_event_branch()
    assert branch.status == "unique_root"
    assert branch.unique_for_every_parameter
    samples = [
        complex(real, imag)
        for real in (epsilon.re.lo, epsilon.re.mid, epsilon.re.hi)
        for imag in (epsilon.im.lo, epsilon.im.mid, epsilon.im.hi)
    ]
    rng = random.Random(18003)
    samples.extend(
        complex(
            rng.uniform(epsilon.re.lo, epsilon.re.hi),
            rng.uniform(epsilon.im.lo, epsilon.im.hi),
        )
        for _ in range(4)
    )
    with mp.workdps(70):
        for sample in samples:
            truth = _event_time(mp.mpc(sample.real, sample.imag))
            assert branch.enclosure.re.contains(float(mp.re(truth)))
            assert branch.enclosure.im.contains(float(mp.im(truth)))


def test_local_complex_branch_replays_real_first_hit_without_earning_g3() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-complex-event-branch-v1"
    assert payload["branch_status"] == "unique_root"
    assert payload["complex_normal_event_branch_certified"] is True
    assert payload["real_first_hit_status"] == "certified"
    assert payload["real_first_hit_replayed"] is True
    assert payload["complex_physical_return_family_certified"] is False
    assert payload["actual_return_ln_membership_proved"] is False
    assert payload["g3_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_complex_normal_event_branch")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False

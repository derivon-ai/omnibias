# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Uniform complex event cover across the cubic model's separation limits."""

from __future__ import annotations

import random

import mpmath as mp
from omnibias.dynamics.complex_separation_cover import (
    certify_complex_separation_branch,
    report,
    separation_cell,
)
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


def _event_time(separation: mp.mpc) -> mp.mpc:
    epsilon = mp.mpf(1) / 16
    ell = 1 - separation**2 / 4
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
        time -= (state[0] - target) / field(state)[0]
    return time


def test_complex_separation_branch_contains_grid_and_random_truth() -> None:
    separation = separation_cell()
    branch = certify_complex_separation_branch()
    assert branch.status == "unique_root"
    assert branch.unique_for_every_parameter
    samples = [
        complex(real, imag)
        for real in (separation.re.lo, separation.re.mid, separation.re.hi)
        for imag in (separation.im.lo, separation.im.mid, separation.im.hi)
    ]
    rng = random.Random(18004)
    samples.extend(
        complex(
            rng.uniform(separation.re.lo, separation.re.hi),
            rng.uniform(separation.im.lo, separation.im.hi),
        )
        for _ in range(4)
    )
    with mp.workdps(70):
        for sample in samples:
            truth = _event_time(mp.mpc(sample.real, sample.imag))
            assert branch.enclosure.re.contains(float(mp.re(truth)))
            assert branch.enclosure.im.contains(float(mp.im(truth)))


def test_separation_cover_spans_both_limits_but_not_physical_matching() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-complex-separation-cover-v1"
    assert payload["branch_status"] == "unique_root"
    assert payload["d_c_endpoint_included"] is True
    assert payload["chart_o_endpoint_included"] is True
    assert payload["derivative_modulus"][0] > 0
    assert payload["complex_separation_event_cover_certified"] is True
    assert payload["real_first_hit_status"] == "certified"
    assert payload["real_first_hit_replayed"] is True
    assert payload["physical_overlap_matching_proved"] is False
    assert payload["actual_return_ln_membership_proved"] is False
    assert payload["g3_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_complex_separation_cover")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    assert not any(derived_parent_flags(ledger).values())

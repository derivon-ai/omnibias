# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Complex physical outgoing-section cover across both separation limits."""

from __future__ import annotations

import random

import mpmath as mp
from omnibias.dynamics.complex_physical_e_out import (
    certify_complex_physical_e_out_cover,
    physical_e_out_cells,
    report,
)
from omnibias.dynamics.hilbert16_ledger import (
    default_h16_ledger,
    derived_parent_flags,
)


def _event_time(separation: mp.mpc) -> mp.mpc:
    epsilon = mp.mpf(1) / 16
    rho = mp.mpf(1) / 4
    coefficient = mp.mpf(2)
    ell = 1 - separation**2 / 4

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

    time = mp.mpc("31")
    for _ in range(6):
        v_coord, height = endpoint(time)
        vdot, hdot = field((v_coord, height))
        event = (
            v_coord
            + rho
            + epsilon * rho * height
            + coefficient * epsilon**2 * rho * height**2
        )
        derivative = (
            vdot
            + epsilon * rho * hdot
            + 2 * coefficient * epsilon**2 * rho * height * hdot
        )
        time -= event / derivative
    return time - 30


def test_physical_e_out_cells_contain_grid_and_random_complex_roots() -> None:
    cells = physical_e_out_cells()
    branches = certify_complex_physical_e_out_cover()
    rng = random.Random(18005)
    with mp.workdps(50):
        for cell, branch in zip(cells, branches, strict=True):
            samples = [
                complex(real, imag)
                for real in (
                    cell.separation.re.lo,
                    cell.separation.re.mid,
                    cell.separation.re.hi,
                )
                for imag in (
                    cell.separation.im.lo,
                    cell.separation.im.mid,
                    cell.separation.im.hi,
                )
            ]
            samples.append(
                complex(
                    rng.uniform(
                        cell.separation.re.lo,
                        cell.separation.re.hi,
                    ),
                    rng.uniform(
                        cell.separation.im.lo,
                        cell.separation.im.hi,
                    ),
                )
            )
            for sample in samples:
                truth = _event_time(mp.mpc(sample.real, sample.imag))
                assert branch.enclosure.re.contains(float(mp.re(truth)))
                assert branch.enclosure.im.contains(float(mp.im(truth)))


def test_physical_e_out_cover_is_local_and_does_not_earn_g3() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-complex-physical-e-out-v1"
    assert payload["d_c_endpoint_included"] is True
    assert payload["chart_o_endpoint_included"] is True
    assert payload["adjacent_branches_matched"] is True
    assert payload["complex_physical_e_out_cover_certified"] is True
    assert payload["real_first_hit_cover_replayed"] is True
    assert payload["incoming_physical_branch_certified"] is False
    assert payload["physical_overlap_matching_proved"] is False
    assert payload["actual_return_ln_membership_proved"] is False
    assert payload["g3_passed"] is False
    assert payload["full_hilbert16_solved"] is False
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_complex_physical_e_out_cover")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    assert not any(derived_parent_flags(ledger).values())

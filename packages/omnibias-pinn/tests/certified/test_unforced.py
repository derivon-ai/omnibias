# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-18 / 07-19: force-free BKM slabs and continuation."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from omnibias.core.proof.catalog import catalog_entry
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_AB_EXTERNAL_PREMISES,
    check_ledger,
    navier_stokes_ab_architecture_ledger,
)
from omnibias.pinn.certified.unforced import (
    BKM_BUDGET,
    LOCKED_HORIZON,
    TG_OMEGA_LINF_FACTOR,
    UNFORCED_CONTINUATION_LEFTOVER,
    Continue,
    Halt,
    UnforcedSlab,
    enclose_bkm_integral,
    force_free_bkm_slab,
    honesty_payload,
    locked_force_free_slab,
    locked_two_slab_continuation,
    try_continue_slab,
)


def test_module_does_not_import_navier_stokes() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "omnibias"
        / "pinn"
        / "certified"
        / "unforced.py"
    ).read_text(encoding="utf-8")
    assert "from omnibias.pinn.certified.navier_stokes" not in src
    assert "import omnibias.pinn.certified.navier_stokes" not in src
    assert "from omnibias.pinn.certified import navier_stokes" not in src


def test_g1_fixture_force_is_zero() -> None:
    report = force_free_bkm_slab()
    assert report["force_zero"] is True
    assert report["plant_unforced"] is True
    assert report["omega_factor_locked"] is True
    assert TG_OMEGA_LINF_FACTOR == 2


def test_g2_weak_form_covers_box() -> None:
    report = force_free_bkm_slab()
    assert report["weak_covers"] is True
    assert report["weak_misses"] == 0


def test_g3_bkm_enclosure_contains_exact_and_samples() -> None:
    report = force_free_bkm_slab()
    assert report["bkm_bounded"] is True
    assert report["bkm_contains_exact"] is True
    assert report["integrand_grid_and_sample"] is True
    slab = report["slab"]
    assert isinstance(slab, UnforcedSlab)
    enc = enclose_bkm_integral(slab)
    assert enc.hi < float(BKM_BUDGET)


def test_g4_honesty_leftovers() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["continuum_navier_stokes_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    assert flags["three_d_claim"] is False
    assert flags["infinite_time_leftover"] is True
    assert flags["all_data_leftover"] is True
    assert flags["bridge_theorem_leftover"] is True
    assert flags["unforced_majorant_leftover"] is True
    report = force_free_bkm_slab()
    assert report["honesty"]["navier_stokes_proof_claim"] is False


def test_g5_ab_ledger_conditional() -> None:
    report = force_free_bkm_slab()
    ledger = navier_stokes_ab_architecture_ledger()
    checked = check_ledger(ledger)
    assert checked.strength == "CONDITIONAL"
    assert checked.holds
    assert NS_AB_EXTERNAL_PREMISES
    assert report["ledger_strength"] == "CONDITIONAL"
    assert report["ab_premises_nonempty"] is True


def test_two_locked_slabs_accept_on_decaying_tg() -> None:
    first = locked_force_free_slab()
    result = try_continue_slab(
        first,
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(result, Continue)
    assert result.next_slab.t0 == Fraction(1, 2)
    assert result.next_slab.horizon == LOCKED_HORIZON
    assert result.next_slab.growing is False
    chain = locked_two_slab_continuation()
    assert chain["accepted"] is True
    assert chain["n_slabs"] == 2
    assert chain["second_t0"] == "1/2"
    assert chain["covers_infinite_time"] is False


def test_growing_vorticity_is_blocked() -> None:
    growing = locked_force_free_slab(growing=True)
    result = try_continue_slab(
        growing,
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(result, Halt)
    assert result.reason == "BLOCKED"
    assert result.detail == "growing_vorticity"
    chain = locked_two_slab_continuation()
    assert chain["growing_reason"] == "BLOCKED"
    assert chain["growing_detail"] == "growing_vorticity"


def test_empty_budget_is_search_incomplete() -> None:
    slab = locked_force_free_slab()
    result = try_continue_slab(
        slab,
        remaining_budget=0,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(result, Halt)
    assert result.reason == "search_incomplete"
    chain = locked_two_slab_continuation()
    assert chain["empty_budget_reason"] == "search_incomplete"
    assert chain["honesty"]["navier_stokes_proof_claim"] is False


def test_leftover_57_is_one_object() -> None:
    leftover = UNFORCED_CONTINUATION_LEFTOVER
    assert leftover["leftover_id"] == 57
    assert leftover["covers_infinite_time"] is False
    assert leftover["all_data"] is False
    assert leftover["three_d"] is False
    assert leftover["bridge_theorem"] is False
    chain = locked_two_slab_continuation()
    assert chain["leftover_id"] == 57


def test_catalog_kinds_are_ab_open() -> None:
    import omnibias.pinn.certified.machine  # noqa: F401

    slab = catalog_entry("unforced_bkm_slab")
    assert slab is not None
    assert slab.parent == "Navier-Stokes unforced regularity (Clay A/B)"
    assert slab.parent_status == "open"
    cont = catalog_entry("unforced_slab_continuation")
    assert cont is not None
    assert cont.parent == "Navier-Stokes unforced regularity (Clay A/B)"
    assert cont.parent_status == "open"
    abc = catalog_entry("unforced_abc_slab")
    assert abc is not None
    assert abc.parent == "Navier-Stokes unforced regularity (Clay A/B)"
    assert abc.parent_status == "open"


def test_forcing_array_is_numpy_zero() -> None:
    from omnibias.pinn.certified.fluid_fixtures import taylor_green_vortex

    sample = taylor_green_vortex(16, viscosity=0.1, density=1.0, time=0.0)
    assert np.all(sample.forcing == 0.0)


def test_abc_g1_fixture_is_3d_force_free() -> None:
    from omnibias.pinn.certified.unforced import force_free_abc_bkm_slab

    report = force_free_abc_bkm_slab()
    assert report["force_zero"] is True
    assert report["plant_unforced"] is True
    assert report["dimension"] == 3
    assert report["grid_omega_inside_bound"] is True


def test_abc_g2_bkm_enclosure() -> None:
    from omnibias.pinn.certified.unforced import (
        BKM_BUDGET,
        enclose_abc_bkm_integral,
        force_free_abc_bkm_slab,
    )

    report = force_free_abc_bkm_slab()
    assert report["bkm_bounded"] is True
    assert report["bkm_contains_exact"] is True
    assert report["integrand_grid_and_sample"] is True
    enc = enclose_abc_bkm_integral(report["slab"])
    assert enc.hi < float(BKM_BUDGET)


def test_abc_g3_two_slabs_and_growing_halt() -> None:
    from omnibias.pinn.certified.unforced import (
        LOCKED_HORIZON,
        AbcSlab,
        Continue,
        Halt,
        locked_force_free_abc_slab,
        locked_two_abc_slab_continuation,
        try_continue_abc_slab,
    )

    first = locked_force_free_abc_slab()
    result = try_continue_abc_slab(
        first,
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(result, Continue)
    assert isinstance(result.next_slab, AbcSlab)
    assert result.next_slab.t0 == Fraction(1, 2)
    chain = locked_two_abc_slab_continuation()
    assert chain["accepted"] is True
    assert chain["n_slabs"] == 2
    assert chain["second_t0"] == "1/2"
    growing = try_continue_abc_slab(
        locked_force_free_abc_slab(growing=True),
        remaining_budget=1,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(growing, Halt)
    assert growing.reason == "BLOCKED"
    assert growing.detail == "growing_vorticity"
    assert chain["growing_reason"] == "BLOCKED"
    empty = try_continue_abc_slab(
        first,
        remaining_budget=0,
        next_horizon=LOCKED_HORIZON,
    )
    assert isinstance(empty, Halt)
    assert empty.reason == "search_incomplete"


def test_abc_g4_honesty_reuses_leftover_57() -> None:
    from omnibias.pinn.certified.unforced import (
        UNFORCED_CONTINUATION_LEFTOVER,
        abc_honesty_payload,
        force_free_abc_bkm_slab,
        locked_two_abc_slab_continuation,
    )

    flags = abc_honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["three_d_claim"] is False
    assert flags["exact_3d_abc_plant"] is True
    assert flags["infinite_time_leftover"] is True
    assert flags["all_data_leftover"] is True
    assert flags["bridge_theorem_leftover"] is True
    assert flags["unforced_majorant_leftover"] is True
    report = force_free_abc_bkm_slab()
    assert report["honesty"]["three_d_claim"] is False
    assert report["honesty"]["exact_3d_abc_plant"] is True
    assert report["leftover_id"] == 57
    chain = locked_two_abc_slab_continuation()
    assert chain["leftover_id"] == 57
    assert UNFORCED_CONTINUATION_LEFTOVER["leftover_id"] == 57
    assert UNFORCED_CONTINUATION_LEFTOVER["three_d"] is False


def test_abc_g5_three_d_premise_stays() -> None:
    from omnibias.pinn.certified.unforced import (
        THREE_D_AB_PREMISE,
        force_free_abc_bkm_slab,
    )

    report = force_free_abc_bkm_slab()
    ledger = navier_stokes_ab_architecture_ledger()
    checked = check_ledger(ledger)
    assert checked.strength == "CONDITIONAL"
    assert THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES
    assert THREE_D_AB_PREMISE in ledger.external_premises
    assert report["three_d_premise_present"] is True
    assert report["ledger_strength"] == "CONDITIONAL"

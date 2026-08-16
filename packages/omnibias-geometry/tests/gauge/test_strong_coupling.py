# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Crude strong-coupling polymer bound: domain, honesty, replay, not continuum."""

from __future__ import annotations

import math

import pytest
from omnibias.core.proof import Conjecture, seal_certificate
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    STRONG_COUPLING_KIND,
    STRONG_COUPLING_SCHEMA_VERSION,
    replay_strong_coupling_gap,
    seal_strong_coupling_certificate,
    strong_coupling_schema_errors,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    BACKTRACK_POLYMER_METHOD,
    BETA_LOCK,
    BETA_LOCK_CRUDE,
    CLUSTER_POLYMER_METHOD,
    CRUDE_POLYMER_METHOD,
    POLYMER_METHOD,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_gap,
    polymer_coordination,
    polymer_coordination_backtrack,
    polymer_first_step,
    su2_wilson_activity,
)


def _bessel_i(order: int, z: float, terms: int = 40) -> float:
    """Independent ascending series for ``I_n(z)``, used as a containment sample."""
    acc = 0.0
    for k in range(terms):
        acc += (0.5 * z) ** (2 * k + order) / (
            math.factorial(k) * math.factorial(k + order)
        )
    return acc


def test_coordination_is_locked_and_documented() -> None:
    assert polymer_coordination(4) == 24
    assert polymer_coordination(3) == 16
    assert polymer_coordination_backtrack(4) == 15
    assert polymer_coordination_backtrack(3) == 9
    assert polymer_first_step(4) == 20
    assert polymer_first_step(3) == 12
    assert polymer_coordination_backtrack(4) < polymer_first_step(4)
    with pytest.raises(ValueError, match="spacetime_dim"):
        polymer_coordination(1)
    with pytest.raises(ValueError, match="spacetime_dim"):
        polymer_coordination_backtrack(1)
    with pytest.raises(ValueError, match="spacetime_dim"):
        polymer_first_step(1)


def test_locked_beta_is_in_domain_and_positive() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK)
    assert result.in_convergence_domain is True
    assert result.certified is True
    assert result.method == POLYMER_METHOD
    assert result.counting == "two_scale"
    assert result.coordination == 15
    assert result.first_step == 20
    assert result.spacetime_dim == 4
    assert result.spectral_gap_lower > 0.0
    assert 0.0 < result.subdominant_ratio_upper < 1.0
    assert result.tail_bound is not None


def test_single_scale_backtrack_is_not_sold_as_n2() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK, counting="backtrack")
    assert result.method == BACKTRACK_POLYMER_METHOD
    assert result.coordination == 15
    assert result.first_step is None
    assert result.certified is True


def test_cluster_counting_certifies_at_the_lock() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK, counting="cluster")
    assert result.certified is True
    assert result.method == CLUSTER_POLYMER_METHOD
    assert result.counting == "cluster"
    assert result.n_keep == 3
    assert result.first_step == 20
    assert result.tail_bound is not None
    activity = su2_wilson_activity(BETA_LOCK)
    mid = 0.5 * (activity.lo + activity.hi)
    scale_a, scale_b = 20.0, 15.0
    terms = [mid, scale_a * mid**2, scale_a * scale_b * mid**3]
    last = terms[-1]
    ratio = scale_b * mid
    sample = sum(terms) + last * ratio / (1.0 - ratio)
    assert result.activity_times_c.contains(sample)
    too_large = certified_strong_coupling_glueball_bound(0.25, counting="cluster")
    assert too_large.certified is False


def test_quarter_beta_is_too_large_for_two_scale() -> None:
    result = certified_strong_coupling_glueball_bound(0.25)
    assert result.certified is False
    assert result.subdominant_ratio_upper >= 1.0


def test_crude_lock_still_certifies_only_the_overcount() -> None:
    crude = certified_strong_coupling_glueball_bound(
        BETA_LOCK_CRUDE, counting="crude"
    )
    assert crude.certified is True
    assert crude.method == CRUDE_POLYMER_METHOD
    assert crude.coordination == 24
    at_quarter = certified_strong_coupling_glueball_bound(
        BETA_LOCK, counting="crude"
    )
    assert at_quarter.certified is False


def test_out_of_domain_is_not_certified_and_must_not_seal() -> None:
    result = certified_strong_coupling_glueball_bound(1)
    assert result.in_convergence_domain is False
    assert result.certified is False
    assert result.spectral_gap_lower == 0.0
    assert result.subdominant_ratio_upper >= 1.0
    with pytest.raises(ValueError, match="uncertified"):
        seal_strong_coupling_certificate(result)


def test_activity_interval_contains_independent_sample() -> None:
    activity = su2_wilson_activity(BETA_LOCK)
    sample = _bessel_i(2, float(BETA_LOCK)) / _bessel_i(1, float(BETA_LOCK))
    assert activity.lo <= sample <= activity.hi


def test_wilson_character_gap_is_positive_and_tighter_than_polymer() -> None:
    wilson = certified_wilson_character_gap(BETA_LOCK)
    polymer = certified_strong_coupling_glueball_bound(BETA_LOCK)
    assert wilson.certified is True
    assert wilson.spectral_gap_lower > polymer.spectral_gap_lower
    assert wilson.subdominant_ratio_upper < polymer.subdominant_ratio_upper


def test_seal_replay_and_honesty_flags() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK)
    cert = seal_strong_coupling_certificate(result)
    assert cert["schema_version"] == STRONG_COUPLING_SCHEMA_VERSION
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["honesty"]["continuum_claim"] is False
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["honesty"]["fixed_spacing"] is True
    assert cert["method"] == POLYMER_METHOD
    assert cert["counting"] == "two_scale"
    assert cert["first_step"] == 20
    assert "two_scale_polymer_count" in cert["honesty"]["note"]
    assert "NOT" in cert["honesty"]["note"]
    assert strong_coupling_schema_errors(cert) == []
    assert verify_certificate_digest(cert)
    assert replay_strong_coupling_gap(cert) is True
    source = generate_obligation(cert)
    assert source is not None
    assert "spectral_gap_pos" in source


def test_schema_rejects_continuum_and_yang_mills_flags() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK)
    cert = dict(seal_strong_coupling_certificate(result))
    cert["continuum_claim"] = True
    errors = strong_coupling_schema_errors(cert)
    assert any("continuum_claim must be False" in e for e in errors)
    cert = dict(seal_strong_coupling_certificate(result))
    honesty = dict(cert["honesty"])
    honesty["yang_mills_claim"] = True
    cert["honesty"] = honesty
    errors = strong_coupling_schema_errors(cert)
    assert any("yang_mills_claim must be False" in e for e in errors)


def test_forged_tighter_ratio_fails_replay() -> None:
    result = certified_strong_coupling_glueball_bound(BETA_LOCK)
    forged = dict(seal_strong_coupling_certificate(result))
    forged.pop("digest")
    forged["subdominant_ratio_upper"] = 1e-12
    resealed = dict(seal_certificate(forged))
    assert verify_certificate_digest(resealed)
    assert replay_strong_coupling_gap(resealed) is False


def test_proofmachine_proves_locked_beta_and_blocks_out_of_domain() -> None:
    machine = build_gauge_machine()
    proved = machine.evaluate(
        Conjecture("polymer", STRONG_COUPLING_KIND, {"beta": BETA_LOCK})
    )
    assert proved.status == "PROVED"
    assert proved.schema_ok is True
    assert proved.replay_ok is True
    assert proved.honesty_ok is True
    blocked = machine.evaluate(
        Conjecture("too weak", STRONG_COUPLING_KIND, {"beta": 1.0})
    )
    assert blocked.status == "BLOCKED"


def test_asserting_yang_mills_is_blocked() -> None:
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "overclaim",
            STRONG_COUPLING_KIND,
            {"beta": BETA_LOCK},
            claims={"yang_mills_claim": True},
        )
    )
    assert verdict.status == "BLOCKED"
    assert verdict.honesty_ok is False


def test_tiny_lattice_plaquette_is_consistent_with_a_positive_gap() -> None:
    """Sanity, not a proof: a few heat-bath sweeps at the locked β stay in (0, 1)."""
    torch = pytest.importorskip("torch")
    from omnibias.geometry.gauge.lattice.observables import average_plaquette
    from omnibias.geometry.gauge.lattice.su2 import random_links, sweep

    result = certified_strong_coupling_glueball_bound(BETA_LOCK)
    assert result.certified is True
    links = random_links((4, 4, 4, 4), device="cpu", dtype=torch.float64)
    gen = torch.Generator().manual_seed(7)
    sweep(links, float(BETA_LOCK), generator=gen)
    sweep(links, float(BETA_LOCK), generator=gen)
    plaquette = float(average_plaquette(links))
    assert 0.0 < plaquette < 1.0

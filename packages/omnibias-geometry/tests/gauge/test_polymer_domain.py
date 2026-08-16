# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Polymer majorant domain on a locked dyadic beta grid. Not continuum."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.proof import Conjecture, seal_certificate
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    POLYMER_DOMAIN_KIND,
    POLYMER_DOMAIN_SCHEMA_VERSION,
    polymer_domain_schema_errors,
    replay_polymer_domain,
    seal_polymer_domain_certificate,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    POLYMER_BETA_DOMAIN_METHOD,
    POLYMER_BETA_GRID,
    certified_polymer_beta_domain,
    certified_strong_coupling_glueball_bound,
)


def test_two_scale_domain_has_a_certifying_point_and_a_failure() -> None:
    result = certified_polymer_beta_domain()
    assert result.certified is True
    assert result.method == POLYMER_BETA_DOMAIN_METHOD
    assert result.counting == "two_scale"
    assert result.grid == POLYMER_BETA_GRID
    assert result.beta_certified < result.beta_outside
    assert result.continuum_claim is False
    assert result.yang_mills_claim is False
    assert certified_strong_coupling_glueball_bound(result.beta_certified).certified
    assert not certified_strong_coupling_glueball_bound(result.beta_outside).certified
    assert result.beta_outside <= Fraction(1, 4) or result.beta_certified < Fraction(1, 4)


def test_cluster_domain_uses_the_same_grid() -> None:
    result = certified_polymer_beta_domain(counting="cluster")
    assert result.certified is True
    assert result.counting == "cluster"
    assert result.n_keep == 3
    assert result.grid == POLYMER_BETA_GRID
    assert result.beta_certified < result.beta_outside
    assert certified_strong_coupling_glueball_bound(
        result.beta_certified, counting="cluster"
    ).certified
    assert not certified_strong_coupling_glueball_bound(
        result.beta_outside, counting="cluster"
    ).certified


def test_grid_must_contain_a_failure() -> None:
    with pytest.raises(ValueError, match="first failure"):
        certified_polymer_beta_domain(grid=(Fraction(1, 32), Fraction(1, 16)))


def test_seal_replay_and_honesty_flags() -> None:
    result = certified_polymer_beta_domain()
    cert = seal_polymer_domain_certificate(result)
    assert cert["schema_version"] == POLYMER_DOMAIN_SCHEMA_VERSION
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["honesty"]["continuum_claim"] is False
    assert cert["method"] == POLYMER_BETA_DOMAIN_METHOD
    assert polymer_domain_schema_errors(cert) == []
    assert verify_certificate_digest(cert)
    assert replay_polymer_domain(cert) is True


def test_schema_rejects_continuum_and_yang_mills_flags() -> None:
    result = certified_polymer_beta_domain()
    cert = dict(seal_polymer_domain_certificate(result))
    cert["continuum_claim"] = True
    errors = polymer_domain_schema_errors(cert)
    assert any("continuum_claim must be False" in item for item in errors)
    cert = dict(seal_polymer_domain_certificate(result))
    honesty = dict(cert["honesty"])
    honesty["yang_mills_claim"] = True
    cert["honesty"] = honesty
    errors = polymer_domain_schema_errors(cert)
    assert any("yang_mills_claim must be False" in item for item in errors)


def test_forged_larger_certified_beta_fails_replay() -> None:
    result = certified_polymer_beta_domain()
    forged = dict(seal_polymer_domain_certificate(result))
    forged.pop("digest")
    forged["beta_certified"] = [
        int(result.beta_outside.numerator),
        int(result.beta_outside.denominator),
    ]
    resealed = dict(seal_certificate(forged))
    assert verify_certificate_digest(resealed)
    assert replay_polymer_domain(resealed) is False


def test_proofmachine_proves_the_domain_and_blocks_yang_mills() -> None:
    machine = build_gauge_machine()
    proved = machine.evaluate(Conjecture("domain", POLYMER_DOMAIN_KIND, {}))
    assert proved.status == "PROVED"
    assert proved.schema_ok is True
    assert proved.replay_ok is True
    assert proved.honesty_ok is True
    blocked = machine.evaluate(
        Conjecture(
            "overclaim",
            POLYMER_DOMAIN_KIND,
            {},
            claims={"yang_mills_claim": True},
        )
    )
    assert blocked.status == "BLOCKED"
    assert blocked.honesty_ok is False

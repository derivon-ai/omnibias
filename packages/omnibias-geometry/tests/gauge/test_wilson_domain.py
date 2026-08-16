# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wilson-character beta-domain past the polymer cutoff. Not 4-D YM."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.proof import Conjecture, seal_certificate
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    WILSON_CHARACTER_DOMAIN_KIND,
    WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION,
    replay_wilson_character_domain,
    seal_wilson_character_domain_certificate,
    wilson_character_domain_schema_errors,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    WILSON_CHARACTER_BETA_DOMAIN_METHOD,
    WILSON_CHARACTER_BETA_GRID,
    WILSON_CHARACTER_CONTRAST_BETA,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_beta_domain,
    certified_wilson_character_gap,
)


def test_wilson_certifies_at_quarter_and_past_the_polymer_cutoff() -> None:
    result = certified_wilson_character_beta_domain()
    assert result.certified is True
    assert result.method == WILSON_CHARACTER_BETA_DOMAIN_METHOD
    assert result.grid == WILSON_CHARACTER_BETA_GRID
    assert result.quarter_certified is True
    assert result.grid_exhausted is True
    assert result.beta_outside is None
    assert result.beta_certified > WILSON_CHARACTER_CONTRAST_BETA
    assert result.continuum_claim is False
    assert result.yang_mills_claim is False
    assert certified_wilson_character_gap(Fraction(1, 4)).certified
    assert certified_wilson_character_gap(Fraction(2)).certified
    assert not certified_strong_coupling_glueball_bound(
        Fraction(1, 4), counting="two_scale"
    ).certified


def test_grid_must_include_the_polymer_contrast() -> None:
    with pytest.raises(ValueError, match="1/4"):
        certified_wilson_character_beta_domain(grid=(Fraction(1, 2), Fraction(1)))


def test_seal_replay_and_honesty_flags() -> None:
    result = certified_wilson_character_beta_domain()
    cert = seal_wilson_character_domain_certificate(result)
    assert cert["schema_version"] == WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION
    assert cert["continuum_claim"] is False
    assert cert["honesty"]["yang_mills_claim"] is False
    assert cert["honesty"]["continuum_claim"] is False
    assert cert["grid_exhausted"] is True
    assert cert["beta_outside"] is None
    assert cert["method"] == WILSON_CHARACTER_BETA_DOMAIN_METHOD
    assert wilson_character_domain_schema_errors(cert) == []
    assert verify_certificate_digest(cert)
    assert replay_wilson_character_domain(cert) is True


def test_schema_rejects_continuum_and_yang_mills_flags() -> None:
    result = certified_wilson_character_beta_domain()
    cert = dict(seal_wilson_character_domain_certificate(result))
    cert["continuum_claim"] = True
    errors = wilson_character_domain_schema_errors(cert)
    assert any("continuum_claim must be False" in item for item in errors)
    cert = dict(seal_wilson_character_domain_certificate(result))
    honesty = dict(cert["honesty"])
    honesty["yang_mills_claim"] = True
    cert["honesty"] = honesty
    errors = wilson_character_domain_schema_errors(cert)
    assert any("yang_mills_claim must be False" in item for item in errors)


def test_forged_larger_certified_beta_fails_replay() -> None:
    result = certified_wilson_character_beta_domain()
    forged = dict(seal_wilson_character_domain_certificate(result))
    forged.pop("digest")
    forged["beta_certified"] = [5, 1]
    resealed = dict(seal_certificate(forged))
    assert verify_certificate_digest(resealed)
    assert replay_wilson_character_domain(resealed) is False


def test_proofmachine_proves_the_domain_and_blocks_yang_mills() -> None:
    machine = build_gauge_machine()
    proved = machine.evaluate(Conjecture("wilson-domain", WILSON_CHARACTER_DOMAIN_KIND, {}))
    assert proved.status == "PROVED"
    assert proved.schema_ok is True
    assert proved.replay_ok is True
    assert proved.honesty_ok is True
    blocked = machine.evaluate(
        Conjecture(
            "overclaim",
            WILSON_CHARACTER_DOMAIN_KIND,
            {},
            claims={"yang_mills_claim": True},
        )
    )
    assert blocked.status == "BLOCKED"
    assert blocked.honesty_ok is False

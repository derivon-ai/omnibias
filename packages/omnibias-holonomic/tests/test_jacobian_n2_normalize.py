# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""O1 normalization and Case C/D leftover replay. Proof-claim stays False."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnibias.holonomic.jacobian_n2 import JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED
from omnibias.holonomic.jacobian_n2_normalize import (
    classify_leftover_chart,
    named_case_c_shear,
    named_case_d_content,
    o1_normalize,
    replay_leftover_certificate,
    seal_leftover_certificate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_o1_normalizes_scaled_shear_to_the_same_representative() -> None:
    monic = named_case_c_shear()
    scaled = named_case_d_content()
    norm = o1_normalize(scaled)
    assert o1_normalize(monic) == monic
    assert norm[0] == monic[0]
    assert norm[1].homogeneous_part(2) == monic[1].homogeneous_part(2)


def test_case_c_leftover_is_zero() -> None:
    chart = classify_leftover_chart(named_case_c_shear())
    assert chart.case == "C"
    assert chart.leftover == 0
    assert chart.replay_ok is True
    assert chart.honesty()["jacobian_conjecture_proof_claim"] is False


def test_case_d_leftover_is_const_nonzero() -> None:
    chart = classify_leftover_chart(named_case_d_content())
    assert chart.case == "D"
    assert chart.leftover != 0
    assert chart.leftover == 2
    assert chart.replay_ok is True
    assert chart.honesty()["jacobian_conjecture_proof_claim"] is False


def test_ci_replays_sealed_case_c_and_d_fixtures() -> None:
    constructors = {
        "named_case_c_shear": named_case_c_shear,
        "named_case_d_content": named_case_d_content,
    }
    for name in ("jacobian_n2_case_c.json", "jacobian_n2_case_d.json"):
        spec = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        chart = classify_leftover_chart(constructors[spec["constructor"]]())
        assert chart.case == spec["case"]
        assert str(chart.leftover) == spec["leftover"]
        sealed = seal_leftover_certificate(chart, name=spec["name"])
        assert replay_leftover_certificate(sealed) is True
        assert sealed["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_proof_claim_flag_stays_false() -> None:
    assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False
    sealed = seal_leftover_certificate(
        classify_leftover_chart(named_case_c_shear()), name="case_c_shear"
    )
    with pytest.raises(ValueError):
        replay_leftover_certificate(
            {**sealed, "honesty": {**sealed["honesty"], "jacobian_conjecture_proof_claim": True}}
        )

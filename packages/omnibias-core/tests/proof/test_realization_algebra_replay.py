# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.proof.realization_algebra_replay import (
    algebraic_realization_replay_certificate,
    laurent_closure_replay_certificate,
    moment_identity_replay_certificate,
    polynomial_box_replay_certificate,
    sturm_isolation_replay_certificate,
)
from omnibias.core.proof.realization_replay import (
    generate_replay_obligation,
    source_digest,
    verify_replay_certificate,
)
from omnibias.core.realization.algebraic import RealAlgebraicField
from omnibias.core.realization.membership import classify_binary_quadratic
from omnibias.core.realization.polynomial import SparsePolynomial


def _examples():
    real = classify_binary_quadratic([[1, 0, 2], [0, 1, 0]])
    boundary = classify_binary_quadratic([[1, 0, 0], [0, 1, 0]])
    x = SparsePolynomial.variable(1, 0)
    tree = {
        "axis": 0,
        "midpoint": [1, 2],
        "left": {"equation": 0, "enclosure": [[1, 2], [5, 4]]},
        "right": {"equation": 0, "enclosure": [[1, 4], [3, 2]]},
    }
    return [
        sturm_isolation_replay_certificate(RealAlgebraicField.sqrt(2)),
        sturm_isolation_replay_certificate(RealAlgebraicField((-2, -1, 0, 1), 1, 2)),
        algebraic_realization_replay_certificate(real.source, real.witness),
        laurent_closure_replay_certificate(boundary.source, boundary.closure_path, radius=Q(1, 10)),
        polynomial_box_replay_certificate((x * x - x + 1,), ((0, 1),), tree),
    ]


@pytest.mark.parametrize("certificate", _examples())
def test_generic_algebra_replay_is_exact_and_emits_both_registers(certificate):
    assert verify_replay_certificate(certificate)
    assert "decide +kernel" in generate_replay_obligation(certificate)
    assert "OmnibiasAnalytic.RealizationAlgebra" in generate_replay_obligation(
        certificate, mathlib=True
    )


@pytest.mark.parametrize("index", range(5))
def test_resealed_arithmetic_tampering_is_not_hidden_by_source_seal(index):
    false = deepcopy(_examples()[index])
    p, s = false["payload"], false["payload"]["source"]
    if index <= 1:
        s["chain"][1][0] = [999, 1]
    elif index == 2:
        p["quotients"][0] = [[999, 1]]
    elif index == 3:
        s["parameters"][0] = [[-1, [1, 1]]]
    else:
        p["tree"]["midpoint"] = [2, 1]
    p["source_digest"] = source_digest(s)
    false = seal_certificate(false)
    assert not verify_replay_certificate(false)
    assert generate_replay_obligation(false) is not None


@pytest.mark.parametrize("certificate", _examples())
def test_actual_kernel_algebra_build_when_toolchain_available(certificate):
    from omnibias.core.proof.lean_check import check_certificate, lean_check_available

    if not lean_check_available():
        pytest.skip("Lean toolchain unavailable")
    assert check_certificate(certificate).verified


def test_moment_identity_embeds_original_offsets_weights_and_center():
    certificate = moment_identity_replay_certificate(
        (1, -1), (2, 3), center=0, expected_moments=(5, -1, Q(5, 2), -Q(1, 6), Q(5, 24))
    )
    assert verify_replay_certificate(certificate)
    assert len(certificate["payload"]["point"]) == 5
    false = deepcopy(certificate)
    false["payload"]["point"][0] = [3, 1]
    assert not verify_replay_certificate(seal_certificate(false))
    with pytest.raises(ValueError):
        moment_identity_replay_certificate((1, -1), (2, 3), center=0, expected_moments=(5, 0))

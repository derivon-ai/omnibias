# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from copy import deepcopy

import pytest
import torch
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.proof.lean_check import lean_check_available
from omnibias.core.proof.realization_replay import verify_replay_certificate
from omnibias.core.refine import RefinedPack
from omnibias.core.verified.interval import Interval
from omnibias.torch.confluent_bank import ConfluentPackBank
from omnibias.verify.neuromanifold import certify_transition
from omnibias.verify.neuromanifold.formal import (
    _transition_coordinate_replays,
    formalize_confluence,
)


def _result(kind):
    bank = ConfluentPackBank(
        [
            RefinedPack(center=-0.001, weight=0.7, scale=1.25, order=0),
            RefinedPack(center=0.002, weight=-0.2, scale=1.25, order=0),
            RefinedPack(center=0.003, weight=0.1, scale=1.25, order=0),
        ],
        max_packs=4,
        max_order=4,
        dtype=torch.float64,
    )
    proposal = (
        bank.propose_pair(0, 1, error_budget=1e-5)
        if kind == "pair"
        else bank.propose_cluster((0, 1, 2), order=2, error_budget=1e-5)
    )
    return certify_transition(bank.portable_snapshot(), proposal, domain=Interval(-1, 1))


@pytest.mark.parametrize("kind", ["pair", "cluster"])
def test_coordinate_replay_binds_affine_biases_moments_rho_and_rounded_target(kind):
    result = _result(kind)
    assert result.accepted
    certificates = _transition_coordinate_replays(result.certificate["meta"])
    assert len(certificates) == 3 and all(verify_replay_certificate(c) for c in certificates)
    # Changing an original stored coefficient or target center cannot borrow
    # the old coordinate/rounding proof, even after re-sealing.
    for certificate_index, point_index in ((0, 0), (2, 6 if kind == "pair" else 9)):
        false = deepcopy(certificates[certificate_index])
        false["payload"]["point"][point_index] = [100, 1]
        assert not verify_replay_certificate(seal_certificate(false))


@pytest.mark.parametrize("kind", ["pair", "cluster"])
def test_actual_pair_and_cluster_formalization(kind):
    if not lean_check_available():
        pytest.skip("Lean toolchain unavailable")
    from omnibias.formal.mathlib_check import mathlib_check_available

    result = formalize_confluence(_result(kind), mathlib=mathlib_check_available())
    assert result.theorem_prover_verified
    if mathlib_check_available():
        assert result.mathlib_verified
    assert len(result.certificates) == 4

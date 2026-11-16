# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
from __future__ import annotations

import math

import pytest

from omnibias.core.ladder import (
    Normalization,
    hermite_function,
    oscillator_hamiltonian_apply,
)
from omnibias.core.local_jet import (
    LocalJetForbidden,
    mlp_param_count,
    reject_local_jet_flood,
)


def test_oscillator_ground_state_retains_legacy_ladder_surface() -> None:
    x = 0.3
    psi0 = hermite_function(0, x, normalization=Normalization.OSCILLATOR)

    assert psi0 == pytest.approx(math.exp(-0.5 * x * x))
    assert oscillator_hamiltonian_apply(
        0, x, normalization=Normalization.OSCILLATOR
    ) == pytest.approx(0.5 * psi0)


def test_local_jet_parameter_count_and_flood_refusal_are_public() -> None:
    assert mlp_param_count([(4, 4), (1, None)]) == 9

    with pytest.raises(LocalJetForbidden, match="n_directions=4 >= n_params=4"):
        reject_local_jet_flood(4, 4, allow_full=False)

    reject_local_jet_flood(4, 4, allow_full=True)

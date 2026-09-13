# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable realization descriptions and explicit physical-parameter providers."""

from dataclasses import replace

import pytest
from omnibias.core.parameter_jets import ParameterJetSpec, mixed_jet
from omnibias.core.realization import (
    ObservationSpec,
    ParameterBlock,
    ParameterLayout,
    RealizationSpec,
)


def test_layout_is_stable_and_versioned() -> None:
    spec = RealizationSpec.dense((2, 3, 1))
    assert spec.layout.size == 13
    assert spec.layout.block("W1").shape == (1, 3)
    assert spec.layout.block_slice("b1") == slice(12, 13)
    assert spec.fingerprint() == RealizationSpec.dense([2, 3, 1]).fingerprint()
    assert spec.layout.fingerprint() != replace(spec.layout, version=2).fingerprint()
    with pytest.raises(ValueError, match="contiguous"):
        ParameterLayout((ParameterBlock("x", (2,), 1),))
    with pytest.raises(ValueError, match="unique"):
        ParameterLayout((ParameterBlock("x", (2,), 0), ParameterBlock("x", (1,), 2)))


def test_observation_scope_changes_identity_without_claiming_determination() -> None:
    sample = ObservationSpec(points=((0.0,),), label="one sample")
    assert sample.scope == "finite_observations"
    original = RealizationSpec.dense((1, 1))
    assert replace(original, observations=(sample,)).fingerprint() != original.fingerprint()
    with pytest.raises(ValueError, match="finite"):
        ObservationSpec(points=((float("inf"),),))


def test_supplied_provider_is_used_and_no_field_is_silently_ignored() -> None:
    class Provider:
        def mixed_parameter_jet(self, coords: tuple[float, float], parameters: float, *, spec: ParameterJetSpec) -> float:
            return 7.0 * parameters + coords[0] + spec.param_order

    assert mixed_jet(Provider(), (2.0, 3.0), 5.0) == 38.0
    with pytest.raises(TypeError, match="mixed_parameter_jet"):
        mixed_jet(lambda x, p: x[0] * p, (2.0, 3.0), 5.0)
    with pytest.raises(NotImplementedError, match="autodiff"):
        mixed_jet(None, (1.0, 1.0), 1.0, spec=ParameterJetSpec(method="autodiff"))

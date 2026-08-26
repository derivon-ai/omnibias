# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named-collapse schema: founding three stay distinct; rebrands fail."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    FOUNDING_COLLAPSES,
    FOUNDING_NAMES,
    CollapseRegistry,
    CollapseSpec,
    are_distinct,
    default_honesty,
    get_collapse,
    list_collapses,
    list_rejected_collapses,
    register_collapse,
    reject_collapse,
    require_sound_enclosure,
    reset_collapse_registry,
)
from omnibias.core.verified.interval import Interval


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_founding_three_are_seeded_and_named() -> None:
    names = {spec.name for spec in list_collapses()}
    assert FOUNDING_NAMES <= names
    assert {spec.name for spec in FOUNDING_COLLAPSES} == FOUNDING_NAMES
    for spec in FOUNDING_COLLAPSES:
        assert spec.founding is True
        assert get_collapse(spec.name).founding is True


def test_founding_three_are_pairwise_distinct() -> None:
    specs = list(FOUNDING_COLLAPSES)
    for i, left in enumerate(specs):
        for right in specs[i + 1 :]:
            report = are_distinct(left, right)
            assert report.distinct is True, (left.name, right.name, report.reasons)
            assert "parameter" in report.reasons or "surviving_object" in report.reasons


def test_bias_and_enclosure_share_limit_zero_but_are_not_rebrands() -> None:
    bias = get_collapse("bias")
    enclosure = get_collapse("enclosure")
    assert bias.limit == enclosure.limit == "0"
    assert are_distinct(bias, enclosure).distinct is True


def test_rebrand_of_enclosure_is_rejected() -> None:
    clone = CollapseSpec(
        name="width_squeeze",
        parameter="width",
        limit="0",
        surviving_object="point_plus_proof",
        failure="Inconclusive",
        home="omnibias.core.collapse.schema",
        register="verified",
    )
    with pytest.raises(ValueError, match="rebrand of 'enclosure'"):
        register_collapse(clone)


def test_distinct_new_spec_registers_and_can_be_rejected() -> None:
    spec = CollapseSpec(
        name="probe",
        parameter="probe_parameter",
        limit="0",
        surviving_object="probe_object",
        failure="probe_failure",
        home="omnibias.core.collapse.schema",
        register="verified",
    )
    assert register_collapse(spec) == spec
    assert get_collapse("probe") == spec
    record = reject_collapse("probe", "placeholder; not yet implemented")
    assert record.reason.startswith("placeholder")
    assert "probe" not in {item.name for item in list_collapses()}
    assert list_rejected_collapses()[0].spec.name == "probe"
    with pytest.raises(ValueError, match="was rejected"):
        register_collapse(spec)


def test_cannot_register_or_reject_a_founding_sense() -> None:
    with pytest.raises(ValueError, match="founding sense"):
        register_collapse(
            CollapseSpec(
                name="bias",
                parameter="other",
                limit="0",
                surviving_object="other",
                failure="x",
                home="omnibias.core.collapse.schema",
                register="differentiable",
            )
        )
    with pytest.raises(ValueError, match="cannot reject founding"):
        reject_collapse("bias", "no")
    with pytest.raises(ValueError, match="reserved for"):
        CollapseSpec(
            name="invented",
            parameter="x",
            limit="0",
            surviving_object="y",
            failure="z",
            home="omnibias.core.collapse.schema",
            register="verified",
            founding=True,
        )


def test_isolated_registry_does_not_seed_when_asked() -> None:
    empty = CollapseRegistry(seed_founding=False)
    assert empty.list_active() == ()
    empty.register(
        CollapseSpec(
            name="local",
            parameter="p",
            limit="0",
            surviving_object="s",
            failure="f",
            home="omnibias.core.collapse.schema",
            register="verified",
        )
    )
    assert [spec.name for spec in empty.list_active()] == ["local"]
    assert "local" not in {spec.name for spec in list_collapses()}


def test_require_sound_enclosure_refuses_a_float() -> None:
    with pytest.raises(TypeError, match="float residual is not a certificate"):
        require_sound_enclosure(0.0)
    boxed = require_sound_enclosure(Interval.point(0.0))
    assert boxed.contains_zero()


def test_honesty_never_forges_kernel_flags() -> None:
    for name in ("bias", "temperature", "enclosure", "probe"):
        payload = default_honesty(spec_name=name)
        assert payload["float_residual_is_proof"] is False
        assert payload["theorem_prover_verified"] is False
        assert payload["mathlib_verified"] is False
        assert payload["continuum_parent_inferred"] is False
    assert default_honesty(spec_name="bias")["founding_bias_collapse"] is True
    assert default_honesty(spec_name="temperature")["temperature_collapse"] is True
    assert default_honesty(spec_name="enclosure")["enclosure_collapse"] is True


def test_empty_fields_and_reject_reason_fail() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        CollapseSpec(
            name="  ",
            parameter="p",
            limit="0",
            surviving_object="s",
            failure="f",
            home="omnibias.core.collapse.schema",
            register="verified",
        )
    with pytest.raises(ValueError, match="reject reason"):
        reject_collapse("enclosure", "   ")
    with pytest.raises(KeyError, match="unknown collapse"):
        get_collapse("missing")

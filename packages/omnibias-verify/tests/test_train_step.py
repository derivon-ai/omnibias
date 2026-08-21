# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified I/O step (theory 08-09)."""

from __future__ import annotations

import random

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    CertifiedStepConfig,
    CertifiedStepForbidden,
    CertifiedStepResult,
    Network,
    ReLULayer,
    affine_layer,
    apply_flat_theta,
    certified_accept,
    honesty_payload,
    select_certified_theta,
)


def _affine(w: float) -> Network:
    return Network([affine_layer([[w]], [0.0])])


def _cfg(p_max: float = 2.0) -> CertifiedStepConfig:
    return CertifiedStepConfig(
        property="lipschitz",
        p_max=p_max,
        cell=[Interval(0.0, 1.0)],
    )


def test_section5_accept_and_reject() -> None:
    net = _affine(1.5)
    acc = certified_accept(net, (1.8, 0.0), config=_cfg())
    rej = certified_accept(net, (2.5, 0.0), config=_cfg())
    assert acc.accepted and acc.reason == "ok"
    assert acc.bound_hi is not None and acc.bound_hi <= 2.0
    assert not rej.accepted and rej.reason == "violates"
    assert rej.bound_hi is not None and rej.bound_hi > 2.0
    kept = select_certified_theta((1.5, 0.0), (2.5, 0.0), rej)
    assert kept == (1.5, 0.0)


def test_g1_accepted_steps_stay_in_cap() -> None:
    net = _affine(1.5)
    cfg = _cfg(2.0)
    accepted = 0
    rejected = 0
    for w in [0.5, 1.0, 1.5, 1.8, 1.99, 2.0, 2.01, 2.5, 3.0]:
        result = certified_accept(net, (w, 0.0), config=cfg)
        true_lip = abs(w)
        if result.accepted:
            accepted += 1
            assert result.bound_hi is not None
            assert result.bound_hi <= cfg.p_max
            assert true_lip <= result.bound_hi + 1e-12
        else:
            rejected += 1
            assert result.reason == "violates"
            assert true_lip > cfg.p_max - 1e-12
    assert accepted >= 1
    assert rejected >= 1


def test_g1_soundness_grid_and_random() -> None:
    w = 1.8
    result = certified_accept(_affine(0.0), (w, 0.0), config=_cfg())
    assert result.accepted
    assert result.bound_hi is not None
    xs = [i / 20.0 for i in range(21)]
    for x in xs:
        assert abs(w * x) <= result.bound_hi + 1e-12
    rng = random.Random(0)
    for _ in range(50):
        x = rng.uniform(0.0, 1.0)
        assert abs(w * x) <= result.bound_hi + 1e-12


def test_g2_vacuous_empty_cell() -> None:
    result = certified_accept(
        _affine(1.0),
        config=CertifiedStepConfig(p_max=2.0, cell=None),
    )
    assert result.accepted is False
    assert result.reason == "vacuous"
    assert result.bound_hi is None


def test_g2_vacuous_exploding_relu() -> None:
    net = Network(
        [
            affine_layer([[1.0e300]], [0.0]),
            ReLULayer(),
            affine_layer([[1.0e300]], [0.0]),
        ]
    )
    result = certified_accept(
        net,
        config=CertifiedStepConfig(
            property="lipschitz",
            p_max=2.0,
            cell=[Interval(-1.0e300, 1.0e300)],
        ),
    )
    assert result.accepted is False
    assert result.reason == "vacuous"
    assert result.bound_hi is None


def test_g3_honesty_sealed() -> None:
    payload = honesty_payload()
    assert payload["robust_without_enclosure"] is False
    payload["robust_without_enclosure"] = True
    assert honesty_payload()["robust_without_enclosure"] is False
    with pytest.raises(CertifiedStepForbidden, match="robust_without_enclosure"):
        CertifiedStepResult(
            accepted=False,
            bound_hi=None,
            reason="vacuous",
            p_max=2.0,
            robust_without_enclosure=True,
        )


def test_apply_flat_theta() -> None:
    net = apply_flat_theta(_affine(0.0), (1.25, 0.0))
    layer = net.layers[0]
    assert layer.weight == ((1.25,),)  # type: ignore[union-attr]


def test_output_box_accept() -> None:
    cfg = CertifiedStepConfig(
        property="output_box",
        p_max=2.0,
        cell=[Interval(0.0, 1.0)],
    )
    acc = certified_accept(_affine(1.5), config=cfg)
    assert acc.accepted
    assert acc.bound_hi is not None and acc.bound_hi <= 2.0
    rej = certified_accept(_affine(3.0), config=cfg)
    assert not rej.accepted
    assert rej.reason == "violates"


def test_temperature_collapse_refused() -> None:
    with pytest.raises(ValueError, match="bias collapse"):
        CertifiedStepConfig(collapse="temperature_collapse")


def test_cannot_accept_over_cap() -> None:
    with pytest.raises(CertifiedStepForbidden, match="exceeds"):
        CertifiedStepResult(
            accepted=True,
            bound_hi=2.5,
            reason="ok",
            p_max=2.0,
        )

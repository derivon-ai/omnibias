# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""08-01 recommended stack on the one-layer closed-form loss jet."""

from __future__ import annotations

import pytest
from omnibias.core.line_search import JetLineSearchConfig
from omnibias.core.train_stack import (
    DISCLAIMER,
    TrainStackConfig,
    gradient_descent_step,
    honesty_payload,
    recommended_stack_step,
    skill_vs_zero,
    stack_minimize,
    worked_example,
)
from omnibias.core.weight_loss_jet import one_layer_forward, one_layer_loss

_HIDDEN = 1
_DIM = 1
_XS = ((0.4,), (-0.3,), (0.8,), (-0.6,), (0.15,), (-0.9,))
_TEACHER = [0.0, 1.2, 0.1, 0.7]
_YS = tuple(one_layer_forward(x, _TEACHER, _HIDDEN, _DIM, "tanh") for x in _XS)
_STUDENT = [0.05, 0.9, 0.0, 0.4]


def _cfg(*, kantorovich: bool) -> TrainStackConfig:
    return TrainStackConfig(
        use_kantorovich=kantorovich,
        use_sharpness=True,
        line_search=JetLineSearchConfig(
            order=3, trust_radius=1.0, verify=True, max_step=1.0
        ),
    )


def test_honesty_and_disclaimer() -> None:
    payload = honesty_payload()
    assert payload["closed_form"] is True
    assert payload["one_layer_only"] is True
    assert payload["used_composed_by_default"] is False
    assert payload["skip_chain_rule"] is False
    assert payload["full_parameter_jacobian"] is False
    assert payload["global_min_claim"] is False
    assert payload["stretch_claim"] is False
    assert "chain rule" in DISCLAIMER
    assert "global min" in DISCLAIMER


def test_worked_example() -> None:
    example = worked_example()
    assert example["first_step_never_worse"] is True
    assert example["used_composed"] is False
    assert example["beat_start"] is True
    assert example["beat_gd"] is True
    assert example["skill_positive"] is True


def test_indefinite_hessian_still_descends() -> None:
    """4-point teacher/student has λ_min < 0; damping must restore descent."""
    _theta, report = recommended_stack_step(
        ((0.4,), (-0.3,), (0.8,), (-0.6,)),
        tuple(
            one_layer_forward(x, _TEACHER, _HIDDEN, _DIM, "tanh")
            for x in ((0.4,), (-0.3,), (0.8,), (-0.6,))
        ),
        _STUDENT,
        _HIDDEN,
        _DIM,
        config=_cfg(kantorovich=False),
    )
    assert report.lambda_min < 0.0
    assert report.loss1 < report.loss0
    assert report.step > 0.0


def test_one_step_never_worse() -> None:
    theta, report = recommended_stack_step(
        _XS, _YS, _STUDENT, _HIDDEN, _DIM, config=_cfg(kantorovich=False)
    )
    assert report.used_composed is False
    assert report.used_line_search is True
    assert report.loss1 <= report.loss0 + 1e-15
    assert one_layer_loss(_XS, _YS, theta, _HIDDEN, _DIM, "tanh") == pytest.approx(
        report.loss1
    )


def test_stack_beats_gd_and_zero_predictor() -> None:
    fitted, reports = stack_minimize(
        _XS, _YS, _STUDENT, _HIDDEN, _DIM, steps=6, config=_cfg(kantorovich=False)
    )
    assert all(item.used_composed is False for item in reports)
    start = one_layer_loss(_XS, _YS, _STUDENT, _HIDDEN, _DIM, "tanh")
    final = one_layer_loss(_XS, _YS, fitted, _HIDDEN, _DIM, "tanh")
    gd = list(_STUDENT)
    for _ in range(6):
        gd = gradient_descent_step(_XS, _YS, gd, _HIDDEN, _DIM, lr=0.15)
    gd_loss = one_layer_loss(_XS, _YS, gd, _HIDDEN, _DIM, "tanh")
    assert final < start
    assert final <= gd_loss + 1e-12
    assert skill_vs_zero(final, _YS) > 0.0


def test_kantorovich_path_is_never_worse() -> None:
    _theta, report = recommended_stack_step(
        _XS, _YS, _STUDENT, _HIDDEN, _DIM, config=_cfg(kantorovich=True)
    )
    assert report.used_kantorovich is True
    assert report.loss1 <= report.loss0 + 1e-15
    assert report.reason in {"ball", "empty", "bounds_failed", "never_worse", "step"}


def test_kantorovich_accepts_near_teacher() -> None:
    close = [0.0, 1.15, 0.1, 0.68]
    _theta, report = recommended_stack_step(
        _XS, _YS, close, _HIDDEN, _DIM, config=_cfg(kantorovich=True)
    )
    assert report.reason == "ball"
    assert report.accepted is True
    assert report.loss1 < report.loss0


def test_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="jet_order"):
        TrainStackConfig(jet_order=1)
    with pytest.raises(ValueError, match="damping"):
        TrainStackConfig(damping=-0.1)

# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Sound enclosure of the terminal-substitution policy-gradient bias (theory 10-03)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.control.certified.gradient_bias import (
    GradientBiasReport,
    honesty_payload,
    terminal_adjoint_error_bound,
    truncation_bias_bound,
    worked_example,
)


def test_honesty_payload_all_false():
    assert not any(honesty_payload().values())


def test_truncation_bias_bound_zero_error_gives_zero_bound():
    m = np.eye(2)
    dpi = np.ones((1, 2))
    b = np.ones((2, 1))
    report = truncation_bias_bound([dpi, dpi], [b, b], [m, m], terminal_error_bound=0.0)
    assert report.bound == 0.0
    assert report.certified


def test_truncation_bias_bound_scales_linearly_with_error():
    m = np.eye(2)
    dpi = np.ones((1, 2))
    b = np.ones((2, 1))
    r1 = truncation_bias_bound([dpi], [b], [m], terminal_error_bound=1.0)
    r2 = truncation_bias_bound([dpi], [b], [m], terminal_error_bound=2.0)
    assert r2.bound == pytest.approx(2.0 * r1.bound)


def test_truncation_bias_bound_rejects_length_mismatch():
    with pytest.raises(ValueError):
        truncation_bias_bound([np.ones((1, 2))], [np.ones((2, 1)), np.ones((2, 1))], [np.eye(2)], 1.0)


def test_truncation_bias_bound_rejects_negative_error():
    with pytest.raises(ValueError):
        truncation_bias_bound([np.ones((1, 2))], [np.ones((2, 1))], [np.eye(2)], -1.0)


def test_terminal_adjoint_error_bound_lipschitz_extension():
    from omnibias.core.verified.interval import Interval
    from omnibias.verify import LinearLayer, Network

    weight = ((0.5, 0.0), (0.0, 0.5))
    net = Network([LinearLayer(weight=weight, bias=(0.0, 0.0))])
    box = [Interval(-0.1, 0.1), Interval(-0.1, 0.1)]
    bound = terminal_adjoint_error_bound(net, box, point_residual=0.01)
    assert bound == pytest.approx(0.06, abs=1e-9)


def test_terminal_adjoint_error_bound_rejects_negative_residual():
    from omnibias.core.verified.interval import Interval
    from omnibias.verify import LinearLayer, Network

    net = Network([LinearLayer(weight=((1.0,),), bias=(0.0,))])
    with pytest.raises(ValueError):
        terminal_adjoint_error_bound(net, [Interval(-0.1, 0.1)], point_residual=-1.0)


def test_terminal_adjoint_error_bound_rejects_unknown_norm():
    from omnibias.core.verified.interval import Interval
    from omnibias.verify import LinearLayer, Network

    net = Network([LinearLayer(weight=((1.0,),), bias=(0.0,))])
    with pytest.raises(ValueError):
        terminal_adjoint_error_bound(net, [Interval(-0.1, 0.1)], point_residual=0.0, norm="bogus")


def test_worked_example_gate_g6():
    result = worked_example()
    assert result["g6_earned"] is True
    assert isinstance(result["gradient_bias_bound"], float)


def test_gradient_bias_report_is_dataclass_with_expected_fields():
    report = truncation_bias_bound([np.ones((1, 2))], [np.ones((2, 1))], [np.eye(2)], 1.0)
    assert isinstance(report, GradientBiasReport)
    assert report.horizon == 1
    assert len(report.per_step_terms) == 1
    assert math.isfinite(report.bound)


def test_gradient_bias_skill_gate_g6():
    from omnibias.control.certified.gradient_bias import gradient_bias_skill

    result = gradient_bias_skill(n=150, seed=11)
    assert result["g6_earned"] is True
    assert result["coverage"] == 1.0

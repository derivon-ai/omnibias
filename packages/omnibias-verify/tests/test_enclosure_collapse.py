# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 01-14: six first-class squeeze contracts."""

from __future__ import annotations

import math

from omnibias.core.remainder_train import exp_jet
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    WidthBudget,
    lohner_flow_jet,
    named_tower_fields,
    tower_field,
    tower_jacobian,
)
from omnibias.core.verified.kantorovich import polynomial_sqrt2_maps
from omnibias.core.verified.pde_certificate import helmholtz
from omnibias.verify import MLPArchitecture
from omnibias.verify.enclosure_collapse import (
    SqueezeReport,
    report_digest_ok,
    squeeze,
    squeeze_existence,
    squeeze_flow,
    squeeze_identifiability,
    squeeze_peak,
    squeeze_remainder,
    squeeze_residual,
)
from omnibias.verify.localization import Inconclusive, ScanResponse


def test_g4_six_callables_and_dispatch() -> None:
    names = (
        squeeze_peak,
        squeeze_residual,
        squeeze_identifiability,
        squeeze_existence,
        squeeze_remainder,
        squeeze_flow,
        squeeze,
    )
    assert all(callable(fn) for fn in names)


def test_g3_residual_helmholtz_splits_decrease() -> None:
    w1, w2 = 1.3, -0.7
    k = math.hypot(w1, w2)
    layers = [([[w1, w2]], [0.0], "cos")]
    domain = [(0.0, 1.0), (0.0, 1.0)]
    pde = helmholtz(2, k)
    mags: list[float] = []
    widths: list[float] = []
    for splits in (1, 4, 16, 64):
        report = squeeze_residual(layers, domain, pde, splits=splits)
        assert isinstance(report, SqueezeReport)
        assert report.kind == "residual"
        assert report.scope == "model_problem"
        assert report.honesty["continuum_navier_stokes_claim"] is False
        assert report.honesty["interval_verified"] is True
        assert report.width is not None
        assert report.inner is not None
        mags.append(float(report.inner.mag))
        widths.append(report.width)
    assert mags[0] > 1.0
    assert mags[-1] < 0.2
    assert widths == sorted(widths, reverse=True)
    assert all(w > 0.0 for w in widths)
    until = squeeze("residual", layers=layers, domain=domain, pde=pde, until="ulps", max_splits=8)
    assert until.honesty["continuum_navier_stokes_claim"] is False
    cap = squeeze_residual(layers, domain, pde, width_cap=2.0, max_splits=4)
    assert cap.status in {"certified", "inconclusive"}


def test_g5_peak_flat_is_inconclusive() -> None:
    report = squeeze_peak(ScanResponse.flat_max(), box=Interval(-0.5, 0.5))
    assert report.status == "inconclusive"
    assert isinstance(report.inner, Inconclusive)
    one = squeeze_peak(ScanResponse.sech2_peak(-0.3, alpha=5.0), box=Interval(-0.40, -0.20))
    assert one.status == "certified"
    assert one.scope == "local_box"
    assert one.certificate is not None
    assert report_digest_ok(one)
    assert one.honesty["theorem_prover_verified"] is False


def test_g4_identifiability_tiny_net() -> None:
    arch = MLPArchitecture(dims=(1, 1), activation="tanh")
    data = [((-1.0,), (1.0,)), ((1.0,), (-1.0,))]
    bounds = [(0.5, 1.5)] * arch.n_params
    report = squeeze_identifiability(arch, data, bounds, tol=1e-2, max_boxes=8_000)
    assert report.kind == "identifiability"
    assert report.scope == "parameter_box"
    assert report.honesty["p_vs_np_claim"] is False
    assert report.inner.result.f_lower > 0.0
    assert report.certificate is not None
    assert report_digest_ok(report)


def test_g5_existence_empty_ball_does_not_raise() -> None:
    empty = squeeze_existence(y0=1.0, z0=1.0, z1=1.0, z2=1.0)
    assert empty.status == "rejected"
    assert empty.inner is None
    func, jac, lip = polynomial_sqrt2_maps()
    good = squeeze_existence(
        func=func,
        jacobian=jac,
        a_inv=[[1.0 / 3.0]],
        trial_params=[1.5],
        lipschitz_df=lip,
        r_max=0.2,
    )
    assert good.status == "certified"
    assert good.honesty["continuum_pde_claim"] is False
    if good.certificate is not None:
        assert report_digest_ok(good)


def test_g4_remainder_tagged_as_truncation() -> None:
    xs = [-0.2, 0.0, 0.2]
    jet = exp_jet(2)
    values = [math.exp(x) for x in xs]
    report = squeeze_remainder(values, jet, xs)
    assert report.kind == "remainder"
    assert report.scope == "model_problem"
    assert isinstance(report.budget, WidthBudget)
    assert report.budget.dominant == "truncation"
    assert report.action is not None
    assert report.action.action == "raise_order"
    assert report.honesty["is_03_10"] is False


def test_g4_flow_reuses_widthbudget() -> None:
    spec = named_tower_fields()[0]
    run = lohner_flow_jet(
        tower_field(spec),
        tower_jacobian(spec),
        [Interval(0.2, 0.3)],
        0.05,
        4,
        order=6,
    )
    report = squeeze_flow(run)
    assert report.kind == "flow"
    assert report.scope == "finite_horizon"
    assert report.budget is run.budget
    assert report.action is not None
    assert report.honesty["navier_stokes_regularity_claim"] is False
    assert report.certificate is not None
    assert report_digest_ok(report)
    assert report.honesty["theorem_prover_verified"] is False
    assert report.honesty["mathlib_verified"] is False

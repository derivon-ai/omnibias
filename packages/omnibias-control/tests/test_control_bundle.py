# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The proof-carrying controller bundle (theory 10-02/10-03, Phase 3)."""

from __future__ import annotations

import numpy as np
from omnibias.control.bundle import build_bundle, bundle_status, honesty_payload, summary
from omnibias.control.certified.gradient_bias import truncation_bias_bound
from omnibias.control.horizon import certified_horizon
from omnibias.control.problem import RecoverableCertificate


def test_honesty_payload_all_false():
    assert not any(honesty_payload().values())


def test_bundle_status_uncertified_when_empty():
    bundle = build_bundle(theta=np.zeros(2))
    assert bundle.verdict == "uncertified"
    assert bundle_status(bundle) == "uncertified"


def test_bundle_status_certified_when_all_present_and_true():
    horizon = certified_horizon([np.diag([0.5, 0.5])] * 10, radius=0.0, tol=0.05)
    gb = truncation_bias_bound([np.ones((1, 2))], [np.ones((2, 1))], [np.eye(2)], 0.0)
    recoverable = RecoverableCertificate(f_lower=0.1, f_upper=0.2, boxes_explored=10, converged=True)
    bundle = build_bundle(theta=np.zeros(2), gradient_bias=gb, horizon=horizon, recoverable=recoverable)
    assert bundle.verdict == "certified"


def test_bundle_status_partial_when_one_slot_missing():
    horizon = certified_horizon([np.diag([0.5, 0.5])] * 10, radius=0.0, tol=0.05)
    bundle = build_bundle(theta=np.zeros(2), horizon=horizon)
    assert bundle.verdict == "partial"


def test_bundle_status_partial_when_one_certificate_false():
    horizon = certified_horizon([np.diag([1.5, 1.5])] * 3, radius=0.0, tol=0.05, max_horizon=3)
    gb = truncation_bias_bound([np.ones((1, 2))], [np.ones((2, 1))], [np.eye(2)], 0.0)
    recoverable = RecoverableCertificate(f_lower=0.1, f_upper=0.2, boxes_explored=10, converged=True)
    assert not horizon.certified
    bundle = build_bundle(theta=np.zeros(2), gradient_bias=gb, horizon=horizon, recoverable=recoverable)
    assert bundle.verdict == "partial"


def test_bundle_status_uncertified_when_all_present_but_false():
    horizon = certified_horizon([np.diag([1.5, 1.5])] * 3, radius=0.0, tol=0.05, max_horizon=3)
    recoverable = RecoverableCertificate(f_lower=-1.0, f_upper=2.0, boxes_explored=10, converged=True)
    assert not horizon.certified
    assert not recoverable.certified
    bundle = build_bundle(theta=np.zeros(2), horizon=horizon, recoverable=recoverable)
    assert bundle.verdict == "uncertified"


def test_summary_never_claims_certified_safe_control():
    bundle = build_bundle(theta=np.zeros(2))
    text = summary(bundle)
    assert "not certified safe control" in text
    assert "model-relative" in text

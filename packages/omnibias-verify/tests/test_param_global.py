# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the global training certificate (rigorous parameter-space global minimum).

Soundness is checked the way the repo requires: the certified global-minimum lower
bound ``f_lower`` is confirmed against a dense deterministic grid **and** a random
sample of the *true* training loss (an independent float reference, not the interval
machinery under test) -- ``f_lower <= L(theta)`` must hold at every sampled point, so
``f_lower <= min_theta L``. Honest non-convergence on a tiny budget still returns a
sound enclosure; the sealed certificate is tamper-evident and its positive-minimum
obligation is Lean well-formed.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.verify import (
    MLPArchitecture,
    certify_trained_global_min,
)

Data = list[tuple[tuple[float, ...], tuple[float, ...]]]


# --------------------------------------------------------------------------- #
# Independent float reference loss (no Interval arithmetic under test).
# --------------------------------------------------------------------------- #
def _net_out(arch: MLPArchitecture, theta: Sequence[float], x: Sequence[float]) -> list[float]:
    off = 0
    layers: list[tuple[list[list[float]], list[float]]] = []
    for n_out, n_in in arch.layer_shapes:
        weight = [[theta[off + r * n_in + c] for c in range(n_in)] for r in range(n_out)]
        off += n_out * n_in
        bias = [theta[off + r] for r in range(n_out)]
        off += n_out
        layers.append((weight, bias))
    a = [float(xi) for xi in x]
    last = len(layers) - 1
    for depth, (weight, bias) in enumerate(layers):
        z = [bias[o] + math.fsum(weight[o][j] * a[j] for j in range(len(weight[o]))) for o in range(len(weight))]
        a = z if depth == last else [math.tanh(zo) for zo in z]
    return a


def _loss_float(arch: MLPArchitecture, theta: Sequence[float], data: Data) -> float:
    total = 0.0
    for x, y in data:
        out = _net_out(arch, theta, x)
        total += math.fsum((out[o] - yo) ** 2 for o, yo in enumerate(y))
    return total / len(data)


def _objective_float(arch: MLPArchitecture, theta: Sequence[float], data: Data, l2: float) -> float:
    reg = l2 * math.fsum(t * t for t in theta) if l2 != 0.0 else 0.0
    return _loss_float(arch, theta, data) + reg


def _assert_f_lower_sound(
    arch: MLPArchitecture, data: Data, bounds: Sequence[tuple[float, float]], f_lower: float,
    *, per_axis: int, samples: int, l2: float = 0.0, seed: int = 0,
) -> None:
    """``f_lower <= J(theta)`` on a dense grid AND a random sample (so ``f_lower <= min J``)."""
    tol = 1e-9
    axes = [
        [lo + (hi - lo) * i / (per_axis - 1) for i in range(per_axis)] if per_axis > 1 else [0.5 * (lo + hi)]
        for lo, hi in bounds
    ]

    def _walk(prefix: list[float], depth: int) -> None:
        if depth == len(axes):
            assert f_lower <= _objective_float(arch, prefix, data, l2) + tol
            return
        for v in axes[depth]:
            _walk([*prefix, v], depth + 1)

    _walk([], 0)
    rng = random.Random(seed)
    for _ in range(samples):
        pt = [rng.uniform(lo, hi) for lo, hi in bounds]
        assert f_lower <= _objective_float(arch, pt, data, l2) + tol


# --------------------------------------------------------------------------- #
# Fixtures: an affine (P=2, convex) model and a tanh (P=4) model.
# --------------------------------------------------------------------------- #
ARCH_AFF = MLPArchitecture(dims=(1, 1), activation="tanh")  # affine readout (P=2)
ARCH_TANH = MLPArchitecture(dims=(1, 1, 1), activation="tanh")  # one tanh unit (P=4)

DATA_AFF_REAL: Data = [((-1.0,), (-0.5,)), ((0.0,), (0.0,)), ((1.0,), (0.5,))]  # y = 0.5 x
DATA_NONREAL: Data = [((-1.0,), (1.0,)), ((1.0,), (-1.0,))]


# --------------------------------------------------------------------------- #
# Convergence + soundness.
# --------------------------------------------------------------------------- #
def test_affine_realizable_global_min_is_near_zero() -> None:
    bounds = [(-1.0, 1.0)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_AFF_REAL, bounds, tol=1e-3, max_boxes=20_000)
    assert cert.converged
    assert cert.certified
    assert cert.verified
    assert cert.result.f_lower <= 1e-6  # realizable: global min is 0
    assert cert.result.f_upper <= 1e-3
    _assert_f_lower_sound(ARCH_AFF, DATA_AFF_REAL, bounds, cert.result.f_lower, per_axis=21, samples=5_000)


def test_nonrealizable_global_min_is_strictly_positive() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(
        ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000, strict_local_min=True
    )
    assert cert.converged
    assert cert.result.f_lower > 0.0  # architecture cannot fit this data on this box
    assert cert.strict_local_min is True  # convex quadratic -> PD Hessian at the argmin
    assert cert.verified
    _assert_f_lower_sound(ARCH_AFF, DATA_NONREAL, bounds, cert.result.f_lower, per_axis=21, samples=5_000)


def test_tanh_activation_path_is_sound() -> None:
    r = 0.2
    bounds = [(-r, r)] * ARCH_TANH.n_params
    cert = certify_trained_global_min(ARCH_TANH, DATA_NONREAL, bounds, tol=1e-2, max_boxes=3_000)
    assert cert.converged
    assert cert.verified
    assert cert.result.f_lower > 0.0
    _assert_f_lower_sound(ARCH_TANH, DATA_NONREAL, bounds, cert.result.f_lower, per_axis=7, samples=5_000)


def test_honest_nonconvergence_still_sound() -> None:
    r = 0.5
    bounds = [(-r, r)] * ARCH_TANH.n_params
    cert = certify_trained_global_min(ARCH_TANH, DATA_NONREAL, bounds, tol=1e-9, max_boxes=8)
    assert not cert.converged  # too tiny a budget to close the gap
    assert not cert.certified
    assert cert.result.f_lower <= cert.result.f_upper  # enclosure is still sound
    _assert_f_lower_sound(ARCH_TANH, DATA_NONREAL, bounds, cert.result.f_lower, per_axis=5, samples=2_000)


def test_l2_regularized_objective_is_recorded_and_sound() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    l2 = 0.1
    cert = certify_trained_global_min(ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000, l2=l2)
    assert cert.l2 == l2
    assert cert.certificate["meta"]["l2"] == l2
    assert cert.verified
    # the sealed enclosure is for the regularized objective J = L + l2||theta||^2.
    _assert_f_lower_sound(ARCH_AFF, DATA_NONREAL, bounds, cert.result.f_lower, per_axis=21, samples=5_000, l2=l2)


# --------------------------------------------------------------------------- #
# Certificate integrity + Lean gate.
# --------------------------------------------------------------------------- #
def test_certificate_tamper_is_detected() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000)
    assert cert.verified
    forged = copy.deepcopy(cert.certificate)
    forged["payload"]["interval"]["lo"] = "0.0"  # forge a weaker (lower) global-min bound
    assert not verify_certificate_digest(forged)


def test_positive_min_obligation_is_lean_well_formed() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000)
    assert cert.result.f_lower > 0.0
    obligation = generate_obligation(cert.certificate)
    assert obligation is not None
    assert "enclosed_quantity_pos" in obligation


def test_lean_gate_runs_or_degrades_gracefully() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000, lean=True)
    assert cert.result.f_lower > 0.0
    assert cert.lean is not None
    if lean_check_available():
        assert cert.lean.verified is True
        assert cert.theorem_prover_verified is True
    else:
        assert cert.lean.available is False
        assert cert.theorem_prover_verified is False


def test_lean_gate_skipped_when_min_includes_zero() -> None:
    bounds = [(-1.0, 1.0)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_AFF_REAL, bounds, tol=1e-3, max_boxes=20_000, lean=True)
    assert cert.result.f_lower <= 1e-6  # realizable: no positive obligation
    assert cert.lean is None
    assert cert.theorem_prover_verified is False


# --------------------------------------------------------------------------- #
# strict-min sub-claim: kernel-verified matrix positive-definiteness at the argmin
# --------------------------------------------------------------------------- #
def test_strict_min_seals_matrix_pd_pivot_vector() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(
        ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000, strict_local_min=True
    )
    assert cert.strict_local_min is True
    assert cert.positive_definite is True
    assert cert.pd_certificate is not None
    assert cert.verified is True
    payload = cert.pd_certificate["payload"]
    assert payload["type"] == "positive_definite"
    assert payload["n"] == ARCH_AFF.n_params
    obligation = generate_obligation(cert.pd_certificate)
    assert obligation is not None
    assert "allPivotsPos" in obligation


def test_no_pd_payload_without_strict_min_request() -> None:
    bounds = [(0.5, 1.5)] * ARCH_AFF.n_params
    cert = certify_trained_global_min(ARCH_AFF, DATA_NONREAL, bounds, tol=1e-3, max_boxes=20_000)
    assert cert.strict_local_min is None
    assert cert.positive_definite is False
    assert cert.pd_certificate is None


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
def test_param_bounds_length_must_match() -> None:
    with pytest.raises(ValueError, match="param_bounds"):
        certify_trained_global_min(ARCH_AFF, DATA_NONREAL, [(-1.0, 1.0)], tol=1e-3)


def test_param_bounds_need_lo_below_hi() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        certify_trained_global_min(ARCH_AFF, DATA_NONREAL, [(1.0, 1.0), (0.0, 1.0)], tol=1e-3)


def test_tol_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tol"):
        certify_trained_global_min(ARCH_AFF, DATA_NONREAL, [(-1.0, 1.0)] * 2, tol=0.0)


def test_negative_l2_rejected() -> None:
    with pytest.raises(ValueError, match="l2"):
        certify_trained_global_min(ARCH_AFF, DATA_NONREAL, [(-1.0, 1.0)] * 2, l2=-0.1)


def test_strict_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="strict_radius"):
        certify_trained_global_min(ARCH_AFF, DATA_NONREAL, [(-1.0, 1.0)] * 2, strict_radius=0.0)

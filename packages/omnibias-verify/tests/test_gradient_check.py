# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""W6-ext: certified finite-difference gradient checking.

A correct autodiff gradient must PASS (every residual band contains 0) and a
wrong one must be REJECTED with a sign-definite residual that yields a finite,
Lean-checkable obligation. The true partials are enclosed by the closed-form
tower intersected with the FD sandwich, so the residual bracket is tight
regardless of the stencil step.
"""

from __future__ import annotations

import importlib
import math

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.difference import sigma_deriv_bound
from omnibias.verify import (
    GradientCheckCertificate,
    Network,
    TanhLayer,
    affine_layer,
    certified_gradient_check,
    gradient_residual_certificate,
    mlp_axis_oracles,
    network_axis_oracles,
)


def _mpmath():  # type: ignore[no-untyped-def]
    try:
        return importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover
        return None


def _tanh_axis(t: float) -> float:
    return math.tanh(t)


def _tanh_bound(k: int, box: Interval) -> Interval:
    return sigma_tower_interval("tanh", box, k)[k]


class TestCertifiedGradientCheck:
    def test_correct_gradient_passes(self) -> None:
        # f(x, y) = tanh(x) + tanh(y); grad = (sech^2 x, sech^2 y).
        pt = [0.3, -0.7]
        grad = [1.0 / math.cosh(0.3) ** 2, 1.0 / math.cosh(-0.7) ** 2]
        gc = certified_gradient_check(
            [_tanh_axis, _tanh_axis], [_tanh_bound, _tanh_bound], grad, pt, step=1e-2
        )
        assert isinstance(gc, GradientCheckCertificate)
        assert gc.passed
        assert gc.mismatched_coordinates() == ()
        assert gc.max_abs_residual < 1e-12

    def test_true_partials_enclose_truth(self) -> None:
        pt = [0.4, 0.9]
        grad = [1.0 / math.cosh(0.4) ** 2, 1.0 / math.cosh(0.9) ** 2]
        gc = certified_gradient_check(
            [_tanh_axis, _tanh_axis], [_tanh_bound, _tanh_bound], grad, pt
        )
        for i in range(2):
            assert gc.true_partials[i].contains(grad[i])

    def test_scaled_wrong_gradient_rejected(self) -> None:
        pt = [0.3, -0.7]
        grad = [1.0 / math.cosh(0.3) ** 2, 1.0 / math.cosh(-0.7) ** 2]
        wrong = [1.5 * grad[0], grad[1]]  # coord 0 scaled wrong
        gc = certified_gradient_check(
            [_tanh_axis, _tanh_axis], [_tanh_bound, _tanh_bound], wrong, pt
        )
        assert not gc.passed
        assert gc.mismatched_coordinates() == (0,)

    def test_mismatch_yields_lean_obligation(self) -> None:
        pt = [0.5, 0.5]
        grad = [1.0 / math.cosh(0.5) ** 2, 1.0 / math.cosh(0.5) ** 2]
        wrong = [grad[0] + 0.25, grad[1]]
        gc = certified_gradient_check(
            [_tanh_axis, _tanh_axis], [_tanh_bound, _tanh_bound], wrong, pt
        )
        cert = gradient_residual_certificate(gc, 0)
        assert verify_certificate_digest(cert)  # sealed, tamper-evident
        obligation = generate_obligation(cert)
        assert obligation is not None
        assert "enclosed_quantity_pos" in obligation

    def test_passing_coordinate_has_no_obligation(self) -> None:
        pt = [0.5, 0.5]
        grad = [1.0 / math.cosh(0.5) ** 2, 1.0 / math.cosh(0.5) ** 2]
        gc = certified_gradient_check(
            [_tanh_axis, _tanh_axis], [_tanh_bound, _tanh_bound], grad, pt
        )
        cert = gradient_residual_certificate(gc, 0)
        # A residual band straddling 0 (a pass) carries no finite sign obligation.
        assert generate_obligation(cert) is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            certified_gradient_check([_tanh_axis], [_tanh_bound, _tanh_bound], [1.0], [0.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            certified_gradient_check([], [], [], [])

    def test_bad_coordinate_raises(self) -> None:
        pt = [0.3]
        grad = [1.0 / math.cosh(0.3) ** 2]
        gc = certified_gradient_check([_tanh_axis], [_tanh_bound], grad, pt)
        with pytest.raises(ValueError):
            gradient_residual_certificate(gc, 5)


class TestMLPAxisOracles:
    def _mlp(self) -> tuple[list[list[float]], list[float], list[float], list[float]]:
        w = [[0.5, -0.3, 0.2], [0.1, 0.4, -0.6]]
        b = [0.05, -0.1]
        v = [1.2, -0.7]
        pt = [0.3, -0.4, 0.8]
        return w, b, v, pt

    def _f(self, w, b, v, x):  # type: ignore[no-untyped-def]
        return sum(
            v[j] * math.tanh(sum(w[j][i] * x[i] for i in range(3)) + b[j]) for j in range(2)
        )

    def _grad(self, w, b, v, x):  # type: ignore[no-untyped-def]
        g = [0.0, 0.0, 0.0]
        for j in range(2):
            z = sum(w[j][i] * x[i] for i in range(3)) + b[j]
            s = 1.0 / math.cosh(z) ** 2
            for i in range(3):
                g[i] += v[j] * w[j][i] * s
        return g

    def test_axis_fn_matches_restricted_f(self) -> None:
        w, b, v, pt = self._mlp()
        fns, _ = mlp_axis_oracles(w, b, v, "tanh", pt)
        assert fns[0](pt[0]) == pytest.approx(self._f(w, b, v, pt), abs=1e-12)

    def test_mlp_correct_gradient_passes(self) -> None:
        w, b, v, pt = self._mlp()
        grad = self._grad(w, b, v, pt)
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, grad, pt, step=1e-2)
        assert gc.passed
        assert gc.max_abs_residual < 1e-12

    def test_mlp_wrong_coordinate_rejected(self) -> None:
        w, b, v, pt = self._mlp()
        grad = self._grad(w, b, v, pt)
        grad[1] *= 1.3
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, grad, pt)
        assert not gc.passed
        assert 1 in gc.mismatched_coordinates()

    def test_row_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            mlp_axis_oracles([[0.1, 0.2]], [0.0], [1.0], "tanh", [0.0, 0.0, 0.0])

    def test_head_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            mlp_axis_oracles([[0.1, 0.2, 0.3]], [0.0, 0.0], [1.0], "tanh", [0.0, 0.0, 0.0])


class TestNetworkAxisOracles:
    def _net(self) -> tuple[Network, list[list[float]], list[float], list[float], list[float]]:
        w = [[0.5, -0.3, 0.2], [0.1, 0.4, -0.6]]
        b = [0.05, -0.1]
        v = [1.2, -0.7]
        pt = [0.3, -0.4, 0.8]
        # A trained scalar MLP as the torch/jax frontends ingest it: [Linear, tanh, Linear].
        net = Network(
            [
                affine_layer(w, b),
                TanhLayer(),
                affine_layer([v], [0.42]),  # readout bias is a constant -> no gradient effect
            ]
        )
        return net, w, b, v, pt

    def _grad(self, w, b, v, x):  # type: ignore[no-untyped-def]
        g = [0.0, 0.0, 0.0]
        for j in range(2):
            z = sum(w[j][i] * x[i] for i in range(3)) + b[j]
            s = 1.0 / math.cosh(z) ** 2
            for i in range(3):
                g[i] += v[j] * w[j][i] * s
        return g

    def test_ingested_network_correct_gradient_passes(self) -> None:
        net, w, b, v, pt = self._net()
        grad = self._grad(w, b, v, pt)
        fns, bounds = network_axis_oracles(net, pt)
        gc = certified_gradient_check(fns, bounds, grad, pt, step=1e-2)
        assert gc.passed
        assert gc.max_abs_residual < 1e-12

    def test_ingested_network_wrong_gradient_rejected(self) -> None:
        net, w, b, v, pt = self._net()
        grad = self._grad(w, b, v, pt)
        grad[0] *= 1.4
        fns, bounds = network_axis_oracles(net, pt)
        gc = certified_gradient_check(fns, bounds, grad, pt)
        assert not gc.passed and 0 in gc.mismatched_coordinates()

    def test_non_scalar_readout_raises(self) -> None:
        net = Network([affine_layer([[0.1, 0.2]], [0.0]), TanhLayer(), affine_layer([[1.0], [2.0]], [0.0, 0.0])])
        with pytest.raises(ValueError):
            network_axis_oracles(net, [0.0, 0.0])

    def test_wrong_shape_network_raises(self) -> None:
        net = Network([affine_layer([[0.1, 0.2]], [0.0]), TanhLayer()])
        with pytest.raises(ValueError):
            network_axis_oracles(net, [0.0, 0.0])


class TestGradientCheckSigmaBound:
    def test_matches_sigma_deriv_bound_helper(self) -> None:
        # The difference sigma_deriv_bound gives the same oracle we hand-build.
        pt = [0.2, 0.6]
        grad = [1.0 / math.cosh(0.2) ** 2, 1.0 / math.cosh(0.6) ** 2]
        db = sigma_deriv_bound("tanh")
        gc = certified_gradient_check([_tanh_axis, _tanh_axis], [db, db], grad, pt)
        assert gc.passed

    @pytest.mark.skipif(_mpmath() is None, reason="mpmath not installed")
    def test_gaussian_gradient_passes(self) -> None:
        mp = _mpmath()
        # f(x) = gaussian(x) = exp(-x^2/2); f'(x) = -x exp(-x^2/2).
        pt = [0.7]

        def axis(t: float) -> float:
            return math.exp(-t * t / 2.0)

        def bound(k: int, box: Interval) -> Interval:
            return sigma_tower_interval("gaussian", box, k)[k]

        with mp.workdps(30):
            true = float(-mp.mpf("0.7") * mp.e ** (-mp.mpf("0.7") ** 2 / 2))
        gc = certified_gradient_check([axis], [bound], [true], pt)
        assert gc.passed

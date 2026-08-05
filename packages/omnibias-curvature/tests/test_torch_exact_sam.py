# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Behavioural tests for :class:`omnibias.curvature.torch.ExactSAM`.

ExactSAM packages the exact-curvature sharpness functionals as a drop-in,
amortised optimizer. These tests pin down that it:

1. descends the loss for every ``measure``, ``base``, and in both ``penalty`` / ``ascent`` modes
   (and the Adam base with ``lam=0`` is bit-for-bit Adam);
2. does the thing it promises -- an exact sharpness *penalty* reaches a strictly
   *flatter* minimum (lower exact ``||H||_F^2``) than the ``lam=0`` baseline, and
   than Adam (the generalisation-first mechanism; H4 of the double-descent study);
3. amortises correctly (a stale sharpness probe still descends);
4. validates its hyper-parameters.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from omnibias.curvature.torch import ExactSAM  # noqa: E402
from omnibias.curvature.torch import sharpness as S  # noqa: E402
from omnibias.torch.architectures import JetMLP  # noqa: E402


def _problem(
    seed: int = 0, in_dim: int = 2, hidden: int = 4, depth: int = 2, n: int = 12
) -> tuple[JetMLP, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(seed)
    net = JetMLP(in_dim, hidden, 1, depth=depth, base="tanh").double()
    x = torch.randn(n, in_dim, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    params = [p for p in net.parameters() if p.requires_grad]
    return net, x, y, params


def _mse(net: JetMLP, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((net(x).squeeze(-1) - y) ** 2).mean()


def _frobenius_sq(net: JetMLP, x: torch.Tensor, y: torch.Tensor, params: list[torch.Tensor]) -> float:
    """Exact ``||H||_F^2`` of the loss Hessian on a small net (dense oracle, no sampling)."""
    hess = S.dense_hessian(_mse(net, x, y), params)
    return float(S.hessian_frobenius_sq(hess))


# ---------------------------------------------------------------------------
# 1. Descent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("measure", ["frobenius", "trace", "top_eig"])
def test_exact_sam_reduces_loss_for_every_measure(measure: str) -> None:
    net, x, y, params = _problem(seed=0)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    opt = ExactSAM(params, lr=1e-2, lam=1e-3, measure=measure, n_samples=4, iters=15, seed=0)
    last = None
    for _ in range(60):
        last = opt.step(closure)
    assert last is not None and torch.isfinite(last)
    assert float(closure().detach()) < f0


def test_exact_sam_ascent_mode_reduces_loss() -> None:
    net, x, y, params = _problem(seed=3)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    opt = ExactSAM(params, lr=1e-2, mode="ascent", rho=0.05, iters=15, seed=0)
    for _ in range(60):
        opt.step(closure)
    assert float(closure().detach()) < f0


def test_exact_sam_sign_momentum_reduces_loss() -> None:
    net, x, y, params = _problem(seed=4)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    opt = ExactSAM(params, lr=2e-3, lam=1e-3, sign_momentum=True, seed=0)
    for _ in range(100):
        opt.step(closure)
    assert float(closure().detach()) < f0


# ---------------------------------------------------------------------------
# 1b. Adaptive base (precondition the sharpness-augmented gradient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", ["sgd", "adam", "frugal"])
def test_exact_sam_every_base_reduces_loss(base: str) -> None:
    net, x, y, params = _problem(seed=4)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    lr = 1e-1 if base == "sgd" else 1e-2  # adaptive bases normalise -> Adam-scale lr
    opt = ExactSAM(params, lr=lr, lam=1e-3, base=base, n_samples=4, seed=0)  # type: ignore[arg-type]
    for _ in range(80):
        opt.step(closure)
    assert float(closure().detach()) < f0


def test_exact_sam_adam_base_matches_adam_when_lam_zero() -> None:
    """``base="adam"`` with ``lam=0`` is exactly Adam applied to the loss gradient --
    the property that lets the Adam base degrade gracefully to Adam on registers where
    the sharpness penalty would hurt (the mse_tanh finding)."""
    net_esa, x, y, p_esa = _problem(seed=6)
    net_adam, _, _, p_adam = _problem(seed=6)  # identical init + data

    esa = ExactSAM(p_esa, lr=5e-3, lam=0.0, base="adam", momentum=0.9, beta2=0.999, eps=1e-8)
    adam = torch.optim.Adam(p_adam, lr=5e-3, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(40):
        esa.step(lambda: _mse(net_esa, x, y))
        adam.zero_grad()
        _mse(net_adam, x, y).backward()
        adam.step()

    for a, b in zip(p_esa, p_adam, strict=True):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)


def test_exact_sam_base_state_footprint() -> None:
    """The Adam base carries a second-moment buffer ``_v`` (extra ``O(P)`` state); the
    frugal base keeps only per-tensor scalars ``_c`` (``O(#tensors)``)."""

    def run(base: str) -> tuple[ExactSAM, int, int]:
        net, x, y, params = _problem(seed=0)
        opt = ExactSAM(params, lr=1e-2, lam=1e-3, base=base, seed=0)  # type: ignore[arg-type]
        for _ in range(3):
            opt.step(lambda: _mse(net, x, y))
        return opt, sum(p.numel() for p in params), len(params)

    sgd, _, _ = run("sgd")
    adam, numel, _ = run("adam")
    frugal, _, ntensors = run("frugal")
    assert sgd._v is None and frugal._v is None
    assert adam._v is not None and sum(v.numel() for v in adam._v) == numel
    assert frugal._c is not None and sum(c.numel() for c in frugal._c) == ntensors
    assert sgd._c is None


# ---------------------------------------------------------------------------
# 1c. Auto-lambda: the fit-preservation cap
# ---------------------------------------------------------------------------


def test_exact_sam_auto_lam_cap_direction() -> None:
    """``lam_eff = min(lam, lam_safety * ||gL||^2 / |<gL,gS>|)`` when the penalty opposes the
    loss gradient, and rides at ``lam`` when it does not -- verified by injecting a ``gS``
    parallel / anti-parallel to ``gL``."""

    def run(sign: float) -> float:
        p = torch.nn.Parameter(torch.randn(40, dtype=torch.float64))
        target = torch.randn(40, dtype=torch.float64)
        opt = ExactSAM([p], lr=1e-3, lam=1.0, lam_auto=True, lam_safety=0.5)
        # Inject gS = sign * gL so <gL, gS> = sign * ||gL||^2 exactly.
        opt._sharpness_grad = lambda loss: [  # type: ignore[method-assign]
            sign * torch.autograd.grad(loss, [p], retain_graph=True)[0].detach()
        ]
        opt.step(lambda: ((p - target) ** 2).mean())
        assert opt._lam_eff is not None
        return opt._lam_eff

    assert run(+1.0) == pytest.approx(1.0)  # aligned: uncapped at lam
    assert run(-1.0) == pytest.approx(0.5)  # opposed (gS = -gL): lam_eff = lam_safety


def test_exact_sam_auto_lam_shrinks_when_penalty_fights_fit() -> None:
    """On a real problem where the exact sharpness gradient opposes the fit, the effective
    penalty collapses well below the ``lam`` upper bound while the loss still descends."""
    net, x, y, params = _problem(seed=7)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    opt = ExactSAM(params, lr=1e-2, lam=1e-1, lam_auto=True, base="adam", n_samples=4, seed=0)
    lam_effs = []
    for _ in range(80):
        opt.step(closure)
        lam_effs.append(opt._lam_eff)
    assert min(le for le in lam_effs if le is not None) < 0.5 * 1e-1  # capping clearly engaged
    assert float(closure().detach()) < f0


def test_exact_sam_fixed_lam_reports_lam_eff() -> None:
    """With ``lam_auto=False`` the telemetry reports the constant ``lam`` (backward-compat)."""
    net, x, y, params = _problem(seed=0)
    opt = ExactSAM(params, lr=1e-2, lam=1e-3)
    opt.step(lambda: _mse(net, x, y))
    assert opt._lam_eff == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# 2. Mechanism: the exact penalty reaches a flatter minimum
# ---------------------------------------------------------------------------


def test_exact_sam_penalty_lowers_sharpness_vs_ablation() -> None:
    """Same init + data + base step: ``lam>0`` ends at a strictly lower exact ``||H||_F^2``
    than the ``lam=0`` ablation, while still reducing the data loss."""
    net_reg, x, y, p_reg = _problem(seed=1)
    net_base, _, _, p_base = _problem(seed=1)  # identical init and data (same seed)

    def train(net: JetMLP, params: list[torch.Tensor], lam: float) -> None:
        opt = ExactSAM(params, lr=5e-3, lam=lam, measure="frobenius", n_samples=6, seed=0)
        for _ in range(150):
            opt.step(lambda: _mse(net, x, y))

    f0 = _frobenius_sq(net_base, x, y, p_base)
    l0 = float(_mse(net_base, x, y).detach())
    train(net_base, p_base, lam=0.0)
    train(net_reg, p_reg, lam=3e-2)

    frob_base = _frobenius_sq(net_base, x, y, p_base)
    frob_reg = _frobenius_sq(net_reg, x, y, p_reg)
    loss_reg = float(_mse(net_reg, x, y).detach())

    assert loss_reg < l0  # the penalised run still trained
    assert frob_reg < frob_base  # ... to a strictly flatter minimum
    assert frob_base <= f0 + 1.0  # sanity: the ablation did not blow curvature up


# ---------------------------------------------------------------------------
# 3. Amortisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe_every", [1, 5])
def test_exact_sam_amortized_probe_still_descends(probe_every: int) -> None:
    net, x, y, params = _problem(seed=2)

    def closure() -> torch.Tensor:
        return _mse(net, x, y)

    f0 = float(closure().detach())
    opt = ExactSAM(params, lr=1e-2, lam=1e-3, probe_every=probe_every, n_samples=4, seed=0)
    for _ in range(60):
        opt.step(closure)
    assert opt._gS is not None  # a sharpness direction was cached
    assert float(closure().detach()) < f0


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------


def test_exact_sam_validation() -> None:
    p = [torch.zeros(2, requires_grad=True)]
    with pytest.raises(ValueError):
        ExactSAM(p, lr=0.0)
    with pytest.raises(ValueError):
        ExactSAM(p, lam=-1.0)
    with pytest.raises(ValueError):
        ExactSAM(p, mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExactSAM(p, measure="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExactSAM(p, rho=0.0)
    with pytest.raises(ValueError):
        ExactSAM(p, momentum=1.0)
    with pytest.raises(ValueError):
        ExactSAM(p, probe_every=0)
    with pytest.raises(ValueError):
        ExactSAM(p, n_samples=0)
    with pytest.raises(ValueError):
        ExactSAM(p, iters=0)
    with pytest.raises(ValueError):
        ExactSAM(p, base="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExactSAM(p, base="adam", sign_momentum=True)  # sign only valid for the sgd base
    with pytest.raises(ValueError):
        ExactSAM(p, beta2=1.0)
    with pytest.raises(ValueError):
        ExactSAM(p, eps=0.0)
    with pytest.raises(ValueError):
        ExactSAM(p, lam_auto=True, mode="ascent")  # no lambda knob in ascent mode
    with pytest.raises(ValueError):
        ExactSAM(p, lam_auto=True, lam=0.0)  # lam is the upper bound, must be > 0
    with pytest.raises(ValueError):
        ExactSAM(p, lam_safety=0.0)
    with pytest.raises(ValueError):
        ExactSAM(p, lam_safety=1.5)
    with pytest.raises(ValueError):
        ExactSAM([torch.zeros(2)])  # no parameter requires grad


# ---------------------------------------------------------------------------
# 5. Generalisation-first mechanism vs Adam (slow; the real proof is MNIST-1D)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_exact_sam_flatter_than_adam_at_comparable_loss() -> None:
    """From the same init/data, ExactSAM (exact Frobenius penalty) reaches a strictly
    flatter minimum than Adam while still fitting the data. The generalisation win itself
    is demonstrated at scale in ``examples/mnist1d_double_descent``; here we pin the
    exact-curvature mechanism that drives it."""
    net_sam, x, y, p_sam = _problem(seed=5, hidden=6, n=10)
    net_adam, _, _, p_adam = _problem(seed=5, hidden=6, n=10)

    opt = ExactSAM(p_sam, lr=5e-3, lam=3e-2, measure="frobenius", n_samples=6, seed=0)
    for _ in range(250):
        opt.step(lambda: _mse(net_sam, x, y))

    adam = torch.optim.Adam(p_adam, lr=5e-3)
    for _ in range(250):
        adam.zero_grad()
        _mse(net_adam, x, y).backward()
        adam.step()

    loss_sam = float(_mse(net_sam, x, y).detach())
    frob_sam = _frobenius_sq(net_sam, x, y, p_sam)
    frob_adam = _frobenius_sq(net_adam, x, y, p_adam)

    assert loss_sam < 1.0  # ExactSAM actually fit the (unit-variance) targets
    assert frob_sam < frob_adam  # ... at a strictly flatter minimum than Adam

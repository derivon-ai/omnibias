# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The optimizer arms: baselines, the omnibias second-order suite, and sharpness levers.

Each :class:`OptimizerArm` bundles a *driver* (how :mod:`~examples.mnist1d_double_descent.train`
must honour the optimizer's closure convention) with a *factory* that constructs it. The
driver values map onto the closure contract in ``omnibias.torch.optim``:

* ``standard``  -- plain :class:`torch.optim.Optimizer` (Adam / SGD): ``backward()`` then ``step()``.
* ``scalar``    -- ``step(closure)`` where the closure returns the *scalar loss* with the graph
  intact (``CubicNewton``, ``TrustRegionNewtonCG``, ``StochasticNewtonCG``, ``JetLBFGSOptimizer``,
  ``JetSubspaceTensor``, ``DiagonalCurvature`` hutchinson).
* ``residual``  -- ``step(closure)`` where the closure returns the *residual vector*
  ``r = f(x) - onehot(y)`` (``CubicGaussNewton``, ``DiagonalCurvature`` gauss-newton). The
  Gauss-Newton family only makes sense on a least-squares objective, so those arms are
  ``mse_tanh``-only.
* ``natural``   -- ``NaturalGradient`` with a (dense) Gauss-Newton Fisher metric attached by the
  trainer; the dense ``(P, P)`` Fisher caps it to narrow widths (``dense_only``).
* ``kfac``      -- ``KFAC`` (constructed with the module, steps a scalar-loss closure internally).
* ``sharpness`` -- an Adam base whose objective is replaced by an exact / stochastic sharpness
  functional (P3); the specific functional is ``sharpness_kind`` with ``hypers``.
* ``eos``       -- plain gradient descent whose learning rate is set *online* from the exact
  top Hessian eigenvalue, ``eta = c * 2 / lambda_max(H)`` (the exact edge-of-stability
  controller in :mod:`~examples.mnist1d_double_descent.eos`); ``hypers`` configure the controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from omnibias.curvature.torch import ExactSAM
from omnibias.torch.optim import (
    KFAC,
    CubicGaussNewton,
    CubicNewton,
    DiagonalCurvature,
    FrugalCurvature,
    JetLBFGSOptimizer,
    JetSubspaceTensor,
    NaturalGradient,
    StochasticNewtonCG,
    TrustRegionNewtonCG,
)
from torch import nn

from examples.mnist1d_double_descent.models import REGISTERS

#: The recognised driver tags (how the trainer honours the closure convention).
DRIVERS = ("standard", "scalar", "residual", "natural", "kfac", "sharpness", "eos")

_BOTH = REGISTERS
_MSE = ("mse_tanh",)

Factory = Callable[[nn.Module, float], torch.optim.Optimizer]


@dataclass(frozen=True)
class OptimizerArm:
    """One optimizer arm: a driver + a factory + its defaults and register constraints."""

    name: str
    driver: str
    factory: Factory
    lr: float = 1e-2
    registers: tuple[str, ...] = _BOTH
    dense_only: bool = False
    sharpness_kind: str | None = None
    hypers: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.driver not in DRIVERS:
            raise ValueError(f"driver must be one of {DRIVERS}, got {self.driver!r}")
        for reg in self.registers:
            if reg not in REGISTERS:
                raise ValueError(f"unknown register {reg!r} in arm {self.name!r}")

    def build(self, model: nn.Module, *, lr: float) -> torch.optim.Optimizer:
        """Construct the optimizer for ``model`` (``lr`` is honoured only where meaningful)."""
        return self.factory(model, lr)

    def valid_in(self, register: str) -> bool:
        return register in self.registers


# ---------------------------------------------------------------------------
# Factories (keyword-only optimizer constructors; lr used only where meaningful)
# ---------------------------------------------------------------------------


def _adam(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def _sgd(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def _eos_sgd(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # Plain GD carrier; the trainer overrides ``lr`` each step from the exact
    # edge-of-stability controller. Momentum 0 keeps the 2/lambda_max limit exact.
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)


def _cubic_newton(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return CubicNewton(model.parameters(), sigma=1.0, krylov_dim=20)


def _trust_region(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return TrustRegionNewtonCG(model.parameters(), radius=1.0)


def _stochastic_newton(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return StochasticNewtonCG(model.parameters(), damping=1.0)


def _jet_lbfgs(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return JetLBFGSOptimizer(model.parameters(), history_size=10)


def _diag_hutch(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return DiagonalCurvature(model.parameters(), lr=lr, curvature="hutchinson")


def _diag_gn(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return DiagonalCurvature(model.parameters(), lr=lr, curvature="gauss_newton")


def _frugal_hutch(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return FrugalCurvature(model.parameters(), lr=lr, curvature="hutchinson", curvature_every=5)


def _frugal_gn(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # ``curvature_subsample`` caps the exact per-row GN backward passes at a random
    # (unbiased) subset each refresh, so the arm stays tractable on large residuals.
    return FrugalCurvature(
        model.parameters(), lr=lr, curvature="gauss_newton", curvature_every=5,
        curvature_subsample=64, clip=1.0,
    )


def _exact_sam(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # ``probe_every=5`` amortises the expensive exact-sharpness probe (several
    # double-backward HVPs) over 5 cheap gradient steps -- the core "<= SAM cost" lever.
    return ExactSAM(
        model.parameters(), lr=lr, lam=1e-3, measure="frobenius",
        n_samples=4, iters=15, probe_every=5,
    )


def _exact_sam_adam(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # Adam-preconditioned base: "Adam + exact sharpness penalty". Adam-scale lr, so the
    # per-coordinate adaptivity handles the fit (esp. mse_tanh) while the penalty flattens.
    return ExactSAM(
        model.parameters(), lr=lr, lam=1e-3, measure="frobenius",
        n_samples=4, iters=15, probe_every=5, base="adam",
    )


def _exact_sam_frugal(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # Memory-lean adaptive base (per-tensor RMS): adaptivity at ~half the adam-base state.
    return ExactSAM(
        model.parameters(), lr=lr, lam=1e-3, measure="frobenius",
        n_samples=4, iters=15, probe_every=5, base="frugal",
    )


def _exact_sam_auto(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # SGD base + register/fit-aware auto-lambda: lam is the *upper bound* (3e-3, the ce_relu
    # sweep optimum); the fit-preservation cap collapses it where the penalty fights the fit.
    return ExactSAM(
        model.parameters(), lr=lr, lam=3e-3, lam_auto=True, lam_safety=0.5,
        measure="frobenius", n_samples=4, iters=15, probe_every=5,
    )


def _exact_sam_adam_auto(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # Adam base + auto-lambda -- the "beat Adam on both registers" candidate: on mse_tanh the
    # cap drives lambda->0 and it degrades to Adam; on ce_relu the penalty rides at the bound.
    return ExactSAM(
        model.parameters(), lr=lr, lam=3e-3, lam_auto=True, lam_safety=0.5,
        measure="frobenius", n_samples=4, iters=15, probe_every=5, base="adam",
    )


def _kfac(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return KFAC(model, lr=lr, damping=1e-2)


def _natural(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return NaturalGradient(model.parameters(), lr=lr, damping=1e-3)


def _cubic_gauss_newton(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return CubicGaussNewton(model.parameters(), sigma=1.0)


def _jet_subspace(order: int) -> Factory:
    def factory(model: nn.Module, lr: float) -> torch.optim.Optimizer:
        return JetSubspaceTensor(model.parameters(), subspace_dim=5, order=order)

    return factory


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_ARMS: dict[str, OptimizerArm] = {
    "adam": OptimizerArm(
        "adam", "standard", _adam, lr=1e-2, description="Adam (the paper's optimizer)"
    ),
    "sgd": OptimizerArm(
        "sgd", "standard", _sgd, lr=1e-1, description="SGD + momentum baseline"
    ),
    "eos": OptimizerArm(
        "eos", "eos", _eos_sgd, lr=1e-2,
        hypers={
            # measure_every amortises the exact-lambda_max HVP probe (>1 reuses the
            # last estimate between probes); probe_iters is the power-iteration count.
            # momentum widens the target to 2c(1+beta) and must match the SGD carrier.
            "c": 0.9, "momentum": 0.0, "eta_min": 1e-4, "eta_max": 1.0,
            "probe_iters": 10, "measure_every": 4, "ema": 0.5,
        },
        description="Exact edge-of-stability GD: eta = c*2/lambda_max(H) via omnibias top_eigenvalue",
    ),
    "cubic_newton": OptimizerArm(
        "cubic_newton", "scalar", _cubic_newton, lr=1.0,
        description="Adaptive cubic-regularised Newton on the exact full Hessian (ARC)",
    ),
    "trust_region": OptimizerArm(
        "trust_region", "scalar", _trust_region, lr=1.0,
        description="Trust-region Newton-CG on the exact full Hessian (Steihaug)",
    ),
    "stochastic_newton": OptimizerArm(
        "stochastic_newton", "scalar", _stochastic_newton, lr=1.0,
        description="Levenberg-damped Newton-CG (full-batch => Newton-CG)",
    ),
    "jet_lbfgs": OptimizerArm(
        "jet_lbfgs", "scalar", _jet_lbfgs, lr=1.0,
        description="L-BFGS with exact-curvature initial scale + Taylor line search",
    ),
    "diag_hutch": OptimizerArm(
        "diag_hutch", "scalar", _diag_hutch, lr=1e-1,
        description="Diagonal-curvature preconditioner (exact Hutchinson diagonal)",
    ),
    "kfac": OptimizerArm(
        "kfac", "kfac", _kfac, lr=1e-1,
        description="K-FAC natural gradient (Kronecker-factored Fisher)",
    ),
    "jet_subspace_o2": OptimizerArm(
        "jet_subspace_o2", "scalar", _jet_subspace(2), lr=1.0,
        description="Subspace tensor method, order 2 (quadratic subspace model)",
    ),
    "jet_subspace_o3": OptimizerArm(
        "jet_subspace_o3", "scalar", _jet_subspace(3), lr=1.0,
        description="Subspace tensor method, order 3 (cubic model; H7 redemption test)",
    ),
    "cubic_gauss_newton": OptimizerArm(
        "cubic_gauss_newton", "residual", _cubic_gauss_newton, lr=1.0, registers=_MSE,
        description="Cubic-regularised Gauss-Newton on the one-hot residual (ARC on J^T J)",
    ),
    "diag_gn": OptimizerArm(
        "diag_gn", "residual", _diag_gn, lr=1e-1, registers=_MSE,
        description="Diagonal Gauss-Newton preconditioner on the one-hot residual",
    ),
    "natural_gradient": OptimizerArm(
        "natural_gradient", "natural", _natural, lr=1.0, registers=_MSE, dense_only=True,
        description="Fisher-scoring natural gradient (dense Gauss-Newton Fisher metric)",
    ),
    "sam_stochastic": OptimizerArm(
        "sam_stochastic", "sharpness", _adam, lr=1e-2,
        sharpness_kind="sam_stochastic", hypers={"rho": 0.05},
        description="Classic SAM proxy: penalise ||grad L|| (the linear sharpness shadow)",
    ),
    "sam_exact": OptimizerArm(
        "sam_exact", "sharpness", _adam, lr=1e-2,
        sharpness_kind="sam_exact", hypers={"rho": 0.05, "iters": 20},
        description="SAM-done-right: exact ascent-free sam_objective (grad + top-eig term)",
    ),
    "sharpness_reg": OptimizerArm(
        "sharpness_reg", "sharpness", _adam, lr=1e-2,
        sharpness_kind="sharpness_reg",
        hypers={"lam": 1e-3, "measure": "frobenius", "n_samples": 4},
        description="Exact Frobenius sharpness regulariser L + lam*||H||_F^2 (rides sigma''')",
    ),
    # -- Phase 1: attack Adam's real weak spots (test error / optimizer memory) --
    "exact_sam": OptimizerArm(
        # Heavy-ball SGD base (like the ``sgd`` arm) => SGD-scale lr, not Adam's 1e-2.
        "exact_sam", "scalar", _exact_sam, lr=1e-1,
        description="ExactSAM: amortised exact-sharpness-penalty optimizer (generalisation-first; H4 as an optimizer)",
    ),
    "exact_sam_adam": OptimizerArm(
        # Adam base => Adam-scale lr (1e-2): "Adam + exact sharpness penalty".
        "exact_sam_adam", "scalar", _exact_sam_adam, lr=1e-2,
        description="ExactSAM with an Adam base: per-coordinate adaptivity + exact-sharpness penalty",
    ),
    "exact_sam_frugal": OptimizerArm(
        # Memory-lean adaptive base (per-tensor RMS) => Adam-scale lr (1e-2).
        "exact_sam_frugal", "scalar", _exact_sam_frugal, lr=1e-2,
        description="ExactSAM with a memory-lean per-tensor-RMS adaptive base + exact-sharpness penalty",
    ),
    "exact_sam_auto": OptimizerArm(
        # SGD base + auto-lambda (fit-preservation cap) => SGD-scale lr (1e-1).
        "exact_sam_auto", "scalar", _exact_sam_auto, lr=1e-1,
        description="ExactSAM (SGD base) with register/fit-aware auto-lambda (exact fit-preservation cap)",
    ),
    "exact_sam_adam_auto": OptimizerArm(
        # Adam base + auto-lambda => Adam-scale lr (1e-2): the "beat Adam on both" candidate.
        "exact_sam_adam_auto", "scalar", _exact_sam_adam_auto, lr=1e-2,
        description="ExactSAM (Adam base) with register/fit-aware auto-lambda (exact fit-preservation cap)",
    ),
    "frugal_hutch": OptimizerArm(
        # Diagonal-curvature preconditioner (like ``diag_hutch``) => lr=1e-1, not Adam's 1e-2.
        "frugal_hutch", "scalar", _frugal_hutch, lr=1e-1,
        description="FrugalCurvature: 1 momentum buffer + per-tensor exact Hutchinson curvature (~half Adam's state)",
    ),
    "frugal_gn": OptimizerArm(
        "frugal_gn", "residual", _frugal_gn, lr=1e-1, registers=_MSE,
        description="FrugalCurvature on the one-hot residual (per-tensor exact Gauss-Newton; lean state)",
    ),
}

#: Every arm, in a stable order (baselines, full-Hessian, quasi/diag, subspace, GN, sharpness).
ARMS: tuple[str, ...] = tuple(_ARMS)

#: A representative subset for quick sweeps (``--arms core``).
CORE_ARMS: tuple[str, ...] = (
    "adam",
    "sgd",
    "cubic_newton",
    "trust_region",
    "jet_lbfgs",
    "diag_hutch",
    "cubic_gauss_newton",
    "natural_gradient",
    "exact_sam",
    "frugal_hutch",
    "frugal_gn",
)

#: The P3 sharpness-intervention arms plus the Adam baseline (``--arms sharpness``).
SHARPNESS_ARMS: tuple[str, ...] = ("adam", "sam_stochastic", "sam_exact", "sharpness_reg")


def get_arm(name: str) -> OptimizerArm:
    """Look up an :class:`OptimizerArm` by name (one of :data:`ARMS`)."""
    try:
        return _ARMS[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; choose from {ARMS}") from None


def arms_for_register(register: str, arm_names: tuple[str, ...]) -> tuple[str, ...]:
    """The subset of ``arm_names`` valid in ``register`` (Gauss-Newton family is mse_tanh-only)."""
    return tuple(name for name in arm_names if get_arm(name).valid_in(register))


__all__ = [
    "ARMS",
    "CORE_ARMS",
    "DRIVERS",
    "OptimizerArm",
    "SHARPNESS_ARMS",
    "arms_for_register",
    "get_arm",
]

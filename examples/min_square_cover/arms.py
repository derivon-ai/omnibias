# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The optimiser *menu*: one coverage energy, many ways to descend it.

Every arm minimises the **same** annealed soft-coverage energy over the same square
centers + gates; they differ only in the optimiser -- a first-order baseline (Adam / SGD)
versus omnibias' exact-curvature second-order methods (cubic-regularised Newton /
Gauss-Newton and trust-region Newton-CG). The second-order arms consume the closed-form
``sigma^(n)`` tower through matrix-free Hessian-vector products, which is the whole point:
the coverage energy is riddled with saddles and plateaus that first-order methods stall on.

The ``kind`` selects the optimiser's closure contract:

* ``"first_order"`` -- standard ``zero_grad / backward / step`` (Adam, SGD);
* ``"scalar"`` -- closure returns the scalar energy, graph intact (CubicNewton, TR-NCG,
  JetLBFGS, JetSubspaceTensor);
* ``"residual"`` -- closure returns the residual vector (CubicGaussNewton,
  DiagonalCurvature in its Gauss-Newton mode);
* ``"functional_gn"`` -- the *functional* Gauss-Newton (Levenberg-Marquardt): it consumes a
  ``residual_fn(flat_params) -> residual`` and returns new flat params, so the solve loop packs
  / unpacks ``[centers, gate_logits]`` around it;
* ``"closed_form"`` -- the closed-form dense-Hessian saddle-free Newton step
  (``closed_form_newton_step``): no torch optimiser object, the solve loop applies the exact
  ``coverage_energy_hessian`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from omnibias.torch.optim import (
    CubicGaussNewton,
    CubicNewton,
    DiagonalCurvature,
    GaussNewton,
    JetLBFGSOptimizer,
    JetSubspaceTensor,
    TrustRegionNewtonCG,
)
from torch import Tensor

_OPTIMIZERS = (
    "adam",
    "sgd",
    "cubic_newton",
    "cubic_gauss_newton",
    "trust_region",
    "jet_lbfgs",
    "jet_subspace",
    "diagonal_curvature",
    "gauss_newton",
    "closed_form_newton",
)
_KINDS = ("first_order", "scalar", "residual", "functional_gn", "closed_form")

_KIND_OF: dict[str, str] = {
    "adam": "first_order",
    "sgd": "first_order",
    "cubic_newton": "scalar",
    "trust_region": "scalar",
    "cubic_gauss_newton": "residual",
    "jet_lbfgs": "scalar",
    "jet_subspace": "scalar",
    "diagonal_curvature": "residual",
    "gauss_newton": "functional_gn",
    "closed_form_newton": "closed_form",
}


@dataclass(frozen=True)
class Arm:
    """One solver arm: which optimiser drives the shared coverage energy."""

    name: str
    optimizer: str
    loss: str
    lr: float
    description: str

    @property
    def kind(self) -> str:
        """Closure contract implied by the optimiser (``first_order`` / ``scalar`` / ``residual``)."""
        return _KIND_OF[self.optimizer]

    def __post_init__(self) -> None:
        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; choose from {_OPTIMIZERS}")
        if self.loss not in ("softplus", "sq_hinge"):
            raise ValueError(f"loss must be 'softplus' or 'sq_hinge', got {self.loss!r}")
        least_squares = ("cubic_gauss_newton", "diagonal_curvature", "gauss_newton")
        if self.optimizer in least_squares and self.loss != "sq_hinge":
            raise ValueError(f"{self.optimizer} implies the sq_hinge (least-squares) objective")

    def make_optimizer(
        self, params: Iterable[Tensor]
    ) -> torch.optim.Optimizer | GaussNewton | None:
        """Instantiate the optimiser for this arm over ``params``.

        Returns a :class:`torch.optim.Optimizer` for the closure-based arms, the functional
        :class:`~omnibias.torch.optim.GaussNewton` for the ``gauss_newton`` arm (driven by the
        solve loop's ``functional_gn`` branch), or ``None`` for ``closed_form_newton`` (the solve
        loop applies the closed-form Hessian step directly).
        """
        param_list = list(params)
        if self.optimizer == "closed_form_newton":
            return None
        if self.optimizer == "adam":
            return torch.optim.Adam(param_list, lr=self.lr)
        if self.optimizer == "sgd":
            return torch.optim.SGD(param_list, lr=self.lr, momentum=0.9)
        if self.optimizer == "cubic_newton":
            return CubicNewton(param_list)
        if self.optimizer == "trust_region":
            return TrustRegionNewtonCG(param_list)
        if self.optimizer == "cubic_gauss_newton":
            return CubicGaussNewton(param_list)
        if self.optimizer == "jet_lbfgs":
            return JetLBFGSOptimizer(param_list)
        if self.optimizer == "jet_subspace":
            return JetSubspaceTensor(param_list, subspace_dim=5, order=3)
        if self.optimizer == "diagonal_curvature":
            return DiagonalCurvature(param_list, lr=self.lr, curvature="gauss_newton")
        return GaussNewton(solver="qr", damping_strategy="nielsen")


def _arm(name: str, optimizer: str, description: str, *, loss: str = "softplus", lr: float = 0.05) -> Arm:
    return Arm(name=name, optimizer=optimizer, loss=loss, lr=lr, description=description)


_ARMS: dict[str, Arm] = {
    "adam": _arm("adam", "adam", "first-order Adam baseline on the softplus energy", lr=0.1),
    "sgd": _arm("sgd", "sgd", "first-order SGD+momentum baseline", lr=0.05),
    "cubic_newton": _arm(
        "cubic_newton", "cubic_newton", "exact-Hessian cubic-regularised Newton (ARC), saddle-escaping"
    ),
    "trust_region": _arm(
        "trust_region", "trust_region", "exact-Hessian trust-region Newton-CG (Steihaug), saddle-escaping"
    ),
    "cubic_gauss_newton": _arm(
        "cubic_gauss_newton",
        "cubic_gauss_newton",
        "cubic-regularised Gauss-Newton on the coverage residual (PSD metric)",
        loss="sq_hinge",
    ),
    "jet_lbfgs": _arm(
        "jet_lbfgs",
        "jet_lbfgs",
        "limited-memory BFGS with exact-curvature initial scale + Taylor line search",
    ),
    "jet_subspace": _arm(
        "jet_subspace",
        "jet_subspace",
        "exact third-order tensor method in a Krylov subspace (matrix-free)",
    ),
    "diagonal_curvature": _arm(
        "diagonal_curvature",
        "diagonal_curvature",
        "exact Gauss-Newton diagonal preconditioner (Adam substitute)",
        loss="sq_hinge",
    ),
    "gauss_newton": _arm(
        "gauss_newton",
        "gauss_newton",
        "functional Levenberg-Marquardt Gauss-Newton (QR, Nielsen damping)",
        loss="sq_hinge",
    ),
    "closed_form_newton": _arm(
        "closed_form_newton",
        "closed_form_newton",
        "closed-form dense-Hessian saddle-free Newton (exact coverage_energy_hessian)",
    ),
}

#: Canonical arm order for sweeps / tables (baselines first, then second-order).
ARMS: tuple[str, ...] = tuple(_ARMS)


def get_arm(name: str) -> Arm:
    """Look up an :class:`Arm` by name (one of :data:`ARMS`)."""
    try:
        return _ARMS[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; choose from {ARMS}") from None


__all__ = ["ARMS", "Arm", "get_arm"]

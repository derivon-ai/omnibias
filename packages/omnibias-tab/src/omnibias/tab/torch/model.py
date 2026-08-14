# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Trainable PyTorch soft decision-tree ensemble (bit-identical to the numpy / jax twins).

:class:`SoftTreeEnsemble` is an ``nn.Module`` whose parameters are exactly the arrays of
:class:`omnibias.tab._core.params.TabParams`, so a model trained here round-trips to numpy
for certification (:mod:`omnibias.tab.certify`) and to the jax twin
(:mod:`omnibias.tab.jax.model`). ``forward`` is a tensor-in / tensor-out layer
(``X`` shape ``(..., d)`` -> scores ``(..., n_outputs)``) and composes with any
encoder; constructors stay float64 CPU, so call ``.to(device, dtype)`` before
plugging into a host net. The forward is differentiable in every parameter -- the
oblique directions ``W``, the thresholds ``t`` and the leaves -- so the whole tree
(splits included) is trained by the exact-curvature optimizers in
:mod:`omnibias.tab.torch.train`.

Terminology: the split gate ``sigmoid(beta (W.x - t))`` hardens as ``beta -> inf`` -- the
feasibility / temperature sense of "collapse", distinct from the **founding bias
collapse** (the multi-bias ``delta -> 0`` limit to the closed-form derivative
``sigma^(K-1)``; see :mod:`omnibias.torch.unit`). The tower's exact gate curvature is what
the second-order trainer consumes.
"""

from __future__ import annotations

import numpy as np
import torch
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.params import TabParams, init_params, leaf_code_matrix
from torch import Tensor, nn

_DTYPE = torch.float64


class SoftTreeEnsemble(nn.Module):
    r"""Ensemble of ``n_trees`` oblivious soft trees of ``depth`` gates.

    Parameters
    ----------
    config:
        The :class:`~omnibias.tab._core.config.SoftTreeConfig` shape descriptor.
    params:
        Optional :class:`~omnibias.tab._core.params.TabParams` to load; otherwise the
        ensemble is randomly initialised from ``config.seed``.
    """

    def __init__(
        self,
        config: SoftTreeConfig,
        params: TabParams | None = None,
        *,
        learnable_beta: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        p = params if params is not None else init_params(config)
        self.W = nn.Parameter(torch.as_tensor(p.W, dtype=_DTYPE))
        self.t = nn.Parameter(torch.as_tensor(p.t, dtype=_DTYPE))
        self.leaves = nn.Parameter(torch.as_tensor(p.leaves, dtype=_DTYPE))
        self.b0 = nn.Parameter(torch.as_tensor(p.b0, dtype=_DTYPE))
        codes = torch.as_tensor(leaf_code_matrix(config.depth), dtype=_DTYPE)
        self.register_buffer("_codes", codes)
        beta0 = torch.tensor(float(config.beta_final), dtype=_DTYPE)
        if learnable_beta:
            self._beta = nn.Parameter(beta0)
        else:
            self.register_buffer("_beta", beta0, persistent=True)

    # ----- beta (gate sharpness) ---------------------------------------- #
    @property
    def beta(self) -> float:
        return float(self._beta.detach().cpu().item())

    def set_beta(self, beta: float) -> None:
        with torch.no_grad():
            self._beta.fill_(float(beta))

    # ----- forward ------------------------------------------------------ #
    def forward(self, X: Tensor, beta: float | None = None) -> Tensor:
        r"""Raw ensemble scores ``F`` of shape ``(..., n_outputs)``.

        ``X`` may have leading batch dims ``(..., d)``. Plugin use: call
        ``.to(device=z.device, dtype=z.dtype)`` then ``forward(z)`` inside the
        host graph. Constructors stay float64 CPU for certify / G3 parity.
        """
        b = self._beta if beta is None else X.new_tensor(float(beta), dtype=self.W.dtype)
        if X.ndim < 2:
            raise ValueError("X must have shape (..., n_features)")
        leading = tuple(int(s) for s in X.shape[:-1])
        rows = X.reshape(-1, X.shape[-1])
        z = torch.einsum("nd,mjd->nmj", rows, self.W) - self.t.unsqueeze(0)
        g = torch.sigmoid(b * z)  # (n, T, D)
        codes = self._codes  # (L, D)
        assert isinstance(codes, Tensor)  # a registered buffer (narrow the nn.Module union)
        gexp = g.unsqueeze(2)  # (n, T, 1, D)
        bexp = codes.view(1, 1, codes.shape[0], codes.shape[1])  # (1, 1, L, D)
        factors = bexp * gexp + (1.0 - bexp) * (1.0 - gexp)  # (n, T, L, D)
        memberships = factors.prod(dim=-1)  # (n, T, L)
        out = torch.einsum("nml,mlk->nk", memberships, self.leaves) + self.b0.unsqueeze(0)
        return out.reshape(*leading, out.shape[-1])

    # ----- numpy conveniences ------------------------------------------- #
    def _to_tensor(self, X: np.ndarray) -> Tensor:
        return torch.as_tensor(
            np.asarray(X, dtype=np.float64), dtype=self.W.dtype, device=self.W.device
        )

    def score(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        with torch.no_grad():
            F = self.forward(self._to_tensor(X), beta=beta)
        arr: np.ndarray = F.detach().cpu().numpy()
        return arr

    def predict_proba(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        F = self.score(X, beta=beta)
        if self.config.task == "binary":
            from omnibias.tab._core.forward import sigmoid_np

            prob: np.ndarray = sigmoid_np(F[:, 0])
            return prob
        if self.config.task == "multiclass":
            from omnibias.tab._core.forward import softmax_np

            probm: np.ndarray = softmax_np(F)
            return probm
        raise ValueError("predict_proba is only defined for classification tasks")

    def predict(self, X: np.ndarray, beta: float | None = None) -> np.ndarray:
        F = self.score(X, beta=beta)
        if self.config.task == "binary":
            labels: np.ndarray = (F[:, 0] > 0.0).astype(np.float64)
            return labels
        if self.config.task == "multiclass":
            idx: np.ndarray = np.argmax(F, axis=-1).astype(np.float64)
            return idx
        out: np.ndarray = F if F.shape[1] > 1 else F[:, 0]
        return out

    # ----- additive (depth-1) Linear -> Sigmoid -> Linear reparam -------- #
    def to_additive_sequential(self, beta: float | None = None) -> nn.Sequential:
        r"""The depth-1 model as an ``nn.Sequential(Linear, Sigmoid, Linear)`` (``beta`` folded in).

        Exactly reproduces the ensemble score with ``beta`` baked into the first layer's
        weights (``W1 = beta W``, ``b1 = -beta t``) and the leaves folded into the readout
        (``W2 = leaves[:, 1] - leaves[:, 0]``, ``b2 = b0 + sum_m leaves[:, 0]``). This is the
        form ``omnibias-verify`` ingests for certification and the form ``KFAC`` (a natural-
        gradient preconditioner for ``nn.Linear`` stacks) trains. Requires ``depth == 1``.
        """
        if self.config.depth != 1:
            raise ValueError("to_additive_sequential requires depth == 1 (the additive tier)")
        b = float(self._beta.detach().cpu().item()) if beta is None else float(beta)
        d, T, k = self.config.n_features, self.config.n_trees, self.config.n_outputs
        lin1 = nn.Linear(d, T).to(_DTYPE)
        lin2 = nn.Linear(T, k).to(_DTYPE)
        with torch.no_grad():
            lin1.weight.copy_(b * self.W[:, 0, :])
            lin1.bias.copy_(-b * self.t[:, 0])
            u = self.leaves[:, 1, :] - self.leaves[:, 0, :]  # (T, k)
            lin2.weight.copy_(u.t())
            lin2.bias.copy_(self.b0 + self.leaves[:, 0, :].sum(dim=0))
        return nn.Sequential(lin1, nn.Sigmoid(), lin2)

    def load_from_additive_sequential(self, seq: nn.Sequential, beta: float) -> None:
        r"""Inverse of :meth:`to_additive_sequential` (writes back to ``W`` / ``t`` / leaves)."""
        if self.config.depth != 1:
            raise ValueError("load_from_additive_sequential requires depth == 1")
        lin1, lin2 = seq[0], seq[2]
        assert isinstance(lin1, nn.Linear) and isinstance(lin2, nn.Linear)
        with torch.no_grad():
            self.W.copy_((lin1.weight / beta).unsqueeze(1))
            self.t.copy_((-lin1.bias / beta).unsqueeze(1))
            self.leaves[:, 0, :].zero_()
            self.leaves[:, 1, :].copy_(lin2.weight.t())
            self.b0.copy_(lin2.bias)
        self.set_beta(beta)

    # ----- (de)serialisation to the backend-agnostic container ---------- #
    def to_params(self) -> TabParams:
        r"""Snapshot the current parameters as a numpy :class:`TabParams`."""
        return TabParams(
            self.config,
            self.W.detach().cpu().numpy(),
            self.t.detach().cpu().numpy(),
            self.leaves.detach().cpu().numpy(),
            self.b0.detach().cpu().numpy(),
        )

    def load_params(self, params: TabParams) -> None:
        r"""Copy a numpy :class:`TabParams` into the module's parameters (in place)."""
        dt = self.W.dtype
        with torch.no_grad():
            self.W.copy_(torch.as_tensor(params.W, dtype=dt, device=self.W.device))
            self.t.copy_(torch.as_tensor(params.t, dtype=dt, device=self.t.device))
            self.leaves.copy_(torch.as_tensor(params.leaves, dtype=dt, device=self.leaves.device))
            self.b0.copy_(torch.as_tensor(params.b0, dtype=dt, device=self.b0.device))


__all__ = ["SoftTreeEnsemble"]

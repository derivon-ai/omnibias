# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CvxLayer-OMBU: differentiable embedded convex solvers.

Two reference architectures:

- :class:`CvxLasso`: depth-T unrolled ISTA for LASSO regression. The
  soft-thresholding step is realised as a multi-bias-Huber K=2 collapse
  (``clip(z, -lambda, lambda)`` is the closed-form first derivative of
  the Huber loss).

- :class:`CvxLogistic`: depth-T unrolled gradient descent for logistic
  regression. Each step computes the predicted probabilities via the
  K=2 collapse of softplus (which equals sigmoid).

Backpropagating through the unrolled loop gives gradients w.r.t. the
sensing matrix, the step sizes, and the (optional) shrinkage threshold,
realising the classical deep-unfolding pattern (LISTA, ALISTA, ...) in
the OMBU framework.
"""

from __future__ import annotations

from omnibias.torch.activations.proximal import make_huber_spec
from omnibias.torch.blocks import OperatorBlock

import torch
import torch.nn as nn
from torch import Tensor


class CvxLasso(nn.Module):
    """Depth-T unrolled ISTA for ``min_x 0.5 ||A x - y||^2 + tau ||x||_1``.

    Iteration::

        z_t      = x_t - eta_t * A^T (A x_t - y)
        x_{t+1}  = z_t - clip(z_t, -lambda_t, lambda_t)            # soft-thresh

    where ``lambda_t = eta_t * tau``. The ``clip`` operator is the K=2
    bias-collapse of the Huber loss with ``tau = 1``; we apply the
    standard scaling identity ``clip(z, -lambda, lambda) = lambda *
    clip(z / lambda, -1, 1)`` to support a per-layer threshold without
    rebuilding the spec.

    Parameters
    ----------
    n_features : int
        Dimension of the unknown ``x``.
    n_obs : int
        Dimension of the observation ``y``.
    T : int, default 20
        Number of unrolled iterations.
    tau : float, default 0.1
        L1-regularisation strength.
    init_step : float, default 0.1
        Initial step size (per-layer learnable scalar).
    learn_A : bool, default True
        Whether the sensing matrix ``A`` is learnable.
    learn_tau : bool, default False
        Whether ``tau`` is a learnable scalar.
    """

    def __init__(
        self,
        n_features: int,
        n_obs: int,
        T: int = 20,
        tau: float = 0.1,
        init_step: float = 0.1,
        learn_A: bool = True,
        learn_tau: bool = False,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        self.n_features = n_features
        self.n_obs = n_obs
        self.T = T

        A_init = torch.randn(n_obs, n_features) / max(n_obs, n_features) ** 0.5
        if learn_A:
            self.A = nn.Parameter(A_init)
        else:
            self.register_buffer("A", A_init)

        self.steps = nn.Parameter(torch.full((T,), init_step))
        if learn_tau:
            self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
        else:
            self.register_buffer("log_tau", torch.tensor(float(tau)).log())

        # Huber with tau=1; we rescale per-layer to get the right threshold.
        spec = make_huber_spec(tau=1.0)
        self.shrink = OperatorBlock(op="grad", base=spec, channels=n_features)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def forward(
        self, y: Tensor, *, return_trace: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Solve the LASSO problem for the batch ``y``.

        Parameters
        ----------
        y : Tensor of shape ``(B, n_obs)``
        return_trace : bool, default False
            If True, also return the list of intermediate iterates.

        Returns
        -------
        x : Tensor of shape ``(B, n_features)``
            Final iterate.
        """
        if y.dim() != 2 or y.size(-1) != self.n_obs:
            raise ValueError(f"y must have shape (B, {self.n_obs}), got {tuple(y.shape)}.")
        B = y.size(0)
        x = torch.zeros(B, self.n_features, dtype=y.dtype, device=y.device)
        trace = [x.clone()] if return_trace else None
        eps = 1e-8
        for t in range(self.T):
            residual = x @ self.A.T - y  # (B, n_obs)
            grad = residual @ self.A  # (B, n_features)
            z = x - self.steps[t] * grad
            lam = (self.steps[t] * self.tau).abs().clamp(min=eps)
            # clip(z, -lam, lam) = lam * clip(z/lam, -1, 1) = lam * grad-Huber(z/lam; 1)
            clip_z = lam * self.shrink(z / lam)
            x = z - clip_z  # soft-threshold(z, lam)
            if trace is not None:
                trace.append(x.clone())
        return (x, trace) if trace is not None else x

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """LASSO objective ``0.5 ||A x - y||^2 + tau ||x||_1``."""
        residual = x @ self.A.T - y
        return 0.5 * (residual**2).sum(dim=-1) + self.tau * x.abs().sum(dim=-1)


class CvxLogistic(nn.Module):
    """Depth-T unrolled gradient descent for binary logistic regression.

    Iteration::

        p_t      = sigmoid(X w_t)                              # K=2 collapse of softplus
        grad     = X^T (p_t - y) / B  + l2_reg * w_t
        w_{t+1}  = w_t - eta_t * grad

    Parameters
    ----------
    n_features : int
    T : int, default 20
    init_step : float, default 0.1
    l2_reg : float, default 0.0
    """

    def __init__(
        self,
        n_features: int,
        T: int = 20,
        init_step: float = 0.1,
        l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        self.n_features = n_features
        self.T = T
        self.l2_reg = l2_reg
        self.steps = nn.Parameter(torch.full((T,), init_step))
        # K=2 collapse of softplus = sigmoid; uses the analytic-derivative path.
        self.sigmoid_op = OperatorBlock(op="grad", base="softplus", channels=1)

    def _sigmoid(self, z: Tensor) -> Tensor:
        # OperatorBlock works on (..., channels=1); reshape to add the channel axis.
        out: Tensor = self.sigmoid_op(z.unsqueeze(-1)).squeeze(-1)
        return out

    def forward(
        self, X: Tensor, y: Tensor, *, return_trace: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Solve the logistic problem for the batch ``(X, y)``.

        Parameters
        ----------
        X : Tensor of shape ``(B, n_features)``
        y : Tensor of shape ``(B,)`` with values in ``{0, 1}``
        return_trace : bool, default False
        """
        if X.size(-1) != self.n_features:
            raise ValueError(f"X last dim must be {self.n_features}, got {X.size(-1)}.")
        if y.shape != X.shape[:-1]:
            raise ValueError(f"y shape {tuple(y.shape)} must match X.shape[:-1].")
        B = X.size(0)
        w = torch.zeros(self.n_features, dtype=X.dtype, device=X.device)
        trace = [w.clone()] if return_trace else None
        for t in range(self.T):
            logits = X @ w
            probs = self._sigmoid(logits)
            grad = X.T @ (probs - y.to(X.dtype)) / B
            if self.l2_reg > 0:
                grad = grad + self.l2_reg * w
            w = w - self.steps[t] * grad
            if trace is not None:
                trace.append(w.clone())
        return (w, trace) if trace is not None else w

    def predict_proba(self, X: Tensor, w: Tensor) -> Tensor:
        return self._sigmoid(X @ w)


__all__ = ["CvxLasso", "CvxLogistic"]

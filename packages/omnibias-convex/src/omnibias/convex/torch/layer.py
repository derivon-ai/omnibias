# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable ``argmin`` for LP/QP (torch) -- the KKT implicit-function layer.

Torch twin of :mod:`omnibias.convex.jax.layer`: a :class:`torch.autograd.Function`
whose forward is the interior-point solve and whose backward solves the linearised
KKT system ``K^T [y_x; y_l] = -[g; 0]`` once for the adjoint (OptNet style). Runs
in float64.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from omnibias.convex.problem import BarrierOptions
from omnibias.convex.torch.solver import solve_qp
from torch import Tensor


class _QPLayerFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        Q: Tensor,
        c: Tensor,
        A: Tensor,
        b: Tensor,
        options: BarrierOptions | None,
    ) -> Tensor:
        sol = solve_qp(Q, c, A, b, options=options)
        ctx.save_for_backward(sol.x, sol.dual, sol.slack, torch.as_tensor(Q, dtype=torch.float64),
                              torch.as_tensor(A, dtype=torch.float64))
        ctx.in_dtypes = (
            torch.as_tensor(Q).dtype,
            torch.as_tensor(c).dtype,
            torch.as_tensor(A).dtype,
            torch.as_tensor(b).dtype,
        )
        return cast(Tensor, sol.x)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, g: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None]:
        x, lam, slack, Q, A = ctx.saved_tensors
        n = x.shape[0]
        m = slack.shape[0]
        g64 = g.to(torch.float64)
        top = torch.cat([Q, A.T], dim=1)
        bottom = torch.cat([lam[:, None] * A, torch.diag(-slack)], dim=1)
        kkt = torch.cat([top, bottom], dim=0)
        rhs = -torch.cat([g64, torch.zeros((m,), dtype=torch.float64)])
        y = torch.linalg.solve(kkt.T, rhs)
        y_x = y[:n]
        y_l = y[n:]
        grad_Q = 0.5 * (torch.outer(y_x, x) + torch.outer(x, y_x))
        grad_c = y_x
        grad_A = torch.outer(lam, y_x) + torch.outer(lam * y_l, x)
        grad_b = -lam * y_l
        dq, dc, da, db = ctx.in_dtypes
        return grad_Q.to(dq), grad_c.to(dc), grad_A.to(da), grad_b.to(db), None


def qp_layer(
    Q: Tensor, c: Tensor, A: Tensor, b: Tensor, options: BarrierOptions | None = None
) -> Tensor:
    """Differentiable QP optimiser ``x*`` of ``min 1/2 x^T Q x + c^T x s.t. A x <= b``."""
    x: Tensor = _QPLayerFn.apply(Q, c, A, b, options)  # type: ignore[no-untyped-call]
    return x


def lp_layer(
    c: Tensor, A: Tensor, b: Tensor, options: BarrierOptions | None = None
) -> Tensor:
    """Differentiable LP optimiser ``x*`` of ``min c^T x s.t. A x <= b`` (``Q = 0``)."""
    n = c.shape[0]
    Q = torch.zeros((n, n), dtype=torch.float64)
    x: Tensor = _QPLayerFn.apply(Q, c, A, b, options)  # type: ignore[no-untyped-call]
    return x


__all__ = ["lp_layer", "qp_layer"]

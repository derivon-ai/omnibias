# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form Newton leaves for a frozen soft-tree membership (theory 05-04).

Leaves enter the ensemble score **linearly** through the partition-of-unity
memberships ``P``: ``F = einsum("nml,mlk->nk", P, leaves) + b0``. Given a Newton
residual ``r = -g/h`` and Hessian weights ``h``, the unique minimizer of the
weighted ridge

    sum_n h_n (r_n - P_n . ell)^2 + leaf_l2 ||ell||^2

is an ``L x L`` solve per tree (``L = 2**depth``). This is the GBM leaf step
LightGBM / CatBoost take exactly; :func:`omnibias.tab.torch.boosting._fit_weak_learner`
previously approximated it with Adam-MSE.

Terminology: memberships come from ``sigmoid(beta (W.x - t))``. The ``beta -> inf``
hardening is temperature collapse (feasibility sense), distinct from the founding
``delta -> 0`` bias collapse. No founding collapse appears here.
"""

from __future__ import annotations

import numpy as np
from omnibias.tab._core.params import FloatArray

_RIDGE = 1e-12


def closed_form_leaves(
    P: FloatArray,
    residual: FloatArray,
    weight: FloatArray,
    leaf_l2: float = 1e-4,
) -> FloatArray:
    r"""Weighted-ridge Newton leaves ``(T, L, k)`` for frozen memberships ``P``.

    Parameters
    ----------
    P:
        Soft (or hard) leaf memberships ``(n, T, L)``. Rows per tree should be a
        partition of unity; the solver does not check this.
    residual:
        Newton target ``-g/h`` of shape ``(n, k)``.
    weight:
        Per-sample Hessian weights ``h`` of shape ``(n, k)`` (non-negative).
    leaf_l2:
        Ridge on ``||leaves||^2``. Must be ``>= 0``.
    """
    Pv = np.asarray(P, dtype=np.float64)
    rv = np.asarray(residual, dtype=np.float64)
    hv = np.asarray(weight, dtype=np.float64)
    if Pv.ndim != 3:
        raise ValueError(f"P must have shape (n, T, L), got {Pv.shape}")
    n, T, L = Pv.shape
    rv = rv.reshape(n, -1)
    hv = hv.reshape(n, -1)
    if rv.shape[0] != n or hv.shape[0] != n:
        raise ValueError("residual and weight must have n rows matching P")
    if rv.shape[1] != hv.shape[1]:
        raise ValueError("residual and weight must share an output width k")
    if leaf_l2 < 0.0:
        raise ValueError(f"leaf_l2 must be >= 0, got {leaf_l2}")
    k = int(rv.shape[1])
    ridge = float(leaf_l2) + _RIDGE
    out = np.zeros((T, L, k), dtype=np.float64)
    eye = np.eye(L, dtype=np.float64)
    for m in range(T):
        Pm = Pv[:, m, :]  # (n, L)
        for o in range(k):
            w = hv[:, o]  # (n,)
            Pw = Pm * w[:, None]  # (n, L)
            A = Pm.T @ Pw + ridge * eye
            b = Pw.T @ rv[:, o]
            out[m, :, o] = np.linalg.solve(A, b)
    return out


def newton_leaf_loss(
    P: FloatArray,
    residual: FloatArray,
    weight: FloatArray,
    leaves: FloatArray,
) -> float:
    r"""Mean weighted squared error of ``P @ leaves`` against the Newton residual."""
    Pv = np.asarray(P, dtype=np.float64)
    rv = np.asarray(residual, dtype=np.float64).reshape(Pv.shape[0], -1)
    hv = np.asarray(weight, dtype=np.float64).reshape(Pv.shape[0], -1)
    lv = np.asarray(leaves, dtype=np.float64)
    pred = np.einsum("nml,mlk->nk", Pv, lv)
    return float(np.mean(hv * (rv - pred) ** 2))


__all__ = ["closed_form_leaves", "newton_leaf_loss"]

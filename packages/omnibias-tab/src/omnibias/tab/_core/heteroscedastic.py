# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Closed-form heteroscedastic Gaussian NLL (theory 05-03).

Placed beside :mod:`omnibias.tab._core.loss` (not in ``omnibias.core``) because it is the
same shape of object as :func:`omnibias.tab._core.loss.score_grad_hess`: a per-sample loss
gradient consumed only by this package's own training / boosting machinery, with no other
consumer in the repo.

Model: ``y | x ~ N(u(x), s(x)**2)`` with ``s = exp(v)``, ``v = log s`` unconstrained, so the
per-sample negative log-likelihood in ``(u, v)`` is

.. math:: \mathrm{NLL}(u, v; y) = \tfrac12 \log(2\pi) + v + \tfrac12 (y-u)^2 e^{-2v}

Differentiating once gives the two closed-form gradients; differentiating twice and taking
the expectation over ``y ~ N(u, s^2)`` (using ``E[y-u] = 0``, ``E[(y-u)^2] = s^2``) gives the
**Fisher information** ``diag(1/s**2, 2)`` exactly -- positive-definite by inspection, unlike
the observed (per-sample) Hessian, whose ``du dv`` cross term changes sign with the residual.
No collapse limit (founding or feasibility) appears here; this is a plain likelihood
computation.
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.tab._core.params import FloatArray

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(u: FloatArray, v: FloatArray, y: FloatArray) -> FloatArray:
    r"""Per-sample heteroscedastic Gaussian NLL, ``0.5 log(2 pi) + v + 0.5 (y-u)^2 e^{-2v}``.

    ``u``, ``v`` and ``y`` broadcast together; no ``1/n`` reduction is applied (mirrors
    :func:`omnibias.tab._core.loss.score_grad_hess`'s per-sample convention).
    """
    uv = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    resid = yv - uv
    return _HALF_LOG_2PI + vv + 0.5 * resid * resid * np.exp(-2.0 * vv)


def gaussian_nll_grad_hess(
    u: FloatArray, v: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Closed-form ``(grad_u, grad_v, fisher)`` for the heteroscedastic Gaussian NLL.

    ``grad_u = -(y-u)/s**2``, ``grad_v = 1 - (y-u)**2/s**2`` (``s = e^v``), both broadcast to
    ``u``'s shape. ``fisher`` has shape ``u.shape + (2, 2)``: the **exact** Fisher block
    ``diag(1/s**2, 2)`` (population expectation over ``y ~ N(u, s**2)``), not the indefinite
    observed Hessian -- safe to use as a natural-gradient / Gauss-Newton metric where the
    observed Hessian is not positive-definite.
    """
    uv = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    resid = yv - uv
    inv_s2 = np.exp(-2.0 * vv)
    grad_u = -resid * inv_s2
    grad_v = 1.0 - resid * resid * inv_s2
    shape = np.broadcast_shapes(uv.shape, vv.shape, yv.shape)
    fisher = np.zeros((*shape, 2, 2), dtype=np.float64)
    fisher[..., 0, 0] = np.broadcast_to(inv_s2, shape)
    fisher[..., 1, 1] = 2.0
    return grad_u, grad_v, fisher


def log_scale_from_variance(variance: FloatArray) -> FloatArray:
    r"""``v = log(s) = 0.5 * log(variance)``, the ``log_scale`` inputs this module expects.

    Matches the CatBoost ``RMSEWithUncertainty`` convention (:func:`omnibias.tab.bench`):
    ``predict`` returns raw ``(mean, w)`` with ``variance = exp(w)``, so
    ``log_scale = w / 2``, which is exactly ``log_scale_from_variance(exp(w))``.
    """
    var = np.asarray(variance, dtype=np.float64)
    return 0.5 * np.log(var)


__all__ = [
    "gaussian_nll",
    "gaussian_nll_grad_hess",
    "log_scale_from_variance",
]

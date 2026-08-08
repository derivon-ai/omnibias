# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NTK-rebalance helpers (torch).

In NTK theory, the convergence rate of each loss term scales with the
trace of its Neural-Tangent-Kernel block. To equalise the per-term
convergence rates we use weights :math:`w_i = \\bar t / t_i` where
:math:`t_i` is the trace of NTK block :math:`i` and :math:`\\bar t` is
the geometric mean.

We expose a cheap Hutchinson-style trace estimator
(:func:`estimate_ntk_trace`) using a single backward pass and a
combiner (:func:`ntk_balanced_loss`) so the user can mix the two as
they see fit. :func:`ntk_eigenspectrum` and :func:`spectral_bias_index`
expose the eigenvalue *decay* that is the theory-of-record instrument
for spectral bias.

Reference
---------
Wang, Yu, Perdikaris, *When and why PINNs fail to train: A neural
tangent kernel perspective*, JCP 2022.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def ntk_balanced_loss(
    losses: dict[str, Tensor],
    *,
    ntk_traces: dict[str, Tensor] | None = None,
    epsilon: float = 1e-12,
) -> tuple[Tensor, dict[str, float]]:
    """Combine multiple loss terms with NTK-balanced weights.

    Parameters
    ----------
    losses
        ``{name: scalar_loss_tensor}`` mapping.
    ntk_traces
        ``{name: scalar_trace_tensor}`` mapping. If ``None``, all
        weights are 1.
    epsilon
        Floor for traces before taking ``log`` (avoids ``-inf``).

    Returns
    -------
    total
        Scalar tensor for ``loss.backward()``.
    weights
        Plain Python dict of the actual weights used (for logging).
    """
    if not losses:
        raise ValueError("ntk_balanced_loss: empty losses dict")
    if ntk_traces is None:
        weights = {k: 1.0 for k in losses}
    else:
        if set(ntk_traces) != set(losses):
            raise ValueError(
                f"ntk_traces keys {sorted(ntk_traces)!r} do not match "
                f"losses keys {sorted(losses)!r}"
            )
        log_t: dict[str, float] = {}
        for k, t in ntk_traces.items():
            t_val = float(t.detach().clamp(min=epsilon))
            log_t[k] = math.log(t_val)
        mean_log = sum(log_t.values()) / len(log_t)
        weights = {k: math.exp(mean_log - log_t[k]) for k in losses}

    total: Tensor | None = None
    for k, L in losses.items():
        term = float(weights[k]) * L
        total = term if total is None else total + term
    assert total is not None
    return total, {k: float(weights[k]) for k in losses}


def estimate_ntk_trace(
    loss: Tensor,
    parameters: list[torch.nn.Parameter],
) -> Tensor:
    """Cheap NTK trace estimator: ``sum_p (d loss / d p)^2``.

    For a per-sample squared loss this is exactly the trace of the
    NTK block; for batch-averaged losses it is a useful (and very
    cheap) proxy that scales with the same quantities -- enough to
    re-balance loss terms in practice.

    The autograd graph is *retained* so multiple traces can share
    backward passes.
    """
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False,
    )
    out = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for g in grads:
        if g is None:
            continue
        out = out + (g * g).sum()
    return out


def ntk_eigenspectrum(
    residual_fn: Callable[[], Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    n_eigen: int = 16,
) -> Tensor:
    """Leading empirical-NTK eigenvalues of a residual vector.

    Forms the Jacobian ``J = dr / dtheta`` (flattened) and returns the
    largest singular values of ``J`` squared -- i.e. the eigenvalues of
    ``J J^T`` / ``J^T J`` -- sorted descending. Intended for small
    residual batches / networks (CPU smoke, diagnostics); for large
    problems prefer a Lanczos / Hutchinson estimator.

    Honesty: this is a *measurement* of spectral bias, not a certificate.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("parameters must contain at least one requires_grad tensor")

    flat0 = torch.cat([p.detach().reshape(-1) for p in params])
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def _unflatten(flat: Tensor) -> list[Tensor]:
        out: list[Tensor] = []
        offset = 0
        for shape, sz in zip(shapes, sizes, strict=True):
            out.append(flat[offset : offset + sz].reshape(shape))
            offset += sz
        return out

    def f_flat(flat: Tensor) -> Tensor:
        chunks = _unflatten(flat)
        tokens = [p.data for p in params]
        try:
            for p, c in zip(params, chunks, strict=True):
                p.data = c
            return residual_fn().reshape(-1)
        finally:
            for p, t in zip(params, tokens, strict=True):
                p.data = t

    # Jacobian: (n_out, n_params)
    J = torch.autograd.functional.jacobian(f_flat, flat0, create_graph=False)
    # Singular values of J; eigenvalues of J J^T are s^2.
    s = torch.linalg.svdvals(J)
    evals = s * s
    k = min(int(n_eigen), int(evals.numel()))
    return torch.sort(evals, descending=True).values[:k]


def spectral_bias_index(eigenvalues: Tensor, *, n_head: int = 4) -> float:
    """Ratio of tail-mean to head-mean eigenvalue (smaller => stronger bias).

    ``eigenvalues`` should be sorted descending. Returns
    ``mean(tail) / mean(head)`` in ``[0, 1]`` for a typical decaying spectrum.
    """
    ev = eigenvalues.detach().reshape(-1)
    if ev.numel() < 2:
        return 1.0
    n_head = max(1, min(int(n_head), ev.numel() // 2))
    head = ev[:n_head].mean()
    tail = ev[n_head:].mean()
    if float(head) <= 0.0:
        return 0.0
    return float((tail / head).clamp(min=0.0, max=1.0))


__all__ = [
    "estimate_ntk_trace",
    "ntk_balanced_loss",
    "ntk_eigenspectrum",
    "spectral_bias_index",
]

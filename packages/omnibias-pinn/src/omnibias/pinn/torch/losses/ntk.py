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
import torch.nn as nn
from omnibias.torch.optim import lanczos_tridiag
from torch import Tensor


def ntk_balanced_loss(
    losses: dict[str, Tensor],
    *,
    ntk_traces: dict[str, Tensor] | None = None,
    epsilon: float = 1e-12,
) -> tuple[Tensor, dict[str, float]]:
    """Combine multiple loss terms with NTK-balanced weights."""
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
    """Cheap NTK trace estimator: ``sum_p (d loss / d p)^2``."""
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False,
    )
    out = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for g in grads:
        if g is None:
            continue
        out = out + (g * g).sum()
    return out


def _grad_params(
    output: Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    grad_outputs: Tensor,
) -> list[Tensor | None]:
    return list(
        torch.autograd.grad(
            output,
            parameters,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _flatten_grads(
    grads: Sequence[Tensor | None], parameters: Sequence[torch.nn.Parameter]
) -> Tensor:
    chunks: list[Tensor] = []
    for g, p in zip(grads, parameters, strict=True):
        if g is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(g.reshape(-1))
    if not chunks:
        raise ValueError("parameters must contain at least one requires_grad tensor")
    return torch.cat(chunks)


def _jjt_matvec(
    residual: Tensor,
    parameters: Sequence[torch.nn.Parameter],
    v: Tensor,
) -> Tensor:
    """Matrix-free multiply ``(J J^T) v`` for ``J = dr/dtheta``."""
    jt_v = _grad_params(residual, parameters, grad_outputs=v.reshape_as(residual))
    flat_jt_v = _flatten_grads(jt_v, parameters)
    out = torch.zeros_like(residual)
    for m in range(int(residual.numel())):
        e_m = torch.zeros_like(residual)
        e_m.reshape(-1)[m] = 1.0
        jt_e = _grad_params(residual, parameters, grad_outputs=e_m)
        out.reshape(-1)[m] = torch.dot(_flatten_grads(jt_e, parameters), flat_jt_v)
    return out


def empirical_jacobian(
    residual_fn: Callable[[], Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    module: nn.Module | None = None,
) -> Tensor:
    """Empirical Jacobian ``J = dr/dtheta`` with shape ``(n_out, n_params)``.

    Row ``m`` is ``grad r_m / dtheta`` via VJP; no ``Parameter.data`` mutation.
    The optional ``module`` argument is accepted for API compatibility.
    """
    del module
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("parameters must contain at least one requires_grad tensor")
    residual = residual_fn().reshape(-1)
    rows: list[Tensor] = []
    for m in range(int(residual.numel())):
        e_m = torch.zeros_like(residual)
        e_m.reshape(-1)[m] = 1.0
        jt_e = _grad_params(residual, params, grad_outputs=e_m)
        rows.append(_flatten_grads(jt_e, params))
    return torch.stack(rows, dim=0)


def ntk_eigenspectrum(
    residual_fn: Callable[[], Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    module: nn.Module | None = None,
    n_eigen: int = 16,
    lanczos_k: int | None = None,
) -> Tensor:
    """Leading empirical-NTK eigenvalues of a residual vector.

    Returns the largest eigenvalues of ``J J^T`` (squared singular values
    of ``J``), sorted descending. Small residual batches use a dense
    Jacobian; larger ones use matrix-free Lanczos via
    :func:`omnibias.torch.optim.lanczos_tridiag`.
    """
    del module
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("parameters must contain at least one requires_grad tensor")

    residual = residual_fn().reshape(-1)
    n_out = int(residual.numel())
    k = min(int(n_eigen), n_out)
    if k < 1:
        return torch.zeros(0, device=residual.device, dtype=residual.dtype)

    dense_limit = 64
    if n_out <= dense_limit:
        j = empirical_jacobian(residual_fn, params)
        s = torch.linalg.svdvals(j)
        evals = s * s
        return torch.sort(evals, descending=True).values[:k]

    krylov = lanczos_k if lanczos_k is not None else max(k + 4, 2 * k)
    krylov = min(int(krylov), n_out)

    def matvec(v: Tensor) -> Tensor:
        return _jjt_matvec(residual, params, v)

    b = residual.detach().reshape(-1)
    if float(torch.linalg.vector_norm(b)) == 0.0:
        b = torch.ones_like(b)
    _, tri = lanczos_tridiag(matvec, b, krylov)
    evals = torch.linalg.eigvalsh(tri)
    evals = torch.sort(evals, descending=True).values
    evals = torch.clamp(evals, min=0.0)
    return evals[:k]


def ntk_tail_head_index(eigenvalues: Tensor, *, n_head: int = 4) -> float:
    """Legacy tail/head eigenvalue ratio (smaller => stronger decay)."""
    ev = eigenvalues.detach().reshape(-1)
    if ev.numel() < 2:
        return 1.0
    n_head = max(1, min(int(n_head), ev.numel() // 2))
    head = ev[:n_head].mean()
    tail = ev[n_head:].mean()
    if float(head) <= 0.0:
        return 0.0
    return float((tail / head).clamp(min=0.0, max=1.0))


def fourier_mode_learning_rates(
    residual_fn: Callable[[], Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    coords: Tensor,
    modes: Sequence[int],
    L: float = 1.0,
    window_axis: int = 0,
) -> Tensor:
    """Per-Fourier-mode NTK learning-rate proxy on a uniform 1-D grid."""
    if not modes:
        raise ValueError("modes must be non-empty")
    residual = residual_fn().reshape(-1)
    x = coords[:, int(window_axis)].detach()
    n = int(x.numel())
    if n != int(residual.numel()):
        raise ValueError(
            f"coords rows {n} must match residual length {int(residual.numel())}"
        )
    rates: list[Tensor] = []
    two_pi = 2.0 * math.pi
    for k in modes:
        phi = torch.sin(two_pi * float(k) * x / float(L))
        phi = phi / (torch.linalg.vector_norm(phi) + 1e-12)
        jt_phi = _grad_params(residual, parameters, grad_outputs=phi)
        flat = _flatten_grads(jt_phi, parameters)
        rates.append((flat * flat).sum())
    return torch.stack(rates)


def kernel_task_alignment(
    mode_rates: Tensor,
    task_coeffs: Sequence[float],
) -> float:
    """Cosine alignment between task energy and per-mode kernel capacity."""
    rates = mode_rates.detach().reshape(-1).to(dtype=torch.float64)
    task = torch.tensor(list(task_coeffs), dtype=torch.float64, device=rates.device)
    if rates.numel() != task.numel():
        raise ValueError(
            f"mode_rates length {rates.numel()} != task_coeffs {task.numel()}"
        )
    num = float((rates * task).sum())
    den = float(torch.linalg.vector_norm(rates) * torch.linalg.vector_norm(task))
    if den <= 0.0:
        return 0.0
    return num / den


def spectral_bias_index(
    mode_rates: Tensor,
    *,
    n_low: int = 2,
    n_high: int = 2,
) -> float:
    """Low- versus high-frequency NTK response (smaller => stronger bias)."""
    rates = mode_rates.detach().reshape(-1)
    if rates.numel() < 2:
        return 1.0
    n_low = max(1, min(int(n_low), rates.numel() - 1))
    n_high = max(1, min(int(n_high), rates.numel() - n_low))
    low = rates[:n_low].mean()
    high = rates[-n_high:].mean()
    if float(high) <= 0.0:
        return 0.0
    return float((low / high).clamp(min=0.0, max=1.0))


__all__ = [
    "empirical_jacobian",
    "estimate_ntk_trace",
    "fourier_mode_learning_rates",
    "kernel_task_alignment",
    "ntk_balanced_loss",
    "ntk_eigenspectrum",
    "ntk_tail_head_index",
    "spectral_bias_index",
]

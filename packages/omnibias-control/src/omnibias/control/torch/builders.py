# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""CBF-row builders for control-affine and learned-Lagrangian dynamics (torch).

Bit-identical twin of :mod:`omnibias.control.jax.builders`; Lie derivatives use
``torch.func`` (grad / vmap) instead of ``jax``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from omnibias.control.problem import CBFSpec
from torch import Tensor
from torch.func import grad, vmap


def actuator_box(a_max: float, dim: int) -> tuple[Tensor, Tensor]:
    r"""The symmetric actuator box ``||a||_inf <= a_max`` as ``2*dim`` rows.

    Returns ``(G_box (2*dim, dim), h_box (2*dim,))`` (float64; cast to the state's
    dtype / device before concatenating).
    """
    eye = torch.eye(dim, dtype=torch.float64)
    g_box = torch.cat([eye, -eye], dim=0)
    h_box = torch.full((2 * dim,), float(a_max), dtype=torch.float64)
    return g_box, h_box


def control_affine_cbf_rows(
    f: Callable[[Tensor], Tensor],
    g: Callable[[Tensor], Tensor],
    barrier: Callable[[Tensor], Tensor],
    x: Tensor,
    spec: CBFSpec,
) -> tuple[Tensor, Tensor]:
    r"""Exponential-CBF rows for ``x_dot = f(x) + g(x) a`` via autodiff Lie derivatives.

    See the JAX twin :func:`omnibias.control.jax.builders.control_affine_cbf_rows` for
    the full contract; ``(G (B,m,d), h (B,m))``, differentiable in ``x``.
    """
    gains = spec.gains

    def per_sample(xi: Tensor) -> tuple[Tensor, Tensor]:
        if spec.relative_degree == 1:
            grad_h = grad(barrier)(xi)              # (n,)
            lf_h = grad_h @ f(xi)
            lg_h = grad_h @ g(xi)                   # (d,)
            row = -lg_h
            h_val = lf_h + gains[0] * barrier(xi)
        else:

            def lf_h_fn(z: Tensor) -> Tensor:
                return cast(Tensor, grad(barrier)(z) @ f(z))

            lf_h = lf_h_fn(xi)
            grad_lf_h = grad(lf_h_fn)(xi)           # (n,)
            lf2_h = grad_lf_h @ f(xi)
            lg_lf_h = grad_lf_h @ g(xi)             # (d,)
            row = -lg_lf_h
            g0, g1 = gains
            h_val = lf2_h + (g0 + g1) * lf_h + g0 * g1 * barrier(xi)
        return row, h_val

    rows, h_vals = vmap(per_sample)(x)              # (B, d), (B,)
    G = rows[:, None, :]                            # (B, 1, d)
    h = h_vals[:, None]                             # (B, 1)
    if spec.a_max is not None:
        d = rows.shape[1]
        g_box, h_box = actuator_box(spec.a_max, d)
        g_box = g_box.to(x)
        h_box = h_box.to(x)
        n_batch = x.shape[0]
        G = torch.cat([G, g_box.expand(n_batch, *g_box.shape)], dim=1)
        h = torch.cat([h, h_box.expand(n_batch, *h_box.shape)], dim=1)
    return G, h


def lagrangian_cbf_rows(
    lagrangian: object,
    B_input: Tensor,
    barrier: Callable[[Tensor], Tensor],
    q: Tensor,
    qdot: Tensor,
    t: Tensor,
    spec: CBFSpec,
) -> tuple[Tensor, Tensor]:
    r"""CBF rows for dynamics from a (possibly learned) Lagrangian.

    See the JAX twin :func:`omnibias.control.jax.builders.lagrangian_cbf_rows`.
    """
    from omnibias.variational.torch.ops.dynamics import acceleration, mass_matrix

    n = q.shape[-1]
    m = B_input.shape[-1]
    t0 = t[:1]

    def f(xi: Tensor) -> Tensor:
        qi, qdi = xi[:n], xi[n:]
        acc = acceleration(lagrangian, qi[None], qdi[None], t0)[0]
        return torch.cat([qdi, acc])

    def g(xi: Tensor) -> Tensor:
        qi, qdi = xi[:n], xi[n:]
        mass = mass_matrix(lagrangian, qi[None], qdi[None], t0)[0]
        minv_b = torch.linalg.solve(mass, B_input)
        zeros = torch.zeros((n, m), dtype=xi.dtype, device=xi.device)
        return torch.cat([zeros, minv_b], dim=0)

    x = torch.cat([q, qdot], dim=1)
    return control_affine_cbf_rows(f, g, barrier, x, spec)


__all__ = ["actuator_box", "control_affine_cbf_rows", "lagrangian_cbf_rows"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Second-order curvature *through* a soft-DP value (torch bridge to omnibias-curvature).

The soft Viterbi value ``V(E) = lse_beta over paths of score(E)`` is a log-sum-exp of
functions **linear** in the emissions ``E`` -- hence convex, with a positive-semidefinite
Hessian. These helpers expose that curvature by routing the unrolled soft-DP layer through
the generic exact machinery in :mod:`omnibias.curvature.torch` (``hvp`` /
``dense_hessian`` / ``hessian_top_eigenvalue``). The exact Hessian-vector product agrees to
machine precision with the closed-form directional curvature ``2 * dp_value_jet[2]`` from
:mod:`omnibias.struct.torch.jets`, giving two independent second-order routes.

Torch only: the JAX curvature surface is one-layer-Riccati, so the second-order bridge is
not mirrored there (the ``delta -> 0`` closed-form jet in :mod:`omnibias.struct.jax.jets`
is the JAX-side higher-order route).
"""

from __future__ import annotations

from omnibias.curvature.torch import dense_hessian, hessian_top_eigenvalue, hvp
from omnibias.struct.torch.soft_dp import soft_viterbi
from torch import Tensor


def chain_value_and_hvp(
    emissions: Tensor,
    transitions: Tensor,
    vector: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Soft Viterbi value and the exact Hessian-vector product ``H @ vector`` in ``E``.

    ``H = d^2 soft_viterbi / d emissions^2``; ``vector`` has the emission shape ``(T, S)``
    and the returned ``Hv`` matches it. Uses :func:`omnibias.curvature.torch.hvp` (a double
    backward), so it is exact, not finite-difference.
    """
    leaf = emissions.detach().clone().requires_grad_(True)
    loss = soft_viterbi(leaf, transitions, beta, start=start)
    hv = hvp(loss, [leaf], [vector])
    return loss.detach(), hv[0]


def chain_directional_curvature(
    emissions: Tensor,
    transitions: Tensor,
    direction: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Directional curvature ``direction^T H direction`` (equals ``2 * chain_lse_jet[2]``)."""
    _, hv = chain_value_and_hvp(emissions, transitions, direction, beta, start=start)
    return (hv * direction).sum()


def chain_hessian(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Dense emission Hessian of the soft Viterbi value, shape ``(T*S, T*S)`` (PSD)."""
    leaf = emissions.detach().clone().requires_grad_(True)
    loss = soft_viterbi(leaf, transitions, beta, start=start)
    h: Tensor = dense_hessian(loss, [leaf])
    return h


def chain_sharpness(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Top Hessian eigenvalue (sharpness) of the soft Viterbi loss landscape in ``E``.

    Non-negative because ``soft_viterbi`` is convex in the emissions; grows with ``beta``
    (the relaxation sharpens towards the hard optimum).
    """
    top: Tensor = hessian_top_eigenvalue(chain_hessian(emissions, transitions, beta, start=start))
    return top


__all__ = [
    "chain_directional_curvature",
    "chain_hessian",
    "chain_sharpness",
    "chain_value_and_hvp",
]

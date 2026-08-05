# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""SU(2) lattice Langevin / stochastic-quantisation updater (torch backend).

Parisi-Wu stochastic quantisation of the lattice gauge field: each link follows
overdamped Langevin dynamics on ``SU(2) = S^3`` whose stationary distribution is
the Wilson-action weight ``exp(-S)`` -- the same distribution the Kennedy-Pendleton
heat bath samples directly. This is the lattice companion of the continuum
DeTurck-gauged gradient-flow drift in :func:`omnibias.geometry.gauge.torch.ops.gauge_flow_rhs`
(the Chandra-Chevyrev-Hairer-Shen picture); gauge-invariant observables of this
chain agree with the heat-bath chain within statistics. Fixed-spacing numerical
evidence only (``unproven_claim=False``).
"""

from __future__ import annotations

import torch
from omnibias.geometry.gauge.lattice._core import kernels
from omnibias.geometry.gauge.lattice.su2 import checkerboard_mask, staple_sum


def langevin_update_links(
    links: torch.Tensor,
    mu: int,
    mask: torch.Tensor,
    beta: float,
    eps: float,
    *,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
) -> None:
    """In-place projected-sphere Langevin update of ``links[mu]`` where ``mask`` is True."""
    staple = staple_sum(links, mu)
    active = mask
    if not active.any():
        return
    u = links[mu][active]
    sigma = staple[active]
    if noise is None:
        xi = torch.randn((*u.shape[:-1], 3), device=links.device, dtype=links.dtype, generator=generator)
    else:
        xi = noise
    links[mu][active] = kernels.langevin_link_step(torch, u, sigma, beta, eps, xi)


def langevin_sweep(
    links: torch.Tensor,
    beta: float,
    eps: float = 0.05,
    *,
    generator: torch.Generator | None = None,
    n_sub: int = 1,
) -> None:
    """One Langevin sweep: ``n_sub`` checkerboard passes over all directions."""
    lattice_shape = links.shape[1:-1]
    device = links.device
    dtype = links.dtype
    for _ in range(n_sub):
        for mu in range(4):
            for parity in (0, 1):
                mask = checkerboard_mask(lattice_shape, parity, device=device, dtype=dtype)
                langevin_update_links(links, mu, mask, beta, eps, generator=generator)


__all__ = ["langevin_sweep", "langevin_update_links"]

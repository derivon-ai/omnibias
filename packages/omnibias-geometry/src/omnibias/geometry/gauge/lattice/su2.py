# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(2) lattice links as unit quaternions (torch backend).

The deterministic quaternion / staple math is delegated to the backend-agnostic
:mod:`omnibias.geometry.gauge.lattice._core.kernels` (shared bit-identically with the jax
backend); this module owns the torch-specific RNG (``torch.Generator``) and the
in-place checkerboard updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.geometry.gauge.lattice._core import kernels

if TYPE_CHECKING:
    from collections.abc import Sequence


def identity_links(
    lattice_shape: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return cold-start links: identity at every site, shape (4, *shape, 4)."""
    shape = tuple(int(s) for s in lattice_shape)
    links = torch.zeros((4, *shape, 4), device=device, dtype=dtype)
    links[..., 0] = 1.0
    return links


def random_links(
    lattice_shape: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly random SU(2) links, shape (4, *shape, 4)."""
    shape = tuple(int(s) for s in lattice_shape)
    q = torch.randn((4, *shape, 4), device=device, dtype=dtype, generator=generator)
    return normalize_quaternion(q)


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Project ``q`` onto the unit quaternion sphere (last dim = 4)."""
    return kernels.normalize_quaternion(torch, q)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate / inverse for unit links."""
    return kernels.quat_conj(torch, q)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Group product matching ``U(q_a q_b) = U(q_a) @ U(q_b)`` (left path order)."""
    return kernels.quat_mul(torch, a, b)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Map unit quaternion ``(q0,q1,q2,q3)`` to 2x2 complex SU(2) matrix."""
    return kernels.quat_to_matrix(torch, q.to(dtype=torch.float64))


def matrix_to_quat(u: torch.Tensor) -> torch.Tensor:
    """Project a 2x2 matrix onto the nearest unit quaternion (last two dims 2x2)."""
    return kernels.matrix_to_quat(torch, u.to(dtype=torch.complex128))


def _shift(x: torch.Tensor, mu: int, amount: int) -> torch.Tensor:
    """Periodic shift along lattice axis ``mu`` (0=x, 1=y, 2=z, 3=t)."""
    return kernels.shift(torch, x, mu, amount)


def staple_sum(links: torch.Tensor, mu: int) -> torch.Tensor:
    """Sum of forward + backward staples for direction ``mu`` (quaternion-valued)."""
    return kernels.staple_sum(torch, links, mu)


def staple_hat_and_magnitude(staple: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit direction ``U_hat`` and scalar magnitude ``a = ||staple||``."""
    return kernels.staple_hat_and_magnitude(torch, staple)


def _sample_q0_kp(
    a: torch.Tensor,
    beta: float,
    *,
    generator: torch.Generator | None,
    max_iter: int = 40,
) -> torch.Tensor:
    r"""Exact Kennedy-Pendleton ``q0`` sample, density ``~ sqrt(1-q0^2) exp(w q0)``.

    ``w = beta * a``. A candidate is drawn from the exponential part
    ``p_exp(q0) ~ exp(w q0)`` by inverse-CDF (uniform on ``[-1,1]`` in the
    ``w -> 0`` limit) and accepted with probability ``sqrt(1-q0^2)`` via the
    Kennedy-Pendleton trick ``u^2 <= 1 - q0^2``. The ``sqrt(1-q0^2)`` factor is
    the SU(2) Haar marginal, so this reproduces the true heat-bath conditional
    (validated against numerical quadrature in the tests). Sites unaccepted after
    ``max_iter`` rounds (probability ``~ (1 - <accept>)^max_iter``, negligible)
    keep their last candidate.
    """
    w = beta * a
    nonzero = w.abs() > 1e-10
    w_safe = torch.where(nonzero, w, torch.ones_like(w))
    q0 = torch.zeros_like(a)
    cand = torch.zeros_like(a)
    accepted = torch.zeros_like(a, dtype=torch.bool)
    for _ in range(max_iter):
        if bool(accepted.all()):
            break
        r = torch.rand(a.shape, device=a.device, dtype=a.dtype, generator=generator)
        cand_exp = 1.0 + torch.log(r + (1.0 - r) * torch.exp(-2.0 * w_safe)) / w_safe
        cand_unif = 2.0 * torch.rand(a.shape, device=a.device, dtype=a.dtype, generator=generator) - 1.0
        cand = torch.where(nonzero, cand_exp, cand_unif)
        u = torch.rand(a.shape, device=a.device, dtype=a.dtype, generator=generator)
        accept = (u * u <= (1.0 - cand * cand).clamp_min(0.0)) & (~accepted)
        q0 = torch.where(accept, cand, q0)
        accepted = accepted | accept
    return torch.where(accepted, q0, cand)


def _sample_unit_sphere(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    v = torch.randn((*shape, 3), device=device, dtype=dtype, generator=generator)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-30)


def heatbath_update_links(
    links: torch.Tensor,
    mu: int,
    mask: torch.Tensor,
    beta: float,
    *,
    generator: torch.Generator | None = None,
) -> None:
    """In-place Kennedy-Pendleton heat bath on ``links[mu]`` at sites where ``mask`` is True."""
    staple = staple_sum(links, mu)
    u_hat, a = staple_hat_and_magnitude(staple)
    active = mask
    if not active.any():
        return

    a_act = a[active]
    q0 = _sample_q0_kp(a_act, beta, generator=generator)
    r = _sample_unit_sphere(q0.shape, device=links.device, dtype=links.dtype, generator=generator)
    radial = torch.sqrt((1.0 - q0 * q0).clamp_min(0.0))
    v = torch.cat((q0.unsqueeze(-1), r * radial.unsqueeze(-1)), dim=-1)
    u_new = quat_mul(v, u_hat[active])

    mu_links = links[mu]
    mu_links[active] = u_new


def overrelax_update_links(
    links: torch.Tensor,
    mu: int,
    mask: torch.Tensor,
) -> None:
    """In-place over-relaxation: ``U' = U_hat * U^{-1} * U_hat``."""
    staple = staple_sum(links, mu)
    u_hat, _ = staple_hat_and_magnitude(staple)
    active = mask
    if not active.any():
        return
    u = links[mu][active]
    u_inv = quat_conj(u)
    u_new = quat_mul(quat_mul(u_hat[active], u_inv), u_hat[active])
    links[mu][active] = u_new


def checkerboard_mask(
    lattice_shape: Sequence[int],
    parity: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Boolean mask for even (parity=0) or odd (parity=1) sites."""
    shape = tuple(int(s) for s in lattice_shape)
    coords = torch.meshgrid(
        *(torch.arange(s, device=device, dtype=dtype) for s in shape),
        indexing="ij",
    )
    parity_sum = coords[0]
    for c in coords[1:]:
        parity_sum = parity_sum + c
    return (parity_sum.remainder(2) == parity).to(torch.bool)


def sweep(
    links: torch.Tensor,
    beta: float,
    *,
    generator: torch.Generator | None = None,
    n_overrelax: int = 2,
) -> None:
    """One MC sweep: heat bath (both colors, all directions) + over-relaxation hits."""
    lattice_shape = links.shape[1:-1]
    device = links.device
    dtype = links.dtype
    for mu in range(4):
        for parity in (0, 1):
            mask = checkerboard_mask(lattice_shape, parity, device=device, dtype=dtype)
            heatbath_update_links(links, mu, mask, beta, generator=generator)
    for _ in range(n_overrelax):
        for mu in range(4):
            for parity in (0, 1):
                mask = checkerboard_mask(lattice_shape, parity, device=device, dtype=dtype)
                overrelax_update_links(links, mu, mask)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(3) lattice kernels (numpy). Fixed-spacing evidence, not continuum QCD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_SU2_PAIRS = ((0, 1), (1, 2), (0, 2))


def identity_su3_links(lattice_shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(size) for size in lattice_shape)
    links = np.zeros((4, *shape, 3, 3), dtype=np.complex128)
    links[..., 0, 0] = 1.0
    links[..., 1, 1] = 1.0
    links[..., 2, 2] = 1.0
    return links


def reunitarize(u: np.ndarray) -> np.ndarray:
    """Project 3×3 matrices onto SU(3) by Gram–Schmidt plus det phase."""
    r0 = u[..., 0, :]
    r0 = r0 / np.maximum(np.linalg.norm(r0, axis=-1, keepdims=True), 1e-30)
    r1 = u[..., 1, :]
    r1 = r1 - np.sum(np.conjugate(r0) * r1, axis=-1, keepdims=True) * r0
    r1 = r1 / np.maximum(np.linalg.norm(r1, axis=-1, keepdims=True), 1e-30)
    r2 = np.conjugate(
        np.stack(
            (
                r0[..., 1] * r1[..., 2] - r0[..., 2] * r1[..., 1],
                r0[..., 2] * r1[..., 0] - r0[..., 0] * r1[..., 2],
                r0[..., 0] * r1[..., 1] - r0[..., 1] * r1[..., 0],
            ),
            axis=-1,
        )
    )
    stacked = np.stack((r0, r1, r2), axis=-2)
    det = np.linalg.det(stacked)
    phase = np.conjugate(det) / np.maximum(np.abs(det), 1e-30)
    stacked = stacked * phase[..., None, None]
    return stacked


def random_su3_links(
    lattice_shape: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    shape = tuple(int(size) for size in lattice_shape)
    raw = rng.normal(size=(4, *shape, 3, 3)) + 1j * rng.normal(size=(4, *shape, 3, 3))
    return reunitarize(raw)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...jk->...ik", a, b)


def _dag(u: np.ndarray) -> np.ndarray:
    return np.conjugate(np.swapaxes(u, -1, -2))


def su3_staple(links: np.ndarray, mu: int) -> np.ndarray:
    """Forward + backward staples for direction ``mu``."""
    staple = np.zeros(links.shape[1:], dtype=np.complex128)
    for nu in range(4):
        if nu == mu:
            continue
        u_nu = links[nu]
        u_mu_shift_nu = np.roll(links[mu], -1, axis=nu)
        u_nu_shift_mu = np.roll(links[nu], -1, axis=mu)
        fwd = _mul(_mul(u_nu, u_mu_shift_nu), _dag(u_nu_shift_mu))
        u_nu_bwd = np.roll(links[nu], 1, axis=nu)
        u_mu_bwd = np.roll(links[mu], 1, axis=nu)
        u_nu_bwd_shift = np.roll(u_nu_bwd, -1, axis=mu)
        bwd = _mul(_mul(_dag(u_nu_bwd), u_mu_bwd), u_nu_bwd_shift)
        staple = staple + fwd + bwd
    return staple


def su3_plaquette_trace(links: np.ndarray) -> np.ndarray:
    traces = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            u_mu = links[mu]
            u_nu_shift = np.roll(links[nu], -1, axis=mu)
            u_mu_shift = np.roll(links[mu], -1, axis=nu)
            u_nu = links[nu]
            loop = _mul(_mul(_mul(u_mu, u_nu_shift), _dag(u_mu_shift)), _dag(u_nu))
            traces.append(np.real(np.trace(loop, axis1=-2, axis2=-1)) / 3.0)
    return np.mean(np.stack(traces, axis=0), axis=0)


def average_su3_plaquette(links: np.ndarray) -> float:
    return float(np.mean(su3_plaquette_trace(links)))


def su3_polyakov_field(links: np.ndarray, *, t_dir: int = 3) -> np.ndarray:
    line = np.array(links[t_dir], copy=True)
    for _ in range(1, links.shape[1 + t_dir]):
        line = _mul(line, np.roll(links[t_dir], -_, axis=t_dir))
    return np.real(np.trace(line, axis1=-2, axis2=-1)) / 3.0


def average_su3_polyakov(links: np.ndarray, *, t_dir: int = 3) -> float:
    return float(np.mean(su3_polyakov_field(links, t_dir=t_dir)))


def su3_wilson_loop_trace(
    links: np.ndarray, mu: int, r_extent: int, t_extent: int, *, t_dir: int = 3
) -> np.ndarray:
    def _line(start: np.ndarray, direction: int, length: int) -> np.ndarray:
        acc = np.array(start, copy=True)
        cur = start
        for _ in range(1, length):
            cur = np.roll(cur, -1, axis=direction)
            acc = _mul(acc, cur)
        return acc

    u_mu = links[mu]
    u_t = links[t_dir]
    fwd_r = _line(u_mu, mu, r_extent)
    fwd_t = _line(np.roll(u_t, -r_extent, axis=mu), t_dir, t_extent)
    bwd_r = _dag(_line(np.roll(u_mu, -t_extent, axis=t_dir), mu, r_extent))
    bwd_t = _dag(_line(u_t, t_dir, t_extent))
    loop = _mul(_mul(_mul(fwd_r, fwd_t), bwd_r), bwd_t)
    return np.real(np.trace(loop, axis1=-2, axis2=-1)) / 3.0


def average_su3_wilson(
    links: np.ndarray, r_extent: int, t_extent: int, *, t_dir: int = 3
) -> float:
    traces = [
        su3_wilson_loop_trace(links, mu, r_extent, t_extent, t_dir=t_dir)
        for mu in range(3)
    ]
    return float(np.mean(np.stack(traces, axis=0)))


def _matrix_to_quat(u: np.ndarray) -> np.ndarray:
    """Nearest unit quaternion to a 2×2 block (batched)."""
    a = u[..., 0, 0]
    b = u[..., 0, 1]
    q0 = np.real(a)
    q3 = np.imag(a)
    q2 = np.real(b)
    q1 = np.imag(b)
    q = np.stack((q0, q1, q2, q3), axis=-1)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-30)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        (
            np.stack((q0 + 1j * q3, q2 + 1j * q1), axis=-1),
            np.stack((-q2 + 1j * q1, q0 - 1j * q3), axis=-1),
        ),
        axis=-2,
    )


def _sample_q0(a: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    w = beta * a
    q0 = np.zeros_like(a)
    accepted = np.zeros(a.shape, dtype=bool)
    cand = np.zeros_like(a)
    for _ in range(24):
        r = rng.random(a.shape)
        w_safe = np.where(np.abs(w) > 1e-10, w, 1.0)
        cand_exp = 1.0 + np.log(r + (1.0 - r) * np.exp(-2.0 * w_safe)) / w_safe
        cand_unif = 2.0 * rng.random(a.shape) - 1.0
        cand = np.where(np.abs(w) > 1e-10, cand_exp, cand_unif)
        u = rng.random(a.shape)
        accept = (u * u <= np.maximum(1.0 - cand * cand, 0.0)) & (~accepted)
        q0 = np.where(accept, cand, q0)
        accepted = accepted | accept
    return np.where(accepted, q0, cand)


def _su2_project(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a 2×2 block to (radius, SU(2) polar factor)."""
    a = block[..., 0, 0]
    b = block[..., 0, 1]
    c = block[..., 1, 0]
    d = block[..., 1, 1]
    q = np.stack(
        (
            np.real(a + d),
            np.imag(b + c),
            np.real(b - c),
            np.imag(a - d),
        ),
        axis=-1,
    )
    radius = np.linalg.norm(q, axis=-1)
    unit = q / np.maximum(radius[..., None], 1e-30)
    return radius, _quat_to_matrix(unit)


def _heatbath_su2(staple2: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """SU(2) factor ``A`` for a right-multiply ``U ← U A`` maximizing ``Re Tr(U† Σ)``.

    ``staple2`` is the 2×2 block of ``U† Σ``. Cooling would set ``A = h`` (polar
    factor); the heat-bath draws ``A = h g†`` with ``g`` Kennedy–Pendleton near
    the identity. ``beta`` is the SU(3) Wilson coupling.
    """
    radius, u_hat = _su2_project(staple2)
    q0 = _sample_q0(radius, float(beta) / 3.0, rng)
    sphere = rng.normal(size=(*radius.shape, 3))
    sphere = sphere / np.maximum(np.linalg.norm(sphere, axis=-1, keepdims=True), 1e-30)
    radial = np.sqrt(np.maximum(1.0 - q0 * q0, 0.0))
    v = np.concatenate((q0[..., None], sphere * radial[..., None]), axis=-1)
    return _mul(u_hat, _dag(_quat_to_matrix(v)))


def _extract_block(mat: np.ndarray, i: int, j: int) -> np.ndarray:
    return mat[..., np.ix_((i, j), (i, j))]


def _embed_su2(block: np.ndarray, i: int, j: int) -> np.ndarray:
    eye = np.zeros((*block.shape[:-2], 3, 3), dtype=np.complex128)
    eye[..., 0, 0] = 1.0
    eye[..., 1, 1] = 1.0
    eye[..., 2, 2] = 1.0
    eye[..., i, i] = block[..., 0, 0]
    eye[..., i, j] = block[..., 0, 1]
    eye[..., j, i] = block[..., 1, 0]
    eye[..., j, j] = block[..., 1, 1]
    return eye


def _parity_mask(lattice_shape: Sequence[int], parity: int) -> np.ndarray:
    grids = np.meshgrid(*(np.arange(size) for size in lattice_shape), indexing="ij")
    return (sum(grids) % 2) == int(parity)


def cabibbo_marinari_update(
    links: np.ndarray,
    mu: int,
    beta: float,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """One Cabibbo–Marinari heat-bath hit on direction ``mu``.

    ``mask`` selects a checkerboard subset. Same-direction staples depend on
    neighboring ``U_μ``, so a full-lattice hit does not satisfy detailed balance.
    """
    staple = su3_staple(links, mu)
    u = np.array(links[mu], copy=True)
    # Same staple convention as the working SU(2) kernel: the Wilson weight is
    # exp((β/3) Re Tr(U† Σ)), so the Cabibbo–Marinari matrix is W = U† Σ.
    w = _mul(_dag(u), staple)
    for i, j in _SU2_PAIRS:
        block = np.stack(
            (
                np.stack((w[..., i, i], w[..., i, j]), axis=-1),
                np.stack((w[..., j, i], w[..., j, j]), axis=-1),
            ),
            axis=-2,
        )
        heat = _heatbath_su2(block, beta, rng)
        embed = _embed_su2(heat, i, j)
        updated = _mul(u, embed)
        if mask is None:
            u = updated
        else:
            u = np.where(mask[..., None, None], updated, u)
        w = _mul(_dag(u), staple)
    out = np.array(links, copy=True)
    out[mu] = reunitarize(u)
    return out


def su3_sweep(
    links: np.ndarray, beta: float, rng: np.random.Generator, *, n_overrelax: int = 0
) -> np.ndarray:
    cur = links
    lattice_shape = tuple(int(size) for size in links.shape[1:5])
    for mu in range(4):
        for parity in (0, 1):
            cur = cabibbo_marinari_update(
                cur, mu, beta, rng, mask=_parity_mask(lattice_shape, parity)
            )
    for _ in range(n_overrelax):
        for mu in range(4):
            for parity in (0, 1):
                mask = _parity_mask(lattice_shape, parity)
                staple = su3_staple(cur, mu)
                polar = reunitarize(staple)
                updated = _mul(_mul(polar, _dag(cur[mu])), polar)
                nxt = np.array(cur, copy=True)
                nxt[mu] = reunitarize(np.where(mask[..., None, None], updated, cur[mu]))
                cur = nxt
    return cur


def su3_landau_overrelax(links: np.ndarray, *, n_steps: int = 8) -> np.ndarray:
    """Site-Jacobi Landau maximisation of ``Re Tr U_μ``."""
    cur = np.array(links, copy=True)
    for _ in range(int(n_steps)):
        acc = np.zeros((*cur.shape[1:5], 3, 3), dtype=np.complex128)
        for mu in range(4):
            acc = acc + cur[mu] + _dag(np.roll(cur[mu], 1, axis=mu))
        gauge = reunitarize(acc)
        nxt = np.zeros_like(cur)
        for mu in range(4):
            nxt[mu] = reunitarize(_mul(_mul(gauge, cur[mu]), _dag(np.roll(gauge, -1, axis=mu))))
        cur = nxt
    return cur


__all__ = [
    "average_su3_plaquette",
    "average_su3_polyakov",
    "average_su3_wilson",
    "cabibbo_marinari_update",
    "identity_su3_links",
    "random_su3_links",
    "reunitarize",
    "su3_landau_overrelax",
    "su3_plaquette_trace",
    "su3_polyakov_field",
    "su3_staple",
    "su3_sweep",
    "su3_wilson_loop_trace",
]

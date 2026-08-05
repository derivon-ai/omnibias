# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form symmetric Jastrow correlation factor for the FermiNet bridge.

A Jastrow factor multiplies the antisymmetric orbital part of a trial
wavefunction by a permutation-symmetric, strictly positive correlation
factor ``exp(J)``:

.. math::

   \psi(r) = e^{J(r)}\,\det M(r),
   \qquad
   \log|\psi(r)| = \log|\det M(r)| + J(r).

Because ``J`` enters ``log|psi|`` additively, its gradient and Laplacian
add to the determinant's, while the local-energy identity's ``|grad|^2``
term picks up a determinant-Jastrow **cross-term** (handled in
:func:`jastrow_slater_local_kinetic_energy`).

The correlation log-factor here is the classic isotropic-pair Pade-Jastrow

.. math::

   J(r) = \sum_{i<j} u_{ee}(r_{ij}) + \sum_{i,a} u_{en}(r_{ia}),
   \qquad
   u(r) = \frac{a\,r}{1 + b\,r},

where ``r_{ij} = \lVert r_i - r_j\rVert`` and ``r_{ia} = \lVert r_i - R_a\rVert``.
Every term is a function of an interparticle *distance*, so ``J`` is manifestly
invariant under electron permutations. The radial factor ``u`` and its two
derivatives are elementary rational functions -- the whole value / gradient /
Laplacian is **closed form**, no autodiff and no finite differences.

Kato cusp conditions
--------------------
The linear coefficient ``a`` is exactly the coalescence slope ``u'(0) = a``,
so the factor enforces the Kato cusp conditions by construction:

* electron-electron: ``u_{ee}'(0) = a_{ee}`` -- the physical value is ``1/2``
  for antiparallel spins and ``1/4`` for parallel spins (a single symmetric
  factor cannot distinguish spin; :func:`jastrow_init_params` defaults to the
  antiparallel ``1/2`` and the choice is documented, not hidden).
* electron-nucleus: ``u_{en}'(0) = a_{en,a}`` -- set to ``-Z_a`` to cancel the
  ``-Z_a / r`` nuclear singularity of the local energy.

Closed-form scope
-----------------
The value, gradient and Laplacian of ``J`` are closed form / exact. Choosing
the *variational* Jastrow parameters ``(a, b)`` that minimise the energy is an
outer optimisation loop (VMC / gradient descent) and is **not** in scope here
-- see ``docs/scope-and-guarantees.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.ferminet.restricted import (
    Tier2Params,
    tier2_grad_laplacian_log_psi,
)

_EPS2 = 1e-30


class JastrowParams(NamedTuple):
    r"""Parameters of the isotropic Pade-Jastrow correlation factor.

    Shapes
    ------
    atoms
        ``(n_atoms, 3)``   -- nuclear positions in Bohr.
    a_ee
        scalar             -- electron-electron cusp slope ``u_{ee}'(0)``.
    b_ee
        scalar             -- electron-electron Pade denominator rate.
    a_en
        ``(n_atoms,)``     -- electron-nucleus cusp slopes ``u_{en}'(0)``
                              (set to ``-Z_a`` for the physical nuclear cusp).
    b_en
        ``(n_atoms,)``     -- electron-nucleus Pade denominator rates.

    All fields are pytree leaves, so :class:`JastrowParams` plays naturally
    with :func:`jax.tree.map` and optimiser state.
    """

    atoms: Array
    a_ee: Array
    b_ee: Array
    a_en: Array
    b_en: Array


def jastrow_init_params(
    atoms: Array,
    charges: Array,
    *,
    a_ee: float = 0.5,
    b_ee: float = 1.0,
    b_en: float = 1.0,
) -> JastrowParams:
    r"""Cusp-satisfying default parameters.

    Sets the electron-nucleus slopes to ``a_{en,a} = -Z_a`` (physical nuclear
    cusp) and the electron-electron slope to ``a_{ee} = 1/2`` (antiparallel-spin
    Kato value) by default. The Pade denominator rates ``b`` control the range
    over which the correlation saturates (``u -> a/b`` as ``r -> inf``) and are
    the natural variational knobs.
    """
    atoms = jnp.asarray(atoms)
    charges = jnp.asarray(charges)
    n_atoms = int(atoms.shape[0])
    dtype = atoms.dtype
    return JastrowParams(
        atoms=atoms,
        a_ee=jnp.asarray(a_ee, dtype=dtype),
        b_ee=jnp.asarray(b_ee, dtype=dtype),
        a_en=-charges.astype(dtype),
        b_en=jnp.full((n_atoms,), b_en, dtype=dtype),
    )


def _pade_u_and_derivs(
    r: Array, a: Array, b: Array
) -> tuple[Array, Array, Array]:
    r"""``u = a r / (1 + b r)`` with its first two radial derivatives.

    Returns ``(u, u', u'')`` where ``u'(0) = a`` is the coalescence cusp slope.
    """
    denom = 1.0 + b * r
    u = a * r / denom
    u_p = a / (denom * denom)
    u_pp = -2.0 * a * b / (denom * denom * denom)
    return u, u_p, u_pp


def jastrow_value_grad_laplacian(
    params: JastrowParams,
    r_flat: Array,
    n_e: int,
) -> tuple[Array, Array, Array]:
    r"""Closed-form ``(J, grad_r J, nabla_r^2 J)`` of the Jastrow log-factor.

    Returns
    -------
    J    : scalar         -- the log-correlation value ``J(r)``.
    grad : ``(n_e, 3)``   -- ``grad_{r_k} J`` per electron.
    lap  : scalar         -- the total Laplacian ``sum_k nabla_{r_k}^2 J``.

    For an isotropic pair term ``u(r)`` with ``r = ||x||`` the 3-D gradient is
    ``u'(r) x/r`` and the radial Laplacian is ``u''(r) + (2/r) u'(r)``; each
    electron-electron pair contributes that radial Laplacian *twice* (once per
    electron), each electron-nucleus term once.
    """
    r = jnp.asarray(r_flat).reshape((n_e, 3))
    atoms = params.atoms

    # ---- electron-electron block -----------------------------------------
    diff_ee = r[:, None, :] - r[None, :, :]  # (n_e, n_e, 3)
    dist_ee = jnp.sqrt(jnp.sum(diff_ee * diff_ee, axis=-1) + _EPS2)  # (n_e, n_e)
    u_ee, up_ee, upp_ee = _pade_u_and_derivs(dist_ee, params.a_ee, params.b_ee)
    offdiag = 1.0 - jnp.eye(n_e, dtype=r.dtype)  # zero the i == j self pair

    # value: sum_{i<j} u = 1/2 sum_{i != j} u
    j_ee = 0.5 * jnp.sum(offdiag * u_ee)
    # gradient wrt r_k: sum_{j != k} u'(r_kj) * (r_k - r_j) / r_kj
    ghat_ee = diff_ee / dist_ee[..., None]  # unit vectors (diag is the 0 vector)
    grad_ee = jnp.sum((offdiag * up_ee)[..., None] * ghat_ee, axis=1)  # (n_e, 3)
    # laplacian: sum_{i<j} 2 (u'' + 2/r u') = sum_{i != j} (u'' + 2/r u')
    radlap_ee = upp_ee + 2.0 * up_ee / dist_ee
    lap_ee = jnp.sum(offdiag * radlap_ee)

    # ---- electron-nucleus block ------------------------------------------
    diff_en = r[:, None, :] - atoms[None, :, :]  # (n_e, n_atoms, 3)
    dist_en = jnp.sqrt(jnp.sum(diff_en * diff_en, axis=-1) + _EPS2)  # (n_e, n_atoms)
    u_en, up_en, upp_en = _pade_u_and_derivs(
        dist_en, params.a_en[None, :], params.b_en[None, :]
    )
    j_en = jnp.sum(u_en)
    ghat_en = diff_en / dist_en[..., None]
    grad_en = jnp.sum(up_en[..., None] * ghat_en, axis=1)  # (n_e, 3)
    radlap_en = upp_en + 2.0 * up_en / dist_en
    lap_en = jnp.sum(radlap_en)

    j_log = j_ee + j_en
    grad = grad_ee + grad_en  # (n_e, 3)
    lap = lap_ee + lap_en
    return j_log, grad, lap


def jastrow_log_value(
    params: JastrowParams,
    r_flat: Array,
    n_e: int,
) -> Array:
    r"""The Jastrow log-correlation value ``J(r)`` (autograd-friendly reference)."""
    r = jnp.asarray(r_flat).reshape((n_e, 3))
    atoms = params.atoms

    diff_ee = r[:, None, :] - r[None, :, :]
    dist_ee = jnp.sqrt(jnp.sum(diff_ee * diff_ee, axis=-1) + _EPS2)
    u_ee = params.a_ee * dist_ee / (1.0 + params.b_ee * dist_ee)
    iu, ju = jnp.triu_indices(n_e, k=1)
    j_ee = jnp.sum(u_ee[iu, ju])

    diff_en = r[:, None, :] - atoms[None, :, :]
    dist_en = jnp.sqrt(jnp.sum(diff_en * diff_en, axis=-1) + _EPS2)
    u_en = params.a_en[None, :] * dist_en / (1.0 + params.b_en[None, :] * dist_en)
    j_en = jnp.sum(u_en)

    out: Array = j_ee + j_en
    return out


def jastrow_slater_local_kinetic_energy(
    t2_params: Tier2Params,
    j_params: JastrowParams,
    r_flat: Array,
    n_e: int,
) -> Array:
    r"""Closed-form ``T_loc`` for the Jastrow-Slater ansatz ``e^{J} det M``.

    Uses the local-energy identity
    ``T_loc = -1/2 (nabla^2 log|psi| + ||nabla log|psi|||^2)`` with
    ``log|psi| = log|det M| + J``. Both the determinant gradient/Laplacian
    (:func:`omnibias.ferminet.tier2_grad_laplacian_log_psi`) and the Jastrow
    gradient/Laplacian (:func:`jastrow_value_grad_laplacian`) are closed form;
    the assembler adds the gradients (so the ``|grad|^2`` cross-term is exact)
    and adds the Laplacians. No autodiff is used at any point.
    """
    grad_det, lap_det = tier2_grad_laplacian_log_psi(t2_params, r_flat, n_e)
    _j, grad_jas, lap_jas = jastrow_value_grad_laplacian(j_params, r_flat, n_e)
    grad = grad_det + grad_jas  # (n_e, 3)
    lap = lap_det + lap_jas
    return -0.5 * (lap + jnp.sum(grad * grad))


def electron_electron_cusp(params: JastrowParams) -> Array:
    r"""The realised e-e coalescence slope ``lim_{r->0} d u_{ee}/dr = a_{ee}``."""
    out: Array = jnp.asarray(params.a_ee)
    return out


def electron_nucleus_cusp(params: JastrowParams) -> Array:
    r"""The realised per-atom e-n coalescence slopes ``u_{en}'(0) = a_{en}``."""
    out: Array = jnp.asarray(params.a_en)
    return out


def jastrow_log_value_np(
    params: JastrowParams, r_flat: Array, n_e: int
) -> float:
    """Pure-numpy reference for ``J(r)`` (independent of the jax code path)."""
    r = np.asarray(r_flat, dtype=np.float64).reshape((n_e, 3))
    atoms = np.asarray(params.atoms, dtype=np.float64)
    a_ee = float(params.a_ee)
    b_ee = float(params.b_ee)
    a_en = np.asarray(params.a_en, dtype=np.float64)
    b_en = np.asarray(params.b_en, dtype=np.float64)

    total = 0.0
    for i in range(n_e):
        for j in range(i + 1, n_e):
            rij = float(np.linalg.norm(r[i] - r[j]))
            total += a_ee * rij / (1.0 + b_ee * rij)
    for i in range(n_e):
        for a in range(atoms.shape[0]):
            ria = float(np.linalg.norm(r[i] - atoms[a]))
            total += a_en[a] * ria / (1.0 + b_en[a] * ria)
    return total


__all__ = [
    "JastrowParams",
    "electron_electron_cusp",
    "electron_nucleus_cusp",
    "jastrow_init_params",
    "jastrow_log_value",
    "jastrow_log_value_np",
    "jastrow_slater_local_kinetic_energy",
    "jastrow_value_grad_laplacian",
]

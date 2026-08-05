# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Molecular (Born-Oppenheimer) electronic-structure local energy (jax).

The electronic Hamiltonian in atomic units for ``n_e`` electrons in the field of
fixed nuclei is

.. math::

    \hat H = -\tfrac12 \sum_j \nabla_{r_j}^2
             - \sum_{j,a} \frac{Z_a}{\lVert r_j - R_a\rVert}
             + \sum_{j<k} \frac{1}{\lVert r_j - r_k\rVert}
             + \sum_{a<b} \frac{Z_a Z_b}{\lVert R_a - R_b\rVert}.

For a trial wavefunction ``psi`` the **local energy** is
``E_L = \hat H\psi / \psi``, which for the kinetic term is the standard
log-derivative identity

.. math::

    T_L = -\tfrac12\bigl(\nabla^2 \log|\psi| + \lVert\nabla\log|\psi|\rVert^2\bigr),
    \qquad E_L = T_L + V(R, r).

This module wires the **closed-form** ``omnibias`` Laplacian of ``log|psi|``
(the multivariate jet tower :func:`omnibias.jax.mlp_jet_mv`) to the bare Coulomb
potential :func:`coulomb_potential`. :func:`log_psi_derivatives` returns the exact
``(grad, lap) log|psi|`` for an MLP ``log|psi|``; :func:`local_energy` assembles
``E_L = T_L + V`` from those derivatives. The FermiNet restricted ansatz supplies
the same drift-form ``T_L`` in closed form through
:func:`omnibias.ferminet.tier2_local_kinetic_energy` -- a caller holding that
kinetic value simply adds ``V = coulomb_potential(...)`` to it.

Closed-form scope
-----------------
The kinetic term and the Coulomb potential are closed form / exact. The
*variational* solution of the Schrodinger equation is **not** in scope here:
VMC Monte-Carlo sampling, SCF/HF/CI/CC self-consistency, and Gaussian-basis
electron-repulsion integrals are iterative / stochastic numerics and are
labelled accordingly (see ``docs/scope-and-guarantees.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from omnibias.jax.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

Array = Any
LayerSpec = tuple[Any, Any, Any]

_COULOMB_EPS2 = 1e-30


def coulomb_potential(R: Array, r_flat: Array, charges: Array, n_e: int) -> Array:
    r"""Bare Coulomb potential energy ``V(R, r)`` in atomic units.

    ``R`` are nuclear positions ``(n_atoms*3,)``, ``r_flat`` electron positions
    ``(n_e*3,)``, ``charges`` the nuclear charges ``Z_a`` ``(n_atoms,)``. Includes
    the electron-nucleus, electron-electron, and nucleus-nucleus terms. This is
    the bit-identical twin of :func:`omnibias.jax.coulomb_potential`.
    """
    charges = jnp.asarray(charges)
    n_atoms = charges.shape[0]
    r_atoms = jnp.asarray(R).reshape((n_atoms, 3))
    r = jnp.asarray(r_flat).reshape((n_e, 3))

    d_en = jnp.sqrt(
        jnp.sum((r[:, None, :] - r_atoms[None, :, :]) ** 2, axis=-1) + _COULOMB_EPS2
    )
    v_en = -jnp.sum(charges[None, :] / d_en)

    if n_e > 1:
        d_ee = jnp.sqrt(
            jnp.sum((r[:, None, :] - r[None, :, :]) ** 2, axis=-1) + _COULOMB_EPS2
        )
        v_ee = jnp.sum(jnp.triu(1.0 / d_ee, k=1))
    else:
        v_ee = jnp.asarray(0.0, dtype=r.dtype)

    if n_atoms > 1:
        d_nn = jnp.sqrt(
            jnp.sum((r_atoms[:, None, :] - r_atoms[None, :, :]) ** 2, axis=-1)
            + _COULOMB_EPS2
        )
        zz = charges[:, None] * charges[None, :]
        v_nn = jnp.sum(jnp.triu(zz / d_nn, k=1))
    else:
        v_nn = jnp.asarray(0.0, dtype=r_atoms.dtype)

    return v_en + v_ee + v_nn


def local_kinetic_energy(grad_log_psi: Array, lap_log_psi: Array) -> Array:
    r"""``T_L = -1/2 (nabla^2 log|psi| + ||nabla log|psi|||^2)`` (drift form)."""
    grad = jnp.asarray(grad_log_psi)
    return -0.5 * (jnp.asarray(lap_log_psi) + jnp.sum(grad * grad))


def local_energy(grad_log_psi: Array, lap_log_psi: Array, potential: Array) -> Array:
    r"""Local energy ``E_L = T_L + V`` for a supplied scalar potential value."""
    return local_kinetic_energy(grad_log_psi, lap_log_psi) + jnp.asarray(potential)


def log_psi_derivatives(
    x0: Array, layers: Sequence[LayerSpec], *, order: int = 2
) -> tuple[Array, Array]:
    r"""Closed-form ``(nabla log|psi|, nabla^2 log|psi|)`` for an MLP ``log|psi|``.

    ``layers`` is the :func:`omnibias.jax.mlp_jet_mv` layer stack whose scalar
    output is ``log|psi|(x)``; the gradient and Laplacian come from the exact
    multivariate jet tower (no autodiff, no finite differences).
    """
    x0 = jnp.asarray(x0)
    dim = int(x0.shape[-1])
    jet = mlp_jet_mv(x0, layers, order)
    grad = jet_gradient(jet, dim, order)  # (D, C)
    hess = jet_hessian(jet, dim, order)  # (D, D, C)
    lap = jnp.trace(hess, axis1=0, axis2=1)  # (C,)
    if grad.ndim == 2 and grad.shape[-1] == 1:
        grad = grad[:, 0]
        lap = lap[0]
    return grad, lap


def molecular_local_energy(
    grad_log_psi: Array,
    lap_log_psi: Array,
    R: Array,
    r_flat: Array,
    charges: Array,
    n_e: int,
) -> Array:
    r"""``E_L = T_L + V_Coulomb`` from the derivatives of ``log|psi|``."""
    return local_energy(
        grad_log_psi, lap_log_psi, coulomb_potential(R, r_flat, charges, n_e)
    )


@dataclass(frozen=True)
class MolecularHamiltonian:
    """A fixed-nuclei electronic Hamiltonian ``H = T + V_Coulomb``.

    Parameters
    ----------
    charges
        Nuclear charges ``Z_a`` ``(n_atoms,)``.
    n_e
        Number of electrons.
    """

    charges: Array
    n_e: int

    def potential(self, R: Array, r_flat: Array) -> Array:
        """Bare Coulomb potential ``V(R, r)``."""
        return coulomb_potential(R, r_flat, self.charges, self.n_e)

    def local_energy(
        self, grad_log_psi: Array, lap_log_psi: Array, R: Array, r_flat: Array
    ) -> Array:
        """Local energy from the derivatives of ``log|psi|``."""
        return molecular_local_energy(
            grad_log_psi, lap_log_psi, R, r_flat, self.charges, self.n_e
        )


__all__ = [
    "MolecularHamiltonian",
    "coulomb_potential",
    "local_energy",
    "local_kinetic_energy",
    "log_psi_derivatives",
    "molecular_local_energy",
]

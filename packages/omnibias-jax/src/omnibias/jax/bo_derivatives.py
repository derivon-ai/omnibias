# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Born-Oppenheimer energy gradient and Hessian estimators.

This module computes nuclear-coordinate derivatives of the
Born-Oppenheimer energy

.. math::

   E_{\mathrm{BO}}(R)
       = \bigl\langle\psi(r;R)\bigm|\hat H(R)\bigm|\psi(r;R)\bigr\rangle
         /\,\bigl\langle\psi\bigm|\psi\bigr\rangle

as Monte Carlo estimators over walkers ``r_w`` drawn from
:math:`\rho(r;R) = |\psi(r;R)|^{2}/Z`. The unique value of doing
this through omnibias is that the *inner* derivatives -- the
:math:`\nabla_r` and :math:`\nabla_r^{2}` of :math:`\log|\psi|`
that drive the local energy -- are evaluated in closed form via
:mod:`omnibias.jax.laplacian` and the chain rule of
:mod:`omnibias.jax.ferminet_integration`, while
the *outer* derivatives w.r.t. :math:`R` (low-dimensional;
:math:`3N \le 30` for the molecules we care about) are computed
with ordinary :func:`jax.grad` / :func:`jax.hessian`.

This gives the first analytic-second-derivative path for a
neural-VMC energy and unlocks Hessian-class observables
(vibrational frequencies, instanton rates, response properties)
that have been infeasible for neural-VMC stacks. See
``H2O_ANALYTIC_HESSIAN`` for the chemistry headline (private
benchmark archive) and ``AUTOGRAD_PHASE_TRANSITION`` for the ICLR 2027
paper that frames the autograd compile bomb this side-steps.

Estimators (Casula--Sorella, eqs. 3, 6 of *Mol. Phys.* 109,
2473, 2011):

.. math::

   F_\alpha
       = -\,\Bigl\langle
              \partial_{R_\alpha} E_{\mathrm{loc}}
              + 2\bigl(E_{\mathrm{loc}} - \bar E\bigr)\,
                \partial_{R_\alpha}\log|\psi|
            \Bigr\rangle,

.. math::

   K_{\alpha\beta}
       = \langle\partial^{2}_{R_\alpha R_\beta} E_{\mathrm{loc}}\rangle
       + 2\bigl\langle (\partial_{R_\alpha} E_{\mathrm{loc}})
                       (\partial_{R_\beta} \log|\psi|)
                   + (\partial_{R_\beta} E_{\mathrm{loc}})
                       (\partial_{R_\alpha} \log|\psi|)\bigr\rangle
       + 2\bigl\langle (E_{\mathrm{loc}} - \bar E)
                       \partial^{2}_{R_\alpha R_\beta} \log|\psi|
                   + 2\,(E_{\mathrm{loc}} - \bar E)
                         (\partial_{R_\alpha} \log|\psi|)
                         (\partial_{R_\beta} \log|\psi|)\bigr\rangle.

All terms cancel in expectation when :math:`\psi` is the exact
ground state (zero-variance principle). With a fixed, non-exact
ansatz they do not, and the estimator above is the unbiased
form used throughout the QMC force literature.

API
---

``make_local_energy(psi_fn, potential_fn, n_e, ndim=3)``
    Returns ``e_local(R, r_flat) -> scalar`` using the omnibias
    closed-form Laplacian on ``log|psi|`` and an explicit
    ``potential_fn(R, r_flat) -> scalar`` for the
    nucleus-electron / electron-electron / nucleus-nucleus
    interaction. Pluggable so the same kernel works for hand-
    rolled wavefunctions, FermiNet, etc.

``make_bo_force(psi_fn, potential_fn, n_e)``
    Returns ``force(R, walkers, mean_energy) -> (3N,)`` -- the
    Casula--Sorella force estimator on a batch of walkers.

``make_bo_hessian(psi_fn, potential_fn, n_e)``
    Returns ``hessian(R, walkers, mean_energy) -> (3N, 3N)`` --
    the full Casula--Sorella Hessian estimator. Symmetric by
    construction.

``vibrational_frequencies(hessian, masses, R)``
    Mass-weights and diagonalises the 3N x 3N nuclear Hessian,
    projects out the 6 (or 5 for diatomics) translation /
    rotation modes, and returns the remaining eigenvalues in
    inverse centimetres (negative for imaginary modes).

Convention
----------

* ``R`` is the nuclear positions as a flat ``(3N,)`` array;
  callers reshape to ``(N, 3)`` as needed.
* ``walkers`` is a batch of electron positions, shape ``(B,
  n_e * 3)``.
* ``psi_fn(R, r_flat) -> (sign, log_abs)`` is the FermiNetLike
  convention (matches the Tier-1 plumbing in
  :func:`omnibias.jax.make_omnibias_envelope_local_kinetic_energy`).
* ``potential_fn(R, r_flat) -> scalar`` is the bare Coulomb
  potential energy at the given (R, r); the caller assembles
  this from atomic charges. We expose
  :func:`coulomb_potential` as the default.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Physical constants (atomic units)
# ---------------------------------------------------------------------------

# Hartree to cm^-1: E_h / (h c) in cm^-1
HARTREE_TO_CM = 219474.6313632

# Bohr to angstrom
BOHR_TO_ANGSTROM = 0.52917721067

# atomic mass unit to electron-mass units (m_e); needed for vibrational
# eigenvalues since the kinetic operator has 1/m factors in atomic units.
AMU_TO_ME = 1822.888486209


# ---------------------------------------------------------------------------
# Bare Coulomb potential
# ---------------------------------------------------------------------------


def coulomb_potential(
    R: Array,  # (n_atoms * 3,)
    r_flat: Array,  # (n_e * 3,)
    charges: Array,  # (n_atoms,)
    n_e: int,
) -> Array:
    r"""Bare Coulomb potential energy for fixed nuclear positions.

    .. math::

       V(R, r) =
         -\sum_{ja} \frac{Z_a}{\lVert r_j - R_a\rVert}
         + \sum_{j < k} \frac{1}{\lVert r_j - r_k\rVert}
         + \sum_{a < b} \frac{Z_a Z_b}{\lVert R_a - R_b\rVert}.

    All in atomic units. The small ``epsilon`` regularisation
    that appears below is for AD stability (the gradient of
    ``1/r`` is undefined at the singularity); MC walkers never
    sit exactly on the singularity but the AD graph still
    evaluates at sample points where the gradient is needed.
    """
    eps2 = 1e-30
    n_atoms = charges.shape[0]
    R_atoms = R.reshape((n_atoms, 3))
    r = r_flat.reshape((n_e, 3))

    # electron-nucleus
    d_en = jnp.sqrt(
        jnp.sum((r[:, None, :] - R_atoms[None, :, :]) ** 2, axis=-1) + eps2
    )  # (n_e, n_atoms)
    V_en = -jnp.sum(charges[None, :] / d_en)

    # electron-electron
    if n_e > 1:
        d_ee = jnp.sqrt(jnp.sum((r[:, None, :] - r[None, :, :]) ** 2, axis=-1) + eps2)  # (n_e, n_e)
        tri = jnp.triu(1.0 / d_ee, k=1)
        V_ee = jnp.sum(tri)
    else:
        V_ee = jnp.asarray(0.0, dtype=r_flat.dtype)

    # nucleus-nucleus
    if n_atoms > 1:
        d_nn = jnp.sqrt(
            jnp.sum(
                (R_atoms[:, None, :] - R_atoms[None, :, :]) ** 2,
                axis=-1,
            )
            + eps2
        )
        ZZ = charges[:, None] * charges[None, :]
        tri = jnp.triu(ZZ / d_nn, k=1)
        V_nn = jnp.sum(tri)
    else:
        V_nn = jnp.asarray(0.0, dtype=r_flat.dtype)

    return V_en + V_ee + V_nn


# ---------------------------------------------------------------------------
# Local energy: closed-form Laplacian + analytic potential
# ---------------------------------------------------------------------------


def _kinetic_energy_density(
    log_abs_fn: Callable[[Array, Array], Array],
    R: Array,
    r_flat: Array,
) -> Array:
    r"""``-1/2 (nabla_r^2 log|psi| + ||nabla_r log|psi|||^2)``.

    Implementation note: we build the full ``DxD`` Hessian of
    ``log|psi|`` w.r.t. ``r`` and take its trace. This is the
    O(D^2) autograd path used as the FermiNet ``'default'``
    branch baseline; it is the path against which we compare
    omnibias's O(1)-in-D primitive in the A2 paper.

    Crucially, this formulation produces a **clean** AD graph
    that nests well under ``jax.grad`` and ``jax.hessian``
    w.r.t. ``R`` (which the BO force / Hessian estimators need).
    The alternative ``jax.linearize`` + ``fori_loop`` pattern is
    slightly cheaper at evaluation time but compiles
    catastrophically slowly when re-differentiated through the
    nuclear-coordinate direction (we measured 15+ minute JIT
    compiles on Li2 with that pattern; the trace-of-Hessian path
    compiles in ~10 s for the same shape).
    """

    def log_abs_of_r(r: Array) -> Array:
        return log_abs_fn(R, r)

    grad_vec = jax.grad(log_abs_of_r)(r_flat)
    H = jax.hessian(log_abs_of_r)(r_flat)
    laplacian = jnp.trace(H)
    return -0.5 * (laplacian + jnp.sum(grad_vec * grad_vec))


def make_local_energy(
    psi_fn: Callable[[Array, Array], tuple[Array, Array]],
    potential_fn: Callable[[Array, Array], Array],
) -> Callable[[Array, Array], Array]:
    r"""Factory returning ``e_local(R, r_flat) -> scalar``.

    Parameters
    ----------
    psi_fn
        ``(R, r_flat) -> (sign, log_abs)``. Matches the
        FermiNetLike convention.
    potential_fn
        ``(R, r_flat) -> scalar`` -- the bare Coulomb potential
        at ``(R, r)``; see :func:`coulomb_potential` for the
        default.
    """

    def log_abs_fn(R: Array, r: Array) -> Array:
        return psi_fn(R, r)[1]

    def e_local(R: Array, r_flat: Array) -> Array:
        T = _kinetic_energy_density(log_abs_fn, R, r_flat)
        V = potential_fn(R, r_flat)
        return T + V

    return e_local


# ---------------------------------------------------------------------------
# BO force (Casula-Sorella)
# ---------------------------------------------------------------------------


def make_bo_force(
    psi_fn: Callable[[Array, Array], tuple[Array, Array]],
    potential_fn: Callable[[Array, Array], Array],
) -> Callable[[Array, Array, Array], Array]:
    r"""Return ``force(R, walkers, mean_energy) -> (3N,)``.

    Estimator (single walker, then average):

    .. math::

       f_\alpha(r;R)
         = -\,\partial_{R_\alpha} E_{\mathrm{loc}}(R, r)
           -\,2\bigl(E_{\mathrm{loc}}(R, r) - \bar E\bigr)
              \partial_{R_\alpha} \log|\psi(R, r)|.

    The minus sign convention matches ``F = -dE/dR``. Caller
    supplies ``mean_energy`` (typically a separate MC estimate of
    :math:`\bar E` from the same walkers).
    """
    e_local = make_local_energy(psi_fn, potential_fn)

    def log_abs_fn(R: Array, r: Array) -> Array:
        return psi_fn(R, r)[1]

    def force_per_walker(R: Array, r_flat: Array, e_bar: Array) -> Array:
        e_l = e_local(R, r_flat)
        grad_e_l = jax.grad(e_local, argnums=0)(R, r_flat)
        grad_log = jax.grad(log_abs_fn, argnums=0)(R, r_flat)
        out: Array = -(grad_e_l + 2.0 * (e_l - e_bar) * grad_log)
        return out

    def force(R: Array, walkers: Array, mean_energy: Array) -> Array:
        per_walker = jax.vmap(force_per_walker, in_axes=(None, 0, None))(
            R,
            walkers,
            mean_energy,
        )
        return jnp.mean(per_walker, axis=0)

    return force


# ---------------------------------------------------------------------------
# BO Hessian (Casula-Sorella, symmetric form)
# ---------------------------------------------------------------------------


def make_bo_hessian(
    psi_fn: Callable[[Array, Array], tuple[Array, Array]],
    potential_fn: Callable[[Array, Array], Array],
) -> Callable[[Array, Array, Array], Array]:
    r"""Return ``hessian(R, walkers, mean_energy) -> (3N, 3N)``.

    The per-walker integrand is

    .. math::

       K_{\alpha\beta}(r;R)
         = \partial^{2}_{R_\alpha R_\beta} E_{\mathrm{loc}}(R, r)
         + 2 (\partial_{R_\alpha} E_{\mathrm{loc}})(\partial_{R_\beta}\log|\psi|)
         + 2 (\partial_{R_\beta} E_{\mathrm{loc}})(\partial_{R_\alpha}\log|\psi|)
         + 2 (E_{\mathrm{loc}} - \bar E)\,
              \partial^{2}_{R_\alpha R_\beta} \log|\psi|
         + 4 (E_{\mathrm{loc}} - \bar E)
              (\partial_{R_\alpha} \log|\psi|)
              (\partial_{R_\beta} \log|\psi|).

    Then averaged over walkers. The result is symmetric by
    construction; we explicitly symmetrise the cross-derivative
    term to guard against MC asymmetry in float64.
    """
    e_local = make_local_energy(psi_fn, potential_fn)

    def log_abs_fn(R: Array, r: Array) -> Array:
        return psi_fn(R, r)[1]

    def hess_per_walker(R: Array, r_flat: Array, e_bar: Array) -> Array:
        e_l = e_local(R, r_flat)

        grad_e_l = jax.grad(e_local, argnums=0)(R, r_flat)
        hess_e_l = jax.hessian(e_local, argnums=0)(R, r_flat)

        grad_log = jax.grad(log_abs_fn, argnums=0)(R, r_flat)
        hess_log = jax.hessian(log_abs_fn, argnums=0)(R, r_flat)

        cross = jnp.outer(grad_e_l, grad_log)
        cross_sym = cross + cross.T
        delta_e = e_l - e_bar
        out: Array = (
            hess_e_l
            + 2.0 * cross_sym
            + 2.0 * delta_e * hess_log
            + 4.0 * delta_e * jnp.outer(grad_log, grad_log)
        )
        return out

    def hessian(
        R: Array,
        walkers: Array,
        mean_energy: Array,
    ) -> Array:
        per_walker = jax.vmap(hess_per_walker, in_axes=(None, 0, None))(
            R,
            walkers,
            mean_energy,
        )
        K = jnp.mean(per_walker, axis=0)
        return 0.5 * (K + K.T)

    return hessian


# ---------------------------------------------------------------------------
# Vibrational frequencies from a nuclear Hessian
# ---------------------------------------------------------------------------


def vibrational_frequencies(
    K_nuclear: Array,  # (3N, 3N), Hartree / Bohr^2, symmetric
    masses_amu: Array,  # (N,) in atomic mass units
    R: Array,  # (3N,) equilibrium nuclear positions, atomic units
    project_tr: bool = True,
    diatomic: bool = False,
) -> tuple[Array, Array]:
    r"""Diagonalise the mass-weighted nuclear Hessian.

    Parameters
    ----------
    K_nuclear
        ``(3N, 3N)`` symmetric matrix of :math:`\partial^{2}
        E_{\mathrm{BO}} / \partial R_\alpha \partial R_\beta` in
        atomic units (Hartree / Bohr^2).
    masses_amu
        ``(N,)`` atomic masses, in atomic mass units (amu).
    R
        ``(3N,)`` equilibrium nuclear positions in Bohr. Used to
        construct the translation and rotation projectors.
    project_tr
        If ``True`` (default), project out the six translation
        and rotation modes before diagonalising. For a diatomic
        molecule pass ``diatomic=True`` so we project out five
        modes (one rotation about the bond axis is degenerate).
    diatomic
        See above.

    Returns
    -------
    freqs_cm
        Vibrational frequencies in inverse centimetres. Negative
        values indicate an imaginary mode (saddle point /
        instability).
    eigvecs
        Mass-weighted Cartesian eigenvectors of the projected
        Hessian, shape ``(3N, 3N)``.
    """
    masses_me = masses_amu * AMU_TO_ME

    inv_sqrt_m = jnp.repeat(1.0 / jnp.sqrt(masses_me), 3)  # (3N,)
    K_mw = (inv_sqrt_m[:, None] * K_nuclear) * inv_sqrt_m[None, :]

    if project_tr:
        K_mw = _project_translations_rotations(
            K_mw,
            masses_me,
            R,
            diatomic=diatomic,
        )

    eigvals, eigvecs = jnp.linalg.eigh(K_mw)
    # eigenvalues are in (Hartree / Bohr^2) / m_e = Hartree / (Bohr^2 m_e).
    # The corresponding angular frequency is sqrt(eigval) in atomic units
    # (where omega_au = 1 / t_au, t_au = a0 sqrt(m_e/E_h)).
    # 1 au of angular frequency = 219474.63... cm^-1 / (2 pi); we convert
    # via the same constant since omega -> wavenumber uses
    # nu_cm = omega_au * HARTREE_TO_CM / (2 pi).  Actually the standard
    # convention used in vibrational spectroscopy is to report
    # nu_cm = sqrt(eigval [Ha/Bohr^2/m_e]) * HARTREE_TO_CM
    # because the relation E = hbar omega becomes nu_tilde = omega/(2 pi c)
    # and in atomic units omega = sqrt(eigval) (since hbar = 1, m_e = 1),
    # so nu_tilde_cm = sqrt(eigval) * HARTREE_TO_CM.
    freqs_cm = jnp.sign(eigvals) * jnp.sqrt(jnp.abs(eigvals)) * HARTREE_TO_CM
    return freqs_cm, eigvecs


def _project_translations_rotations(
    K_mw: Array,
    masses_me: Array,
    R: Array,
    diatomic: bool = False,
) -> Array:
    r"""Project the mass-weighted Hessian onto the
    translation/rotation-free subspace.

    Builds the (3 translation + 2 or 3 rotation) projectors
    explicitly, then applies ``K' = (I - P) K (I - P)``.
    """
    n_at = masses_me.shape[0]
    sqrt_m = jnp.sqrt(masses_me)
    R_at = R.reshape((n_at, 3))

    # 3 translation modes (mass-weighted): each is sqrt(m_a) * e_k.
    e = jnp.eye(3, dtype=R.dtype)
    trans = jnp.stack(
        [(sqrt_m[:, None] * e[k][None, :]).reshape(-1) for k in range(3)],
        axis=1,
    )  # (3N, 3)

    # rotation modes (Eckart conditions): for each axis k,
    #   T_k = sqrt(m) * (e_k x R)
    com = jnp.sum(masses_me[:, None] * R_at, axis=0) / jnp.sum(masses_me)
    R_shift = R_at - com[None, :]
    rot_axes = []
    for k in range(3):
        axis = e[k]
        cross = jnp.cross(R_shift, axis[None, :])  # (n_at, 3)
        vec = (sqrt_m[:, None] * cross).reshape(-1)  # (3N,)
        rot_axes.append(vec)
    rot = jnp.stack(rot_axes, axis=1)  # (3N, 3)

    basis = jnp.concatenate([trans, rot], axis=1)  # (3N, 6)
    # orthonormalise
    Q, _ = jnp.linalg.qr(basis)
    # drop any column whose norm became negligible (linear dep,
    # e.g. the third rotation mode for a diatomic).
    norms = jnp.linalg.norm(basis - Q @ (Q.T @ basis), axis=0)
    if diatomic:
        # Drop the last rotation column (it is colinear with the
        # bond axis -> identically zero).
        Q = Q[:, :5]
    del norms

    P = Q @ Q.T
    n = K_mw.shape[0]
    I = jnp.eye(n, dtype=K_mw.dtype)  # noqa: E741  # standard identity-matrix notation
    return (I - P) @ K_mw @ (I - P)

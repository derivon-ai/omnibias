# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Tier-2 ``omnibias`` integration: closed-form Laplacian through a
restricted-depth FermiNet-class wavefunction.

This is the Tier-2 deliverable from the FermiNet integration plan:
replace the autograd Laplacian on a real (not just envelope)
FermiNet-class graph with closed form. The plan distinguishes three
tiers:

* **Tier 1** (``ferminet_integration.py``) -- closed-form envelope
  derivatives, autograd interior. Autograd-parity in wall-clock by
  construction.
* **Tier 2** (this file) -- closed-form derivatives through the
  *entire* per-electron stack of a restricted FermiNet, including
  the ``tanh`` activation. The actual speedup story.
* **Tier 3** -- a folx-class JAX interpreter that recognises the
  full FermiNet equivariant block. Separate ~6-week workstream.

The Tier-2 wavefunction here is the
"one-layer single-electron restricted FermiNet" -- the smallest
ansatz that exercises closed-form derivatives through
:func:`jax.numpy.tanh` and a real Slater determinant. Per electron
``j`` we form

.. math::

   \mathrm{features}_j(r_j) &= \bigl[r_j - R_a\,;\,
                                     \lVert r_j - R_a\rVert\bigr]_{a=1\ldots A}
       \quad \text{(input feature vector)} \\
   h_j(r_j) &= \tanh\!\bigl(W_1\,\mathrm{features}_j + b_1\bigr)
       \quad \text{(equivariant layer)} \\
   \phi_i(r_j) &= \mathrm{env}_i(r_j)
              \cdot \bigl(W_\mathrm{orb}^{(i)\top} h_j(r_j)\bigr)
       \quad \text{(per-orbital readout)} \\
   M[i, j] &= \phi_i(r_j) \\
   \log|\psi(r)| &= \log\bigl|\det M\bigr|

Two variants are provided:

* :class:`Tier2Params` / :func:`tier2_local_kinetic_energy` --
  **Tier-2-lite**, no symmetric pooling. Every entry ``M[i, j]``
  depends only on ``r_j``; the off-diagonal electron-electron block
  of the position-space Hessian collapses (see formula below).
* :class:`Tier2SymParams` / :func:`tier2sym_local_kinetic_energy` --
  **Tier-2-full**, with symmetric pooling. Each per-electron tanh
  receives the additional input
  ``g(r) = (1/N) sum_l features_l(r_l)``, so every entry ``M[i, j]``
  now depends on *all* electron positions. The closed-form chain
  rule then has an extra electron-electron coupling term but no
  new ``omnibias`` primitives: the symmetric mean of features is
  itself a sum of per-electron features, so the
  closed-form derivative tower still applies (see derivation in
  the docstring of :func:`tier2sym_local_kinetic_energy`).

Lite variant: Laplacian decomposes electron-by-electron

.. math::

   \nabla_r^{2} \log\bigl|\det M\bigr|
       = \sum_{j=1}^{N}
           \Bigl[\,\mathrm{trace}\!\bigl(M^{-1}\,\partial_{r_j}^{2} M\bigr)
                 - \mathrm{trace}\!\bigl((M^{-1}\,\partial_{r_j} M)^{2}\bigr)
           \Bigr]

with ``\partial_{r_j} M`` non-zero only in column ``j``, so the
matrix products collapse to dot products. Every ingredient
(:math:`\phi_i(r_j), \partial_{r_j}\phi_i, \partial^{2}_{r_j}\phi_i`)
is shipped here in **closed form** via
:func:`omnibias.jax.neural_field_value_grad_hessian` on
the tanh equivariant layer and the explicit envelope derivatives
from :func:`omnibias.jax.envelope_value_grad_hessian`.

Full variant: symmetric pool through ``g(r) = mean_l features_l``
---------------------------------------------------------------

For the symmetric-pool variant, every column ``M[i, j]`` now
depends on all ``r_k``, and the position-space Hessian no longer
collapses to a column-of-``j``-only block. The matrix identity is
the same,

.. math::

   \nabla_r^{2} \log\bigl|\det M\bigr|
       = \mathrm{trace}\!\bigl(M^{-1}\,
                               \sum_{k, \mu} \partial^{2}_{r_{k\mu}} M\bigr)
         - \sum_{k, \mu}
             \mathrm{trace}\!\bigl((M^{-1}\,\partial_{r_{k\mu}} M)^{2}\bigr),

but :math:`\partial_{r_{k\mu}} M[i, j]` is now non-zero for every
:math:`(j, k)` pair. The closed form

.. math::

   \frac{\partial z_j[h]}{\partial r_{k, \mu}}
     = \delta_{kj}\,(W_{1,a}\,J_\mathrm{feat}[j])[h, \mu]
       + \tfrac{1}{N}\,(W_{1,b}\,J_\mathrm{feat}[k])[h, \mu]

and the analogous formula for the diagonal Hessian
:math:`\partial^{2} z_j/\partial r_{k, \mu}^{2}` decompose the full
``(n_orb, n_e, n_e, 3)`` Jacobian tensor of ``M`` into a self
(``k = j``) piece and a pool (``W_{1,b}``-only) piece. The
``omnibias`` primitives used for the lite tanh chain rule
extend unchanged: we never differentiate through ``tanh`` with
autograd. See :func:`tier2sym_local_kinetic_energy` for the full
derivation.

Why this is the speedup story
-----------------------------

In Tier 1 the FermiNet ``omnibias_envelope`` branch routed through
:func:`omnibias.jax.folx_compat.forward_laplacian`, which is
mathematically the autograd path -- so it is bit-stable to
FermiNet's ``default`` and ``folx`` but does not beat them in
wall-clock. The Tier-2 path here builds the value, gradient, and
Hessian of every per-electron orbital ``phi_i(r_j)`` from
closed-form omnibias primitives that are **O(1) in D** and **O(1)
in network width** for the Laplacian, and only the small
``N x N`` matrix identity remains. The benchmark numbers that
back the paper's compile-time / scaling claims are produced
off-band on the GPU cluster and summarised in ``docs/benchmarks.md``.

Public API
----------

Lite variant (no symmetric pooling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`Tier2Params` -- the lite-variant wavefunction parameters.
* :func:`tier2_log_abs_psi` -- ``log|psi(r)|`` (autograd path; used
  as the reference for parity tests).
* :func:`tier2_local_kinetic_energy` -- closed-form local kinetic
  energy ``T_loc = -1/2 (Laplacian + |grad|^2) of log|psi|``,
  built entirely from omnibias primitives. The Tier-2 production
  path.
* :func:`tier2_init_params` -- deterministic parameter
  initialization.

Full variant (symmetric pooling through ``g = mean_l features_l``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`Tier2SymParams` -- the full-variant wavefunction
  parameters; splits the lite ``W1`` into ``W1_a`` (acts on the
  per-electron features) and ``W1_b`` (acts on the symmetric mean
  ``g``).
* :func:`tier2sym_log_abs_psi` -- ``log|psi(r)|`` (autograd path).
* :func:`tier2sym_local_kinetic_energy` -- closed-form local
  kinetic energy with the symmetric-pool chain rule.
* :func:`tier2sym_init_params` -- deterministic parameter
  initialization.
* :class:`Tier2SymSpinBlockedParams`,
  :func:`tier2sym_spin_blocked_init_params`,
  :func:`tier2sym_blocked_log_abs_psi`,
  :func:`tier2sym_blocked_local_kinetic_energy` -- spin-blocked
  symmetric-pool variant. The pool is *per spin block* (so alpha
  electrons do not feed beta's interior); :math:`\psi = \det M_\alpha \cdot \det M_\beta`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.ferminet.integration import envelope_value_grad_hessian

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class Tier2Params(NamedTuple):
    r"""Parameters of the Tier-2 wavefunction.

    Shapes
    ------
    W1
        (H, D_in)             -- equivariant tanh layer weights.
    b1
        (H,)                  -- tanh bias.
    W_orb
        (n_orb, H)            -- orbital readout (one row per orbital).
    sigmas
        (n_orb, n_atoms)      -- envelope amplitudes per (orbital, atom).
    alphas
        (n_orb, n_atoms)      -- envelope decay rates per (orbital, atom).
    atoms
        (n_atoms, 3)          -- nuclear positions in Bohr.

    With ``D_in = 4 * n_atoms`` for the default feature builder
    (``(r_j - R_a)`` per atom plus the distance ``|r_j - R_a|``),
    and ``n_orb == n_e`` for a closed Slater determinant.
    """

    W1: Array
    b1: Array
    W_orb: Array
    sigmas: Array
    alphas: Array
    atoms: Array


def tier2_init_params(
    n_atoms: int,
    n_orb: int,
    hidden: int,
    *,
    seed: int = 0,
    atoms: Array | None = None,
) -> Tier2Params:
    rng = np.random.default_rng(seed)
    D_in = 4 * n_atoms
    if atoms is None:
        atoms = jnp.asarray(
            rng.normal(scale=1.0, size=(n_atoms, 3)),
            dtype=jnp.float64,
        )
    W1 = jnp.asarray(
        rng.normal(scale=1.0 / np.sqrt(D_in), size=(hidden, D_in)),
        dtype=jnp.float64,
    )
    b1 = jnp.asarray(
        rng.normal(scale=0.05, size=(hidden,)),
        dtype=jnp.float64,
    )
    W_orb = jnp.asarray(
        rng.normal(scale=0.5, size=(n_orb, hidden)),
        dtype=jnp.float64,
    )
    sigmas = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(n_orb, n_atoms)),
        dtype=jnp.float64,
    )
    alphas = jnp.asarray(
        rng.uniform(0.8, 1.2, size=(n_orb, n_atoms)),
        dtype=jnp.float64,
    )
    return Tier2Params(
        W1=W1,
        b1=b1,
        W_orb=W_orb,
        sigmas=sigmas,
        alphas=alphas,
        atoms=atoms,
    )


# ---------------------------------------------------------------------------
# Per-electron features
# ---------------------------------------------------------------------------


def _features(r_j: Array, atoms: Array) -> Array:
    r"""Build the per-electron feature vector.

    Returns shape ``(4 * n_atoms,)`` consisting of, for each
    atom ``a``,
    ``[r_j - R_a, |r_j - R_a|]`` flattened.

    This matches the simplest FermiNet-class input head: relative
    position vectors plus norms.
    """
    delta = r_j[None, :] - atoms  # (n_atoms, 3)
    dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-30)  # (n_atoms,)
    return jnp.concatenate(
        [delta, dist[:, None]],
        axis=-1,
    ).reshape(-1)  # (4 * n_atoms,)


def _features_jac_and_hessian(
    r_j: Array,
    atoms: Array,
) -> tuple[Array, Array, Array]:
    r"""Closed-form ``(features, dfeatures/dr_j, d^2 features/dr_j^2)``.

    ``features = [r_j - R_a, |r_j - R_a|]`` for ``a = 1..n_atoms``
    gives, after flattening,

    .. math::

       \partial_{(r_j)_\mu}\,(r_j - R_a)_\nu &= \delta_{\mu\nu} \\
       \partial_{(r_j)_\mu}\,\lVert r_j - R_a\rVert
           &= (r_j - R_a)_\mu /\,\lVert r_j - R_a\rVert \\
       \partial^{2}_{(r_j)_\mu (r_j)_\nu}\,(r_j - R_a)
           &= 0 \\
       \partial^{2}_{(r_j)_\mu (r_j)_\nu}\,
                 \lVert r_j - R_a\rVert
           &= (\delta_{\mu\nu} - \hat r_\mu \hat r_\nu)\,
              /\,\lVert r_j - R_a\rVert.

    Returns
    -------
    features : (4*n_atoms,)
    J        : (4*n_atoms, 3)      -- d features / d r_j
    H        : (4*n_atoms, 3, 3)   -- d^2 features / d r_j^2

    The "H" tensor is per-feature: ``H[k, mu, nu]`` is the (mu, nu)
    entry of the 3x3 Hessian of the k-th feature.
    """
    eps2 = 1e-30
    eye3 = jnp.eye(3, dtype=r_j.dtype)
    delta = r_j[None, :] - atoms  # (n_a, 3)
    dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + eps2)  # (n_a,)
    inv_d = 1.0 / dist  # (n_a,)
    rhat = delta * inv_d[:, None]  # (n_a, 3)

    # features = concat([delta (n_a, 3), dist (n_a, 1)], axis=-1).reshape(-1)
    # The reshape stride is (delta_x, delta_y, delta_z, dist) per atom.
    n_a = atoms.shape[0]

    # Pre-allocate J of shape (4 n_a, 3)
    # For each atom a in [0, n_a):
    #   J[4 a + 0 : 4 a + 3, :] = I3  (d (r - R_a)_nu / d r_j_mu = delta_{mu nu})
    #   J[4 a + 3, :]            = (r - R_a) / |r - R_a|  (= rhat_a)
    block_delta = jnp.broadcast_to(eye3, (n_a, 3, 3))  # (n_a, 3, 3)
    block_dist = rhat[:, None, :]  # (n_a, 1, 3)
    J = jnp.concatenate([block_delta, block_dist], axis=1)  # (n_a, 4, 3)
    J = J.reshape(4 * n_a, 3)

    # H: 4 n_a Hessians of shape (3, 3).
    # For the (r - R_a) entries: H = 0
    # For the dist entry: H = (I - rhat outer rhat) / dist
    H_delta = jnp.zeros((n_a, 3, 3, 3), dtype=r_j.dtype)  # 3 features x (3,3)
    # H_dist[a, mu, nu] = (I - rhat outer rhat)[mu, nu] / dist[a]
    P_a = eye3[None, :, :] - jnp.einsum(
        "am,an->amn",
        rhat,
        rhat,
    )  # (n_a, 3, 3)
    H_dist = (P_a * inv_d[:, None, None])[:, None, :, :]  # (n_a, 1, 3, 3)
    H_block = jnp.concatenate([H_delta, H_dist], axis=1)  # (n_a, 4, 3, 3)
    H = H_block.reshape(4 * n_a, 3, 3)

    # features:
    features = jnp.concatenate(
        [delta, dist[:, None]],
        axis=-1,
    ).reshape(-1)
    return features, J, H


# ---------------------------------------------------------------------------
# log|psi| -- reference path (autograd-friendly), and the orbital matrix
# ---------------------------------------------------------------------------


def _orbital_matrix(params: Tier2Params, r_flat: Array, n_e: int) -> Array:
    r"""Build the ``n_orb x n_e`` orbital matrix ``M[i, j] = phi_i(r_j)``.

    Each column ``j`` is a function of ``r_j`` only.
    """
    r = r_flat.reshape((n_e, 3))

    def column(r_j: Array) -> Array:
        features = _features(r_j, params.atoms)
        h = jnp.tanh(params.W1 @ features + params.b1)  # (H,)
        # env values per orbital at r_j
        delta = r_j[None, :] - params.atoms  # (n_atoms, 3)
        dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-30)
        env = jnp.sum(
            params.sigmas * jnp.exp(-params.alphas * dist[None, :]),
            axis=-1,
        )  # (n_orb,)
        orb = params.W_orb @ h  # (n_orb,)
        return env * orb  # (n_orb,)

    return jax.vmap(column)(r).T  # (n_orb, n_e)


def tier2_log_abs_psi(
    params: Tier2Params,
    r_flat: Array,
    n_e: int,
) -> Array:
    """Autograd-friendly ``log|det M|``. Reference for parity tests."""
    M = _orbital_matrix(params, r_flat, n_e)
    _, logdet = jnp.linalg.slogdet(M)
    out: Array = logdet
    return out


def tier2_psi_fn(
    params: Tier2Params, n_e: int
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """FermiNetLike adapter: returns ``(R_flat, r_flat) -> (sign, log_abs)``.

    The wavefunction's nuclear positions are stored in
    ``params.atoms``; for the BO derivative pipeline we expose them
    as an explicit ``R_flat`` argument by patching the params each
    call. This is the natural surface for
    :mod:`omnibias.jax.bo_derivatives`.
    """

    def psi_fn(R_flat: Array, r_flat: Array) -> tuple[Array, Array]:
        atoms = R_flat.reshape(params.atoms.shape)
        params_R = params._replace(atoms=atoms)
        M = _orbital_matrix(params_R, r_flat, n_e)
        sign, logdet = jnp.linalg.slogdet(M)
        return sign, logdet

    return psi_fn


# ---------------------------------------------------------------------------
# Closed-form column derivatives: (phi_i(r_j), d phi_i / d r_j, d^2 phi_i / d r_j^2)
# ---------------------------------------------------------------------------


def _column_value_jac_hess(
    params: Tier2Params,
    r_j: Array,
) -> tuple[Array, Array, Array]:
    r"""Closed-form ``(phi(r_j), d phi / d r_j, d^2 phi / d r_j^2)``.

    Shapes
    ------
    phi    : (n_orb,)
    J_phi  : (n_orb, 3)         -- gradient of each phi_i wrt r_j
    H_phi  : (n_orb, 3, 3)      -- Hessian of each phi_i wrt r_j

    Algorithm
    ---------
    phi_i(r_j) = env_i(r_j) * orb_i(r_j) where
        orb_i(r_j) = W_orb[i, :] @ tanh(W_1 features(r_j) + b_1).

    We obtain:

    * ``env_i, grad_env_i, H_env_i``   --
      from :func:`envelope_value_grad_hessian` (closed form).
    * ``orb_i, grad_orb_i, H_orb_i``  --
      by a chain rule:
      orb is a one-layer ``tanh`` field over the
      ``features(r_j)`` input space; the
      :func:`neural_field_value_grad_hessian` primitive gives us the
      gradient and Hessian *in feature space*; we then chain through
      the Jacobian / Hessian of the features map.

    The final product rule:
        d phi      = (d env) orb + env (d orb)
        d^2 phi    = (d^2 env) orb + 2 (d env) outer (d orb) + env (d^2 orb).
    """
    # ------ features ------
    features, J_feat, H_feat = _features_jac_and_hessian(r_j, params.atoms)
    # features: (D_in,), J_feat: (D_in, 3), H_feat: (D_in, 3, 3).

    # ------ orbital readout (one-layer tanh field) ------
    # For each orbital i, define a 1-layer multi-bias field over
    # features-space:
    #     orb_i(features) = W_orb[i, :] @ tanh(W_1 features + b_1)
    # The closed-form derivative tower in feature space is
    #     d orb_i / d features = sum_h W_orb[i, h] sigma'(z_h) W1[h, :]
    #     H_features orb_i     = sum_h W_orb[i, h] sigma''(z_h) W1[h, :] W1[h, :]^T
    # but materialising ``H_features`` is O(n_orb * D_in^2) which is
    # wasteful here -- we only need the 3x3 Hessian in r_j space
    # after the chain rule. Push the J_feat contraction inside so
    # the hidden state we carry has size ``(H, 3)`` rather than
    # ``(D_in, D_in)``.
    z = params.W1 @ features + params.b1  # (H,)
    h = jnp.tanh(z)  # (H,)
    h_prime = 1.0 - h * h  # (H,) = sigma'(z)
    h_pprime = -2.0 * h * h_prime  # (H,) = sigma''(z)
    orb_value = params.W_orb @ h  # (n_orb,)

    Wj = params.W1 @ J_feat  # (H, 3)
    # grad_orb[i, mu] = sum_h W_orb[i, h] * sigma'(z_h) * Wj[h, mu]
    grad_orb = (params.W_orb * h_prime[None, :]) @ Wj  # (n_orb, 3)

    # term_a[i, mu, nu]
    #   = sum_{k, l, h} W_orb[i, h] sigma''(z_h) W1[h, k] W1[h, l] J_feat[k, mu] J_feat[l, nu]
    #   = sum_h        W_orb[i, h] sigma''(z_h) Wj[h, mu] Wj[h, nu]
    weights = params.W_orb * h_pprime[None, :]  # (n_orb, H)
    term_a = jnp.einsum("ih,hm,hn->imn", weights, Wj, Wj)  # (n_orb, 3, 3)
    # term_b[i, mu, nu] = sum_k (d orb_i / d features_k) * H_feat[k, mu, nu]
    grad_orb_feat = (params.W_orb * h_prime[None, :]) @ params.W1  # (n_orb, D_in)
    term_b = jnp.einsum("ik,kmn->imn", grad_orb_feat, H_feat)
    H_orb = term_a + term_b  # (n_orb, 3, 3)

    # ------ envelope: closed form ------
    # envelope_value_grad_hessian takes r of shape (n_e, 3) and
    # returns (env (n_orb, n_e), grad (n_orb, n_e, 3), H (n_orb, n_e, 3, 3)).
    # We need just the j-th column, so we evaluate at a 1-electron config.
    r_j_batch = r_j[None, :]  # (1, 3)
    env, grad_env, H_env = envelope_value_grad_hessian(
        r_j_batch,
        params.atoms,
        params.sigmas,
        params.alphas,
    )
    env = env[:, 0]  # (n_orb,)
    grad_env = grad_env[:, 0, :]  # (n_orb, 3)
    H_env = H_env[:, 0, :, :]  # (n_orb, 3, 3)

    # ------ product rule for phi = env * orb ------
    phi = env * orb_value  # (n_orb,)
    grad_phi = grad_env * orb_value[:, None] + env[:, None] * grad_orb  # (n_orb, 3)

    # d^2 (env * orb) = d^2 env * orb
    #                 + (d env) outer (d orb) + (d orb) outer (d env)
    #                 + env * d^2 orb
    H_phi = (
        H_env * orb_value[:, None, None]
        + jnp.einsum("im,in->imn", grad_env, grad_orb)
        + jnp.einsum("im,in->imn", grad_orb, grad_env)
        + env[:, None, None] * H_orb
    )  # (n_orb, 3, 3)

    return phi, grad_phi, H_phi


# ---------------------------------------------------------------------------
# Closed-form local kinetic energy
# ---------------------------------------------------------------------------


def tier2_grad_laplacian_log_psi(
    params: Tier2Params,
    r_flat: Array,
    n_e: int,
) -> tuple[Array, Array]:
    r"""Closed-form ``(grad_r log|psi|, nabla_r^2 log|psi|)`` for the lite ansatz.

    Returns the per-electron gradient ``t`` of shape ``(n_e, 3)`` and the
    scalar Laplacian ``nabla_r^2 log|det M|`` from the matrix identity

    .. math::

       \nabla_{r_j}^{2}\,\log|\det M|
           = \mathrm{trace}\!\bigl(M^{-1}\,\partial^{2}_{r_j} M\bigr)
             - \mathrm{trace}\!\bigl((M^{-1}\,\partial_{r_j} M)^{2}\bigr).

    This is the split used by :func:`tier2_local_kinetic_energy` and by any
    caller that must combine the determinant with a separate log-factor
    (e.g. a Jastrow correlation term) whose ``|grad|^2`` cross-term does not
    cancel. Because the lite ansatz has no symmetric pooling, every matrix
    product collapses to a row-of-``M^{-1}`` dotted with a column-of-``dM``,
    so each electron costs ``O(n_orb)`` after one ``M^{-1}`` solve.
    """
    r = r_flat.reshape((n_e, 3))

    # Per-electron (value, grad, Hessian) of every orbital.
    column_fn = jax.vmap(_column_value_jac_hess, in_axes=(None, 0))
    phi_per_e, grad_per_e, H_per_e = column_fn(params, r)
    # phi_per_e:   (n_e, n_orb)
    # grad_per_e:  (n_e, n_orb, 3)
    # H_per_e:     (n_e, n_orb, 3, 3)

    # Orbital matrix M[i, j] = phi_per_e[j, i]
    M = phi_per_e.T  # (n_orb, n_e)

    # For each electron j, the partial derivative wrt r_j_mu has
    # only column j of M non-zero:
    #     (dM / d r_j_mu)[i, k] = delta_{kj} * grad_per_e[j, i, mu].
    # Define A = M^{-1} (dM / d r_j_mu). Then
    #     A[alpha, beta]
    #       = sum_gamma M^{-1}[alpha, gamma] * dM[gamma, beta]
    #       = delta_{j, beta} * sum_gamma M^{-1}[alpha, gamma]
    #                              * grad_per_e[beta, gamma, mu]
    #       = delta_{j, beta} * v[alpha],
    # i.e. only column ``j`` of A is non-zero. So
    #     trace(A)   = A[j, j] = v[j]
    #              = sum_gamma M^{-1}[j, gamma] * grad_per_e[j, gamma, mu],
    # which is ROW ``j`` of ``M^{-1}`` dotted with the gradient column.
    #
    # A^2 has the same column-j-only structure with
    #     A^2[alpha, j] = v[alpha] * v[j]
    # so trace(A^2) = v[j]^2.
    #
    # For the second derivative the same algebra gives
    #     trace(M^{-1} d^2 M / d r_{j, mu}^2)
    #       = sum_gamma M^{-1}[j, gamma] * H_per_e[j, gamma, mu, mu].
    # Form the inverse via solve(M, I) (the canonical stable inverse; jnp's
    # inv() routes through solve() anyway). The element-wise trace contractions
    # below need every entry of M^{-1}, so the full inverse is genuinely needed.
    Minv = jnp.linalg.solve(M, jnp.eye(M.shape[-1], dtype=M.dtype))  # (n_orb, n_e)

    # t[j, mu] = trace(M^{-1} d M / d r_{j, mu}) = grad log|psi|_{j, mu}
    #         = sum_gamma Minv[j, gamma] * grad_per_e[j, gamma, mu]
    t = jnp.einsum("jg,jgm->jm", Minv, grad_per_e)  # (n_e, 3)

    # Laplacian Term 1: sum_{j, mu} trace(M^{-1} d^2 M / d r_{j, mu}^2)
    #   = sum_{j, gamma} Minv[j, gamma] * (sum_mu H_per_e[j, gamma, mu, mu])
    trace_H = jnp.einsum("jgmm->jg", H_per_e)  # (n_e, n_orb)
    term_lap1 = jnp.einsum("jg,jg->", Minv, trace_H)  # scalar

    # Laplacian Term 2: -sum_{j, mu} t[j, mu]^2
    term_lap2 = -jnp.sum(t * t)  # scalar

    laplacian = term_lap1 + term_lap2  # scalar
    return t, laplacian


def tier2_local_kinetic_energy(
    params: Tier2Params,
    r_flat: Array,
    n_e: int,
) -> Array:
    r"""``T_loc = -1/2 (nabla_r^2 log|psi| + ||nabla_r log|psi|||^2)``.

    Built entirely from closed-form per-electron column derivatives
    (:func:`tier2_grad_laplacian_log_psi`) plus the local-energy identity
    ``psi^{-1} nabla^2 psi = nabla^2 log|psi| + ||nabla log|psi|||^2``.
    """
    t, laplacian = tier2_grad_laplacian_log_psi(params, r_flat, n_e)
    grad_log_psi_sq = jnp.sum(t * t)  # |grad log|psi||^2
    return -0.5 * (laplacian + grad_log_psi_sq)


class Tier2SpinBlockedParams(NamedTuple):
    r"""Spin-blocked Tier-2 wavefunction parameters.

    For a system with ``n_alpha`` spin-up and ``n_beta`` spin-down
    electrons we form

    .. math::

       \psi(r) = \det\!\bigl(M_\alpha(r_\alpha)\bigr)
                 \,\det\!\bigl(M_\beta (r_\beta )\bigr),

    each block being a Tier-2 orbital matrix on its electrons. The
    Slater determinant is square per spin block; for restricted
    closed-shell ansatzes the two blocks can share parameters but
    we keep them independent by default so the ansatz is genuinely
    UHF-style.

    Both fields are pytree leaves, so this object plays naturally
    with :func:`jax.tree.map` and Adam state. The electron counts
    are inferred from ``params.alpha.W_orb.shape[0]`` and
    ``params.beta.W_orb.shape[0]`` (both static compile-time
    integers under JIT).
    """

    alpha: Tier2Params
    beta: Tier2Params


def _spin_block_sizes(params: Tier2SpinBlockedParams) -> tuple[int, int]:
    """Static ``(n_alpha, n_beta)`` from the params' W_orb shapes."""
    return int(params.alpha.W_orb.shape[0]), int(params.beta.W_orb.shape[0])


def tier2_spin_blocked_init_params(
    n_atoms: int,
    n_alpha: int,
    n_beta: int,
    hidden: int,
    *,
    seed: int = 0,
    atoms: Array | None = None,
) -> Tier2SpinBlockedParams:
    p_alpha = tier2_init_params(
        n_atoms=n_atoms,
        n_orb=n_alpha,
        hidden=hidden,
        seed=seed,
        atoms=atoms,
    )
    p_beta = tier2_init_params(
        n_atoms=n_atoms,
        n_orb=n_beta,
        hidden=hidden,
        seed=seed + 1,
        atoms=p_alpha.atoms,
    )
    return Tier2SpinBlockedParams(alpha=p_alpha, beta=p_beta)


def tier2_blocked_log_abs_psi(
    params: Tier2SpinBlockedParams,
    r_flat: Array,
) -> Array:
    """``log|psi| = log|det M_alpha| + log|det M_beta|`` for spin-blocked Tier-2."""
    n_alpha, n_beta = _spin_block_sizes(params)
    r_alpha = jax.lax.dynamic_slice_in_dim(r_flat, 0, 3 * n_alpha)
    r_beta = jax.lax.dynamic_slice_in_dim(r_flat, 3 * n_alpha, 3 * n_beta)
    return tier2_log_abs_psi(params.alpha, r_alpha, n_e=n_alpha) + tier2_log_abs_psi(
        params.beta, r_beta, n_e=n_beta
    )


def tier2_blocked_local_kinetic_energy(
    params: Tier2SpinBlockedParams,
    r_flat: Array,
) -> Array:
    r"""Tier-2 spin-blocked local kinetic energy.

    Because the two spin blocks share NO coordinates,
    :math:`\nabla^{2}_r \log|\psi| = \nabla^{2}_{r_\alpha} \log|\det M_\alpha|
    + \nabla^{2}_{r_\beta} \log|\det M_\beta|`
    and similarly for :math:`\lVert\nabla\log|\psi|\rVert^{2}` since the
    gradients sit in disjoint coordinate subspaces. So the total
    closed-form ``T_loc`` is the sum of per-block closed-form
    ``T_loc``.
    """
    n_alpha, n_beta = _spin_block_sizes(params)
    r_alpha = jax.lax.dynamic_slice_in_dim(r_flat, 0, 3 * n_alpha)
    r_beta = jax.lax.dynamic_slice_in_dim(r_flat, 3 * n_alpha, 3 * n_beta)
    T_alpha = tier2_local_kinetic_energy(params.alpha, r_alpha, n_e=n_alpha)
    T_beta = tier2_local_kinetic_energy(params.beta, r_beta, n_e=n_beta)
    return T_alpha + T_beta


def tier2_value_grad_log_psi(
    params: Tier2Params,
    r_flat: Array,
    n_e: int,
) -> tuple[Array, Array]:
    r"""``(log|psi(r)|, grad_r log|psi(r)|)`` from closed-form columns.

    Used by VMC training loops to avoid recomputing the column
    derivatives twice. The Laplacian path above does not need
    ``log|psi|`` itself for ``T_loc``; this helper is a convenience
    for callers that want the full ``(value, grad, T_loc)`` triple.
    """
    r = r_flat.reshape((n_e, 3))
    column_fn = jax.vmap(_column_value_jac_hess, in_axes=(None, 0))
    phi_per_e, grad_per_e, _ = column_fn(params, r)
    M = phi_per_e.T
    sign, logdet = jnp.linalg.slogdet(M)
    del sign
    # solve(M, I) is the canonical stable inverse; the per-electron contraction
    # below reads whole rows of M^{-1}, so the full inverse is needed here.
    Minv = jnp.linalg.solve(M, jnp.eye(M.shape[-1], dtype=M.dtype))
    t = jnp.einsum("jg,jgm->jm", Minv, grad_per_e)  # (n_e, 3)
    return logdet, t.reshape(-1)


# ---------------------------------------------------------------------------
# Tier-2-full: symmetric pooling through g(r) = mean_l features_l(r_l)
# ---------------------------------------------------------------------------


class Tier2SymParams(NamedTuple):
    r"""Parameters of the **Tier-2-full** symmetric-pool wavefunction.

    Shapes
    ------
    W1_a
        (H, D_in)             -- tanh weight on the per-electron feature
                                 ``features_j(r_j)``.
    W1_b
        (H, D_in)             -- tanh weight on the symmetric mean
                                 ``g(r) = (1/N) sum_l features_l(r_l)``.
    b1
        (H,)                  -- tanh bias.
    W_orb
        (n_orb, H)            -- orbital readout.
    sigmas
        (n_orb, n_atoms)      -- envelope amplitudes.
    alphas
        (n_orb, n_atoms)      -- envelope decay rates.
    atoms
        (n_atoms, 3)          -- nuclear positions in Bohr.

    The per-electron pre-activation is

    .. math::

       z_j[h] = (W_{1,a}\,\mathrm{features}_j)[h]
              + (W_{1,b}\,g)[h] + b_1[h],

    so :class:`Tier2SymParams` reduces to :class:`Tier2Params` at
    ``W1_b = 0`` (and the closed-form Laplacian below reduces to
    :func:`tier2_local_kinetic_energy` in that limit -- a parity
    test is shipped in
    ``tests/test_jax_ferminet_restricted.py``).
    """

    W1_a: Array
    W1_b: Array
    b1: Array
    W_orb: Array
    sigmas: Array
    alphas: Array
    atoms: Array


def tier2sym_init_params(
    n_atoms: int,
    n_orb: int,
    hidden: int,
    *,
    seed: int = 0,
    atoms: Array | None = None,
    pool_scale: float = 1.0,
) -> Tier2SymParams:
    r"""Deterministic initializer for :class:`Tier2SymParams`.

    ``pool_scale`` rescales ``W1_b`` at init time; the default
    matches ``W1_a`` (so the per-electron and pooled pathways have
    comparable initial weight). Setting ``pool_scale=0`` recovers
    the Tier-2-lite ansatz exactly (the closed-form Laplacian path
    is then bit-identical to :func:`tier2_local_kinetic_energy`).
    """
    rng = np.random.default_rng(seed)
    D_in = 4 * n_atoms
    if atoms is None:
        atoms = jnp.asarray(
            rng.normal(scale=1.0, size=(n_atoms, 3)),
            dtype=jnp.float64,
        )
    W1_a = jnp.asarray(
        rng.normal(scale=1.0 / np.sqrt(D_in), size=(hidden, D_in)),
        dtype=jnp.float64,
    )
    W1_b = jnp.asarray(
        pool_scale
        * rng.normal(
            scale=1.0 / np.sqrt(D_in),
            size=(hidden, D_in),
        ),
        dtype=jnp.float64,
    )
    b1 = jnp.asarray(
        rng.normal(scale=0.05, size=(hidden,)),
        dtype=jnp.float64,
    )
    W_orb = jnp.asarray(
        rng.normal(scale=0.5, size=(n_orb, hidden)),
        dtype=jnp.float64,
    )
    sigmas = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(n_orb, n_atoms)),
        dtype=jnp.float64,
    )
    alphas = jnp.asarray(
        rng.uniform(0.8, 1.2, size=(n_orb, n_atoms)),
        dtype=jnp.float64,
    )
    return Tier2SymParams(
        W1_a=W1_a,
        W1_b=W1_b,
        b1=b1,
        W_orb=W_orb,
        sigmas=sigmas,
        alphas=alphas,
        atoms=atoms,
    )


def _orbital_matrix_sym(
    params: Tier2SymParams,
    r_flat: Array,
    n_e: int,
) -> Array:
    r"""``M[i, j] = env_i(r_j) * (W_orb[i] . tanh(W1_a f_j + W1_b g + b1))``.

    Each column ``j`` depends on all ``r_k`` through the symmetric
    mean ``g = (1/N) sum_l features_l(r_l)``.
    """
    r = r_flat.reshape((n_e, 3))
    feat_fn = jax.vmap(_features, in_axes=(0, None))
    features_per_e = feat_fn(r, params.atoms)  # (n_e, D_in)
    g = jnp.mean(features_per_e, axis=0)  # (D_in,)

    Z = (
        features_per_e @ params.W1_a.T + (g @ params.W1_b.T)[None, :] + params.b1[None, :]
    )  # (n_e, H)
    h = jnp.tanh(Z)  # (n_e, H)

    def env_at(r_j: Array) -> Array:
        delta = r_j[None, :] - params.atoms  # (n_atoms, 3)
        dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-30)
        return jnp.sum(
            params.sigmas * jnp.exp(-params.alphas * dist[None, :]),
            axis=-1,
        )  # (n_orb,)

    env_per_e = jax.vmap(env_at)(r)  # (n_e, n_orb)
    env = env_per_e.T  # (n_orb, n_e)
    orb_val = (h @ params.W_orb.T).T  # (n_orb, n_e)
    return env * orb_val


def tier2sym_log_abs_psi(
    params: Tier2SymParams,
    r_flat: Array,
    n_e: int,
) -> Array:
    """Autograd-friendly ``log|det M|`` for the Tier-2-full ansatz."""
    M = _orbital_matrix_sym(params, r_flat, n_e)
    _sign, logdet = jnp.linalg.slogdet(M)
    out: Array = logdet
    return out


def tier2sym_psi_fn(
    params: Tier2SymParams, n_e: int
) -> Callable[[Array, Array], tuple[Array, Array]]:
    r"""``FermiNetLike`` adapter: ``(R_flat, r_flat) -> (sign, log_abs)``.

    Mirrors :func:`tier2_psi_fn` for the symmetric-pool variant;
    the nuclear positions are re-bound through
    ``params._replace(atoms=...)``  so the BO derivative pipeline
    can take ``grad_R`` cleanly.
    """

    def psi_fn(R_flat: Array, r_flat: Array) -> tuple[Array, Array]:
        atoms = R_flat.reshape(params.atoms.shape)
        params_R = params._replace(atoms=atoms)
        M = _orbital_matrix_sym(params_R, r_flat, n_e)
        sign, logdet = jnp.linalg.slogdet(M)
        return sign, logdet

    return psi_fn


def tier2sym_local_kinetic_energy(
    params: Tier2SymParams,
    r_flat: Array,
    n_e: int,
) -> Array:
    r"""``T_loc`` for the symmetric-pool variant, fully closed form.

    Derivation
    ----------
    The orbital matrix is ``M[i, j] = env_i(r_j) * orb_i(r_j; g(r))``
    with the per-electron tanh interior

    .. math::

       z_j[h] &= (W_{1,a}\,f_j)[h] + (W_{1,b}\,g)[h] + b_1[h] \\
       h_j    &= \tanh(z_j), \qquad
       \mathrm{orb}_i(r_j; g) = (W_\mathrm{orb}[i, :] \cdot h_j)

    and ``g(r) = (1/N) sum_l f_l(r_l)`` the symmetric mean. The
    closed-form ``z`` Jacobian and diagonal Hessian are

    .. math::

       \frac{\partial z_j[h]}{\partial r_{k, \mu}}
         &= \delta_{kj}\,(W_{1,a}\,J_f[j])[h, \mu]
           + \tfrac{1}{N}\,(W_{1,b}\,J_f[k])[h, \mu], \\
       \frac{\partial^{2} z_j[h]}{\partial r_{k, \mu}^{2}}
         &= \delta_{kj}\,(W_{1,a}\,H_f[j])[h, \mu, \mu]
           + \tfrac{1}{N}\,(W_{1,b}\,H_f[k])[h, \mu, \mu],

    with :math:`J_f[k]`, :math:`H_f[k]` the closed-form Jacobian and
    Hessian of the features map at electron ``k`` (shipped by
    :func:`_features_jac_and_hessian`). Chain-rule through ``tanh``
    gives ``∂h_j/∂r_{k, μ}`` and ``∂²h_j/∂r_{k, μ}²``; chain rule
    through the linear orbital readout ``W_orb`` then gives the
    closed-form full ``(n_orb, n_e, n_e, 3)`` Jacobian tensor
    ``∂M/∂r`` and the per-entry trace
    ``sum_{k, μ} ∂²M[i, j]/∂r_{k, μ}²`` of ``M``.

    The matrix-identity Laplacian is then assembled directly:

    .. math::

       \nabla_r^{2} \log|\det M|
         = \mathrm{trace}(M^{-1}\,\mathrm{LapM})
           - \sum_{k, \mu} \mathrm{trace}\!\bigl(
              (M^{-1}\,\partial_{r_{k\mu}} M)^{2}\bigr).

    No autograd through ``tanh`` is used at any point.

    Complexity
    ----------
    With ``N = n_e``, ``D = 4 n_\text{atoms}``, ``H`` hidden width,
    the dominant work per walker is

    * one ``(n_orb, n_e, n_e, 3)`` ``∂M/∂r`` tensor build:
      :math:`O(n_\text{orb}\,n_e^{2}\,H\,d)` with ``d = 3``,
    * one ``trace((Minv ∂M)^2)`` quadratic form:
      :math:`O(n_e^{4})`,
    * one ``Minv`` solve: :math:`O(n_e^{3})`.

    For square ``n_orb = n_e`` and the FermiNet-class widths
    ``H ~ 64..256`` this is dominated by the ``n_e^{4}``
    quadratic form, which is the same scaling as the symmetric
    ``folx`` backend on the equivariant interior (both have to
    contract a (k, μ) - indexed jacobian against itself).
    """
    r = r_flat.reshape((n_e, 3))
    N = n_e

    # ---- Per-electron features + closed-form J_f, H_f --------------------
    feat_fn = jax.vmap(_features_jac_and_hessian, in_axes=(0, None))
    features_per_e, J_feat_per_e, H_feat_per_e = feat_fn(r, params.atoms)
    # features_per_e: (n_e, D_in)
    # J_feat_per_e:   (n_e, D_in, 3)
    # H_feat_per_e:   (n_e, D_in, 3, 3)

    # ---- Pooled mean g and tanh layer ------------------------------------
    g = jnp.mean(features_per_e, axis=0)  # (D_in,)
    Z = (
        features_per_e @ params.W1_a.T + (g @ params.W1_b.T)[None, :] + params.b1[None, :]
    )  # (n_e, H)
    h_act = jnp.tanh(Z)
    sigma_p = 1.0 - h_act * h_act  # sigma'(z)
    sigma_pp = -2.0 * h_act * sigma_p  # sigma''(z)

    # ---- Envelope (value, grad, full Hessian) ----------------------------
    env, grad_env, H_env = envelope_value_grad_hessian(
        r,
        params.atoms,
        params.sigmas,
        params.alphas,
    )  # (n_orb, n_e), (n_orb, n_e, 3), (n_orb, n_e, 3, 3)

    # ---- Orbital matrix M ------------------------------------------------
    orb_val = (h_act @ params.W_orb.T).T  # (n_orb, n_e)
    M = env * orb_val  # (n_orb, n_e), square

    # ---- Per-electron z-derivatives via W1_a (self) and W1_b (pool) ------
    # R_pre[j, h, mu] = (W1_a @ J_feat[j])[h, mu]
    # Q_pre[k, h, mu] = (W1_b @ J_feat[k])[h, mu]
    R_pre = jnp.einsum("ha,jam->jhm", params.W1_a, J_feat_per_e)
    Q_pre = jnp.einsum("ha,jam->jhm", params.W1_b, J_feat_per_e)

    # trace_{mu mu} of features Hessian, per electron.
    trH_feat = jnp.einsum("jamm->ja", H_feat_per_e)  # (n_e, D_in)
    R_H_trace_sum = trH_feat @ params.W1_a.T  # (n_e, H)
    Q_H_trace_sum = trH_feat @ params.W1_b.T  # (n_e, H)

    # ---- Closed-form orbital Jacobian wrt r (n_orb, n_e, n_e, 3) ---------
    # C[i, j, h] = W_orb[i, h] * sigma'(z_j)[h]
    C = params.W_orb[None, :, :] * sigma_p[:, None, :]  # (n_e, n_orb, H)
    # transpose to (n_orb, n_e, H) for natural einsum
    C = jnp.transpose(C, (1, 0, 2))  # (n_orb, n_e, H)

    # dorb_self[i, j, mu] = sum_h C[i, j, h] * R_pre[j, h, mu]   -- only k = j piece
    dorb_self = jnp.einsum("ijh,jhm->ijm", C, R_pre)
    # dorb_pool[i, j, k, mu] = (1/N) sum_h C[i, j, h] * Q_pre[k, h, mu]   -- pool piece
    dorb_pool = (1.0 / N) * jnp.einsum("ijh,khm->ijkm", C, Q_pre)

    # dM[i, j, k, mu]:
    #   = env[i, j] * dorb_pool[i, j, k, mu]
    #   + delta_{kj} * (env[i, j] * dorb_self[i, j, mu] + grad_env[i, j, mu] * orb_val[i, j])
    dM = env[:, :, None, None] * dorb_pool  # (n_orb, n_e, n_e, 3)
    diag_piece = env[:, :, None] * dorb_self + grad_env * orb_val[:, :, None]  # (n_orb, n_e, 3)
    j_idx = jnp.arange(n_e)
    dM = dM.at[:, j_idx, j_idx, :].add(diag_piece)

    # ---- LapM[i, j] = sum_{k, mu} d^2 M[i, j] / d r_{k, mu}^2 ------------
    # 1. Total ||d z_j[h] / d r||_F^2:
    #    sum_k sum_mu (d z_j[h] / d r_{k, mu})^2
    #      = ||W_eff_J[j, h, :]||^2  +  (1/N^2)
    #         * (sum_k ||Q_pre[k, h, :]||^2 - ||Q_pre[j, h, :]||^2)
    W_eff_J = R_pre + (1.0 / N) * Q_pre  # (n_e, H, 3)
    Z_J_sq = jnp.sum(W_eff_J * W_eff_J, axis=-1)  # (n_e, H)
    Q_sumsq = jnp.sum(Q_pre * Q_pre, axis=-1)  # (n_e, H)
    sum_Q_sumsq_all = jnp.sum(Q_sumsq, axis=0, keepdims=True)  # (1, H)
    Z_J_sq_total = Z_J_sq + (1.0 / (N * N)) * (sum_Q_sumsq_all - Q_sumsq)  # (n_e, H)

    # 2. Total sum_k sum_mu d^2 z_j[h] / d r_{k, mu}^2
    #      = R_H_trace_sum[j, h] + (1/N) sum_k Q_H_trace_sum[k, h]
    sum_Q_H_trace_all = jnp.sum(Q_H_trace_sum, axis=0, keepdims=True)
    Z_HH_total = R_H_trace_sum + (1.0 / N) * sum_Q_H_trace_all  # (n_e, H)

    # 3. Lap_h[j, h] = sigma''[j, h] * Z_J_sq_total + sigma'[j, h] * Z_HH_total
    Lap_h = sigma_pp * Z_J_sq_total + sigma_p * Z_HH_total  # (n_e, H)

    # 4. Lap_orb[i, j] = sum_h W_orb[i, h] * Lap_h[j, h]
    Lap_orb = (Lap_h @ params.W_orb.T).T  # (n_orb, n_e)

    # 5. Cross term: grad_env . grad_orb_full_at_self
    #    grad_orb_full_at_self[i, j, mu] = sum_h C[i, j, h] * W_eff_J[j, h, mu]
    grad_orb_full_at_self = jnp.einsum("ijh,jhm->ijm", C, W_eff_J)
    cross = jnp.sum(grad_env * grad_orb_full_at_self, axis=-1)  # (n_orb, n_e)

    # 6. trace_mu mu of envelope Hessian, per (orbital, electron)
    trace_H_env = jnp.einsum("ijmm->ij", H_env)  # (n_orb, n_e)

    # 7. LapM[i, j] = trace_H_env[i, j] * orb_val[i, j]
    #               + 2 * cross[i, j]
    #               + env[i, j] * Lap_orb[i, j]
    LapM = trace_H_env * orb_val + 2.0 * cross + env * Lap_orb

    # ---- Trace identities (via linear solves, not an explicit inverse) ----
    # Every trace below is a contraction of ``M^{-1} @ RHS``. Solving
    # ``M Y = RHS`` directly is more accurate than forming ``M^{-1}`` and
    # multiplying, especially for ill-conditioned orbital matrices.
    n_sq = M.shape[-1]

    # trace(M^{-1} LapM) = trace(solve(M, LapM)).
    S_lap = jnp.linalg.solve(M, LapM)  # (n_orb, n_e) = M^{-1} LapM
    trace1 = jnp.einsum("aa->", S_lap)  # scalar

    # X[a, c, k, mu] = sum_b (M^{-1})[a, b] dM[b, c, k, mu] = (M^{-1} dM)[a, ...].
    X = jnp.linalg.solve(M, dM.reshape(n_sq, -1)).reshape(dM.shape)
    trace2 = jnp.einsum("ackm,cakm->", X, X)  # scalar

    # tval[k, mu] = sum_{a, b} (M^{-1})[a, b] dM[b, a, k, mu] = sum_a X[a, a, k, mu].
    tval = jnp.einsum("aakm->km", X)  # (n_e, 3)
    grad_sq = jnp.sum(tval * tval)

    L = trace1 - trace2
    return -0.5 * (L + grad_sq)


def tier2sym_value_grad_log_psi(
    params: Tier2SymParams,
    r_flat: Array,
    n_e: int,
) -> tuple[Array, Array]:
    r"""``(log|psi|, grad_r log|psi|)`` for the Tier-2-full ansatz.

    Mirrors :func:`tier2_value_grad_log_psi`. Uses
    :func:`jax.value_and_grad` on the autograd reference (cheap for
    a single value + gradient; the closed-form gain is on the
    Laplacian).
    """
    log_abs = tier2sym_log_abs_psi(params, r_flat, n_e)
    grad = jax.grad(
        lambda x: tier2sym_log_abs_psi(params, x, n_e),
    )(r_flat)
    return log_abs, grad


class Tier2SymSpinBlockedParams(NamedTuple):
    r"""Spin-blocked Tier-2-full wavefunction parameters.

    Two independent :class:`Tier2SymParams` blocks; the pool is
    *per spin block*, so

    .. math::

       \psi(r) = \det\!\bigl(M_\alpha(r_\alpha)\bigr)\,
                 \det\!\bigl(M_\beta (r_\beta )\bigr),

    with ``g_\alpha(r_\alpha) = (1/n_\alpha)\sum_l f(r_{\alpha, l})``
    and analogously for ``\beta``. The two blocks share no
    coordinates, so the closed-form Laplacian is the sum of
    per-block closed-form Laplacians (no cross terms).
    """

    alpha: Tier2SymParams
    beta: Tier2SymParams


def _spin_block_sym_sizes(
    params: Tier2SymSpinBlockedParams,
) -> tuple[int, int]:
    """Static ``(n_alpha, n_beta)`` from ``W_orb`` shapes."""
    return (
        int(params.alpha.W_orb.shape[0]),
        int(params.beta.W_orb.shape[0]),
    )


def tier2sym_spin_blocked_init_params(
    n_atoms: int,
    n_alpha: int,
    n_beta: int,
    hidden: int,
    *,
    seed: int = 0,
    atoms: Array | None = None,
    pool_scale: float = 1.0,
) -> Tier2SymSpinBlockedParams:
    p_alpha = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_alpha,
        hidden=hidden,
        seed=seed,
        atoms=atoms,
        pool_scale=pool_scale,
    )
    p_beta = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_beta,
        hidden=hidden,
        seed=seed + 1,
        atoms=p_alpha.atoms,
        pool_scale=pool_scale,
    )
    return Tier2SymSpinBlockedParams(alpha=p_alpha, beta=p_beta)


def tier2sym_blocked_log_abs_psi(
    params: Tier2SymSpinBlockedParams,
    r_flat: Array,
) -> Array:
    """``log|psi| = log|det M_alpha| + log|det M_beta|`` for Tier-2-full."""
    n_alpha, n_beta = _spin_block_sym_sizes(params)
    r_alpha = jax.lax.dynamic_slice_in_dim(r_flat, 0, 3 * n_alpha)
    r_beta = jax.lax.dynamic_slice_in_dim(r_flat, 3 * n_alpha, 3 * n_beta)
    return tier2sym_log_abs_psi(params.alpha, r_alpha, n_e=n_alpha) + tier2sym_log_abs_psi(
        params.beta, r_beta, n_e=n_beta
    )


def tier2sym_blocked_local_kinetic_energy(
    params: Tier2SymSpinBlockedParams,
    r_flat: Array,
) -> Array:
    r"""Tier-2-full spin-blocked local kinetic energy.

    The two spin blocks share no coordinates, so the closed-form
    ``T_loc`` is the sum of per-block closed-form ``T_loc``
    (mirrors :func:`tier2_blocked_local_kinetic_energy`).
    """
    n_alpha, n_beta = _spin_block_sym_sizes(params)
    r_alpha = jax.lax.dynamic_slice_in_dim(r_flat, 0, 3 * n_alpha)
    r_beta = jax.lax.dynamic_slice_in_dim(r_flat, 3 * n_alpha, 3 * n_beta)
    T_alpha = tier2sym_local_kinetic_energy(
        params.alpha,
        r_alpha,
        n_e=n_alpha,
    )
    T_beta = tier2sym_local_kinetic_energy(
        params.beta,
        r_beta,
        n_e=n_beta,
    )
    return T_alpha + T_beta

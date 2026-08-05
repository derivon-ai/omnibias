# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Direct Galerkin generalized-eigenvalue solver (torch backend).

The classical PINN approach to bound-state eigenvalue problems trains
a single wavefunction :math:`\psi(x)` against a Rayleigh-quotient loss
:math:`\mathcal R[\psi] = \langle\psi|H|\psi\rangle /
\langle\psi|\psi\rangle` using Adam or LBFGS. This works but suffers
from two structural defects:

1. **Adam is a poor eigenvalue optimizer.** In a high-precision
   eigenvalue problem the signal is :math:`\langle\psi|H|\psi\rangle -
   E_{\rm true} = \mathcal O(\varepsilon^2)` smaller than the overall
   loss scale; gradient noise dominates long before spectroscopic
   accuracy is reached.
2. **Sequential deflation for excited states is unstable.** Errors in
   :math:`\psi_0` leak into :math:`\psi_1` through the Gram-Schmidt
   projector; v=1 NH3 splittings in the v0.0.2a1 benchmark batch had 200%
   relative error for exactly this reason.

This module replaces the eigenvalue extraction with a *direct*
generalized eigensolver. The trainable :math:`K`-channel
:class:`FieldBase` provides a *basis* :math:`\{\phi_k(x)\}_{k=1}^{K}`
(real-valued or complex). The user picks a fixed quadrature grid and
weights :math:`(x_q, w_q)`. We then assemble in closed form:

.. math::

    S_{ij} &= \sum_q w_q\,\phi_i^*(x_q)\,\phi_j(x_q), \\
    H_{ij} &= \sum_q w_q\,\phi_i^*(x_q)\,\big(\hat H \phi_j\big)(x_q),

where :math:`\hat H \phi_j` is computed via the omnibias closed-form
Laplacian + the user-supplied potential. Solving the generalized
eigenproblem :math:`H c = E S c` via :func:`scipy.linalg.eigh` yields
the :math:`K` lowest variational eigenpairs in **one** call. The
network's only remaining job is to provide a *good* basis -- a much
easier optimization than nailing the eigenvalue itself.

For maximum bit-stable accuracy the omnibias Laplacian is computed
through the same backbone the PINN trains on, so the matrix elements
are exact to float64 precision. Variational improvement of the basis
via :func:`galerkin_trace_loss` (the trace of the Galerkin
:math:`S^{-1} H` -- a variational upper bound on the sum of the
:math:`K` lowest eigenvalues) drives the basis toward the true
eigenstates without ever taking a gradient step *on* the eigenvalue.

This is the textbook *spectral-Galerkin* / *Rayleigh-Ritz* method but
with a *trainable*, adaptive basis. The reduction in K relative to a
uniform DVR is typically 5-10x for confined eigenstates because the
network places its Gaussians where the wavefunction lives.

Backend
-------
**Torch only** (alpha). The Galerkin assembly consumes a torch
:class:`~omnibias.pinn.torch.fields.base.FieldBase` basis and the torch
closed-form Laplacian, then solves ``H c = E S c`` with
:func:`scipy.linalg.eigh`. A JAX twin (a ``jax.scipy.linalg.eigh`` assembly on
an ``omnibias.pinn.jax`` basis) is on the roadmap; until it lands
``omnibias.qpinn.jax`` intentionally ships no ``eigensolvers`` submodule rather
than a silent partial one. The *equation / cage / diagnostics* residuals remain
bit-identical across the torch and JAX backends.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

try:
    import scipy.linalg as _scipy_linalg
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.fields.base import FieldBase


@dataclass(frozen=True)
class GalerkinEigenResult:
    """Output of a single generalized Galerkin eigensolve.

    Attributes
    ----------
    eigenvalues
        Float64 numpy array of length ``n_states``, sorted ascending.
    eigenvectors
        Coefficient matrix of shape ``(K, n_states)``: each column gives
        the expansion of the i-th eigenstate in the trained basis.
    overlap
        The :math:`K \\times K` Gram matrix :math:`S`.
    hamiltonian
        The :math:`K \\times K` reduced Hamiltonian :math:`H`.
    cond_S
        Condition number of :math:`S`; values much larger than ``1e10``
        indicate near-linear dependence in the basis -- typically a
        sign that ``K`` is too high relative to the basis expressiveness.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    overlap: np.ndarray
    hamiltonian: np.ndarray
    cond_S: float


def _basis_values(
    state: FieldState,
    basis_names: Sequence[str],
) -> Tensor:
    """Return ``(B_quad, K)`` stack of basis values at the quadrature grid."""
    cols = [state.ops.value(state, n) for n in basis_names]
    return torch.stack(cols, dim=-1)


def _basis_laplacian(
    state: FieldState,
    basis_names: Sequence[str],
) -> Tensor:
    """Return ``(B_quad, K)`` stack of basis Laplacians at the quadrature grid."""
    cols = [state.ops.laplacian(state, n) for n in basis_names]
    return torch.stack(cols, dim=-1)


def galerkin_matrices(
    *,
    field: FieldBase,
    quadrature_coords: Tensor,
    quadrature_weights: Tensor,
    basis_names: Sequence[str],
    potential_fn: Callable[[Tensor], Tensor],
    kinetic_prefactor: float,
    diagonal_shift: float = 0.0,
) -> tuple[Tensor, Tensor]:
    r"""Build the Galerkin :math:`(S, H)` matrices for the bound-state
    Hamiltonian :math:`\hat H = -\alpha\,\Delta + V(x)`.

    Parameters
    ----------
    field
        Trainable :class:`FieldBase` exposing the basis components as
        named outputs. For NH3 this is a parity-projected,
        boundary-pinned multi-channel :class:`OneLayerVectorField`.
    quadrature_coords
        Float64 quadrature grid of shape ``(B_quad, D)``.
    quadrature_weights
        Float64 quadrature weights of shape ``(B_quad,)``.
    basis_names
        Names of the basis components on ``field``. Length is ``K``.
    potential_fn
        Callable mapping ``coords (B_quad, D) -> V(B_quad)``.
    kinetic_prefactor
        :math:`\alpha = \hbar^2 / (2 m)`. In atomic units with
        :math:`\hbar = 1`, this is ``1 / (2 * reduced_mass)``.
    diagonal_shift
        Optional :math:`\sigma I` added to the Hamiltonian; useful when
        ``V`` is unbounded below and the user wants to invert
        :math:`H + \sigma I` for shift-and-invert iteration. Default
        ``0``.

    Returns
    -------
    S, H
        The :math:`K \times K` Gram and reduced Hamiltonian matrices as
        torch tensors. Both are real (the basis is assumed real).
    """
    state = field(quadrature_coords)
    B = _basis_values(state, basis_names)              # (B_quad, K)
    Lap = _basis_laplacian(state, basis_names)          # (B_quad, K)
    V = potential_fn(quadrature_coords)                 # (B_quad,)
    if V.ndim != 1 or V.shape[0] != B.shape[0]:
        raise ValueError(
            f"potential_fn must return shape ({B.shape[0]},); got "
            f"{tuple(V.shape)}"
        )
    if quadrature_weights.ndim != 1 or quadrature_weights.shape[0] != B.shape[0]:
        raise ValueError(
            f"quadrature_weights must be shape ({B.shape[0]},); got "
            f"{tuple(quadrature_weights.shape)}"
        )
    w = quadrature_weights.to(B.dtype)
    # H phi_j = (-alpha) Lap_j + V * phi_j  + diagonal_shift * phi_j
    H_psi = (-kinetic_prefactor) * Lap + V.unsqueeze(-1) * B
    if diagonal_shift != 0.0:
        H_psi = H_psi + diagonal_shift * B
    # S_{ij} = sum_q w_q B[q,i] B[q,j]
    Bw = B * w.unsqueeze(-1)
    S = Bw.transpose(-1, -2) @ B
    # H_{ij} = sum_q w_q B[q,i] (H phi_j)[q]
    H = Bw.transpose(-1, -2) @ H_psi
    # Symmetrise to suppress numerical asymmetry (it would otherwise
    # leak into scipy.linalg.eigh's symmetric-matrix assumption).
    S = 0.5 * (S + S.transpose(-1, -2))
    H = 0.5 * (H + H.transpose(-1, -2))
    return S, H


def galerkin_eigh(
    *,
    field: FieldBase,
    quadrature_coords: Tensor,
    quadrature_weights: Tensor,
    basis_names: Sequence[str],
    potential_fn: Callable[[Tensor], Tensor],
    kinetic_prefactor: float,
    n_states: int | None = None,
    diagonal_shift: float = 0.0,
    overlap_pseudoinverse_tol: float = 1e-10,
) -> GalerkinEigenResult:
    r"""End-to-end Galerkin eigensolve: build matrices, then ``eigh(H, S)``.

    The generalized eigenproblem :math:`H c = E S c` is solved with
    :func:`scipy.linalg.eigh` when SciPy is available (uses LAPACK
    ``dsygvd``), falling back to a numerically-stable Cholesky-based
    rewrite :math:`L^{-1} H L^{-T} \tilde c = E\,\tilde c` followed by
    :func:`torch.linalg.eigh`. Both paths return real eigenvalues in
    ascending order.

    For ill-conditioned :math:`S` (e.g. when ``K`` is close to the
    number of effectively-independent basis functions), the routine
    automatically symmetrically diagonalizes :math:`S`, drops eigenvalues
    below ``overlap_pseudoinverse_tol * max(eig(S))``, and solves the
    eigenproblem in the reduced subspace. The dropped subspace is
    silent (no warning) but is reported via the returned ``cond_S``.

    Parameters and Returns
    ----------------------
    See :class:`GalerkinEigenResult` for the output schema.

    Notes
    -----
    The returned eigenvalues / eigenvectors are computed in pure
    numpy (float64) on CPU. The torch graph is detached -- this is a
    one-shot diagnostic / extraction step, not a training operation.
    Use :func:`galerkin_trace_loss` for gradient-driven basis
    optimization.
    """
    S_t, H_t = galerkin_matrices(
        field=field,
        quadrature_coords=quadrature_coords,
        quadrature_weights=quadrature_weights,
        basis_names=basis_names,
        potential_fn=potential_fn,
        kinetic_prefactor=kinetic_prefactor,
        diagonal_shift=diagonal_shift,
    )
    S = S_t.detach().double().cpu().numpy()
    H = H_t.detach().double().cpu().numpy()
    # Condition number for diagnostics.
    s_eigvals = np.linalg.eigvalsh(S)
    s_min = float(s_eigvals.min())
    s_max = float(s_eigvals.max())
    cond_S = float("inf") if s_min <= 0 else s_max / max(s_min, 1e-300)
    if n_states is None:
        n_states = S.shape[0]
    n_states = min(int(n_states), S.shape[0])

    # If S is well-conditioned, use the standard generalized eigh.
    well_conditioned = (s_min > overlap_pseudoinverse_tol * max(s_max, 1.0))
    if well_conditioned and _HAVE_SCIPY:
        eigvals, eigvecs = _scipy_linalg.eigh(H, S)
    elif well_conditioned:
        # Cholesky-based rewrite. S = L L^T => H' = L^{-1} H L^{-T}.
        L = np.linalg.cholesky(S)
        L_inv = np.linalg.solve(L, np.eye(S.shape[0]))
        Hp = L_inv @ H @ L_inv.T
        Hp = 0.5 * (Hp + Hp.T)
        eigvals, V_hat = np.linalg.eigh(Hp)
        eigvecs = L_inv.T @ V_hat
    else:
        # Symmetric pseudo-inverse path. S = U diag(s) U^T,
        # keep s_k > tol * s_max; solve in reduced subspace.
        s, U = np.linalg.eigh(S)
        s_keep = s > overlap_pseudoinverse_tol * max(s_max, 1.0)
        if not s_keep.any():
            raise RuntimeError(
                "Galerkin overlap matrix is singular: no eigenvalues "
                f"above tol={overlap_pseudoinverse_tol}*s_max={s_max:.2e}"
            )
        s_red = s[s_keep]
        U_red = U[:, s_keep]
        S_inv_half = U_red / np.sqrt(s_red)  # (K, k)
        Hp = S_inv_half.T @ H @ S_inv_half
        Hp = 0.5 * (Hp + Hp.T)
        eigvals_red, V_red = np.linalg.eigh(Hp)
        eigvecs = S_inv_half @ V_red
        eigvals = eigvals_red

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return GalerkinEigenResult(
        eigenvalues=np.ascontiguousarray(eigvals[:n_states] - diagonal_shift),
        eigenvectors=np.ascontiguousarray(eigvecs[:, :n_states]),
        overlap=S,
        hamiltonian=H,
        cond_S=cond_S,
    )


def galerkin_eigh_real_basis(
    *,
    field: FieldBase,
    quadrature_coords: Tensor,
    quadrature_weights: Tensor,
    basis_re_names: Sequence[str],
    potential_fn: Callable[[Tensor], Tensor],
    kinetic_prefactor: float,
    n_states: int | None = None,
    diagonal_shift: float = 0.0,
    overlap_pseudoinverse_tol: float = 1e-10,
) -> GalerkinEigenResult:
    """Convenience wrapper for purely-real basis functions.

    Equivalent to :func:`galerkin_eigh` with ``basis_names = basis_re_names``;
    exists to make the API consistent with the
    ``(psi_re, psi_im)`` complex pair convention used elsewhere in
    omnibias-qpinn.
    """
    return galerkin_eigh(
        field=field,
        quadrature_coords=quadrature_coords,
        quadrature_weights=quadrature_weights,
        basis_names=tuple(basis_re_names),
        potential_fn=potential_fn,
        kinetic_prefactor=kinetic_prefactor,
        n_states=n_states,
        diagonal_shift=diagonal_shift,
        overlap_pseudoinverse_tol=overlap_pseudoinverse_tol,
    )


def galerkin_trace_loss(
    *,
    field: FieldBase,
    quadrature_coords: Tensor,
    quadrature_weights: Tensor,
    basis_names: Sequence[str],
    potential_fn: Callable[[Tensor], Tensor],
    kinetic_prefactor: float,
    overlap_regulariser: float = 1e-8,
) -> Tensor:
    r"""Variational *trace* loss :math:`\mathrm{tr}\,(S^{-1} H)`.

    Equal to the sum of the :math:`K` lowest eigenvalues of the reduced
    eigenproblem :math:`H c = E S c` when the basis is orthonormal;
    in general it is a (very tight) variational upper bound on
    :math:`\sum_{k<K} E_k`. Minimising this drives the trainable basis
    toward the :math:`K`-dimensional invariant subspace spanned by the
    :math:`K` lowest eigenstates.

    The implementation is :math:`\mathrm{tr}\,(L^{-T} H L^{-1})` with
    :math:`L L^T = S + \epsilon I`, which is numerically stable and
    fully differentiable through :func:`torch.linalg.cholesky_solve`.
    The ``epsilon I`` regulariser avoids loss explosions in the early
    iterations before the basis spreads.
    """
    S, H = galerkin_matrices(
        field=field,
        quadrature_coords=quadrature_coords,
        quadrature_weights=quadrature_weights,
        basis_names=basis_names,
        potential_fn=potential_fn,
        kinetic_prefactor=kinetic_prefactor,
    )
    K = S.shape[0]
    eye = torch.eye(K, dtype=S.dtype, device=S.device)
    S_reg = S + overlap_regulariser * eye
    L = torch.linalg.cholesky(S_reg)
    # H_red = L^{-1} H L^{-T}, trace = tr(L^{-1} H L^{-T}) = tr(L^{-T} L^{-1} H) = tr(S^{-1} H).
    L_inv_H = torch.linalg.solve_triangular(L, H, upper=False)
    H_red = torch.linalg.solve_triangular(
        L.transpose(-1, -2), L_inv_H, upper=True, left=False,
    )
    return torch.diagonal(H_red, dim1=-2, dim2=-1).sum()

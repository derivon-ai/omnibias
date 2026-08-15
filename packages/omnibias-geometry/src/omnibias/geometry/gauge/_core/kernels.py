# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic gauge-theory numeric kernels.

Every function takes the array module ``xp`` (``torch`` or ``jax.numpy``) as its
first argument and operates purely through ``xp.einsum`` and arithmetic, so the
torch and jax operator surfaces are bit-identical twins by construction (they
share this one implementation). Inputs are backend tensors; structure constants,
the Levi-Civita symbol and the signature diagonal are passed in already as
backend tensors.

Index conventions (``B`` = batch, ``d`` = spacetime dim, ``n`` = adjoint dim):

* ``A[B, mu, a]``           connection ``A_mu^a``
* ``dA[B, rho, nu, a]``     ``d_rho A_nu^a``
* ``ddA[B, rho, mu, nu, a]``  ``d_rho d_mu A_nu^a``
* ``F[B, mu, nu, a]``       field strength ``F_{mu nu}^a`` (antisymmetric in mu, nu)

The metric is flat and diagonal with entries ``eta`` (each ``+/-1``); since
``eta^{-1} = eta`` it doubles as the inverse-metric diagonal, and ``|det g| = 1``.
"""

from __future__ import annotations

from typing import Any


def lie_bracket(xp: Any, x: Any, y: Any, f: Any) -> Any:
    r"""Component Lie bracket ``[x, y]^c = f^{abc} x^a y^b`` for ``(B, n)`` inputs."""
    return xp.einsum("abc,Ba,Bb->Bc", f, x, y)


def field_strength(xp: Any, A: Any, dA: Any, f: Any, coupling: float) -> Any:
    r"""``F_{mu nu}^a = d_mu A_nu^a - d_nu A_mu^a + g f^{abc} A_mu^b A_nu^c``.

    ``dA[B, mu, nu, a] = d_mu A_nu^a``; the exterior part is its antisymmetrization
    in ``(mu, nu)`` and the bracket term is the non-abelian contribution.
    """
    exterior = dA - xp.swapaxes(dA, 1, 2)
    bracket = coupling * xp.einsum("pqc,Bmp,Bnq->Bmnc", f, A, A)
    return exterior + bracket


def field_strength_partials(
    xp: Any, A: Any, dA: Any, ddA: Any, f: Any, coupling: float
) -> Any:
    r"""``d_rho F_{mu nu}^a`` of shape ``(B, rho, mu, nu, a)``."""
    exterior = ddA - xp.swapaxes(ddA, 2, 3)
    term1 = xp.einsum("pqc,Brmp,Bnq->Brmnc", f, dA, A)
    term2 = xp.einsum("pqc,Bmp,Brnq->Brmnc", f, A, dA)
    return exterior + coupling * (term1 + term2)


def covariant_derivative_field_strength(
    xp: Any, A: Any, dA: Any, ddA: Any, f: Any, coupling: float
) -> Any:
    r"""``(D_rho F_{mu nu})^a`` of shape ``(B, rho, mu, nu, a)``.

    The gauge-covariant derivative of the adjoint 2-form,

    .. math::

        (D_\rho F_{\mu\nu})^a
        = \partial_\rho F_{\mu\nu}^a + g\,f^{abc}\,A_\rho^b F_{\mu\nu}^c.

    This is the uncontracted fiber of a first-order covariant jet. The Yang-Mills
    operator :func:`covariant_divergence` is its metric contraction
    ``eta^{rho mu} eta^{nu lambda} (D_rho F_{mu lambda})^a``.
    """
    dF = field_strength_partials(xp, A, dA, ddA, f, coupling)
    fld = field_strength(xp, A, dA, f, coupling)
    bracket = coupling * xp.einsum("pqa,Brp,Bmnq->Brmna", f, A, fld)
    return dF + bracket


def covariant_divergence(
    xp: Any, A: Any, dA: Any, ddA: Any, f: Any, coupling: float, eta: Any
) -> Any:
    r"""The Yang-Mills operator ``(D_mu F^{mu nu})^a`` of shape ``(B, nu, a)``.

    ``D_mu F^{mu nu, a} = d_mu F^{mu nu, a} + g f^{abc} A_mu^b F^{mu nu, c}`` with
    the upper indices raised by the flat metric ``eta``.
    """
    fld = field_strength(xp, A, dA, f, coupling)
    dF = field_strength_partials(xp, A, dA, ddA, f, coupling)
    div = xp.einsum("m,n,Bmmna->Bna", eta, eta, dF)
    bracket = coupling * xp.einsum("m,n,pqa,Bmp,Bmnq->Bna", eta, eta, f, A, fld)
    return div + bracket


def yang_mills_gradient_flow_rhs(
    xp: Any,
    A: Any,
    dA: Any,
    ddA: Any,
    f: Any,
    coupling: float,
    eta: Any,
    *,
    deturck: float = 0.0,
) -> Any:
    r"""Yang-Mills gradient-flow / stochastic-quantisation drift, shape ``(B, nu, a)``.

    Returns the Wilson/Luscher gradient-flow drift ``-delta S / delta A_nu^a``,
    which equals the Yang-Mills operator ``(D_mu F^{mu nu})^a`` and vanishes on a
    classical solution (e.g. an instanton). This is the deterministic part of the
    Parisi-Wu Langevin equation ``dA = (drift) dtau + dW``.

    With ``deturck != 0`` the DeTurck-Zwanziger gauge term
    ``deturck * (D_nu (d_mu A^mu))^a`` is added. This term -- the gradient of the
    gauge-fixing functional dragged along by the flow -- makes the otherwise
    only-weakly-parabolic flow strongly parabolic (Donaldson / Zwanziger; it is
    the lattice/continuum analogue of the DeTurck trick used by
    Chandra-Chevyrev-Hairer-Shen to renormalise the stochastic Yang-Mills flow).
    The added term is a pure gauge direction, so it does not change the physical
    (gauge-invariant) fixed points.
    """
    drift = covariant_divergence(xp, A, dA, ddA, f, coupling, eta)
    if deturck != 0.0:
        div_a = xp.einsum("m,Bmma->Ba", eta, dA)
        d_div_a = xp.einsum("m,Brmma->Bra", eta, ddA)
        gauge_term = covariant_derivative_adjoint(xp, div_a, d_div_a, A, f, coupling)
        drift = drift + deturck * gauge_term
    return drift


def dual_field_strength(xp: Any, F: Any, eps: Any, eta: Any) -> Any:
    r"""Hodge dual ``\tilde F_{mu nu}^a = (1/2) eps_{mu nu rho sigma} F^{rho sigma, a}``."""
    f_up = xp.einsum("p,q,Bpqa->Bpqa", eta, eta, F)
    return 0.5 * xp.einsum("mnpq,Bpqa->Bmna", eps, f_up)


def bianchi(
    xp: Any,
    A: Any,
    dA: Any,
    ddA: Any,
    f: Any,
    coupling: float,
    eta: Any,
    eps: Any,
) -> Any:
    r"""Bianchi operator ``(D_mu \tilde F^{mu nu})^a`` -> ``(B, nu, a)`` (identically 0).

    The Bianchi identity ``D \tilde F = 0`` holds for *any* connection, so this is
    a closed-form consistency check: its output is zero up to rounding.
    """
    fld = field_strength(xp, A, dA, f, coupling)
    dF = field_strength_partials(xp, A, dA, ddA, f, coupling)
    f_dual = dual_field_strength(xp, fld, eps, eta)
    dF_up = xp.einsum("p,q,Brpqa->Brpqa", eta, eta, dF)
    d_dual = 0.5 * xp.einsum("mnpq,Brpqa->Brmna", eps, dF_up)
    div = xp.einsum("m,n,Bmmna->Bna", eta, eta, d_dual)
    bracket = coupling * xp.einsum("m,n,pqa,Bmp,Bmnq->Bna", eta, eta, f, A, f_dual)
    return div + bracket


def self_duality_defect(xp: Any, F: Any, eps: Any, eta: Any) -> Any:
    r"""``F - \tilde F`` (zero exactly for a self-dual field; e.g. an instanton)."""
    return F - dual_field_strength(xp, F, eps, eta)


def anti_self_duality_defect(xp: Any, F: Any, eps: Any, eta: Any) -> Any:
    r"""``F + \tilde F`` (zero for an anti-self-dual field; e.g. an anti-instanton)."""
    return F + dual_field_strength(xp, F, eps, eta)


def action_density(xp: Any, F: Any, eta: Any) -> Any:
    r"""Yang-Mills Lagrangian density ``(1/4) F_{mu nu}^a F^{mu nu, a}`` -> ``(B,)``."""
    return 0.25 * xp.einsum("m,n,Bmna,Bmna->B", eta, eta, F, F)


def topological_charge_density(xp: Any, F: Any, eps: Any) -> Any:
    r"""Topological charge density (instanton-number density) -> ``(B,)``.

    The second Chern number is ``Q = (1/8 pi^2) int tr(F ^ F)``; with the
    fundamental normalization ``tr(T^a T^b) = 1/2 delta^{ab}`` this is

    .. math::

        q(x) = \frac{1}{32\pi^2}\,F_{\mu\nu}^a\,\tilde F^{\mu\nu, a}
             = \frac{1}{64\pi^2}\,\varepsilon^{\mu\nu\rho\sigma}
               F_{\mu\nu}^a F_{\rho\sigma}^a,

    so a single BPST instanton integrates to ``Q = 1``. The density is
    metric-independent (it uses the Levi-Civita *symbol*).
    """
    pi = 3.141592653589793
    contracted = xp.einsum("mnpq,Bmna,Bpqa->B", eps, F, F)
    return contracted / (64.0 * pi * pi)


def covariant_derivative_adjoint(
    xp: Any, phi: Any, dphi: Any, A: Any, f: Any, coupling: float
) -> Any:
    r"""``(D_mu phi)^a = d_mu phi^a + g f^{abc} A_mu^b phi^c`` for an adjoint scalar.

    ``phi[B, a]``, ``dphi[B, mu, a] = d_mu phi^a`` -> ``(B, mu, a)``. With
    ``phi = omega`` (a gauge parameter) this is the infinitesimal gauge variation
    ``delta A_mu = D_mu omega``.
    """
    bracket = coupling * xp.einsum("pqa,Bmp,Bq->Bma", f, A, phi)
    return dphi + bracket


def gauge_variation_field_strength(
    xp: Any, F: Any, omega: Any, f: Any, coupling: float
) -> Any:
    r"""Infinitesimal homogeneous variation ``delta F_{mu nu}^a = g f^{abc} omega^b F_{mu nu}^c``."""
    return coupling * xp.einsum("pqa,Bp,Bmnq->Bmna", f, omega, F)


def covariant_derivative_commutator(
    xp: Any, phi: Any, dphi: Any, ddphi: Any, A: Any, dA: Any, f: Any, coupling: float
) -> Any:
    r"""Commutator ``([D_mu, D_nu] phi)^a`` of an adjoint scalar -> ``(B, mu, nu, a)``.

    Computed the *derivative way* by nesting :func:`covariant_derivative_adjoint`
    (``D_mu(D_nu phi) - D_nu(D_mu phi)``) from ``phi[B, a]``, its partials
    ``dphi[B, mu, a] = d_mu phi^a`` and ``ddphi[B, mu, nu, a] = d_mu d_nu phi^a``,
    the connection ``A[B, mu, a]`` and ``dA[B, rho, nu, a] = d_rho A_nu^a``. By the
    Ricci identity this equals :func:`curvature_action_on_adjoint` of the
    closed-form :func:`field_strength` -- the *local* nonintegrability of ``D``.
    """
    cov_phi = covariant_derivative_adjoint(xp, phi, dphi, A, f, coupling)  # D_nu phi
    d_cov_phi = ddphi + coupling * (
        xp.einsum("abc,Bmnb,Bc->Bmna", f, dA, phi)
        + xp.einsum("abc,Bnb,Bmc->Bmna", f, A, dphi)
    )
    dd = d_cov_phi + coupling * xp.einsum("abc,Bmb,Bnc->Bmna", f, A, cov_phi)
    return dd - xp.swapaxes(dd, 1, 2)


def curvature_action_on_adjoint(xp: Any, F: Any, phi: Any, f: Any, coupling: float) -> Any:
    r"""Homogeneous curvature action ``g f^{abc} F_{mu nu}^b phi^c`` -> ``(B, mu, nu, a)``.

    The closed-form right-hand side of the Ricci identity
    ``[D_mu, D_nu] phi = g\,f\,F\,phi`` (the adjoint action of the field strength
    on ``phi``).
    """
    return coupling * xp.einsum("abc,Bmnb,Bc->Bmna", f, F, phi)


def to_matrix(xp: Any, A_comp: Any, generators: Any) -> Any:
    r"""Map adjoint components to fundamental matrices ``A = A^a T^a``.

    ``A_comp[..., a]``, ``generators[a, i, j]`` -> ``(..., i, j)`` matrices.
    """
    return xp.einsum("...a,aij->...ij", A_comp, generators)


def from_matrix(xp: Any, A_mat: Any, generators: Any) -> Any:
    r"""Project fundamental matrices back to real adjoint components ``A^a = 2 Re tr(T^a A)``."""
    comps = 2.0 * xp.einsum("aji,...ij->...a", generators, A_mat)
    return xp.real(comps) if hasattr(xp, "real") else comps.real


def wilson_line_exponents(
    xp: Any, A_path: Any, tangents: Any, generators: Any, coupling: float, dt: float
) -> Any:
    r"""Per-segment path-ordered exponents ``X_i = -i g\,(A_mu^a \dot x^mu\,dt)\,T^a``.

    Assembles the anti-Hermitian generators of the parallel transport / Wilson
    line from the connection sampled along a curve: ``A_path[N, d, a]`` is the
    connection at ``N`` ordered path points, ``tangents[N, d] = \dot x^mu`` the
    curve tangents, ``generators[a, i, j]`` the (Hermitian) representation
    matrices ``T^a``, and ``dt`` the parameter step. Returns ``X[N, i, j]``
    (complex); matrix-exponentiating each and ordered-multiplying yields the
    holonomy ``U(C) = P exp(-i g \int A_mu^a T^a\,dx^mu)``. The ``-i g`` factor is
    fixed so an infinitesimal-loop holonomy reproduces :func:`field_strength`.
    """
    a_dot = xp.einsum("Nda,Nd->Na", A_path, tangents) * dt
    a_dot = xp.asarray(a_dot, dtype=generators.dtype)
    a_mat = to_matrix(xp, a_dot, generators)
    return (-1j * coupling) * a_mat


__all__ = [
    "action_density",
    "anti_self_duality_defect",
    "bianchi",
    "covariant_derivative_adjoint",
    "covariant_derivative_commutator",
    "covariant_derivative_field_strength",
    "covariant_divergence",
    "curvature_action_on_adjoint",
    "dual_field_strength",
    "field_strength",
    "field_strength_partials",
    "from_matrix",
    "gauge_variation_field_strength",
    "lie_bracket",
    "self_duality_defect",
    "to_matrix",
    "topological_charge_density",
    "wilson_line_exponents",
    "yang_mills_gradient_flow_rhs",
]

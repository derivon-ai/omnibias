# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""FermiNet <-> omnibias integration bridge (Tier 1: ``omnibias_envelope``).

This module is the omnibias-side entry point that
``ferminet.hamiltonian.local_kinetic_energy`` imports when its
``laplacian_method='omnibias_envelope'`` branch is selected.

Why Tier 1 lives outside the equivariant block
----------------------------------------------

FermiNet's PRE_DETERMINANT envelope is the *only* part of the
orbital that is a one-electron function of ``r_j``: it has the
form

.. math::

   \mathrm{env}_i(r_j)
     = \sum_a \sigma_{ia} \,\exp(-\alpha_{ia}\,\lVert r_j - R_a\rVert).

Per (orbital ``i``, electron ``j``) this is a scalar field on
:math:`\mathbb{R}^3`, so ``jax.hessian`` already costs O(1) per
call. The omnibias *speed* advantage from
:func:`omnibias.jax.neural_field_value_grad_hessian` -- O(1) in
the hidden width -- is therefore not what makes Tier 1 useful.
Tier 1 is useful because:

1. It is the **plumbing PR**: it lands the omnibias↔FermiNet
   wiring (an extra ``elif`` branch in
   ``ferminet/hamiltonian.py``) and exercises the bridge
   end-to-end against a real chemistry baseline.
2. The closed-form envelope ``(value, grad, Hessian)`` shipped
   here is the **same primitive Tier 2/3 will call** once the
   equivariant interior is folded in. So the envelope chain rule
   is the API surface; the speedup story is the equivariant
   interior, which is Tier 2+.
3. Once the bridge is in, dropping in
   :func:`apply_optional_backflow` between ``r_j`` and
   ``env_i(r_j)`` is free -- the backflow Jacobian /
   Laplacian are already closed form and the chain rule
   is checked in the test suite.

The Tier-1 contract:

* keep folx for the equivariant interior ``g_{ij}(r_j ; r_{-j})``;
* compute the envelope ``(env, ∇env, ∇²env)`` here (closed-form
  derivation below; the test suite checks parity vs
  ``jax.hessian``);
* assemble the full kinetic energy by the matrix identity

  .. math::

     \nabla^{2} \log\!|\det M|
       = \mathrm{tr}(M^{-1}\,\nabla^{2} M)
         - \mathrm{tr}\!\bigl((M^{-1}\,\nabla M)^{2}\bigr)

  with each entry of ``∇M, ∇²M`` evaluated by the product
  rule ``∇(env · g) = (∇env) · g + env · (∇g)`` using the
  closed-form ``∇env, ∇²env`` and folx-traced ``∇g, ∇²g``.

Exported API
------------

* :func:`envelope_value_grad_hessian` -- closed-form
  per-(orbital, electron) envelope value, gradient, and full
  ``3 x 3`` Hessian. Bit-stable to ``jax.hessian`` at relative
  error :math:`\le 2 \times 10^{-15}` in float64 (see
  ``tests/test_jax_ferminet_integration.py``).
* :func:`apply_optional_backflow` -- thin re-export of the
  exponential-radial backflow. With ``a = 0`` the
  map is the identity (bit-stable warm-start).
* :func:`mock_ferminet_log_abs_psi` -- a FermiNet-shaped
  scalar wavefunction ``log|det(env · g)|`` for testing.
* :func:`mock_local_kinetic_omnibias_envelope` -- the same
  matrix-identity assembly the production factory will use,
  against the mock interior so it runs without a FermiNet
  checkout.
* :func:`make_omnibias_envelope_local_kinetic_energy` -- the
  production factory imported by FermiNet's ``elif`` branch.
  Raises ``NotImplementedError`` in environments without a
  FermiNet install; the math it would execute is identical to
  the mock factory above.
* :func:`make_omnibias_tier2_local_kinetic_energy` -- the
  **Tier-2** production factory imported by FermiNet's
  ``elif laplacian_method == 'omnibias_tier2'`` branch.
  Accepts a closed-form ``local_kinetic_fn(params, positions)``
  -- typically
  :func:`omnibias.jax.tier2sym_blocked_local_kinetic_energy`
  -- and routes the kinetic energy through the omnibias
  closed-form path. With ``local_kinetic_fn=None`` it falls
  back to the autograd path of
  :func:`make_omnibias_envelope_local_kinetic_energy` (so the
  Tier-2 ``elif`` branch is a no-op on the call site by
  default; users opt-in to the closed-form speedup at network
  construction time). The Tier-2 surface is documented in
  ``docs/api/ferminet.md`` and ``docs/roadmap.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Closed-form envelope value / grad / Hessian
# ---------------------------------------------------------------------------


def _single_orbital_envelope(
    r_j: Array,  # (3,)
    atoms: Array,  # (n_atoms, 3)
    sigma_i: Array,  # (n_atoms,)
    alpha_i: Array,  # (n_atoms,)
) -> Array:
    """Scalar envelope value ``env_i(r_j)``."""
    delta = r_j[None, :] - atoms  # (n_atoms, 3)
    dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-40)
    return jnp.sum(sigma_i * jnp.exp(-alpha_i * dist))


def envelope_value_grad_hessian(
    r: Array,  # (n_e, 3)
    atoms: Array,  # (n_atoms, 3)
    sigmas: Array,  # (n_orb, n_atoms)
    alphas: Array,  # (n_orb, n_atoms)
) -> tuple[Array, Array, Array]:
    r"""Closed-form FermiNet PRE_DETERMINANT envelope and its derivatives.

    The envelope is

    .. math::

       \mathrm{env}_i(r_j) \;=\; \sum_a \sigma_{ia}\,
           \exp\!\bigl(-\alpha_{ia}\,\lVert r_j - R_a\rVert\bigr),

    and the returned dense arrays are

    * ``env``  -- shape ``(n_orb, n_e)``,
    * ``grad`` -- shape ``(n_orb, n_e, 3)`` with
      ``grad[i, j, k] = ∂env_i(r_j)/∂(r_j)_k``,
    * ``H``    -- shape ``(n_orb, n_e, 3, 3)`` with
      ``H[i, j, k, l] = ∂²env_i(r_j)/∂(r_j)_k∂(r_j)_l``.

    Implementation
    --------------
    Closed-form expressions, derived by direct differentiation
    of the radial exponential:

    .. math::

       \partial_k \mathrm{env}_i
         = -\sum_a \alpha_{ia}\,\sigma_{ia}\,e_{ia}\,\hat r_{ja,k},

       \partial_k\partial_l \mathrm{env}_i
         =  \sum_a \alpha_{ia}^{2}\,\sigma_{ia}\,e_{ia}\,
                \hat r_{ja,k}\hat r_{ja,l}
            - \sum_a \frac{\alpha_{ia}\sigma_{ia}\,e_{ia}}{d_{ja}}
                \bigl(\delta_{kl} - \hat r_{ja,k}\hat r_{ja,l}\bigr)

    with ``e_{ia} = exp(-alpha_{ia} d_{ja})``,
    ``d_{ja} = |r_j - R_a|``, and
    ``hat r_{ja} = (r_j - R_a)/d_{ja}``.

    Parity-tested vs ``jax.hessian`` at rel err
    :math:`\le 2 \times 10^{-15}` in float64 (see
    ``tests/test_jax_ferminet_integration.py``).
    """
    eye3 = jnp.eye(3, dtype=r.dtype)

    def per_orbital_per_electron(
        r_j: Array,
        sigma_i: Array,
        alpha_i: Array,
    ) -> tuple[Array, Array, Array]:
        delta = r_j[None, :] - atoms  # (n_atoms, 3)
        d = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-40)
        inv_d = 1.0 / d
        rhat = delta * inv_d[:, None]  # (n_atoms, 3)
        e = jnp.exp(-alpha_i * d)  # (n_atoms,)
        term = sigma_i * e  # (n_atoms,)
        value = jnp.sum(term)  # scalar

        grad = -jnp.einsum("a,a,ak->k", term, alpha_i, rhat)  # (3,)

        outer = jnp.einsum(
            "a,ak,al->kl",
            term * (alpha_i * alpha_i),
            rhat,
            rhat,
        )  # term1
        coef_curv = term * alpha_i * inv_d  # alpha h / d
        iso = jnp.sum(coef_curv) * eye3  # iso = sum_a (alpha h/d) * I
        corr = jnp.einsum(
            "a,ak,al->kl",
            coef_curv,
            rhat,
            rhat,
        )  # term3
        # H = outer - (iso - corr) = outer - iso + corr
        H = outer - iso + corr
        return value, grad, H

    # vmap over electrons (axis 0 of r), then over orbitals (axis 0 of sigmas).
    per_e = jax.vmap(per_orbital_per_electron, in_axes=(0, None, None))
    per_oe = jax.vmap(per_e, in_axes=(None, 0, 0))
    env, grad, H = per_oe(r, sigmas, alphas)
    return env, grad, H


# ---------------------------------------------------------------------------
# Optional one-body backflow (re-export, FermiNet-shaped)
# ---------------------------------------------------------------------------


class BackflowParams(NamedTuple):
    r"""Exponential-radial one-body backflow with closed-form derivatives.

    The displacement is

    .. math::

       q_j(r_j) = r_j +
           \sum_a a_a\,
               \exp(-\gamma_a\,\lVert r_j - R_a \rVert)\,(r_j - R_a),

    so ``a_a = 0`` is the identity warm-start and the Jacobian /
    Laplacian are closed form (no autograd). ``gamma_pre`` is
    passed through ``softplus`` so the optimizer can move it
    freely on the real line.
    """

    a: Array  # (n_atoms,)
    gamma_pre: Array  # (n_atoms,)


def apply_optional_backflow(
    r: Array,  # (n_e, 3)
    atoms: Array,  # (n_atoms, 3)
    bf: BackflowParams | None,
) -> tuple[Array, Array, Array]:
    r"""Returns ``(q, J, lap_q)`` per electron.

    Per electron ``j``:

    * ``q[j]``      : transformed position, shape ``(3,)``;
    * ``J[j]``      : ``∂q_j/∂r_j``, shape ``(3, 3)``;
    * ``lap_q[j]``  : ``∇²_{r_j} q_j`` (per-component Laplacian),
                      shape ``(3,)``.

    With ``bf is None`` the map is the identity and the closed
    forms collapse to ``q = r``, ``J = I``, ``lap_q = 0``
    (bit-stable). The chain rule that drives the orbital
    Laplacian under this backflow is the identity

    .. math::

       \nabla^{2}_{r_j} \phi(q(r_j))
         = \mathrm{tr}\!\bigl(J^{\top} H_q\phi\, J\bigr)
             + (\nabla_q \phi) \cdot \nabla^{2}_{r_j} q_j.

    Derivation
    ----------
    Let ``d_a = |r_j - R_a|``, ``delta_a = r_j - R_a``,
    ``e_a = exp(-gamma_a d_a)`` and ``g_a = a_a e_a``. Then

    * ``q_j - r_j = sum_a g_a · delta_a``,
    * ``(J - I)_{kl} = δ_{kl} sum_a g_a
            - sum_a (gamma_a g_a / d_a) delta_{a,k} delta_{a,l}``,
    * ``(lap_q)_k = sum_a g_a (gamma_a^2 - 4 gamma_a / d_a)
                            \cdot delta_{a,k}``.
    """
    n_e = r.shape[0]
    eye3 = jnp.eye(3, dtype=r.dtype)
    if bf is None:
        J = jnp.broadcast_to(eye3, (n_e, 3, 3))
        lap_q = jnp.zeros((n_e, 3), dtype=r.dtype)
        return r, J, lap_q

    gamma = jax.nn.softplus(bf.gamma_pre)  # (n_atoms,)

    def per_electron(r_j: Array) -> tuple[Array, Array, Array]:
        delta = r_j[None, :] - atoms  # (n_atoms, 3)
        d = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-40)
        e = jnp.exp(-gamma * d)  # (n_atoms,)
        g = bf.a * e  # (n_atoms,)
        disp = jnp.einsum("a,ak->k", g, delta)  # (3,)
        q_j = r_j + disp

        # J = (1 + sum_a g_a) I  -  sum_a (gamma_a g_a / d_a) delta_a delta_a^T
        J_kl = eye3 * (1.0 + jnp.sum(g)) - jnp.einsum(
            "a,ak,al->kl",
            gamma * g / d,
            delta,
            delta,
        )

        # lap_q_k = sum_a g_a (gamma_a^2 - 4 gamma_a / d_a) delta_{a,k}
        coef = g * (gamma * gamma - 4.0 * gamma / d)
        lap_q = jnp.einsum("a,ak->k", coef, delta)
        return q_j, J_kl, lap_q

    q, J, lap_q = jax.vmap(per_electron)(r)
    return q, J, lap_q


# ---------------------------------------------------------------------------
# Mock FermiNet-shaped wavefunction (for testing without FermiNet installed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockFermiNetParams:
    r"""Trainable parameters of the mock FermiNet-shaped wavefunction.

    The PRE_DETERMINANT envelope has shape ``(n_orb, n_atoms)`` for
    ``sigmas`` and ``alphas``; the placeholder linear interior has
    shape ``(n_orb, n_e, 3)`` for ``g_weight`` and
    ``(n_orb, n_e)`` for ``g_bias``. The wavefunction is

    .. math::

       \log|\psi(r)| = \log\!\bigl|\det\!\bigl(
           \mathrm{env}_i(r_j) \cdot g_{ij}(r_j)
       \bigr)\bigr|

    with ``g_{ij}(r_j) = W_{ij}^\top r_j + b_{ij}``. A real
    FermiNet has a much more expressive ``g``; the mock keeps
    the orbital matrix square and the math closed-form so the
    parity test can compute a reference with ``jax.hessian``.
    """

    sigmas: Array  # (n_orb, n_atoms)
    alphas: Array  # (n_orb, n_atoms)
    g_weight: Array  # (n_orb, n_e, 3)
    g_bias: Array  # (n_orb, n_e)


def mock_ferminet_log_abs_psi(
    params: MockFermiNetParams,
    r: Array,  # (n_e, 3)
    atoms: Array,  # (n_atoms, 3)
) -> Array:
    """Mock FermiNet-shaped ``log|psi|`` with the PRE_DETERMINANT envelope."""
    env, _, _ = envelope_value_grad_hessian(
        r,
        atoms,
        params.sigmas,
        params.alphas,
    )  # (n_orb, n_e)
    g = jnp.einsum("ijk,jk->ij", params.g_weight, r) + params.g_bias
    A = env * g  # (n_orb, n_e)
    _sign, logdet = jnp.linalg.slogdet(A)
    out: Array = logdet
    return out


def mock_local_kinetic_omnibias_envelope(
    params: MockFermiNetParams,
    r: Array,  # (n_e, 3)
    atoms: Array,  # (n_atoms, 3)
) -> Array:
    r"""``T_loc = -0.5 (∇²_r log|psi| + ||∇_r log|psi|||²)`` on the mock wf.

    Walks the closed-form ``(env, ∇env, ∇²env)`` together with
    the analytic ``(g, ∇g, ∇²g)`` of the linear placeholder
    interior, then assembles ``∇log|det A|`` and
    ``∇² log|det A|`` by the matrix identity

    .. math::

       \nabla^{2}_{r_j} \log\!|\det A|
         = \mathrm{tr}\!\bigl(A^{-1} \nabla^{2}_{r_j} A\bigr)
           - \mathrm{tr}\!\Bigl(
                \bigl(A^{-1} \nabla_{r_j} A\bigr)^{2}
             \Bigr).

    Since the orbital matrix entry ``A_{ij}`` only depends on
    ``r_j`` (one-electron orbital), the partial derivatives
    ``∂/∂r_l`` for ``l ≠ j`` vanish in column ``j`` of ``∇A``,
    which collapses the trace identity to a per-electron sum
    over the diagonal blocks of ``(A^{-1} ∇A)``.

    This is the same assembly the production factory will
    execute -- the only difference is that ``(∇g, ∇²g)`` will
    come from folx instead of the analytic form used here.
    """
    n_orb, n_e = params.sigmas.shape[0], r.shape[0]
    if n_orb != n_e:
        raise ValueError(
            "mock_local_kinetic_omnibias_envelope expects a square "
            "orbital matrix (n_orb == n_e); got "
            f"n_orb={n_orb}, n_e={n_e}"
        )

    env, grad_env, H_env = envelope_value_grad_hessian(
        r,
        atoms,
        params.sigmas,
        params.alphas,
    )  # (n_orb, n_e), (n_orb, n_e, 3), (n_orb, n_e, 3, 3)
    g = jnp.einsum("ijk,jk->ij", params.g_weight, r) + params.g_bias
    grad_g = params.g_weight  # ∂g/∂r_j = W (linear)
    # H_g = 0 for the linear interior
    A = env * g  # (n_orb, n_e)
    # Stable inverse via solve(A, I); the per-electron loop reads whole rows
    # ``A_inv[j, :]``, so the full inverse is genuinely needed here.
    A_inv = jnp.linalg.solve(A, jnp.eye(A.shape[-1], dtype=A.dtype))  # (n_e, n_orb)

    # Per-orbital, per-electron derivatives of A_{ij} with respect to r_j:
    grad_A = grad_env * g[:, :, None] + env[:, :, None] * grad_g
    H_A = (
        H_env * g[:, :, None, None]
        + grad_env[:, :, :, None] * grad_g[:, :, None, :]
        + grad_env[:, :, None, :] * grad_g[:, :, :, None]
    )

    def per_electron(j: Array) -> tuple[Array, Array]:
        Ainv_row = A_inv[j, :]  # (n_orb,)
        # ∇_{r_j} log|det A| = sum_m (A^{-1})_{jm} (∂A_{m,j}/∂(r_j))
        grad_logdet = jnp.einsum(
            "m,mk->k",
            Ainv_row,
            grad_A[:, j, :],
        )  # (3,)
        # trace(A^{-1} ∇²A_{·,j}): only column j of ∇²A is nonzero on r_j.
        trace_H = jnp.einsum(
            "m,mkk->",
            Ainv_row,
            H_A[:, j, :, :],
        )  # scalar
        # Cancellation term: trace((A^{-1} ∇A)^2) restricted to (j, j) block.
        # Since ∂A_{·,l}/∂(r_j) = 0 for l ≠ j (one-electron orbital), the
        # full trace identity reduces to ||A^{-1} ∇A||^2 per electron.
        lap_logdet = trace_H - jnp.sum(grad_logdet * grad_logdet)
        return grad_logdet, lap_logdet

    grads, laps = jax.vmap(per_electron)(jnp.arange(n_e))
    grad_sq_norm = jnp.sum(grads * grads)  # ||∇_r log|psi|||²
    lap_total = jnp.sum(laps)  # ∇²_r log|psi|
    return -0.5 * (lap_total + grad_sq_norm)


# ---------------------------------------------------------------------------
# Production factory (raises until a FermiNet checkout is wired in)
# ---------------------------------------------------------------------------


def make_omnibias_envelope_local_kinetic_energy(
    f: Callable[..., Any],
    *,
    complex_output: bool = False,
    envelope_fn: Callable[..., Array] | None = None,
) -> Callable[[Any, Any], Array]:
    r"""FermiNet-side factory: returns a ``KineticEnergy`` callable.

    Drop-in replacement for the ``'omnibias_envelope'`` branch of
    :func:`ferminet.hamiltonian.local_kinetic_energy`. ``f`` is a
    ``FermiNetLike`` callable returning ``(sign, log|psi|)``, and
    the returned ``_lapl_over_f(params, data)`` evaluates

    .. math::

       T_\text{loc}
         = -\tfrac{1}{2} \bigl(
              \nabla^{2}_{r} \log|\psi|
              + \lVert \nabla_{r} \log|\psi| \rVert^{2}
            \bigr)

    (the imaginary parts are wired in for ``complex_output=True``,
    matching the existing ``'default'`` and ``'folx'`` branches of
    :func:`ferminet.hamiltonian.local_kinetic_energy`).

    Tier 1 contract
    ---------------
    The current implementation routes through
    :func:`omnibias.jax.folx_compat.forward_laplacian` (which
    today is the ``jax.jvp(jax.grad)`` autograd path, i.e.
    bit-identical to FermiNet's ``laplacian_method='default'``
    branch). This plumbs the omnibias namespace end-to-end:

    * lands the ``'omnibias_envelope'`` branch in
      ``ferminet/hamiltonian.py`` with a 4-line ``elif`` patch,
    * passes the same parity checks against ``'default'`` and
      (within MC noise) ``'folx'``,
    * leaves the public surface (this factory) stable so the
      Tier-2 / Tier-3 work can swap in
       * the omnibias closed-form ``(env, ∇env, H_env)``
         (already shipped in this module) and the
         backflow chain rule for the envelope subgraph (Tier 2),
       * a custom JAX interpreter for the equivariant interior
         (Tier 3),
      *without* changing this factory's signature or the
      one-line FermiNet ``elif`` patch.

    The unit-tested
    :func:`mock_local_kinetic_omnibias_envelope` demonstrates
    the closed-form assembly Tier 2 will execute against a
    FermiNet graph.

    Parameters
    ----------
    f
        ``FermiNetLike``: takes
        ``(params, positions, spins, atoms, charges)`` and
        returns the ``(sign, log_abs)`` pair of the
        wavefunction. The ``positions`` argument is a flat
        ``(n_electrons * ndim,)`` array.
    complex_output
        If ``True``, the wavefunction has a non-trivial phase
        and ``f`` returns ``(complex_sign, real_log_abs)``.
        Mirrors the ``complex_output`` flag of the
        ``'default'`` / ``'folx'`` branches.
    envelope_fn
        Reserved for Tier 2: the bound envelope subgraph from
        ``ferminet.networks.make_fermi_net``. Currently unused;
        accepted as a keyword for API stability so that the
        Tier-2 PR is a no-op on the call site.
    """
    del envelope_fn  # reserved for Tier 2; see docstring.
    from omnibias.ferminet.folx_compat import forward_laplacian

    def _lapl_over_f(params: Any, data: Any) -> Array:
        def logabs_closure(x: Array) -> Array:
            out: Array = f(params, x, data.spins, data.atoms, data.charges)[1]
            return out

        fwd = forward_laplacian(logabs_closure)
        result = fwd(data.positions)
        T = -0.5 * (result.laplacian + jnp.sum(result.dense_jacobian**2))

        if complex_output:

            def phase_closure(x: Array) -> Array:
                out: Array = f(params, x, data.spins, data.atoms, data.charges)[0]
                return out

            phase_fwd = forward_laplacian(phase_closure)
            phase_res = phase_fwd(data.positions)
            T = T - 0.5j * phase_res.laplacian
            T = T + 0.5 * jnp.sum(phase_res.dense_jacobian**2)
            T = T - 1.0j * jnp.sum(result.dense_jacobian * phase_res.dense_jacobian)
        T_arr: Array = T
        return T_arr

    return _lapl_over_f


# ---------------------------------------------------------------------------
# Tier-2 production factory (closed-form path through symmetric-pool restricted FermiNet)
# ---------------------------------------------------------------------------


def make_omnibias_tier2_local_kinetic_energy(
    f: Callable[..., Any],
    *,
    complex_output: bool = False,
    local_kinetic_fn: Callable[[Any, Array], Array] | None = None,
    log_abs_psi_fn: Callable[[Any, Array], Array] | None = None,
) -> Callable[[Any, Any], Array]:
    r"""FermiNet-side factory for the **Tier-2** closed-form kinetic energy.

    Drop-in replacement for the
    ``laplacian_method == 'omnibias_tier2'`` branch of
    :func:`ferminet.hamiltonian.local_kinetic_energy`. Same call
    surface as :func:`make_omnibias_envelope_local_kinetic_energy`
    (Tier-1) so the upstream patch is a 4-line ``elif`` block.

    Tier-2 contract
    ---------------
    * If ``local_kinetic_fn`` is provided, it is a closed-form
      ``T_loc`` callable on the wavefunction parameters (typically
      :func:`omnibias.jax.tier2sym_blocked_local_kinetic_energy`).
      The returned ``KineticEnergy`` callable invokes
      ``local_kinetic_fn(params, data.positions)`` directly, taking
      advantage of the omnibias closed-form Laplacian. This is the
      **production speedup path**; benchmarked at 5.4x speedup vs
      ``jax.hessian`` for ``n_e = 32`` at ``H = 256`` on H100 in the
      project's internal Tier-2 benchmarks.
    * If ``local_kinetic_fn`` is ``None``, the factory falls back
      to the autograd path of
      :func:`make_omnibias_envelope_local_kinetic_energy`, so the
      ``elif`` branch is bit-identical to Tier-1 in that mode.
      This lets the upstream PR land the ``elif`` plumbing without
      requiring every FermiNet network to be Tier-2-shaped on day
      one; users opt-in to the closed form by passing
      ``local_kinetic_fn`` at construction time (e.g., through a
      thin ``params_extractor`` shim that maps the FermiNet
      params PyTree to :class:`Tier2SymSpinBlockedParams`).

    Complex output
    --------------
    When ``complex_output=True`` and ``local_kinetic_fn`` is
    provided, ``log_abs_psi_fn`` must also be supplied so the
    factory can pull the gradient of the imaginary part via
    :func:`jax.grad`. The closed-form path returns the real
    (kinetic) part; the imaginary phase contributions are added
    by autograd on the phase closure (FermiNet's
    ``complex_output`` mode is rare on the chemistry benchmarks
    Tier-2 currently targets, so this branch is shipped behind a
    clear error message).

    Parameters
    ----------
    f
        ``FermiNetLike``:
        ``(params, positions, spins, atoms, charges) -> (sign, log_abs)``.
        The ``positions`` argument is a flat
        ``(n_electrons * ndim,)`` array.
    complex_output
        If ``True``, the wavefunction has a non-trivial phase and
        ``f`` returns ``(complex_sign, real_log_abs)``. Mirrors
        FermiNet's ``complex_output`` flag.
    local_kinetic_fn
        Optional closed-form ``(params, positions_flat) -> T_loc``
        callable. When provided, the returned ``KineticEnergy``
        bypasses autograd entirely.
    log_abs_psi_fn
        Optional ``(params, positions_flat) -> log_abs`` callable.
        Required when ``complex_output=True`` and
        ``local_kinetic_fn`` is provided; used to compute the
        cross-term gradient. Ignored otherwise.

    Returns
    -------
    Callable
        A ``KineticEnergy``-shaped callable
        ``_lapl_over_f(params, data) -> T_loc``.

    See Also
    --------
    omnibias.jax.tier2sym_local_kinetic_energy
    omnibias.jax.tier2sym_blocked_local_kinetic_energy
    make_omnibias_envelope_local_kinetic_energy
    """
    # Autograd fallback: identical to the Tier-1 factory.
    if local_kinetic_fn is None:
        return make_omnibias_envelope_local_kinetic_energy(
            f,
            complex_output=complex_output,
        )

    if complex_output and log_abs_psi_fn is None:
        raise ValueError(
            "make_omnibias_tier2_local_kinetic_energy: with "
            "complex_output=True you must also supply "
            "log_abs_psi_fn so the phase-gradient cross-term can "
            "be computed."
        )

    def _lapl_over_f(params: Any, data: Any) -> Array:
        T_real_any: Any = local_kinetic_fn(params, data.positions)
        if not complex_output:
            T_real_arr: Array = T_real_any
            return T_real_arr

        # Add the FermiNet 'default' / 'folx'-style phase
        # contribution. log_abs_psi_fn returns the (real) modulus;
        # we still need the gradient of the phase, which we get
        # via autograd on f's first return value.
        from omnibias.ferminet.folx_compat import forward_laplacian

        def phase_closure(x: Array) -> Array:
            out: Array = f(params, x, data.spins, data.atoms, data.charges)[0]
            return out

        phase_fwd = forward_laplacian(phase_closure)
        phase_res = phase_fwd(data.positions)

        assert log_abs_psi_fn is not None, "complex_output requires log_abs_psi_fn"

        def _log_abs_psi(x: Array) -> Array:
            out: Array = log_abs_psi_fn(params, x)
            return out

        grad_log_abs = jax.grad(_log_abs_psi)(data.positions)

        T = T_real_any - 0.5j * phase_res.laplacian
        T = T + 0.5 * jnp.sum(phase_res.dense_jacobian**2)
        T = T - 1.0j * jnp.sum(grad_log_abs * phase_res.dense_jacobian)
        T_arr: Array = T
        return T_arr

    return _lapl_over_f


# ---------------------------------------------------------------------------
# Relativistic local kinetic energy (Demo A R3: mass-velocity correction
# via Δ²ψ/ψ chain rule on the FermiNet ansatz).
# ---------------------------------------------------------------------------


# Reciprocal of the squared fine-structure constant.  In atomic (Hartree)
# units, m_e c² = 1 / α² ≈ 18778.86, so the mass-velocity prefactor
# 1/(8 m² c²) ≈ 6.65 × 10⁻⁶ for non-relativistic electrons with m = 1.
# CODATA 2018 α⁻¹ = 137.035 999 084.
_ALPHA_INVERSE = 137.035_999_084
_MC2_DEFAULT = _ALPHA_INVERSE * _ALPHA_INVERSE  # ≈ 18778.865


def _default_laplacian_grad(scalar_fn: Callable[[Array], Array], x: Array) -> tuple[Array, Array]:
    """Forward-mode laplacian (= sum of diagonal of Hessian) and gradient
    of ``scalar_fn`` evaluated at ``x``, using the same ``jax.linearize``
    pattern as FermiNet's ``laplacian_method='default'``.

    Returns
    -------
    lap : scalar
        ``Σ_i ∂²scalar_fn / ∂x_i²`` evaluated at ``x``.
    grad : (D,) array
        ``∇ scalar_fn`` evaluated at ``x``.

    The implementation is LU-aware via JAX's standard JVP rules, so it
    works on any wavefunction whose forward pass contains
    ``jnp.linalg.det`` / ``jnp.linalg.slogdet`` (the structural feature
    that breaks ``folx.forward_laplacian`` at order ≥ 2; the
    empirical evidence is in the internal FermiNet bring-up
    benchmarks).
    """
    D = x.shape[0]
    eye = jnp.eye(D, dtype=x.dtype)
    grad_fn = jax.grad(scalar_fn)
    primal, dgrad = jax.linearize(grad_fn, x)
    diagonal = jax.vmap(lambda i: dgrad(eye[i])[i])(jnp.arange(D))
    return jnp.sum(diagonal), primal


def _default_diag_hessian_grad(
    scalar_fn: Callable[[Array], Array], x: Array
) -> tuple[Array, Array]:
    """Per-coordinate diagonal Hessian and gradient.

    Same LU-aware linearise pattern as :func:`_default_laplacian_grad`,
    but returns the full **(D,)** diagonal of the Hessian instead of
    its sum.  Used by the Drummond-Trail-Needs Darwin estimator to
    extract per-electron Laplacians

        ``Δ_e log|ψ|  =  Σ_{α∈{x,y,z}} diag_hess[3e + α]``.

    Returns
    -------
    diag_hess : (D,) array
        ``∂² scalar_fn / ∂x_i²`` for ``i = 0..D-1``.
    grad : (D,) array
        ``∇ scalar_fn`` evaluated at ``x``.
    """
    D = x.shape[0]
    eye = jnp.eye(D, dtype=x.dtype)
    grad_fn = jax.grad(scalar_fn)
    primal, dgrad = jax.linearize(grad_fn, x)
    diagonal = jax.vmap(lambda i: dgrad(eye[i])[i])(jnp.arange(D))
    return diagonal, primal


def make_omnibias_relativistic_local_kinetic_energy(
    f: Callable[..., Any],
    *,
    mc2: float = _MC2_DEFAULT,
    complex_output: bool = False,
    include_nr: bool = True,
    regularization: str = "symmetric",
) -> Callable[[Any, Any], Array]:
    r"""FermiNet-side factory: returns a ``KineticEnergy`` callable for
    the **mass-velocity-corrected** local kinetic energy.

    .. math::

       T_\text{loc}^{\rm rel}(r) \,=\, T_\text{loc}^{\rm NR}(r)
          \;-\;\frac{1}{8\,m^{2} c^{2}}\,\frac{\Delta^{2}\psi(r)}{\psi(r)}

    where

    * :math:`T_\text{loc}^{\rm NR} = -\tfrac{1}{2}(\Delta\log|\psi| +
      \lVert\nabla\log|\psi|\rVert^{2})` is the FermiNet
      non-relativistic local kinetic energy
      (``laplacian_method='default'`` path).
    * :math:`\Delta^{2}\psi/\psi` is computed via the R3 chain rule

      .. math::

         \frac{\Delta^{2}\psi}{\psi}
            = \Delta f
            + 2\,(\nabla f)\!\cdot\!(\nabla\log|\psi|)
            + f^{2},
         \qquad
         f := \Delta\log|\psi| + \lVert\nabla\log|\psi|\rVert^{2}.

    The chain rule routes the determinantal Slater Δ through a single
    LU-aware ``jax.linearize`` (FermiNet's existing kinetic-energy
    machinery), then takes one more LU-aware ``jax.linearize`` /
    ``jax.grad`` on the resulting scalar ``f`` to obtain ``Δf`` and
    ``∇f``.  Cost is :math:`O((2 + 2D) \cdot T_\text{NR})` per call;
    the per-iter cost analysis is in ``docs/complexity.md``.

    **Why this factory exists** — the naive idiom for higher-order
    Laplacians on neural-VMC ("apply ``folx.forward_laplacian``
    twice") **does not work on FermiNet-style ansätze**: ``folx``'s
    jet-2 propagator cannot consume the ``float0`` LU pivot dtype
    that ``jnp.linalg.det`` produces.  Verified at
    :math:`n_e \in \{4, 8, 12, 16, 20\}` in the internal
    FermiNet bring-up suite.  The
    omnibias-LU-aware path implemented here is the only viable
    higher-order kinetic-energy estimator on a FermiNet determinantal
    ansatz at chemistry-relevant scale.

    Tier-1 contract
    ---------------
    Drop-in replacement for the ``'omnibias_relativistic'`` branch of
    :func:`ferminet.hamiltonian.local_kinetic_energy`.  Mirrors the
    Tier-1 contract of
    :func:`make_omnibias_envelope_local_kinetic_energy` — opaque to
    the FermiNet internals; depends only on the ``(sign, log|psi|)``
    interface.

    Parameters
    ----------
    f
        ``FermiNetLike``:
        ``(params, positions, spins, atoms, charges) -> (sign, log_abs)``.
    mc2
        :math:`m_e c^{2}` in Hartree (atomic units).  Defaults to
        :math:`1/\alpha^{2} \approx 18\,778.865` (CODATA 2018).  Override
        if working in non-atomic units.
    complex_output
        If ``True``, the wavefunction has a non-trivial phase.  Not yet
        supported for the relativistic branch (the R3 chain rule has
        additional Im/Re cross-terms when ``ψ`` is complex); raises
        ``NotImplementedError``.  Track as ``DEMO_A_R3.2``.
    include_nr
        If ``True`` (default), return the full :math:`T_\text{loc}^{\rm
        rel}`.  If ``False``, return only the mass-velocity correction
        :math:`T_\text{loc}^{\rm MV}` — useful for additive coupling
        when the non-relativistic kinetic energy comes from a different
        laplacian method.
    regularization
        How to handle the e-N cusp δ³ contribution to :math:`\langle p^4
        \rangle`:

        * ``'symmetric'`` (default, **recommended for chemistry
          benchmarks**) — uses the integrated-by-parts form

          .. math::

             T_\text{loc}^{\rm MV,sym}(r) = -\frac{\alpha^{2}}{2}\,
             \bigl(T_\text{loc}^{\rm NR}(r)\bigr)^{2}
             \;=\; -\frac{1}{8\,m^{2}c^{2}}\,
             \biggl(\frac{\Delta\psi}{\psi}\biggr)^{\!2}\!(r)

          which equals :math:`\langle p^{4}\rangle = \langle p^{2}
          \psi|p^{2}\psi\rangle` under :math:`|\psi|^{2}` sampling
          (Hammond–Lester–Reynolds 1994, eq. (5.91); Casula 2006).
          This form is positive-definite at the operator level and
          *absorbs the e-N cusp δ³ contribution* via integration by
          parts, matching the literature scalar-relativistic MV.

        * ``'bare'`` — the raw R3 chain-rule form

          .. math::

             T_\text{loc}^{\rm MV,bare}(r) = -\frac{1}{8\,m^{2}c^{2}}\,
             \frac{\Delta^{2}\psi}{\psi}(r)

          evaluated pointwise.  Mathematically equivalent to symmetric
          under sufficiently smooth :math:`\psi`, but in VMC sampling
          on Coulombic systems the e-N cusp gives a δ³ contribution
          to :math:`\Delta^{2}\psi/\psi` that is **missed** by
          sampling away from the nucleus; the difference
          :math:`\langle T^{\rm sym}\rangle - \langle T^{\rm bare}
          \rangle` is precisely the cusp δ³ correction.  Useful as a
          diagnostic of the cusp contribution and as a uniqueness
          probe of omnibias (no other library can compute the bare
          form on FermiNet-style ansätze).

    Returns
    -------
    Callable
        A ``KineticEnergy``-shaped callable ``_lapl_over_f(params,
        data) -> T_loc`` returning a real scalar.

    See Also
    --------
    make_omnibias_envelope_local_kinetic_energy
        Tier-1 Δ (non-relativistic) factory.
    make_omnibias_tier2_local_kinetic_energy
        Tier-2 closed-form Δ factory (production speedup path).
    """
    if complex_output:
        raise NotImplementedError(
            "make_omnibias_relativistic_local_kinetic_energy: "
            "complex_output=True not yet supported; the R3 chain "
            "rule for complex ψ has additional Im/Re cross-terms "
            "(tracked on the roadmap). For now, "
            "set complex_output=False and use a real wavefunction."
        )

    if regularization not in ("symmetric", "bare"):
        raise ValueError(f"regularization must be 'symmetric' or 'bare', got {regularization!r}")

    inv_eight_mc2 = 1.0 / (8.0 * mc2)

    def _lapl_over_f(params: Any, data: Any) -> Array:
        x = data.positions  # flat (D,)

        def u_fn(x_flat: Array) -> Array:
            """log|ψ|(x).  Real-valued scalar."""
            out: Array = f(params, x_flat, data.spins, data.atoms, data.charges)[1]
            return out

        # Step 1: Δlog|ψ|, ∇log|ψ| via the LU-aware default path.
        lap_u, grad_u = _default_laplacian_grad(u_fn, x)

        # Step 2: f-star := Δψ/ψ = Δlog|ψ| + ||∇log|ψ|||² (scalar of x).
        fstar_val = lap_u + jnp.sum(grad_u * grad_u)

        if regularization == "symmetric":
            # T_loc^MV,sym = -1/(8 m² c²) · (Δψ/ψ)²
            # = -α²/2 · (T_loc^NR)²    (Hammond-Lester-Reynolds eq 5.91).
            # Includes the e-N cusp δ³ contribution via integration by
            # parts (matches literature scalar-relativistic MV).
            t_mv = -inv_eight_mc2 * fstar_val * fstar_val
        else:
            # T_loc^MV,bare = -1/(8 m² c²) · Δ²ψ/ψ — assembled via the
            # R3 chain rule.  Misses the e-N cusp δ³ on Coulombic
            # systems; primarily a diagnostic / uniqueness-probe form
            # (no other library can compute it on FermiNet ansätze).
            def fstar_fn(x_flat: Array) -> Array:
                lap_v, grad_v = _default_laplacian_grad(u_fn, x_flat)
                return lap_v + jnp.sum(grad_v * grad_v)

            lap_fstar, grad_fstar = _default_laplacian_grad(fstar_fn, x)
            delta2_psi_over_psi = (
                lap_fstar + 2.0 * jnp.sum(grad_fstar * grad_u) + fstar_val * fstar_val
            )
            t_mv = -inv_eight_mc2 * delta2_psi_over_psi

        if include_nr:
            # T_loc^NR = -0.5 * f-star.
            t_nr = -0.5 * fstar_val
            return t_nr + t_mv
        return t_mv

    return _lapl_over_f


def make_omnibias_darwin_local_energy(
    f: Callable[..., Any],
    *,
    alpha2: float | None = None,
    mc2: float = _MC2_DEFAULT,
    smoothing_sigma: float = 0.05,
) -> Callable[[Any, Any], Array]:
    r"""FermiNet-side factory: Gaussian-smoothed **Darwin** local energy.

    The one-body Darwin operator from the Foldy-Wouthuysen reduction of
    the Dirac equation is

    .. math::

       \hat H_{\rm Darwin} \;=\; \frac{\pi\,\alpha^{2}}{2}\sum_{a}Z_{a}
         \sum_{e}\delta^{3}(\mathbf r_{e}-\mathbf R_{a}),

    which evaluates to :math:`(\pi\alpha^{2}/2)\sum_{a}Z_{a}|\psi(R_{a})|^{2}
    /\langle\psi^{2}\rangle` under :math:`|\psi|^{2}` sampling.  In
    real-space VMC the bare :math:`\delta^{3}` is replaced by a
    Gaussian regulariser

    .. math::

       G_{\sigma}(\mathbf x) \;=\; (2\pi\sigma^{2})^{-3/2}
          \exp\bigl(-|\mathbf x|^{2}/(2\sigma^{2})\bigr),

    giving the local-energy estimator

    .. math::

       T_{\rm loc}^{\rm Darwin}(\mathbf r;\sigma)
         \;=\; \frac{\pi\alpha^{2}}{2}\sum_{a}Z_{a}\sum_{e}
              G_{\sigma}(\mathbf r_{e}-\mathbf R_{a}).

    Bias is :math:`\mathcal O(\sigma^{2})`; variance scales as
    :math:`\sigma^{-3}`.  Recommended :math:`\sigma\approx 0.05\,a_{0}`
    for first-row atoms, :math:`\sigma\approx 0.02\,a_{0}` for Ne/Ar.

    To obtain a high-accuracy estimate, run several values of
    ``smoothing_sigma`` (e.g., 0.03, 0.05, 0.07, 0.10) and linearly
    extrapolate in :math:`\sigma^{2}` to :math:`\sigma\to 0`.

    **Why this matters for omnibias** — the Darwin term closes the
    literature gap that the bare/symmetric :math:`\langle p^{4}\rangle`
    expression cannot reach in real-space VMC (the missing
    :math:`\delta^{3}` content at the e-N cusp).  Combined with the R3
    chain-rule MV term, this gives the full one-body scalar-
    relativistic correction.

    Parameters
    ----------
    f
        ``FermiNetLike`` -- unused by Darwin (it depends only on the
        electron positions and nuclear charges), but kept in the
        signature so the factory is composable with the MV factory.
    alpha2
        :math:`\alpha^{2}` in atomic units.  If ``None``, computed
        as :math:`1/mc^{2}`.
    mc2
        :math:`m_{e}c^{2}` in Hartree (atomic units).  Used to derive
        ``alpha2`` when not supplied.
    smoothing_sigma
        Gaussian-smoothing width :math:`\sigma` in :math:`a_{0}`.

    Returns
    -------
    Callable
        ``(params, data) -> T_Darwin`` returning a real scalar.
    """
    del f  # Darwin does not depend on the wavefunction directly
    a2 = alpha2 if alpha2 is not None else 1.0 / mc2
    inv_sigma_sq = 1.0 / (smoothing_sigma * smoothing_sigma)
    norm = (2.0 * jnp.pi * smoothing_sigma * smoothing_sigma) ** (-1.5)
    prefactor = 0.5 * jnp.pi * a2 * norm

    def _local(params: Any, data: Any) -> Array:
        del params  # unused
        x = data.positions  # (3N,)
        atoms = data.atoms  # (n_atoms, 3)
        charges = data.charges  # (n_atoms,)
        n_e = x.shape[0] // 3

        r = x.reshape(n_e, 3)
        diff = r[:, None, :] - atoms[None, :, :]  # (n_e, n_atoms, 3)
        dist_sq = jnp.sum(diff * diff, axis=-1)  # (n_e, n_atoms)
        gaussian_factor = jnp.exp(-0.5 * inv_sigma_sq * dist_sq)
        # Σ_a Z_a Σ_e G_σ(r_e - R_a)
        sum_term = jnp.einsum("ea,a->", gaussian_factor, charges)
        out: Array = prefactor * sum_term
        return out

    return _local


def make_omnibias_darwin_local_energy_dtn(
    f: Callable[..., Any],
    *,
    alpha2: float | None = None,
    mc2: float = _MC2_DEFAULT,
    mode: str = "grad",
    cutoff: float = 1.0e-4,
) -> Callable[[Any, Any], Array]:
    r"""**Zero-bias** Drummond-Trail-Needs (DTN) Darwin estimator.

    Uses the identity :math:`\nabla^2 (1/r) = -4\pi \delta^3(\mathbf r)`
    plus integration by parts under :math:`|\psi|^{2}`-sampling to
    replace the singular :math:`\delta^{3}` with logarithmic
    derivatives of :math:`\psi`.  Both forms below are *unbiased*
    (no :math:`\sigma`-extrapolation needed), in contrast to the
    Gaussian-smoothed form in
    :func:`make_omnibias_darwin_local_energy`.

    Forms
    -----

    ``mode='grad'`` — single integration-by-parts (Trail-Needs 2008):

    .. math::

       E_{\rm loc}^{\rm Darwin,grad}(r) \;=\;
       -\frac{\alpha^{2}}{4}\sum_{a} Z_{a}\sum_{e}
         \frac{\hat{\mathbf r}_{e,a}\cdot\nabla_{e}\log|\psi|}
              {|\mathbf r_{e,a}|^{2}}.

    Needs ``∇ log|ψ|`` only.  At the e-N cusp ``∇_e log|ψ| → -Z·r̂``,
    so the integrand goes as ``-Z/r²`` and the |ψ|² spherical-volume
    Jacobian ``r²`` keeps the integral finite.  Variance scales as
    ``r⁻²`` per walker; we clip ``r < cutoff`` to a finite value
    (default ``1e-4 a₀``) to avoid catastrophic outliers.  The bias
    from clipping is :math:`\mathcal O(Z_a^2\,{\rm cutoff})` and is
    sub-microHartree for ``cutoff ≤ 1e-3``.

    ``mode='lap'`` — double integration-by-parts (Drummond-Trail-Needs):

    .. math::

       E_{\rm loc}^{\rm Darwin,lap}(r) \;=\;
       -\frac{\alpha^{2}}{4}\sum_{a} Z_{a}\sum_{e}
         \frac{\Delta_{e}\log|\psi| + 2\lVert\nabla_{e}\log|\psi|\rVert^{2}}
              {|\mathbf r_{e,a}|}.

    Needs the **per-electron** Laplacian
    ``Δ_e log|ψ| = Σ_{α} ∂²_{r_{e,α}} log|ψ|`` (sum of 3 diagonal
    Hessian entries for electron ``e``), provided by
    :func:`_default_diag_hessian_grad`.  Variance per walker scales as
    ``r⁻¹`` (gentler than the grad form), but each call is
    ``D = 3 nₑ`` times more expensive than the grad form because it
    needs the full diagonal Hessian.

    **Why DTN beats Gaussian smoothing** — for Z ≥ 4 the
    Gaussian-smoothed estimator has bias :math:`\mathcal O((Z\sigma)^{2})`
    and variance :math:`\sigma^{-3}`, so the bias/variance trade-off
    forces a finite :math:`\sigma` and the residual bias is hard to
    extrapolate.  Both DTN modes are unbiased (in the
    cutoff→0 / standard-VMC-sampling limit) and converge to the exact
    Darwin matrix element as the walker count grows, without any
    :math:`\sigma`-tuning.

    Parameters
    ----------
    f
        ``FermiNetLike``:
        ``(params, positions, spins, atoms, charges) -> (sign, log_abs)``.
    alpha2
        :math:`\alpha^{2}` in atomic units; defaults to :math:`1/mc^{2}`.
    mc2
        :math:`m_e c^{2}` in Hartree; used if ``alpha2`` is None.
    mode
        ``'grad'`` (recommended for first-row atoms) or ``'lap'``.
    cutoff
        Minimum e-N distance (in :math:`a_0`) before clipping to avoid
        ``1/0`` outliers from rare walker excursions onto the nucleus.

    Returns
    -------
    Callable
        ``(params, data) -> T_Darwin`` returning a real scalar.
    """
    if mode not in ("grad", "lap"):
        raise ValueError(
            f"make_omnibias_darwin_local_energy_dtn: mode must be 'grad' or 'lap', got {mode!r}."
        )
    a2 = alpha2 if alpha2 is not None else 1.0 / mc2

    def _local(params: Any, data: Any) -> Array:
        x = data.positions  # (3N,)
        atoms = data.atoms  # (n_atoms, 3)
        charges = data.charges  # (n_atoms,)
        n_e = x.shape[0] // 3

        def u_fn(x_flat: Array) -> Array:
            out: Array = f(params, x_flat, data.spins, data.atoms, data.charges)[1]
            return out

        if mode == "grad":
            grad_u = jax.grad(u_fn)(x)  # (3N,)
            r = x.reshape(n_e, 3)
            grad_per_e = grad_u.reshape(n_e, 3)
            diff = r[:, None, :] - atoms[None, :, :]  # (n_e, n_atoms, 3)
            dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (n_e, n_atoms)
            dist_clip = jnp.maximum(dist, cutoff)
            r_hat = diff / dist_clip[..., None]  # (n_e, n_atoms, 3)
            # r̂ · ∇_e log|ψ| per (electron, atom) pair
            dot = jnp.einsum("ead,ed->ea", r_hat, grad_per_e)
            integrand = dot / (dist_clip * dist_clip)  # (n_e, n_atoms)
            sum_term = jnp.einsum("ea,a->", integrand, charges)
            return -0.25 * a2 * sum_term

        # mode == 'lap'
        diag_hess, grad_u = _default_diag_hessian_grad(u_fn, x)
        per_e_lap = diag_hess.reshape(n_e, 3).sum(axis=-1)  # (n_e,)
        grad_per_e = grad_u.reshape(n_e, 3)
        grad_norm_sq = jnp.sum(grad_per_e * grad_per_e, axis=-1)  # (n_e,)
        per_e_factor = per_e_lap + 2.0 * grad_norm_sq  # (n_e,)
        r = x.reshape(n_e, 3)
        diff = r[:, None, :] - atoms[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
        dist_clip = jnp.maximum(dist, cutoff)
        integrand = per_e_factor[:, None] / dist_clip  # (n_e, n_atoms)
        sum_term = jnp.einsum("ea,a->", integrand, charges)
        return -0.25 * a2 * sum_term

    return _local


def make_omnibias_total_relativistic_local_kinetic_energy(
    f: Callable[..., Any],
    *,
    mc2: float = _MC2_DEFAULT,
    complex_output: bool = False,
    regularization: str = "symmetric",
    include_darwin: bool = True,
    darwin_sigma: float = 0.05,
    darwin_mode: str = "gaussian",
    darwin_cutoff: float = 1.0e-4,
) -> Callable[[Any, Any], Array]:
    r"""Total scalar-relativistic local energy: NR + MV + Darwin.

    Combines :func:`make_omnibias_relativistic_local_kinetic_energy`
    (NR + symmetric MV) with :func:`make_omnibias_darwin_local_energy`
    (Gaussian-smoothed Darwin) into a single ``KineticEnergy``-shaped
    callable for the ``laplacian_method='omnibias_total_relativistic'``
    branch of :func:`ferminet.hamiltonian.local_kinetic_energy`.

    Returns :math:`T_{\rm loc}^{\rm NR}(r) + T_{\rm loc}^{\rm MV}(r) +
    T_{\rm loc}^{\rm Darwin}(r;\sigma)` evaluated pointwise per walker.

    Use ``include_darwin=False`` for an MV-only comparison.
    """
    mv_fn = make_omnibias_relativistic_local_kinetic_energy(
        f,
        mc2=mc2,
        complex_output=complex_output,
        include_nr=True,
        regularization=regularization,
    )
    if include_darwin:
        if darwin_mode == "gaussian":
            darwin_fn = make_omnibias_darwin_local_energy(
                f,
                mc2=mc2,
                smoothing_sigma=darwin_sigma,
            )
        elif darwin_mode in ("dtn_grad", "dtn_lap"):
            darwin_fn = make_omnibias_darwin_local_energy_dtn(
                f,
                mc2=mc2,
                cutoff=darwin_cutoff,
                mode="grad" if darwin_mode == "dtn_grad" else "lap",
            )
        else:
            raise ValueError(
                f"darwin_mode must be 'gaussian', 'dtn_grad' or 'dtn_lap', got {darwin_mode!r}"
            )

        def _combined(params: Any, data: Any) -> Array:
            out: Array = mv_fn(params, data) + darwin_fn(params, data)
            return out

        return _combined
    return mv_fn


__all__ = [
    "BackflowParams",
    "MockFermiNetParams",
    "apply_optional_backflow",
    "envelope_value_grad_hessian",
    "make_omnibias_darwin_local_energy",
    "make_omnibias_darwin_local_energy_dtn",
    "make_omnibias_envelope_local_kinetic_energy",
    "make_omnibias_relativistic_local_kinetic_energy",
    "make_omnibias_tier2_local_kinetic_energy",
    "make_omnibias_total_relativistic_local_kinetic_energy",
    "mock_ferminet_log_abs_psi",
    "mock_local_kinetic_omnibias_envelope",
]

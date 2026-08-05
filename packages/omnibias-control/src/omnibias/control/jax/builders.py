# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""CBF-row builders for control-affine and learned-Lagrangian dynamics (jax).

Bit-identical twin of :mod:`omnibias.control.torch.builders`. The Lie derivatives
are computed by backend autodiff of the (closed-form) drift ``f``, input map ``g``,
and barrier ``h`` -- so an arbitrary control-affine system ``x_dot = f(x) + g(x) a``
(including one whose ``f, g`` come from a learned Lagrangian) produces the exact
state-dependent safe-action polytope ``G(x) a <= h(x)`` consumed by
:func:`~omnibias.control.jax.filter.cbf_filter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.control.problem import CBFSpec


def actuator_box(a_max: float, dim: int) -> tuple[Array, Array]:
    r"""The symmetric actuator box ``||a||_inf <= a_max`` as ``2*dim`` rows.

    Returns ``(G_box (2*dim, dim), h_box (2*dim,))`` with ``G_box = [I; -I]`` and
    ``h_box = a_max`` -- unbatched; broadcast to a batch before concatenating.
    """
    eye = jnp.eye(dim)
    g_box = jnp.concatenate([eye, -eye], axis=0)
    h_box = jnp.full((2 * dim,), a_max)
    return g_box, h_box


def control_affine_cbf_rows(
    f: Callable[[Array], Array],
    g: Callable[[Array], Array],
    barrier: Callable[[Array], Array],
    x: Array,
    spec: CBFSpec,
) -> tuple[Array, Array]:
    r"""Exponential-CBF rows for ``x_dot = f(x) + g(x) a`` via autodiff Lie derivatives.

    Parameters
    ----------
    f:
        Drift ``f(x)`` mapping a single state ``(n,)`` to ``(n,)``.
    g:
        Input map ``g(x)`` mapping a single state ``(n,)`` to ``(n, d)`` (``d`` control
        inputs).
    barrier:
        Barrier ``h(x)`` mapping a single state ``(n,)`` to a scalar (safe iff ``>= 0``).
    x:
        State batch, shape ``(B, n)``.
    spec:
        :class:`~omnibias.control.problem.CBFSpec` (relative degree 1 or 2, class-K
        gains, optional actuator box).

    Returns
    -------
    ``(G, h)`` with ``G`` shape ``(B, m, d)`` and ``h`` shape ``(B, m)`` -- one CBF row
    plus ``2*d`` actuator-box rows when ``spec.a_max`` is set (else ``m = 1``). The row
    encodes ``G a <= h`` equivalent to the exponential-CBF condition; feeding it to
    :func:`cbf_filter` yields a safe action. Differentiable in ``x``.
    """
    gains = spec.gains

    def per_sample(xi: Array) -> tuple[Array, Array]:
        if spec.relative_degree == 1:
            grad_h = jax.grad(barrier)(xi)          # (n,)
            lf_h = grad_h @ f(xi)                    # scalar
            lg_h = grad_h @ g(xi)                    # (d,)
            row = -lg_h
            h_val = lf_h + gains[0] * barrier(xi)
        else:

            def lf_h_fn(z: Array) -> Array:
                return cast(Array, jax.grad(barrier)(z) @ f(z))

            lf_h = lf_h_fn(xi)                       # scalar
            grad_lf_h = jax.grad(lf_h_fn)(xi)        # (n,)
            lf2_h = grad_lf_h @ f(xi)                # scalar
            lg_lf_h = grad_lf_h @ g(xi)             # (d,)
            row = -lg_lf_h
            g0, g1 = gains
            h_val = lf2_h + (g0 + g1) * lf_h + g0 * g1 * barrier(xi)
        return row, h_val

    rows, h_vals = jax.vmap(per_sample)(x)          # (B, d), (B,)
    G = rows[:, None, :]                            # (B, 1, d)
    h = h_vals[:, None]                             # (B, 1)
    if spec.a_max is not None:
        d = rows.shape[1]
        g_box, h_box = actuator_box(spec.a_max, d)
        n_batch = x.shape[0]
        G = jnp.concatenate([G, jnp.broadcast_to(g_box, (n_batch, *g_box.shape))], axis=1)
        h = jnp.concatenate([h, jnp.broadcast_to(h_box, (n_batch, *h_box.shape))], axis=1)
    return G, h


def lagrangian_cbf_rows(
    lagrangian: object,
    B_input: Array,
    barrier: Callable[[Array], Array],
    q: Array,
    qdot: Array,
    t: Array,
    spec: CBFSpec,
) -> tuple[Array, Array]:
    r"""CBF rows for a system whose dynamics come from a (possibly learned) Lagrangian.

    The Lagrangian defines a control-affine system on ``x = [q, qdot]``:

    .. math::
        \dot q = \dot q, \qquad \ddot q = \underbrace{M(q,\dot q)^{-1} F}_{\text{drift}}
        \; + \; \underbrace{M(q,\dot q)^{-1} B}_{g}\, a,

    with ``M = d^2L/d\dot q^2`` and ``F`` the generalized force -- both from
    :mod:`omnibias.variational.jax.ops`. We assemble ``f, g`` and defer to
    :func:`control_affine_cbf_rows`. This is how the learned-Lagrangian demo plugs
    straight into the generic builder.

    Parameters
    ----------
    lagrangian:
        An :class:`omnibias.variational.Lagrangian` (order 1).
    B_input:
        Constant input matrix ``B`` mapping actions to generalized forces, ``(n, d)``.
    barrier:
        Barrier over the full state ``x = [q, qdot]`` (``(2n,) -> scalar``).
    q, qdot, t:
        State batch (``(B, n)``, ``(B, n)``, ``(B, 1)``). ``t`` is taken as a uniform
        time slice ``t[:1]`` (a CBF snapshot).
    spec:
        :class:`~omnibias.control.problem.CBFSpec`.

    Returns
    -------
    ``(G, h)`` as in :func:`control_affine_cbf_rows`.
    """
    from omnibias.variational.jax.ops.dynamics import acceleration, mass_matrix

    n = q.shape[-1]
    m = B_input.shape[-1]
    t0 = t[:1]

    def f(xi: Array) -> Array:
        qi, qdi = xi[:n], xi[n:]
        acc = acceleration(lagrangian, qi[None], qdi[None], t0)[0]
        return jnp.concatenate([qdi, acc])

    def g(xi: Array) -> Array:
        qi, qdi = xi[:n], xi[n:]
        mass = mass_matrix(lagrangian, qi[None], qdi[None], t0)[0]
        minv_b = jnp.linalg.solve(mass, B_input)
        return jnp.concatenate([jnp.zeros((n, m)), minv_b], axis=0)

    x = jnp.concatenate([q, qdot], axis=1)
    return control_affine_cbf_rows(f, g, barrier, x, spec)


__all__ = ["actuator_box", "control_affine_cbf_rows", "lagrangian_cbf_rows"]

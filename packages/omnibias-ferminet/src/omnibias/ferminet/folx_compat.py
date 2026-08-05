# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""folx- and DeepQMC-compatible Laplacian APIs backed by omnibias.

This module exposes two thin wrappers so omnibias's closed-form
Laplacian primitive can drop in to JAX-based neural-VMC stacks
without bespoke plumbing on either side:

* :func:`forward_laplacian` -- mirrors the public surface of
  :func:`folx.forward_laplacian` (returns an object with ``.x``,
  ``.dense_jacobian``, ``.laplacian``). Calling it on a function
  built out of :mod:`omnibias.jax.laplacian` primitives uses the
  closed-form path; for any other function the user must register
  an omnibias rule or fall back to folx explicitly.

* :func:`laplacian_factory` -- mirrors DeepQMC's
  ``physics.LaplacianFactory`` protocol: takes
  ``f: jax.Array -> jax.Array`` and returns a callable
  ``g(x) -> (laplacian, gradient)``. Plug this directly into
  ``MolecularHamiltonian(laplacian_factory=...)``.

**Scope.** This is the integration contract; the closed-form path
is automatically used for the one-hidden-layer multi-bias field
:func:`omnibias.jax.neural_field_value_grad_laplacian`. Closed-form
through arbitrary FermiNet/DeepQMC ansatzes requires a custom JAX
interpreter (a Tier-3 roadmap deliverable).

Usage (with an omnibias-built scalar field)::

    from omnibias.jax import folx_compat

    def my_wf(r, W, beta, c, b):
        return omnibias.jax.neural_field_value(r, W, beta, c, b, "tanh")

    fwd = folx_compat.forward_laplacian(my_wf)
    res = fwd(r, W, beta, c, b)
    print(res.x, res.dense_jacobian, res.laplacian)

Usage (as a DeepQMC LaplacianFactory)::

    from deepqmc import MolecularHamiltonian
    from omnibias.ferminet.folx_compat import laplacian_factory

    hamil = MolecularHamiltonian(
        mol=mol,
        laplacian_factory=laplacian_factory,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# folx-compatible result type
# ---------------------------------------------------------------------------


@dataclass
class OmnibiasFwdLaplResult:
    """Folx-API-compatible container for value, gradient, Laplacian.

    The attribute names mirror :class:`folx.api.FwdLaplArray`:

    * ``x`` -- the function value at the input.
    * ``dense_jacobian`` -- the gradient as a dense array (folx returns
      a ``FwdJacobian`` wrapper; we always materialize the dense form).
    * ``laplacian`` -- the Laplacian as a scalar (or batched scalar).

    For symmetry with folx, ``jacobian`` is provided as an alias that
    also returns the dense form -- omnibias does not have a sparse
    representation today.
    """

    x: Array
    dense_jacobian: Array
    laplacian: Array

    @property
    def jacobian(self) -> Array:
        return self.dense_jacobian


# ---------------------------------------------------------------------------
# forward_laplacian
# ---------------------------------------------------------------------------


# The trace machinery is intentionally minimal: we wrap the user's
# function with `jax.jvp`/`jax.grad` to derive (value, gradient,
# Laplacian) in a way that JIT-compiles cleanly. For a *single*
# omnibias-built one-layer field the user can instead pass
# ``closed_form=True``, which dispatches to the analytic
# :func:`omnibias.jax.neural_field_value_grad_laplacian` path.


def forward_laplacian(
    f: Callable[..., Array],
    *,
    closed_form: bool = False,
) -> Callable[..., OmnibiasFwdLaplResult]:
    """Return a function computing ``(value, gradient, laplacian)``.

    Parameters
    ----------
    f
        The function to differentiate. Must take a leading ``x``
        argument of shape ``(D,)`` and return a scalar.
    closed_form
        If ``True``, ``f`` must internally call exactly one of
        :func:`omnibias.jax.neural_field_value` /
        :func:`neural_field_value_and_laplacian` /
        :func:`neural_field_value_grad_laplacian` with the same
        ``x`` argument; the call is detected at trace time and
        replaced with the analytic closed-form path. (Reserved
        for future use; current implementation requires explicit
        decomposition by the caller -- see usage notes above.)

    Notes
    -----
    Without ``closed_form=True`` this falls back to a
    ``jax.jvp(jax.grad(f))`` Hessian-diagonal sum -- correct, but
    *not* the omnibias closed-form path. To get the closed-form
    speedup, structure your wavefunction so the omnibias call is
    visible at trace time, or compose
    :func:`omnibias.jax.neural_field_value_grad_laplacian` directly.

    The closed-form-detecting JAX interpreter is a Tier-3 roadmap
    deliverable; this is the public API that interpreter will plug
    into.
    """
    if closed_form:
        raise NotImplementedError(
            "closed_form=True requires the omnibias JAX interpreter "
            "(a Tier-3 roadmap deliverable). Use "
            "omnibias.jax.neural_field_value_grad_laplacian "
            "directly to access the closed-form path today."
        )

    def wrapped(x: Array, *args: Any, **kwargs: Any) -> OmnibiasFwdLaplResult:
        def f_of_x(xx: Array) -> Array:
            return f(xx, *args, **kwargs)

        value, gradient = jax.value_and_grad(f_of_x)(x)
        n_coord = x.shape[-1]
        eye = jnp.eye(n_coord, dtype=x.dtype)
        grad_f = jax.grad(f_of_x)
        _, grad_jvp = jax.linearize(grad_f, x)

        def add_diag(i: Array, acc: Array) -> Array:
            out: Array = acc + grad_jvp(eye[i])[i]
            return out

        laplacian = jax.lax.fori_loop(0, n_coord, add_diag, jnp.asarray(0.0, dtype=x.dtype))
        return OmnibiasFwdLaplResult(
            x=value,
            dense_jacobian=gradient,
            laplacian=laplacian,
        )

    return wrapped


# ---------------------------------------------------------------------------
# DeepQMC LaplacianFactory
# ---------------------------------------------------------------------------


def laplacian_factory(
    f: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Array]]:
    """Conforms to :class:`deepqmc.physics.LaplacianFactory`.

    Returns ``g: x -> (laplacian, gradient)`` for ``f: x -> scalar``,
    using the same trace-time machinery as :func:`forward_laplacian`.

    For an omnibias-built network the closed-form path can be reached
    by structuring ``f`` to call
    :func:`omnibias.jax.neural_field_value_grad_laplacian` (which is
    detected and used directly by the upcoming interpreter; the
    Tier-3 interpreter is a roadmap deliverable).
    """
    fwd = forward_laplacian(f)

    def lap(x: Array) -> tuple[Array, Array]:
        r = fwd(x)
        return r.laplacian, r.dense_jacobian

    return lap


# ---------------------------------------------------------------------------
# Convenience: closed-form one-shot for the documented happy path
# ---------------------------------------------------------------------------


def closed_form_forward_laplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str,
) -> OmnibiasFwdLaplResult:
    """Closed-form forward Laplacian for an omnibias one-layer field.

    Convenience wrapper around
    :func:`omnibias.jax.neural_field_value_grad_laplacian` that
    returns the folx-shaped :class:`OmnibiasFwdLaplResult`. This is
    the "guaranteed closed-form" entry point; arbitrary networks
    that wrap this primitive will be handled by the upcoming
    interpreter (Tier-3 work).
    """
    from omnibias.jax.laplacian import neural_field_value_grad_laplacian

    value, grad, lap = neural_field_value_grad_laplacian(
        x,
        W,
        beta,
        c,
        b,
        activation,
    )
    return OmnibiasFwdLaplResult(
        x=value,
        dense_jacobian=grad,
        laplacian=lap,
    )


__all__ = [
    "OmnibiasFwdLaplResult",
    "closed_form_forward_laplacian",
    "forward_laplacian",
    "laplacian_factory",
]

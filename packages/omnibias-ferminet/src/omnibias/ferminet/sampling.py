# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Reproducible quantum Monte-Carlo sampling substrate (JAX).

This module is the **sampling** foundation for a future variational Monte
Carlo (VMC) trainer: reproducible Metropolis / Langevin walkers over
electron positions, a local-energy estimator, autocorrelation-aware error
diagnostics, and a walker-level log-derivative accumulator. It is
deliberately **ansatz-agnostic** -- every function operates on a
caller-supplied ``log_abs_psi_fn`` callable, so it plugs into
:mod:`omnibias.ferminet.restricted`'s Tier-2 wavefunction (or any future
ansatz) without modification.

Ansatz interface contract
--------------------------
Every function in this module takes a two-argument callable

.. math::

   \texttt{log\_abs\_psi\_fn}(\texttt{params}, r) \;\longmapsto\; \log|\psi(r)|,

i.e. the **log of the wavefunction amplitude** (not the log-density), for a
*single* walker's flat coordinate vector ``r`` of shape ``(D,)`` (``D`` is
whatever the ansatz expects -- ``3 * n_e`` for electron positions, or a
toy 1-D/few-D coordinate for the test targets below). ``params`` is an
arbitrary JAX pytree; this module never inspects its structure and works
generically via :func:`jax.vmap`, :func:`jax.grad`, and
:func:`jax.flatten_util.ravel_pytree`.

The **target density** every walker below actually samples is

.. math::

   p(r) \;\propto\; |\psi(r)|^{2} \;=\; \exp\bigl(2\,\texttt{log\_abs\_psi\_fn}(\texttt{params}, r)\bigr),

so all acceptance/proposal math is phrased in terms of ``2 * log_abs_psi_fn(...)``
rather than requiring the caller to square anything themselves. This
convention matches :mod:`omnibias.ferminet.restricted`'s
``tier2_log_abs_psi`` / ``tier2sym_log_abs_psi`` / ``tier2_blocked_log_abs_psi``
/ ``tier2sym_blocked_log_abs_psi`` functions exactly, so any of those can be
passed directly (after binding their extra ``n_e`` argument with
``functools.partial`` or a lambda, since this module's contract is strictly
two arguments).

Batched functions (the walkers, the local-energy estimator, and the
log-derivative accumulator) expect a **leading batch axis**: ``positions``
has shape ``(n_walkers, D)`` for the walkers and ``(n_samples, D)`` for the
per-sample estimators. Internally every batched function is built with
:func:`jax.vmap` over ``log_abs_psi_fn`` (never by asking ``log_abs_psi_fn``
itself to be batch-aware), so a single-walker ansatz such as
``tier2_log_abs_psi`` works unmodified.

Closed-form / autodiff / numerical honesty
-------------------------------------------
Per the repository's derivative-tower doctrine, this module is careful to
label each piece by its actual register -- **do not blur these**:

* **Closed form (the omnibias derivative tower)** -- nothing in *this*
  module computes a closed-form derivative itself. It is designed to
  *consume* one: pass :func:`omnibias.ferminet.restricted.tier2_local_kinetic_energy`
  (or ``tier2sym_...`` / ``jastrow_slater_local_kinetic_energy``) as the
  ``kinetic_fn`` of :func:`local_energy` to route the kinetic term through
  the package's closed-form Laplacian machinery.
* **Plain JAX autodiff (ordinary ``jax.grad`` / ``jax.hessian``, not a
  closed-form-tower claim)** -- the Langevin drift
  :math:`\nabla_r \log|\psi|` (:func:`langevin_step` /
  :func:`langevin_sample`), the general-ansatz Laplacian fallback
  (:func:`autodiff_grad_and_laplacian`), and the walker-level parameter
  log-derivative accumulator (:func:`log_derivative_accumulator`) are all
  ordinary reverse-mode (or forward-over-reverse, for the Hessian trace)
  autodiff. None of these reuse the polynomial coefficient tower in
  ``omnibias.core.polynomials``; they differentiate whatever ``log_abs_psi_fn``
  the caller supplies, generically.
* **Numerical / statistical estimator** -- the Metropolis-Hastings and
  Langevin *acceptance mechanics* are the standard MCMC algorithms (not a
  closed-form primitive at all), and the autocorrelation / integrated
  autocorrelation time / effective-sample-size machinery
  (:func:`autocorrelation_function`, :func:`integrated_autocorrelation_time`,
  :func:`effective_sample_size`, :func:`standard_error_of_mean`) is an
  ordinary windowed-estimator computation over a finite sample: its accuracy
  is set by chain length, exactly the "Numerical" register in
  ``docs/scope-and-guarantees.md``, not a sound enclosure.

Reproducibility
----------------
Every stochastic function takes an **explicit, caller-controlled**
:func:`jax.random.PRNGKey`. There is no hidden or global random state
anywhere in this module: calling any sampler twice with the same key,
params, and positions returns bit-identical output.

``jit`` / ``vmap`` / ``scan`` safety
--------------------------------------
Per ``.cursor/rules/omnibias.md`` (JAX tracing): every walker step is written as a
pure function of its arguments (no mutation, no Python-level branching on a
*traced* array value), parallel walkers are vectorised with
:func:`jax.vmap`, and multi-step trajectories are unrolled with
:func:`jax.lax.scan`. The one static (non-traced) Python control-flow
choice is :func:`langevin_step`'s ``adjusted`` flag, which selects between
two closed-form branches at *trace time* (like ``n_steps``, it must stay a
plain Python ``bool``/``int``, not a traced array, and should be passed via
``static_argnames`` if the caller wraps these functions in ``jax.jit``).

Every function here also takes at least one plain **Python callable**
(``log_abs_psi_fn``, or ``kinetic_fn`` / ``potential_fn`` for
:func:`local_energy`). If a caller wraps any of these functions in
``jax.jit`` directly, that callable argument must be listed in
``static_argnames`` too -- JAX cannot trace a function object as an
abstract array, and will raise a ``TypeError`` ("as an abstract array ...
was of type <class 'function'>") otherwise. For example::

    jax.jit(metropolis_sample, static_argnames=("log_abs_psi_fn", "n_steps"))
    jax.jit(langevin_sample, static_argnames=("log_abs_psi_fn", "n_steps", "adjusted"))
    jax.jit(local_energy, static_argnames=("kinetic_fn", "potential_fn"))
    jax.jit(log_derivative_accumulator, static_argnames=("log_abs_psi_fn",))

(this is exercised directly by ``tests/test_sampling.py::TestJitSafety``).

Deliberately NOT built here
-----------------------------
This module is the sampling substrate only. It does **not** implement
stochastic reconfiguration (the natural-gradient covariance-matrix solve
that :mod:`omnibias.curvature.natural_gradient` already provides the
generic ``(P, P)`` / ``(P,)`` machinery for), lattice Hamiltonians,
antisymmetric-ansatz changes, or ground-state certificates. Those are
separate, later work items that consume exactly the
``(n_samples, n_params)`` array :func:`log_derivative_accumulator` returns
here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

#: Arbitrary JAX pytree of ansatz parameters (never inspected by this module).
Params = Any

#: ``(params, position) -> log|psi(position)|`` for a single walker,
#: ``position`` of shape ``(D,)``. See the module docstring's "Ansatz
#: interface contract".
LogAbsPsiFn = Callable[[Params, Array], Array]


# ---------------------------------------------------------------------------
# 1. Random-walk Metropolis-Hastings walker
# ---------------------------------------------------------------------------


class MetropolisResult(NamedTuple):
    r"""Output of :func:`metropolis_sample`.

    Shapes (``n_walkers`` independent chains, ``n_steps`` recorded steps)
    ----------------------------------------------------------------------
    positions
        ``(n_walkers, D)`` -- final positions after ``n_steps``.
    trajectory
        ``(n_steps, n_walkers, D)`` -- positions recorded after every step
        (index 0 is the state after the *first* step, not the initial state).
    log_abs_psi
        ``(n_walkers,)`` -- ``log|psi|`` at the final positions.
    n_accepted
        ``(n_walkers,)`` int32 -- number of accepted proposals per walker.
    acceptance_rate
        ``(n_walkers,)`` -- ``n_accepted / n_steps``, the standard Metropolis
        acceptance-rate diagnostic (healthy random-walk Metropolis is usually
        tuned via ``step_size`` to land around 0.2-0.5).
    """

    positions: Array
    trajectory: Array
    log_abs_psi: Array
    n_accepted: Array
    acceptance_rate: Array


def metropolis_step(
    log_abs_psi_fn: LogAbsPsiFn,
    params: Params,
    positions: Array,
    log_abs_psi_current: Array,
    key: Array,
    step_size: float | Array,
) -> tuple[Array, Array, Array]:
    r"""One symmetric-Gaussian random-walk Metropolis-Hastings step, all walkers.

    Numerical / statistical register (standard MCMC): for each of the
    ``n_walkers`` independent chains, proposes
    ``r' = r + step_size * xi``, ``xi ~ N(0, I)`` (a symmetric proposal, so
    no Hastings correction term is needed) and accepts with probability
    ``min(1, |psi(r')|^2 / |psi(r)|^2)``, computed **stably in log-space** as
    ``log(u) <= log_ratio`` with ``log_ratio = 2 * (log_abs_psi_fn(params, r')
    - log_abs_psi_fn(params, r))`` -- no ``exp`` of a potentially large
    argument is ever evaluated.

    Parallel walkers use :func:`jax.vmap` (one JAX PRNG sub-key per walker,
    split from ``key``); this makes the function ``jit``-safe and composes
    with :func:`jax.lax.scan` for multi-step trajectories (see
    :func:`metropolis_sample`).

    Parameters
    ----------
    positions
        ``(n_walkers, D)`` current positions.
    log_abs_psi_current
        ``(n_walkers,)`` -- ``log_abs_psi_fn(params, positions)``, cached by
        the caller so this function never redundantly recomputes it.
    key
        A single ``jax.random.PRNGKey``; split internally into one sub-key
        per walker plus an internal proposal/acceptance split, so the same
        key always reproduces the same step bit-for-bit.
    step_size
        The Gaussian proposal standard deviation.

    Returns
    -------
    new_positions, new_log_abs_psi, accepted
        ``(n_walkers, D)``, ``(n_walkers,)``, ``(n_walkers,)`` bool.
    """
    n_walkers = positions.shape[0]
    walker_keys = jax.random.split(key, n_walkers)

    def _step_one(
        position: Array, log_abs_psi_here: Array, walker_key: Array
    ) -> tuple[Array, Array, Array]:
        key_prop, key_acc = jax.random.split(walker_key)
        proposal = position + step_size * jax.random.normal(
            key_prop, shape=position.shape, dtype=position.dtype
        )
        log_abs_psi_proposal = log_abs_psi_fn(params, proposal)
        log_ratio = 2.0 * (log_abs_psi_proposal - log_abs_psi_here)
        log_u = jnp.log(jax.random.uniform(key_acc, dtype=position.dtype))
        accept = log_u <= log_ratio
        new_position = jnp.where(accept, proposal, position)
        new_log_abs_psi = jnp.where(accept, log_abs_psi_proposal, log_abs_psi_here)
        return new_position, new_log_abs_psi, accept

    result: tuple[Array, Array, Array] = jax.vmap(_step_one)(
        positions, log_abs_psi_current, walker_keys
    )
    return result


def metropolis_sample(
    log_abs_psi_fn: LogAbsPsiFn,
    params: Params,
    init_positions: Array,
    key: Array,
    *,
    n_steps: int,
    step_size: float | Array,
) -> MetropolisResult:
    r"""Run ``n_steps`` of random-walk Metropolis-Hastings on all walkers.

    Wraps :func:`metropolis_step` in a :func:`jax.lax.scan` over
    ``n_steps`` (each step's PRNG sub-key split off ``key`` up front via
    ``jax.random.split(key, n_steps)``), so the whole trajectory is a pure,
    ``jit``-compatible function of ``(params, init_positions, key)``: the
    same key reproduces the same trajectory bit-for-bit (see
    ``test_metropolis_reproducibility``). ``n_steps`` must be a static
    Python ``int`` (it sizes the scan and the PRNG key array), so mark it
    ``static_argnames=("n_steps",)`` if wrapping this function in
    ``jax.jit``.

    Parameters
    ----------
    init_positions
        ``(n_walkers, D)`` initial positions of every independent walker.
    n_steps
        Number of sequential Metropolis steps (static).
    step_size
        The Gaussian proposal standard deviation, shared by all walkers.
    """
    init_log_abs_psi = jax.vmap(log_abs_psi_fn, in_axes=(None, 0))(params, init_positions)
    step_keys = jax.random.split(key, n_steps)

    def body(
        carry: tuple[Array, Array], step_key: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        positions, log_abs_psi_current = carry
        new_positions, new_log_abs_psi, accepted = metropolis_step(
            log_abs_psi_fn, params, positions, log_abs_psi_current, step_key, step_size
        )
        return (new_positions, new_log_abs_psi), (new_positions, accepted)

    (final_positions, final_log_abs_psi), (trajectory, accepted_traj) = jax.lax.scan(
        body, (init_positions, init_log_abs_psi), step_keys
    )
    n_accepted = jnp.sum(accepted_traj.astype(jnp.int32), axis=0)
    acceptance_rate = n_accepted.astype(init_positions.dtype) / n_steps
    return MetropolisResult(
        positions=final_positions,
        trajectory=trajectory,
        log_abs_psi=final_log_abs_psi,
        n_accepted=n_accepted,
        acceptance_rate=acceptance_rate,
    )


# ---------------------------------------------------------------------------
# 2. Langevin (MALA-style) walker
# ---------------------------------------------------------------------------


class LangevinResult(NamedTuple):
    r"""Output of :func:`langevin_sample`. Same shapes as :class:`MetropolisResult`.

    When ``adjusted=False`` (unadjusted Langevin, ULA) every proposal is
    taken unconditionally: ``n_accepted`` and ``acceptance_rate`` are then
    reported as ``n_steps`` / ``1.0`` for every walker **by convention**,
    not because the chain was genuinely tested against a target -- ULA has
    no Metropolis test at all, and (unlike MALA) is only exact in the
    ``step_size -> 0`` limit.
    """

    positions: Array
    trajectory: Array
    log_abs_psi: Array
    n_accepted: Array
    acceptance_rate: Array


def _mala_log_transition_density(
    x_to: Array, x_from: Array, grad_log_p_from: Array, step_size: float | Array
) -> Array:
    r"""Log Gaussian MALA proposal density ``log q(x_to | x_from)`` (up to a constant).

    ``q(x' | x) = N(x'; x + (step_size^2 / 2) grad_log_p(x), step_size^2 I)``.
    The ``(2 pi step_size^2)^{-D/2}`` normaliser is dropped because it is
    identical for the forward and reverse transition (same ``step_size``,
    same dimension) and cancels exactly in the Metropolis-Hastings log-ratio;
    only the quadratic exponent is needed.
    """
    mean = x_from + 0.5 * step_size**2 * grad_log_p_from
    diff = x_to - mean
    out: Array = -0.5 * jnp.sum(diff * diff) / (step_size**2)
    return out


def langevin_step(
    log_abs_psi_fn: LogAbsPsiFn,
    params: Params,
    positions: Array,
    log_abs_psi_current: Array,
    key: Array,
    step_size: float | Array,
    *,
    adjusted: bool = True,
) -> tuple[Array, Array, Array]:
    r"""One (unadjusted or Metropolis-adjusted) Langevin step, all walkers.

    The drift is the **plain JAX autodiff** gradient of ``log|psi|`` with
    respect to the *walker position* -- an ordinary ``jax.grad`` of the
    caller's ``log_abs_psi_fn``, **not** a closed-form-derivative-tower
    claim (contrast with :func:`omnibias.ferminet.restricted.tier2_grad_laplacian_log_psi`,
    which *is* closed form but differentiates with respect to electron
    position for a specific ansatz, not a generic ``log_abs_psi_fn``).

    Proposal: ``r' = r + (step_size^2 / 2) * grad_log_p(r) + step_size * xi``,
    ``xi ~ N(0, I)``, with ``grad_log_p(r) = 2 * grad_r log_abs_psi_fn(params, r)``
    (the target density is ``|psi|^2``, so its log-gradient is twice the
    log-amplitude gradient).

    * ``adjusted=True`` (**MALA**, the default): accepts/rejects with the
      exact Metropolis-Hastings log-ratio (target log-ratio plus the
      asymmetric-proposal correction ``log q(r|r') - log q(r'|r)``), so the
      stationary distribution is exactly ``|psi|^2`` for any ``step_size``.
    * ``adjusted=False`` (**ULA**, unadjusted Langevin): every proposal is
      taken unconditionally (no accept/reject test at all); the stationary
      distribution is only ``|psi|^2`` in the ``step_size -> 0`` limit, with
      an ``O(step_size)`` bias at finite step size.

    ``adjusted`` is a plain Python ``bool`` that selects between the two
    branches **at trace time** (not a traced array), exactly like
    ``n_steps`` elsewhere in this module; pass it through
    ``static_argnames=("adjusted",)`` if wrapping this function in
    ``jax.jit``.
    """
    n_walkers = positions.shape[0]
    walker_keys = jax.random.split(key, n_walkers)
    grad_log_abs_psi_fn = jax.grad(log_abs_psi_fn, argnums=1)

    def _grad_log_p(position: Array) -> Array:
        out: Array = 2.0 * grad_log_abs_psi_fn(params, position)
        return out

    def _step_one(
        position: Array, log_abs_psi_here: Array, walker_key: Array
    ) -> tuple[Array, Array, Array]:
        key_prop, key_acc = jax.random.split(walker_key)
        grad_log_p_here = _grad_log_p(position)
        noise = jax.random.normal(key_prop, shape=position.shape, dtype=position.dtype)
        proposal = position + 0.5 * step_size**2 * grad_log_p_here + step_size * noise
        log_abs_psi_proposal = log_abs_psi_fn(params, proposal)

        if not adjusted:
            accept_always = jnp.asarray(True)
            return proposal, log_abs_psi_proposal, accept_always

        grad_log_p_proposal = _grad_log_p(proposal)
        log_q_forward = _mala_log_transition_density(
            proposal, position, grad_log_p_here, step_size
        )
        log_q_reverse = _mala_log_transition_density(
            position, proposal, grad_log_p_proposal, step_size
        )
        log_ratio = 2.0 * (log_abs_psi_proposal - log_abs_psi_here) + (
            log_q_reverse - log_q_forward
        )
        log_u = jnp.log(jax.random.uniform(key_acc, dtype=position.dtype))
        accept = log_u <= log_ratio
        new_position = jnp.where(accept, proposal, position)
        new_log_abs_psi = jnp.where(accept, log_abs_psi_proposal, log_abs_psi_here)
        return new_position, new_log_abs_psi, accept

    result: tuple[Array, Array, Array] = jax.vmap(_step_one)(
        positions, log_abs_psi_current, walker_keys
    )
    return result


def langevin_sample(
    log_abs_psi_fn: LogAbsPsiFn,
    params: Params,
    init_positions: Array,
    key: Array,
    *,
    n_steps: int,
    step_size: float | Array,
    adjusted: bool = True,
) -> LangevinResult:
    r"""Run ``n_steps`` of the Langevin walker on all walkers.

    Mirrors :func:`metropolis_sample`'s ``vmap``-over-walkers,
    ``scan``-over-steps structure and reproducibility guarantee; see
    :func:`langevin_step` for the ``adjusted`` (MALA) vs. unadjusted (ULA)
    distinction. ``n_steps`` and ``adjusted`` are both static.
    """
    init_log_abs_psi = jax.vmap(log_abs_psi_fn, in_axes=(None, 0))(params, init_positions)
    step_keys = jax.random.split(key, n_steps)

    def body(
        carry: tuple[Array, Array], step_key: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        positions, log_abs_psi_current = carry
        new_positions, new_log_abs_psi, accepted = langevin_step(
            log_abs_psi_fn,
            params,
            positions,
            log_abs_psi_current,
            step_key,
            step_size,
            adjusted=adjusted,
        )
        return (new_positions, new_log_abs_psi), (new_positions, accepted)

    (final_positions, final_log_abs_psi), (trajectory, accepted_traj) = jax.lax.scan(
        body, (init_positions, init_log_abs_psi), step_keys
    )
    n_accepted = jnp.sum(accepted_traj.astype(jnp.int32), axis=0)
    acceptance_rate = n_accepted.astype(init_positions.dtype) / n_steps
    return LangevinResult(
        positions=final_positions,
        trajectory=trajectory,
        log_abs_psi=final_log_abs_psi,
        n_accepted=n_accepted,
        acceptance_rate=acceptance_rate,
    )


# ---------------------------------------------------------------------------
# 3. Local-energy estimator
# ---------------------------------------------------------------------------


def kinetic_energy_from_grad_lap(grad: Array, lap: Array) -> Array:
    r"""Local kinetic energy from ``(grad, lap)`` of ``log|psi|`` at one position.

    Implements the log-domain identity

    .. math::

       \frac{\nabla^2 \psi}{\psi} = \nabla^2 \log|\psi| + \lVert\nabla \log|\psi|\rVert^2,

    so that :math:`T_\mathrm{loc} = -\tfrac12 \nabla^2\psi/\psi` is computed
    as :math:`-\tfrac12\bigl(\nabla^2\log|\psi| + \lVert\nabla\log|\psi|\rVert^2\bigr)`
    **without ever dividing by** :math:`\psi` -- ``psi`` itself can be
    arbitrarily close to zero (e.g. near a nodal surface) while ``log|psi|``
    and its derivatives stay finite. This is exactly the identity
    :mod:`omnibias.ferminet.restricted`'s Tier-2 kinetic-energy functions
    (e.g. ``tier2_local_kinetic_energy``) already use internally; this
    helper exists so a caller with a raw ``(grad, lap)`` pair (rather than a
    pre-fused ``T_loc`` function) can assemble the same quantity.

    ``grad`` is ``(D,)``, ``lap`` is a scalar; returns a scalar.
    """
    out: Array = -0.5 * (lap + jnp.sum(grad * grad))
    return out


def autodiff_grad_and_laplacian(
    log_abs_psi_fn: LogAbsPsiFn, params: Params, position: Array
) -> tuple[Array, Array]:
    r"""Plain-autodiff ``(grad, Laplacian)`` of ``log|psi|`` at one position.

    **Not the omnibias closed-form derivative tower.** This is an ordinary
    ``jax.grad`` plus a ``jax.hessian`` trace, provided as a general-ansatz
    fallback for callers that do not have a closed-form Laplacian available
    (e.g. a toy target such as the harmonic-oscillator oracle in the test
    suite). For an ansatz that *does* have one --
    :func:`omnibias.ferminet.restricted.tier2_grad_laplacian_log_psi` and
    its symmetric-pool / spin-blocked siblings -- use that instead and skip
    this function entirely; it is asymptotically more expensive
    (``O(D)`` forward-over-reverse Hessian evaluations vs. the closed-form
    machinery's ``O(1)``-in-width cost) and is not what
    ``docs/scope-and-guarantees.md`` calls "closed form".

    Returns ``(grad, lap)`` with ``grad`` of shape ``(D,)`` and ``lap`` a
    scalar (``trace`` of the ``(D, D)`` Hessian).
    """
    grad = jax.grad(log_abs_psi_fn, argnums=1)(params, position)
    hessian = jax.hessian(log_abs_psi_fn, argnums=1)(params, position)
    lap = jnp.trace(hessian)
    return grad, lap


def local_energy(
    kinetic_fn: Callable[[Params, Array], Array],
    potential_fn: Callable[[Array], Array],
    params: Params,
    positions: Array,
) -> Array:
    r"""Per-sample local energy ``E_L(r) = T_loc(r) + V(r)``.

    ``E_L(r) = H psi(r) / psi(r)`` for a Hamiltonian ``H = T + V``; a true
    eigenstate has ``E_L`` constant everywhere (see
    ``test_local_energy_harmonic_oscillator_is_exactly_constant``), which is
    why it is the standard VMC diagnostic and training signal.

    ``kinetic_fn(params, position) -> scalar`` supplies the local kinetic
    energy :math:`T_\mathrm{loc}(r)` for a *single* walker/sample and may be:

    * a **closed-form** fused kinetic-energy function from
      :mod:`omnibias.ferminet.restricted`, e.g.
      ``lambda params, r: tier2_local_kinetic_energy(params, r, n_e=n_e)``
      (binding the ansatz's extra static ``n_e`` argument); or
    * a thin wrapper around a caller-supplied ``(grad, lap)``-style
      Laplacian, e.g.
      ``lambda params, r: kinetic_energy_from_grad_lap(*my_grad_lap_fn(params, r))``,
      composing with :func:`kinetic_energy_from_grad_lap`; or
    * the general-ansatz autodiff fallback,
      ``lambda params, r: kinetic_energy_from_grad_lap(*autodiff_grad_and_laplacian(log_abs_psi_fn, params, r))``.

    ``potential_fn(position) -> scalar`` supplies :math:`V(r)` for a single
    walker/sample (e.g. :func:`omnibias.jax.coulomb_potential` bound to
    fixed nuclear positions and charges via ``functools.partial``).

    ``positions`` is ``(n_samples, D)``; both callables are vectorised over
    the sample axis with :func:`jax.vmap`. Returns ``(n_samples,)``.
    """

    def _per_sample(position: Array) -> Array:
        return kinetic_fn(params, position) + potential_fn(position)

    result: Array = jax.vmap(_per_sample)(positions)
    return result


# ---------------------------------------------------------------------------
# 4. Autocorrelation / error diagnostics
# ---------------------------------------------------------------------------


def autocorrelation_function(x: Array, *, max_lag: int) -> Array:
    r"""Normalized sample autocorrelation function ``rho(0..max_lag)``.

    Numerical / statistical estimator (not closed form, not autodiff):

    .. math::

       \rho(k) = \frac{\frac1N\sum_{i=0}^{N-k-1} (x_i - \bar x)(x_{i+k} - \bar x)}
                      {\frac1N \sum_{i=0}^{N-1} (x_i - \bar x)^2},

    the standard **biased** (divide-by-``N``, not ``N - k``) autocovariance
    estimator, with :math:`\rho(0) = 1` exactly. Biased-but-bounded-variance
    is the deliberate, standard choice for autocorrelation-*time* estimation
    (Sokal 1989; the same convention used by e.g. ``emcee.autocorr``): the
    "unbiased" ``1 / (N - k)`` normalization has *growing* variance as
    ``k -> N``, which would corrupt exactly the large-lag tail that
    :func:`integrated_autocorrelation_time`'s windowing has to inspect to
    decide when to stop.

    Computed via a single zero-padded FFT (Wiener-Khinchin: the
    autocovariance is the inverse FFT of the power spectrum), **not** a
    per-lag Python loop -- a per-lag loop of up to ``N/2`` un-jitted
    reductions is asymptotically fine but prohibitively slow to *dispatch*
    eagerly (every iteration is a separate small XLA call with its own
    dispatch overhead). ``x`` may carry arbitrary leading batch dimensions
    (e.g. ``(n_chains, n_samples)``); the autocorrelation is always computed
    along the **last** axis, so this function composes with plain
    broadcasting for multiple independent chains. ``max_lag`` must satisfy
    ``0 <= max_lag < N`` (a static Python ``int``, since it only sizes the
    static output shape -- this is ``jit``-safe). Returns shape
    ``(..., max_lag + 1)``.

    A chain with exactly zero sample variance (e.g. local-energy samples of
    an exact eigenstate, which is constant at every point) has an undefined
    ``0 / 0`` correlation coefficient; this is handled explicitly by
    returning the correlation of a constant signal with itself --
    :math:`\rho(0) = 1`, :math:`\rho(k) = 0` for :math:`k > 0` -- rather than
    ``nan``, so downstream (:func:`integrated_autocorrelation_time`,
    :func:`standard_error_of_mean`) see a well-defined ``tau = 1`` and a
    correctly-zero standard error instead of propagating ``nan``.
    """
    x = jnp.asarray(x)
    n = x.shape[-1]
    if not (0 <= max_lag < n):
        raise ValueError(f"max_lag must satisfy 0 <= max_lag < {n}, got {max_lag}")
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    spectrum = jnp.fft.rfft(centered, n=n_fft, axis=-1)
    # Zero-padding to n_fft >= 2n means no circular wraparound for lags < n,
    # so autocovariance[..., k] = sum_{i=0}^{n-1-k} centered[i] * centered[i+k].
    autocovariance = jnp.fft.irfft(spectrum * jnp.conj(spectrum), n=n_fft, axis=-1)[..., :n]
    variance0 = autocovariance[..., 0:1]
    is_constant = variance0 == 0
    safe_variance0 = jnp.where(is_constant, jnp.ones_like(variance0), variance0)
    lag_is_zero = jnp.arange(n) == 0
    constant_chain_rho = jnp.where(lag_is_zero, jnp.ones_like(autocovariance), 0.0)
    rho_full = jnp.where(is_constant, constant_chain_rho, autocovariance / safe_variance0)
    out: Array = rho_full[..., : max_lag + 1]
    return out


def integrated_autocorrelation_time(
    x: Array, *, c: float = 5.0, max_lag: int | None = None
) -> Array:
    r"""Integrated autocorrelation time via Sokal's automatic-windowing estimator.

    Numerical / statistical estimator. The integrated autocorrelation time
    at window ``M`` is ``tau(M) = 1 + 2 * sum_{k=1}^{M} rho(k)``; the
    automatic window is the smallest ``M`` with ``M >= c * tau(M)`` (Sokal's
    self-consistent windowing, ``c`` typically ``5``-``10``: a larger
    window includes more (noisy) lags, so the window is stopped as soon as
    it is comfortably larger than the correlation time it is estimating).
    Falls back to the largest available window (``max_lag``) if the
    self-consistency condition is never met within ``max_lag`` (a sign the
    chain is too short to resolve ``tau`` reliably -- the returned value is
    then only a lower-bound-flavoured estimate).

    ``max_lag`` defaults to ``N // 2`` (a standard rule of thumb: lags
    beyond half the chain length are dominated by estimator noise). ``x``
    may carry leading batch dimensions, matching
    :func:`autocorrelation_function`. Returns shape ``x.shape[:-1]`` (a
    scalar for a plain 1-D chain).
    """
    x = jnp.asarray(x)
    n = x.shape[-1]
    resolved_max_lag = max(1, n // 2) if max_lag is None else int(max_lag)
    rho = autocorrelation_function(x, max_lag=resolved_max_lag)  # (..., resolved_max_lag + 1)
    cumulative = jnp.cumsum(rho[..., 1:], axis=-1)  # sum_{k=1}^{M} rho(k), M = 1..resolved_max_lag
    tau_of_window = 1.0 + 2.0 * cumulative  # (..., resolved_max_lag)
    windows = jnp.arange(1, resolved_max_lag + 1, dtype=tau_of_window.dtype)
    window_is_self_consistent = windows >= c * tau_of_window
    any_self_consistent = jnp.any(window_is_self_consistent, axis=-1)
    first_self_consistent = jnp.argmax(window_is_self_consistent, axis=-1)
    fallback = jnp.full_like(first_self_consistent, resolved_max_lag - 1)
    chosen_window = jnp.where(any_self_consistent, first_self_consistent, fallback)
    tau: Array = jnp.take_along_axis(
        tau_of_window, chosen_window[..., None], axis=-1
    )[..., 0]
    return tau


def effective_sample_size(x: Array, *, c: float = 5.0, max_lag: int | None = None) -> Array:
    r"""Autocorrelation-corrected effective sample size ``N_eff = N / tau``.

    Numerical / statistical estimator built on
    :func:`integrated_autocorrelation_time`. ``N`` correlated samples carry
    the statistical information of only ``N / tau`` independent ones.
    """
    n = x.shape[-1]
    tau = integrated_autocorrelation_time(x, c=c, max_lag=max_lag)
    out: Array = n / tau
    return out


def standard_error_of_mean(x: Array, *, c: float = 5.0, max_lag: int | None = None) -> Array:
    r"""Autocorrelation-corrected standard error of the mean, ``sqrt(Var(x) / N_eff)``.

    Numerical / statistical estimator: the sample variance (``ddof=1``)
    divided by the effective sample size
    (:func:`effective_sample_size`), matching the usual Monte-Carlo error
    bar on a correlated chain's mean.
    """
    x = jnp.asarray(x)
    n = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    sample_variance = jnp.sum((x - mean) ** 2, axis=-1) / (n - 1)
    n_eff = effective_sample_size(x, c=c, max_lag=max_lag)
    out: Array = jnp.sqrt(sample_variance / n_eff)
    return out


class ChainDiagnostics(NamedTuple):
    """Bundled Monte-Carlo error diagnostics for one scalar sample chain."""

    mean: Array
    standard_error: Array
    integrated_autocorr_time: Array
    effective_sample_size: Array


def energy_chain_diagnostics(
    samples: Array, *, c: float = 5.0, max_lag: int | None = None
) -> ChainDiagnostics:
    r"""Convenience bundle: mean, autocorrelation-corrected SEM, tau, and N_eff.

    Typically called on a local-energy chain (:func:`local_energy` applied
    along a :func:`metropolis_sample` / :func:`langevin_sample` trajectory)
    to report the VMC energy estimate with an honest, autocorrelation-aware
    error bar. Numerical / statistical register throughout -- see
    :func:`integrated_autocorrelation_time`.
    """
    samples = jnp.asarray(samples)
    return ChainDiagnostics(
        mean=jnp.mean(samples, axis=-1),
        standard_error=standard_error_of_mean(samples, c=c, max_lag=max_lag),
        integrated_autocorr_time=integrated_autocorrelation_time(samples, c=c, max_lag=max_lag),
        effective_sample_size=effective_sample_size(samples, c=c, max_lag=max_lag),
    )


# ---------------------------------------------------------------------------
# 5. Walker-level logarithmic-derivative accumulator
# ---------------------------------------------------------------------------


def log_derivative_accumulator(
    log_abs_psi_fn: LogAbsPsiFn, params: Params, positions: Array
) -> Array:
    r"""Per-sample parameter log-derivative ``O_p(r) = d log|psi(params, r)| / d theta_p``.

    **Plain JAX autodiff (ordinary reverse-mode ``jax.grad``), not a
    closed-form-derivative-tower claim.** This differentiates the caller's
    ``log_abs_psi_fn`` with respect to its *parameters* (as opposed to
    :func:`langevin_step`, which differentiates with respect to the
    *position*); nothing here reuses ``omnibias.core.polynomials``.

    ``params`` may be an arbitrary JAX pytree (a ``NamedTuple`` such as
    :class:`omnibias.ferminet.restricted.Tier2Params`, a dict, ...); it is
    flattened with :func:`jax.flatten_util.ravel_pytree` into a single
    ``(n_params,)`` vector purely to get a stable, deterministic parameter
    ordering to index the output columns by -- the ordering is whatever
    ``ravel_pytree`` produces for that pytree structure (stable as long as
    the structure itself does not change), so a downstream consumer that
    needs to map a flat parameter-space vector back onto the pytree should
    call ``jax.flatten_util.ravel_pytree(params)`` again to get the matching
    ``unravel`` function.

    This is exactly the walker-level ingredient a stochastic-reconfiguration
    covariance matrix needs -- ``S_ij = <O_i* O_j> - <O_i><O_j>`` over the
    sample axis -- and is designed to plug into
    :func:`omnibias.curvature.natural_gradient.damped_solve` /
    ``natural_gradient_step``'s existing ``(P, P)`` Fisher / ``(P,)`` grad
    convention once a future work item wires up the actual covariance
    solve; **that wiring is not built here** (see the module docstring).

    ``positions`` is ``(n_samples, D)``. Returns ``(n_samples, n_params)``.
    """
    flat_params, unravel = ravel_pytree(params)

    def _log_abs_psi_of_flat(flat_p: Array, position: Array) -> Array:
        return log_abs_psi_fn(unravel(flat_p), position)

    per_sample_grad: Array = jax.vmap(
        jax.grad(_log_abs_psi_of_flat, argnums=0), in_axes=(None, 0)
    )(flat_params, positions)
    return per_sample_grad


__all__ = [
    "ChainDiagnostics",
    "LangevinResult",
    "LogAbsPsiFn",
    "MetropolisResult",
    "Params",
    "autocorrelation_function",
    "autodiff_grad_and_laplacian",
    "effective_sample_size",
    "energy_chain_diagnostics",
    "integrated_autocorrelation_time",
    "kinetic_energy_from_grad_lap",
    "langevin_sample",
    "langevin_step",
    "local_energy",
    "log_derivative_accumulator",
    "metropolis_sample",
    "metropolis_step",
    "standard_error_of_mean",
]

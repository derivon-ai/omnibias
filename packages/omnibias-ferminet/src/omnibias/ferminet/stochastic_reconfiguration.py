# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stochastic reconfiguration (SR): the natural-gradient VMC optimizer step.

This module is the first work item that actually *trains* on top of
:mod:`omnibias.ferminet.sampling`'s sampling substrate. It is explicitly a
**composition**, not a new optimizer: the only new math here is the
Monte-Carlo overlap-matrix / energy-gradient estimators
(:func:`sr_overlap_matrix`, :func:`sr_energy_gradient`); the linear solve
that turns those estimates into a parameter update is delegated, unmodified,
to :mod:`omnibias.curvature.natural_gradient` (:func:`damped_solve` /
:func:`natural_gradient_step`) -- functions written with no knowledge of
quantum Monte Carlo at all, for a generic ``(P, P)`` Fisher-like matrix and
``(P,)`` gradient. Feeding them a Monte-Carlo overlap matrix instead of the
closed-form GLM Fisher they were designed for is exactly the point of this
module; neither function's signature or body is touched.

Stochastic reconfiguration in one paragraph
--------------------------------------------
For a variational wavefunction :math:`\psi_\theta`, stochastic
reconfiguration (Sorella, *Phys. Rev. Lett.* 80, 4558 (1998); Sorella,
*Phys. Rev. B* 64, 024512 (2001)) preconditions the plain VMC energy
gradient by the **quantum geometric tensor** / overlap matrix

.. math::

   S_{ij} = \langle O_i^{*} O_j\rangle - \langle O_i^{*}\rangle\langle O_j\rangle,
   \qquad O_p(r) = \frac{\partial \log|\psi_\theta(r)|}{\partial \theta_p},

where :math:`\langle\cdot\rangle` is the Monte-Carlo sample average over
positions drawn from :math:`p(r)\propto|\psi_\theta(r)|^2`. This ``S`` is
exactly Fisher-information-shaped on parameter space -- the same role
:mod:`omnibias.curvature.natural_gradient` was built to precondition with --
so an SR step *is* a natural-gradient step with ``S`` standing in for the GLM
Fisher and the VMC energy gradient ``g`` standing in for the GLM loss
gradient. The update

.. math::

   \Delta\theta = (S + \lambda I)^{-1} g

is exactly :func:`omnibias.curvature.natural_gradient.damped_solve`, and
``theta - \mathrm{lr}\cdot\Delta\theta`` is exactly
:func:`omnibias.curvature.natural_gradient.natural_gradient_step`.
:func:`stochastic_reconfiguration_step` below does nothing more than: sample
positions (:mod:`omnibias.ferminet.sampling`'s walkers), accumulate ``O`` and
``E_L`` (:func:`omnibias.ferminet.sampling.log_derivative_accumulator` /
:func:`omnibias.ferminet.sampling.local_energy`), estimate ``(S, g)``, and
call that solve.

Conventions used here (stated explicitly, since the literature varies)
------------------------------------------------------------------------
* **Overlap matrix** (:func:`sr_overlap_matrix`): the *population*
  (``ddof=0``, divide by ``n_samples``, not ``n_samples - 1``) sample
  covariance of the log-derivative matrix ``O``, in matrix form
  ``S = <O^H O> - <O>^H <O>`` (``O^H`` the conjugate transpose; ``<.>``
  the sample mean over the leading axis). For a real-valued ``O`` -- the
  only case :mod:`omnibias.ferminet.sampling` currently produces, since
  ``log_abs_psi_fn`` returns the real quantity ``log|psi|`` and
  :func:`~omnibias.ferminet.sampling.log_derivative_accumulator`
  differentiates it with plain ``jax.grad`` -- this is exactly the real
  sample covariance matrix of ``O``'s columns, ``S_ij = <O_i O_j> - <O_i><O_j>``,
  with no conjugation to worry about. The conjugate-transpose form and a
  trailing :func:`jax.numpy.real` are kept only so the formula stays correct
  (and the result stays a real, symmetric matrix, matching
  :func:`omnibias.curvature.natural_gradient.damped_solve`'s ``(P, P)``
  real-matrix contract) for a hypothetical complex-parameter ansatz;
  ``jnp.conj`` is the identity on real dtypes, so nothing changes for the
  real case exercised by every test in this module.
* **Energy gradient** (:func:`sr_energy_gradient`):
  ``g_p = 2 * Re[<O_p^{*} E_L> - <O_p^{*}><E_L>]`` -- twice the (real part
  of the) sample covariance of ``conj(O_p)`` with the local energy ``E_L``.
  This is the standard VMC "log-derivative trick" energy-gradient estimator;
  see the derivation below for the exact sign/factor convention.

Derivation of the energy-gradient estimator
----------------------------------------------
Write the variational energy as an expectation over the sampled density
``p(r) = |psi_theta(r)|^2 / Z(theta)``:

.. math::

   E(\theta) = \int p(r)\, E_L(r)\, dr,
   \qquad E_L(r) = \frac{H\psi_\theta(r)}{\psi_\theta(r)}.

Differentiating the *log*-density with respect to ``theta_p`` (using
``p \propto |\psi|^2 = \exp(2\log|\psi|)``, so ``d log Z/d theta_p = 2<O_p>``):

.. math::

   \frac{\partial \log p(r)}{\partial \theta_p} = 2 O_p(r) - 2\langle O_p\rangle.

The score-function / log-derivative identity ``d<f>/dtheta = <f * dlogp/dtheta> + <df/dtheta>``
then gives

.. math::

   \frac{\partial E}{\partial \theta_p}
     = 2\langle O_p E_L\rangle - 2\langle O_p\rangle\langle E_L\rangle
       + \Bigl\langle \frac{\partial E_L}{\partial \theta_p}\Bigr\rangle.

The last term has zero expectation under ``p`` for *any* trial wavefunction
-- the standard VMC "zero-variance" identity (Umrigar, Wilson & Wilkins,
*Phys. Rev. Lett.* 60, 1719 (1988); Sorella 2001, Eq. 4) -- but only in
expectation over an *infinite* sample, not pointwise for a finite Monte-Carlo
batch, so dropping it turns an exact identity into the standard, finite-sample
**statistical estimator** used throughout the VMC/SR literature:

.. math::

   g_p \;:=\; 2\bigl(\langle O_p E_L\rangle - \langle O_p\rangle\langle E_L\rangle\bigr)
   \;\approx\; \frac{\partial E}{\partial \theta_p}.

**Sign/factor convention, stated explicitly**: the overall factor is ``+2``
(never ``-2`` or ``1``), and it is ``Cov(O_p, E_L)`` (not ``Cov(E_L, O_p)``
with a different conjugation order) -- consistent with treating ``theta``
as being updated by *descending* ``g``, i.e.
``theta <- theta - lr * (S + damping*I)^{-1} g``, exactly
:func:`omnibias.curvature.natural_gradient.natural_gradient_step`'s own
convention.

Statistical-estimator vs. exact-linear-algebra honesty
-----------------------------------------------------------
Per the repository's derivative-tower doctrine, keep these three registers
distinct -- do not blur them:

* **Monte-Carlo statistical estimator** (:func:`sr_overlap_matrix`,
  :func:`sr_energy_gradient`) -- both carry ordinary Monte-Carlo sampling
  noise that shrinks only as more (decorrelated) samples are drawn; neither
  is exact for a finite batch, and neither reuses the omnibias closed-form
  derivative tower.
* **Exact linear algebra** (:func:`omnibias.curvature.natural_gradient.damped_solve`
  / :func:`~omnibias.curvature.natural_gradient.natural_gradient_step`,
  called here unmodified) -- given the *already-estimated* ``(S, g)``, the
  damped solve ``(S + damping*I)^{-1} g`` is an exact, deterministic linear
  solve with no additional randomness; every bit of statistical noise in the
  final update lives upstream, in ``S`` and ``g`` themselves.
* **Plain JAX autodiff** (:func:`omnibias.ferminet.sampling.log_derivative_accumulator`,
  and :func:`omnibias.ferminet.sampling.local_energy`'s autodiff fallback) --
  already labeled as such in ``sampling.py``; this module never touches
  ``omnibias.core.polynomials`` and makes no closed-form-tower claim
  anywhere.

Why no conjugate-gradient solver here
----------------------------------------
:func:`omnibias.curvature.natural_gradient.damped_solve` currently does a
dense ``jnp.linalg.solve``, and the only existing conjugate-gradient
implementation in the repository is ``omnibias.torch.optim.conjugate_gradient``
-- a PyTorch, not JAX, primitive, so it cannot be *reused* here without a
from-scratch JAX port (which would not be "reusing ... as-is"). Every test in
this module -- the hand-computable unit test and the toy harmonic-oscillator
convergence demonstration -- uses ``n_params`` on the order of 1-6, for which
the dense ``(P, P)`` direct solve ``damped_solve`` already performs is fast
and numerically robust; there is no problem size in scope here where CG is
actually needed.
:attr:`StochasticReconfigurationResult.cg_iterations` is kept as a field
(always ``None`` today) so a future CG-based solve path -- the standard
practical choice once ``S`` grows large and ill-conditioned, per Sorella
(2001) -- can be wired in later without changing this result type's shape.

``jit`` safety
---------------
Like every function in :mod:`omnibias.ferminet.sampling`,
:func:`stochastic_reconfiguration_step` takes several plain **Python
callables** (``log_abs_psi_fn``, ``kinetic_fn``, ``potential_fn``,
``sample_fn``); wrapping it in :func:`jax.jit` requires listing all four in
``static_argnames``, plus ``damping`` (``omnibias.curvature.natural_gradient.damped_solve``
branches on ``damping``'s *concrete* value, so it cannot be a traced
array under ``jit``)::

    jax.jit(
        stochastic_reconfiguration_step,
        static_argnames=("kinetic_fn", "potential_fn", "sample_fn", "damping"),
    )

(exercised directly by ``tests/test_stochastic_reconfiguration.py::TestJitSafety``).
``learning_rate`` has no such restriction (it is only ever used arithmetically)
and may safely be a traced array.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from omnibias.curvature.natural_gradient import natural_gradient_step
from omnibias.ferminet.sampling import (
    LangevinResult,
    LogAbsPsiFn,
    MetropolisResult,
    Params,
    local_energy,
    log_derivative_accumulator,
    standard_error_of_mean,
)

#: A pre-configured walker: ``(log_abs_psi_fn, params, init_positions, key) ->
#: MetropolisResult | LangevinResult``. Callers bind ``n_steps`` / ``step_size``
#: (and, for the Langevin walker, ``adjusted``) with :func:`functools.partial`
#: before passing it in, exactly like ``kinetic_fn`` / ``potential_fn`` in
#: :func:`omnibias.ferminet.sampling.local_energy`.
SampleFn = Callable[[LogAbsPsiFn, Params, Array, Array], "MetropolisResult | LangevinResult"]


def sr_overlap_matrix(log_derivatives: Array) -> Array:
    r"""Monte-Carlo overlap (quantum geometric tensor) matrix ``S``.

    **Monte-Carlo statistical estimator** -- carries ordinary sampling
    noise, exact only in the infinite-sample limit. Computes the
    *population* (``ddof=0``) sample covariance of ``log_derivatives``'
    columns,

    .. math::

       S_{ij} = \frac{1}{N}\sum_{n=1}^N \overline{(O_{n,i} - \langle O_i\rangle)}\,
                (O_{n,j} - \langle O_j\rangle),

    which expands to the textbook SR overlap matrix
    ``S_ij = <O_i^* O_j> - <O_i^*><O_j>`` (see the module docstring for the
    real-vs-complex convention). A trailing :func:`jax.numpy.real` guards
    against a spurious near-zero imaginary part, so the result is always a
    real, symmetric ``(n_params, n_params)`` matrix -- exactly
    :func:`omnibias.curvature.natural_gradient.damped_solve`'s expected
    ``(P, P)`` shape.

    ``log_derivatives`` is ``(n_samples, n_params)``, e.g. the output of
    :func:`omnibias.ferminet.sampling.log_derivative_accumulator`. Returns
    ``(n_params, n_params)``.
    """
    if log_derivatives.ndim != 2:
        raise ValueError(
            "log_derivatives must be (n_samples, n_params), got shape "
            f"{tuple(log_derivatives.shape)}"
        )
    n_samples = log_derivatives.shape[0]
    mean = jnp.mean(log_derivatives, axis=0)  # (n_params,)
    centered = log_derivatives - mean[None, :]
    cov = (jnp.conj(centered).T @ centered) / n_samples
    out: Array = jnp.real(cov)
    return out


def sr_energy_gradient(log_derivatives: Array, local_energies: Array) -> Array:
    r"""Monte-Carlo VMC energy-gradient estimator ``g``.

    **Monte-Carlo statistical estimator** -- carries ordinary sampling
    noise, exact only in the infinite-sample limit (see the module
    docstring for the full derivation and the exact sign/factor
    convention):

    .. math::

       g_p = 2\,\mathrm{Re}\bigl[\langle O_p^{*} E_L\rangle
                                  - \langle O_p^{*}\rangle\langle E_L\rangle\bigr].

    ``log_derivatives`` is ``(n_samples, n_params)`` (e.g.
    :func:`omnibias.ferminet.sampling.log_derivative_accumulator`'s output)
    and ``local_energies`` is ``(n_samples,)`` (e.g.
    :func:`omnibias.ferminet.sampling.local_energy`'s output), sharing the
    same sample axis. Returns ``(n_params,)``.
    """
    if log_derivatives.ndim != 2:
        raise ValueError(
            "log_derivatives must be (n_samples, n_params), got shape "
            f"{tuple(log_derivatives.shape)}"
        )
    if local_energies.ndim != 1 or local_energies.shape[0] != log_derivatives.shape[0]:
        raise ValueError(
            "local_energies must be (n_samples,) with n_samples = "
            f"{log_derivatives.shape[0]}, got shape {tuple(local_energies.shape)}"
        )
    mean_O = jnp.mean(log_derivatives, axis=0)  # (n_params,)
    mean_E = jnp.mean(local_energies)  # scalar
    cross = jnp.mean(jnp.conj(log_derivatives) * local_energies[:, None], axis=0)  # (n_params,)
    out: Array = 2.0 * jnp.real(cross - jnp.conj(mean_O) * mean_E)
    return out


class StochasticReconfigurationResult(NamedTuple):
    r"""Output of one :func:`stochastic_reconfiguration_step`.

    Attributes
    ----------
    params
        The updated ansatz parameters (same pytree structure as the input
        ``params``).
    positions
        ``(n_walkers, D)`` -- the sampled positions this step actually used
        (the walker's final positions after its burn-in/decorrelation
        steps), so a training loop can pass them back in as the next
        step's ``init_positions`` to warm-start the chain.
    energy_mean
        Scalar Monte-Carlo estimate of the energy, ``mean(local_energies)``.
    energy_standard_error
        Scalar autocorrelation-corrected standard error of ``energy_mean``,
        from :func:`omnibias.ferminet.sampling.standard_error_of_mean`.
    overlap_matrix
        The ``(n_params, n_params)`` Monte-Carlo overlap matrix ``S`` this
        step estimated (:func:`sr_overlap_matrix`) -- exposed so a caller
        can cross-check the update against an independent call to
        :func:`omnibias.curvature.natural_gradient.damped_solve` /
        :func:`~omnibias.curvature.natural_gradient.natural_gradient_step`.
    energy_gradient
        The ``(n_params,)`` Monte-Carlo energy gradient ``g`` this step
        estimated (:func:`sr_energy_gradient`), exposed for the same reason.
    damping
        The Tikhonov damping ``lambda`` actually used in
        ``(S + lambda I)^{-1} g``.
    cg_iterations
        Reserved for a future conjugate-gradient solve path; always
        ``None`` today, since this module reuses
        :func:`omnibias.curvature.natural_gradient.natural_gradient_step`'s
        existing exact direct solve (see the module docstring, "Why no
        conjugate-gradient solver here").
    """

    params: Params
    positions: Array
    energy_mean: Array
    energy_standard_error: Array
    overlap_matrix: Array
    energy_gradient: Array
    damping: float
    cg_iterations: int | None


def stochastic_reconfiguration_step(
    log_abs_psi_fn: LogAbsPsiFn,
    params: Params,
    init_positions: Array,
    key: Array,
    *,
    kinetic_fn: Callable[[Params, Array], Array],
    potential_fn: Callable[[Array], Array],
    sample_fn: SampleFn,
    damping: float = 1e-3,
    learning_rate: float = 1.0,
) -> StochasticReconfigurationResult:
    r"""One sample -> accumulate -> solve -> update stochastic-reconfiguration step.

    Composes, in order:

    1. **Sample** -- ``sample_fn(log_abs_psi_fn, params, init_positions, key)``,
       a caller-bound :func:`omnibias.ferminet.sampling.metropolis_sample` or
       :func:`omnibias.ferminet.sampling.langevin_sample` (bind ``n_steps`` /
       ``step_size`` / ``adjusted`` with :func:`functools.partial`). Only the
       walker's final ``positions`` are used as this step's Monte-Carlo
       batch.
    2. **Accumulate** -- the per-sample parameter log-derivative ``O``
       (:func:`omnibias.ferminet.sampling.log_derivative_accumulator`) and
       local energy ``E_L`` (:func:`omnibias.ferminet.sampling.local_energy`,
       given the caller-supplied ``kinetic_fn`` / ``potential_fn``).
    3. **Solve** -- the Monte-Carlo overlap matrix ``S``
       (:func:`sr_overlap_matrix`) and energy gradient ``g``
       (:func:`sr_energy_gradient`), fed *unmodified* into
       :func:`omnibias.curvature.natural_gradient.natural_gradient_step`
       (which itself calls
       :func:`omnibias.curvature.natural_gradient.damped_solve`) on the
       flattened parameter vector (:func:`jax.flatten_util.ravel_pytree`,
       the same flattening :func:`~omnibias.ferminet.sampling.log_derivative_accumulator`
       already uses, so the ordering of ``S`` / ``g`` matches the ordering
       ``ravel_pytree`` gives the updated ``theta`` here).
    4. **Update** -- unravel the new flat parameter vector back onto
       ``params``'s original pytree structure.

    Returns a :class:`StochasticReconfigurationResult` bundling the updated
    params with the energy estimate, its standard error, the ``(S, g)``
    this step estimated, the damping used, and ``cg_iterations`` (always
    ``None``; see the module docstring).

    See the module docstring's "``jit`` safety" section before wrapping this
    function in :func:`jax.jit`.
    """
    sample_result = sample_fn(log_abs_psi_fn, params, init_positions, key)
    positions = sample_result.positions  # (n_walkers, D)

    log_derivatives = log_derivative_accumulator(log_abs_psi_fn, params, positions)
    local_energies = local_energy(kinetic_fn, potential_fn, params, positions)

    overlap = sr_overlap_matrix(log_derivatives)
    grad = sr_energy_gradient(log_derivatives, local_energies)

    flat_params, unravel = ravel_pytree(params)
    new_flat_params = natural_gradient_step(
        flat_params, grad, overlap, learning_rate=learning_rate, damping=damping
    )
    new_params = unravel(new_flat_params)

    return StochasticReconfigurationResult(
        params=new_params,
        positions=positions,
        energy_mean=jnp.mean(local_energies),
        energy_standard_error=standard_error_of_mean(local_energies),
        overlap_matrix=overlap,
        energy_gradient=grad,
        damping=damping,
        cg_iterations=None,
    )


__all__ = [
    "SampleFn",
    "StochasticReconfigurationResult",
    "sr_energy_gradient",
    "sr_overlap_matrix",
    "stochastic_reconfiguration_step",
]

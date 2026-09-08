# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fermi-Dirac occupancy and thermodynamic potentials (theory 04-03).

With ``z = -beta * (e - mu)`` the Fermi-Dirac occupancy *is* the sigmoid:
``f(e) = sigma(z)``. Every derivative of ``f`` with respect to energy or
chemical potential is therefore a direct read of the closed-form sigmoid
tower in :mod:`omnibias.core.polynomials`, one sigmoid evaluation
regardless of order. The Fermi entropy per state and the grand-potential
density are the ``softplus`` half of the same tower (``softplus' =
sigmoid``), so both are exact to arbitrary order as well:

.. math::

    f(e) = \sigma(z), \qquad
    s(z) = \mathrm{softplus}(z) - z\,\sigma(z), \qquad
    \omega(z) = -\tfrac{1}{\beta}\,\mathrm{softplus}(z).

Two collapse senses are named in this module and must not be conflated:

* The founding bias collapse (spread ``delta -> 0``, ``K`` biases coalesce
  into a smooth ``sigma^(K-1)``) never appears here; every identity in
  this module is a finite-order derivative read, not a ``delta -> 0``
  limit.
* :func:`zero_temperature_occupancy` evaluates the ``beta -> inf`` limit,
  the T=0 Fermi step. That limit is *literally* the founding
  **temperature collapse** (``omnibias.core.collapse.schema``'s founding
  spec: parameter ``beta``, limit ``inf``, surviving object
  ``indicator``, the feasibility sense). This module therefore never
  requests a new collapse registry slot for it; here ``beta`` is a
  genuine physical inverse temperature rather than an annealing metaphor,
  but the limit and its surviving 0/1 indicator are the same object
  already minted by the founding spec.

Do not conflate either sense with the other, and do not read a
``beta -> inf`` step as evidence of anything about interactions,
density-functional theory, or a thermodynamic limit -- see
:func:`honesty_payload`.

Scope. Non-interacting fermions in a single band, with an externally
supplied (or absent) density of states. Not a continuum thermodynamic
limit, not a many-body solve, not density-functional theory, and no
phase-transition claim -- mirroring the scope wall in
:mod:`omnibias.core.verified.lattice_ground_state`.

Note on the module name: :mod:`omnibias.core` re-exports :func:`occupancy`
(this module's own function) at package level, per the repository's
"regenerate ``__all__``" convention. Because of that, once
:mod:`omnibias.core` has been imported, ``omnibias.core.occupancy`` is the
*function*, not this submodule -- ordinary Python attribute shadowing, not a
bug. Import specific names directly (``from omnibias.core.occupancy import
occupancy_derivatives``), which resolves against the fully-qualified module
in ``sys.modules`` and is unaffected by the shadow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.core.polynomials import sigmoid_polynomial_coeffs
from omnibias.core.verified.coeffs import bernoulli_number_exact


def honesty_payload() -> dict[str, bool]:
    """Permanently-false claims this module must never assert.

    ``founding_bias_collapse`` and ``temperature_collapse`` describe this
    module's own axis (``beta``, a physical inverse temperature): neither
    limit is taken by the exact-derivative functions in this module.
    ``requests_new_collapse_registry_slot`` records that
    :func:`zero_temperature_occupancy` evaluates the founding temperature
    collapse's value directly rather than seeking a new
    ``omnibias.core.collapse`` registry entry for it.
    """
    return {
        "founding_bias_collapse": False,
        "temperature_collapse": False,
        "requests_new_collapse_registry_slot": False,
        "dft_solved_claim": False,
        "many_body_solved_claim": False,
        "interacting_system_claim": False,
        "thermodynamic_limit_taken": False,
        "phase_transition_proved": False,
        "theorem_prover_verified": False,
    }


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + float(c)
    return acc


def _sigmoid_value(z: float) -> float:
    """Numerically stable ``sigmoid(z)``, mirroring the two-branch form used
    by the framework-native ``torch.sigmoid`` / ``jax.nn.sigmoid`` twins."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _softplus_value(z: float) -> float:
    """Numerically stable ``softplus(z) = ln(1 + e^z)``."""
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _sigma_tower(z: float, order: int) -> list[float]:
    """``[sigma(z), sigma'(z), ..., sigma^(order)(z)]`` -- one sigmoid call,
    then Horner over the shared Eulerian coefficients."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    s = _sigmoid_value(z)
    rows = [s]
    for k in range(1, order + 1):
        rows.append(_horner(sigmoid_polynomial_coeffs(k), s))
    return rows


def _entropy_z_tower(z: float, order: int) -> list[float]:
    r"""``[s(z), s'(z), ..., s^(order)(z)]``, the Fermi-entropy tower in ``z``.

    ``s(z) = softplus(z) - z sigma(z)``. Differentiating,
    ``s'(z) = -z sigma'(z)``, and by the Leibniz rule on the product
    ``z * sigma'(z)``, for ``n >= 1``:
    ``s^(n)(z) = -(z sigma^(n)(z) + (n - 1) sigma^(n - 1)(z))``.
    """
    sigma_tower = _sigma_tower(z, order)
    rows = [_softplus_value(z) - z * sigma_tower[0]]
    for n in range(1, order + 1):
        rows.append(-(z * sigma_tower[n] + (n - 1) * sigma_tower[n - 1]))
    return rows


@dataclass(frozen=True)
class FermiModel:
    """Non-interacting Fermi-Dirac occupancy model.

    ``beta`` is the inverse temperature (``1 / (k_B T)``, must be finite and
    strictly positive) and ``mu`` is the chemical potential.
    """

    beta: float
    mu: float = 0.0

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be finite and > 0, got {self.beta!r}")
        mu = float(self.mu)
        if not math.isfinite(mu):
            raise ValueError(f"mu must be finite, got {self.mu!r}")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "mu", mu)


def reduced_argument(model: FermiModel, energy: float) -> float:
    """``z = -beta * (energy - mu)``, the sigmoid tower's native argument."""
    return -model.beta * (float(energy) - model.mu)


def occupancy(model: FermiModel, energy: float) -> float:
    """Fermi-Dirac occupancy ``f(e) = sigma(z)``, ``z = -beta (e - mu)``."""
    z = reduced_argument(model, energy)
    return _sigmoid_value(z)


def occupancy_derivatives(
    model: FermiModel, energy: float, *, order: int
) -> tuple[float, ...]:
    """``(f(e), df/de, ..., d^order f / de^order)``.

    ``d^n f / de^n = (-beta)^n * sigma^(n)(z)`` since ``dz/de = -beta``.
    """
    z = reduced_argument(model, energy)
    tower = _sigma_tower(z, order)
    beta = model.beta
    return tuple(((-beta) ** n) * tower[n] for n in range(order + 1))


def occupancy_mu_derivatives(
    model: FermiModel, energy: float, *, order: int
) -> tuple[float, ...]:
    """``(f(e), df/dmu, ..., d^order f / dmu^order)``.

    ``mu`` enters ``z`` affinely with the opposite sign of ``e``
    (``dz/dmu = +beta``), so ``d^n f / dmu^n = beta^n * sigma^(n)(z)``.
    """
    z = reduced_argument(model, energy)
    tower = _sigma_tower(z, order)
    beta = model.beta
    return tuple((beta**n) * tower[n] for n in range(order + 1))


def thermal_broadening(model: FermiModel, energy: float) -> float:
    """``-df/de = beta * sigma'(z)``, the thermal smearing width of the step."""
    _, d1 = occupancy_derivatives(model, energy, order=1)
    return -d1


def entropy_derivatives(
    model: FermiModel, energy: float, *, order: int
) -> tuple[float, ...]:
    """``(s(e), ds/de, ..., d^order s / de^order)`` of the Fermi entropy per state.

    ``d^n s / de^n = (-beta)^n * s^(n)(z)`` via :func:`_entropy_z_tower`.
    """
    z = reduced_argument(model, energy)
    tower = _entropy_z_tower(z, order)
    beta = model.beta
    return tuple(((-beta) ** n) * tower[n] for n in range(order + 1))


def entropy_per_state(model: FermiModel, energy: float) -> float:
    """Fermi entropy per state, ``s(z) = softplus(z) - z sigma(z)``.

    ``s(0) = ln 2`` (maximum entropy at half filling) and ``s -> 0`` in both
    tails (``sigma -> 0`` or ``sigma -> 1``).
    """
    return entropy_derivatives(model, energy, order=0)[0]


def grand_potential_density(model: FermiModel, energy: float) -> float:
    r"""Grand-potential density per state, ``omega(z) = -(1/beta) softplus(z)``.

    ``d omega / d mu = -sigma(z) = -f(e)``, reproducing ``dOmega/dmu = -N``
    exactly (one sigmoid evaluation, no finite difference).
    """
    z = reduced_argument(model, energy)
    return -_softplus_value(z) / model.beta


def occupancy_window(model: FermiModel, e_lo: float, e_hi: float) -> float:
    r"""Exact electron count in ``[e_lo, e_hi]`` for a *constant* density of
    states, the ``band`` role of :mod:`omnibias.core.spec`.

    ``d/de [-(1/beta) softplus(z(e))] = sigma(z(e)) = f(e)``, so
    ``-(1/beta) softplus(z(e))`` is a closed-form antiderivative of ``f``.
    Hence

    .. math::

        \int_{e_{lo}}^{e_{hi}} f(e)\,de
            = \tfrac{1}{\beta}\bigl[\mathrm{softplus}(z_{lo})
                - \mathrm{softplus}(z_{hi})\bigr],

    nonnegative because ``softplus`` is increasing and ``z`` decreases in
    ``e``. Requires ``e_lo <= e_hi``.
    """
    lo, hi = float(e_lo), float(e_hi)
    if lo > hi:
        raise ValueError(f"occupancy_window requires e_lo <= e_hi, got {lo} > {hi}")
    z_lo = reduced_argument(model, lo)
    z_hi = reduced_argument(model, hi)
    return (_softplus_value(z_lo) - _softplus_value(z_hi)) / model.beta


def _zeta_even_float(m: int) -> float:
    r"""``zeta(2m)`` via the exact Bernoulli-number closed form, narrowed to
    ``float`` once: ``zeta(2m) = (-1)^(m+1) B_{2m} (2 pi)^{2m} / (2 (2m)!)``."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    b2m = bernoulli_number_exact(2 * m)
    sign = -1.0 if (m % 2 == 0) else 1.0
    return sign * float(b2m) * (2.0 * math.pi) ** (2 * m) / (2.0 * math.factorial(2 * m))


def sommerfeld_moment(order: int) -> float:
    r"""Raw moment ``M_order = integral_{-inf}^{inf} sigma'(z) z^order dz``.

    Odd orders vanish exactly by the symmetry ``sigma'(-z) = sigma'(z)``.
    ``order = 0`` gives ``1`` (``sigma`` rises from ``0`` to ``1``). Even
    orders ``order = 2n`` (``n >= 1``) are closed form,
    ``M_{2n} = 2 (2n)! eta(2n)`` with the Dirichlet eta function
    ``eta(s) = (1 - 2^{1-s}) zeta(s)``, read off the exact Bernoulli-number
    closed form for ``zeta(2n)`` (:func:`_zeta_even_float`).
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return 1.0
    if order % 2 == 1:
        return 0.0
    n = order // 2
    zeta_2n = _zeta_even_float(n)
    eta_2n = (1.0 - 2.0 ** (1 - 2 * n)) * zeta_2n
    return 2.0 * math.factorial(order) * eta_2n


def sommerfeld_coefficient(n: int) -> float:
    r"""Standard dimensionless Sommerfeld coefficient
    ``a_n = sommerfeld_moment(2n) / (2n)! = 2 eta(2n)``.

    Ashcroft & Mermin convention: ``a_1 = pi^2 / 6``,
    ``a_2 = 7 pi^4 / 360``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return sommerfeld_moment(2 * n) / math.factorial(2 * n)


def zero_temperature_occupancy(model: FermiModel, energy: float) -> float:
    r"""The T=0 step function value, the ``beta -> inf`` limit.

    This is the founding **temperature collapse** (``omnibias.core.
    collapse.schema``'s founding spec: parameter ``beta``, limit ``inf``,
    surviving object ``indicator``). :class:`FermiModel` never takes this
    limit itself; this function evaluates the closed-form step directly as
    the honest reference for what ``beta -> inf`` collapses to, and does
    not request a new collapse registry slot (see :func:`honesty_payload`).
    At ``energy == mu`` the convention is the symmetric ``0.5``, matching
    ``sigmoid(0)``.
    """
    e = float(energy)
    if e < model.mu:
        return 1.0
    if e > model.mu:
        return 0.0
    return 0.5


__all__ = [
    "FermiModel",
    "entropy_derivatives",
    "entropy_per_state",
    "grand_potential_density",
    "honesty_payload",
    "occupancy",
    "occupancy_derivatives",
    "occupancy_mu_derivatives",
    "occupancy_window",
    "reduced_argument",
    "sommerfeld_coefficient",
    "sommerfeld_moment",
    "thermal_broadening",
    "zero_temperature_occupancy",
]

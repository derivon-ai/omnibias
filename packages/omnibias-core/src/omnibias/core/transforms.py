# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form integral-transform identities for the activation dictionary.

This module is the single source of truth for *which* Laplace / Fourier /
Mellin transforms omnibias ships in closed form, what each one equals, and
where it converges. It is pure Python -- strings, floats and frozen
dataclasses, no tensor library -- exactly like :mod:`omnibias.core.polynomials`
is the shared source of the derivative-tower coefficients. The backend twins
:mod:`omnibias.torch.transforms` and :mod:`omnibias.jax.transforms` evaluate
these identities on tensors and are checked against this table by a coverage
test, so the documentation and the code cannot drift apart.

Conventions
-----------
Fixed once, here, so both backends and all documentation agree:

.. math::

    \mathcal{L}[\sigma](s) &= \int_0^\infty \sigma(z)\, e^{-sz}\, dz, \\
    \mathcal{F}[\sigma](\xi) &= \int_{-\infty}^{\infty} \sigma(z)\,
        e^{-i\xi z}\, dz, \\
    \mathcal{M}[\sigma](s) &= \int_0^\infty \sigma(z)\, z^{s-1}\, dz.

The Fourier convention is the non-unitary *angular-frequency* one: no
``1/sqrt(2 pi)`` prefactor, and ``e^{-i xi z}`` in the kernel. Under this
convention the Gaussian is its own transform up to ``sqrt(2 pi)``, and
``sech`` is its own transform up to ``pi`` and a ``pi/2`` dilation.

Honesty labels
--------------
Every entry below is **closed form**: a finite composition of elementary and
standard special functions (``exp``, ``erfcx``, ``digamma``, ``lgamma``,
``zeta``), evaluated in one shot with no quadrature, series truncation or
iteration. That is a stronger claim than "analytic", and it is the reason the
transforms are exact to machine precision rather than to a tolerance.

What is deliberately *not* registered matters just as much. Three families of
gap appear in the tables, each recorded as an :class:`ExcludedTransform` with
its reason:

* **Divergent.** The defining integral does not converge for the activation as
  omnibias registers it -- for example the Mellin transform of ``exp``, since
  omnibias's ``exp`` is :math:`e^{+z}` and :math:`\int_0^\infty e^{z} z^{s-1}
  dz` diverges at every ``s``. The classical ``Gamma(s)`` pair belongs to the
  *decaying* exponential :math:`e^{-z}`, which is a different function.
* **Distributional.** The transform exists only as a tempered distribution --
  Dirac deltas and Cauchy principal values -- which no tensor-valued kernel can
  return. ``relu``, ``sigmoid`` and ``tanh`` are not in :math:`L^1(\mathbb{R})`,
  so their Fourier transforms fall here. Shipping a "kernel" for these would be
  silently wrong, so they stay ``None``.
* **Complementary.** The convergent classical identity belongs to the
  activation's *complement* rather than the activation itself. The Fermi-Dirac
  integral :math:`\Gamma(s)\eta(s)` is the Mellin transform of
  :math:`1 - \mathrm{sigmoid}(z) = \mathrm{sigmoid}(-z)`, not of ``sigmoid``,
  whose own Mellin integral diverges. omnibias ships that identity under its
  own name (``fermi_dirac_mellin`` in each backend) instead of mislabelling it
  as ``sigmoid.transforms.mellin``.
* **Conditional.** The integral converges only conditionally (as an Abel- or
  Cesaro-regularised limit), never absolutely, so there is no quadrature that
  could validate it. The Mellin transforms of ``sin`` and ``cos`` fall here.
* **Unavailable.** A closed form exists on paper but calls a special function
  neither backend provides, so evaluating it would mean shipping a truncated
  series behind a "closed form" label. The Mellin transform of ``sech`` needs
  the Dirichlet beta function and falls here.

Scope wall on the Fermi-Dirac integral
--------------------------------------
:data:`FERMI_DIRAC_MELLIN` is registered for ``Re(s) > 1`` only. The integral
itself converges for ``Re(s) > 0``, but omnibias evaluates ``eta(s) =
(1 - 2^{1-s}) zeta(s)`` through the Riemann zeta function, and the rigorous
zeta machinery in :mod:`omnibias.core.verified.dirichlet` is sound on
``Re(s) > 1`` only: ``zeta_enclosure`` raises below that line, and analytic
continuation (hence anything touching the critical strip or the Riemann
hypothesis) is a recorded *external obligation* that omnibias never infers.
The transform layer honours the same wall rather than quietly extending past
the region where the project can defend its numbers.

Tempering
---------
Transform kernels propagate through :func:`omnibias.core.spec.tempered` by the
exact scaling laws in :func:`omnibias.core.spec.make_tempered_transforms`; the
laws are recorded here as :data:`TEMPERING_LAWS` for documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: The three transforms omnibias models.
TransformName = Literal["laplace", "fourier", "mellin"]

#: Reason codes for a transform that is *not* registered.
ExclusionReason = Literal[
    "divergent", "distributional", "complementary", "conditional", "unavailable"
]

#: All transform names, in the canonical order used by tables and docs.
TRANSFORM_NAMES: tuple[TransformName, ...] = ("laplace", "fourier", "mellin")

#: The spectral variable each transform is evaluated at.
SPECTRAL_VARIABLE: dict[TransformName, str] = {
    "laplace": "s",
    "fourier": "xi",
    "mellin": "s",
}

#: Defining integral of each transform, as rendered mathematics.
TRANSFORM_DEFINITION: dict[TransformName, str] = {
    "laplace": "int_0^inf sigma(z) exp(-s z) dz",
    "fourier": "int_{-inf}^{inf} sigma(z) exp(-i xi z) dz",
    "mellin": "int_0^inf sigma(z) z^(s-1) dz",
}

#: Exact behaviour of each transform under ``f_beta(z) = beta^-p g(beta z)``
#: for ``beta > 0``. Implemented by
#: :func:`omnibias.core.spec.make_tempered_transforms`.
TEMPERING_LAWS: dict[TransformName, str] = {
    "laplace": "L[f_beta](s) = beta^-(p+1) L[g](s / beta)",
    "fourier": "F[f_beta](xi) = beta^-(p+1) F[g](xi / beta)",
    "mellin": "M[f_beta](s) = beta^(-p-s) M[g](s)",
}


@dataclass(frozen=True)
class TransformIdentity:
    """One closed-form transform pair that omnibias registers.

    Attributes
    ----------
    activation : str
        Registry name of the activation, as returned by ``list_activations()``.
    transform : TransformName
        Which of the three transforms this identity gives.
    expression : str
        The closed form, written in the spectral variable of ``transform``.
    region : str
        Region of convergence of the defining integral, as a condition on the
        spectral variable. ``"all real s"`` marks an entire function.
    evaluated_with : str
        The special functions the backend kernels actually call. Recorded
        because the evaluation route, not just the formula, determines the
        numerical behaviour (see ``erfcx`` on the Gaussian Laplace).
    min_argument : float, optional
        Infimum of the real region of convergence: the identity is valid for
        spectral arguments strictly greater than this. ``None`` marks an entire
        function, valid on the whole real line. This is the machine-readable
        form of :attr:`region`, and it is what the backend layer wrappers use
        to keep a *learnable* spectral variable inside the region during
        training instead of drifting into the divergent half-plane.
    note : str
        Anything a caller must know beyond the formula itself.
    """

    activation: str
    transform: TransformName
    expression: str
    region: str
    evaluated_with: str
    min_argument: float | None = None
    note: str = ""


@dataclass(frozen=True)
class ExcludedTransform:
    """A transform omnibias deliberately does *not* register, and why.

    These entries are the honest half of the coverage story: they turn "we did
    not get to it" into a checkable statement about the mathematics. The
    registry-coverage test asserts that every excluded pair really is ``None``
    on the corresponding :class:`~omnibias.core.spec.TransformKernels`.
    """

    activation: str
    transform: TransformName
    reason: ExclusionReason
    detail: str


# --------------------------------------------------------------------------- #
# Laplace:  L[sigma](s) = int_0^inf sigma(z) exp(-s z) dz
# --------------------------------------------------------------------------- #
LAPLACE_IDENTITIES: tuple[TransformIdentity, ...] = (
    TransformIdentity(
        activation="exp",
        transform="laplace",
        expression="1 / (s - 1)",
        region="s > 1",
        evaluated_with="elementary",
        min_argument=1.0,
        note="Simple pole at s = 1, where exp(z) stops being dominated by exp(-s z).",
    ),
    TransformIdentity(
        activation="relu",
        transform="laplace",
        expression="1 / s^2",
        region="s > 0",
        evaluated_with="elementary",
        min_argument=0.0,
        note="On [0, inf) relu(z) = z, so this is the Laplace transform of the ramp.",
    ),
    TransformIdentity(
        activation="sin",
        transform="laplace",
        expression="1 / (s^2 + 1)",
        region="s > 0",
        evaluated_with="elementary",
        min_argument=0.0,
    ),
    TransformIdentity(
        activation="cos",
        transform="laplace",
        expression="s / (s^2 + 1)",
        region="s > 0",
        evaluated_with="elementary",
        min_argument=0.0,
    ),
    TransformIdentity(
        activation="sinh",
        transform="laplace",
        expression="1 / (s^2 - 1)",
        region="s > 1",
        evaluated_with="elementary",
        min_argument=1.0,
        note="Pole at s = 1: sinh grows like exp(z) / 2.",
    ),
    TransformIdentity(
        activation="cosh",
        transform="laplace",
        expression="s / (s^2 - 1)",
        region="s > 1",
        evaluated_with="elementary",
        min_argument=1.0,
        note="Pole at s = 1: cosh grows like exp(z) / 2.",
    ),
    TransformIdentity(
        activation="gaussian",
        transform="laplace",
        expression="sqrt(pi/2) exp(s^2/2) erfc(s / sqrt 2)",
        region="all real s",
        evaluated_with="erfcx",
        min_argument=None,
        note=(
            "Entire in s because exp(-z^2/2) outruns every exponential. Evaluated as "
            "sqrt(pi/2) erfcx(s / sqrt 2): the exp(s^2/2) and erfc factors overflow "
            "and underflow respectively past s ~ 38, but their product is O(1/s), and "
            "the scaled complementary error function computes it directly."
        ),
    ),
    TransformIdentity(
        activation="sigmoid",
        transform="laplace",
        expression="Phi(-1, 1, s) = (psi((s+1)/2) - psi(s/2)) / 2",
        region="s > 0",
        evaluated_with="digamma",
        min_argument=0.0,
        note=(
            "Expanding 1/(1 + exp(-z)) as an alternating geometric series gives the "
            "Lerch transcendent Phi(-1, 1, s) = sum_k (-1)^k / (s + k); the digamma "
            "form is the same number in one closed-form call. omnibias evaluates the "
            "digamma form and uses omnibias.fractional's lerch as a test-only oracle, "
            "so omnibias-torch/-jax never depend on omnibias-fractional."
        ),
    ),
    TransformIdentity(
        activation="tanh",
        transform="laplace",
        expression="Phi(-1, 1, s/2) - 1/s = (psi(s/4 + 1/2) - psi(s/4)) / 2 - 1/s",
        region="s > 0",
        evaluated_with="digamma",
        min_argument=0.0,
        note=(
            "tanh(z) = 1 - 2 sum_{k>=1} (-1)^(k-1) exp(-2kz); the -1/s is the "
            "transform of the constant 1 that tanh saturates to. Both terms blow up "
            "like 1/s as s -> 0+ but only partially cancel, so the transform itself "
            "diverges there -- correctly, since int_0^inf tanh diverges."
        ),
    ),
    TransformIdentity(
        activation="sech",
        transform="laplace",
        expression="Phi(-1, 1, (s+1)/2) = (psi((s+3)/4) - psi((s+1)/4)) / 2",
        region="s > -1",
        evaluated_with="digamma",
        min_argument=-1.0,
        note=(
            "sech decays like 2 exp(-z), so the region of convergence extends to the "
            "left of the origin, unlike the saturating activations. L[sech](0) = pi/2, "
            "the total area under sech."
        ),
    ),
)

# --------------------------------------------------------------------------- #
# Fourier:  F[sigma](xi) = int_R sigma(z) exp(-i xi z) dz
# --------------------------------------------------------------------------- #
FOURIER_IDENTITIES: tuple[TransformIdentity, ...] = (
    TransformIdentity(
        activation="gaussian",
        transform="fourier",
        expression="sqrt(2 pi) exp(-xi^2 / 2)",
        region="all real xi",
        evaluated_with="elementary",
        min_argument=None,
        note="The fixed point of the Fourier transform, up to the sqrt(2 pi) of this convention.",
    ),
    TransformIdentity(
        activation="sech",
        transform="fourier",
        expression="pi sech(pi xi / 2)",
        region="all real xi",
        evaluated_with="elementary",
        min_argument=None,
        note=(
            "The second self-reciprocal profile: sech transforms to a dilated sech. "
            "This is why sech is the natural soliton / Poschl-Teller envelope."
        ),
    ),
)

# --------------------------------------------------------------------------- #
# Mellin:  M[sigma](s) = int_0^inf sigma(z) z^(s-1) dz
# --------------------------------------------------------------------------- #
MELLIN_IDENTITIES: tuple[TransformIdentity, ...] = (
    TransformIdentity(
        activation="gaussian",
        transform="mellin",
        expression="2^(s/2 - 1) Gamma(s/2)",
        region="s > 0",
        evaluated_with="lgamma",
        min_argument=0.0,
        note=(
            "Substituting u = z^2/2 turns the Gaussian Mellin integral into a Gamma "
            "integral. Evaluated as exp((s/2 - 1) log 2 + lgamma(s/2)) because lgamma "
            "is the only gamma function both torch and jax expose; the result itself "
            "still overflows once s exceeds roughly 300, as Gamma must."
        ),
    ),
)

#: Closed-form Mellin transform of the *complementary* sigmoid,
#: ``1 - sigmoid(z) = sigmoid(-z) = 1/(1 + exp(z))`` -- the Fermi-Dirac
#: distribution at unit temperature. Shipped under its own name in each
#: backend (``fermi_dirac_mellin``) rather than attached to the ``sigmoid``
#: spec, because it is the transform of a different function; see the module
#: docstring for the ``Re(s) > 1`` scope wall.
FERMI_DIRAC_MELLIN = TransformIdentity(
    activation="sigmoid_complement",
    transform="mellin",
    expression="Gamma(s) eta(s) = Gamma(s) (1 - 2^(1-s)) zeta(s)",
    region="s > 1",
    evaluated_with="lgamma, zeta",
        min_argument=1.0,
    note=(
        "The Fermi-Dirac integral. The defining integral converges for s > 0, but "
        "omnibias routes eta through the Riemann zeta function, and the project's "
        "rigorous zeta scope (omnibias.core.verified.dirichlet) is Re(s) > 1 only: "
        "zeta_enclosure raises below that line and analytic continuation is a "
        "recorded external obligation. The kernel honours the same wall."
    ),
)

# --------------------------------------------------------------------------- #
# Deliberate gaps.
# --------------------------------------------------------------------------- #
EXCLUDED_TRANSFORMS: tuple[ExcludedTransform, ...] = (
    ExcludedTransform(
        activation="relu",
        transform="fourier",
        reason="distributional",
        detail=(
            "relu grows linearly, so it is not in L^1(R). Its Fourier transform "
            "exists only as a tempered distribution, combining a derivative of a "
            "Dirac delta with a Hadamard finite part of 1/xi^2. No tensor-valued "
            "kernel can return that."
        ),
    ),
    ExcludedTransform(
        activation="sigmoid",
        transform="fourier",
        reason="distributional",
        detail=(
            "sigmoid saturates to 1 at +inf and 0 at -inf, so it is not in L^1(R). "
            "Its Fourier transform is pi delta(xi) plus a principal-value term "
            "-i pi / sinh(pi xi / 2) -- distributional, and complex off xi = 0."
        ),
    ),
    ExcludedTransform(
        activation="tanh",
        transform="fourier",
        reason="distributional",
        detail=(
            "tanh saturates to +-1 and is not in L^1(R). Its Fourier transform is "
            "the principal value -i pi / sinh(pi xi / 2), a distribution with a "
            "non-integrable singularity at the origin."
        ),
    ),
    ExcludedTransform(
        activation="exp",
        transform="fourier",
        reason="divergent",
        detail="exp(z) grows without bound; the Fourier integral diverges at +inf.",
    ),
    ExcludedTransform(
        activation="exp",
        transform="mellin",
        reason="divergent",
        detail=(
            "omnibias's exp activation is exp(+z), and int_0^inf exp(z) z^(s-1) dz "
            "diverges for every s. The textbook Gamma(s) pair is the Mellin "
            "transform of the decaying exponential exp(-z), which is a different "
            "function and is not in the activation dictionary."
        ),
    ),
    ExcludedTransform(
        activation="sigmoid",
        transform="mellin",
        reason="complementary",
        detail=(
            "sigmoid tends to 1 at +inf, so int_0^inf sigmoid(z) z^(s-1) dz diverges "
            "for every s. The convergent Fermi-Dirac identity Gamma(s) eta(s) belongs "
            "to the complement 1 - sigmoid(z), shipped as fermi_dirac_mellin."
        ),
    ),
    ExcludedTransform(
        activation="tanh",
        transform="mellin",
        reason="complementary",
        detail=(
            "tanh tends to 1 at +inf, so its Mellin integral diverges. The convergent "
            "companion is the complement 1 - tanh(z) = 2 / (exp(2z) + 1), whose Mellin "
            "transform is 2^(1-s) Gamma(s) eta(s) -- the same Fermi-Dirac object at "
            "half temperature, reachable through fermi_dirac_mellin."
        ),
    ),
    ExcludedTransform(
        activation="relu",
        transform="mellin",
        reason="divergent",
        detail=(
            "relu(z) = z on the half line, so int_0^inf z^s dz diverges at the upper "
            "limit for every s. Mellin transforms need decay at infinity."
        ),
    ),
    ExcludedTransform(
        activation="sin",
        transform="fourier",
        reason="distributional",
        detail=(
            "sin is bounded but not integrable over R. Its Fourier transform is the "
            "pair of Dirac deltas i pi (delta(xi + 1) - delta(xi - 1)), which is a "
            "measure, not a function of xi."
        ),
    ),
    ExcludedTransform(
        activation="cos",
        transform="fourier",
        reason="distributional",
        detail=(
            "As for sin: the transform is pi (delta(xi - 1) + delta(xi + 1)), a pair "
            "of Dirac masses at the carrier frequency."
        ),
    ),
    ExcludedTransform(
        activation="sin",
        transform="mellin",
        reason="conditional",
        detail=(
            "int_0^inf sin(z) z^(s-1) dz equals Gamma(s) sin(pi s / 2) on the strip "
            "0 < Re(s) < 1, but only as a conditionally convergent (Abel-regularised) "
            "integral -- sin does not decay, so the integrand is not Lebesgue "
            "integrable anywhere. omnibias registers absolutely convergent transforms "
            "only, because a regularised value cannot be validated against quadrature."
        ),
    ),
    ExcludedTransform(
        activation="cos",
        transform="mellin",
        reason="conditional",
        detail=(
            "As for sin: the value Gamma(s) cos(pi s / 2) on 0 < Re(s) < 1 exists only "
            "as a conditionally convergent integral, since cos does not decay."
        ),
    ),
    ExcludedTransform(
        activation="sinh",
        transform="mellin",
        reason="divergent",
        detail="sinh grows like exp(z)/2; the Mellin integral diverges at infinity for every s.",
    ),
    ExcludedTransform(
        activation="cosh",
        transform="mellin",
        reason="divergent",
        detail="cosh grows like exp(z)/2; the Mellin integral diverges at infinity for every s.",
    ),
    ExcludedTransform(
        activation="sinh",
        transform="fourier",
        reason="divergent",
        detail="sinh grows without bound in both directions; the Fourier integral diverges.",
    ),
    ExcludedTransform(
        activation="cosh",
        transform="fourier",
        reason="divergent",
        detail="cosh grows without bound in both directions; the Fourier integral diverges.",
    ),
    ExcludedTransform(
        activation="sech",
        transform="mellin",
        reason="unavailable",
        detail=(
            "The identity exists -- M[sech](s) = 2 Gamma(s) beta(s) with beta the "
            "Dirichlet beta function -- but neither torch nor jax ships Dirichlet "
            "beta, and omnibias will not substitute a truncated series for a kernel "
            "advertised as closed form. Unregistered until a backend provides it."
        ),
    ),
)


def identities(transform: TransformName) -> tuple[TransformIdentity, ...]:
    """Return the registered identities for one transform.

    Raises
    ------
    ValueError
        If ``transform`` is not one of :data:`TRANSFORM_NAMES`.
    """
    table = {
        "laplace": LAPLACE_IDENTITIES,
        "fourier": FOURIER_IDENTITIES,
        "mellin": MELLIN_IDENTITIES,
    }
    if transform not in table:
        raise ValueError(f"transform must be one of {TRANSFORM_NAMES}, got {transform!r}")
    return table[transform]


def registered_activations(transform: TransformName) -> tuple[str, ...]:
    """Activation names carrying a closed-form ``transform``, in table order."""
    return tuple(identity.activation for identity in identities(transform))


def find_identity(activation: str, transform: TransformName) -> TransformIdentity | None:
    """Return the identity for ``(activation, transform)``, or ``None`` if unregistered."""
    for identity in identities(transform):
        if identity.activation == activation:
            return identity
    return None


def find_exclusion(activation: str, transform: TransformName) -> ExcludedTransform | None:
    """Return the recorded reason ``(activation, transform)`` is unregistered, if any."""
    for excluded in EXCLUDED_TRANSFORMS:
        if excluded.activation == activation and excluded.transform == transform:
            return excluded
    return None


__all__ = [
    "EXCLUDED_TRANSFORMS",
    "ExcludedTransform",
    "ExclusionReason",
    "FERMI_DIRAC_MELLIN",
    "FOURIER_IDENTITIES",
    "LAPLACE_IDENTITIES",
    "MELLIN_IDENTITIES",
    "SPECTRAL_VARIABLE",
    "TEMPERING_LAWS",
    "TRANSFORM_DEFINITION",
    "TRANSFORM_NAMES",
    "TransformIdentity",
    "TransformName",
    "find_exclusion",
    "find_identity",
    "identities",
    "registered_activations",
]

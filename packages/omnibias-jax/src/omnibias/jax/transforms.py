# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form integral transforms of the activation dictionary (JAX).

Bit-identical twin of :mod:`omnibias.torch.transforms`. omnibias already
computes every derivative of an activation in closed form; this module is the
other side of that coin, evaluating the Laplace, Fourier and Mellin transforms
of the activations whose transform is itself elementary -- in one shot, with no
quadrature, series truncation or iteration.

Which pairs are shipped, what each equals, where it converges, and *why the
gaps are gaps* is decided once in the pure-Python table
:mod:`omnibias.core.transforms`; this module is the thin tensor evaluation of
that table. The coverage and parity tests walk the table against both
registries, so the two backends and the documentation cannot drift.

Conventions (from :class:`omnibias.core.spec.TransformKernels`):

.. math::

    \mathcal{L}[\sigma](s) = \int_0^\infty \sigma(z) e^{-sz} dz, \qquad
    \mathcal{F}[\sigma](\xi) = \int_{\mathbb{R}} \sigma(z) e^{-i\xi z} dz, \qquad
    \mathcal{M}[\sigma](s) = \int_0^\infty \sigma(z) z^{s-1} dz.

Usage
-----

>>> import jax.numpy as jnp
>>> from omnibias.jax.transforms import laplace_transform
>>> laplace_transform("relu", jnp.array([1.0, 2.0, 4.0]))  # 1 / s^2
Array([1.    , 0.25  , 0.0625], dtype=float32)

Outside the region of convergence the kernels return what the closed form
returns -- ``inf`` at a pole, ``nan`` past a branch -- rather than silently
clamping. Use :func:`region_of_convergence` to check, or the layer wrappers,
which keep a learnable spectral variable inside the region by construction.

Honesty labels
--------------
Every kernel here is **closed form**. The numerical oracle used to validate
them (``lebesgue_integral`` from :mod:`omnibias.measure`) appears in the tests
only. Activations without an entry genuinely have no elementary transform under
omnibias's registration -- see ``EXCLUDED_TRANSFORMS`` in
:mod:`omnibias.core.transforms` for the reason attached to each gap.

Tracing note: every kernel is a pure array expression and is safe under
``jit`` / ``grad`` / ``vmap``. The one exception is
:func:`fermi_dirac_mellin`, whose ``s > 1`` scope check reads a concrete value;
it skips the check under tracing (documented on the function) rather than
raising a ``ConcretizationTypeError``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from omnibias.core.spec import ActivationSpec, TransformKernels
from omnibias.core.transforms import (
    TRANSFORM_NAMES,
    TransformName,
    find_exclusion,
    find_identity,
)

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softplus
from jax.scipy.special import digamma, erfcx, gammaln, zeta

_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Laplace kernels.  L[sigma](s) = int_0^inf sigma(z) exp(-s z) dz
# --------------------------------------------------------------------------- #
def _exp_laplace(s: Array) -> Array:
    """``1 / (s - 1)`` for ``s > 1``."""
    return 1.0 / (s - 1.0)


def _relu_laplace(s: Array) -> Array:
    """``1 / s^2`` for ``s > 0``."""
    return 1.0 / (s * s)


def _sin_laplace(s: Array) -> Array:
    """``1 / (s^2 + 1)`` for ``s > 0``."""
    return 1.0 / (s * s + 1.0)


def _cos_laplace(s: Array) -> Array:
    """``s / (s^2 + 1)`` for ``s > 0``."""
    return s / (s * s + 1.0)


def _sinh_laplace(s: Array) -> Array:
    """``1 / (s^2 - 1)`` for ``s > 1``."""
    return 1.0 / (s * s - 1.0)


def _cosh_laplace(s: Array) -> Array:
    """``s / (s^2 - 1)`` for ``s > 1``."""
    return s / (s * s - 1.0)


def _gaussian_laplace(s: Array) -> Array:
    r"""``sqrt(pi/2) exp(s^2/2) erfc(s/sqrt 2)``, entire in ``s``.

    Evaluated through the *scaled* complementary error function
    ``erfcx(x) = exp(x^2) erfc(x)``: the two naive factors overflow and
    underflow past ``s ~ 38`` while their product decays like ``1/s``, so the
    unscaled form returns ``inf * 0 = nan`` exactly where the answer is small
    and well-conditioned.
    """
    return _SQRT_HALF_PI * erfcx(s * _INV_SQRT_TWO)


def _sigmoid_laplace(s: Array) -> Array:
    r"""``Phi(-1, 1, s) = (psi((s+1)/2) - psi(s/2)) / 2`` for ``s > 0``."""
    return 0.5 * (digamma(0.5 * (s + 1.0)) - digamma(0.5 * s))


def _tanh_laplace(s: Array) -> Array:
    r"""``Phi(-1, 1, s/2) - 1/s = (psi(s/4 + 1/2) - psi(s/4)) / 2 - 1/s`` for ``s > 0``."""
    quarter = 0.25 * s
    return 0.5 * (digamma(quarter + 0.5) - digamma(quarter)) - 1.0 / s


def _sech_laplace(s: Array) -> Array:
    r"""``(psi((s+3)/4) - psi((s+1)/4)) / 2`` for ``s > -1``."""
    return 0.5 * (digamma(0.25 * (s + 3.0)) - digamma(0.25 * (s + 1.0)))


# --------------------------------------------------------------------------- #
# Fourier kernels.  F[sigma](xi) = int_R sigma(z) exp(-i xi z) dz
# --------------------------------------------------------------------------- #
def _gaussian_fourier(xi: Array) -> Array:
    """``sqrt(2 pi) exp(-xi^2 / 2)``: the Gaussian is its own transform."""
    return _SQRT_TWO_PI * jnp.exp(-0.5 * xi * xi)


def _sech_fourier(xi: Array) -> Array:
    """``pi sech(pi xi / 2)``: sech transforms to a dilated sech."""
    return math.pi / jnp.cosh(0.5 * math.pi * xi)


# --------------------------------------------------------------------------- #
# Mellin kernels.  M[sigma](s) = int_0^inf sigma(z) z^(s-1) dz
# --------------------------------------------------------------------------- #
def _gaussian_mellin(s: Array) -> Array:
    """``2^(s/2 - 1) Gamma(s/2)`` for ``s > 0``.

    Assembled in log space and exponentiated once, because ``gammaln`` is the
    only gamma function both backends expose. The result still overflows past
    ``s ~ 300``, as any Gamma-valued quantity must.
    """
    half = 0.5 * s
    return jnp.exp((half - 1.0) * math.log(2.0) + gammaln(half))


def fermi_dirac_mellin(s: Array) -> Array:
    r"""Mellin transform of the *complementary* sigmoid, for ``s > 1``.

    .. math::

        \int_0^\infty \frac{z^{s-1}}{1 + e^{z}}\, dz
            = \Gamma(s)\, \eta(s)
            = \Gamma(s)\, (1 - 2^{1-s})\, \zeta(s).

    This is the Fermi-Dirac integral at unit temperature. It is *not*
    ``mellin_transform("sigmoid", s)``: the integrand here is
    ``1 - sigmoid(z) = sigmoid(-z)``, and ``sigmoid``'s own Mellin integral
    diverges because ``sigmoid`` tends to 1 rather than decaying. It is shipped
    under its own name so the label matches the mathematics.

    Restricted to ``s > 1`` even though the integral converges for ``s > 0``:
    the evaluation routes ``eta`` through the Riemann zeta function, and
    omnibias's rigorous zeta scope
    (:mod:`omnibias.core.verified.dirichlet`) is ``Re(s) > 1`` only --
    ``zeta_enclosure`` raises below that line, and analytic continuation into
    the critical strip is a recorded external obligation the project never
    infers.

    The scope check reads a concrete value, so it is skipped when ``s`` is a
    tracer (under ``jit``, ``grad`` or ``vmap``), matching how
    ``omnibias.measure.jax.integraleq`` handles its solvability check. Validate
    in eager mode; the arithmetic itself is trace-safe either way.

    Unlike the torch twin, this kernel *is* differentiable in ``s``:
    ``jax.scipy.special.zeta`` defines a gradient rule where
    ``torch.special.zeta`` does not. The asymmetry lives in the special-function
    libraries, not in omnibias.

    The complementary ``tanh`` is reachable from here too:
    ``M[1 - tanh](s) = 2^(1-s) Gamma(s) eta(s)``.
    """
    if not isinstance(s, jax.core.Tracer) and bool(jnp.any(jnp.asarray(s) <= 1.0)):
        raise ValueError(
            "fermi_dirac_mellin requires s > 1: eta(s) is evaluated through the "
            "Riemann zeta function, whose sound scope in omnibias "
            "(omnibias.core.verified.dirichlet) is Re(s) > 1. Continuation below "
            "that line is a recorded external obligation, not something this "
            "kernel may assume."
        )
    eta = -jnp.expm1((1.0 - s) * math.log(2.0)) * zeta(s, 1.0)
    return jnp.exp(gammaln(s)) * eta


# --------------------------------------------------------------------------- #
# Per-activation kernel bundles, imported by the activation registry so the
# specs carry their transforms at construction time (no post-hoc mutation of a
# frozen dataclass, and no import cycle: this module reaches only into
# ``core`` and ``jax`` at module scope).
# --------------------------------------------------------------------------- #
EXP_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_exp_laplace)
RELU_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_relu_laplace)
SIN_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_sin_laplace)
COS_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_cos_laplace)
SINH_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_sinh_laplace)
COSH_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_cosh_laplace)
SIGMOID_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_sigmoid_laplace)
TANH_TRANSFORMS: TransformKernels[Array] = TransformKernels(laplace=_tanh_laplace)
SECH_TRANSFORMS: TransformKernels[Array] = TransformKernels(
    laplace=_sech_laplace, fourier=_sech_fourier
)
GAUSSIAN_TRANSFORMS: TransformKernels[Array] = TransformKernels(
    laplace=_gaussian_laplace, fourier=_gaussian_fourier, mellin=_gaussian_mellin
)


# --------------------------------------------------------------------------- #
# Resolvers.
# --------------------------------------------------------------------------- #
def _resolve(
    activation: str | ActivationSpec[Array], transform: TransformName
) -> tuple[ActivationSpec[Array], Callable[[Array], Array]]:
    from omnibias.jax.activations import get_activation

    spec = get_activation(activation)
    kernels = spec.transforms
    kernel = None if kernels is None else getattr(kernels, transform)
    if kernel is None:
        excluded = find_exclusion(spec.name, transform)
        reason = (
            f" ({excluded.reason}: {excluded.detail})"
            if excluded is not None
            else " No closed form is registered for this pair."
        )
        raise TypeError(
            f"{transform}_transform requires a base activation with a closed-form "
            f"{transform} kernel; activation {spec.name!r} has none.{reason}"
        )
    return spec, kernel


def laplace_transform(activation: str | ActivationSpec[Array], s: Array) -> Array:
    r"""Evaluate ``L[sigma](s) = int_0^inf sigma(z) exp(-s z) dz`` in closed form.

    Parameters
    ----------
    activation:
        Registry name (``"sigmoid"``) or an :class:`ActivationSpec`.
    s:
        Real spectral variable, any shape. Values outside the region of
        convergence (see :func:`region_of_convergence`) return the analytic
        continuation of the closed form, including ``inf`` at a pole.

    Raises
    ------
    TypeError
        If the activation has no registered Laplace kernel. The message carries
        the recorded reason -- divergent, distributional, complementary or
        unavailable -- rather than just reporting absence.
    """
    _, kernel = _resolve(activation, "laplace")
    return kernel(s)


def fourier_transform(activation: str | ActivationSpec[Array], xi: Array) -> Array:
    r"""Evaluate ``F[sigma](xi) = int_R sigma(z) exp(-i xi z) dz`` in closed form.

    Only ``L^1(R)`` activations are registered, and for those the transform is
    real and even, so this returns a real array. Saturating or growing
    activations have *distributional* transforms (Dirac masses, principal
    values) and raise instead of returning a silently wrong kernel.
    """
    _, kernel = _resolve(activation, "fourier")
    return kernel(xi)


def mellin_transform(activation: str | ActivationSpec[Array], s: Array) -> Array:
    r"""Evaluate ``M[sigma](s) = int_0^inf sigma(z) z^(s-1) dz`` in closed form.

    Mellin transforms need decay at both ends of the half line, which most of
    the activation dictionary does not have; ``gaussian`` is the registered
    case. For the saturating activations the convergent classical identity
    belongs to the complement -- see :func:`fermi_dirac_mellin`.
    """
    _, kernel = _resolve(activation, "mellin")
    return kernel(s)


def has_transform(activation: str | ActivationSpec[Array], transform: TransformName) -> bool:
    """True if ``activation`` carries a closed-form ``transform`` kernel."""
    from omnibias.jax.activations import get_activation

    if transform not in TRANSFORM_NAMES:
        raise ValueError(f"transform must be one of {TRANSFORM_NAMES}, got {transform!r}")
    kernels = get_activation(activation).transforms
    return kernels is not None and getattr(kernels, transform) is not None


def region_of_convergence(
    activation: str | ActivationSpec[Array], transform: TransformName
) -> tuple[float | None, str]:
    """Return ``(min_argument, region)`` for a registered transform.

    ``min_argument`` is the infimum of the real region of convergence, or
    ``None`` when the transform is entire; ``region`` is the human-readable
    condition. Raises :class:`TypeError` for an unregistered pair, with the
    recorded reason.
    """
    spec, _ = _resolve(activation, transform)
    identity = find_identity(spec.name, transform)
    if identity is None:  # pragma: no cover - kept in sync by the coverage test
        raise TypeError(
            f"activation {spec.name!r} has a {transform} kernel but no entry in "
            f"omnibias.core.transforms; the identity table is out of sync."
        )
    return identity.min_argument, identity.region


# --------------------------------------------------------------------------- #
# Trainable layers (functional: explicit parameter pytrees, JAX style).
# --------------------------------------------------------------------------- #
class TransformBlock:
    r"""Learnable spectral feature map built from a closed-form transform.

    Functional twin of :class:`omnibias.torch.transforms.TransformBlock`. The
    torch version owns its parameters; this one is a pure description that
    hands you an initial parameter pytree via :meth:`init` and evaluates
    ``T[sigma](scale * x + shift)`` through :meth:`apply`, so it composes with
    ``jit`` / ``grad`` and any JAX optimizer without a module system.

    Because the transform is closed form, the whole layer is a single
    elementary expression -- there is no quadrature to differentiate through,
    and the gradients in ``scale`` and ``shift`` are exact to machine precision
    rather than to a quadrature tolerance.

    A learnable spectral variable can wander out of the region of convergence,
    where the closed form is meaningless (a pole, or the wrong branch). The
    block therefore reparameterises: it stores a raw offset and maps it into
    the open region with a softplus,

    ``argument = min_argument + softplus(raw) + scale * x``

    so no optimizer step can drive the offset below the boundary, however large
    the step. The guarantee is one-sided and closed: ``softplus`` underflows to
    exactly zero for extreme raw values, so a sufficiently hostile update can
    land the offset *on* the boundary, but never past it into the divergent
    half-plane. Transforms that are entire (the Gaussian Fourier and Laplace
    kernels) have no boundary and use a plain additive shift.
    """

    def __init__(
        self,
        activation: str | ActivationSpec[Array],
        transform: TransformName,
        *,
        features: int = 1,
        init_shift: float | None = None,
        init_scale: float = 1.0,
    ) -> None:
        if features < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        spec, kernel = _resolve(activation, transform)
        self.activation_name = spec.name
        self.transform = transform
        self.features = features
        self._kernel = kernel
        lower, region = region_of_convergence(spec, transform)
        self.min_argument = lower
        self.region = region

        if init_shift is None:
            init_shift = 0.0 if lower is None else lower + 1.0
        elif lower is not None and init_shift <= lower:
            raise ValueError(
                f"init_shift={init_shift} is outside the region of convergence "
                f"({region}) of the {transform} transform of {spec.name!r}."
            )
        self.init_shift = float(init_shift)
        self.init_scale = float(init_scale)

    def init(self) -> dict[str, Array]:
        """Initial parameter pytree ``{"raw_shift": ..., "scale": ...}``."""
        if self.min_argument is None:
            raw = jnp.full((self.features,), self.init_shift)
        else:
            gap = self.init_shift - self.min_argument
            raw = jnp.full((self.features,), math.log(math.expm1(gap)))
        return {"raw_shift": raw, "scale": jnp.full((self.features,), self.init_scale)}

    def argument(self, params: dict[str, Array], x: Array) -> Array:
        """Spectral argument ``scale * x + shift`` this block evaluates at.

        Exposed because the reparameterisation makes the effective shift a
        function of the raw parameter; inspecting it is how a caller sees where
        in the region of convergence training has moved.
        """
        raw = params["raw_shift"]
        shift = raw if self.min_argument is None else self.min_argument + softplus(raw)
        return params["scale"] * x + shift

    def apply(self, params: dict[str, Array], x: Array) -> Array:
        """Evaluate the transform at ``scale * x + shift``, broadcasting over features."""
        return self._kernel(self.argument(params, jnp.atleast_1d(x)))

    def __repr__(self) -> str:
        return (
            f"TransformBlock(activation={self.activation_name!r}, "
            f"transform={self.transform!r}, features={self.features}, region={self.region!r})"
        )


class LaplaceTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Laplace transform."""

    def __init__(self, activation: str | ActivationSpec[Array], **kwargs: object) -> None:
        super().__init__(activation, "laplace", **kwargs)  # type: ignore[arg-type]


class FourierTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Fourier transform."""

    def __init__(self, activation: str | ActivationSpec[Array], **kwargs: object) -> None:
        super().__init__(activation, "fourier", **kwargs)  # type: ignore[arg-type]


class MellinTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Mellin transform."""

    def __init__(self, activation: str | ActivationSpec[Array], **kwargs: object) -> None:
        super().__init__(activation, "mellin", **kwargs)  # type: ignore[arg-type]


__all__ = [
    "COSH_TRANSFORMS",
    "COS_TRANSFORMS",
    "EXP_TRANSFORMS",
    "FourierTransform",
    "GAUSSIAN_TRANSFORMS",
    "LaplaceTransform",
    "MellinTransform",
    "RELU_TRANSFORMS",
    "SECH_TRANSFORMS",
    "SIGMOID_TRANSFORMS",
    "SINH_TRANSFORMS",
    "SIN_TRANSFORMS",
    "TANH_TRANSFORMS",
    "TransformBlock",
    "fermi_dirac_mellin",
    "fourier_transform",
    "has_transform",
    "laplace_transform",
    "mellin_transform",
    "region_of_convergence",
]

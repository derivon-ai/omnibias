# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form integral transforms of the activation dictionary (PyTorch).

omnibias already computes every derivative of an activation in closed form.
This module is the other side of that coin: for the activations whose Laplace,
Fourier or Mellin transform is itself elementary, it evaluates the transform in
one shot -- no quadrature, no series truncation, no iteration -- and makes it a
differentiable tensor op that drops into a network.

Which pairs are shipped, what each equals, where it converges, and *why the
gaps are gaps* is decided once in the pure-Python table
:mod:`omnibias.core.transforms`; this module is the thin tensor evaluation of
that table, and :mod:`omnibias.jax.transforms` is its bit-identical twin. The
coverage test walks the table against the registry, so the two cannot drift.

Conventions (from :class:`omnibias.core.spec.TransformKernels`):

.. math::

    \mathcal{L}[\sigma](s) = \int_0^\infty \sigma(z) e^{-sz} dz, \qquad
    \mathcal{F}[\sigma](\xi) = \int_{\mathbb{R}} \sigma(z) e^{-i\xi z} dz, \qquad
    \mathcal{M}[\sigma](s) = \int_0^\infty \sigma(z) z^{s-1} dz.

Usage
-----

>>> import torch
>>> from omnibias.torch.transforms import laplace_transform
>>> torch.set_default_dtype(torch.float64)
>>> s = torch.tensor([1.0, 2.0, 4.0])
>>> laplace_transform("relu", s)          # 1 / s^2
tensor([1.0000, 0.2500, 0.0625], dtype=torch.float64)

Outside the region of convergence the kernels return what the closed form
returns -- ``inf`` at a pole, ``nan`` past a branch -- rather than silently
clamping. Use :func:`region_of_convergence` to check, or the layer wrappers,
which keep a learnable spectral variable inside the region by construction.

Honesty labels
--------------
Every kernel here is **closed form**. Nothing in this module is a quadrature
approximation; the numerical oracle used to validate them (``lebesgue_integral``
from :mod:`omnibias.measure`) appears in the tests only. Activations without an
entry genuinely have no elementary transform under omnibias's registration --
see ``EXCLUDED_TRANSFORMS`` in :mod:`omnibias.core.transforms` for the reason
attached to each gap. Keras is not covered: it reads
:class:`~omnibias.core.spec.ActivationSpec` from core, so the ``transforms``
field is visible there with its default ``None``, but no keras kernels ship.
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

import torch
from torch import Tensor, nn

_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Laplace kernels.  L[sigma](s) = int_0^inf sigma(z) exp(-s z) dz
# --------------------------------------------------------------------------- #
def _exp_laplace(s: Tensor) -> Tensor:
    """``1 / (s - 1)`` for ``s > 1``."""
    return 1.0 / (s - 1.0)


def _relu_laplace(s: Tensor) -> Tensor:
    """``1 / s^2`` for ``s > 0``."""
    return 1.0 / (s * s)


def _sin_laplace(s: Tensor) -> Tensor:
    """``1 / (s^2 + 1)`` for ``s > 0``."""
    return 1.0 / (s * s + 1.0)


def _cos_laplace(s: Tensor) -> Tensor:
    """``s / (s^2 + 1)`` for ``s > 0``."""
    return s / (s * s + 1.0)


def _sinh_laplace(s: Tensor) -> Tensor:
    """``1 / (s^2 - 1)`` for ``s > 1``."""
    return 1.0 / (s * s - 1.0)


def _cosh_laplace(s: Tensor) -> Tensor:
    """``s / (s^2 - 1)`` for ``s > 1``."""
    return s / (s * s - 1.0)


def _gaussian_laplace(s: Tensor) -> Tensor:
    r"""``sqrt(pi/2) exp(s^2/2) erfc(s/sqrt 2)``, entire in ``s``.

    Evaluated through the *scaled* complementary error function
    ``erfcx(x) = exp(x^2) erfc(x)``: the two naive factors overflow and
    underflow past ``s ~ 38`` while their product decays like ``1/s``, so the
    unscaled form returns ``inf * 0 = nan`` exactly where the answer is small
    and well-conditioned.
    """
    scaled: Tensor = torch.special.erfcx(s * _INV_SQRT_TWO)
    return _SQRT_HALF_PI * scaled


def _sigmoid_laplace(s: Tensor) -> Tensor:
    r"""``Phi(-1, 1, s) = (psi((s+1)/2) - psi(s/2)) / 2`` for ``s > 0``."""
    return 0.5 * (torch.digamma(0.5 * (s + 1.0)) - torch.digamma(0.5 * s))


def _tanh_laplace(s: Tensor) -> Tensor:
    r"""``Phi(-1, 1, s/2) - 1/s = (psi(s/4 + 1/2) - psi(s/4)) / 2 - 1/s`` for ``s > 0``."""
    quarter = 0.25 * s
    return 0.5 * (torch.digamma(quarter + 0.5) - torch.digamma(quarter)) - 1.0 / s


def _sech_laplace(s: Tensor) -> Tensor:
    r"""``(psi((s+3)/4) - psi((s+1)/4)) / 2`` for ``s > -1``."""
    return 0.5 * (torch.digamma(0.25 * (s + 3.0)) - torch.digamma(0.25 * (s + 1.0)))


# --------------------------------------------------------------------------- #
# Fourier kernels.  F[sigma](xi) = int_R sigma(z) exp(-i xi z) dz
# --------------------------------------------------------------------------- #
def _gaussian_fourier(xi: Tensor) -> Tensor:
    """``sqrt(2 pi) exp(-xi^2 / 2)``: the Gaussian is its own transform."""
    return _SQRT_TWO_PI * torch.exp(-0.5 * xi * xi)


def _sech_fourier(xi: Tensor) -> Tensor:
    """``pi sech(pi xi / 2)``: sech transforms to a dilated sech."""
    return math.pi / torch.cosh(0.5 * math.pi * xi)


# --------------------------------------------------------------------------- #
# Mellin kernels.  M[sigma](s) = int_0^inf sigma(z) z^(s-1) dz
# --------------------------------------------------------------------------- #
def _gaussian_mellin(s: Tensor) -> Tensor:
    """``2^(s/2 - 1) Gamma(s/2)`` for ``s > 0``.

    Assembled in log space and exponentiated once, because ``lgamma`` is the
    only gamma function both backends expose. The result still overflows past
    ``s ~ 300``, as any Gamma-valued quantity must.
    """
    half = 0.5 * s
    return torch.exp((half - 1.0) * math.log(2.0) + torch.lgamma(half))


def fermi_dirac_mellin(s: Tensor) -> Tensor:
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
    infers. Arguments at or below 1 raise rather than silently crossing the
    same wall the verified layer refuses to cross.

    **Not differentiable in** ``s`` **on this backend.** ``torch.special.zeta``
    ships no derivative rule, so any attempt to backpropagate through this
    kernel raises ``NotImplementedError`` from torch. The JAX twin
    (:func:`omnibias.jax.transforms.fermi_dirac_mellin`) *is* differentiable,
    since ``jax.scipy.special.zeta`` defines its gradient. This is a genuine
    backend asymmetry in the underlying special-function libraries, recorded
    rather than hidden; every other kernel in this module is differentiable on
    both backends.

    The complementary ``tanh`` is reachable from here too:
    ``M[1 - tanh](s) = 2^(1-s) Gamma(s) eta(s)``.
    """
    if bool(torch.any(s <= 1.0)):
        raise ValueError(
            "fermi_dirac_mellin requires s > 1: eta(s) is evaluated through the "
            "Riemann zeta function, whose sound scope in omnibias "
            "(omnibias.core.verified.dirichlet) is Re(s) > 1. Continuation below "
            "that line is a recorded external obligation, not something this "
            "kernel may assume."
        )
    eta: Tensor = -torch.expm1((1.0 - s) * math.log(2.0)) * torch.special.zeta(
        s, torch.ones_like(s)
    )
    gamma: Tensor = torch.exp(torch.lgamma(s))
    return gamma * eta


# --------------------------------------------------------------------------- #
# Per-activation kernel bundles, imported by the activation modules so the
# specs carry their transforms at construction time (no post-hoc mutation of a
# frozen dataclass, and no import cycle: this module reaches only into
# ``core`` and ``torch``).
# --------------------------------------------------------------------------- #
EXP_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_exp_laplace)
RELU_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_relu_laplace)
SIN_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_sin_laplace)
COS_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_cos_laplace)
SINH_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_sinh_laplace)
COSH_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_cosh_laplace)
SIGMOID_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_sigmoid_laplace)
TANH_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(laplace=_tanh_laplace)
SECH_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(
    laplace=_sech_laplace, fourier=_sech_fourier
)
GAUSSIAN_TRANSFORMS: TransformKernels[Tensor] = TransformKernels(
    laplace=_gaussian_laplace, fourier=_gaussian_fourier, mellin=_gaussian_mellin
)


# --------------------------------------------------------------------------- #
# Resolvers.
# --------------------------------------------------------------------------- #
def _resolve(
    activation: str | ActivationSpec[Tensor], transform: TransformName
) -> tuple[ActivationSpec[Tensor], Callable[[Tensor], Tensor]]:
    from omnibias.torch.activations import get_activation

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


def laplace_transform(activation: str | ActivationSpec[Tensor], s: Tensor) -> Tensor:
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


def fourier_transform(activation: str | ActivationSpec[Tensor], xi: Tensor) -> Tensor:
    r"""Evaluate ``F[sigma](xi) = int_R sigma(z) exp(-i xi z) dz`` in closed form.

    Only ``L^1(R)`` activations are registered, and for those the transform is
    real and even, so this returns a real tensor. Saturating or growing
    activations have *distributional* transforms (Dirac masses, principal
    values) and raise instead of returning a silently wrong kernel.
    """
    _, kernel = _resolve(activation, "fourier")
    return kernel(xi)


def mellin_transform(activation: str | ActivationSpec[Tensor], s: Tensor) -> Tensor:
    r"""Evaluate ``M[sigma](s) = int_0^inf sigma(z) z^(s-1) dz`` in closed form.

    Mellin transforms need decay at both ends of the half line, which most of
    the activation dictionary does not have; ``gaussian`` is the registered
    case. For the saturating activations the convergent classical identity
    belongs to the complement -- see :func:`fermi_dirac_mellin`.
    """
    _, kernel = _resolve(activation, "mellin")
    return kernel(s)


def has_transform(activation: str | ActivationSpec[Tensor], transform: TransformName) -> bool:
    """True if ``activation`` carries a closed-form ``transform`` kernel."""
    from omnibias.torch.activations import get_activation

    if transform not in TRANSFORM_NAMES:
        raise ValueError(f"transform must be one of {TRANSFORM_NAMES}, got {transform!r}")
    kernels = get_activation(activation).transforms
    return kernels is not None and getattr(kernels, transform) is not None


def region_of_convergence(
    activation: str | ActivationSpec[Tensor], transform: TransformName
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
# Trainable layers.
# --------------------------------------------------------------------------- #
class TransformBlock(nn.Module):
    r"""Learnable spectral feature map built from a closed-form transform.

    Evaluates ``T[sigma](scale * x + shift)`` per output feature, with ``scale``
    and ``shift`` trainable. Because the transform is closed form, the whole
    layer is a single elementary expression -- there is no quadrature to
    backpropagate through, and the gradients in ``scale`` and ``shift`` are
    exact to machine precision rather than to a quadrature tolerance.

    A learnable spectral variable can wander out of the region of convergence,
    where the closed form is meaningless (a pole, or the wrong branch). The
    block therefore reparameterises: it stores a raw offset and maps it into the
    open region with a softplus,

    ``argument = min_argument + softplus(raw) + scale * x``

    so no optimizer step can drive the offset below the boundary, however large
    the step. The guarantee is one-sided and closed: ``softplus`` underflows to
    exactly zero for raw values below about ``-745`` in float64, so a
    sufficiently hostile update can land the offset *on* the boundary, but
    never past it into the divergent half-plane. Transforms that are entire
    (the Gaussian Fourier and Laplace kernels) have no boundary and use a plain
    additive shift.

    Parameters
    ----------
    activation:
        Registry name or spec. Must carry a kernel for ``transform``.
    transform:
        ``"laplace"``, ``"fourier"`` or ``"mellin"``.
    features:
        Number of output features (spectral nodes). Default ``1``.
    init_shift:
        Initial value of the *argument offset*, i.e. of ``T``'s spectral
        variable at ``x = 0``. Must lie strictly inside the region of
        convergence. Defaults to one unit above ``min_argument`` (or ``0`` for
        an entire transform).
    init_scale:
        Initial multiplier on the input. Default ``1``.
    learnable:
        If ``False``, ``scale`` and the offset are registered as buffers rather
        than parameters, giving a fixed spectral probe.
    """

    def __init__(
        self,
        activation: str | ActivationSpec[Tensor],
        transform: TransformName,
        *,
        features: int = 1,
        init_shift: float | None = None,
        init_scale: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        spec, kernel = _resolve(activation, transform)
        self.activation_name = spec.name
        self.transform = transform
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

        dtype = torch.get_default_dtype()
        if lower is None:
            raw_init = torch.full((features,), float(init_shift), dtype=dtype)
        else:
            # softplus^-1 so that min_argument + softplus(raw) == init_shift.
            gap = float(init_shift) - lower
            raw_init = torch.full((features,), math.log(math.expm1(gap)), dtype=dtype)
        scale_init = torch.full((features,), float(init_scale), dtype=dtype)

        if learnable:
            self.raw_shift = nn.Parameter(raw_init)
            self.scale = nn.Parameter(scale_init)
        else:
            self.register_buffer("raw_shift", raw_init)
            self.register_buffer("scale", scale_init)

    def argument(self, x: Tensor) -> Tensor:
        """Spectral argument ``scale * x + shift`` this block evaluates at.

        Exposed because the reparameterisation makes the effective shift a
        function of the raw parameter; inspecting it is how a caller sees where
        in the region of convergence training has moved.
        """
        shift = (
            self.raw_shift
            if self.min_argument is None
            else self.min_argument + nn.functional.softplus(self.raw_shift)
        )
        return self.scale * x + shift

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the transform at ``scale * x + shift``, broadcasting over features.

        ``x`` of shape ``(...,)`` or ``(..., 1)`` broadcasts to ``(..., features)``;
        ``x`` of shape ``(..., features)`` is used per feature.
        """
        if x.ndim == 0:
            x = x.reshape(1)
        return self._kernel(self.argument(x))

    def extra_repr(self) -> str:
        return (
            f"activation={self.activation_name!r}, transform={self.transform!r}, "
            f"features={self.scale.shape[0]}, region={self.region!r}"
        )


class LaplaceTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Laplace transform."""

    def __init__(self, activation: str | ActivationSpec[Tensor], **kwargs: object) -> None:
        super().__init__(activation, "laplace", **kwargs)  # type: ignore[arg-type]


class FourierTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Fourier transform."""

    def __init__(self, activation: str | ActivationSpec[Tensor], **kwargs: object) -> None:
        super().__init__(activation, "fourier", **kwargs)  # type: ignore[arg-type]


class MellinTransform(TransformBlock):
    """:class:`TransformBlock` specialised to the Mellin transform."""

    def __init__(self, activation: str | ActivationSpec[Tensor], **kwargs: object) -> None:
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic activation specification.

The omnibias framework parameterises every closed-form derivative-tower
kernel by an :class:`ActivationSpec`, a small bundle of metadata describing
one base activation (forward, derivative, fastpath, Riccati polynomial,
GLM noise model, operator role).

This module defines the *generic* spec that does not bind to any specific
tensor library. The :mod:`omnibias.torch.activations.registry` and
:mod:`omnibias.jax.activations` modules each instantiate concrete dataclasses
that pin ``TensorT`` to their respective array type
(:class:`torch.Tensor` and :class:`jax.Array`).

Keeping the spec definition in :mod:`omnibias.core` guarantees the shape of
every backend's registry is identical by construction, which is what the
cross-backend parity test (`tests/test_cross_backend_parity.py`) checks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

# Generic over the backend tensor type. Both backends ship a concrete
# subclass / specialisation that fixes TensorT to torch.Tensor or jax.Array.
TensorT = TypeVar("TensorT")

#: Type alias for ``sigma: TensorT -> TensorT``.
TensorFn = Callable[[TensorT], TensorT]
#: Type alias for ``sigma^(n): (TensorT, int) -> TensorT``.
NthDerivativeFn = Callable[[TensorT, int], TensorT]
#: Type alias for an antiderivative ``S`` such that ``dS/dz = sigma``.
IntegralFn = Callable[[TensorT], TensorT]
#: Type alias for an integral transform ``T[sigma]`` evaluated at its own
#: spectral variable (``s`` for Laplace / Mellin, ``xi`` for Fourier).
TransformFn = Callable[[TensorT], TensorT]


@dataclass(frozen=True)
class TransformKernels(Generic[TensorT]):
    r"""Closed-form integral transforms of one activation.

    Each field is the transform *of the activation as registered*, evaluated at
    its spectral variable, or ``None`` when no closed form exists (or when the
    defining integral diverges for that activation). A ``None`` field is a
    deliberate, documented gap -- never a placeholder for an unwritten kernel.
    The half-plane / region of convergence of every shipped kernel is recorded
    in :mod:`omnibias.core.transforms`.

    Conventions (fixed once here so both backends agree):

    .. math::

        \mathcal{L}[\sigma](s) = \int_0^\infty \sigma(z)\, e^{-sz}\, dz, \qquad
        \mathcal{F}[\sigma](\xi) = \int_{-\infty}^{\infty} \sigma(z)\,
            e^{-i\xi z}\, dz, \qquad
        \mathcal{M}[\sigma](s) = \int_0^\infty \sigma(z)\, z^{s-1}\, dz.

    The Fourier convention is the non-unitary angular-frequency one (no
    ``1/sqrt(2 pi)`` prefactor). Only activations whose transform is *real* on
    the real axis are registered, so kernels return real tensors.

    Attributes
    ----------
    laplace : Callable[[TensorT], TensorT], optional
        ``L[sigma](s)`` for real ``s`` in the region of convergence.
    fourier : Callable[[TensorT], TensorT], optional
        ``F[sigma](xi)``. Registered only for ``L^1(R)`` activations; a
        saturating or growing activation has a *distributional* Fourier
        transform (Dirac deltas, principal values) that no tensor kernel can
        represent honestly, so it stays ``None``.
    mellin : Callable[[TensorT], TensorT], optional
        ``M[sigma](s)`` for real ``s`` in the region of convergence.
    """

    laplace: Callable[[TensorT], TensorT] | None = None
    fourier: Callable[[TensorT], TensorT] | None = None
    mellin: Callable[[TensorT], TensorT] | None = None


@dataclass(frozen=True)
class ActivationSpec(Generic[TensorT]):
    """Full description of a base activation, generic over the tensor type.

    Attributes
    ----------
    name : str
        Lookup key (lowercase, no whitespace).
    forward : Callable[[TensorT], TensorT]
        ``sigma(z)``.
    derivative : Callable[[TensorT], TensorT], optional
        First derivative ``sigma'(z)`` in closed form. ``None`` when not
        available analytically.
    fastpath : Callable[[TensorT, int], TensorT], optional
        Closed-form ``sigma^(n)(z)`` for any non-negative integer order
        ``n``. ``None`` when no general derivative formula is known.
    integral : Callable[[TensorT], TensorT], optional
        Closed-form antiderivative ``S(z)`` with ``S'(z) = sigma(z)``.
        Definite bias-window integrals are evaluated as
        ``S(z + b_hi) - S(z + b_lo)``. ``None`` when no stable primitive is
        provided by the backend.
    riccati_polynomial : tuple[float, ...], optional
        Coefficients of the polynomial ``P`` such that
        ``sigma'(z) = P(sigma(z))`` (Riccati class). ``None`` for
        activations not in the Riccati class.
    noise_model : str
        GLM family for which ``sigma`` is the log-partition function;
        ``"none"`` if the activation does not arise as a log-partition.
    operator_role : str
        One-line description of the canonical operator role of the K=2
        bias-collapse unit with this activation.
    aliases : Sequence[str]
        Additional lookup names (lowercase, no whitespace).
    limit_pos_inf : float, optional
        Saturation limit ``lim_{z->+inf} sigma(z)`` when it is finite; ``None``
        when the activation diverges or has no recorded right asymptote. This is
        the ``beta->inf`` saturation reused by the binary surrogates and the jet
        ``lim`` operator.
    limit_neg_inf : float, optional
        Saturation limit ``lim_{z->-inf} sigma(z)`` when finite; ``None``
        otherwise.
    transforms : TransformKernels[TensorT], optional
        Closed-form Laplace / Fourier / Mellin transforms of ``sigma``.
        ``None`` (the default) means the backend registered no transform at
        all for this activation, which is distinct from registering a
        :class:`TransformKernels` whose individual fields are ``None``.
    """

    name: str
    forward: Callable[[TensorT], TensorT]
    derivative: Callable[[TensorT], TensorT] | None = None
    fastpath: Callable[[TensorT, int], TensorT] | None = None
    integral: Callable[[TensorT], TensorT] | None = None
    riccati_polynomial: tuple[float, ...] | None = None
    noise_model: str = "none"
    operator_role: str = ""
    aliases: Sequence[str] = field(default_factory=tuple)
    limit_pos_inf: float | None = None
    limit_neg_inf: float | None = None
    transforms: TransformKernels[TensorT] | None = None


def saturation_limit(spec: ActivationSpec[TensorT], sign: float) -> float | None:
    """Return the saturation limit ``lim_{z -> sign*inf} sigma(z)`` if recorded.

    ``sign > 0`` selects :attr:`ActivationSpec.limit_pos_inf`; ``sign < 0``
    selects :attr:`ActivationSpec.limit_neg_inf`. Returns ``None`` when the
    backend did not record a finite asymptote (e.g. diverging activations such
    as ``exp`` or ``softplus`` on the ``+inf`` side).
    """
    if sign > 0:
        return spec.limit_pos_inf
    if sign < 0:
        return spec.limit_neg_inf
    raise ValueError("sign must be nonzero")


def make_tempered_fastpath(
    base_fastpath: NthDerivativeFn[TensorT],
    beta: float | TensorT,
    *,
    scale_power: int = 0,
) -> NthDerivativeFn[TensorT]:
    r"""Beta-scaled derivative tower of a temperature-smoothed surrogate.

    Given a base activation ``g`` with a closed-form tower ``g^(n)`` and a
    temperature ``beta > 0``, the *tempered* surrogate

    .. math::

        f_\beta(z) = \beta^{-p}\, g(\beta z), \qquad p \in \{0, 1\}

    has, by the chain rule, the closed-form tower

    .. math::

        f_\beta^{(n)}(z) = \beta^{\,n-p}\, g^{(n)}(\beta z).

    ``scale_power = 0`` gives ``g(beta z)`` (bounded surrogates such as
    ``tanh(beta z) -> sign(z)`` and ``sigmoid(beta z) -> Heaviside``);
    ``scale_power = 1`` gives ``g(beta z) / beta`` (e.g.
    ``softplus(beta z) / beta -> relu``). ``n = 0`` reproduces ``f_beta``
    itself, so ``forward`` and ``derivative`` can both route through the
    returned kernel.

    ``beta`` may be a Python float or a backend tensor (an ``nn.Parameter`` /
    traced ``Array``), making the surrogate differentiable in the temperature.
    Negative and unimplemented orders raise through ``base_fastpath``.
    """

    def tempered_fastpath(z: TensorT, n: int) -> TensorT:
        # Intermediate arithmetic is typed ``Any``: the tensor TypeVar is
        # unconstrained (no known ``__mul__`` / ``__pow__``), but every backend
        # tensor supports these ops. ``base_fastpath`` still enforces the
        # ``n < 0`` / unimplemented-order contract.
        b: Any = beta
        coeff: Any = b ** (n - scale_power)
        scaled: Any = b * z
        out: TensorT = coeff * base_fastpath(scaled, n)
        return out

    return tempered_fastpath


def make_tempered_transforms(
    base: TransformKernels[TensorT],
    beta: float | TensorT,
    *,
    scale_power: int = 0,
) -> TransformKernels[TensorT]:
    r"""Integral transforms of a temperature-smoothed surrogate.

    For ``f_beta(z) = beta^{-p} g(beta z)`` with ``beta > 0``, each transform
    obeys an *exact* scaling law obtained by the substitution ``u = beta z``:

    .. math::

        \mathcal{L}[f_\beta](s) &= \beta^{-(p+1)}\,\mathcal{L}[g](s/\beta), \\
        \mathcal{F}[f_\beta](\xi) &= \beta^{-(p+1)}\,\mathcal{F}[g](\xi/\beta), \\
        \mathcal{M}[f_\beta](s) &= \beta^{-p-s}\,\mathcal{M}[g](s).

    The Laplace and Fourier laws rescale the spectral variable; the Mellin law
    does not (it is the multiplicative-convolution character of the Mellin
    transform), which is why the temperature enters through the exponent
    ``-p - s`` instead. Every kernel that is ``None`` on ``base`` stays
    ``None``: tempering never manufactures a transform that the base activation
    does not have.

    The laws assume a strictly positive real ``beta``; tempering by a
    non-positive temperature reflects or collapses the integration domain and
    is not covered. ``beta`` may be a backend tensor, so the surrogate's
    transforms stay differentiable in the temperature.
    """
    b: Any = beta
    base_laplace = base.laplace
    base_fourier = base.fourier
    base_mellin = base.mellin

    laplace: Callable[[TensorT], TensorT] | None = None
    if base_laplace is not None:

        def _laplace(s: TensorT) -> TensorT:
            out: TensorT = base_laplace(s / b) / (b ** (scale_power + 1))
            return out

        laplace = _laplace

    fourier: Callable[[TensorT], TensorT] | None = None
    if base_fourier is not None:

        def _fourier(xi: TensorT) -> TensorT:
            out: TensorT = base_fourier(xi / b) / (b ** (scale_power + 1))
            return out

        fourier = _fourier

    mellin: Callable[[TensorT], TensorT] | None = None
    if base_mellin is not None:

        def _mellin(s: TensorT) -> TensorT:
            # As in make_tempered_fastpath: the tensor TypeVar is unconstrained,
            # so the exponent arithmetic is typed Any. Every backend tensor
            # supports __rsub__ and __rpow__ against a Python scalar.
            power: Any = -scale_power
            exponent: Any = power - s
            out: TensorT = base_mellin(s) * b**exponent
            return out

        mellin = _mellin

    return TransformKernels(laplace=laplace, fourier=fourier, mellin=mellin)


def tempered(
    base: ActivationSpec[TensorT],
    beta: float | TensorT,
    *,
    scale: str = "unit",
    name: str | None = None,
    noise_model: str = "none",
    operator_role: str = "",
    aliases: Sequence[str] = (),
    limit_pos_inf: float | None = None,
    limit_neg_inf: float | None = None,
) -> ActivationSpec[TensorT]:
    r"""Wrap a base activation into its beta-tempered surrogate :class:`ActivationSpec`.

    The base must carry a fastpath (its closed-form tower ``g^(n)``). The result
    is a new spec whose whole tower is :func:`make_tempered_fastpath` of the
    base -- so it inherits the base's maximum supported order for free. As
    ``beta -> inf`` the surrogate approaches a hard activation (``softplus ->
    relu``, ``tanh -> sign``, ``sigmoid -> Heaviside``), with the singular
    (Dirac) part emerging as the beta-scaled bump in the higher orders.

    Parameters
    ----------
    base
        Base :class:`ActivationSpec` with a non-``None`` ``fastpath``.
    beta
        Temperature (float or backend tensor). Larger is sharper.
    scale
        ``"unit"`` for ``g(beta z)`` or ``"one_over_beta"`` for
        ``g(beta z) / beta``.
    name, noise_model, operator_role, aliases, limit_pos_inf, limit_neg_inf
        Metadata for the produced spec (limits are recorded explicitly rather
        than auto-derived, so callers stay honest about the surrogate's
        asymptotes).
    """
    if base.fastpath is None:
        raise ValueError(
            f"tempered() requires a base activation with a fastpath; {base.name!r} has none."
        )
    if scale == "one_over_beta":
        p = 1
    elif scale == "unit":
        p = 0
    else:
        raise ValueError(f"scale must be 'unit' or 'one_over_beta', got {scale!r}")

    fastpath = make_tempered_fastpath(base.fastpath, beta, scale_power=p)
    base_integral = base.integral

    def _forward(z: TensorT) -> TensorT:
        return fastpath(z, 0)

    def _derivative(z: TensorT) -> TensorT:
        return fastpath(z, 1)

    integral: Callable[[TensorT], TensorT] | None = None
    if base_integral is not None:

        def _integral(z: TensorT) -> TensorT:
            b: Any = beta
            out: TensorT = base_integral(b * z) / (b ** (p + 1))
            return out

        integral = _integral

    transforms: TransformKernels[TensorT] | None = None
    if base.transforms is not None:
        transforms = make_tempered_transforms(base.transforms, beta, scale_power=p)

    return ActivationSpec(
        name=name if name is not None else f"tempered_{base.name}",
        forward=_forward,
        derivative=_derivative,
        fastpath=fastpath,
        integral=integral,
        riccati_polynomial=None,
        noise_model=noise_model,
        operator_role=operator_role,
        aliases=tuple(aliases),
        limit_pos_inf=limit_pos_inf,
        limit_neg_inf=limit_neg_inf,
        transforms=transforms,
    )


__all__ = [
    "ActivationSpec",
    "IntegralFn",
    "NthDerivativeFn",
    "TensorFn",
    "TensorT",
    "TransformFn",
    "TransformKernels",
    "make_tempered_fastpath",
    "make_tempered_transforms",
    "saturation_limit",
    "tempered",
]

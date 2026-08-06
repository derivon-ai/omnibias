# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX-backed activation registry.

Bit-identical mirror of the PyTorch activation dictionary. Families:

* Smooth Riccati-class (closed-form fastpath for every n): ``sigmoid``,
  ``tanh``, ``softplus``, ``gaussian``, ``exp``, and the trig / hyperbolic set.
* Analytic products, **exact all orders**: ``silu``, ``gelu``, ``mish``.
* Proximal-class: ``arctan``, ``log1pu2`` (``n in {0, 1, 2}``).
* Piecewise, **almost-everywhere / regular-part all-orders** tower (singular
  delta dropped): ``relu``, ``huber``, ``leaky_relu``, ``prelu``, ``relu6``,
  ``hardtanh``, ``hardsigmoid``, ``hardswish``, ``elu``, ``selu``, ``celu``,
  ``softshrink``, ``hardshrink``, ``threshold``, ``abs``, ``sign``, ``step``,
  ``softsign``. See :mod:`omnibias.jax._activations_piecewise`.
* Beta-tempered smooth surrogates (``-> hard`` as ``beta -> inf``):
  ``soft_relu``, ``soft_step``, ``soft_sign`` (= ``smooth_sign``), ``soft_abs``
  (= ``softabs``). See :mod:`omnibias.jax._activations_tempered`.

Every spec exposes ``forward(z)``, ``derivative(z)`` (== fastpath at ``n=1``),
and ``fastpath(z, n)`` (closed form up to the maximum supported order).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeAlias

from omnibias.core.polynomials import mish_inner_coeffs
from omnibias.core.spec import ActivationSpec as _CoreActivationSpec
from omnibias.jax import _fastpath
from omnibias.jax.transforms import (
    COS_TRANSFORMS,
    COSH_TRANSFORMS,
    EXP_TRANSFORMS,
    GAUSSIAN_TRANSFORMS,
    RELU_TRANSFORMS,
    SECH_TRANSFORMS,
    SIGMOID_TRANSFORMS,
    SIN_TRANSFORMS,
    SINH_TRANSFORMS,
    TANH_TRANSFORMS,
)

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

ArrayFn = Callable[[Array], Array]
NthDerivativeFn = Callable[[Array, int], Array]
IntegralFn = Callable[[Array], Array]


# ---------------------------------------------------------------------------
# Spec + registry
# ---------------------------------------------------------------------------


#: JAX-specialised :class:`omnibias.core.spec.ActivationSpec`. The field
#: shape is identical to the torch backend's spec; ``TensorT`` is pinned
#: to :class:`jax.Array`. Re-exported under the ``JaxActivationSpec`` name
#: to preserve the v0.1 public API.
JaxActivationSpec: TypeAlias = _CoreActivationSpec[Array]


_REGISTRY: dict[str, JaxActivationSpec] = {}


def register_activation(spec: JaxActivationSpec) -> JaxActivationSpec:
    if not spec.name:
        raise ValueError("JaxActivationSpec.name must be a non-empty string.")
    name = spec.name.lower()
    if name in _REGISTRY and _REGISTRY[name] is not spec:
        raise ValueError(f"Activation {name!r} is already registered with a different spec.")
    _REGISTRY[name] = spec
    for alias in spec.aliases:
        akey = alias.lower()
        if akey in _REGISTRY and _REGISTRY[akey] is not spec:
            raise ValueError(f"Alias {alias!r} for activation {name!r} clashes.")
        _REGISTRY[akey] = spec
    return spec


def get_activation(name: str | JaxActivationSpec) -> JaxActivationSpec:
    if isinstance(name, _CoreActivationSpec):
        return name
    if not isinstance(name, str):
        raise TypeError(f"Expected str or JaxActivationSpec, got {type(name).__name__}: {name!r}")
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted({s.name for s in _REGISTRY.values()}))
        raise KeyError(f"Unknown activation {name!r}. Known: {known}.")
    return _REGISTRY[key]


def list_activations() -> list[str]:
    return sorted({s.name for s in _REGISTRY.values()})


def is_registered(name: str) -> bool:
    return isinstance(name, str) and name.lower() in _REGISTRY


# ---------------------------------------------------------------------------
# Smooth Riccati-class
# ---------------------------------------------------------------------------


def _sigmoid(z: Array) -> Array:
    return _fastpath.jax_sigmoid(z)


def _sigmoid_derivative(z: Array) -> Array:
    s = _fastpath.jax_sigmoid(z)
    return s * (1.0 - s)


def _sigmoid_integral(z: Array) -> Array:
    return jnp.logaddexp(jnp.zeros_like(z), z)


SIGMOID = register_activation(
    JaxActivationSpec(
        name="sigmoid",
        transforms=SIGMOID_TRANSFORMS,
        forward=_sigmoid,
        derivative=_sigmoid_derivative,
        fastpath=_fastpath.sigmoid_nth_derivative,
        integral=_sigmoid_integral,
        riccati_polynomial=(0.0, 1.0, -1.0),
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> Bernoulli variance s(1-s); Newton/IRLS step for logistic regression."
        ),
        aliases=("logistic",),
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )
)


def _tanh(z: Array) -> Array:
    return jnp.tanh(z)


def _tanh_derivative(z: Array) -> Array:
    t = jnp.tanh(z)
    return 1.0 - t * t


def _tanh_integral(z: Array) -> Array:
    return z + jnp.logaddexp(jnp.zeros_like(z), -2.0 * z) - math.log(2.0)


TANH = register_activation(
    JaxActivationSpec(
        name="tanh",
        transforms=TANH_TRANSFORMS,
        forward=_tanh,
        derivative=_tanh_derivative,
        fastpath=_fastpath.tanh_nth_derivative,
        integral=_tanh_integral,
        riccati_polynomial=(1.0, 0.0, -1.0),
        noise_model="symmetric_bernoulli",
        operator_role="K=2 collapse -> 1 - tanh^2; symmetric IRLS bell.",
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


def _softplus(z: Array) -> Array:
    # Numerically stable log(1 + exp(z)).
    return jnp.logaddexp(jnp.zeros_like(z), z)


def _softplus_derivative(z: Array) -> Array:
    return _fastpath.jax_sigmoid(z)


SOFTPLUS = register_activation(
    JaxActivationSpec(
        name="softplus",
        forward=_softplus,
        derivative=_softplus_derivative,
        fastpath=_fastpath.softplus_nth_derivative,
        riccati_polynomial=None,
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> sigmoid (Bernoulli mean); "
            "canonical PINN base for diffusion-class PDEs."
        ),
        limit_neg_inf=0.0,  # softplus -> 0 as z -> -inf; diverges as z -> +inf
    )
)


def _gaussian(z: Array) -> Array:
    return _fastpath.gaussian_forward(z)


def _gaussian_derivative(z: Array) -> Array:
    return -z * _fastpath.gaussian_forward(z)


def _gaussian_integral(z: Array) -> Array:
    return math.sqrt(math.pi / 2.0) * erf(z / math.sqrt(2.0))


GAUSSIAN = register_activation(
    JaxActivationSpec(
        name="gaussian",
        transforms=GAUSSIAN_TRANSFORMS,
        forward=_gaussian,
        derivative=_gaussian_derivative,
        fastpath=_fastpath.gaussian_nth_derivative,
        integral=_gaussian_integral,
        riccati_polynomial=None,
        noise_model="gaussian_kernel",
        operator_role=(
            "K=2 collapse -> -z * exp(-z^2/2); RBF/Hermite spectral basis, "
            "Laplacian-of-Gaussian conv kernel."
        ),
        aliases=("rbf",),
        limit_pos_inf=0.0,
        limit_neg_inf=0.0,
    )
)


def _exp_forward(z: Array) -> Array:
    return jnp.exp(z)


def _exp_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return jnp.exp(z)


EXP = register_activation(
    JaxActivationSpec(
        name="exp",
        transforms=EXP_TRANSFORMS,
        forward=_exp_forward,
        derivative=_exp_forward,
        fastpath=_exp_fastpath,
        integral=_exp_forward,
        riccati_polynomial=(0.0, 1.0),
        noise_model="poisson",
        operator_role=(
            "K=2 collapse -> exp(z) (eigenfunction of d/dz); "
            "log-link Newton step for Poisson regression."
        ),
        limit_neg_inf=0.0,  # exp -> 0 as z -> -inf; diverges as z -> +inf
    )
)


# ---------------------------------------------------------------------------
# Proximal-class (fastpath n in {0, 1, 2})
# ---------------------------------------------------------------------------


def _arctan_forward(z: Array) -> Array:
    return jnp.arctan(z)


def _arctan_derivative(z: Array) -> Array:
    return 1.0 / (1.0 + z * z)


def _arctan_integral(z: Array) -> Array:
    return z * jnp.arctan(z) - 0.5 * jnp.log1p(z * z)


def _arctan_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _arctan_forward(z)
    if n == 1:
        return _arctan_derivative(z)
    if n == 2:
        denom = (1.0 + z * z) ** 2
        return -2.0 * z / denom
    raise NotImplementedError(f"arctan fast path only implements n in {{0, 1, 2}}, got {n}.")


ARCTAN = register_activation(
    JaxActivationSpec(
        name="arctan",
        forward=_arctan_forward,
        derivative=_arctan_derivative,
        fastpath=_arctan_fastpath,
        integral=_arctan_integral,
        riccati_polynomial=None,
        noise_model="cauchy",
        operator_role="K=2 collapse -> 1/(1+z^2); Cauchy IRLS weight.",
        aliases=("atan",),
        limit_pos_inf=math.pi / 2.0,
        limit_neg_inf=-math.pi / 2.0,
    )
)


def _log1pu2_forward(z: Array) -> Array:
    return jnp.log1p(z * z)


def _log1pu2_derivative(z: Array) -> Array:
    return 2.0 * z / (1.0 + z * z)


def _log1pu2_integral(z: Array) -> Array:
    return z * jnp.log1p(z * z) - 2.0 * z + 2.0 * jnp.arctan(z)


def _log1pu2_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _log1pu2_forward(z)
    if n == 1:
        return _log1pu2_derivative(z)
    if n == 2:
        denom = (1.0 + z * z) ** 2
        return 2.0 * (1.0 - z * z) / denom
    raise NotImplementedError(f"log1pu2 fast path only implements n in {{0, 1, 2}}, got {n}.")


LOG1PU2 = register_activation(
    JaxActivationSpec(
        name="log1pu2",
        forward=_log1pu2_forward,
        derivative=_log1pu2_derivative,
        fastpath=_log1pu2_fastpath,
        integral=_log1pu2_integral,
        riccati_polynomial=None,
        noise_model="redescending_m",
        operator_role=("K=2 collapse -> 2 z / (1 + z^2); redescending M-estimator."),
        aliases=("logcosh_dual",),
    )
)


# ---------------------------------------------------------------------------
# Kinked and analytic-product activations (all orders; ``huber`` a.e.)
# ---------------------------------------------------------------------------


_DEFAULT_HUBER_TAU = 1.0


def _huber_forward(z: Array, tau: float = _DEFAULT_HUBER_TAU) -> Array:
    abs_z = jnp.abs(z)
    quad = 0.5 * z * z
    lin = tau * (abs_z - 0.5 * tau)
    return jnp.where(abs_z <= tau, quad, lin)


def _huber_derivative(z: Array, tau: float = _DEFAULT_HUBER_TAU) -> Array:
    return jnp.clip(z, -tau, tau)


def _huber_integral(z: Array, tau: float = _DEFAULT_HUBER_TAU) -> Array:
    abs_z = jnp.abs(z)
    inside = z * z * z / 6.0
    outside_mag = 0.5 * tau * abs_z * abs_z - 0.5 * tau * tau * abs_z + tau**3 / 6.0
    outside = jnp.sign(z) * outside_mag
    return jnp.where(abs_z <= tau, inside, outside)


def _huber_fastpath(z: Array, n: int, tau: float = _DEFAULT_HUBER_TAU) -> Array:
    """Closed-form ``huber^(n)`` on the almost-everywhere / regular-part convention.

    ``n = 2`` is the indicator of the quadratic region, ``n >= 3`` is zero; the
    singular part at the ``+/-tau`` kinks is dropped.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _huber_forward(z, tau)
    if n == 1:
        return _huber_derivative(z, tau)
    if n == 2:
        return (jnp.abs(z) <= tau).astype(z.dtype)
    return jnp.zeros_like(z)


HUBER = register_activation(
    JaxActivationSpec(
        name="huber",
        forward=_huber_forward,
        derivative=_huber_derivative,
        fastpath=_huber_fastpath,
        integral=_huber_integral,
        riccati_polynomial=None,
        noise_model="laplace_smoothed",
        operator_role=("K=2 collapse -> clip(z, -tau, tau); ISTA soft-shrink step."),
    )
)


def _silu_forward(z: Array) -> Array:
    return z * _fastpath.jax_sigmoid(z)


def _silu_derivative(z: Array) -> Array:
    s = _fastpath.jax_sigmoid(z)
    return s + z * s * (1.0 - s)


def _silu_fastpath(z: Array, n: int) -> Array:
    """Exact closed-form ``silu^(n)`` (all orders) via Leibniz on ``z * sigmoid(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _silu_forward(z)
    return z * _fastpath.sigmoid_nth_derivative(z, n) + n * _fastpath.sigmoid_nth_derivative(
        z, n - 1
    )


SILU = register_activation(
    JaxActivationSpec(
        name="silu",
        forward=_silu_forward,
        derivative=_silu_derivative,
        fastpath=_silu_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "K=2 collapse -> sigmoid + z * sigmoid * (1 - sigmoid); "
            "smoothed gate for transformer FFN compatibility."
        ),
        aliases=("swish",),
    )
)


_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: Array) -> Array:
    return 0.5 * (1.0 + erf(z * _INV_SQRT_2))


def _normal_pdf(z: Array) -> Array:
    return _INV_SQRT_2PI * jnp.exp(-0.5 * z * z)


def _gelu_forward(z: Array) -> Array:
    return z * _normal_cdf(z)


def _gelu_derivative(z: Array) -> Array:
    return _normal_cdf(z) + z * _normal_pdf(z)


def _gelu_integral(z: Array) -> Array:
    cdf = _normal_cdf(z)
    pdf = _normal_pdf(z)
    return 0.5 * ((z * z - 1.0) * cdf + z * pdf)


def _normal_cdf_nth(z: Array, k: int) -> Array:
    """``Phi^(k)(z)``: the CDF for ``k = 0``, else ``phi^(k-1)`` via the Hermite tower."""
    if k == 0:
        return _normal_cdf(z)
    return _fastpath.gaussian_nth_derivative(z, k - 1) * _INV_SQRT_2PI


def _gelu_fastpath(z: Array, n: int) -> Array:
    """Exact closed-form ``gelu^(n)`` (all orders) via Leibniz on ``z * Phi(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _gelu_forward(z)
    return z * _normal_cdf_nth(z, n) + n * _normal_cdf_nth(z, n - 1)


GELU = register_activation(
    JaxActivationSpec(
        name="gelu",
        forward=_gelu_forward,
        derivative=_gelu_derivative,
        fastpath=_gelu_fastpath,
        integral=_gelu_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="K=2 collapse -> Phi(z) + z * phi(z); smoothed half-space gate.",
    )
)


def _relu_forward(z: Array) -> Array:
    return jnp.maximum(z, 0.0)


def _relu_derivative(z: Array) -> Array:
    # Matches the PyTorch convention H(0) = 0.
    return (z > 0).astype(z.dtype)


def _relu_integral(z: Array) -> Array:
    r = jnp.maximum(z, 0.0)
    return 0.5 * r * r


def _relu_fastpath(z: Array, n: int) -> Array:
    """Closed-form ``relu^(n)`` on the almost-everywhere / regular-part convention.

    ``n = 0`` value, ``n = 1`` Heaviside (``H(0) = 0``), ``n >= 2`` zero away
    from the kink (the singular delta at ``z = 0`` is dropped). Smooth twin:
    ``soft_relu``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _relu_forward(z)
    if n == 1:
        return _relu_derivative(z)
    return jnp.zeros_like(z)


RELU = register_activation(
    JaxActivationSpec(
        name="relu",
        transforms=RELU_TRANSFORMS,
        forward=_relu_forward,
        derivative=_relu_derivative,
        fastpath=_relu_fastpath,
        integral=_relu_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=("K=2 collapse -> Heaviside step; equality-constraint indicator."),
    )
)


# ---------------------------------------------------------------------------
# NQS-friendly activations (added for lattice work).
#
# These all carry closed-form first and second derivatives so the omnibias
# higher-order derivative tower can use them directly in the SR / quantum-
# geometric-tensor pipeline. Where applicable the (real-valued) third
# derivative is also closed-form.
# ---------------------------------------------------------------------------


def _log_cosh_forward(z: Array) -> Array:
    absz = jnp.abs(z)
    return absz + jnp.log1p(jnp.exp(-2.0 * absz)) - jnp.log(2.0)


def _log_cosh_derivative(z: Array) -> Array:
    return jnp.tanh(z)


def _log_cosh_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _log_cosh_forward(z)
    if n == 1:
        return _log_cosh_derivative(z)
    if n == 2:
        t = jnp.tanh(z)
        return 1.0 - t * t
    if n == 3:
        t = jnp.tanh(z)
        return -2.0 * t * (1.0 - t * t)
    raise NotImplementedError(f"log_cosh fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


LOG_COSH = register_activation(
    JaxActivationSpec(
        name="log_cosh",
        forward=_log_cosh_forward,
        derivative=_log_cosh_derivative,
        fastpath=_log_cosh_fastpath,
        riccati_polynomial=None,
        noise_model="laplace_smoothed",
        operator_role=(
            "Smooth-|z| activation: log(cosh(z)) ≈ |z| for large |z|, "
            "≈ z²/2 for small. Standard log-amplitude in spin-NQS; "
            "first derivative is tanh, second is 1-tanh²."
        ),
        aliases=("logcosh",),
    )
)


_DEFAULT_SOFTABS_EPS = 1e-3


def _softabs_forward(z: Array, eps: float = _DEFAULT_SOFTABS_EPS) -> Array:
    return jnp.sqrt(z * z + eps * eps) - eps


def _softabs_derivative(z: Array, eps: float = _DEFAULT_SOFTABS_EPS) -> Array:
    return z / jnp.sqrt(z * z + eps * eps)


def _softabs_integral(z: Array, eps: float = _DEFAULT_SOFTABS_EPS) -> Array:
    root = jnp.sqrt(z * z + eps * eps)
    return 0.5 * (z * root + eps * eps * jnp.arcsinh(z / eps)) - eps * z


def _softabs_fastpath(z: Array, n: int, eps: float = _DEFAULT_SOFTABS_EPS) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _softabs_forward(z, eps)
    if n == 1:
        return _softabs_derivative(z, eps)
    if n == 2:
        denom = (z * z + eps * eps) ** 1.5
        return eps * eps / denom
    raise NotImplementedError(f"softabs fast path only implements n in {{0, 1, 2}}, got {n}.")


SOFTABS = register_activation(
    JaxActivationSpec(
        name="softabs",
        forward=_softabs_forward,
        derivative=_softabs_derivative,
        fastpath=_softabs_fastpath,
        integral=_softabs_integral,
        riccati_polynomial=None,
        noise_model="huber_smoothed",
        operator_role=(
            "Smooth absolute value: sqrt(z² + ε²) - ε. C² everywhere; the "
            "eps-tempered abs surrogate (-> |z| as eps -> 0). Useful in "
            "Jastrow factors and complex-amplitude magnitudes."
        ),
        aliases=("soft_abs",),
    )
)


_DEFAULT_SMOOTH_SIGN_T = 1.0


def _smooth_sign_forward(z: Array, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Array:
    return jnp.tanh(z / T)


def _smooth_sign_derivative(z: Array, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Array:
    t = jnp.tanh(z / T)
    return (1.0 - t * t) / T


def _smooth_sign_integral(z: Array, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Array:
    return T * _log_cosh_forward(z / T)


def _smooth_sign_fastpath(z: Array, n: int, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Array:
    """Closed-form all-orders ``d^n/dz^n tanh(z / T) = tanh^(n)(z/T) / T^n``.

    The beta-tempered ``sign`` surrogate (``beta = 1 / T``).
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _smooth_sign_forward(z, T)
    return _fastpath.tanh_nth_derivative(z / T, n) / (T**n)


SMOOTH_SIGN = register_activation(
    JaxActivationSpec(
        name="smooth_sign",
        forward=_smooth_sign_forward,
        derivative=_smooth_sign_derivative,
        fastpath=_smooth_sign_fastpath,
        integral=_smooth_sign_integral,
        riccati_polynomial=None,
        noise_model="symmetric_bernoulli_tempered",
        operator_role=(
            "Temperature-controlled smooth sign: tanh(z/T) → sign(z) as "
            "T → 0. The beta-tempered sign surrogate; used in variational "
            "annealing schedules."
        ),
        aliases=("soft_sign",),
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


def _mish_forward(z: Array) -> Array:
    # mish(z) = z * tanh(softplus(z)).
    sp = jnp.logaddexp(jnp.zeros_like(z), z)  # softplus
    return z * jnp.tanh(sp)


def _mish_inner_nth(z: Array, n: int) -> Array:
    """``g^(n)(z)`` for ``g(z) = tanh(softplus(z))`` via the shared ``(t, s)`` tower."""
    t = jnp.tanh(jnp.logaddexp(jnp.zeros_like(z), z))
    if n == 0:
        return t
    s = _fastpath.jax_sigmoid(z)
    acc = jnp.zeros_like(z)
    for i, j, c in mish_inner_coeffs(n):
        acc = acc + c * t**i * s**j
    return acc


def _mish_derivative(z: Array) -> Array:
    return _mish_fastpath(z, 1)


def _mish_fastpath(z: Array, n: int) -> Array:
    """Exact closed-form ``mish^(n)`` (all orders); ``mish = z * g``, Leibniz + ``(t, s)`` tower."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _mish_forward(z)
    return z * _mish_inner_nth(z, n) + n * _mish_inner_nth(z, n - 1)


MISH = register_activation(
    JaxActivationSpec(
        name="mish",
        forward=_mish_forward,
        derivative=_mish_derivative,
        fastpath=_mish_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Mish: z * tanh(softplus(z)). Self-gated residual activation "
            "useful in transformer FFN blocks; smoother than ReLU."
        ),
    )
)


# ---------------------------------------------------------------------------
# Trigonometric / hyperbolic activations (physics).
#
# These mirror :mod:`omnibias.activations.trigonometric` in PyTorch so the
# parity test in ``tests/test_jax_parity.py`` covers them. Closed-form n-th
# derivatives:
#
#   sin / cos       — every n via shifted-argument identity
#   sinh / cosh     — every n via even/odd alternation
#   tan / cot       — Riccati class, n in {0, 1, 2, 3}
#   sech            — n in {0, 1, 2, 3}
#   coth            — Riccati class, n in {0, 1, 2, 3}
# ---------------------------------------------------------------------------


_HALF_PI = 0.5 * math.pi


def _sin_forward(z: Array) -> Array:
    return jnp.sin(z)


def _sin_derivative(z: Array) -> Array:
    return jnp.cos(z)


def _sin_integral(z: Array) -> Array:
    return -jnp.cos(z)


def _sin_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return jnp.sin(z + n * _HALF_PI)


SIN = register_activation(
    JaxActivationSpec(
        name="sin",
        transforms=SIN_TRANSFORMS,
        forward=_sin_forward,
        derivative=_sin_derivative,
        fastpath=_sin_fastpath,
        integral=_sin_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding; sin(z + n*pi/2) gives every order in closed form."
        ),
    )
)


def _cos_forward(z: Array) -> Array:
    return jnp.cos(z)


def _cos_derivative(z: Array) -> Array:
    return -jnp.sin(z)


def _cos_integral(z: Array) -> Array:
    return jnp.sin(z)


def _cos_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return jnp.cos(z + n * _HALF_PI)


COS = register_activation(
    JaxActivationSpec(
        name="cos",
        transforms=COS_TRANSFORMS,
        forward=_cos_forward,
        derivative=_cos_derivative,
        fastpath=_cos_fastpath,
        integral=_cos_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding; cos(z + n*pi/2) gives every order in closed form."
        ),
    )
)


def _sinh_forward(z: Array) -> Array:
    return jnp.sinh(z)


def _sinh_derivative(z: Array) -> Array:
    return jnp.cosh(z)


def _sinh_integral(z: Array) -> Array:
    return jnp.cosh(z)


def _sinh_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return jnp.sinh(z) if (n & 1) == 0 else jnp.cosh(z)


SINH = register_activation(
    JaxActivationSpec(
        name="sinh",
        transforms=SINH_TRANSFORMS,
        forward=_sinh_forward,
        derivative=_sinh_derivative,
        fastpath=_sinh_fastpath,
        integral=_sinh_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=("Hyperbolic odd primitive; eigenfunction of d^2/dz^2 with eigenvalue +1."),
    )
)


def _cosh_forward(z: Array) -> Array:
    return jnp.cosh(z)


def _cosh_derivative(z: Array) -> Array:
    return jnp.sinh(z)


def _cosh_integral(z: Array) -> Array:
    return jnp.sinh(z)


def _cosh_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return jnp.cosh(z) if (n & 1) == 0 else jnp.sinh(z)


COSH = register_activation(
    JaxActivationSpec(
        name="cosh",
        transforms=COSH_TRANSFORMS,
        forward=_cosh_forward,
        derivative=_cosh_derivative,
        fastpath=_cosh_fastpath,
        integral=_cosh_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=("Hyperbolic even primitive; eigenfunction of d^2/dz^2 with eigenvalue +1."),
    )
)


def _tan_forward(z: Array) -> Array:
    return jnp.tan(z)


def _tan_derivative(z: Array) -> Array:
    t = jnp.tan(z)
    return 1.0 + t * t


def _tan_integral(z: Array) -> Array:
    return -jnp.log(jnp.abs(jnp.cos(z)))


def _tan_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _tan_forward(z)
    t = jnp.tan(z)
    s2 = 1.0 + t * t
    if n == 1:
        return s2
    if n == 2:
        return 2.0 * t * s2
    if n == 3:
        return 2.0 * s2 * (1.0 + 3.0 * t * t)
    raise NotImplementedError(f"tan fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


TAN = register_activation(
    JaxActivationSpec(
        name="tan",
        forward=_tan_forward,
        derivative=_tan_derivative,
        fastpath=_tan_fastpath,
        integral=_tan_integral,
        riccati_polynomial=(1.0, 0.0, 1.0),
        noise_model="none",
        operator_role=("Riccati-class periodic activation; tan'(z) = 1 + tan^2(z)."),
    )
)


def _cot_forward(z: Array) -> Array:
    return 1.0 / jnp.tan(z)


def _cot_derivative(z: Array) -> Array:
    c = _cot_forward(z)
    return -(1.0 + c * c)


def _cot_integral(z: Array) -> Array:
    return jnp.log(jnp.abs(jnp.sin(z)))


def _cot_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _cot_forward(z)
    c = _cot_forward(z)
    csc2 = 1.0 + c * c
    if n == 1:
        return -csc2
    if n == 2:
        return 2.0 * c * csc2
    if n == 3:
        return -2.0 * csc2 * (1.0 + 3.0 * c * c)
    raise NotImplementedError(f"cot fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


COT = register_activation(
    JaxActivationSpec(
        name="cot",
        forward=_cot_forward,
        derivative=_cot_derivative,
        fastpath=_cot_fastpath,
        integral=_cot_integral,
        riccati_polynomial=(-1.0, 0.0, -1.0),
        noise_model="none",
        operator_role=("Riccati-class periodic activation; cot'(z) = -(1 + cot^2(z))."),
        aliases=("ctg", "ctan"),
    )
)


def _coth_forward(z: Array) -> Array:
    return 1.0 / jnp.tanh(z)


def _coth_derivative(z: Array) -> Array:
    c = _coth_forward(z)
    return 1.0 - c * c


def _coth_integral(z: Array) -> Array:
    return jnp.log(jnp.abs(jnp.sinh(z)))


def _coth_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _coth_forward(z)
    c = _coth_forward(z)
    one_minus_c2 = 1.0 - c * c
    if n == 1:
        return one_minus_c2
    if n == 2:
        return -2.0 * c * one_minus_c2
    if n == 3:
        return -2.0 * one_minus_c2 * (1.0 - 3.0 * c * c)
    raise NotImplementedError(f"coth fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


COTH = register_activation(
    JaxActivationSpec(
        name="coth",
        forward=_coth_forward,
        derivative=_coth_derivative,
        fastpath=_coth_fastpath,
        integral=_coth_integral,
        riccati_polynomial=(1.0, 0.0, -1.0),
        noise_model="none",
        operator_role=(
            "Hyperbolic Riccati: coth'(z) = 1 - coth^2(z); same polynomial "
            "shape as tanh, but coth has a pole at z=0."
        ),
    )
)


def _sech_forward(z: Array) -> Array:
    return 1.0 / jnp.cosh(z)


def _sech_derivative(z: Array) -> Array:
    s = _sech_forward(z)
    return -s * jnp.tanh(z)


def _sech_integral(z: Array) -> Array:
    return 2.0 * jnp.arctan(jnp.tanh(0.5 * z))


def _sech_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _sech_forward(z)
    s = _sech_forward(z)
    t = jnp.tanh(z)
    if n == 1:
        return -s * t
    if n == 2:
        return s * (1.0 - 2.0 * s * s)
    if n == 3:
        return s * t * (6.0 * s * s - 1.0)
    raise NotImplementedError(f"sech fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


SECH = register_activation(
    JaxActivationSpec(
        name="sech",
        transforms=SECH_TRANSFORMS,
        forward=_sech_forward,
        derivative=_sech_derivative,
        fastpath=_sech_fastpath,
        integral=_sech_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Soliton bound-state amplitude (Poschl-Teller, sine-Gordon, KdV, NLS soliton profile)."
        ),
    )
)


# ---------------------------------------------------------------------------
# Piecewise (almost-everywhere) + beta-tempered families.
#
# These mirror :mod:`omnibias.torch.activations.piecewise` /
# :mod:`omnibias.torch.activations.tempered`. They are imported *here* -- after
# the base specs above -- so the tempered surrogates can reuse the
# already-registered SIGMOID / SOFTPLUS towers. Import triggers registration.
# ---------------------------------------------------------------------------
from omnibias.jax._activations_piecewise import (  # noqa: E402  (must follow base specs)
    ABS,
    CELU,
    ELU,
    HARDSHRINK,
    HARDSIGMOID,
    HARDSWISH,
    HARDTANH,
    LEAKY_RELU,
    PRELU,
    RELU6,
    SELU,
    SIGN,
    SOFTSHRINK,
    SOFTSIGN,
    STEP,
    THRESHOLD,
    make_celu_spec,
    make_elu_spec,
    make_hardshrink_spec,
    make_hardtanh_spec,
    make_leaky_relu_spec,
    make_softshrink_spec,
    make_threshold_spec,
)
from omnibias.jax._activations_tempered import (  # noqa: E402  (must follow base specs)
    SOFT_RELU,
    SOFT_STEP,
    make_soft_leaky_relu_spec,
    make_soft_relu_spec,
    make_soft_step_spec,
    make_swish_spec,
    tempered_activation,
)

__all__ = [
    "ABS",
    "ARCTAN",
    "CELU",
    "COS",
    "COSH",
    "COT",
    "COTH",
    "ELU",
    "EXP",
    "GAUSSIAN",
    "GELU",
    "HARDSHRINK",
    "HARDSIGMOID",
    "HARDSWISH",
    "HARDTANH",
    "HUBER",
    "IntegralFn",
    "JaxActivationSpec",
    "LEAKY_RELU",
    "LOG1PU2",
    "LOG_COSH",
    "MISH",
    "PRELU",
    "RELU",
    "RELU6",
    "SECH",
    "SELU",
    "SIGMOID",
    "SIGN",
    "SILU",
    "SIN",
    "SINH",
    "SMOOTH_SIGN",
    "SOFTABS",
    "SOFTPLUS",
    "SOFTSHRINK",
    "SOFTSIGN",
    "SOFT_RELU",
    "SOFT_STEP",
    "STEP",
    "TAN",
    "TANH",
    "THRESHOLD",
    "get_activation",
    "is_registered",
    "list_activations",
    "make_celu_spec",
    "make_elu_spec",
    "make_hardshrink_spec",
    "make_hardtanh_spec",
    "make_leaky_relu_spec",
    "make_soft_leaky_relu_spec",
    "make_soft_relu_spec",
    "make_soft_step_spec",
    "make_softshrink_spec",
    "make_swish_spec",
    "make_threshold_spec",
    "register_activation",
    "tempered_activation",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Compactified / line-domain CCF self-similar residual (jax twin).

DeepMind-style line-domain toolkit:

* **lambda-tied compactification** (paper eq. 5)
  :math:`q = (1+y^2)^{-1/(2(1+\lambda))}` mapping the line into ``(0,1]``;
* optional algebraic decay envelope;
* residual factorisation :math:`\mathcal{E}=F\cdot\mathcal{R}`;
* Hilbert modes:

  - ``hardy_exact`` -- closed-form whole-line Hilbert on a Cauchy-Hardy
    profile (preferred for the line);
  - ``truncated_line_resampled_periodic_fft`` -- numerical nonlocal on a
    truncated interval (kept for periodic-path compatibility; not
    recommended for competitive line CCF).

A legacy rational map ``q = y/sqrt(1+y^2)`` remains as
:func:`compactify_y_rational` for callers that need the odd coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import CCFCompactifiedOutput
from omnibias.pinn.jax.hilbert import hilbert_transform

_FORMS = ("transport", "flux")
_HILBERT_MODES = ("hardy_exact", "truncated_line_resampled_periodic_fft")


def alpha_from_lambda(lam: Array | float) -> Array:
    """Self-similar far-field exponent ``alpha = 1/(1+lambda)``."""
    lam = jnp.asarray(lam, dtype=jnp.float64)
    return 1.0 / (1.0 + lam)


def compactify_y_lambda(y: Array, lam: Array | float) -> Array:
    r"""Paper eq. 5 (1-D): ``q = (1+y^2)^{-1/(2(1+lambda))}`` into ``(0, 1]``."""
    y = jnp.asarray(y)
    alpha = alpha_from_lambda(lam)
    return jnp.power(1.0 + y * y, -0.5 * alpha)


def compactify_y(y: Array, lam: Array | float | None = None) -> Array:
    """Compactify ``y``.  If ``lam`` is given, use the lambda-tied map."""
    if lam is None:
        return compactify_y_rational(y)
    return compactify_y_lambda(y, lam)


def compactify_y_rational(y: Array) -> Array:
    r"""Legacy rational map ``q = y / sqrt(1+y^2)`` into ``(-1, 1)``."""
    y = jnp.asarray(y)
    return y / jnp.sqrt(1.0 + y * y)


def decompactify_q(q: Array) -> Array:
    r"""Inverse of the rational map: ``y = q / sqrt(1-q^2)`` for ``|q| < 1``."""
    q = jnp.asarray(q)
    return q / jnp.sqrt(jnp.maximum(1.0 - q * q, 1e-30))


def dq_dy(y: Array) -> Array:
    r"""Jacobian of the rational map: ``dq/dy = (1+y^2)^{-3/2}``."""
    y = jnp.asarray(y)
    return jnp.power(1.0 + y * y, -1.5)


def compactified_grid(
    n: int,
    *,
    q_max: float = 0.999,
    lam: float | None = None,
) -> tuple[Array, Array]:
    """Return ``(q, y)`` samples.

    With ``lam is None``: uniform rational ``q`` in ``(-q_max, q_max)``.
    With ``lam`` set: uniform ``y`` on a symmetric window whose lambda-tied
    ``q`` spans roughly ``[q_floor, 1]``, returned as ``(q_lambda, y)``.
    """
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    if lam is None:
        if not (0.0 < q_max < 1.0):
            raise ValueError(f"q_max must lie in (0, 1), got {q_max}")
        q = jnp.linspace(-q_max, q_max, int(n), dtype=jnp.float64)
        return q, decompactify_q(q)
    # Choose y_max so lambda-tied q(y_max) ~= small floor.
    alpha = float(alpha_from_lambda(lam))
    # q = (1+y^2)^{-alpha/2} => y = sqrt(q^{-2/alpha} - 1)
    q_floor = max(1.0 - q_max, 1e-4)
    y_max = float(np_sqrt_safe(q_floor ** (-2.0 / alpha) - 1.0))
    y = jnp.linspace(-y_max, y_max, int(n), dtype=jnp.float64)
    q = compactify_y_lambda(y, lam)
    return q, y


def np_sqrt_safe(x: float) -> float:
    return float(max(x, 0.0) ** 0.5)


def decay_envelope(y: Array, *, power: float = 1.0) -> tuple[Array, Array]:
    r"""Return ``(E, E')`` for :math:`E=(1+y^2)^{-p/2}`."""
    y = jnp.asarray(y)
    p = float(power)
    one_y2 = 1.0 + y * y
    env = jnp.power(one_y2, -0.5 * p)
    env_y = -p * y * jnp.power(one_y2, -(0.5 * p + 1.0))
    return env, env_y


def apply_envelope(
    y: Array,
    hat: Array,
    hat_y: Array,
    *,
    power: float = 1.0,
) -> tuple[Array, Array]:
    """Lift ``(hat, hat_y)`` through the decay envelope to ``(Theta, Theta')``."""
    env, env_y = decay_envelope(y, power=power)
    theta = env * hat
    theta_y = env_y * hat + env * hat_y
    return theta, theta_y


def residual_weight(y: Array, *, kind: str = "one_plus_abs") -> Array:
    """Positive factorisation weight ``F(y)`` so ``E = F * R``."""
    y = jnp.asarray(y)
    if kind == "one":
        return jnp.ones_like(y)
    if kind == "one_plus_abs":
        return 1.0 + jnp.abs(y)
    if kind == "one_plus_y2":
        return 1.0 + y * y
    raise ValueError(f"unknown residual weight kind {kind!r}")


# --------------------------------------------------------------------------- #
# Cauchy-Hardy closed-form line Hilbert
# --------------------------------------------------------------------------- #


def hardy_even(y: Array, a: Array | float, alpha: Array | float) -> Array:
    """``P_{a,alpha}(y) = r^{-alpha} cos(alpha phi)``."""
    y = jnp.asarray(y)
    a = jnp.asarray(a, dtype=y.dtype)
    alpha = jnp.asarray(alpha, dtype=y.dtype)
    r = jnp.sqrt(a * a + y * y)
    phi = jnp.arctan(y / a)
    return jnp.power(r, -alpha) * jnp.cos(alpha * phi)


def hardy_odd(y: Array, a: Array | float, alpha: Array | float) -> Array:
    """``Q_{a,alpha}(y) = r^{-alpha} sin(alpha phi)``."""
    y = jnp.asarray(y)
    a = jnp.asarray(a, dtype=y.dtype)
    alpha = jnp.asarray(alpha, dtype=y.dtype)
    r = jnp.sqrt(a * a + y * y)
    phi = jnp.arctan(y / a)
    return jnp.power(r, -alpha) * jnp.sin(alpha * phi)


def hardy_even_deriv(y: Array, a: Array | float, alpha: Array | float) -> Array:
    """``P' = -alpha Q_{a, alpha+1}``."""
    alpha = jnp.asarray(alpha)
    return -alpha * hardy_odd(y, a, alpha + 1.0)


def hardy_odd_deriv(y: Array, a: Array | float, alpha: Array | float) -> Array:
    """``Q' = alpha P_{a, alpha+1}``."""
    alpha = jnp.asarray(alpha)
    return alpha * hardy_even(y, a, alpha + 1.0)


def hardy_profile(
    y: Array,
    coeffs: Array,
    scales: Array,
    alpha: Array | float,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(Theta, Theta', HTheta, (HTheta)')`` for a Hardy even sum."""
    y = jnp.asarray(y).reshape(-1)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64).reshape(-1)
    scales = jnp.asarray(scales, dtype=jnp.float64).reshape(-1)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    th = jnp.zeros_like(y, dtype=jnp.float64)
    thp = jnp.zeros_like(y, dtype=jnp.float64)
    hth = jnp.zeros_like(y, dtype=jnp.float64)
    hthp = jnp.zeros_like(y, dtype=jnp.float64)
    for c, a in zip(coeffs, scales, strict=True):
        th = th + c * hardy_even(y, a, alpha)
        thp = thp + c * hardy_even_deriv(y, a, alpha)
        hth = hth + c * hardy_odd(y, a, alpha)
        hthp = hthp + c * hardy_odd_deriv(y, a, alpha)
    return th, thp, hth, hthp


def hilbert_transform_truncated_line(
    y: Array,
    values: Array,
    *,
    n_uniform: int | None = None,
) -> Array:
    r"""Truncated-line Hilbert via uniform resampling + periodic spectral ``H``.

    Honesty: numerical nonlocal on a truncated interval; **not** closed-form
    whole-line Hilbert. Prefer :func:`hardy_profile` for line CCF.
    """
    y = jnp.asarray(y).reshape(-1)
    values = jnp.asarray(values).reshape(-1)
    if y.shape != values.shape:
        raise ValueError(f"y and values shape mismatch: {y.shape} vs {values.shape}")
    n = int(y.shape[0])
    if n < 4:
        raise ValueError(f"need at least 4 samples, got {n}")
    order = jnp.argsort(y)
    y_s = y[order]
    v_s = values[order]
    n_u = int(n if n_uniform is None else n_uniform)
    if n_u < 4:
        raise ValueError(f"n_uniform must be >= 4, got {n_u}")
    y_u = jnp.linspace(y_s[0], y_s[-1], n_u, dtype=y.dtype)
    v_u = jnp.interp(y_u, y_s, v_s)
    h_u = hilbert_transform(v_u)
    h_s = jnp.interp(y_s, y_u, jnp.real(h_u))
    out = jnp.empty_like(values)
    out = out.at[order].set(h_s)
    return out


def ccf_hardy_residual_samples(
    y: Array,
    coeffs: Array,
    scales: Array,
    lam: Array | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    weight_kind: str = "one_plus_abs",
    alpha: Array | float | None = None,
) -> tuple[Array, Array, Array, dict[str, Array]]:
    """CCF residual on a Hardy even profile with exact whole-line Hilbert."""
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    lam_arr = jnp.asarray(lam, dtype=jnp.float64)
    alpha_arr = alpha_from_lambda(lam_arr) if alpha is None else jnp.asarray(alpha)
    th, thp, hth, hthp = hardy_profile(y, coeffs, scales, alpha_arr)
    linear = (1.0 + lam_arr) * y * thp - lam_arr * th
    if form == "transport":
        nonlocal_term = hth * thp
    else:
        nonlocal_term = thp * hth + th * hthp
    equation = linear + velocity_sign * nonlocal_term
    weight = residual_weight(y, kind=weight_kind)
    factored = equation / weight
    fields = {"theta": th, "theta_y": thp, "hilbert": hth, "hilbert_y": hthp}
    return equation, factored, weight, fields


def ccf_compactified_residual_samples(
    y: Array,
    theta: Array,
    theta_y: Array,
    lam: Array | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    weight_kind: str = "one_plus_abs",
    hilbert_n_uniform: int | None = None,
    hilbert_mode: str = "truncated_line_resampled_periodic_fft",
    hilbert_values: Array | None = None,
    hilbert_y_values: Array | None = None,
) -> tuple[Array, Array, Array]:
    r"""Return ``(equation_residual, factored_R, weight_F)`` on line samples.

    For ``hilbert_mode='hardy_exact'``, pass precomputed ``hilbert_values``
    (and ``hilbert_y_values`` for flux form) from :func:`hardy_profile`.
    """
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    if hilbert_mode not in _HILBERT_MODES:
        raise ValueError(f"hilbert_mode must be one of {_HILBERT_MODES}, got {hilbert_mode!r}")
    y = jnp.asarray(y)
    theta = jnp.asarray(theta)
    theta_y = jnp.asarray(theta_y)
    if hilbert_mode == "hardy_exact":
        if hilbert_values is None:
            raise ValueError("hilbert_mode='hardy_exact' requires hilbert_values")
        h_theta = jnp.asarray(hilbert_values)
    else:
        h_theta = hilbert_transform_truncated_line(
            y, theta, n_uniform=hilbert_n_uniform
        )
    if form == "transport":
        nonlocal_term = h_theta * theta_y
    else:
        if hilbert_mode == "hardy_exact":
            if hilbert_y_values is None:
                raise ValueError("flux + hardy_exact requires hilbert_y_values")
            h_theta_y = jnp.asarray(hilbert_y_values)
        else:
            h_theta_y = hilbert_transform_truncated_line(
                y, theta_y, n_uniform=hilbert_n_uniform
            )
        nonlocal_term = theta_y * h_theta + theta * h_theta_y
    linear = (1.0 + lam) * y * theta_y - lam * theta
    equation = linear + velocity_sign * nonlocal_term
    weight = residual_weight(y, kind=weight_kind)
    factored = equation / weight
    return equation, factored, weight


@dataclass
class CordobaCordobaFontelosCompactified:
    r"""Line-domain / compactified CCF residual on a 1-D spatial ``FieldState``."""

    lam: float = 0.6057
    component: str = "theta"
    form: str = "transport"
    velocity_sign: float = 1.0
    weight_kind: str = "one_plus_abs"
    hilbert_n_uniform: int | None = None
    hilbert_mode: str = "truncated_line_resampled_periodic_fft"

    def __call__(self, state: FieldState) -> CCFCompactifiedOutput:
        if state.coordinate_spec.time_axis is not None:
            raise ValueError(
                "CCF compactified residual is steady; the field must have no "
                f"time axis (got time_axis={state.coordinate_spec.time_axis!r})"
            )
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != 1:
            raise ValueError(
                f"CCF compactified residual requires exactly 1 spatial axis, got "
                f"{len(spatial)} ({spatial!r})"
            )
        ax = spatial[0]
        ax_i = state.coordinate_spec.axis_index(ax)
        y = state.coords[:, ax_i]
        if self.hilbert_mode == "hardy_exact":
            raise ValueError(
                "FieldState path does not carry Hardy coeffs; use "
                "ccf_hardy_residual_samples for hilbert_mode='hardy_exact'"
            )
        theta = state.ops.value(state, self.component)
        theta_y = state.ops.derivative(state, self.component, axis=ax, order=1)
        equation, factored, weight = ccf_compactified_residual_samples(
            y,
            theta,
            theta_y,
            self.lam,
            form=self.form,
            velocity_sign=self.velocity_sign,
            weight_kind=self.weight_kind,
            hilbert_n_uniform=self.hilbert_n_uniform,
            hilbert_mode=self.hilbert_mode,
        )
        h_theta = hilbert_transform_truncated_line(
            y, theta, n_uniform=self.hilbert_n_uniform
        )
        q = compactify_y_lambda(y, self.lam)
        return CCFCompactifiedOutput(
            residual=factored,
            equation_residual=equation,
            hilbert=h_theta,
            weight=weight,
            q=q,
            diag={
                "mean_sq_residual": jnp.mean(factored * factored),
                "max_abs_residual": jnp.max(jnp.abs(factored)),
                "max_abs_equation_residual": jnp.max(jnp.abs(equation)),
                "domain": "line_compactified",
                "hilbert_convention": self.hilbert_mode,
                "compactification": "lambda_tied_eq5",
                "alpha": alpha_from_lambda(self.lam),
            },
        )


def cordoba_cordoba_fontelos_compactified(
    state: FieldState,
    *,
    lam: float = 0.6057,
    component: str = "theta",
    form: str = "transport",
    velocity_sign: float = 1.0,
    weight_kind: str = "one_plus_abs",
    hilbert_n_uniform: int | None = None,
    hilbert_mode: str = "truncated_line_resampled_periodic_fft",
) -> CCFCompactifiedOutput:
    """Stateless one-shot wrapper around :class:`CordobaCordobaFontelosCompactified`."""
    return CordobaCordobaFontelosCompactified(
        lam=lam,
        component=component,
        form=form,
        velocity_sign=velocity_sign,
        weight_kind=weight_kind,
        hilbert_n_uniform=hilbert_n_uniform,
        hilbert_mode=hilbert_mode,
    )(state)


__all__ = [
    "CordobaCordobaFontelosCompactified",
    "alpha_from_lambda",
    "apply_envelope",
    "ccf_compactified_residual_samples",
    "ccf_hardy_residual_samples",
    "compactified_grid",
    "compactify_y",
    "compactify_y_lambda",
    "compactify_y_rational",
    "cordoba_cordoba_fontelos_compactified",
    "decay_envelope",
    "decompactify_q",
    "dq_dy",
    "hardy_even",
    "hardy_even_deriv",
    "hardy_odd",
    "hardy_odd_deriv",
    "hardy_profile",
    "hilbert_transform_truncated_line",
    "residual_weight",
]

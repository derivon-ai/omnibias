# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Compactified / line-domain CCF self-similar residual (torch twin).

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

Bit-parity companion of
:mod:`omnibias.pinn.jax.equations.ccf_compactified`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import CCFCompactifiedOutput
from omnibias.pinn.torch.hilbert import hilbert_transform
from torch import Tensor

_FORMS = ("transport", "flux")
_HILBERT_MODES = ("hardy_exact", "truncated_line_resampled_periodic_fft")


def alpha_from_lambda(lam: Tensor | float) -> Tensor:
    """Self-similar far-field exponent ``alpha = 1/(1+lambda)``."""
    lam = torch.as_tensor(lam, dtype=torch.float64)
    return 1.0 / (1.0 + lam)


def compactify_y_lambda(y: Tensor, lam: Tensor | float) -> Tensor:
    r"""Paper eq. 5 (1-D): ``q = (1+y^2)^{-1/(2(1+lambda))}`` into ``(0, 1]``."""
    y = torch.as_tensor(y)
    alpha = alpha_from_lambda(lam)
    return torch.pow(1.0 + y * y, -0.5 * alpha)


def compactify_y(y: Tensor, lam: Tensor | float | None = None) -> Tensor:
    """Compactify ``y``.  If ``lam`` is given, use the lambda-tied map."""
    if lam is None:
        return compactify_y_rational(y)
    return compactify_y_lambda(y, lam)


def compactify_y_rational(y: Tensor) -> Tensor:
    r"""Legacy rational map ``q = y / sqrt(1+y^2)`` into ``(-1, 1)``."""
    y = torch.as_tensor(y)
    return y / torch.sqrt(1.0 + y * y)


def decompactify_q(q: Tensor) -> Tensor:
    r"""Inverse of the rational map: ``y = q / sqrt(1-q^2)`` for ``|q| < 1``."""
    q = torch.as_tensor(q)
    return q / torch.sqrt(torch.clamp(1.0 - q * q, min=1e-30))


def dq_dy(y: Tensor) -> Tensor:
    r"""Jacobian of the rational map: ``dq/dy = (1+y^2)^{-3/2}``."""
    y = torch.as_tensor(y)
    return torch.pow(1.0 + y * y, -1.5)


def compactified_grid(
    n: int,
    *,
    q_max: float = 0.999,
    lam: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(q, y)`` samples.

    With ``lam is None``: uniform rational ``q`` in ``(-q_max, q_max)``.
    With ``lam`` set: uniform ``y`` on a symmetric window whose lambda-tied
    ``q`` spans roughly ``[q_floor, 1]``, returned as ``(q_lambda, y)``.
    """
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    dt = torch.get_default_dtype()
    if lam is None:
        if not (0.0 < q_max < 1.0):
            raise ValueError(f"q_max must lie in (0, 1), got {q_max}")
        q = torch.linspace(-q_max, q_max, int(n), dtype=dt)
        return q, decompactify_q(q)
    # Choose y_max so lambda-tied q(y_max) ~= small floor.
    alpha = float(alpha_from_lambda(lam))
    # q = (1+y^2)^{-alpha/2} => y = sqrt(q^{-2/alpha} - 1)
    q_floor = max(1.0 - q_max, 1e-4)
    y_max = float(np_sqrt_safe(q_floor ** (-2.0 / alpha) - 1.0))
    y = torch.linspace(-y_max, y_max, int(n), dtype=dt)
    q = compactify_y_lambda(y, lam)
    return q, y


def np_sqrt_safe(x: float) -> float:
    return float(max(x, 0.0) ** 0.5)


def decay_envelope(y: Tensor, *, power: float = 1.0) -> tuple[Tensor, Tensor]:
    r"""Return ``(E, E')`` for :math:`E=(1+y^2)^{-p/2}`."""
    y = torch.as_tensor(y)
    p = float(power)
    one_y2 = 1.0 + y * y
    env = torch.pow(one_y2, -0.5 * p)
    env_y = -p * y * torch.pow(one_y2, -(0.5 * p + 1.0))
    return env, env_y


def apply_envelope(
    y: Tensor,
    hat: Tensor,
    hat_y: Tensor,
    *,
    power: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Lift ``(hat, hat_y)`` through the decay envelope to ``(Theta, Theta')``."""
    env, env_y = decay_envelope(y, power=power)
    theta = env * hat
    theta_y = env_y * hat + env * hat_y
    return theta, theta_y


def residual_weight(y: Tensor, *, kind: str = "one_plus_abs") -> Tensor:
    """Positive factorisation weight ``F(y)`` so ``E = F * R``."""
    y = torch.as_tensor(y)
    if kind == "one":
        return torch.ones_like(y)
    if kind == "one_plus_abs":
        return 1.0 + torch.abs(y)
    if kind == "one_plus_y2":
        return 1.0 + y * y
    raise ValueError(f"unknown residual weight kind {kind!r}")


# --------------------------------------------------------------------------- #
# Cauchy-Hardy closed-form line Hilbert
# --------------------------------------------------------------------------- #


def hardy_even(y: Tensor, a: Tensor | float, alpha: Tensor | float) -> Tensor:
    """``P_{a,alpha}(y) = r^{-alpha} cos(alpha phi)``."""
    y = torch.as_tensor(y)
    a = torch.as_tensor(a, dtype=y.dtype)
    alpha = torch.as_tensor(alpha, dtype=y.dtype)
    r = torch.sqrt(a * a + y * y)
    phi = torch.atan(y / a)
    return torch.pow(r, -alpha) * torch.cos(alpha * phi)


def hardy_odd(y: Tensor, a: Tensor | float, alpha: Tensor | float) -> Tensor:
    """``Q_{a,alpha}(y) = r^{-alpha} sin(alpha phi)``."""
    y = torch.as_tensor(y)
    a = torch.as_tensor(a, dtype=y.dtype)
    alpha = torch.as_tensor(alpha, dtype=y.dtype)
    r = torch.sqrt(a * a + y * y)
    phi = torch.atan(y / a)
    return torch.pow(r, -alpha) * torch.sin(alpha * phi)


def hardy_even_deriv(y: Tensor, a: Tensor | float, alpha: Tensor | float) -> Tensor:
    """``P' = -alpha Q_{a, alpha+1}``."""
    alpha = torch.as_tensor(alpha)
    return -alpha * hardy_odd(y, a, alpha + 1.0)


def hardy_odd_deriv(y: Tensor, a: Tensor | float, alpha: Tensor | float) -> Tensor:
    """``Q' = alpha P_{a, alpha+1}``."""
    alpha = torch.as_tensor(alpha)
    return alpha * hardy_even(y, a, alpha + 1.0)


def hardy_profile(
    y: Tensor,
    coeffs: Tensor,
    scales: Tensor,
    alpha: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(Theta, Theta', HTheta, (HTheta)')`` for a Hardy even sum."""
    y = torch.as_tensor(y).reshape(-1)
    coeffs = torch.as_tensor(coeffs, dtype=torch.float64).reshape(-1)
    scales = torch.as_tensor(scales, dtype=torch.float64).reshape(-1)
    alpha = torch.as_tensor(alpha, dtype=torch.float64)
    th = torch.zeros_like(y, dtype=torch.float64)
    thp = torch.zeros_like(y, dtype=torch.float64)
    hth = torch.zeros_like(y, dtype=torch.float64)
    hthp = torch.zeros_like(y, dtype=torch.float64)
    for c, a in zip(coeffs, scales, strict=True):
        th = th + c * hardy_even(y, a, alpha)
        thp = thp + c * hardy_even_deriv(y, a, alpha)
        hth = hth + c * hardy_odd(y, a, alpha)
        hthp = hthp + c * hardy_odd_deriv(y, a, alpha)
    return th, thp, hth, hthp


def hilbert_transform_truncated_line(
    y: Tensor,
    values: Tensor,
    *,
    n_uniform: int | None = None,
) -> Tensor:
    r"""Truncated-line Hilbert via uniform resampling + periodic spectral ``H``.

    Honesty: numerical nonlocal on a truncated interval; **not** closed-form
    whole-line Hilbert. Prefer :func:`hardy_profile` for line CCF.
    """
    y = torch.as_tensor(y).reshape(-1)
    values = torch.as_tensor(values).reshape(-1)
    if y.shape != values.shape:
        raise ValueError(f"y and values shape mismatch: {y.shape} vs {values.shape}")
    n = int(y.shape[0])
    if n < 4:
        raise ValueError(f"need at least 4 samples, got {n}")
    order = torch.argsort(y)
    y_s = y[order]
    v_s = values[order]
    n_u = int(n if n_uniform is None else n_uniform)
    if n_u < 4:
        raise ValueError(f"n_uniform must be >= 4, got {n_u}")
    y_u = torch.linspace(
        float(y_s[0]), float(y_s[-1]), n_u, dtype=y.dtype, device=y.device
    )
    # torch has no jnp.interp; use searchsorted linear interp
    v_u = _interp1d(y_u, y_s, v_s)
    h_u = hilbert_transform(v_u)
    h_s = _interp1d(y_s, y_u, h_u.real if torch.is_complex(h_u) else h_u)
    out = torch.empty_like(values)
    out[order] = h_s
    return out


def _interp1d(x_new: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Linear interpolation of ``(x, y)`` at ``x_new`` (1-D, sorted ``x``)."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    x_new = x_new.reshape(-1)
    n = x.numel()
    idx = torch.searchsorted(x, x_new, right=True).clamp(1, n - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    t = (x_new - x0) / torch.clamp(x1 - x0, min=1e-30)
    return y0 + t * (y1 - y0)


def ccf_hardy_residual_samples(
    y: Tensor,
    coeffs: Tensor,
    scales: Tensor,
    lam: Tensor | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    weight_kind: str = "one_plus_abs",
    alpha: Tensor | float | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """CCF residual on a Hardy even profile with exact whole-line Hilbert."""
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    lam_arr = torch.as_tensor(lam, dtype=torch.float64)
    alpha_arr = alpha_from_lambda(lam_arr) if alpha is None else torch.as_tensor(alpha)
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
    y: Tensor,
    theta: Tensor,
    theta_y: Tensor,
    lam: Tensor | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    weight_kind: str = "one_plus_abs",
    hilbert_n_uniform: int | None = None,
    hilbert_mode: str = "truncated_line_resampled_periodic_fft",
    hilbert_values: Tensor | None = None,
    hilbert_y_values: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return ``(equation_residual, factored_R, weight_F)`` on line samples.

    For ``hilbert_mode='hardy_exact'``, pass precomputed ``hilbert_values``
    (and ``hilbert_y_values`` for flux form) from :func:`hardy_profile`.
    """
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    if hilbert_mode not in _HILBERT_MODES:
        raise ValueError(f"hilbert_mode must be one of {_HILBERT_MODES}, got {hilbert_mode!r}")
    y = torch.as_tensor(y)
    theta = torch.as_tensor(theta)
    theta_y = torch.as_tensor(theta_y)
    if hilbert_mode == "hardy_exact":
        if hilbert_values is None:
            raise ValueError("hilbert_mode='hardy_exact' requires hilbert_values")
        h_theta = torch.as_tensor(hilbert_values)
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
            h_theta_y = torch.as_tensor(hilbert_y_values)
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
                "mean_sq_residual": float((factored.detach() ** 2).mean()),
                "max_abs_residual": float(factored.detach().abs().max()),
                "max_abs_equation_residual": float(equation.detach().abs().max()),
                "domain": "line_compactified",
                "hilbert_convention": self.hilbert_mode,
                "compactification": "lambda_tied_eq5",
                "alpha": float(alpha_from_lambda(self.lam)),
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

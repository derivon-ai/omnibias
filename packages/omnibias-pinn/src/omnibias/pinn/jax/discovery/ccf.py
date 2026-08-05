# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Deterministic CCF self-similar profile discovery / refinement harness (jax).

This is a small, dependency-light, x64, fully-deterministic harness that searches
for a smooth self-similar profile :math:`\Theta(y)` of the 1D Córdoba-Córdoba-
Fontelos model (see
:mod:`omnibias.pinn.jax.equations.cordoba_cordoba_fontelos` for the derivation
and honesty caveats).

Design choices
--------------
* **Periodic tractable target.** The profile is sought on a periodic interval
  ``[-L, L)``. This is a tractable truncation of the unbounded-domain problem and
  makes the spectral Hilbert transform exact. It is deliberately *not* claimed to
  reproduce the published line-domain admissible :math:`\lambda` values; doing so
  needs an explicit compactification of the infinite domain (future work).
* **omnibias exact derivatives.** The profile is a one-hidden-layer ``tanh``
  network on a *periodic* feature embedding ``[cos y, sin y]``; its first (and, for
  diagnostics, second) derivative are evaluated with omnibias's closed-form
  ``fastpath`` activation-derivative tower, not autodiff -- the same primitive
  that makes the whole library bit-stable.
* **Gauges.** A normalization anchor ``Theta(y_norm) = C`` fixes the trivial-zero
  scaling symmetry, and a far-field penalty pushes ``Theta -> 0`` near the period
  boundary so the bump stays localized.
* **Method of manufactured solutions.** ``manufactured_forcing`` lets a known
  closed-form profile define a forcing term so the harness can be validated
  against a *known* answer (a standard, rigorous PDE-solver check).

The harness reports residual diagnostics (max / RMS of the self-similar residual,
analytic d1 residual, spectral tail), the recovered :math:`\lambda`, and the
sampled profile -- everything needed by :mod:`omnibias.pinn.jax.discovery.cap`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.jax.activations import get_activation
from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import ccf_residual_samples
from omnibias.pinn.jax.hilbert import hilbert_transform

_TANH = get_activation("tanh")
if _TANH.fastpath is None:  # pragma: no cover - tanh always ships a fastpath
    raise RuntimeError("tanh activation is missing its closed-form derivative fastpath")
_TANH_FASTPATH: Callable[..., Array] = cast("Callable[..., Array]", _TANH.fastpath)
_PARITIES = ("even", "none")


# --------------------------------------------------------------------------- #
# Config / result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CCFDiscoveryConfig:
    """Configuration for :func:`run_ccf_discovery`.

    Parameters
    ----------
    hidden
        Hidden width of the one-layer tanh profile network.
    n_grid
        Number of uniform periodic collocation points over ``[-half_period,
        half_period)``.
    half_period
        Half the period ``L``; the domain is ``[-L, L)``.
    parity
        ``"even"`` (default, the parity-consistent CCF class -- uses only the
        ``cos`` feature) or ``"none"`` (uses ``[cos, sin]``).
    form
        CCF residual form, ``"transport"`` (default) or ``"flux"``.
    velocity_sign
        Sign on the nonlocal term.
    lam_init
        Initial self-similar scaling parameter.
    train_lam
        If ``True``, ``lambda`` is optimized jointly with the profile.
    norm_value, norm_point, norm_weight
        Normalization gauge: penalize ``(Theta(norm_point) - norm_value)**2``.
    far_field_frac, far_field_weight
        Far-field gauge: penalize ``Theta**2`` mean over ``|y| > frac * L``.
    seed, weight_scale
        Deterministic initialisation controls.
    """

    hidden: int = 24
    n_grid: int = 256
    half_period: float = math.pi
    parity: str = "even"
    form: str = "transport"
    velocity_sign: float = 1.0
    lam_init: float = 0.6057
    train_lam: bool = False
    norm_value: float = 1.0
    norm_point: float = 0.0
    norm_weight: float = 1.0
    far_field_frac: float = 0.6
    far_field_weight: float = 1.0
    seed: int = 0
    weight_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.parity not in _PARITIES:
            raise ValueError(f"parity must be one of {_PARITIES}, got {self.parity!r}")
        if self.form not in ("transport", "flux"):
            raise ValueError(f"form must be 'transport' or 'flux', got {self.form!r}")
        if self.n_grid < 8:
            raise ValueError(f"n_grid must be >= 8, got {self.n_grid}")


@dataclass(frozen=True)
class CCFDiscoveryResult:
    """Result of a discovery run (numpy arrays for portability)."""

    lam: float
    y: np.ndarray
    theta: np.ndarray
    theta_y: np.ndarray
    residual: np.ndarray
    loss_history: np.ndarray
    diagnostics: dict[str, float]
    params: dict[str, np.ndarray]
    config: CCFDiscoveryConfig
    forced: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Profile model (periodic, omnibias closed-form derivatives)
# --------------------------------------------------------------------------- #
def make_grid(cfg: CCFDiscoveryConfig) -> Array:
    """Uniform periodic grid over ``[-half_period, half_period)``."""
    n, L = cfg.n_grid, cfg.half_period
    return -L + 2.0 * L * jnp.arange(n, dtype=jnp.float64) / n


def _periodic_features(y: Array, parity: str) -> tuple[Array, Array, Array]:
    """Return ``(f, f', f'')`` for the periodic embedding, shape ``(N, F)``.

    For ``cos``/``sin`` features, ``f'' = -f`` exactly.
    """
    if parity == "even":
        f = jnp.stack([jnp.cos(y)], axis=-1)
        df = jnp.stack([-jnp.sin(y)], axis=-1)
    else:  # "none"
        f = jnp.stack([jnp.cos(y), jnp.sin(y)], axis=-1)
        df = jnp.stack([-jnp.sin(y), jnp.cos(y)], axis=-1)
    return f, df, -f


def init_params(cfg: CCFDiscoveryConfig) -> dict[str, Array]:
    """Deterministic parameter initialisation (pytree dict)."""
    n_feat = 1 if cfg.parity == "even" else 2
    key = jax.random.PRNGKey(cfg.seed)
    k_w, k_c = jax.random.split(key)
    W = jax.random.normal(k_w, (cfg.hidden, n_feat), dtype=jnp.float64) * cfg.weight_scale
    beta = jnp.zeros((cfg.hidden,), dtype=jnp.float64)
    c = jax.random.normal(k_c, (cfg.hidden,), dtype=jnp.float64) * (
        cfg.weight_scale / math.sqrt(cfg.hidden)
    )
    b = jnp.zeros((), dtype=jnp.float64)
    lam = jnp.asarray(cfg.lam_init, dtype=jnp.float64)
    return {"W": W, "beta": beta, "c": c, "b": b, "lam": lam}


def profile(params: dict[str, Array], y: Array, parity: str) -> tuple[Array, Array]:
    """Return ``(Theta, Theta')`` using omnibias's exact tanh fastpath."""
    f, df, _ = _periodic_features(y, parity)
    z = f @ params["W"].T + params["beta"]
    dz = df @ params["W"].T
    sig = _TANH.forward(z)
    sig_p = _TANH_FASTPATH(z, 1)  # closed-form sigma'(z)
    theta = params["b"] + sig @ params["c"]
    theta_y = (sig_p * dz) @ params["c"]
    return theta, theta_y


def profile_with_hessian(
    params: dict[str, Array], y: Array, parity: str
) -> tuple[Array, Array, Array]:
    """Return ``(Theta, Theta', Theta'')`` via closed-form fastpath orders 1, 2."""
    f, df, ddf = _periodic_features(y, parity)
    z = f @ params["W"].T + params["beta"]
    dz = df @ params["W"].T
    ddz = ddf @ params["W"].T
    sig = _TANH.forward(z)
    sig_p = _TANH_FASTPATH(z, 1)
    sig_pp = _TANH_FASTPATH(z, 2)
    theta = params["b"] + sig @ params["c"]
    theta_y = (sig_p * dz) @ params["c"]
    theta_yy = ((sig_pp * dz * dz) + (sig_p * ddz)) @ params["c"]
    return theta, theta_y, theta_yy


# --------------------------------------------------------------------------- #
# Loss + training
# --------------------------------------------------------------------------- #
def _loss(
    params: dict[str, Array], y: Array, cfg: CCFDiscoveryConfig, forcing: Array
) -> Array:
    theta, theta_y = profile(params, y, cfg.parity)
    res = ccf_residual_samples(
        y, theta, theta_y, params["lam"],
        form=cfg.form, velocity_sign=cfg.velocity_sign,
    )
    res = res - forcing
    d0 = jnp.mean(res * res)
    # normalization gauge
    yn = jnp.asarray([cfg.norm_point], dtype=jnp.float64)
    theta_n = profile(params, yn, cfg.parity)[0][0]
    norm_pen = (theta_n - cfg.norm_value) ** 2
    # far-field gauge
    far_mask = (jnp.abs(y) > cfg.far_field_frac * cfg.half_period).astype(y.dtype)
    denom = jnp.maximum(jnp.sum(far_mask), 1.0)
    far_pen = jnp.sum(theta * theta * far_mask) / denom
    return d0 + cfg.norm_weight * norm_pen + cfg.far_field_weight * far_pen


def run_ccf_discovery(
    cfg: CCFDiscoveryConfig,
    *,
    steps: int = 600,
    lr: float = 5e-3,
    forcing: Array | np.ndarray | None = None,
    init: dict[str, Array] | None = None,
) -> CCFDiscoveryResult:
    """Run deterministic Adam discovery/refinement and return diagnostics.

    Parameters
    ----------
    cfg
        Discovery configuration.
    steps, lr
        Adam step count and learning rate.
    forcing
        Optional forcing array ``g(y)`` (method of manufactured solutions); the
        objective then targets ``residual == g``. ``None`` -> homogeneous CCF.
    init
        Optional warm-start parameters (for multi-stage / continuation refinement).
    """
    y = make_grid(cfg)
    params = init_params(cfg) if init is None else dict(init)
    forced = forcing is not None
    forcing_arr = jnp.zeros_like(y) if forcing is None else jnp.asarray(forcing, dtype=jnp.float64)

    loss_and_grad = jax.value_and_grad(lambda p: _loss(p, y, cfg, forcing_arr))
    b1, b2, eps = 0.9, 0.999, 1e-8
    train_lam = cfg.train_lam

    @jax.jit
    def step(
        params: dict[str, Array],
        m: dict[str, Array],
        v: dict[str, Array],
        t: Array,
    ) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array]:
        loss, grad = loss_and_grad(params)
        if not train_lam:
            grad = {**grad, "lam": jnp.zeros_like(grad["lam"])}
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, grad)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, grad)
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        params = jax.tree_util.tree_map(
            lambda p, mm, vv: p - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps),
            params, m, v,
        )
        return params, m, v, loss

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    losses: list[float] = []
    for t in range(1, int(steps) + 1):
        params, m, v, loss = step(params, m, v, jnp.asarray(float(t)))
        losses.append(float(loss))

    diagnostics, residual, theta, theta_y = _diagnostics(params, y, cfg, forcing_arr)
    return CCFDiscoveryResult(
        lam=float(params["lam"]),
        y=np.asarray(y),
        theta=np.asarray(theta),
        theta_y=np.asarray(theta_y),
        residual=np.asarray(residual),
        loss_history=np.asarray(losses),
        diagnostics=diagnostics,
        params={k: np.asarray(val) for k, val in params.items()},
        config=cfg,
        forced=forced,
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _diagnostics(
    params: dict[str, Array], y: Array, cfg: CCFDiscoveryConfig, forcing: Array
) -> tuple[dict[str, float], Array, Array, Array]:
    theta, theta_y, theta_yy = profile_with_hessian(params, y, cfg.parity)
    residual = (
        ccf_residual_samples(
            y, theta, theta_y, params["lam"],
            form=cfg.form, velocity_sign=cfg.velocity_sign,
        )
        - forcing
    )
    max_abs = float(jnp.max(jnp.abs(residual)))
    rms = float(jnp.sqrt(jnp.mean(residual * residual)))
    # analytic d1 residual (transport form only): exercises closed-form Theta''.
    d1_max = float("nan")
    if cfg.form == "transport" and not bool(jnp.any(forcing != 0.0)):
        h_theta = hilbert_transform(theta)
        h_theta_y = hilbert_transform(theta_y)
        lam = params["lam"]
        res_d1 = (
            (1.0 + lam) * theta_y
            + (1.0 + lam) * y * theta_yy
            - lam * theta_y
            + cfg.velocity_sign * (h_theta_y * theta_y + h_theta * theta_yy)
        )
        d1_max = float(jnp.max(jnp.abs(res_d1)))
    # spectral tail: fraction of |fft(theta)| energy in the top half of modes.
    spectrum = jnp.abs(jnp.fft.fft(theta))
    n = theta.shape[0]
    half = n // 4
    tail = float(jnp.sum(spectrum[half : n - half] ** 2) / (jnp.sum(spectrum**2) + 1e-300))
    far_mask = jnp.abs(y) > cfg.far_field_frac * cfg.half_period
    far_max = float(jnp.max(jnp.abs(jnp.where(far_mask, theta, 0.0))))
    diagnostics = {
        "max_abs_residual": max_abs,
        "rms_residual": rms,
        "max_abs_d1_residual": d1_max,
        "spectral_tail_fraction": tail,
        "far_field_max_abs": far_max,
        "theta_peak": float(jnp.max(jnp.abs(theta))),
        "lam": float(params["lam"]),
    }
    return diagnostics, residual, theta, theta_y


# --------------------------------------------------------------------------- #
# Method of manufactured solutions
# --------------------------------------------------------------------------- #
def manufactured_forcing(
    cfg: CCFDiscoveryConfig,
    theta_star: Callable[[Array], Array],
    lam_star: float,
) -> tuple[Array, Array, Array]:
    """Build the forcing ``g = R[theta_star; lam_star]`` for an MMS check.

    Returns ``(forcing, theta_star_samples, theta_star_prime_samples)`` on the
    grid. ``theta_star`` must be a jax-traceable scalar function of ``y``; its
    derivative is taken with autodiff (the manufactured *reference*, distinct
    from the omnibias closed-form path the harness itself uses).
    """
    y = make_grid(cfg)
    th = theta_star(y)
    th_y = jax.vmap(jax.grad(theta_star))(y)
    forcing = ccf_residual_samples(
        y, th, th_y, lam_star, form=cfg.form, velocity_sign=cfg.velocity_sign
    )
    return forcing, th, th_y


def default_manufactured_profile(half_period: float = math.pi) -> Callable[[Array], Array]:
    """A smooth, even, periodic reference profile for MMS validation."""

    def theta_star(y: Array) -> Array:
        return jnp.exp(-(1.0 - jnp.cos(y))) + 0.25 * jnp.cos(2.0 * y)

    return theta_star


__all__ = [
    "CCFDiscoveryConfig",
    "CCFDiscoveryResult",
    "default_manufactured_profile",
    "init_params",
    "make_grid",
    "manufactured_forcing",
    "profile",
    "profile_with_hessian",
    "run_ccf_discovery",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Compactified neural vorticity PINN for line-domain CCF (torch).

DeepMind / Lucas-style ingredients, omnibias weapons:

* λ-tied compactification ``q = (1+y^2)^{-α/2}``;
* odd lift ``Ω=y·E·hat`` with decay ``E=(1+y²)^{-(α+1)/2}`` (so ``Ω∼|y|^{-α}``)
  and **softplus / exp-style** core;
* OMBU / closed-form ``sigma`` tower for ``Ω_y``;
* **gradient-normalized** Wang residual (train); absolute residual for Rung;
* d0 + d1 + d2 residual stack + adaptive / hybrid collocation;
* **Two arms:**
  - ``earn`` (default): Hardy train Hilbert + CubicGaussNewton (Adam forbidden);
  - ``reproduce``: spectral/PV Hilbert + Martens–Grosse Gauss–Newton (exact JVP),
    Adam warmup allowed, dense residual gated on the neural profile itself.

CPU smoke configs stay small; ``--full`` / submit uses richer budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from omnibias.pinn.torch.equations.ccf_compactified import (
    alpha_from_lambda,
    apply_envelope,
    compactify_y_lambda,
    hardy_odd,
    hilbert_transform_truncated_line,
)
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.architectures.pinn import PINNOMBU, JetMLP
from omnibias.torch.optim import (
    CubicGaussNewton,
    GaussNewton,
    functional_residual_fn,
    martens_grosse_gauss_newton_minimize,
)
from torch import Tensor

TrainHilbert = Literal[
    "truncated_line_spectral",
    "hardy_projection",
    "pv_line",
    "hardy_corrected_pv",
]
OptimizerName = Literal["cubic_gauss_newton", "martens_grosse"]
ArmName = Literal["earn", "reproduce"]


@dataclass(frozen=True)
class CCFVorticityNeuralConfig:
    """Compactified neural vorticity discovery configuration."""

    lam: float = 0.6057
    n_grid: int = 257
    y_max: float = 40.0
    hidden: int = 32
    depth: int = 1  # JetMLP hidden-layer count; 1 keeps single-OMBU path
    activation: str = "tanh"
    n_scales: int = 8
    n_gamma_multiples: int = 4
    gauge_point: float = 0.5
    gauge_value: float = 0.05
    gauge_weight: float = 40.0
    nontrivial_weight: float = 5.0
    omega_peak_floor: float = 0.02  # soft floor on max|Ω|; raise for reproduce
    proj_defect_weight: float = 1.0
    seed: int = 0
    cubic_gn_steps: int = 40
    qr_gn_steps: int = 20
    mg_steps: int = 40
    mg_solver: Literal["qr", "cgls", "dense", "cg"] = "qr"
    optimizer: OptimizerName = "cubic_gauss_newton"
    arm: ArmName = "earn"
    adam_warmup_steps: int = 0  # earn path must keep 0 (Adam forbidden for Rung-1)
    adam_lr: float = 1e-2
    dtype: torch.dtype = torch.float64
    # Grad-norm (DeepMind follow-up α≈2 for CCF)
    use_grad_norm: bool = True
    grad_norm_alpha: float = 2.0
    grad_norm_eps: float = 1e-8
    exp_core: bool = True
    # d1/d2 + adaptive collocation
    d1_weight: float = 0.1
    d2_weight: float = 0.01
    adaptive_power: float = 2.0
    resample_every: int = 10
    n_adaptive: int | None = None  # default = n_grid
    origin_fraction: float = 0.25  # fraction of points forced near origin / gauge
    # Hilbert: earn default matches Rung/CAP (Hardy). Reproduce defaults to
    # hardy_corrected_pv via reproduce_deepmind_config (spectral/PV are diagnostic).
    train_hilbert: TrainHilbert = "hardy_projection"
    hilbert_n_uniform: int | None = None
    dense_n_val: int = 4001
    device: str = "cpu"  # "cuda" when available; T1200 supports float64
    # Random Fourier features of compactified q (depth>=2 JetMLP only).
    n_fourier: int = 0
    fourier_scale: float = 1.0


def reproduce_deepmind_config(**overrides: Any) -> CCFVorticityNeuralConfig:
    """DeepMind-faithful reproduction defaults (neural + corrected Hilbert + MG)."""
    device = str(overrides.pop("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    base = CCFVorticityNeuralConfig(
        arm="reproduce",
        optimizer="martens_grosse",
        # Exact H on Hardy projection + PV on remainder (PV diag autograd-safe).
        train_hilbert="hardy_corrected_pv",
        # Pull neural Ω into Hardy span so the corrected Hilbert → exact.
        proj_defect_weight=float(overrides.pop("proj_defect_weight", 25.0)),
        adam_warmup_steps=int(overrides.pop("adam_warmup_steps", 50)),
        mg_steps=int(overrides.pop("mg_steps", 80)),
        mg_solver=overrides.pop("mg_solver", "qr"),
        cubic_gn_steps=int(overrides.pop("cubic_gn_steps", 0)),
        qr_gn_steps=int(overrides.pop("qr_gn_steps", 20)),
        use_grad_norm=True,
        exp_core=True,
        # Modest dictionary keeps QR/MG Jacobians inside 4GB laptop GPUs.
        n_scales=int(overrides.pop("n_scales", 6)),
        n_gamma_multiples=int(overrides.pop("n_gamma_multiples", 4)),
        nontrivial_weight=float(overrides.pop("nontrivial_weight", 50.0)),
        omega_peak_floor=float(overrides.pop("omega_peak_floor", 0.08)),
        device=device,
    )
    return replace(base, **overrides) if overrides else base


@dataclass
class CCFVorticityNeuralResult:
    """Neural discovery result with projected Hardy dictionary."""

    lam: float
    y: np.ndarray
    omega: np.ndarray
    omega_y: np.ndarray
    residual: np.ndarray
    coeffs: np.ndarray
    scales: np.ndarray
    gammas: np.ndarray
    diagnostics: dict[str, float]
    config: CCFVorticityNeuralConfig
    extra: dict[str, Any] = field(default_factory=dict)


class CompactifiedOmegaOMBU(nn.Module):
    """Compactified ``q ↦ raw`` map with closed-form ``d raw / dq``.

    ``depth=1`` keeps the original single-hidden PINNOMBU path. ``depth>=2``
    uses :class:`~omnibias.torch.architectures.pinn.JetMLP` so DeepMind-style
    deeper envelopes still differentiate via the omnibias jet tower (no
    stacked ``autograd.grad``).

    Optional random Fourier features of ``q`` enrich the near-origin map when
    ``n_fourier>0`` (JetMLP path only): input becomes
    ``[q, sin(Bq), cos(Bq)]`` with chain-ruled ``d raw / dq``.
    """

    def __init__(
        self,
        *,
        hidden: int = 32,
        depth: int = 1,
        activation: str = "tanh",
        n_fourier: int = 0,
        fourier_scale: float = 1.0,
        fourier_seed: int = 0,
    ) -> None:
        super().__init__()
        self.depth = max(1, int(depth))
        self.hidden = int(hidden)
        self.activation = str(activation)
        self.n_fourier = max(0, int(n_fourier))
        self.fourier_scale = float(fourier_scale)
        if self.n_fourier > 0 and self.depth < 2:
            raise ValueError("n_fourier>0 requires depth>=2 (JetMLP feature path)")
        in_dim = 1 + 2 * self.n_fourier
        if self.n_fourier > 0:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(fourier_seed))
            b = torch.randn(self.n_fourier, generator=g, dtype=torch.float64)
            b = b * self.fourier_scale
            self.register_buffer("fourier_B", b)
        else:
            self.register_buffer("fourier_B", torch.empty(0, dtype=torch.float64))
        if self.depth == 1:
            self._ombu = _SingleHiddenOMBU(hidden=self.hidden, activation=self.activation)
            self.mlp: JetMLP | None = None
        else:
            self._ombu = None
            self.mlp = JetMLP(
                in_dim,
                self.hidden,
                out_dim=1,
                depth=self.depth,
                base=self.activation,
            ).double()

    def _feature_map(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(feats, dfeats_dq)`` with shapes ``(N, in_dim)``."""
        q1 = q.reshape(-1, 1).to(dtype=torch.float64)
        if self.n_fourier <= 0:
            ones = torch.ones_like(q1)
            return q1, ones
        b = self.fourier_B.to(device=q1.device, dtype=q1.dtype).reshape(1, -1)
        arg = q1 * b
        s = torch.sin(arg)
        c = torch.cos(arg)
        feats = torch.cat([q1, s, c], dim=-1)
        dfeats = torch.cat([torch.ones_like(q1), b * c, -b * s], dim=-1)
        return feats, dfeats

    def forward_value_and_dq(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(raw, d raw / dq)`` via the closed-form tower."""
        if self._ombu is not None:
            return self._ombu.forward_value_and_dq(q)
        assert self.mlp is not None
        feats, dfeats = self._feature_map(q)
        raw = self.mlp.value(feats).reshape(-1)
        # gradient shape (B, in_dim, out_dim=1)
        du_dfeat = self.mlp.gradient(feats).reshape(feats.shape[0], feats.shape[1])
        du = (du_dfeat * dfeats).sum(dim=-1)
        return raw, du


class _SingleHiddenOMBU(PINNOMBU):
    """Original single-hidden OMBU map ``q ↦ raw``."""

    def __init__(self, *, hidden: int = 32, activation: str = "tanh") -> None:
        super().__init__()
        self.spec = get_activation(activation)
        self.W = nn.Linear(1, int(hidden), bias=True, dtype=torch.float64)
        self.c = nn.Linear(int(hidden), 1, bias=True, dtype=torch.float64)
        self._check_fastpath(2)

    def forward_value_and_dq(self, q: Tensor) -> tuple[Tensor, Tensor]:
        q_in = q.reshape(-1, 1)
        u, z = self.base_forward(q_in)
        du_dq = self.first_derivative(z, axis=0)
        return u.reshape(-1), du_dq.reshape(-1)


def hardy_dictionary(
    *,
    lam: float,
    n_scales: int,
    n_gamma_multiples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse scales × ``γ = k α`` ladder (Ω-primary)."""
    alpha0 = float(1.0 / (1.0 + float(lam)))
    if n_scales == 1:
        scales_base = np.array([1.3], dtype=float)
    else:
        scales_base = np.linspace(0.6, 3.5, int(n_scales))
    gammas = alpha0 * np.arange(1, int(n_gamma_multiples) + 1, dtype=float)
    scales = np.repeat(scales_base, int(n_gamma_multiples))
    gs = np.tile(gammas, int(n_scales))
    return scales, gs


def _build_phi(y: Tensor, scales: Tensor, gammas: Tensor) -> Tensor:
    cols: list[Tensor] = []
    for a, g in zip(scales, gammas, strict=True):
        cols.append(hardy_odd(y, a, g))
    return torch.stack(cols, dim=1)


def project_omega_hardy_torch(
    y: Tensor,
    omega: Tensor,
    *,
    scales: Tensor,
    gammas: Tensor,
    ridge: float = 1e-12,
    coeff_cap: float = 1e3,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Differentiable LS projection; exact ``H[Q]=-P`` and ``U=∫H`` on coeffs."""
    phi = _build_phi(y, scales, gammas)
    # Prefer ridge solve: overlapping Hardy atoms make plain gels fragile (DGELS).
    ata = phi.T @ phi
    n = int(ata.shape[0])
    eye = torch.eye(n, dtype=ata.dtype, device=ata.device)
    scale = torch.trace(ata) / max(n, 1)
    # Tiny ridge: enough for multi-atom conditioning, small enough to recover
    # exact single-atom coefficients to ~1e-12.
    ridge_eff = max(float(ridge), 1e-14) * scale
    try:
        coeffs = torch.linalg.solve(ata + ridge_eff * eye, phi.T @ omega.reshape(-1))
    except RuntimeError:
        coeffs = torch.zeros(n, dtype=phi.dtype, device=phi.device)
    if float(torch.max(torch.abs(coeffs)).detach()) > float(coeff_cap):
        coeffs = torch.clamp(coeffs, min=-float(coeff_cap), max=float(coeff_cap))
    om_hat = (phi @ coeffs.reshape(-1, 1)).reshape(-1)
    defect = torch.max(torch.abs(om_hat - omega))
    # Vectorized exact Hardy fields (matches jax hardy_omega_profile).
    yy = y.reshape(-1, 1)
    aa = scales.reshape(1, -1)
    gg = gammas.reshape(1, -1)
    cc = coeffs.reshape(1, -1)
    rr = torch.sqrt(aa * aa + yy * yy)
    phi_ang = torch.atan2(yy, aa)
    h = torch.sum(cc * (-(rr ** (-gg)) * torch.cos(gg * phi_ang)), dim=1)
    omy = torch.sum(
        cc * gg * (rr ** (-(gg + 1.0))) * torch.cos((gg + 1.0) * phi_ang), dim=1
    )
    near1 = torch.abs(gg - 1.0) < 1e-12
    u_gen = -(rr ** (-(gg - 1.0))) * torch.sin((gg - 1.0) * phi_ang) / (gg - 1.0)
    u_g1 = -torch.atan(yy / aa)
    u = torch.sum(cc * torch.where(near1, u_g1, u_gen), dim=1)
    return coeffs, defect, {"omega_proj": om_hat, "H": h, "U": u, "omega_y_proj": omy}


def project_omega_hardy(
    y: Tensor | np.ndarray,
    omega: Tensor | np.ndarray,
    *,
    scales: np.ndarray,
    gammas: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """NumPy convenience wrapper around :func:`project_omega_hardy_torch`."""
    y_t = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    om_t = torch.as_tensor(np.asarray(omega, dtype=float).copy(), dtype=torch.float64).reshape(
        -1
    )
    sc = torch.as_tensor(scales, dtype=torch.float64).reshape(-1)
    gs = torch.as_tensor(gammas, dtype=torch.float64).reshape(-1)
    with torch.no_grad():
        coeffs, defect, fields = project_omega_hardy_torch(
            y_t, om_t, scales=sc, gammas=gs
        )
    out_fields = {k: v.detach().cpu().numpy() for k, v in fields.items()}
    return coeffs.detach().cpu().numpy(), float(defect.item()), out_fields


def _dq_dy(y: Tensor, lam: float) -> Tensor:
    alpha = alpha_from_lambda(lam)
    return (-alpha * y) * torch.pow(1.0 + y * y, -0.5 * alpha - 1.0)


def _integrate_from_zero(y: Tensor, f: Tensor) -> Tensor:
    """Cumulative trapezoid of ``f`` through 0 (odd-friendly for HΩ → U)."""
    order = torch.argsort(y)
    y_s = y[order]
    f_s = f[order]
    n = int(y_s.numel())
    i0 = int(torch.argmin(torch.abs(y_s)).item())
    dy = y_s[1:] - y_s[:-1]
    trap = 0.5 * (f_s[1:] + f_s[:-1]) * dy
    # Prefix sums from left; then shift so u(y≈0)=0
    pref = torch.zeros(n, dtype=f.dtype, device=f.device)
    if n > 1:
        pref[1:] = torch.cumsum(trap, dim=0)
    u_s = pref - pref[i0]
    out = torch.empty_like(f)
    out[order] = u_s
    return out


def spectral_hu_from_omega(
    y: Tensor,
    omega: Tensor,
    *,
    n_uniform: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Truncated-line spectral ``HΩ`` and ``U=∫_0^y HΩ`` (train Hilbert path)."""
    h = hilbert_transform_truncated_line(y, omega, n_uniform=n_uniform)
    u = _integrate_from_zero(y, h)
    return h, u


def hilbert_pv_line(y: Tensor, values: Tensor) -> Tensor:
    """Differentiable trapezoid PV Hilbert on a truncated sorted line.

    Implements
    ``Hf(x)=(1/π)[∫_a^b (f(t)-f(x))/(x-t) dt + f(x) log|(x-a)/(b-x)|]``,
    the standard singularity-subtraction form of the finite-interval PV
    Hilbert transform. Matches Hardy ``H[Q]=-P`` on ``[-Y,Y]`` far better
    than periodized FFT (which errs at ``O(10^{-1})`` for leading atoms).
    """
    y = torch.as_tensor(y).reshape(-1)
    values = torch.as_tensor(values).reshape(-1)
    if y.shape != values.shape:
        raise ValueError(f"y and values shape mismatch: {y.shape} vs {values.shape}")
    order = torch.argsort(y)
    y_s = y[order]
    v_s = values[order]
    a = y_s[0]
    b = y_s[-1]
    # dy[i,j] = x_i - t_j; integrand (f(t)-f(x))/(x-t)
    dy = y_s.unsqueeze(1) - y_s.unsqueeze(0)
    dv_tx = v_s.unsqueeze(0) - v_s.unsqueeze(1)  # [i,j] = f(t_j)-f(x_i)
    # Avoid 0/0 on the diagonal: both branches of where() are evaluated for
    # autograd, so a raw dv/dy NaN poisons Martens–Grosse jacrev.
    diag = dy.abs() < 1e-30
    dy_safe = torch.where(diag, torch.ones_like(dy), dy)
    kern = dv_tx / dy_safe
    kern = torch.where(diag, torch.zeros_like(kern), kern)
    n = int(y_s.numel())
    dt = torch.zeros(n, dtype=y_s.dtype, device=y_s.device)
    if n >= 3:
        dt[1:-1] = 0.5 * (y_s[2:] - y_s[:-2])
    if n >= 2:
        dt[0] = y_s[1] - y_s[0]
        dt[-1] = y_s[-1] - y_s[-2]
    integ = (kern * dt.unsqueeze(0)).sum(dim=1)
    # endpoint log correction; clamp away from endpoints
    xa = torch.clamp(y_s - a, min=1e-12)
    bx = torch.clamp(b - y_s, min=1e-12)
    log_term = v_s * torch.log(xa / bx)
    h_s = (integ + log_term) / math.pi
    out = torch.empty_like(values)
    out[order] = h_s
    return out


def pv_hu_from_omega(y: Tensor, omega: Tensor) -> tuple[Tensor, Tensor]:
    """PV-line ``HΩ`` and ``U=∫_0^y HΩ``."""
    h = hilbert_pv_line(y, omega)
    u = _integrate_from_zero(y, h)
    return h, u


def hardy_corrected_hu_from_omega(
    y: Tensor,
    omega: Tensor,
    *,
    scales: Tensor,
    gammas: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact Hardy ``H`` on the L2 projection + PV Hilbert on the remainder.

    Periodized FFT / truncated PV alone err at ``O(10^{-1})``–``O(10^{-2})`` vs
    ``H[Q]=-P``. Splitting ``Ω = Ω_H + r`` with exact ``H[Ω_H]`` and PV on the
    faster-decaying remainder ``r`` recovers near-machine ``H`` whenever the
    Hardy defect is small — the regime needed for a 1e-13 Wang residual.
    """
    coeffs, defect, fields = project_omega_hardy_torch(
        y, omega, scales=scales, gammas=gammas
    )
    om_h = fields["omega_proj"]
    h_exact = fields["H"]
    u_exact = fields["U"]
    rem = omega - om_h
    rem = torch.nan_to_num(rem, nan=0.0, posinf=0.0, neginf=0.0)
    h_rem = hilbert_pv_line(y, rem)
    u_rem = _integrate_from_zero(y, h_rem)
    h = torch.nan_to_num(h_exact + h_rem, nan=0.0, posinf=0.0, neginf=0.0)
    u = torch.nan_to_num(u_exact + u_rem, nan=0.0, posinf=0.0, neginf=0.0)
    return h, u, coeffs, defect


def omega_from_net(
    net: CompactifiedOmegaOMBU,
    y: Tensor,
    *,
    lam: float,
    exp_core: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Odd-envelope lift with optional exp core.

    Returns ``(omega, omega_y, q, nn_core)`` where ``nn_core`` is the pre-envelope
    network field used for gradient normalization (DeepMind ``NN_Ω``).

    Far-field honesty
    -----------------
    Odd lift is ``Ω = y · E · hat`` with decay envelope
    ``E=(1+y²)^{-(α+1)/2}``. Combined with the explicit ``y``, this yields the
    Wang/DeepMind leading decay ``Ω ∼ |y|^{-α}`` (``α=1/(1+λ)``). Using power
    ``α`` instead of ``α+1`` would grow like ``|y|^{1-α}`` and cannot cancel
    the linear far-field operator.
    """
    alpha = float(alpha_from_lambda(lam))
    q = compactify_y_lambda(y, lam)
    raw, draw_dq = net.forward_value_and_dq(q)
    if exp_core:
        # Softplus-exp hybrid: positive multiscale core without raw exp blow-up.
        # hat = softplus(raw) + eps; dhat/dq = sigmoid(raw) * draw_dq
        hat = torch.nn.functional.softplus(raw) + 1e-6
        dhat_dq = torch.sigmoid(raw) * draw_dq
        nn_core = torch.log(hat)  # for grad-norm ~ log-amplitude
    else:
        hat = raw
        dhat_dq = draw_dq
        nn_core = raw
    # power=α+1 so y·E ∼ |y|^{-α} (not |y|^{1-α}).
    psi, psi_y = apply_envelope(
        y, hat, dhat_dq * _dq_dy(y, lam), power=alpha + 1.0
    )
    omega = y * psi
    omega_y = psi + y * psi_y
    return omega, omega_y, q, nn_core


def wang_residual(
    y: Tensor,
    omega: Tensor,
    omega_y: Tensor,
    u: Tensor,
    uy: Tensor,
    *,
    lam: float,
) -> Tensor:
    return omega + ((1.0 + lam) * y - u) * omega_y - omega * uy


def gradient_normalize(
    r: Tensor,
    nn_core: Tensor,
    *,
    alpha: float,
    eps: float,
) -> Tensor:
    """DeepMind CCF: ``R / (eps + exp(alpha * NN_Ω))``."""
    return r / (float(eps) + torch.exp(float(alpha) * nn_core))


def _grid_derivatives(r: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Central-difference d1/d2 on a sorted grid (then unsort)."""
    order = torch.argsort(y)
    y_s = y[order]
    r_s = r[order]
    # spacing must be a tuple-of-tensors for nonuniform grids
    d1_s = torch.gradient(r_s, spacing=(y_s,))[0]
    d2_s = torch.gradient(d1_s, spacing=(y_s,))[0]
    d1 = torch.empty_like(r)
    d2 = torch.empty_like(r)
    d1[order] = d1_s
    d2[order] = d2_s
    return d1, d2


def vorticity_fields(
    y: Tensor,
    omega: Tensor,
    omega_y: Tensor,
    *,
    lam: float,
    scales: Tensor,
    gammas: Tensor,
    train_hilbert: TrainHilbert,
    hilbert_n_uniform: int | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(r_abs, defect, coeffs, uy)`` for the chosen Hilbert mode."""
    if train_hilbert == "hardy_projection":
        coeffs, defect, fields = project_omega_hardy_torch(
            y, omega, scales=scales, gammas=gammas
        )
        # Consistent projected fields (exact H/U on the Hardy image).
        omega_r = fields["omega_proj"]
        omega_y_r = fields["omega_y_proj"]
        u, uy = fields["U"], fields["H"]
        r = wang_residual(y, omega_r, omega_y_r, u, uy, lam=lam)
        return r, defect, coeffs, uy
    if train_hilbert == "hardy_corrected_pv":
        h, u, coeffs, defect = hardy_corrected_hu_from_omega(
            y, omega, scales=scales, gammas=gammas
        )
        uy = h
        r = wang_residual(y, omega, omega_y, u, uy, lam=lam)
        return r, defect, coeffs, uy
    if train_hilbert == "pv_line":
        uy, u = pv_hu_from_omega(y, omega)
        coeffs = torch.zeros(scales.shape[0], dtype=y.dtype, device=y.device)
        defect = torch.zeros((), dtype=y.dtype, device=y.device)
    else:
        uy, u = spectral_hu_from_omega(y, omega, n_uniform=hilbert_n_uniform)
        coeffs = torch.zeros(scales.shape[0], dtype=y.dtype, device=y.device)
        defect = torch.zeros((), dtype=y.dtype, device=y.device)
    r = wang_residual(y, omega, omega_y, u, uy, lam=lam)
    return r, defect, coeffs, uy


def residual_vector(
    net: CompactifiedOmegaOMBU,
    y: Tensor,
    *,
    cfg: CCFVorticityNeuralConfig,
    scales: Tensor,
    gammas: Tensor,
) -> Tensor:
    """Stacked residual for CubicGaussNewton."""
    omega, omega_y, _, nn_core = omega_from_net(
        net, y, lam=cfg.lam, exp_core=cfg.exp_core
    )
    r_abs, defect, _, _ = vorticity_fields(
        y,
        omega,
        omega_y,
        lam=cfg.lam,
        scales=scales,
        gammas=gammas,
        train_hilbert=cfg.train_hilbert,
        hilbert_n_uniform=cfg.hilbert_n_uniform,
    )
    r_train = (
        gradient_normalize(
            r_abs,
            nn_core,
            alpha=cfg.grad_norm_alpha,
            eps=cfg.grad_norm_eps,
        )
        if cfg.use_grad_norm
        else r_abs
    )
    parts: list[Tensor] = [r_train]
    if cfg.d1_weight > 0.0 or cfg.d2_weight > 0.0:
        d1, d2 = _grid_derivatives(r_train, y)
        if cfg.d1_weight > 0.0:
            parts.append(math.sqrt(cfg.d1_weight) * d1)
        if cfg.d2_weight > 0.0:
            parts.append(math.sqrt(cfg.d2_weight) * d2)
    yg = torch.tensor([cfg.gauge_point], dtype=y.dtype, device=y.device)
    om_g, _, _, _ = omega_from_net(net, yg, lam=cfg.lam, exp_core=cfg.exp_core)
    gauge = (om_g - cfg.gauge_value) * math.sqrt(cfg.gauge_weight)
    max_abs = torch.max(torch.abs(omega))
    anti = torch.relu(
        torch.as_tensor(float(cfg.omega_peak_floor), dtype=y.dtype, device=y.device)
        - max_abs
    )
    anti = anti * math.sqrt(cfg.nontrivial_weight)
    parts.extend([gauge.reshape(-1), anti.reshape(1)])
    if cfg.proj_defect_weight > 0.0:
        parts.append(defect.reshape(1) * math.sqrt(cfg.proj_defect_weight))
    return torch.cat(parts)


def _hybrid_sample_y(
    *,
    y_max: float,
    n: int,
    origin_fraction: float,
    weights: np.ndarray | None,
    y_pool: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix origin/gauge neighborhood with residual-weighted adaptive samples."""
    n_origin = max(1, int(round(float(origin_fraction) * n)))
    n_adapt = max(1, n - n_origin)
    # Dense near origin + gauge
    y_ori = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, size=n_origin // 2),
            rng.uniform(0.2, 1.0, size=n_origin - n_origin // 2),
        ]
    )
    if weights is None or y_pool.size < 2:
        y_ad = rng.uniform(-y_max, y_max, size=n_adapt)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        if float(w.sum()) <= 0.0 or not np.isfinite(w).all():
            w = np.ones_like(w)
        w = w / w.sum()
        idx = rng.choice(y_pool.size, size=n_adapt, replace=True, p=w)
        y_ad = y_pool[idx] + rng.normal(0.0, 0.02 * y_max / max(n, 1), size=n_adapt)
        y_ad = np.clip(y_ad, -y_max, y_max)
    y = np.concatenate([y_ori, y_ad])
    # Ensure unique-ish sorted grid for FD
    y = np.unique(np.round(y, decimals=10))
    if y.size < n:
        extra = rng.uniform(-y_max, y_max, size=n - y.size)
        y = np.unique(np.concatenate([y, extra]))
    if y.size > n:
        y = np.sort(y)[:: max(1, y.size // n)][:n]
    return np.sort(y.astype(float))


class _ResidualModule(nn.Module):
    """Module wrapper so QR GaussNewton can use ``functional_call`` / jacrev."""

    def __init__(
        self,
        net: CompactifiedOmegaOMBU,
        *,
        y: Tensor,
        scales: Tensor,
        gammas: Tensor,
        cfg: CCFVorticityNeuralConfig,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("y", y)
        self.register_buffer("scales", scales)
        self.register_buffer("gammas", gammas)
        self.cfg = cfg

    def set_y(self, y: Tensor) -> None:
        self.y = y.to(dtype=self.y.dtype, device=self.y.device)

    def forward(self) -> Tensor:
        return residual_vector(
            self.net,
            self.y,
            cfg=self.cfg,
            scales=self.scales,
            gammas=self.gammas,
        )


def _anti_ghost_residual(
    dense_max_abs: float,
    *,
    omega_gauge_sample: float,
    omega_max_abs: float,
    gauge_value: float = 0.05,
    gauge_tol: float = 0.01,
    nontrivial_floor: float = 0.02,
) -> float:
    """Floor residual to 1.0 when gauge / nontriviality fails (anti-ghost)."""
    gauge_ok = abs(float(omega_gauge_sample) - float(gauge_value)) <= float(gauge_tol)
    nontrivial = float(omega_max_abs) >= float(nontrivial_floor)
    if not (gauge_ok and nontrivial):
        return max(float(dense_max_abs), 1.0)
    return float(dense_max_abs)


@torch.no_grad()
def dense_neural_vorticity_residual(
    net: CompactifiedOmegaOMBU,
    *,
    lam: float,
    train_hilbert: TrainHilbert,
    scales: Tensor,
    gammas: Tensor,
    y_max: float = 40.0,
    n_val: int = 4001,
    exp_core: bool = True,
    hilbert_n_uniform: int | None = None,
    gauge_point: float = 0.5,
    gauge_value: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """Dense Wang residual on the **neural** profile with matched train Hilbert.

    This is the DeepMind reproduction metric. It does **not** project onto Hardy
    atoms before scoring (Hardy dense gates remain separate for CAP / Rung-1).
    """
    y = torch.linspace(
        -float(y_max), float(y_max), int(n_val), dtype=dtype, device=next(net.parameters()).device
    )
    omega, omega_y, _, _ = omega_from_net(net, y, lam=lam, exp_core=exp_core)
    # Anti-ghost must see the *raw* profile. Hard-rescaling to gauge before the
    # check would make gauge failure unreachable (always pass after rescale).
    y_np0 = y.detach().cpu().numpy()
    om_np0 = omega.detach().cpu().numpy()
    g_raw = float(np.interp(gauge_point, y_np0, om_np0))
    omega_max_raw = float(np.max(np.abs(om_np0)))
    if abs(g_raw) > 1e-14:
        scale = float(gauge_value) / g_raw
        omega = omega * scale
        omega_y = omega_y * scale
    r, defect, _, _ = vorticity_fields(
        y,
        omega,
        omega_y,
        lam=lam,
        scales=scales,
        gammas=gammas,
        train_hilbert=train_hilbert,
        hilbert_n_uniform=hilbert_n_uniform,
    )
    r_np = r.detach().cpu().numpy()
    om_np = omega.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    om_g = float(np.interp(gauge_point, y_np, om_np))
    dense_max = float(np.max(np.abs(r_np)))
    omega_max = float(np.max(np.abs(om_np)))
    defect_f = float(defect.item())
    # On Hardy-projection Hilbert, the residual is that of the projected profile;
    # the gate must also see the projection defect so we cannot claim 1e-13 while
    # the free neural Ω sits far from the Hardy image.
    score = (
        max(dense_max, defect_f)
        if train_hilbert in ("hardy_projection", "hardy_corrected_pv")
        else dense_max
    )
    for_gate = _anti_ghost_residual(
        score,
        omega_gauge_sample=g_raw,
        omega_max_abs=omega_max_raw,
        gauge_value=gauge_value,
    )
    return {
        "reproduction_dense_max_abs": dense_max,
        "reproduction_dense_rms": float(np.sqrt(np.mean(r_np * r_np))),
        "reproduction_dense_max_abs_for_gate": for_gate,
        "omega_gauge_sample": om_g,
        "omega_gauge_sample_raw": g_raw,
        "omega_max_abs": omega_max,
        "omega_max_abs_raw": omega_max_raw,
        "projection_defect_report": defect_f,
        "n_val": float(n_val),
        "y_max": float(y_max),
        "train_hilbert": 0.0,  # placeholder; string in caller extras
    }


def _resample_collocation(
    net: CompactifiedOmegaOMBU,
    residual_mod: _ResidualModule,
    *,
    cfg: CCFVorticityNeuralConfig,
    scales: Tensor,
    gammas: Tensor,
    n_pts: int,
    rng: np.random.Generator,
) -> None:
    with torch.no_grad():
        omega, omega_y, _, nn_core = omega_from_net(
            net, residual_mod.y, lam=cfg.lam, exp_core=cfg.exp_core
        )
        r_abs, _, _, _ = vorticity_fields(
            residual_mod.y,
            omega,
            omega_y,
            lam=cfg.lam,
            scales=scales,
            gammas=gammas,
            train_hilbert=cfg.train_hilbert,
            hilbert_n_uniform=cfg.hilbert_n_uniform,
        )
        r_w = (
            gradient_normalize(
                r_abs,
                nn_core,
                alpha=cfg.grad_norm_alpha,
                eps=cfg.grad_norm_eps,
            )
            if cfg.use_grad_norm
            else r_abs
        )
        w = torch.abs(r_w).detach().cpu().numpy() ** float(cfg.adaptive_power)
        y_new = _hybrid_sample_y(
            y_max=float(cfg.y_max),
            n=n_pts,
            origin_fraction=cfg.origin_fraction,
            weights=w,
            y_pool=residual_mod.y.detach().cpu().numpy(),
            rng=rng,
        )
        residual_mod.set_y(
            torch.as_tensor(y_new, dtype=cfg.dtype, device=residual_mod.y.device)
        )


def _load_flat_params(module: nn.Module, params: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            p.copy_(params[offset : offset + n].reshape(p.shape))
            offset += n


def run_ccf_vorticity_neural_discovery(
    cfg: CCFVorticityNeuralConfig | None = None,
    *,
    warm_state_dict: dict[str, Tensor] | None = None,
) -> CCFVorticityNeuralResult:
    """Train compactified Ω-PINN (CubicGN earn path or Martens–Grosse reproduce).

    Parameters
    ----------
    warm_state_dict
        Optional prior ``net.state_dict()`` (same architecture). Continues MG from
        a previous escalate round instead of cold-start.
    """
    cfg = cfg or CCFVorticityNeuralConfig()
    if cfg.arm == "earn" and cfg.adam_warmup_steps > 0:
        raise ValueError(
            "earn arm forbids Adam warmup (adam_warmup_steps must be 0); "
            "use arm='reproduce' for DeepMind-style warm-start"
        )
    torch.manual_seed(int(cfg.seed))
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    device = torch.device(cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(int(cfg.seed))
    n_pts = int(cfg.n_adaptive if cfg.n_adaptive is not None else cfg.n_grid)
    y_np = _hybrid_sample_y(
        y_max=float(cfg.y_max),
        n=n_pts,
        origin_fraction=cfg.origin_fraction,
        weights=None,
        y_pool=np.linspace(-cfg.y_max, cfg.y_max, max(n_pts, 33)),
        rng=rng,
    )
    y = torch.as_tensor(y_np, dtype=cfg.dtype, device=device)
    scales_np, gammas_np = hardy_dictionary(
        lam=cfg.lam,
        n_scales=cfg.n_scales,
        n_gamma_multiples=cfg.n_gamma_multiples,
    )
    scales = torch.as_tensor(scales_np, dtype=cfg.dtype, device=device)
    gammas = torch.as_tensor(gammas_np, dtype=cfg.dtype, device=device)
    def _new_net() -> CompactifiedOmegaOMBU:
        return CompactifiedOmegaOMBU(
            hidden=cfg.hidden,
            depth=cfg.depth,
            activation=cfg.activation,
            n_fourier=int(cfg.n_fourier),
            fourier_scale=float(cfg.fourier_scale),
            fourier_seed=int(cfg.seed),
        ).to(device=device, dtype=cfg.dtype)

    def _cold_bias_fill(module: CompactifiedOmegaOMBU) -> None:
        with torch.no_grad():
            if module._ombu is not None:
                module._ombu.c.bias.fill_(0.0)
            elif module.mlp is not None:
                last = module.mlp.linears[-1]
                assert isinstance(last, nn.Linear)
                if last.bias is not None:
                    last.bias.fill_(0.0)

    net = _new_net()
    warm_ok = False
    if warm_state_dict is not None:
        try:
            # strict=True: never keep a hybrid net after a partial load.
            net.load_state_dict(warm_state_dict, strict=True)
            warm_ok = True
        except RuntimeError:
            # Architecture / key / size mismatch — full cold rebuild.
            net = _new_net()
            warm_ok = False
    if not warm_ok:
        warm_state_dict = None
        _cold_bias_fill(net)
    residual_mod = _ResidualModule(net, y=y, scales=scales, gammas=gammas, cfg=cfg)

    # Skip Adam when warm-starting — continue second-order from the prior basin.
    adam_steps = 0 if warm_state_dict is not None else int(cfg.adam_warmup_steps)
    if adam_steps > 0:
        opt_adam = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
        for _ in range(adam_steps):
            opt_adam.zero_grad(set_to_none=True)
            rv = residual_mod()
            loss = 0.5 * torch.mean(rv * rv)
            loss.backward()
            opt_adam.step()

    history: list[float] = []
    best_r = float("inf")
    optimizer_label = str(cfg.optimizer)

    if cfg.optimizer == "martens_grosse" and int(cfg.mg_steps) > 0:
        optimizer_label = f"MartensGrosseGN({cfg.mg_solver})"
        # Periodic resample: chunk MG steps between collocation refreshes.
        chunk = max(1, int(cfg.resample_every) if cfg.resample_every > 0 else cfg.mg_steps)
        steps_left = int(cfg.mg_steps)
        mg_failed = False
        while steps_left > 0 and not mg_failed:
            n_chunk = min(chunk, steps_left)
            flat0, residual_fn = functional_residual_fn(residual_mod)
            try:
                params, loss_hist = martens_grosse_gauss_newton_minimize(
                    residual_fn,
                    flat0,
                    steps=n_chunk,
                    damping=1e-3,
                    solver=cfg.mg_solver,
                    use_martens_grosse=True,
                )
            except RuntimeError:
                mg_failed = True
                optimizer_label = "MartensGrosseGN_failed_fallback_CubicGN"
                break
            _load_flat_params(residual_mod, params)
            for loss in loss_hist:
                history.append(float(math.sqrt(2.0 * max(float(loss), 0.0))))
            cur = history[-1] if history else float("inf")
            if cur < best_r:
                best_r = cur
            steps_left -= n_chunk
            if steps_left > 0 and cfg.resample_every > 0:
                _resample_collocation(
                    net,
                    residual_mod,
                    cfg=cfg,
                    scales=scales,
                    gammas=gammas,
                    n_pts=n_pts,
                    rng=rng,
                )
        if mg_failed or (not history and int(cfg.cubic_gn_steps) == 0):
            # Fall back to CubicGN so reproduction arm still makes progress.
            with torch.no_grad():
                r0 = residual_mod()
            if torch.isfinite(r0).all():
                opt = CubicGaussNewton(net.parameters(), sigma=1.0, krylov_dim=16)
                n_fb = max(int(cfg.mg_steps) // 2, 8)
                for step in range(n_fb):
                    if cfg.resample_every > 0 and step > 0 and step % int(cfg.resample_every) == 0:
                        _resample_collocation(
                            net,
                            residual_mod,
                            cfg=cfg,
                            scales=scales,
                            gammas=gammas,
                            n_pts=n_pts,
                            rng=rng,
                        )

                    def closure_fb() -> Tensor:
                        return residual_mod()

                    try:
                        r_out = opt.step(closure_fb)
                    except RuntimeError:
                        break
                    if not torch.isfinite(r_out).all():
                        break
                    r_max = float(torch.max(torch.abs(r_out.detach())))
                    history.append(r_max)
                    if r_max < best_r:
                        best_r = r_max
            else:
                optimizer_label = optimizer_label + "_nonfinite_residual"
    elif int(cfg.cubic_gn_steps) > 0:
        optimizer_label = "CubicGaussNewton"
        opt = CubicGaussNewton(net.parameters(), sigma=1.0, krylov_dim=16)
        best_state: list[Tensor] | None = None
        stall = 0
        for step in range(int(cfg.cubic_gn_steps)):
            if cfg.resample_every > 0 and step > 0 and step % int(cfg.resample_every) == 0:
                _resample_collocation(
                    net,
                    residual_mod,
                    cfg=cfg,
                    scales=scales,
                    gammas=gammas,
                    n_pts=n_pts,
                    rng=rng,
                )

            def closure() -> Tensor:
                return residual_mod()

            r_out = opt.step(closure)
            r_max = float(torch.max(torch.abs(r_out.detach())))
            history.append(r_max)
            if r_max < best_r * 0.999:
                best_r = r_max
                stall = 0
                best_state = [p.detach().clone() for p in net.parameters()]
            else:
                stall += 1
                if stall >= max(8, int(cfg.cubic_gn_steps) // 10) and best_state is not None:
                    with torch.no_grad():
                        for p, b in zip(net.parameters(), best_state, strict=True):
                            p.copy_(b)
                    break
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(net.parameters(), best_state, strict=True):
                    p.copy_(b)

    if cfg.qr_gn_steps > 0 and (best_r < 1.0 or not history):
        flat0, residual_fn = functional_residual_fn(residual_mod)
        gn = GaussNewton(
            solver="qr",
            damping=1e-4,
            damping_strategy="nielsen",
            use_martens_grosse=cfg.optimizer == "martens_grosse",
        )
        params = flat0
        qr_best = params.detach().clone()
        qr_best_loss = best_r if math.isfinite(best_r) else float("inf")
        try:
            for _ in range(int(cfg.qr_gn_steps)):
                params, info = gn.step(residual_fn, params)
                loss_r = float(math.sqrt(2.0 * max(info.loss, 0.0)))
                history.append(loss_r)
                if loss_r < qr_best_loss:
                    qr_best_loss = loss_r
                    qr_best = params.detach().clone()
            _load_flat_params(residual_mod, qr_best)
            best_r = min(best_r, qr_best_loss)
            optimizer_label = optimizer_label + "+QR"
        except RuntimeError:
            optimizer_label = optimizer_label + "+QR_skipped"

    # Dense eval grid for absolute residual + Hardy projection for CAP handoff
    y_eval = torch.linspace(
        -float(cfg.y_max),
        float(cfg.y_max),
        int(cfg.n_grid),
        dtype=cfg.dtype,
        device=device,
    )
    with torch.no_grad():
        omega, omega_y, _, _ = omega_from_net(
            net, y_eval, lam=cfg.lam, exp_core=cfg.exp_core
        )
    y_np = y_eval.detach().cpu().numpy()
    om_np = omega.detach().cpu().numpy()
    omy_np = omega_y.detach().cpu().numpy()
    g_sample = float(np.interp(cfg.gauge_point, y_np, om_np))
    if abs(g_sample) > 1e-14:
        scale = float(cfg.gauge_value) / g_sample
        om_np = om_np * scale
        omy_np = omy_np * scale
        omega = torch.as_tensor(om_np, dtype=cfg.dtype, device=device)
        omega_y = torch.as_tensor(omy_np, dtype=cfg.dtype, device=device)
    with torch.no_grad():
        r_train_h, defect_train, _, _ = vorticity_fields(
            y_eval,
            omega,
            omega_y,
            lam=cfg.lam,
            scales=scales,
            gammas=gammas,
            train_hilbert=cfg.train_hilbert,
            hilbert_n_uniform=cfg.hilbert_n_uniform,
        )
        try:
            coeffs, defect, _ = project_omega_hardy_torch(
                y_eval, omega, scales=scales, gammas=gammas
            )
        except RuntimeError:
            coeffs = torch.zeros(scales.shape[0], dtype=cfg.dtype, device=device)
            defect = torch.zeros((), dtype=cfg.dtype, device=device)
    om_g = float(np.interp(cfg.gauge_point, y_np, om_np))
    dense = dense_neural_vorticity_residual(
        net,
        lam=cfg.lam,
        train_hilbert=cfg.train_hilbert,
        scales=scales,
        gammas=gammas,
        y_max=float(cfg.y_max),
        n_val=int(cfg.dense_n_val),
        exp_core=cfg.exp_core,
        hilbert_n_uniform=cfg.hilbert_n_uniform,
        gauge_point=cfg.gauge_point,
        gauge_value=cfg.gauge_value,
        dtype=cfg.dtype,
    )

    r_np = r_train_h.detach().cpu().numpy()
    return CCFVorticityNeuralResult(
        lam=float(cfg.lam),
        y=y_np,
        omega=om_np,
        omega_y=omy_np,
        residual=r_np,
        coeffs=np.asarray(coeffs.detach().cpu().numpy(), dtype=float),
        scales=scales_np,
        gammas=gammas_np,
        diagnostics={
            "max_abs_vorticity_residual": float(np.max(np.abs(r_np))),
            "rms_vorticity_residual": float(np.sqrt(np.mean(r_np * r_np))),
            "projection_defect": float(defect.item()),
            "projection_defect_train_report": float(defect_train.item()),
            "omega_gauge_sample": float(om_g),
            "omega_max_abs": float(np.max(np.abs(om_np))),
            "final_train_max_abs": float(history[-1]) if history else float("nan"),
            "reproduction_dense_max_abs": float(dense["reproduction_dense_max_abs"]),
            "reproduction_dense_rms": float(dense["reproduction_dense_rms"]),
            "reproduction_dense_max_abs_for_gate": float(
                dense["reproduction_dense_max_abs_for_gate"]
            ),
        },
        config=cfg,
        extra={
            "optimizer": optimizer_label,
            "arm": cfg.arm,
            "train_hilbert": cfg.train_hilbert,
            "rung_hilbert": "hardy_projection_exact",
            "hilbert": cfg.train_hilbert,
            "residual_form": "wang_vorticity",
            "use_grad_norm": cfg.use_grad_norm,
            "train_history_max_abs": history,
            "rung_metric_uses_fft": cfg.train_hilbert == "truncated_line_spectral",
            "net": net,
        },
    )


def grad_norm_downweights_peak(
    *,
    n: int = 201,
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Regression helper: grad-norm reduces peak weight vs absolute L2."""
    y = torch.linspace(-8.0, 8.0, n, dtype=torch.float64)
    # Synthetic sharp residual peak near y=4
    r = torch.exp(-((y - 4.0) ** 2) / 0.05) + 0.01 * torch.ones_like(y)
    nn_core = 3.0 * torch.exp(-((y - 4.0) ** 2) / 0.05)
    r_n = gradient_normalize(r, nn_core, alpha=alpha, eps=eps)
    peak = int(torch.argmax(torch.abs(r)).item())
    return {
        "abs_peak": float(r[peak].abs()),
        "norm_peak": float(r_n[peak].abs()),
        "abs_mean": float(r.abs().mean()),
        "norm_mean": float(r_n.abs().mean()),
        "peak_to_mean_abs": float(r[peak].abs() / (r.abs().mean() + 1e-30)),
        "peak_to_mean_norm": float(r_n[peak].abs() / (r_n.abs().mean() + 1e-30)),
    }


__all__ = [
    "CCFVorticityNeuralConfig",
    "CCFVorticityNeuralResult",
    "CompactifiedOmegaOMBU",
    "dense_neural_vorticity_residual",
    "grad_norm_downweights_peak",
    "gradient_normalize",
    "hardy_corrected_hu_from_omega",
    "hardy_dictionary",
    "hilbert_pv_line",
    "omega_from_net",
    "project_omega_hardy",
    "project_omega_hardy_torch",
    "pv_hu_from_omega",
    "reproduce_deepmind_config",
    "residual_vector",
    "run_ccf_vorticity_neural_discovery",
    "spectral_hu_from_omega",
    "vorticity_fields",
    "wang_residual",
]

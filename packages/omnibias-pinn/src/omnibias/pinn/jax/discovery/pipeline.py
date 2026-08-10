# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Reusable self-similar singularity discovery-and-certification pipeline.

Five stages with swappable problem adapters:

1. **Ansatz** — exponents, envelope, lambda-tied compactification
2. **Exact operators** — closed-form derivatives + Hardy / streamfunction nonlocality
3. **Discovery** — CubicGN / Martens-Grosse GN (Adam forbidden on earn path)
4. **Rigor** — interval residual, sequence-space NK, sealed certificate
5. **Formal** — finite rational obligation for the Lean kernel

IPM, Boussinesq, Euler, and gauge/spectral-gap sub-obligations plug in via
:class:`ProblemAdapter`. Clay parents stay external obligations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# Absolute Rung-1 residual gate (must match benchmarks/_gates.py).
CCF_RUNG1_RESIDUAL_GATE = 1e-11
CCF_RUNG1_LAMBDA = 0.6057


class ProblemAdapter(Protocol):
    """Swappable problem surface for the singularity pipeline."""

    name: str

    def discover(self, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Return a discovery dict with at least ``lam`` and residual diagnostics."""

    def certify(self, discovery: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Build a certificate / CAP bundle from a discovery result."""


@dataclass
class PipelineConfig:
    seed: int = 0
    polish: bool = False
    spectrum: bool = False
    dissipation: bool = False
    residual_gate: float = CCF_RUNG1_RESIDUAL_GATE
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    adapter: str
    discovery: dict[str, Any]
    certificate: dict[str, Any]
    spectrum: dict[str, Any] | None = None
    dissipation: dict[str, Any] | None = None
    honesty: dict[str, Any] = field(default_factory=dict)


def run_singularity_pipeline(
    adapter: ProblemAdapter,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run stages 3–4 (and optional spectrum / dissipation) for ``adapter``."""
    cfg = PipelineConfig() if config is None else config
    discovery = adapter.discover(seed=cfg.seed, **cfg.extra)
    certify_kw = dict(cfg.extra)
    certify_kw.setdefault("residual_gate", cfg.residual_gate)
    certificate = adapter.certify(discovery, **certify_kw)
    spectrum = None
    dissipation = None
    if cfg.spectrum and "coeffs" in discovery and "scales" in discovery:
        from omnibias.pinn.jax.discovery.spectrum import sealed_ccf_unstable_mode_count

        spectrum = sealed_ccf_unstable_mode_count(
            coeffs=list(discovery["coeffs"]),
            scales=list(discovery["scales"]),
            lam=float(discovery["lam"]),
            claimed_order=int(discovery.get("claimed_order", 1)),
        )
    if cfg.dissipation and "lambda_lo" in discovery and "lambda_hi" in discovery:
        from omnibias.pinn.certified.dissipation_threshold import (
            certified_fractional_dissipation_threshold,
        )

        dissipation = certified_fractional_dissipation_threshold(
            lambda_lo=float(discovery["lambda_lo"]),
            lambda_hi=float(discovery["lambda_hi"]),
        )
    honesty = {
        "unproven_claim": False,
        "navier_stokes_proof_claim": False,
        "yang_mills_mass_gap_claim": False,
        "adapter": getattr(adapter, "name", type(adapter).__name__),
        "enabler_surface": True,
        "optimizer_doctrine": "CubicGaussNewton_or_MartensGrosse_GN_only_on_earn_path",
        "notes": (
            "Pipeline supplies numerical candidates and machine-checked local "
            "certificates; Clay parent problems remain external obligations."
        ),
    }
    return PipelineResult(
        adapter=str(honesty["adapter"]),
        discovery=dict(discovery),
        certificate=dict(certificate),
        spectrum=spectrum,
        dissipation=dissipation,
        honesty=honesty,
    )


@dataclass
class CCFHardyAdapter:
    """CCF Hardy vorticity discovery (Martens–Grosse) + whole-line CAP.

    Earn path uses JAX Hardy-Ω Gauss–Newton with Martens–Grosse schedules.
    Adam / Θ-line smoke is intentionally not used here.
    """

    name: str = "ccf_hardy"
    n_scales: int = 4
    n_gamma_multiples: int = 2
    n_grid: int = 65
    steps: int = 20
    lam_init: float = CCF_RUNG1_LAMBDA
    y_max: float = 20.0

    def discover(self, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.jax.discovery import ccf_vorticity

        optimizer = str(kwargs.get("optimizer", "martens_grosse_gn"))
        if optimizer.lower() in {"adam", "sgd"}:
            raise ValueError(
                "CCFHardyAdapter earn path forbids Adam/SGD; use Martens–Grosse GN "
                f"(got optimizer={optimizer!r})"
            )
        cfg = ccf_vorticity.CCFVorticityDiscoveryConfig(
            n_scales=int(kwargs.get("n_scales", self.n_scales)),
            n_gamma_multiples=int(
                kwargs.get("n_gamma_multiples", self.n_gamma_multiples)
            ),
            n_grid=int(kwargs.get("n_grid", self.n_grid)),
            y_max=float(kwargs.get("y_max", self.y_max)),
            lam=float(kwargs.get("lam_init", self.lam_init)),
            seed=int(seed),
            gn_steps=int(kwargs.get("steps", self.steps)),
        )
        result = ccf_vorticity.run_ccf_vorticity_discovery(cfg)
        return {
            "lam": float(result.lam),
            "coeffs": np.asarray(result.coeffs, dtype=float).tolist(),
            "scales": np.asarray(result.scales, dtype=float).tolist(),
            "gammas": np.asarray(result.alphas, dtype=float).tolist(),
            "max_abs_residual": float(
                result.diagnostics["dense_max_abs_vorticity"]
            ),
            "claimed_order": 1,
            "optimizer": "martens_grosse_gn",
            "train_hilbert": "hardy_exact_omega",
            "gn_solver": "qr",
            "result": result,
        }

    def certify(self, discovery: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.certified.ccf_hardy import (
            certified_ccf_hardy_wholeline_blowup_attempt,
        )

        gate = float(kwargs.get("residual_gate", CCF_RUNG1_RESIDUAL_GATE))
        gammas = discovery.get("gammas")
        return certified_ccf_hardy_wholeline_blowup_attempt(
            coeffs=list(discovery["coeffs"]),
            scales=list(discovery["scales"]),
            lam=float(discovery["lam"]),
            form="vorticity",
            gammas=list(gammas) if gammas is not None else None,
            residual_gate=gate,
            velocity_sign=-1.0,
        )


@dataclass
class IPMAdapter:
    """IPM self-similar smoke discovery + CAP bundle (scaffold until absolute gates)."""

    name: str = "ipm"
    n: int = 12
    steps: int = 40
    lam_init: float = 0.5

    def discover(self, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.jax.discovery import ipm
        from omnibias.pinn.jax.discovery.lambda_laws import predict_lambda_init

        order = int(kwargs.get("order", 1))
        if "lam_init" in kwargs:
            lam0 = float(kwargs["lam_init"])
        else:
            lam0 = float(predict_lambda_init(order, family="ipm"))
        cfg = ipm.IPMDiscoveryConfig(
            n=int(kwargs.get("n", self.n)),
            lam_init=lam0,
            seed=int(seed),
            steps=int(kwargs.get("steps", self.steps)),
        )
        out = ipm.run_ipm_discovery(cfg)
        max_r = max(
            float(out.get("max_abs_residual_theta", 0.0)),
            float(out.get("max_abs_residual_psi", 0.0)),
        )
        payload = dict(out)
        return {
            **payload,
            "lam": float(out["lam"]),
            "max_abs_residual": max_r,
            "claimed_order": order,
            "optimizer": "adam_smoke_scaffold",
            "result": out,
        }

    def certify(self, discovery: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.certified.ipm import build_ipm_cap_bundle

        raw = discovery.get("result", discovery)
        return build_ipm_cap_bundle(dict(raw))


@dataclass
class BoussinesqAdapter:
    """Boussinesq self-similar smoke discovery + CAP (scaffold until absolute gates)."""

    name: str = "boussinesq"
    n: int = 12
    steps: int = 40
    lam_init: float = 1.5

    def discover(self, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.jax.discovery import boussinesq
        from omnibias.pinn.jax.discovery.lambda_laws import predict_lambda_init

        order = int(kwargs.get("order", 1))
        try:
            lam0 = float(
                kwargs["lam_init"]
                if "lam_init" in kwargs
                else predict_lambda_init(order, family="boussinesq")
            )
        except Exception:
            lam0 = float(kwargs.get("lam_init", self.lam_init))
        cfg = boussinesq.BoussinesqDiscoveryConfig(
            n=int(kwargs.get("n", self.n)),
            lam_init=lam0,
            seed=int(seed),
            steps=int(kwargs.get("steps", self.steps)),
        )
        out = boussinesq.run_boussinesq_discovery(cfg)
        max_r = max(
            float(out.get("max_abs_residual_omega", 0.0)),
            float(out.get("max_abs_residual_theta", 0.0)),
            float(out.get("max_abs_residual_psi", 0.0)),
        )
        return {
            **dict(out),
            "lam": float(out["lam"]),
            "max_abs_residual": max_r,
            "claimed_order": order,
            "optimizer": "adam_smoke_scaffold",
            "result": out,
        }

    def certify(self, discovery: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.certified.boussinesq import build_boussinesq_cap_bundle

        raw = discovery.get("result", discovery)
        return build_boussinesq_cap_bundle(dict(raw))


__all__ = [
    "BoussinesqAdapter",
    "CCFHardyAdapter",
    "CCF_RUNG1_LAMBDA",
    "CCF_RUNG1_RESIDUAL_GATE",
    "IPMAdapter",
    "PipelineConfig",
    "PipelineResult",
    "ProblemAdapter",
    "run_singularity_pipeline",
]

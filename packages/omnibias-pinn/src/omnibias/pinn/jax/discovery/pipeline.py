# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Reusable self-similar singularity discovery-and-certification pipeline.

Five stages with swappable problem adapters:

1. **Ansatz** — exponents, envelope, lambda-tied compactification
2. **Exact operators** — closed-form derivatives + Hardy / streamfunction nonlocality
3. **Discovery** — GN + Martens-Grosse, linearized multistage, funnel, mpmath polish
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
    residual_gate: float = 1e-6
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
    certificate = adapter.certify(discovery, **cfg.extra)
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
    """CCF line Hardy discovery + whole-line CAP adapter."""

    name: str = "ccf_hardy"
    n_terms: int = 4
    n_grid: int = 32
    steps: int = 20
    lam_init: float = 0.6057

    def discover(self, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        import jax
        import jax.numpy as jnp

        from omnibias.pinn.jax.discovery import ccf_line

        cfg = ccf_line.CCFLineDiscoveryConfig(
            n_terms=int(kwargs.get("n_terms", self.n_terms)),
            n_grid=int(kwargs.get("n_grid", self.n_grid)),
            seed=seed,
            optimizer="adam",
            lam_init=float(kwargs.get("lam_init", self.lam_init)),
        )
        result = ccf_line.run_ccf_line_discovery(
            cfg, steps=int(kwargs.get("steps", self.steps)), lr=5e-3
        )
        log_scales = np.asarray(result.params["log_scales"], dtype=float)
        scales = np.asarray(jax.nn.softplus(jnp.asarray(log_scales)) + 1e-3)
        return {
            "lam": float(result.lam),
            "coeffs": np.asarray(result.params["coeffs"]).tolist(),
            "scales": scales.tolist(),
            "max_abs_residual": float(result.diagnostics["max_abs_residual"]),
            "claimed_order": 1,
            "result": result,
        }

    def certify(self, discovery: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        from omnibias.pinn.certified.ccf_hardy import (
            certified_ccf_hardy_wholeline_blowup_attempt,
        )

        return certified_ccf_hardy_wholeline_blowup_attempt(
            coeffs=list(discovery["coeffs"]),
            scales=list(discovery["scales"]),
            lam=float(discovery["lam"]),
            residual_gate=float(kwargs.get("residual_gate", 1e-6)),
        )


__all__ = [
    "CCFHardyAdapter",
    "PipelineConfig",
    "PipelineResult",
    "ProblemAdapter",
    "run_singularity_pipeline",
]

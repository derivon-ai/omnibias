# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CAP-ready export of a CCF discovery result.

A computer-assisted proof (CAP) via interval arithmetic needs a *self-contained,
independently checkable* description of a candidate solution -- not a neural
network. This module turns a :class:`~omnibias.pinn.jax.discovery.ccf.CCFDiscoveryResult`
into such a bundle:

* the equation metadata (form, sign, self-similar ansatz, ``k(lambda) = lambda``);
* the admissible-candidate ``lambda``;
* the grid + sampled profile ``(y, Theta, Theta')`` so a checker can recompute the
  residual from scratch;
* an **interval-friendly band-limited representation**: the normalized DFT
  coefficients of ``Theta`` above a threshold, plus a rigorous :math:`\ell^1`
  bound on the discarded tail (since each mode has unit modulus, the dropped part
  is bounded in sup-norm by the sum of dropped coefficient magnitudes);
* the residual vector and its max / RMS / d1 diagnostics;
* **honesty flags** that state plainly this is a periodic-truncation candidate, is
  not an exact symbolic solution, and is not a reproduction of a published
  line-domain ``lambda`` value.

The companion :mod:`omnibias.symbolic.ccf` re-evaluates ``validation_inputs`` with
an independent numpy implementation; the schema test asserts every key a checker
needs is present.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn.jax.discovery.ccf import CCFDiscoveryResult

SCHEMA_VERSION = "ccf-cap-1"

#: Top-level keys every CAP bundle must carry (asserted by the schema test).
REQUIRED_CAP_KEYS: tuple[str, ...] = (
    "schema_version",
    "problem",
    "lambda",
    "domain",
    "profile_samples",
    "fourier_representation",
    "residual_diagnostics",
    "residual_samples",
    "validation_inputs",
    "honesty",
    "provenance",
)

#: Keys a checker needs to recompute the residual without the network.
REQUIRED_VALIDATION_KEYS: tuple[str, ...] = (
    "y",
    "theta",
    "theta_y",
    "lambda",
    "form",
    "velocity_sign",
    "hilbert_convention",
)


def _band_limited_representation(
    theta: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """Normalized DFT coefficients above ``threshold`` + l1 tail bound.

    Reconstruction convention (exact on the sampling grid)::

        theta[j] = sum_k c_k * exp(2j*pi*k*j / N)
    """
    n = int(theta.shape[0])
    coeffs = np.fft.fft(theta) / n  # ifft-normalized: theta_j = sum_k c_k e^{i 2pi k j/N}
    freqs = np.fft.fftfreq(n) * n  # integer mode index k
    mag = np.abs(coeffs)
    keep = mag > threshold
    tail_l1 = float(np.sum(mag[~keep]))
    kept = [
        {"k": int(freqs[i]), "real": float(coeffs[i].real), "imag": float(coeffs[i].imag)}
        for i in np.nonzero(keep)[0]
    ]
    kept.sort(key=lambda d: abs(d["k"]))
    return {
        "convention": "theta[j] = sum_k c_k * exp(2*pi*i*k*j/N)",
        "n_grid": n,
        "threshold": float(threshold),
        "n_kept": len(kept),
        "coefficients": kept,
        "tail_l1_sup_bound": tail_l1,
    }


def build_cap_bundle(
    result: CCFDiscoveryResult,
    *,
    fourier_threshold: float = 1e-12,
    reproduces_published_lambda: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a JSON-serializable CAP bundle from a discovery result."""
    cfg = result.config
    theta = np.asarray(result.theta, dtype=float)
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "problem": {
            "model": "cordoba_cordoba_fontelos",
            "description": "1D nonlocal-transport self-similar blow-up profile",
            "form": cfg.form,
            "velocity_sign": float(cfg.velocity_sign),
            "ansatz": "theta(x,t) = (1-t)^{k(lambda)} Theta(y), y = (1-t)^{-(1+lambda)} x",
            "blowup_time": 1.0,
            "scalar_exponent_k": "lambda",
            "stationary_residual_transport": "(1+lambda) y Theta' - lambda Theta + (H Theta) Theta'",
            "parity": cfg.parity,
        },
        "lambda": float(result.lam),
        "domain": {
            "type": "periodic_truncation",
            "half_period": float(cfg.half_period),
            "interval": [-float(cfg.half_period), float(cfg.half_period)],
            "n_grid": int(cfg.n_grid),
            "periodic": True,
            "dtype": "float64",
        },
        "profile_samples": {
            "y": np.asarray(result.y, dtype=float).tolist(),
            "theta": theta.tolist(),
            "theta_y": np.asarray(result.theta_y, dtype=float).tolist(),
        },
        "fourier_representation": _band_limited_representation(
            theta, threshold=fourier_threshold
        ),
        "residual_diagnostics": {k: float(v) for k, v in result.diagnostics.items()},
        "residual_samples": np.asarray(result.residual, dtype=float).tolist(),
        "validation_inputs": {
            "y": np.asarray(result.y, dtype=float).tolist(),
            "theta": theta.tolist(),
            "theta_y": np.asarray(result.theta_y, dtype=float).tolist(),
            "lambda": float(result.lam),
            "form": cfg.form,
            "velocity_sign": float(cfg.velocity_sign),
            "hilbert_convention": "periodic_fft_minus_i_sgn",
        },
        "honesty": {
            "exact_solution_claim": False,
            "domain_note": (
                "Periodic-interval truncation of the unbounded-domain self-similar "
                "profile; the Hilbert transform is the spectral periodic transform."
            ),
            "reproduces_published_lambda": reproduces_published_lambda,
            "is_forced_manufactured_solution": bool(result.forced),
            "navier_stokes_proof_claim": False,
            "notes": notes,
        },
        "provenance": {
            "harness": "omnibias.pinn.jax.discovery.ccf",
            "seed": int(cfg.seed),
            "hidden": int(cfg.hidden),
            "train_lam": bool(cfg.train_lam),
            "derivatives": "omnibias closed-form tanh fastpath (local); FFT Hilbert (nonlocal)",
            "python": platform.python_version(),
        },
    }
    return bundle


def cap_schema_errors(bundle: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty list == valid)."""
    errors: list[str] = []
    for key in REQUIRED_CAP_KEYS:
        if key not in bundle:
            errors.append(f"missing top-level key: {key!r}")
    vin = bundle.get("validation_inputs", {})
    for key in REQUIRED_VALIDATION_KEYS:
        if key not in vin:
            errors.append(f"missing validation_inputs key: {key!r}")
    # length consistency
    try:
        ny = len(vin["y"])
        if len(vin["theta"]) != ny or len(vin["theta_y"]) != ny:
            errors.append("validation_inputs arrays have inconsistent lengths")
        if len(bundle["residual_samples"]) != ny:
            errors.append("residual_samples length != grid length")
    except (KeyError, TypeError):
        errors.append("validation_inputs arrays are missing or not sized")
    if bundle.get("honesty", {}).get("navier_stokes_proof_claim", True) is not False:
        errors.append("honesty.navier_stokes_proof_claim must be False")
    return errors


def write_cap_bundle(bundle: dict[str, Any], out_dir: Path | str) -> Path:
    """Write ``ccf_cap.json`` and a short markdown summary to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ccf_cap.json"
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    diag = bundle["residual_diagnostics"]
    lines = [
        "# CCF self-similar candidate (CAP-ready bundle)",
        "",
        f"- model: `{bundle['problem']['model']}` ({bundle['problem']['form']} form)",
        f"- lambda: `{bundle['lambda']:.10g}`",
        f"- domain: periodic `[-{bundle['domain']['half_period']:.6g}, "
        f"{bundle['domain']['half_period']:.6g})`, n_grid=`{bundle['domain']['n_grid']}`",
        f"- max|residual|: `{diag.get('max_abs_residual', float('nan')):.3e}`",
        f"- rms residual: `{diag.get('rms_residual', float('nan')):.3e}`",
        f"- band-limited modes kept: `{bundle['fourier_representation']['n_kept']}` "
        f"(tail l1 bound `{bundle['fourier_representation']['tail_l1_sup_bound']:.3e}`)",
        "",
        "## Honesty",
        f"- exact symbolic solution: `{bundle['honesty']['exact_solution_claim']}`",
        f"- reproduces published lambda: `{bundle['honesty']['reproduces_published_lambda']}`",
        f"- Navier-Stokes proof claim: `{bundle['honesty']['navier_stokes_proof_claim']}`",
        f"- {bundle['honesty']['domain_note']}",
    ]
    (out / "ccf_cap_summary.md").write_text("\n".join(lines) + "\n")
    return json_path


__all__ = [
    "REQUIRED_CAP_KEYS",
    "REQUIRED_VALIDATION_KEYS",
    "SCHEMA_VERSION",
    "build_cap_bundle",
    "cap_schema_errors",
    "write_cap_bundle",
]

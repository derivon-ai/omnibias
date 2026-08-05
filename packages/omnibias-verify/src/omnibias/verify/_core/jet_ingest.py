# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Extract plain ``(W, b, activation)`` float layers for the *certified jet*.

:func:`verified_layers` turns a trained ``JetMLP``-style network (any object that
exposes ``_layer_specs()`` -- the omnibias torch and jax ``JetMLP`` both do) into
the layer list that
:func:`omnibias.core.verified.jet_mv.mlp_jet_mv` /
:func:`omnibias.core.verified.pde_certificate.certified_interior_residual`
consume: ``[(W, b, name), ...]`` with ``name`` one of the verified-tower
activations (``"tanh"``, ``"sigmoid"``, ``"gaussian"``, ``"silu"``, ``"gelu"``,
``"softplus"``) or ``None`` for an affine readout.

This module is **backend-neutral** -- it duck-types the weight tensors
(``.detach()`` for torch, ``.tolist()`` for jax / numpy) and therefore imports
neither torch nor jax, keeping the verifier core dependency-free.  Activations the
verified tower cannot represent (e.g. ``"sin"`` in a SIREN / spectral net) are
rejected loudly rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from omnibias.core.verified.pde_certificate import (
    AdaptiveResidualDiagnostics,
    BoundaryFace,
    LinearPDE,
    PINNErrorCertificate,
    StabilityEstimate,
    StructuralInvariant,
    adaptive_certified_interior_residual,
    aposteriori_error_certificate,
)

#: A plain certified-jet layer: ``(W, b-or-None, activation-name-or-None)``.
VerifiedLayer = tuple[list[list[float]], list[float] | None, str | None]

#: Activations the verified derivative tower supports.
SUPPORTED_ACTIVATIONS = ("tanh", "sigmoid", "gaussian", "silu", "gelu", "softplus")


@dataclass(frozen=True)
class VerifiedLayerBundle:
    """Certified-jet layers plus reproducibility metadata."""

    layers: list[VerifiedLayer]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CertifiedPDEPipelineResult:
    """Result of the verify-side model -> PDE certificate convenience path."""

    layers: list[VerifiedLayer]
    metadata: dict[str, Any]
    certificate: PINNErrorCertificate
    diagnostics: AdaptiveResidualDiagnostics | None = None


def _to_matrix(weight: Any) -> list[list[float]]:
    if hasattr(weight, "detach"):  # torch tensor
        weight = weight.detach()
    rows = weight.tolist() if hasattr(weight, "tolist") else weight
    return [[float(x) for x in row] for row in rows]


def _to_vector(bias: Any) -> list[float] | None:
    if bias is None:
        return None
    if hasattr(bias, "detach"):
        bias = bias.detach()
    seq = bias.tolist() if hasattr(bias, "tolist") else bias
    return [float(x) for x in seq]


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _activation_name(spec: Any) -> str | None:
    if spec is None:
        return None
    name = spec if isinstance(spec, str) else getattr(spec, "name", None)
    if name in SUPPORTED_ACTIVATIONS:
        return str(name)
    raise ValueError(
        f"activation {name!r} is not supported by the verified jet "
        f"(supported: {', '.join(SUPPORTED_ACTIVATIONS)})"
    )


def verified_layers(net: Any) -> list[VerifiedLayer]:
    """Extract certified-jet ``(W, b, name)`` float layers from a ``JetMLP``-like net.

    ``net`` must expose ``_layer_specs() -> [(weight, bias, spec), ...]`` (the
    omnibias torch / jax ``JetMLP`` do).  ``spec`` is an ``ActivationSpec`` (its
    ``.name`` is mapped to the verified tower name), a bare activation-name string,
    or ``None`` for the affine readout.  Raises ``TypeError`` if ``net`` has no
    ``_layer_specs`` and ``ValueError`` on an unsupported activation.
    """
    specs = getattr(net, "_layer_specs", None)
    if not callable(specs):
        raise TypeError(
            "network must expose a _layer_specs() method (e.g. a JetMLP); "
            f"got {type(net).__name__}"
        )
    return [
        (_to_matrix(weight), _to_vector(bias), _activation_name(spec))
        for weight, bias, spec in specs()
    ]


def _layers_digest(layers: Sequence[VerifiedLayer]) -> str:
    body = json.dumps(layers, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def verified_layer_bundle(
    net: Any,
    *,
    domain: Sequence[Any] | None = None,
    boundary: Sequence[Any] = (),
    provenance: Mapping[str, Any] | None = None,
) -> VerifiedLayerBundle:
    """Extract layers and attach deterministic metadata for certificate payloads."""
    specs_fn = getattr(net, "_layer_specs", None)
    if not callable(specs_fn):
        raise TypeError(
            "network must expose a _layer_specs() method (e.g. a JetMLP); "
            f"got {type(net).__name__}"
        )
    raw_specs = list(specs_fn())
    layers = [
        (_to_matrix(weight), _to_vector(bias), _activation_name(spec))
        for weight, bias, spec in raw_specs
    ]
    activations = [name for _w, _b, name in layers if name is not None]
    dtypes = sorted(
        {
            dtype
            for weight, bias, _spec in raw_specs
            for dtype in (_dtype_name(weight), _dtype_name(bias))
            if dtype is not None
        }
    )
    n_params = 0
    for weight, bias, _name in layers:
        n_params += sum(len(row) for row in weight)
        if bias is not None:
            n_params += len(bias)
    metadata: dict[str, Any] = {
        "network_type": type(net).__name__,
        "n_layers": len(layers),
        "n_parameters": n_params,
        "activations": activations,
        "supported_activation_set": list(SUPPORTED_ACTIVATIONS),
        "dtypes": dtypes,
        "layers_digest": _layers_digest(layers),
        "domain": None if domain is None else list(domain),
        "n_boundary_faces": len(boundary),
        "provenance": dict(provenance) if provenance is not None else {},
    }
    return VerifiedLayerBundle(layers, metadata)


def certify_pinn_aposteriori(
    net: Any,
    domain: Sequence[Any],
    pde: LinearPDE,
    *,
    boundary: Sequence[BoundaryFace] = (),
    stability: StabilityEstimate | None = None,
    stability_interior: float = 1.0,
    stability_boundary: float = 1.0,
    invariants: Sequence[StructuralInvariant] = (),
    target_residual: float | None = None,
    initial_splits: int | Sequence[int] = 1,
    max_splits: int = 16,
    boundary_splits: int | Sequence[int] = 1,
    max_error: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> CertifiedPDEPipelineResult:
    """Convenience path from a JetMLP-like model to a sealed PDE certificate."""
    bundle = verified_layer_bundle(
        net, domain=domain, boundary=boundary, provenance=provenance
    )
    diagnostics = adaptive_certified_interior_residual(
        bundle.layers,
        domain,
        pde,
        target=target_residual,
        initial_splits=initial_splits,
        max_splits=max_splits,
    )
    cert = aposteriori_error_certificate(
        bundle.layers,
        domain,
        pde,
        boundary=boundary,
        stability=stability,
        stability_interior=stability_interior,
        stability_boundary=stability_boundary,
        invariants=invariants,
        model_metadata=bundle.metadata,
        max_error=max_error,
        diagnostics=diagnostics,
        boundary_splits=boundary_splits,
    )
    return CertifiedPDEPipelineResult(bundle.layers, bundle.metadata, cert, diagnostics)


__all__ = [
    "CertifiedPDEPipelineResult",
    "SUPPORTED_ACTIVATIONS",
    "VerifiedLayer",
    "VerifiedLayerBundle",
    "certify_pinn_aposteriori",
    "verified_layer_bundle",
    "verified_layers",
]

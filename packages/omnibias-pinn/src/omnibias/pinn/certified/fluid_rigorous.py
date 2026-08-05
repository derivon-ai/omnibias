# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous *interval* Navier--Stokes residual certificates (streamfunction cage).

This is the **theorem-grade** companion of the FFT-sampled certificate in
:mod:`omnibias.pinn.certified.fluid`.  Where that module samples the residual on
grid nodes (``honesty.interval_verified = False``), this module encloses the
residual over the **whole continuous domain** -- between the grid nodes -- with
validated interval arithmetic, so ``honesty.interval_verified = True``.

The cage
--------
A 2-D incompressible velocity is read off a scalar streamfunction ``\psi`` as
``u = \nabla^\perp\psi = (\psi_y,\,-\psi_x)``.  Then ``\nabla\cdot u =
\psi_{xy}-\psi_{yx}\equiv 0`` *identically* (equality of mixed partials): the
field is divergence free **by construction**, not by penalty.  ``\psi`` is a
small ``tanh`` MLP in the verified :data:`~omnibias.core.verified.jet_mv.Layer`
format, so :func:`~omnibias.core.verified.jet_mv.certified_partials` encloses
every partial of ``\psi`` over an input box and the steady vorticity-transport
residual

.. math::

    R = (u\cdot\nabla)\omega - \nu\,\Delta\omega - f_\omega,
    \qquad \omega = -\Delta\psi,

is a rigorous :class:`~omnibias.core.verified.interval.Interval`.

A streamfunction that depends on ``y`` only induces a parallel shear
``u = (\psi'(y), 0)``; its advection ``(u\cdot\nabla)\omega`` vanishes *exactly*
(``\omega`` is ``x``-independent), so it is a certified **exact steady-Euler
solution** with a whole-domain residual enclosure of machine zero.  A general
``\psi(x,y)`` is a learned-style field whose residual the certificate bounds
rigorously; the bound tightens as ``splits`` subdivides the box.

Honesty / scope
---------------
``interval_verified = True`` here means exactly the validated-arithmetic enclosure
above -- the same standing as the linear a-posteriori PINN certificate.  It is
**not** a continuum regularity / global-regularity result, perfect weather, chaotic tracking,
or a turbulence closure; those honesty flags stay ``False`` and the schema gate
rejects any attempt to flip them.  The cage is intrinsically 2-D; 3-D exact flows
are covered (FFT-sampled) by :func:`~omnibias.pinn.certified.fluid_fixtures.beltrami_abc_flow`.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.core.proof import Conjecture, ProofAttempt
from omnibias.core.proof.certificate import (
    make_certificate,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.verified.jet_mv import Layer
from omnibias.core.verified.pde_certificate import (
    certified_streamfunction_divergence,
    certified_vorticity_transport_residual,
    structural_invariant,
)

NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION = "navier-stokes-streamfunction-residual-1"

#: honesty claims that must stay ``False`` on every streamfunction residual cert.
_FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "unproven_claim",
    "continuum_navier_stokes_claim",
    "chaotic_tracking_claim",
    "perfect_weather_claim",
    "turbulence_closure_claim",
)

_TWO_PI = 2.0 * float(np.pi)


# --------------------------------------------------------------------------- #
# Streamfunction fields (small tanh MLPs in the verified ``Layer`` format).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamfunctionField:
    """A scalar streamfunction MLP plus its JSON-native regeneration descriptor.

    ``layers`` is the in-memory :data:`~omnibias.core.verified.jet_mv.Layer`
    sequence consumed by the verified jet; ``descriptor`` is the JSON-native twin
    (weights, biases, activations, domain) carried inside a certificate so an
    independent verifier can rebuild the field from scratch.
    """

    layers: tuple[Layer, ...]
    domain: tuple[tuple[float, float], ...]
    descriptor: dict[str, Any]
    label: str


def _encode_layers(layers: tuple[Layer, ...]) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for weight, bias, name in layers:
        encoded.append({
            "weight": np.asarray(weight, dtype=float).tolist(),
            "bias": None if bias is None else np.asarray(bias, dtype=float).tolist(),
            "activation": name,
        })
    return encoded


def _decode_layers(encoded: list[dict[str, Any]]) -> tuple[Layer, ...]:
    layers: list[Layer] = []
    for layer in encoded:
        weight = [[float(v) for v in row] for row in layer["weight"]]
        raw_bias = layer.get("bias")
        bias = None if raw_bias is None else [float(v) for v in raw_bias]
        layers.append((weight, bias, layer.get("activation")))
    return tuple(layers)


def _mlp_streamfunction_layers(
    *,
    hidden: int,
    seed: int,
    weight_scale: float,
    y_only: bool,
    frequency: float,
) -> tuple[Layer, ...]:
    """Deterministic small ``tanh`` -> linear streamfunction MLP."""
    rng = np.random.default_rng(seed)
    w_in = weight_scale * rng.standard_normal((hidden, 2))
    if y_only:
        w_in[:, 0] = 0.0  # kill all x-dependence -> exact steady-Euler shear
    w_in = frequency * w_in
    b_in = weight_scale * rng.standard_normal(hidden)
    w_out = (weight_scale * rng.standard_normal((1, hidden))).tolist()
    layers: list[Layer] = [
        (w_in.tolist(), b_in.tolist(), "tanh"),
        (w_out, [0.0], None),
    ]
    return tuple(layers)


def shear_streamfunction(
    *,
    hidden: int = 6,
    seed: int = 0,
    weight_scale: float = 0.7,
    frequency: float = 1.0,
    domain_length: float = _TWO_PI,
) -> StreamfunctionField:
    r"""A ``y``-only ``tanh`` streamfunction: an exact steady-Euler shear flow.

    Because ``\psi`` depends on ``y`` alone the induced velocity is a parallel
    shear ``u = (\psi'(y), 0)`` and the steady-Euler vorticity-transport residual
    is identically zero -- which the certificate enclosure confirms to machine
    precision over the whole domain.
    """
    layers = _mlp_streamfunction_layers(
        hidden=hidden, seed=seed, weight_scale=weight_scale, y_only=True, frequency=frequency
    )
    domain = ((0.0, float(domain_length)), (0.0, float(domain_length)))
    descriptor: dict[str, Any] = {
        "name": "shear_streamfunction",
        "kind": "shear",
        "streamfunction_cage": True,
        "y_only": True,
        "domain": [list(axis) for axis in domain],
        "layers": _encode_layers(layers),
    }
    return StreamfunctionField(layers, domain, descriptor, "y-only tanh shear streamfunction")


def cellular_streamfunction(
    *,
    hidden: int = 6,
    seed: int = 1,
    weight_scale: float = 0.5,
    frequency: float = 1.0,
    domain_length: float = _TWO_PI,
) -> StreamfunctionField:
    r"""A general ``\psi(x,y)`` ``tanh`` streamfunction (a learned-style cellular flow).

    Divergence free by the cage, but **not** an exact steady state: the
    certificate reports a rigorous, finite whole-domain bound on its
    vorticity-transport residual that tightens as ``splits`` grows.
    """
    layers = _mlp_streamfunction_layers(
        hidden=hidden, seed=seed, weight_scale=weight_scale, y_only=False, frequency=frequency
    )
    domain = ((0.0, float(domain_length)), (0.0, float(domain_length)))
    descriptor = {
        "name": "cellular_streamfunction",
        "kind": "cellular",
        "streamfunction_cage": True,
        "y_only": False,
        "domain": [list(axis) for axis in domain],
        "layers": _encode_layers(layers),
    }
    return StreamfunctionField(layers, domain, descriptor, "general tanh cellular streamfunction")


def streamfunction_from_descriptor(descriptor: dict[str, Any]) -> StreamfunctionField:
    """Rebuild a :class:`StreamfunctionField` from its JSON-native descriptor."""
    layers = _decode_layers(list(descriptor["layers"]))
    domain = tuple((float(a), float(b)) for a, b in descriptor["domain"])
    return StreamfunctionField(layers, domain, dict(descriptor), str(descriptor.get("name", "")))


# --------------------------------------------------------------------------- #
# Certificate.
# --------------------------------------------------------------------------- #
def certified_streamfunction_residual(
    field: StreamfunctionField,
    *,
    viscosity: float = 0.0,
    splits: int = 4,
    residual_tol: float = 1e-8,
    notes: str = "",
) -> dict[str, Any]:
    r"""Seal a rigorous interval vorticity-transport residual certificate.

    Parameters
    ----------
    field
        A :class:`StreamfunctionField` (its velocity is divergence free by the
        streamfunction cage).
    viscosity
        Kinematic ``\nu``.  ``0`` (the default) certifies the steady-Euler
        residual ``(u\cdot\nabla)\omega``; a positive value adds the viscous term
        ``-\nu\Delta\omega`` (the forcing-free steady-NS residual).
    splits
        Per-axis sub-box count for the certified enclosure (more = tighter).
    residual_tol
        Tolerance gating ``exact_steady_euler_claim``.

    Returns
    -------
    dict
        A v1 :func:`~omnibias.core.proof.certificate.make_certificate` certificate
        (schema version ``navier-stokes-streamfunction-residual-1``), digest
        sealed and tamper evident.
    """
    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive")
    if splits < 1:
        raise ValueError("splits must be >= 1")
    if viscosity < 0.0:
        raise ValueError("viscosity must be non-negative")

    residual = certified_vorticity_transport_residual(
        field.layers, field.domain, viscosity=viscosity, splits=splits
    )
    divergence = certified_streamfunction_divergence(field.layers, field.domain, splits=splits)
    residual_sup = float(residual.mag)
    divergence_sup = float(divergence.mag)

    is_euler = viscosity == 0.0
    exact_steady_euler_claim = bool(
        is_euler and residual_sup <= residual_tol and divergence_sup <= residual_tol
    )

    cage = structural_invariant(
        "incompressibility",
        "div(grad_perp psi) = psi_xy - psi_yx = 0",
        assumptions=("equality of mixed partials of the streamfunction MLP",),
        method="streamfunction_cage",
    )

    payload: dict[str, Any] = {
        "type": "navier_stokes_streamfunction_residual",
        "schema_version": NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION,
        "observable": "incompressible_steady_vorticity_transport_residual",
        "model": "incompressible_euler" if is_euler else "incompressible_navier_stokes",
        "verification_method": "verified_interval_jet_enclosure",
        "dimension": 2,
        "domain": [list(axis) for axis in field.domain],
        "viscosity": float(viscosity),
        "splits": int(splits),
        "jet_order": 3 if is_euler else 4,
        "residual_tol": float(residual_tol),
        "residual_enclosure": [residual.lo, residual.hi],
        "residual_sup": residual_sup,
        "divergence_enclosure": [divergence.lo, divergence.hi],
        "divergence_sup": divergence_sup,
        "exact_steady_euler_claim": exact_steady_euler_claim,
        "invariants": [cage.to_payload()],
        "streamfunction": field.descriptor,
        "criterion": (
            "the steady vorticity-transport residual of the streamfunction-induced "
            "incompressible velocity is rigorously enclosed over the whole domain "
            "(between grid nodes) by the verified interval jet; an exact-steady-Euler "
            "claim additionally requires the residual and divergence sups <= residual_tol"
        ),
        "theorem_dependency": (
            "omnibias.core.verified.jet_mv.certified_partials (sound interval enclosure "
            "of every mixed partial over the box) + outward-rounded interval arithmetic"
        ),
        "open_obligations": [
            "continuum_navier_stokes_regularity_is_out_of_scope",
            "three_dimensional_streamfunction_cage_not_implemented (use beltrami_abc_flow)",
            "high_reynolds_turbulence_and_long_horizon_chaos_tracking_out_of_scope",
        ],
    }

    honesty = {
        "unproven_claim": False,
        "continuum_navier_stokes_claim": False,
        "chaotic_tracking_claim": False,
        "perfect_weather_claim": False,
        "turbulence_closure_claim": False,
        "interval_verified": True,
        "incompressible_by_construction": True,
        "whole_domain_enclosure": True,
        "exact_steady_euler_claim": exact_steady_euler_claim,
    }
    meta = {
        "harness": "omnibias.pinn.certified.fluid_rigorous.certified_streamfunction_residual",
        "field": field.label,
        "notes": str(notes),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    return make_certificate(claim=str(payload["criterion"]), payload=payload, honesty=honesty, meta=meta)


def certified_shear_streamfunction_residual(
    *,
    hidden: int = 6,
    seed: int = 0,
    frequency: float = 1.0,
    splits: int = 4,
    residual_tol: float = 1e-8,
) -> dict[str, Any]:
    """Certify a ``y``-only ``tanh`` streamfunction as an exact steady-Euler shear."""
    field = shear_streamfunction(hidden=hidden, seed=seed, frequency=frequency)
    return certified_streamfunction_residual(
        field, viscosity=0.0, splits=splits, residual_tol=residual_tol
    )


# --------------------------------------------------------------------------- #
# Schema gate.
# --------------------------------------------------------------------------- #
def streamfunction_residual_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-streamfunction-residual-1`` v1 certificate."""
    errors = schema_errors_v1(cert)
    payload = cert.get("payload")
    if not isinstance(payload, dict):
        return errors + ["payload must be a mapping"]
    if payload.get("schema_version") != NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION:
        errors.append(f"payload.schema_version must be {NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION!r}")
    if payload.get("type") != "navier_stokes_streamfunction_residual":
        errors.append("payload.type must be 'navier_stokes_streamfunction_residual'")

    for key in (
        "residual_sup",
        "divergence_sup",
        "residual_tol",
        "viscosity",
        "splits",
        "domain",
        "streamfunction",
        "residual_enclosure",
        "divergence_enclosure",
    ):
        if key not in payload:
            errors.append(f"payload missing required field {key!r}")

    for key in ("residual_sup", "divergence_sup"):
        val = payload.get(key)
        if not isinstance(val, int | float) or float(val) < 0.0 or not np.isfinite(float(val)):
            errors.append(f"payload.{key} must be a finite non-negative number")

    tol = payload.get("residual_tol")
    if not isinstance(tol, int | float) or float(tol) <= 0.0:
        errors.append("payload.residual_tol must be a positive number")

    honesty = cert.get("honesty", {})
    if not isinstance(honesty, dict):
        errors.append("honesty must be a mapping")
        honesty = {}
    for flag in _FORBIDDEN_CLAIMS:
        if honesty.get(flag, False):
            errors.append(f"honesty.{flag} must be False")
    if not honesty.get("interval_verified", False):
        errors.append("honesty.interval_verified must be True for the interval certificate")

    if payload.get("exact_steady_euler_claim", False):
        if float(payload.get("viscosity", 1.0)) != 0.0:
            errors.append("exact_steady_euler_claim requires viscosity == 0")
        if isinstance(tol, int | float):
            for key in ("residual_sup", "divergence_sup"):
                val = payload.get(key)
                if isinstance(val, int | float) and float(val) > float(tol):
                    errors.append(
                        f"exact_steady_euler_claim requires {key} <= residual_tol "
                        f"({float(val):.3e} > {float(tol):.3e})"
                    )

    streamfunction = payload.get("streamfunction")
    if not isinstance(streamfunction, dict) or "layers" not in streamfunction:
        errors.append("payload.streamfunction must carry the regenerable 'layers'")
    return errors


# --------------------------------------------------------------------------- #
# Proof-machine prover (kind: ``navier_stokes_streamfunction_residual``)
# --------------------------------------------------------------------------- #
def _blocked(detail: str) -> ProofAttempt:
    return ProofAttempt(status="BLOCKED", certificate=None, obligations=(detail,), detail=detail)


def _certificate_from_data(data: dict[str, Any]) -> dict[str, Any]:
    cert = data.get("certificate")
    if isinstance(cert, dict):
        return cert
    residual_tol = float(data.get("residual_tol", 1e-8))
    splits = int(data.get("splits", 4))
    viscosity = float(data.get("viscosity", 0.0))
    descriptor = data.get("streamfunction")
    if isinstance(descriptor, dict):
        field = streamfunction_from_descriptor(descriptor)
    else:
        kind = data.get("kind", "shear")
        if kind == "cellular":
            field = cellular_streamfunction(seed=int(data.get("seed", 1)))
        elif kind == "shear":
            field = shear_streamfunction(seed=int(data.get("seed", 0)))
        else:
            raise ValueError(f"unknown streamfunction kind: {kind!r}")
    return certified_streamfunction_residual(
        field, viscosity=viscosity, splits=splits, residual_tol=residual_tol
    )


def prove_streamfunction_residual(conjecture: Conjecture) -> ProofAttempt:
    """Adjudicate a rigorous interval streamfunction residual certificate."""
    try:
        cert = _certificate_from_data(dict(conjecture.data))
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked(f"could not build streamfunction residual certificate: {exc}")

    payload = cert.get("payload", {})
    if not isinstance(payload, dict):
        return ProofAttempt(status="BLOCKED", certificate=cert, obligations=("payload is invalid",))
    threshold = conjecture.data.get("residual_tolerance")
    if threshold is None:
        threshold = payload.get("residual_tol", 1e-8)
    residual_sup = float(payload.get("residual_sup", float("inf")))
    if residual_sup > float(threshold):
        return ProofAttempt(
            status="BLOCKED",
            certificate=cert,
            obligations=(
                f"certified residual enclosure {residual_sup:.3e} exceeds tolerance "
                f"{float(threshold):.3e}",
            ),
            detail="whole-domain residual enclosure is finite but above tolerance",
        )
    return ProofAttempt(
        status="PROVED",
        certificate=cert,
        detail=(
            "steady incompressible vorticity-transport residual rigorously enclosed "
            "at or below tolerance over the whole domain (verified interval jet)"
        ),
    )


def streamfunction_residual_proof_schema_errors(cert: dict[str, Any]) -> list[str]:
    return streamfunction_residual_schema_errors(cert)


def replay_streamfunction_residual(cert: dict[str, Any]) -> bool | None:
    """Independent numpy finite-difference twin (``None`` if omnibias-symbolic absent)."""
    try:
        from omnibias.symbolic.fluid import verify_streamfunction_residual
    except ImportError:
        return None
    report = verify_streamfunction_residual(cert)
    return bool(report["replay_match"])


__all__ = [
    "NS_STREAMFUNCTION_RESIDUAL_SCHEMA_VERSION",
    "StreamfunctionField",
    "cellular_streamfunction",
    "certified_shear_streamfunction_residual",
    "certified_streamfunction_residual",
    "prove_streamfunction_residual",
    "replay_streamfunction_residual",
    "shear_streamfunction",
    "streamfunction_from_descriptor",
    "streamfunction_residual_proof_schema_errors",
    "streamfunction_residual_schema_errors",
    "verify_certificate_digest",
]

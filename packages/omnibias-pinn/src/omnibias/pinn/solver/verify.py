# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Optional certified mode: seal an a-posteriori error bound for a solved field.

This is **one optional adapter**, not the package's purpose. It wraps
:func:`omnibias.core.verified.pde_certificate.aposteriori_error_certificate`
for a *solved* one-layer omnibias field on the **linear, steady** canonical
problems (Poisson / Helmholtz / screened-Poisson / steady advection-diffusion),
producing a sealed, digest-verifiable certificate whose ``honesty.unproven_claim``
is **always ``False``** (the core seals it that way; this module never overrides
it, see :func:`assert_certificate_is_honest`).

What is and is not certified (honest labelling)
-----------------------------------------------
* The interior residual ``sup_Omega |L u_NN - f|`` and the boundary mismatch
  ``sup_{dOmega} |u_NN - g|`` are computed **rigorously** (interval / certified
  jet) from the trained weights -- this is a *verified* enclosure, not autodiff
  and not sampling.
* The final bound ``||u_NN - u_true||_inf <= C_Omega R_int + C_dOmega R_bnd``
  additionally needs the well-posedness **stability constants**
  ``(C_Omega, C_dOmega)``. Those are a mathematical proof obligation the caller
  supplies; they are recorded (not invented) in the certificate.
* Scope is deliberately **modest-scale, linear, steady**. Nonlinear-system
  certification is hard and per-problem and is intentionally NOT offered here.

The single genuinely reusable piece is :func:`extract_layers`: the
backend-dispatched extraction of a one-layer torch/jax field into the verified
``Layer`` format.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from omnibias.pinn.solver._core.domain import Domain

if TYPE_CHECKING:
    from omnibias.core.verified.jet_mv import Layer
    from omnibias.core.verified.pde_certificate import (
        LinearPDE,
        PINNErrorCertificate,
    )

#: A boundary / source target: a constant or a ``coords -> value`` callable.
Target = float | Callable[[Any], Any]


# --------------------------------------------------------------------------- #
# Backend-dispatched weight extraction (the reusable core of this module).
# --------------------------------------------------------------------------- #
def _to_float_matrix(arr: Any) -> list[list[float]]:
    import numpy as np

    return np.asarray(arr, dtype=float).tolist()


def _to_float_vector(arr: Any) -> list[float]:
    import numpy as np

    return np.asarray(arr, dtype=float).reshape(-1).tolist()


def _backend_of(field: Any) -> str:
    """Report ``"torch"`` / ``"jax"`` (else ``"unknown"``) for a field object."""
    parts = set(type(field).__module__.split("."))
    if "torch" in parts:
        return "torch"
    if "jax" in parts:
        return "jax"
    return "unknown"


def _raw_params(field: Any) -> tuple[Any, Any, Any, Any, str]:
    """Return ``(W, beta, c, b, activation_name)`` for a one-layer torch/jax field.

    Dispatch is structural: the torch field stores its hidden map as an
    ``nn.Linear`` (``field.W.weight`` / ``.bias``); the jax field stores plain
    arrays (``field.W`` / ``field.beta`` / ``field.c`` / ``field.b``).
    """
    if getattr(field, "_omnibias_dispatch", None) != "one_layer":
        raise TypeError(
            "certified mode supports the one-layer omnibias field ansatz; got "
            f"{type(field).__name__!r}"
        )
    activation = field.spec.name
    hidden_map = field.W
    if hasattr(hidden_map, "weight"):  # torch nn.Linear
        w = hidden_map.weight.detach().cpu().numpy()
        beta = hidden_map.bias.detach().cpu().numpy()
        c = field.c.weight.detach().cpu().numpy()
        b = field.c.bias.detach().cpu().numpy()
    else:  # jax arrays
        w, beta, c, b = field.W, field.beta, field.c, field.b
    return w, beta, c, b, activation


def extract_layers(field: Any) -> list[Layer]:
    r"""Extract the verified 2-layer ``Layer`` MLP from a one-layer omnibias field.

    ``f_c(x) = b_c + sum_h c[c,h] sigma(W[h,:] . x + beta[h])`` is exactly the
    two-layer MLP ``[(W, beta, activation), (c, b, None)]`` in the
    :data:`omnibias.core.verified.jet_mv.Layer` format (hidden activation then a
    pure affine readout). Works for both the torch and jax backends.
    """
    w, beta, c, b, activation = _raw_params(field)
    hidden: Layer = (_to_float_matrix(w), _to_float_vector(beta), activation)
    readout: Layer = (_to_float_matrix(c), _to_float_vector(b), None)
    return [hidden, readout]


# --------------------------------------------------------------------------- #
# Domain / boundary geometry for the steady BVP.
# --------------------------------------------------------------------------- #
def spatial_box(domain: Domain) -> list[tuple[float, float]]:
    """The (steady) spatial bounding box; rejects a time axis."""
    if domain.is_time_dependent:
        raise ValueError(
            "certified mode targets steady problems; the domain has a time axis. "
            "Certify a steady/BVP System (e.g. poisson) instead."
        )
    return [domain.bound(ax) for ax in domain.axes]


def boundary_faces(domain: Domain, target: Target = 0.0) -> list[Any]:
    """Every non-periodic spatial boundary face as a certified ``BoundaryFace``.

    A face pins one axis to a bound (a degenerate interval) while the others span
    their box. ``target`` (``g``) is a constant or a ``coords -> value`` callable
    evaluated on each sub-box's interval coordinates.
    """
    from omnibias.core.verified.pde_certificate import BoundaryFace

    box = spatial_box(domain)
    cs = domain.coordinate_spec
    faces: list[Any] = []
    for ai, axis in enumerate(domain.axes):
        if cs.is_periodic(axis):
            continue
        lo, hi = box[ai]
        for value in (lo, hi):
            face_box = [
                (value, value) if j == ai else box[j] for j in range(len(box))
            ]
            faces.append(BoundaryFace(box=face_box, target=target))
    return faces


# --------------------------------------------------------------------------- #
# Certificate drivers (linear, steady).
# --------------------------------------------------------------------------- #
def certify_linear_bvp(
    solution: Any,
    pde: LinearPDE,
    *,
    boundary: Target = 0.0,
    stability_interior: float = 1.0,
    stability_boundary: float = 1.0,
    interior_splits: int = 2,
    boundary_splits: int = 2,
    max_error: float | None = None,
    stability_source: str = "user_supplied",
    stability_assumptions: Sequence[str] = (),
) -> PINNErrorCertificate:
    r"""Seal an a-posteriori error certificate for a solved linear steady BVP.

    ``solution`` is a :class:`FieldSolution` from either backend's
    :func:`solve_least_squares`; ``pde`` is the matching verified
    :class:`~omnibias.core.verified.pde_certificate.LinearPDE` (e.g.
    ``pde_certificate.poisson(dim, source)``). The stability constants are the
    caller's well-posedness obligation and are recorded honestly in the sealed
    certificate. Returns the core's
    :class:`~omnibias.core.verified.pde_certificate.PINNErrorCertificate`
    (``.certificate`` is sealed with ``unproven_claim: False``).
    """
    from omnibias.core.verified.pde_certificate import (
        aposteriori_error_certificate,
        user_stability_estimate,
    )

    system = solution.system
    layers = extract_layers(solution.field)
    box = spatial_box(system.domain)
    faces = boundary_faces(system.domain, boundary)
    estimate = user_stability_estimate(
        stability_interior,
        stability_boundary,
        source=stability_source,
        pde_family="linear_steady",
        domain=repr(system.domain.axes),
        assumptions=tuple(stability_assumptions),
    )
    cert = aposteriori_error_certificate(
        layers,
        box,
        pde,
        boundary=faces,
        stability=estimate,
        model_metadata={
            "ansatz": "one_layer",
            "backend": _backend_of(solution.field),
            "system": system.name or "unnamed",
            "hidden": int(getattr(solution.field, "hidden", 0)),
            "activation": solution.field.spec.name,
        },
        max_error=max_error,
        splits=interior_splits,
        boundary_splits=boundary_splits,
    )
    assert_certificate_is_honest(cert)
    return cert


def certify_poisson(
    solution: Any,
    *,
    source: float = 0.0,
    boundary: Target = 0.0,
    stability_interior: float = 1.0,
    stability_boundary: float = 1.0,
    interior_splits: int = 2,
    boundary_splits: int = 2,
    max_error: float | None = None,
) -> PINNErrorCertificate:
    r"""Certify a solved Poisson field ``Delta u = source`` (constant source).

    Convenience wrapper over :func:`certify_linear_bvp` that builds the verified
    Laplace/Poisson operator over the solution's spatial dimension. ``source``
    must be constant here; use :func:`certify_linear_bvp` with a custom
    :class:`LinearPDE` for variable-coefficient / interval-callback sources.
    """
    from omnibias.core.verified.pde_certificate import poisson as _verified_poisson

    dim = solution.system.domain.ndim
    return certify_linear_bvp(
        solution,
        _verified_poisson(dim, float(source)),
        boundary=boundary,
        stability_interior=stability_interior,
        stability_boundary=stability_boundary,
        interior_splits=interior_splits,
        boundary_splits=boundary_splits,
        max_error=max_error,
    )


def assert_certificate_is_honest(cert: PINNErrorCertificate) -> None:
    """Guard: the sealed certificate must never assert a global-regularity / continuum claim.

    Mirrors :func:`omnibias.pinn.solver._core.honesty.assert_no_unproven_claim`, applied to
    the *sealed* certificate (a ``dict``) the verified core returns.
    """
    from omnibias.pinn.solver._core.honesty import assert_no_unproven_claim

    honesty = dict(cert.certificate.get("honesty", {}))
    assert_no_unproven_claim(honesty)
    if bool(honesty.get("continuum_claim", False)):
        raise ValueError("certified PDE mode must not assert a continuum claim")


__all__ = [
    "Target",
    "assert_certificate_is_honest",
    "boundary_faces",
    "certify_linear_bvp",
    "certify_poisson",
    "extract_layers",
    "spatial_box",
]

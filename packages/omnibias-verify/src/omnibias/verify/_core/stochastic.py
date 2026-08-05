# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Verified stochastic layer: rigorous Fokker-Planck / Ito operator residuals.

The differentiable :mod:`omnibias.score` package composes the SDE operators from
the closed-form field primitives; this module gives them a **rigorous register**,
mirroring the PDE-certificate pipeline
(:func:`omnibias.verify._core.jet_ingest.certify_pinn_aposteriori`).

For an Ito diffusion ``dX = b(X) dt + sigma(X) dW`` with ``a = sigma sigma^T``:

* the **generator** (backward Kolmogorov operator) ``L f = b . grad f + 1/2 a_ij d_i d_j f``;
* the **Fokker-Planck adjoint** (forward operator, spatially-constant ``a``)
  ``L* p = -(div(b) p + b . grad p) + 1/2 a_ij d_i d_j p``.

Given the density / test-function network in the certified-jet
:data:`~omnibias.core.verified.jet_mv.Layer` format, the multivariate certified jet
(:func:`~omnibias.core.verified.jet_mv.certified_partials`) encloses every partial up
to order 2 over a spatial box, so the operator residual is a rigorous interval that
holds for **every** point of the box (not just the grid nodes) -- computed via the
:func:`~omnibias.core.verified.pde_certificate.certified_custom_residual` callback and
sealed into a tamper-evident :class:`~omnibias.core.proof.certificate.Cert`.

Honest scope: this bounds the residual of a *given* network's SDE operator over a
spatial box; the drift ``b``, its divergence ``div b`` and the diffusion ``a`` are the
caller's (analytic) inputs, exactly as in
:func:`omnibias.score.torch.ops.sde.fokker_planck`. It is a rigorous **local**
(finite-box) enclosure -- not a global or open-problem claim, and the diffusion is taken
spatially constant for the Fokker-Planck adjoint (the common constant-noise case, e.g.
Ornstein-Uhlenbeck).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnibias.core.multi_index import MultiIndex
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import BoxLike, Layer
from omnibias.core.verified.pde_certificate import Coeff, certified_custom_residual

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

#: The lowest jet order the SDE operators need (drift is first-order, diffusion second).
_ORDER = 2


def _coeff(c: Coeff, box: Sequence[Interval]) -> Interval:
    """Evaluate a constant / ``box -> value`` coefficient to an :class:`Interval`."""
    if callable(c):
        return Interval.from_value(c(box))
    return Interval.from_value(c)


def _unit(i: int, dim: int) -> MultiIndex:
    return tuple(1 if j == i else 0 for j in range(dim))


def _second(i: int, j: int, dim: int) -> MultiIndex:
    """Multi-index of the mixed second partial ``d_i d_j`` (``2`` at ``i`` when ``i == j``)."""
    idx = [0] * dim
    idx[i] += 1
    idx[j] += 1
    return tuple(idx)


def _normalize_splits(splits: int | Sequence[int], dim: int) -> list[int]:
    if isinstance(splits, int):
        return [splits] * dim
    out = list(splits)
    if len(out) != dim:
        raise ValueError(f"splits must have one entry per axis ({dim}), got {len(out)}")
    return out


def _check_coefficients(
    drift: Sequence[Coeff], diffusion: Sequence[Sequence[Coeff]], dim: int
) -> None:
    if len(drift) != dim:
        raise ValueError(f"drift must have {dim} components (one per axis), got {len(drift)}")
    if len(diffusion) != dim:
        raise ValueError(f"diffusion must be a {dim}x{dim} matrix, got {len(diffusion)} rows")
    for row in diffusion:
        if len(row) != dim:
            raise ValueError(f"diffusion must be a {dim}x{dim} matrix, got a row of length {len(row)}")


def _diffusion_quadratic(
    diffusion: Sequence[Sequence[Coeff]],
    box: Sequence[Interval],
    partials: Mapping[MultiIndex, Sequence[Interval]],
    component: int,
    dim: int,
) -> Interval:
    r"""The contraction ``sum_ij a_ij d_i d_j u`` over the box (``a`` may vary with ``x``)."""
    acc = Interval.point(0.0)
    for i in range(dim):
        for j in range(dim):
            a_ij = _coeff(diffusion[i][j], box)
            acc = acc + a_ij * partials[_second(i, j, dim)][component]
    return acc


# --------------------------------------------------------------------------- #
# Rigorous operator-residual enclosures.
# --------------------------------------------------------------------------- #
def certified_fokker_planck_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    drift: Sequence[Coeff],
    diffusion: Sequence[Sequence[Coeff]],
    drift_divergence: Coeff,
    component: int = 0,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Enclosure of the Fokker-Planck adjoint residual over the box.

    Rigorously encloses ``{ L* p(x) : x in domain }`` with

    .. math::

        L^* p = -\big((\nabla\cdot b)\,p + b\cdot\nabla p\big)
                + \tfrac12\,a_{ij}\,\partial_i\partial_j p,

    the constant-diffusion Fokker-Planck adjoint (the rigorous twin of
    :func:`omnibias.score.torch.ops.sde.fokker_planck`).  ``p`` is the density network
    (:data:`~omnibias.core.verified.jet_mv.Layer` format); ``drift`` is ``b`` (one
    :data:`~omnibias.core.verified.pde_certificate.Coeff` per axis), ``drift_divergence``
    the analytic ``div b``, and ``diffusion`` the ``d x d`` matrix ``a``.  For a true
    stationary density the residual encloses ``0``; ``.mag`` is a certified sup-norm bound
    that tightens as ``splits`` grows.
    """
    dim = len(list(domain))
    _check_coefficients(drift, diffusion, dim)

    def residual(
        box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        p = partials[(0,) * dim][component]
        transport = _coeff(drift_divergence, box) * p
        for i in range(dim):
            transport = transport + _coeff(drift[i], box) * partials[_unit(i, dim)][component]
        diff = _diffusion_quadratic(diffusion, box, partials, component, dim)
        return (Interval.point(0.0) - transport) + Interval.point(0.5) * diff

    return certified_custom_residual(layers, domain, residual, order=_ORDER, splits=splits)


def certified_ito_generator_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    drift: Sequence[Coeff],
    diffusion: Sequence[Sequence[Coeff]],
    source: Coeff = 0.0,
    reaction: Coeff = 0.0,
    component: int = 0,
    splits: int | Sequence[int] = 1,
) -> Interval:
    r"""Enclosure of the Ito-generator residual over the box.

    Rigorously encloses ``{ L f(x) + c(x) f(x) - g(x) : x in domain }`` with the
    backward (Kolmogorov) generator

    .. math::

        L f = b\cdot\nabla f + \tfrac12\,a_{ij}\,\partial_i\partial_j f

    (the rigorous twin of :func:`omnibias.score.torch.ops.sde.ito_generator`).  The
    optional ``reaction`` ``c`` and ``source`` ``g`` express a generator equation
    ``L f + c f = g`` -- e.g. ``reaction = -lambda`` for the eigen-residual
    ``L f - lambda f``, or ``source`` for a Feynman-Kac right-hand side.  ``.mag`` is a
    certified sup-norm bound on the residual.
    """
    dim = len(list(domain))
    _check_coefficients(drift, diffusion, dim)

    def residual(
        box: Sequence[Interval], partials: Mapping[MultiIndex, Sequence[Interval]]
    ) -> Interval:
        f = partials[(0,) * dim][component]
        transport = Interval.point(0.0)
        for i in range(dim):
            transport = transport + _coeff(drift[i], box) * partials[_unit(i, dim)][component]
        diff = _diffusion_quadratic(diffusion, box, partials, component, dim)
        generator = transport + Interval.point(0.5) * diff
        return generator + _coeff(reaction, box) * f - _coeff(source, box)

    return certified_custom_residual(layers, domain, residual, order=_ORDER, splits=splits)


# --------------------------------------------------------------------------- #
# Sealed certificate.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StochasticResidualCertificate:
    """A sealed rigorous sup-norm bound on an SDE operator residual over a box."""

    operator: str
    residual: Interval
    residual_sup: float
    certificate: Cert

    @property
    def verified(self) -> bool:
        """``True`` iff the sealed certificate's digest is intact (tamper-evidence)."""
        from omnibias.core.proof.certificate import verify_certificate_digest

        return verify_certificate_digest(self.certificate)


def _constant_matrix(diffusion: Sequence[Sequence[Coeff]]) -> list[list[float]] | None:
    """The diffusion matrix as floats when every entry is a constant, else ``None``."""
    out: list[list[float]] = []
    for row in diffusion:
        frow: list[float] = []
        for entry in row:
            if callable(entry):
                return None
            frow.append(float(Interval.from_value(entry).mid))
        out.append(frow)
    return out


def _layer_metadata(layers: Sequence[Layer]) -> dict[str, Any]:
    """Deterministic, JSON-safe fingerprint of the certified-jet layers."""
    import hashlib
    import json

    activations = [name for _w, _b, name in layers if name is not None]

    def _f(x: Any) -> float:
        return float(x.mid) if hasattr(x, "mid") else float(x)

    canonical = [
        [
            [[_f(v) for v in row] for row in weight],
            None if bias is None else [_f(v) for v in bias],
            name,
        ]
        for weight, bias, name in layers
    ]
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "n_layers": len(list(layers)),
        "activations": activations,
        "layers_digest": "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def _seal_stochastic_residual(
    *,
    operator: str,
    claim: str,
    residual: Interval,
    dim: int,
    splits: list[int],
    diffusion: Sequence[Sequence[Coeff]],
    extra_payload: Mapping[str, Any],
    max_residual: float | None,
    model_metadata: Mapping[str, Any] | None,
    provenance: Mapping[str, Any] | None,
    layers: Sequence[Layer],
) -> Cert:
    from omnibias.core.proof.certificate import make_certificate

    payload: dict[str, Any] = {
        "type": "stochastic_operator_residual",
        "operator": operator,
        "scope": "local_box",  # a finite-box enclosure, never a global / continuum claim
        "residual_sup": residual.mag,
        "residual_enclosure": [residual.lo, residual.hi],
        "spatial_dim": dim,
        "order": _ORDER,
        "splits": list(splits),
        "diffusion_constant": _constant_matrix(diffusion),
        "model": dict(model_metadata) if model_metadata is not None else _layer_metadata(layers),
    }
    payload.update(dict(extra_payload))
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if max_residual is not None:
        if max_residual < 0.0:
            raise ValueError("max_residual must be non-negative")
        margin = Interval.point(float(max_residual)) - Interval.point(residual.mag)
        payload["finite_obligation"] = {
            "type": "residual_sup_le_threshold",
            "threshold": float(max_residual),
            "margin": [margin.lo, margin.hi],
            "note": (
                "Lean can only check this finite numerical inequality; it does not "
                "formalize the analytic SDE / Fokker-Planck theory."
            ),
        }
    return make_certificate(
        claim=claim,
        payload=payload,
        honesty={
            "unproven_claim": False,
            "continuum_claim": False,
            "interval_verified": True,
        },
    )


def certify_fokker_planck_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    drift: Sequence[Coeff],
    diffusion: Sequence[Sequence[Coeff]],
    drift_divergence: Coeff,
    component: int = 0,
    splits: int | Sequence[int] = 1,
    max_residual: float | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> StochasticResidualCertificate:
    r"""Seal a rigorous sup-norm bound on the Fokker-Planck adjoint residual.

    Encloses ``L* p`` over ``domain`` with :func:`certified_fokker_planck_residual` and
    seals the ``[lo, hi]`` residual enclosure into a tamper-evident certificate.  When a
    ``div b p_inf = 1/2 a p_inf''`` stationary density is supplied the sealed sup-norm is
    (up to interval width) ``0``.  Pass ``max_residual`` to attach a finite ``residual_sup
    <= threshold`` obligation.
    """
    dim = len(list(domain))
    residual = certified_fokker_planck_residual(
        layers,
        domain,
        drift=drift,
        diffusion=diffusion,
        drift_divergence=drift_divergence,
        component=component,
        splits=splits,
    )
    cert = _seal_stochastic_residual(
        operator="fokker_planck",
        claim=(
            "rigorous sup-norm bound on the Fokker-Planck residual "
            "L* p = -(div(b) p + b . grad p) + 1/2 a_ij d_i d_j p over the spatial box"
        ),
        residual=residual,
        dim=dim,
        splits=_normalize_splits(splits, dim),
        diffusion=diffusion,
        extra_payload={"component": int(component)},
        max_residual=max_residual,
        model_metadata=model_metadata,
        provenance=provenance,
        layers=layers,
    )
    return StochasticResidualCertificate("fokker_planck", residual, residual.mag, cert)


def certify_ito_generator_residual(
    layers: Sequence[Layer],
    domain: BoxLike,
    *,
    drift: Sequence[Coeff],
    diffusion: Sequence[Sequence[Coeff]],
    source: Coeff = 0.0,
    reaction: Coeff = 0.0,
    component: int = 0,
    splits: int | Sequence[int] = 1,
    max_residual: float | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> StochasticResidualCertificate:
    r"""Seal a rigorous sup-norm bound on the Ito-generator residual ``L f + c f - g``.

    Encloses the generator residual over ``domain`` with
    :func:`certified_ito_generator_residual` and seals it.  ``reaction`` ``c`` and
    ``source`` ``g`` are optional (default ``0``); ``max_residual`` attaches a finite
    ``residual_sup <= threshold`` obligation.
    """
    dim = len(list(domain))
    residual = certified_ito_generator_residual(
        layers,
        domain,
        drift=drift,
        diffusion=diffusion,
        source=source,
        reaction=reaction,
        component=component,
        splits=splits,
    )
    cert = _seal_stochastic_residual(
        operator="ito_generator",
        claim=(
            "rigorous sup-norm bound on the Ito-generator residual "
            "L f = b . grad f + 1/2 a_ij d_i d_j f over the spatial box"
        ),
        residual=residual,
        dim=dim,
        splits=_normalize_splits(splits, dim),
        diffusion=diffusion,
        extra_payload={"component": int(component)},
        max_residual=max_residual,
        model_metadata=model_metadata,
        provenance=provenance,
        layers=layers,
    )
    return StochasticResidualCertificate("ito_generator", residual, residual.mag, cert)


# --------------------------------------------------------------------------- #
# Schema / replay.
# --------------------------------------------------------------------------- #
def stochastic_residual_schema_errors(cert: Mapping[str, Any]) -> list[str]:
    """Validate the v1 ``stochastic_operator_residual`` certificate payload."""
    from omnibias.core.proof.certificate import schema_errors_v1

    errors = schema_errors_v1(cert)
    payload = cert.get("payload")
    if not isinstance(payload, Mapping):
        return errors + ["payload must be a mapping"]
    if payload.get("type") != "stochastic_operator_residual":
        errors.append("payload.type must be 'stochastic_operator_residual'")
    if payload.get("operator") not in ("fokker_planck", "ito_generator"):
        errors.append("payload.operator must be 'fokker_planck' or 'ito_generator'")
    if payload.get("scope") != "local_box":
        errors.append("payload.scope must be 'local_box' (finite-box enclosure)")
    for key in ("residual_sup", "residual_enclosure", "spatial_dim", "order", "splits"):
        if key not in payload:
            errors.append(f"payload missing required field {key!r}")
    try:
        if float(payload.get("residual_sup", -1.0)) < 0.0:
            errors.append("payload.residual_sup must be non-negative")
    except (TypeError, ValueError):
        errors.append("payload.residual_sup must be numeric")
    enclosure = payload.get("residual_enclosure")
    if not isinstance(enclosure, Sequence) or isinstance(enclosure, str | bytes) or len(enclosure) != 2:
        errors.append("payload.residual_enclosure must be a [lo, hi] pair")
    honesty = cert.get("honesty", {})
    if isinstance(honesty, Mapping) and bool(honesty.get("unproven_claim", False)):
        errors.append("honesty.unproven_claim must be False")
    return errors


def replay_stochastic_residual_certificate(cert: Mapping[str, Any]) -> bool:
    """Independent replay: schema, digest, and the sup-norm / finite-obligation arithmetic."""
    from omnibias.core.proof.certificate import verify_certificate_digest

    if stochastic_residual_schema_errors(cert):
        return False
    if "digest" in cert and not verify_certificate_digest(cert):
        return False
    payload = cert["payload"]
    assert isinstance(payload, Mapping)
    lo, hi = payload["residual_enclosure"]
    expected_sup = Interval(float(lo), float(hi)).mag
    if abs(expected_sup - float(payload["residual_sup"])) > max(1e-12, 1e-12 * abs(expected_sup)):
        return False
    finite = payload.get("finite_obligation")
    if finite is not None:
        if not isinstance(finite, Mapping) or finite.get("type") != "residual_sup_le_threshold":
            return False
        margin = (
            Interval.point(float(finite["threshold"])) - Interval.point(float(payload["residual_sup"]))
        )
        got = finite.get("margin")
        if not isinstance(got, Sequence) or abs(float(got[0]) - margin.lo) > 1e-12:
            return False
    return True


__all__ = [
    "StochasticResidualCertificate",
    "certified_fokker_planck_residual",
    "certified_ito_generator_residual",
    "certify_fokker_planck_residual",
    "certify_ito_generator_residual",
    "replay_stochastic_residual_certificate",
    "stochastic_residual_schema_errors",
]

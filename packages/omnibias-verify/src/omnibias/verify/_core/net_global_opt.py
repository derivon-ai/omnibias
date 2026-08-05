# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified global minimization of an omnibias network's scalar output.

This is the bridge that makes :func:`omnibias.verify.certified_minimize` first-class
for omnibias objectives: instead of hand-writing an interval extension and gradient,
you hand it a network (or its ``(W, b, name)`` layer list) and the **closed-form
verified derivative tower** supplies both.

For an input box, :func:`omnibias.core.verified.jet_mv.mlp_jet_mv` propagates a
*certified multivariate jet* through the MLP in interval arithmetic: row 0 encloses
the network output over the box (the value objective), and the unit-multi-index rows
give an **exact interval enclosure of the gradient** valid for every point of the box
(:func:`jet_gradient`) -- and, at order >= 2, the Hessian (:func:`jet_hessian`).
Those enclosures are exactly what the branch-and-bound needs for its monotonicity
test and mean-value form, so an omnibias network gets a *proof* of its global minimum
with no autodiff and no per-objective bound engineering.

The value row is the interval-bound-propagation forward pass (order-independent); the
gradient enclosure is complete at ``order = 1`` (the chain-rule coefficient only takes
the first-order Cauchy term), so ``order = 1`` is the cheap default for minimization
and ``order = 2`` is used for the strict-local-min Hessian.

Scope / honesty: this inherits the interval-B&B curse of dimensionality (exponential
worst case) *and* interval dependency overestimation that grows with box width and
network depth -- it is for **small networks over low-dimensional input boxes** (a
certified read-out over an input region), not million-parameter training. Supported
activations are those of the verified tower: ``"tanh"``, ``"sigmoid"``, ``"gaussian"``,
the smooth closed-form ``"silu"`` / ``"gelu"`` / ``"softplus"`` (and, for raw layer
lists, ``"sin"`` / ``"cos"``); ``None`` is a pure affine readout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.proof.certificate import Cert, interval_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.jet_mv import (
    Layer,
    jet_gradient,
    jet_hessian,
    mlp_jet_mv,
)
from omnibias.verify._core.global_opt import (
    Box,
    GlobalMinResult,
    certified_minimize,
    certify_strict_local_min,
)
from omnibias.verify._core.jet_ingest import verified_layer_bundle, verified_layers
from omnibias.verify._core.stationary import (
    CriticalPoint,
    FlatnessResult,
    certified_critical_points,
    certified_flatness,
)

#: A cap on the per-solve jet cache (boxes are keyed by their float bounds).
_CACHE_CAP = 256


def _coerce_layers(net_or_layers: Any) -> list[Layer]:
    """Accept either a ``JetMLP``-like network (via ``_layer_specs``) or a raw
    ``[(W, b, name), ...]`` layer list."""
    if hasattr(net_or_layers, "_layer_specs"):
        return list(verified_layers(net_or_layers))
    layers = list(net_or_layers)
    if not layers:
        raise ValueError("layer list is empty")
    return layers


class _LayerListNet:
    """Adapter exposing a raw ``[(W, b, name), ...]`` list as a ``_layer_specs()`` source."""

    def __init__(self, layers: list[Layer]) -> None:
        self._layers = layers

    def _layer_specs(self) -> list[Layer]:
        return list(self._layers)


def _layer_source(net_or_layers: Any) -> Any:
    """Return an object exposing ``_layer_specs()`` (wrap a raw layer list if needed)."""
    if hasattr(net_or_layers, "_layer_specs"):
        return net_or_layers
    return _LayerListNet(_coerce_layers(net_or_layers))


def _readout_width(layers: list[Layer]) -> int:
    weight = layers[-1][0]
    return len(weight)


class _JetCache:
    """Memoize ``mlp_jet_mv`` per box so the value and gradient share one jet.

    ``certified_minimize`` reads ``f(box)`` and ``grad(box)`` on the same box
    back-to-back; one jet answers both.  Keyed by the box's float bounds with a
    hard size cap (cleared wholesale when exceeded -- the B&B frontier only reuses
    the most recent boxes)."""

    def __init__(self, layers: list[Layer], order: int) -> None:
        self._layers = layers
        self._order = order
        self._cache: dict[tuple[tuple[float, float], ...], list[list[Interval]]] = {}

    def jet(self, box: Box) -> list[list[Interval]]:
        key = tuple((iv.lo, iv.hi) for iv in box)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if len(self._cache) >= _CACHE_CAP:
            self._cache.clear()
        j = mlp_jet_mv(box, self._layers, self._order)
        self._cache[key] = j
        return j


def certified_network_minimize(
    net_or_layers: Any,
    box: list[tuple[float, float]] | list[Interval],
    *,
    order: int = 1,
    component: int = 0,
    tol: float = 1e-6,
    max_boxes: int = 100_000,
    use_gradient: bool = True,
    second_order: bool = False,
    use_newton: bool = True,
    min_width: float = 1e-12,
    seeds: Sequence[Sequence[float]] | None = None,
) -> GlobalMinResult:
    r"""Rigorously enclose ``min_{x in box} net(x)[component]`` by interval B&B.

    ``net_or_layers`` is either a ``JetMLP``-like network (any object exposing
    ``_layer_specs()`` -- omnibias torch/jax ``JetMLP`` do) or a raw
    ``[(W, b, name), ...]`` layer list.  The closed-form verified jet supplies the
    value enclosure (jet row 0) and, when ``use_gradient`` is true, an exact
    interval **gradient** enclosure (:func:`jet_gradient`) that powers the
    monotonicity test and mean-value form.  With ``second_order=True`` the jet is
    lifted to (at least) order 2 so the exact interval **Hessian**
    (:func:`jet_hessian`) adds the second-order lower bound and, unless
    ``use_newton=False``, the interval-Newton contractor.  ``seeds`` are optional
    concrete warm-start incumbents (each clamped into the box) forwarded to
    :func:`certified_minimize`; a strong seed -- e.g. from the closed-form
    gradient-descent helpers in :mod:`omnibias.verify.torch.warm_start` /
    :mod:`omnibias.verify.jax.warm_start` -- prunes the search sooner without ever
    affecting the sound enclosure.  Returns the same sound :class:`GlobalMinResult`
    as :func:`certified_minimize`: ``f_lower <= min net <= f_upper`` unconditionally.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1 for the value+gradient jet, got {order}")
    if second_order:
        order = max(order, 2)
    layers = _coerce_layers(net_or_layers)
    dim = len(box)
    if dim < 1:
        raise ValueError("box must have at least one axis")
    width = _readout_width(layers)
    if not 0 <= component < width:
        raise ValueError(
            f"component {component} out of range for readout width {width}"
        )

    cache = _JetCache(layers, order)

    def f(b: Box) -> IntervalLike:
        return cache.jet(b)[0][component]

    def grad(b: Box) -> list[Interval]:
        rows = jet_gradient(cache.jet(b), dim, order)
        return [row[component] for row in rows]

    def hess(b: Box) -> list[list[Interval]]:
        h = jet_hessian(cache.jet(b), dim, order)
        return [[h[i][j][component] for j in range(dim)] for i in range(dim)]

    return certified_minimize(
        f,
        box,
        tol=tol,
        max_boxes=max_boxes,
        grad=grad if use_gradient else None,
        hess=hess if second_order else None,
        use_newton=use_newton,
        min_width=min_width,
        seeds=seeds,
    )


def certify_network_strict_local_min(
    net_or_layers: Any,
    box: list[tuple[float, float]] | list[Interval],
    *,
    component: int = 0,
    order: int = 2,
) -> bool:
    r"""Certify ``net(.)[component]`` is strictly convex on ``box`` (Hessian PD there).

    Uses the closed-form verified Hessian enclosure (:func:`jet_hessian`, ``order >= 2``)
    with the interval ``LDL^T`` inertia test.  ``True`` guarantees the Hessian is
    positive definite at *every* point of ``box``, so any interior stationary point is
    a strict local minimizer; pair it with a :func:`certified_network_minimize` gap to
    upgrade the incumbent to a certified strict minimizer.
    """
    if order < 2:
        raise ValueError(f"hessian needs order >= 2, got {order}")
    layers = _coerce_layers(net_or_layers)
    dim = len(box)
    width = _readout_width(layers)
    if not 0 <= component < width:
        raise ValueError(
            f"component {component} out of range for readout width {width}"
        )

    def hessian(b: Box) -> list[list[Interval]]:
        h = jet_hessian(mlp_jet_mv(b, layers, order), dim, order)
        return [[h[i][j][component] for j in range(dim)] for i in range(dim)]

    return bool(certify_strict_local_min(hessian, box))


def _net_grad_hess(
    net_or_layers: Any, dim: int, component: int, order: int
) -> tuple[list[Layer], Any, Any]:
    if order < 2:
        raise ValueError(f"hessian needs order >= 2, got {order}")
    layers = _coerce_layers(net_or_layers)
    width = _readout_width(layers)
    if not 0 <= component < width:
        raise ValueError(f"component {component} out of range for readout width {width}")
    cache = _JetCache(layers, order)

    def grad(b: Box) -> list[Interval]:
        return [row[component] for row in jet_gradient(cache.jet(b), dim, order)]

    def hess(b: Box) -> list[list[Interval]]:
        h = jet_hessian(cache.jet(b), dim, order)
        return [[h[i][j][component] for j in range(dim)] for i in range(dim)]

    return layers, grad, hess


def certified_network_critical_points(
    net_or_layers: Any,
    box: list[tuple[float, float]] | list[Interval],
    *,
    component: int = 0,
    order: int = 2,
    tol: float = 1e-8,
    max_boxes: int = 200_000,
) -> list[CriticalPoint]:
    r"""Enclose & certify every stationary point of ``net(.)[component]`` in ``box``.

    Bridges the closed-form verified gradient/Hessian jets into
    :func:`omnibias.verify.certified_critical_points`: the rigorous "solve
    ``grad net = 0``" for a small network over a low-dimensional input box, with
    per-point ``min``/``max``/``saddle`` classification and certified Hessian
    eigenvalue bounds.
    """
    dim = len(box)
    _, grad, hess = _net_grad_hess(net_or_layers, dim, component, order)
    points: list[CriticalPoint] = certified_critical_points(grad, hess, box, tol=tol, max_boxes=max_boxes)
    return points


def certified_network_flatness(
    net_or_layers: Any,
    box: list[tuple[float, float]] | list[Interval],
    *,
    component: int = 0,
    order: int = 2,
) -> FlatnessResult:
    r"""Rigorously enclose the extreme Hessian eigenvalues of ``net(.)[component]``.

    A *certified* basin-sharpness read-out over ``box`` (see
    :func:`omnibias.verify.certified_flatness`).  Honest scope: this measures local
    curvature (generalization-relevant flatness), not global optimality.
    """
    dim = len(box)
    _, _, hess = _net_grad_hess(net_or_layers, dim, component, order)
    flatness: FlatnessResult = certified_flatness(hess, box)
    return flatness


@dataclass(frozen=True)
class NetworkCertificate:
    r"""A sealed, tamper-evident proof of a *trained* network's certified read-out.

    Bundles the rigorous :class:`GlobalMinResult` (the enclosure of the network's minimum
    over an input box), an optional :class:`FlatnessResult` (certified Hessian-eigenvalue /
    basin-sharpness enclosure), an optional strict-local-min flag, and the sealed v1
    :class:`~omnibias.core.proof.certificate.Cert` (interval enclosure of the minimum, with the
    ingested-weight digest + box + convergence flag as provenance in ``meta``). :attr:`verified`
    recomputes the certificate digest, so any post-hoc edit to a bound is detected.
    """

    result: GlobalMinResult
    certificate: Cert
    layers_digest: str
    flatness: FlatnessResult | None = None
    strict_local_min: bool | None = None

    @property
    def verified(self) -> bool:
        """``True`` iff the sealed certificate's digest matches its body (untampered)."""
        return verify_certificate_digest(self.certificate)

    @property
    def converged(self) -> bool:
        """``True`` iff the certified optimality gap reached the requested tolerance."""
        return self.result.converged


def certify_trained_network(
    net: Any,
    box: list[tuple[float, float]] | list[Interval],
    *,
    component: int = 0,
    tol: float = 1e-6,
    max_boxes: int = 100_000,
    flatness: bool = False,
    strict_local_min: bool = False,
    provenance: dict[str, Any] | None = None,
) -> NetworkCertificate:
    r"""Train-then-certify convenience: certify a trained network's read-out over a box.

    Runs :func:`certified_network_minimize` (rigorous global minimum enclosure) and, when
    requested, :func:`certified_network_flatness` (certified curvature) and
    :func:`certify_network_strict_local_min`, then **seals** the minimum enclosure as a v1
    certificate whose ``meta`` records the ingested-weight digest (via
    :func:`verified_layer_bundle`), the input box, the argmin, the certified gap, and the honest
    ``converged`` flag. Returns a :class:`NetworkCertificate`; check :attr:`~NetworkCertificate.verified`
    for the digest and :attr:`~NetworkCertificate.converged` for whether the gap reached ``tol``.

    Honest scope (inherited from interval B&B): small networks over low-dimensional input boxes,
    activations ``tanh`` / ``sigmoid`` / ``gaussian`` / ``silu`` / ``gelu`` / ``softplus``. ``net``
    is a ``JetMLP``-like network (exposing ``_layer_specs()``) or a raw ``[(W, b, name), ...]``
    layer list.
    """
    box_iv: list[Interval] = [
        b if isinstance(b, Interval) else Interval(float(b[0]), float(b[1])) for b in box
    ]
    source = _layer_source(net)
    bundle = verified_layer_bundle(source, domain=box_iv, provenance=provenance)
    need_second_order = flatness or strict_local_min
    result = certified_network_minimize(
        source,
        box_iv,
        component=component,
        tol=tol,
        max_boxes=max_boxes,
        second_order=need_second_order,
    )
    flat = certified_network_flatness(source, box_iv, component=component) if flatness else None
    slm = (
        certify_network_strict_local_min(source, box_iv, component=component)
        if strict_local_min
        else None
    )

    meta: dict[str, Any] = {
        "kind": "trained_network_readout",
        "component": component,
        "box": [[iv.lo, iv.hi] for iv in box_iv],
        "argmin": list(result.x),
        "gap": result.gap,
        "tol": result.tol,
        "converged": result.converged,
        "boxes_explored": result.boxes_explored,
        "boxes_remaining": result.boxes_remaining,
        "layers_digest": bundle.metadata["layers_digest"],
        "provenance": dict(provenance) if provenance is not None else {},
    }
    if flat is not None:
        meta["flatness"] = {
            "eig_min": [flat.eig_min.lo, flat.eig_min.hi],
            "eig_max": [flat.eig_max.lo, flat.eig_max.hi],
            "certified_positive_definite": flat.certified_positive_definite,
            "sharpness": flat.sharpness,
        }
    if slm is not None:
        meta["strict_local_min"] = slm

    honesty = {"unproven_claim": False, "global_min_certified": result.converged}
    claim = f"min_{{x in box}} net(x)[{component}] is enclosed by the interval"
    cert = interval_certificate(claim, result.enclosure, honesty=honesty, meta=meta)
    return NetworkCertificate(
        result=result,
        certificate=cert,
        layers_digest=str(bundle.metadata["layers_digest"]),
        flatness=flat,
        strict_local_min=slm,
    )


__all__ = [
    "NetworkCertificate",
    "certified_network_critical_points",
    "certified_network_flatness",
    "certified_network_minimize",
    "certify_network_strict_local_min",
    "certify_trained_network",
]

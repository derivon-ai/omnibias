# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified finite-difference gradient checking for ML.

Autodiff gradients are checked in practice by an ad-hoc fixed-``eps`` finite
difference and a hand-tuned tolerance -- a test that both false-passes a subtly
wrong gradient and false-fails a correct one to float cancellation. This module
replaces that with a **proof**: for a scalar ``f: R^d -> R`` whose per-axis
restriction ``g_i(t) = f(x_1,...,t,...,x_d)`` has a sound derivative-tower /
interval-jet oracle, each true partial ``d f / d x_i`` is enclosed *rigorously*
(reusing :func:`omnibias.difference.certified_fd_error_general`), and the
per-coordinate residual band ``autodiff_grad_i - true_partial_i`` is returned as a
certified :class:`~omnibias.core.verified.interval.Interval`.

* If every residual band contains ``0`` the autodiff gradient is **accepted** (it
  is provably consistent with the true gradient to the certified tolerance).
* If a residual band is sign-definite the autodiff gradient is **rejected** with a
  *proof of mismatch* -- and :func:`gradient_residual_certificate` seals it as a
  v1 interval certificate whose finite ``enclosed_quantity_pos/neg`` obligation the
  Lean kernel can re-check (mirroring the ``omnibias-difference`` sign path).

The true-partial enclosure is the **closed-form** tower value intersected with the
finite-difference sandwich (both contain the truth); the stencil evaluation is
**numerical**. Weights can come from any backend -- :func:`mlp_axis_oracles` builds
the per-axis oracles for a one-hidden-layer scalar MLP whose parameters were
ingested from torch/jax (plain nested lists), so this drives real networks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.proof.certificate import Cert, interval_certificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.difference import DerivBound, certified_fd_error_general
from omnibias.verify._core.network import (
    LinearLayer,
    Network,
    SigmoidLayer,
    TanhLayer,
)


@dataclass(frozen=True)
class GradientCheckCertificate:
    r"""Per-coordinate certified comparison of an autodiff gradient to the truth.

    ``residuals[i]`` rigorously encloses ``autodiff_grad_i - (df/dx_i)`` and
    ``true_partials[i]`` rigorously encloses ``df/dx_i``.  ``passed`` is ``True``
    iff every residual band contains ``0`` (the gradient is provably consistent);
    a sign-definite residual is a *proof* that coordinate is wrong.
    """

    point: tuple[float, ...]
    step: float
    stencil: str
    autodiff_grad: tuple[float, ...]
    true_partials: tuple[Interval, ...]
    residuals: tuple[Interval, ...]
    label: str = "closed-form + numerical"

    @property
    def passed(self) -> bool:
        """Whether every per-coordinate residual band contains zero."""
        return all(r.contains(0.0) for r in self.residuals)

    @property
    def max_abs_residual(self) -> float:
        """The largest residual magnitude across coordinates (outward rounded)."""
        return max((r.mag for r in self.residuals), default=0.0)

    def mismatched_coordinates(self) -> tuple[int, ...]:
        """Indices whose residual band is sign-definite (a proven mismatch)."""
        return tuple(i for i, r in enumerate(self.residuals) if not r.contains(0.0))


def certified_gradient_check(
    axis_fns: Sequence[Callable[[float], float]],
    axis_deriv_bounds: Sequence[DerivBound],
    autodiff_grad: Sequence[float],
    point: Sequence[float],
    *,
    step: float = 1e-2,
    stencil: str = "central",
) -> GradientCheckCertificate:
    r"""Certify an autodiff gradient against the rigorously-enclosed true gradient.

    ``axis_fns[i]`` is the scalar restriction ``g_i(t) = f(..., x_i = t, ...)`` (a
    plain ``float`` map, for the stencil), and ``axis_deriv_bounds[i](k, box)``
    rigorously encloses ``g_i^{(k)}`` over ``box`` (the only per-axis input, so this
    works for any ``f`` with a sound directional derivative tower). For each ``i``
    the true partial is enclosed by intersecting the closed-form tower value with
    the finite-difference sandwich, and the residual band
    ``autodiff_grad_i - true_partial_i`` is returned.
    """
    d = len(point)
    if not (len(axis_fns) == len(axis_deriv_bounds) == len(autodiff_grad) == d):
        raise ValueError("axis_fns, axis_deriv_bounds, autodiff_grad and point must share length d")
    if d == 0:
        raise ValueError("gradient check needs at least one coordinate")

    true_partials: list[Interval] = []
    residuals: list[Interval] = []
    for i in range(d):
        cert = certified_fd_error_general(
            axis_fns[i], axis_deriv_bounds[i], point[i], 1, step, stencil, name=f"partial_{i}"
        )
        # Both the closed-form tower enclosure and the FD sandwich contain the truth;
        # their intersection is the tightest rigorous bracket (fall back if disjoint).
        closed = cert.enclosure
        sandwich = cert.true_derivative_interval
        lo = max(closed.lo, sandwich.lo)
        hi = min(closed.hi, sandwich.hi)
        true_partial = Interval(lo, hi) if lo <= hi else closed
        true_partials.append(true_partial)
        residuals.append(Interval.point(float(autodiff_grad[i])) - true_partial)

    return GradientCheckCertificate(
        point=tuple(float(x) for x in point),
        step=float(step),
        stencil=stencil,
        autodiff_grad=tuple(float(g) for g in autodiff_grad),
        true_partials=tuple(true_partials),
        residuals=tuple(residuals),
    )


def gradient_residual_certificate(
    gc: GradientCheckCertificate, i: int, *, meta: dict[str, Any] | None = None
) -> Cert:
    r"""Seal the coordinate-``i`` residual band as a v1 interval certificate.

    A sign-definite residual (a proven autodiff mismatch) yields a finite
    ``enclosed_quantity_pos`` / ``enclosed_quantity_neg`` obligation via
    :func:`omnibias.core.proof.lean_check.generate_obligation`; a band containing
    ``0`` (a pass) carries none -- the same documented sign-only obligation scope as
    the ``omnibias-difference`` derivative certificates.
    """
    if not 0 <= i < len(gc.residuals):
        raise ValueError(f"coordinate {i} out of range for a {len(gc.residuals)}-vector")
    residual = gc.residuals[i]
    claim = (
        f"autodiff_grad[{i}] - (df/dx_{i}) at {gc.point} is enclosed in "
        f"[{residual.lo!r}, {residual.hi!r}]"
    )
    payload_meta: dict[str, Any] = {
        "coordinate": i,
        "step": gc.step,
        "stencil": gc.stencil,
        "autodiff_grad_i": gc.autodiff_grad[i],
        "true_partial_lo": gc.true_partials[i].lo,
        "true_partial_hi": gc.true_partials[i].hi,
        "label": gc.label,
    }
    if meta:
        payload_meta.update(meta)
    return interval_certificate(claim, residual, honesty={"unproven_claim": False}, meta=payload_meta)


def mlp_axis_oracles(
    hidden_weights: Sequence[Sequence[float]],
    hidden_biases: Sequence[float],
    output_weights: Sequence[float],
    activation: str,
    point: Sequence[float],
) -> tuple[list[Callable[[float], float]], list[DerivBound]]:
    r"""Per-axis ``(g_i, g_i^{(k)}-oracle)`` for a one-hidden-layer scalar MLP.

    Builds the directional oracles for ``f(x) = sum_j v_j sigma(sum_i W_{j,i} x_i +
    b_j)`` with ``sigma = activation`` (a :func:`sigma_tower_interval` name). The
    restriction ``g_i(t) = f(..., x_i = t, ...)`` has
    ``g_i^{(k)}(t) = sum_j v_j W_{j,i}^k sigma^{(k)}(W_{j,i} t + c_j)`` with
    ``c_j = b_j + sum_{l != i} W_{j,l} x_l``, each ``sigma^{(k)}`` enclosed by the
    closed-form tower.  ``hidden_weights`` etc. are plain nested lists -- exactly
    what ``param.detach().cpu().numpy().tolist()`` (torch) or
    ``np.asarray(param).tolist()`` (jax) produce, so this drives an ingested net.
    """
    h = len(hidden_weights)
    d = len(point)
    if not (len(hidden_biases) == len(output_weights) == h):
        raise ValueError("hidden_weights, hidden_biases and output_weights must share length h")
    for row in hidden_weights:
        if len(row) != d:
            raise ValueError("each hidden-weight row must have length d = len(point)")

    def make(i: int) -> tuple[Callable[[float], float], DerivBound]:
        offsets = [
            hidden_biases[j]
            + sum(hidden_weights[j][col] * point[col] for col in range(d) if col != i)
            for j in range(h)
        ]

        def g_float(t: float) -> float:
            total = 0.0
            for j in range(h):
                arg = Interval.point(hidden_weights[j][i] * t + offsets[j])
                total += output_weights[j] * sigma_tower_interval(activation, arg, 0)[0].mid
            return total

        def deriv_bound(k: int, box: Interval) -> Interval:
            total = Interval.point(0.0)
            for j in range(h):
                w = Interval.point(hidden_weights[j][i])
                arg = w * box + Interval.point(offsets[j])
                sig_k = sigma_tower_interval(activation, arg, k)[k]
                total = total + Interval.point(output_weights[j]) * w.pow_int(k) * sig_k
            return total

        return g_float, deriv_bound

    axis_fns: list[Callable[[float], float]] = []
    axis_bounds: list[DerivBound] = []
    for i in range(d):
        g, db = make(i)
        axis_fns.append(g)
        axis_bounds.append(db)
    return axis_fns, axis_bounds


#: Map a smooth activation-layer type to its :func:`sigma_tower_interval` name.
_ACTIVATION_NAME: dict[type, str] = {TanhLayer: "tanh", SigmoidLayer: "sigmoid"}


def network_axis_oracles(
    network: Network, point: Sequence[float]
) -> tuple[list[Callable[[float], float]], list[DerivBound]]:
    r"""Per-axis oracles for an ingested scalar one-hidden-layer MLP :class:`Network`.

    Consumes exactly what ``omnibias.verify.torch.network_from_sequential`` /
    ``omnibias.verify.jax`` produce -- a ``[LinearLayer, activation, LinearLayer]``
    stack with a single output -- so a trained torch/jax net drives the certified
    gradient check with no per-backend glue. The scalar readout bias is a constant
    that cancels in every derivative, so it is (correctly) ignored.
    """
    layers = list(network)
    if (
        len(layers) != 3
        or not isinstance(layers[0], LinearLayer)
        or not isinstance(layers[2], LinearLayer)
    ):
        raise ValueError("network_axis_oracles expects a [Linear, activation, Linear] MLP")
    hidden, activation, readout = layers
    assert isinstance(hidden, LinearLayer) and isinstance(readout, LinearLayer)
    name = _ACTIVATION_NAME.get(type(activation))
    if name is None:
        raise NotImplementedError(
            f"unsupported activation layer {type(activation).__name__} "
            f"(supported: {', '.join(t.__name__ for t in _ACTIVATION_NAME)})"
        )
    if readout.out_features != 1:
        raise ValueError("readout layer must have a single output (scalar network)")
    hidden_weights = [list(row) for row in hidden.weight]
    hidden_biases = list(hidden.bias)
    output_weights = list(readout.weight[0])
    return mlp_axis_oracles(hidden_weights, hidden_biases, output_weights, name, point)


__all__ = [
    "GradientCheckCertificate",
    "certified_gradient_check",
    "gradient_residual_certificate",
    "mlp_axis_oracles",
    "network_axis_oracles",
]

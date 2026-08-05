# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Property certificates for a verifier :class:`Network`.

Four guarantees, all sound (the reported verdict holds for *every* input in the
box, computed with outward-rounded interval / Taylor-model arithmetic):

* **robustness** -- the class margin stays positive over an :math:`L^\infty` ball,
  proved with the branch-and-bound read-out;
* **Lipschitz** -- an upper bound on an induced operator norm of the network,
  from a rigorous interval enclosure of the Jacobian over the box;
* **monotonicity** -- the sign of a single partial derivative over the box;
* **reachable set** -- an axis-aligned enclosure of the output set.

The Jacobian enclosure uses the closed-form activation derivative tower
(``sigma'`` from :mod:`omnibias.core.verified.sigma`), so it is exact per layer;
the interval matrix product is where (sound) over-approximation enters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sigma import sigma_tower_interval, sigma_value_interval
from omnibias.verify._core.activations import INV_SQRT_2PI, gauss_cdf_iv
from omnibias.verify._core.bab import output_range, scalar_readout_range
from omnibias.verify._core.network import (
    GELULayer,
    LinearLayer,
    MaxPoolLayer,
    Network,
    ReLULayer,
    SigmoidLayer,
    TanhLayer,
)
from omnibias.verify._core.propagate import interval_propagate

IntervalMatrix = list[list[Interval]]


# --------------------------------------------------------------------------- #
# Elementwise activation derivative enclosures (closed-form towers).
# --------------------------------------------------------------------------- #
def _relu_deriv(iv: Interval) -> Interval:
    """Sound enclosure of ``relu'`` over ``iv`` (the subgradient on the kink)."""
    if iv.hi < 0.0:
        return Interval.point(0.0)
    if iv.lo > 0.0:
        return Interval.point(1.0)
    return Interval(0.0, 1.0)


def _gelu_deriv(iv: Interval) -> Interval:
    r"""Enclosure of ``GELU' = Phi(x) + x*phi(x)`` over ``iv``."""
    phi = INV_SQRT_2PI * sigma_value_interval("gaussian", iv)
    return gauss_cdf_iv(iv) + iv * phi


def _activation_deriv(name: str, iv: Interval) -> Interval:
    if name == "tanh":
        return sigma_tower_interval("tanh", iv, 1)[1]
    if name == "sigmoid":
        return sigma_tower_interval("sigmoid", iv, 1)[1]
    if name == "gelu":
        return _gelu_deriv(iv)
    if name == "relu":
        return _relu_deriv(iv)
    raise ValueError(f"no derivative for activation {name!r}")  # pragma: no cover


_ELEMENTWISE: dict[type, str] = {
    ReLULayer: "relu",
    TanhLayer: "tanh",
    SigmoidLayer: "sigmoid",
    GELULayer: "gelu",
}


# --------------------------------------------------------------------------- #
# Interval matrix helpers.
# --------------------------------------------------------------------------- #
def _identity(n: int) -> IntervalMatrix:
    return [[Interval.point(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]


def _matmul(a: IntervalMatrix, b: IntervalMatrix) -> IntervalMatrix:
    inner = len(b)
    cols = len(b[0]) if b else 0
    out: IntervalMatrix = []
    for row in a:
        new_row: list[Interval] = []
        for j in range(cols):
            acc = Interval.point(0.0)
            for k in range(inner):
                if row[k].lo == 0.0 and row[k].hi == 0.0:
                    continue
                acc = acc + row[k] * b[k][j]
            new_row.append(acc)
        out.append(new_row)
    return out


def _maxpool_jacobian(layer: MaxPoolLayer, in_box: Sequence[Interval], n_in: int) -> IntervalMatrix:
    """Selection-matrix enclosure of a group ``max``: ``[0,1]`` per live member."""
    rows: IntervalMatrix = []
    for group in layer.groups:
        row = [Interval.point(0.0) for _ in range(n_in)]
        dominant = None
        for i in group:
            if all(in_box[i].lo >= in_box[j].hi for j in group if j != i):
                dominant = i
                break
        if dominant is not None:
            row[dominant] = Interval.point(1.0)
        else:
            for i in group:
                dominated = any(in_box[i].hi < in_box[j].lo for j in group if j != i)
                if not dominated:
                    row[i] = Interval(0.0, 1.0)
        rows.append(row)
    return rows


def interval_jacobian(net: Network, input_box: Sequence[IntervalLike]) -> IntervalMatrix:
    r"""Rigorous enclosure of the network Jacobian ``J(x)`` over the whole box.

    Chain rule with interval matrices: ``J = J_L ... J_1`` where an affine layer
    contributes its (constant) weight matrix, an elementwise activation a diagonal
    of closed-form derivative enclosures ``sigma'([range])``, and a max-pool a
    ``[0,1]`` selection matrix.  Every entry ``J[i][j]`` contains
    ``d out_i / d x_j`` for *every* point in the box.
    """
    res = interval_propagate(net, input_box)
    boxes = res.layer_boxes
    n_in = len(boxes[0])
    jac = _identity(n_in)
    for k, layer in enumerate(net):
        in_box = boxes[k]
        if isinstance(layer, LinearLayer):
            weight = [[Interval.point(w) for w in row] for row in layer.weight]
            jac = _matmul(weight, jac)
        elif isinstance(layer, MaxPoolLayer):
            jac = _matmul(_maxpool_jacobian(layer, in_box, len(in_box)), jac)
        else:
            name = _ELEMENTWISE[type(layer)]
            deriv = [_activation_deriv(name, b) for b in in_box]
            jac = [[deriv[i] * jac[i][j] for j in range(n_in)] for i in range(len(jac))]
    return jac


def _mag(iv: Interval) -> float:
    return max(abs(iv.lo), abs(iv.hi))


def _outward_sum(values: Sequence[float]) -> float:
    acc = Interval.point(0.0)
    for v in values:
        acc = acc + Interval.point(v)
    return acc.hi


# --------------------------------------------------------------------------- #
# Lipschitz bound.
# --------------------------------------------------------------------------- #
def lipschitz_bound(
    net: Network, input_box: Sequence[IntervalLike], *, norm: str = "inf"
) -> float:
    r"""Rigorous upper bound on the network's Lipschitz constant over the box.

    ``norm`` selects the induced operator norm: ``"inf"`` (max abs row sum),
    ``"l1"`` (max abs column sum), or ``"l2"`` (the Holder bound
    ``sqrt(||J||_1 ||J||_inf)``, itself an upper bound on the spectral norm).
    """
    jac = interval_jacobian(net, input_box)
    rows, cols = len(jac), len(jac[0]) if jac else 0
    inf_norm = max(
        (_outward_sum([_mag(jac[i][j]) for j in range(cols)]) for i in range(rows)),
        default=0.0,
    )
    if norm == "inf":
        return inf_norm
    one_norm = max(
        (_outward_sum([_mag(jac[i][j]) for i in range(rows)]) for j in range(cols)),
        default=0.0,
    )
    if norm == "l1":
        return one_norm
    if norm == "l2":
        return (Interval.point(inf_norm) * Interval.point(one_norm)).sqrt().hi
    raise ValueError(f"unknown norm {norm!r}; choose 'inf', 'l1' or 'l2'")


# --------------------------------------------------------------------------- #
# Monotonicity.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MonotonicityCertificate:
    """The certified sign of ``d out[out_index] / d x[in_index]`` over the box."""

    out_index: int
    in_index: int
    verdict: str  # "increasing" | "decreasing" | "unknown"
    derivative: Interval

    @property
    def certified(self) -> bool:
        return self.verdict != "unknown"


def monotonicity(
    net: Network, input_box: Sequence[IntervalLike], out_index: int, in_index: int
) -> MonotonicityCertificate:
    """Certify whether output ``out_index`` is monotone in input ``in_index``."""
    jac = interval_jacobian(net, input_box)
    d = jac[out_index][in_index]
    if d.lo >= 0.0:
        verdict = "increasing"
    elif d.hi <= 0.0:
        verdict = "decreasing"
    else:
        verdict = "unknown"
    return MonotonicityCertificate(out_index, in_index, verdict, d)


# --------------------------------------------------------------------------- #
# Reachable set.
# --------------------------------------------------------------------------- #
def reachable_box(
    net: Network,
    input_box: Sequence[IntervalLike],
    *,
    order: int = 2,
    max_boxes: int = 128,
    tol: float = 1e-6,
) -> tuple[Interval, ...]:
    """Axis-aligned enclosure of the output set, one branch-and-bound face per axis."""
    n_out = len(interval_propagate(net, input_box).output)
    return tuple(
        output_range(net, input_box, i, order=order, max_boxes=max_boxes, tol=tol).enclosure
        for i in range(n_out)
    )


# --------------------------------------------------------------------------- #
# Robustness.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RobustnessCertificate:
    """Certified margin of ``true_label`` against every other class over the ball."""

    certified: bool
    true_label: int
    eps: float
    margins: tuple[Interval, ...]
    boxes_explored: int

    @property
    def min_margin(self) -> float:
        return min((m.lo for m in self.margins), default=float("inf"))


def certify_robustness(
    net: Network,
    x0: Sequence[float],
    eps: float,
    true_label: int,
    *,
    order: int = 2,
    max_boxes: int = 256,
    tol: float = 1e-6,
) -> RobustnessCertificate:
    r"""Certify that ``true_label`` is the argmax output over the ball ``||x-x0||_inf <= eps``.

    For each other class ``j`` the margin ``out[true] - out[j]`` is enclosed with
    branch-and-bound; the network is robust iff every margin's certified lower
    bound is strictly positive.
    """
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    box = [Interval(xi - eps, xi + eps) for xi in x0]
    n_out = len(interval_propagate(net, box).output)
    if not 0 <= true_label < n_out:
        raise ValueError(f"true_label must be in 0..{n_out - 1}")
    margins: list[Interval] = []
    explored = 0
    certified = True
    for j in range(n_out):
        if j == true_label:
            continue
        weights = [0.0] * n_out
        weights[true_label] = 1.0
        weights[j] = -1.0
        res = scalar_readout_range(
            net, box, weights, order=order, max_boxes=max_boxes, tol=tol
        )
        margins.append(res.enclosure)
        explored += res.boxes_explored
        if res.enclosure.lo <= 0.0:
            certified = False
    return RobustnessCertificate(
        certified=certified,
        true_label=true_label,
        eps=eps,
        margins=tuple(margins),
        boxes_explored=explored,
    )


__all__ = [
    "IntervalMatrix",
    "MonotonicityCertificate",
    "RobustnessCertificate",
    "certify_robustness",
    "interval_jacobian",
    "lipschitz_bound",
    "monotonicity",
    "reachable_box",
]

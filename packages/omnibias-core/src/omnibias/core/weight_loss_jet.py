# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form one-layer weight-space loss jets.

A Riccati hidden layer ``f = b + c · σ(W x + β)`` is linear in the
step parameter ``s`` at every preactivation once a weight direction
``d`` is fixed: ``z_h(s) = z_h + s u_h``. Founding bias collapse
(``delta -> 0``) supplies ``σ^(n)`` from
:mod:`omnibias.core.polynomials`. Faà di Bruno / Leibniz then assemble

``φ^(k)(0) = d^k/ds^k L(θ + s d) |_{s=0}``

for MSE ``L = (1/B) Σ_n (f_n - y_n)^2`` without nested reverse-mode
and without materialising ``d h / d θ``.

This is **one-layer** closed form. A deep nest is still the chain
rule; this module does not skip it, does not expand ``L(θ)`` as a
polynomial in every weight, and does not return a full parameter
Jacobian of hidden activations. The dense loss Hessian is assembled
only when ``P <= max_params``. Not a global min. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.composed_curvature import solve_dense
from omnibias.core.polynomials import (
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)

SUPPORTED_ACTIVATIONS: frozenset[str] = frozenset({"tanh", "sigmoid"})
_DEFAULT_MAX_PARAMS: int = 256
_FD_STEP: float = 1e-5

DISCLAIMER = (
    "one-layer closed-form weight-space loss jet; Faà di Bruno is the "
    "chain rule; not a global min, not a full dh/dθ flood, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "closed_form": True,
        "one_layer_only": True,
        "deep_net_closed_form": False,
        "skip_chain_rule": False,
        "full_parameter_jacobian": False,
        "global_min_claim": False,
        "stretch_claim": False,
        "continuum_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class WeightLossJetSpec:
    """Order, activation, and dense-Hessian cap for a one-layer loss jet."""

    order: int = 2
    activation: str = "tanh"
    max_params: int = _DEFAULT_MAX_PARAMS

    def __post_init__(self) -> None:
        if int(self.order) < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        act = str(self.activation).lower()
        if act not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}, "
                f"got {self.activation!r}"
            )
        if int(self.max_params) < 1:
            raise ValueError(f"max_params must be >= 1, got {self.max_params}")
        object.__setattr__(self, "activation", act)


def one_layer_param_count(hidden: int, dim: int) -> int:
    """``P = 1 + 2 H + H D`` for layout ``(b, c, β, W)``."""
    if hidden < 1:
        raise ValueError(f"hidden must be >= 1, got {hidden}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return 1 + 2 * hidden + hidden * dim


def pack_one_layer_params(
    bias: float,
    readout: Sequence[float],
    hidden_bias: Sequence[float],
    weights: Sequence[Sequence[float]],
) -> list[float]:
    """Pack ``(b, c, β, W)`` row-major, matching ``omnibias.curvature.one_layer``."""
    hidden = len(readout)
    if hidden < 1:
        raise ValueError("readout must be non-empty")
    if len(hidden_bias) != hidden:
        raise ValueError(
            f"hidden_bias length {len(hidden_bias)} != hidden {hidden}"
        )
    if len(weights) != hidden:
        raise ValueError(f"weights rows {len(weights)} != hidden {hidden}")
    dim = len(weights[0])
    if dim < 1:
        raise ValueError("weight rows must be non-empty")
    if any(len(row) != dim for row in weights):
        raise ValueError("weight rows must share the same length")
    out = [float(bias)]
    out.extend(float(v) for v in readout)
    out.extend(float(v) for v in hidden_bias)
    for row in weights:
        out.extend(float(v) for v in row)
    return out


def unpack_one_layer_params(
    theta: Sequence[float],
    hidden: int,
    dim: int,
) -> tuple[float, list[float], list[float], list[list[float]]]:
    """Inverse of :func:`pack_one_layer_params`."""
    need = one_layer_param_count(hidden, dim)
    if len(theta) != need:
        raise ValueError(
            f"theta length {len(theta)} != P={need} for hidden={hidden}, dim={dim}"
        )
    bias = float(theta[0])
    readout = [float(theta[1 + h]) for h in range(hidden)]
    hidden_bias = [float(theta[1 + hidden + h]) for h in range(hidden)]
    base = 1 + 2 * hidden
    weights = [
        [float(theta[base + h * dim + j]) for j in range(dim)]
        for h in range(hidden)
    ]
    return bias, readout, hidden_bias, weights


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for coeff in reversed(coeffs):
        acc = acc * x + float(coeff)
    return acc


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def eval_sigma_derivative(z: float, n: int, activation: str) -> float:
    """``σ^(n)(z)`` from the shared polynomial coefficients."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    act = str(activation).lower()
    if act == "tanh":
        return _horner(tanh_polynomial_coeffs(n), math.tanh(z))
    if act == "sigmoid":
        return _horner(sigmoid_polynomial_coeffs(n), _sigmoid(z))
    raise ValueError(
        f"activation must be one of {sorted(SUPPORTED_ACTIVATIONS)}, "
        f"got {activation!r}"
    )


def _check_batch(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    dim: int,
) -> None:
    if not xs:
        raise ValueError("xs must be non-empty")
    if len(xs) != len(ys):
        raise ValueError(f"xs length {len(xs)} != ys length {len(ys)}")
    for row in xs:
        if len(row) != dim:
            raise ValueError(f"each x must have length dim={dim}, got {len(row)}")


def one_layer_forward(
    x: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    activation: str,
) -> float:
    """Scalar field value ``b + c · σ(W x + β)``."""
    if len(x) != dim:
        raise ValueError(f"x length {len(x)} != dim {dim}")
    bias, readout, hidden_bias, weights = unpack_one_layer_params(
        theta, hidden, dim
    )
    acc = bias
    for h in range(hidden):
        pre = hidden_bias[h] + sum(weights[h][j] * float(x[j]) for j in range(dim))
        acc += readout[h] * eval_sigma_derivative(pre, 0, activation)
    return acc


def one_layer_loss(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    activation: str,
) -> float:
    """MSE ``(1/B) Σ (f - y)^2``."""
    _check_batch(xs, ys, dim)
    total = 0.0
    for x, y in zip(xs, ys, strict=True):
        residual = one_layer_forward(x, theta, hidden, dim, activation) - float(y)
        total += residual * residual
    return total / float(len(xs))


def one_layer_output_jet(
    x: Sequence[float],
    theta: Sequence[float],
    direction: Sequence[float],
    order: int,
    hidden: int,
    dim: int,
    activation: str,
) -> list[float]:
    r"""Directional derivatives ``f^(k)(0)`` of the scalar field along ``d``.

    With ``z_h(s) = z_h + s u_h`` and ``c_h(s) = c_h + s dc_h``,

    ``f^(k) = Σ_h [c_h σ^(k) u_h^k + k dc_h σ^(k-1) u_h^{k-1}]``

    for ``k >= 1``, plus ``db`` on the first derivative. That Leibniz
    expansion *is* the chain rule.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if len(x) != dim:
        raise ValueError(f"x length {len(x)} != dim {dim}")
    if len(direction) != len(theta):
        raise ValueError(
            f"direction length {len(direction)} != theta length {len(theta)}"
        )
    bias, readout, hidden_bias, weights = unpack_one_layer_params(
        theta, hidden, dim
    )
    db, dc, dbeta, dweights = unpack_one_layer_params(direction, hidden, dim)
    jets = [0.0] * (order + 1)
    jets[0] = bias
    if order >= 1:
        jets[1] = db
    for h in range(hidden):
        pre = hidden_bias[h] + sum(weights[h][j] * float(x[j]) for j in range(dim))
        shift = dbeta[h] + sum(dweights[h][j] * float(x[j]) for j in range(dim))
        sig = [eval_sigma_derivative(pre, k, activation) for k in range(order + 1)]
        jets[0] += readout[h] * sig[0]
        for k in range(1, order + 1):
            jets[k] += readout[h] * sig[k] * (shift**k)
            jets[k] += float(k) * dc[h] * sig[k - 1] * (shift ** (k - 1))
    return jets


def one_layer_loss_jet(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    direction: Sequence[float],
    *,
    order: int,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> list[float]:
    r"""``φ^(k)(0)`` for ``φ(s) = (1/B) Σ_n r_n(s)^2``.

    ``(r^2)^{(k)} = Σ_{j=0}^k C(k, j) r^{(j)} r^{(k-j)}``.
    """
    spec = WeightLossJetSpec(order=order, activation=activation)
    _check_batch(xs, ys, dim)
    if len(direction) != len(theta):
        raise ValueError(
            f"direction length {len(direction)} != theta length {len(theta)}"
        )
    acc = [0.0] * (spec.order + 1)
    for x, y in zip(xs, ys, strict=True):
        field = one_layer_output_jet(
            x, theta, direction, spec.order, hidden, dim, spec.activation
        )
        residual = [field[0] - float(y), *field[1:]]
        for k in range(spec.order + 1):
            term = 0.0
            for j in range(k + 1):
                term += math.comb(k, j) * residual[j] * residual[k - j]
            acc[k] += term
    scale = 1.0 / float(len(xs))
    return [scale * value for value in acc]


def one_layer_output_grad(
    x: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    activation: str,
) -> list[float]:
    """Per-sample ``∇_θ f``, same layout as :func:`pack_one_layer_params`."""
    if len(x) != dim:
        raise ValueError(f"x length {len(x)} != dim {dim}")
    _bias, readout, hidden_bias, weights = unpack_one_layer_params(
        theta, hidden, dim
    )
    grad = [0.0] * one_layer_param_count(hidden, dim)
    grad[0] = 1.0
    for h in range(hidden):
        pre = hidden_bias[h] + sum(weights[h][j] * float(x[j]) for j in range(dim))
        sig0 = eval_sigma_derivative(pre, 0, activation)
        sig1 = eval_sigma_derivative(pre, 1, activation)
        grad[1 + h] = sig0
        grad[1 + hidden + h] = readout[h] * sig1
        base = 1 + 2 * hidden + h * dim
        for j in range(dim):
            grad[base + j] = readout[h] * sig1 * float(x[j])
    return grad


def one_layer_loss_grad(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> list[float]:
    """``∇_θ L = (2/B) Σ_n r_n ∇_θ f_n``."""
    spec = WeightLossJetSpec(order=1, activation=activation)
    _check_batch(xs, ys, dim)
    need = one_layer_param_count(hidden, dim)
    if len(theta) != need:
        raise ValueError(f"theta length {len(theta)} != P={need}")
    acc = [0.0] * need
    for x, y in zip(xs, ys, strict=True):
        residual = one_layer_forward(x, theta, hidden, dim, spec.activation) - float(
            y
        )
        grad = one_layer_output_grad(x, theta, hidden, dim, spec.activation)
        for i, value in enumerate(grad):
            acc[i] += residual * value
    scale = 2.0 / float(len(xs))
    return [scale * value for value in acc]


def _network_hessian(
    x: Sequence[float],
    readout: Sequence[float],
    hidden_bias: Sequence[float],
    weights: Sequence[Sequence[float]],
    hidden: int,
    dim: int,
    activation: str,
) -> list[list[float]]:
    need = one_layer_param_count(hidden, dim)
    hess = [[0.0] * need for _ in range(need)]
    for h in range(hidden):
        pre = hidden_bias[h] + sum(weights[h][j] * float(x[j]) for j in range(dim))
        sig1 = eval_sigma_derivative(pre, 1, activation)
        sig2 = eval_sigma_derivative(pre, 2, activation)
        idx_c = 1 + h
        idx_beta = 1 + hidden + h
        w0 = 1 + 2 * hidden + h * dim
        hess[idx_c][idx_beta] += sig1
        hess[idx_beta][idx_c] += sig1
        hess[idx_beta][idx_beta] += readout[h] * sig2
        for j in range(dim):
            idx_w = w0 + j
            xj = float(x[j])
            hess[idx_c][idx_w] += sig1 * xj
            hess[idx_w][idx_c] += sig1 * xj
            hess[idx_beta][idx_w] += readout[h] * sig2 * xj
            hess[idx_w][idx_beta] += readout[h] * sig2 * xj
            for jp in range(dim):
                hess[idx_w][w0 + jp] += readout[h] * sig2 * xj * float(x[jp])
    return hess


def one_layer_loss_hessian(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    hidden: int,
    dim: int,
    activation: str = "tanh",
    *,
    max_params: int = _DEFAULT_MAX_PARAMS,
) -> list[list[float]]:
    """Full MSE Hessian ``(2/B) Σ_n (∇f ∇f^T + r ∇²f)``.

    Raises if ``P > max_params`` so a silent ``P × P`` flood cannot hide
    behind this helper. That is the loss Hessian, not ``d h / d θ``.
    """
    spec = WeightLossJetSpec(order=2, activation=activation, max_params=max_params)
    _check_batch(xs, ys, dim)
    need = one_layer_param_count(hidden, dim)
    if need > spec.max_params:
        raise ValueError(
            f"P={need} exceeds max_params={spec.max_params}; "
            "assemble a directional jet instead of a dense Hessian"
        )
    if len(theta) != need:
        raise ValueError(f"theta length {len(theta)} != P={need}")
    _, readout, hidden_bias, weights = unpack_one_layer_params(theta, hidden, dim)
    hess = [[0.0] * need for _ in range(need)]
    for x, y in zip(xs, ys, strict=True):
        residual = one_layer_forward(x, theta, hidden, dim, spec.activation) - float(
            y
        )
        grad = one_layer_output_grad(x, theta, hidden, dim, spec.activation)
        net = _network_hessian(
            x, readout, hidden_bias, weights, hidden, dim, spec.activation
        )
        for i in range(need):
            gi = grad[i]
            for j in range(need):
                hess[i][j] += gi * grad[j] + residual * net[i][j]
    scale = 2.0 / float(len(xs))
    return [[scale * value for value in row] for row in hess]


def one_layer_newton_direction(
    grad: Sequence[float],
    hess: Sequence[Sequence[float]],
    *,
    damping: float = 1e-6,
) -> list[float]:
    """Solve ``(H + λ I) d = -∇L``.

    The direction is a local descent (``∇L · d < 0`` on a PD Hessian).
    A raw length-1 step can overshoot; pair with a jet line search.
    Not a global min.
    """
    if damping < 0.0 or not math.isfinite(damping):
        raise ValueError(f"damping must be a finite number >= 0, got {damping}")
    if len(grad) != len(hess):
        raise ValueError("grad and hess must share dimension")
    rhs = [-float(v) for v in grad]
    return list(solve_dense(hess, rhs, damping=damping))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _quad(
    hess: Sequence[Sequence[float]],
    direction: Sequence[float],
) -> float:
    acc = 0.0
    for i, row in enumerate(hess):
        di = float(direction[i])
        acc += di * _dot(row, direction)
    return acc


def worked_example() -> dict[str, bool | float]:
    """Deterministic one-unit identities used by the cookbook snippet."""
    hidden = 1
    dim = 1
    xs = ((0.5,), (-0.25,))
    ys = (0.1, -0.2)
    theta = pack_one_layer_params(0.0, (1.0,), (0.0,), ((0.5,),))
    direction = pack_one_layer_params(0.0, (0.0,), (0.0,), ((1.0,),))
    jet = one_layer_loss_jet(
        xs,
        ys,
        theta,
        direction,
        order=3,
        hidden=hidden,
        dim=dim,
        activation="tanh",
    )
    loss = one_layer_loss(xs, ys, theta, hidden, dim, "tanh")
    grad = one_layer_loss_grad(xs, ys, theta, hidden, dim, "tanh")
    hess = one_layer_loss_hessian(xs, ys, theta, hidden, dim, "tanh")
    return {
        "jet0_matches_loss": abs(jet[0] - loss) < 1e-14,
        "jet1_matches_gdotd": abs(jet[1] - _dot(grad, direction)) < 1e-13,
        "jet2_matches_dHd": abs(jet[2] - _quad(hess, direction)) < 1e-12,
        "loss": float(loss),
        "phi1": float(jet[1]),
        "phi2": float(jet[2]),
        "phi3": float(jet[3]),
    }


def finite_difference_jet(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    theta: Sequence[float],
    direction: Sequence[float],
    *,
    order: int,
    hidden: int,
    dim: int,
    activation: str = "tanh",
    step: float = _FD_STEP,
) -> list[float]:
    """Central differences of ``φ(s)`` for tests; not the public method."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")

    def phi(scale: float) -> float:
        moved = [
            float(theta[i]) + scale * float(direction[i]) for i in range(len(theta))
        ]
        return one_layer_loss(xs, ys, moved, hidden, dim, activation)

    if order == 0:
        return [phi(0.0)]
    # Recursive central differences on a tiny stencil; tests use order <= 3.
    values = [phi(0.0)]
    plus = phi(step)
    minus = phi(-step)
    values.append((plus - minus) / (2.0 * step))
    if order >= 2:
        values.append((plus - 2.0 * phi(0.0) + minus) / (step * step))
    if order >= 3:
        plus2 = phi(2.0 * step)
        minus2 = phi(-2.0 * step)
        values.append(
            (plus2 - 2.0 * plus + 2.0 * minus - minus2) / (2.0 * step**3)
        )
    return values[: order + 1]


__all__ = [
    "DISCLAIMER",
    "SUPPORTED_ACTIVATIONS",
    "WeightLossJetSpec",
    "eval_sigma_derivative",
    "finite_difference_jet",
    "honesty_payload",
    "one_layer_forward",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "one_layer_output_grad",
    "one_layer_output_jet",
    "one_layer_param_count",
    "pack_one_layer_params",
    "unpack_one_layer_params",
    "worked_example",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sound residual enclosure for a DeepONet over a family of inputs.

Given a coefficient box ``[b_k]`` (the image of a sensor box under the branch)
and a query box, the trunk jet is enclosed by
:func:`omnibias.core.verified.jet_mv.mlp_jet_mv` and the operator residual

    R = u_t - D u_xx   (heat)   or   R = u_t + u u_x - nu u_xx   (Burgers)

is enclosed by contracting the coefficient box with the enclosed trunk
partials. This is a **residual enclosure over a family of input functions**,
not a solution-error bound: converting it into ``||u_NN - u_true||`` requires
a stability constant the caller must supply (the same honesty gate
``aposteriori_error_certificate`` already carries).

The additive DeepONet bias is intentionally absent from these APIs: it occupies
only jet row 0 and cancels from every derivative that enters a residual.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.multi_index import index_position, multi_index_factorial
from omnibias.core.proof.certificate import Cert, interval_certificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import BoxLike, Layer, mlp_jet_mv

IntervalLike = Interval | tuple[float, float] | float


def _as_interval(v: IntervalLike) -> Interval:
    if isinstance(v, Interval):
        return v
    if isinstance(v, tuple):
        return Interval(float(v[0]), float(v[1]))
    return Interval.point(float(v))


def _dot(coeffs: Sequence[Interval], partials: Sequence[Interval]) -> Interval:
    acc = Interval.point(0.0)
    for c, p in zip(coeffs, partials, strict=True):
        acc = acc + c * p
    return acc


def enclose_heat_residual(
    *,
    trunk_layers: Sequence[Layer],
    coeffs: Sequence[IntervalLike],
    query_box: BoxLike,
    diffusivity: float,
    order: int = 2,
) -> Interval:
    """Enclose ``u_t - D u_xx`` of ``G(u)(y) = bias + sum_k c_k t_k(y)`` over ``query_box``.

    ``query_box`` is length 2 with axes ``(x, t)``. ``coeffs`` has length ``p``
    (single-component DeepONet). ``trunk_layers`` ends in an affine readout of
    width ``p``. The DeepONet bias is omitted because it cancels from every
    derivative that enters the residual.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2 to reach u_xx; got {order}")
    c = [_as_interval(v) for v in coeffs]
    jet = mlp_jet_mv(query_box, trunk_layers, order)
    # jet rows are D^alpha t / alpha!; convert to raw partials.
    dim = len(query_box)
    pos = index_position(dim, order)
    # Trunk jet width = p.
    p = len(jet[0])
    if len(c) != p:
        raise ValueError(f"coeffs length {len(c)} != trunk width {p}")

    def partial(alpha: tuple[int, ...]) -> list[Interval]:
        row = jet[pos[alpha]]
        scale = multi_index_factorial(alpha)
        return [row[k] * float(scale) for k in range(p)]

    # u = bias + sum c_k t_k  =>  u_t = sum c_k dt t_k, u_xx = sum c_k dxx t_k
    # (bias drops out of every derivative).
    alpha_t = (0, 1)
    alpha_xx = (2, 0)
    u_t = _dot(c, partial(alpha_t))
    u_xx = _dot(c, partial(alpha_xx))
    return u_t - float(diffusivity) * u_xx


def certify_heat_residual(
    *,
    trunk_layers: Sequence[Layer],
    coeffs: Sequence[IntervalLike],
    query_box: BoxLike,
    diffusivity: float,
    order: int = 2,
    claim: str | None = None,
) -> Cert:
    """Seal an interval certificate for the heat residual enclosure.

    Honesty: this certifies a **residual enclosure**, not a solution-error
    bound. No stability constant is applied.
    """
    enclosure = enclose_heat_residual(
        trunk_layers=trunk_layers,
        coeffs=coeffs,
        query_box=query_box,
        diffusivity=diffusivity,
        order=order,
    )
    return interval_certificate(
        claim=claim
        or (
            "DeepONet heat residual u_t - D u_xx is enclosed over the query box "
            "for every coefficient vector in the supplied coefficient box"
        ),
        interval=enclosure,
        honesty={"residual_enclosure_not_solution_error": True},
        meta={
            "diffusivity": float(diffusivity),
            "order": int(order),
            "kind": "operator_heat_residual",
        },
    )


def branch_coefficient_box(
    sensors_box: Sequence[IntervalLike],
    branch_layers: Sequence[Layer],
) -> tuple[list[Interval], Interval | None]:
    """Propagate a sensor box through an affine+tanh branch to a coefficient box.

    ``branch_layers`` is a sequence of ``(W, b, name)`` with ``name`` in
    ``{None, "tanh", "sigmoid"}``; the final layer must be affine (``name=None``)
    producing ``p`` (or ``p+1`` with per-sample bias) outputs for a
    single-component operator. Returns ``(coeffs, None)`` -- callers that emit
    a trailing bias should split the last entry themselves.
    """
    from omnibias.core.verified.transcend import sigmoid_iv, tanh_iv

    h = [_as_interval(v) for v in sensors_box]
    for weight, bias, name in branch_layers:
        rows = []
        for i, row in enumerate(weight):
            acc = (
                Interval.from_value(bias[i])
                if bias is not None
                else Interval.point(0.0)
            )
            for a, hj in zip(row, h, strict=True):
                acc = acc + Interval.from_value(a) * hj
            rows.append(acc)
        h = rows
        if name is None:
            continue
        if name == "tanh":
            h = [tanh_iv(hi) for hi in h]
        elif name == "sigmoid":
            h = [sigmoid_iv(hi) for hi in h]
        else:
            raise ValueError(
                f"branch_coefficient_box supports tanh/sigmoid/affine; got {name!r}"
            )
    return h, None


__all__ = [
    "branch_coefficient_box",
    "certify_heat_residual",
    "enclose_heat_residual",
]

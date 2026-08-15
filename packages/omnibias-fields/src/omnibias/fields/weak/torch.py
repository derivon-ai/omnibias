# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Weak-form VPINN assembly (torch; theory 02-04)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.weak._core import (
    TestFunctionSpace,
    WeakForm,
    boundary_bound,
    eval_test,
    exact_moment,
    poly_eval,
)
from torch import Tensor

PathName = Literal["exact", "quadrature"]


@dataclass(frozen=True)
class ResidualTerm:
    """One assembled weak-form contribution, with the path recorded."""

    name: str
    path: PathName
    value: Tensor


def _quad_nodes(spec: QuadratureSpec, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    x = torch.as_tensor(spec.nodes[:, 0], dtype=dtype, device=device)
    w = torch.as_tensor(spec.weights, dtype=dtype, device=device)
    return x, w


def weak_residual(
    field: Callable[[Tensor], Tensor] | tuple[float, ...],
    space: TestFunctionSpace,
    *,
    operator: WeakForm,
    source: Callable[[Tensor], Tensor] | tuple[float, ...] | None = None,
    quadrature: QuadratureSpec | None = None,
) -> tuple[Tensor, tuple[ResidualTerm, ...]]:
    """Assemble ``r(v_i)`` for every test function.

    Polynomial ``source`` / ``operator.diffusion`` on a box use the exact
    antiderivative path. A callable field falls back to quadrature on the
    coefficient factor only. Each term records ``exact`` or ``quadrature``.
    """
    if space.window is None:
        raise ValueError("weak_residual needs a box window")
    src = operator.source if source is None else source
    n = space.size
    values = []
    terms: list[ResidualTerm] = []
    sample = torch.zeros((), dtype=torch.get_default_dtype())
    dt = sample.dtype
    device = sample.device
    for i in range(n):
        stiffness, stiff_path = _stiffness_term(
            field, space, i, operator.diffusion, quadrature, dt, device
        )
        load, load_path = _load_term(src, space, i, quadrature, dt, device)
        r = stiffness - load
        values.append(r)
        terms.append(ResidualTerm(f"stiffness[{i}]", stiff_path, stiffness))
        terms.append(ResidualTerm(f"load[{i}]", load_path, load))
    stacked = torch.stack(values)
    return stacked, tuple(terms)


def _stiffness_term(
    field: Callable[[Tensor], Tensor] | tuple[float, ...],
    space: TestFunctionSpace,
    index: int,
    diffusion: tuple[float, ...],
    quadrature: QuadratureSpec | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, PathName]:
    a_poly = all(isinstance(c, (int, float)) for c in diffusion)
    if isinstance(field, tuple) and a_poly:
        # Exact: a and u polynomials => a u' is polynomial; int (a u') v' via moments of v'.
        u_coeffs = field
        # u' coeffs
        du = tuple(float(k) * float(u_coeffs[k]) for k in range(1, len(u_coeffs)))
        prod = _poly_mul(diffusion, du if du else (0.0,))
        acc = 0.0
        # int poly(x) v'(x) dx = sum c_j exact_moment of v' = test with order+1, power j
        bumped = TestFunctionSpace(
            bank=space.bank,
            orders=tuple(space.order_at(index) + 1 for _ in range(space.size)),
            base=space.base,
            window=space.window,
        )
        alpha = space.scale_at(index)
        for j, c in enumerate(prod):
            acc += float(c) * alpha * exact_moment(bumped, j, index)
        return torch.tensor(acc, dtype=dtype, device=device), "exact"
    if quadrature is None:
        raise ValueError("callable field needs a QuadratureSpec for the coefficient factor")
    xs, ws = _quad_nodes(quadrature, dtype, device)
    if isinstance(field, tuple):
        du_c = tuple(float(k) * float(field[k]) for k in range(1, len(field)))
        du = torch.tensor(
            [poly_eval(du_c if du_c else (0.0,), float(x)) for x in xs.tolist()],
            dtype=dtype,
        )
    else:
        xs_req = xs.detach().clone().requires_grad_(True)
        u = field(xs_req.unsqueeze(-1)).reshape(-1)
        du = torch.autograd.grad(u.sum(), xs_req, create_graph=True)[0]
    a = torch.tensor([poly_eval(diffusion, float(x)) for x in xs.tolist()], dtype=dtype, device=device)
    vp = torch.tensor(
        [eval_test(space, index, float(x), deriv=1) for x in xs.tolist()],
        dtype=dtype,
        device=device,
    )
    val = (ws * a * du * vp).sum()
    return val, "quadrature"


def _load_term(
    source: Callable[[Tensor], Tensor] | tuple[float, ...] | None,
    space: TestFunctionSpace,
    index: int,
    quadrature: QuadratureSpec | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, PathName]:
    if source is None:
        return torch.zeros((), dtype=dtype, device=device), "exact"
    if isinstance(source, tuple):
        acc = 0.0
        for j, c in enumerate(source):
            acc += float(c) * exact_moment(space, j, index)
        return torch.tensor(acc, dtype=dtype, device=device), "exact"
    if quadrature is None:
        raise ValueError("callable source needs a QuadratureSpec")
    xs, ws = _quad_nodes(quadrature, dtype, device)
    f = source(xs.unsqueeze(-1)).reshape(-1)
    v = torch.tensor(
        [eval_test(space, index, float(x), deriv=0) for x in xs.tolist()],
        dtype=dtype,
        device=device,
    )
    return (ws * f * v).sum(), "quadrature"


def _poly_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += float(ai) * float(bj)
    return tuple(out)


def weak_loss(
    field: Callable[[Tensor], Tensor] | tuple[float, ...],
    space: TestFunctionSpace,
    *,
    operator: WeakForm,
    source: Callable[[Tensor], Tensor] | tuple[float, ...] | None = None,
    quadrature: QuadratureSpec | None = None,
    include_boundary_bound: bool = True,
    deriv_bound: float = 1.0,
) -> Tensor:
    """Sum of squared weak residuals; boundary bound on by default."""
    residual, _terms = weak_residual(
        field, space, operator=operator, source=source, quadrature=quadrature
    )
    loss = (residual * residual).sum()
    if include_boundary_bound:
        bound = boundary_bound(space, deriv_bound=deriv_bound)
        loss = loss + torch.tensor(float(bound.hi), dtype=loss.dtype, device=loss.device)
    return loss


__all__ = [
    "ResidualTerm",
    "weak_loss",
    "weak_residual",
]

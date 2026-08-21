# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Block / coordinate exact search (theory 08-07), PyTorch twin.

A sparse block direction is handed to the 03-12 jet line search
(``verify=True`` never-worse). Last-layer least squares is exactly
quadratic. This is a coordinate / block sweep, not a global solver
and not CCF stretch.

Do not wrap the driver in ``torch.compile``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from omnibias.core.block_search import (
    BlockSpec,
    apply_block_step,
    arrangement_w_block,
    default_block_config,
    last_linear_block,
    ombu_bias_block,
    resolve_block_mask,
    unit_direction_from_mask,
)
from omnibias.core.line_search import JetLineSearchConfig, LineSearchResult
from omnibias.torch.line_search import jet_line_search

import torch
from torch import Tensor
from torch.func import grad

ScalarFn = Callable[[Tensor], Tensor]


def _as_mask(
    params: Tensor,
    *,
    spec: BlockSpec | None,
    mask: Tensor | Sequence[bool] | None,
) -> tuple[bool, ...]:
    n = int(params.reshape(-1).shape[0])
    flags: Sequence[bool] | None
    if mask is None:
        flags = None
    elif isinstance(mask, Tensor):
        flags = [bool(v) for v in mask.reshape(-1)]
    else:
        flags = mask
    block = spec if spec is not None else BlockSpec(kind="mask")
    return resolve_block_mask(n, block, flags)


def block_direction(
    params: Tensor,
    *,
    spec: BlockSpec | None = None,
    mask: Tensor | Sequence[bool] | None = None,
    probe: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Unit direction supported on a named or boolean block."""
    p = params.reshape(-1)
    flags = _as_mask(p, spec=spec, mask=mask)
    probe_list: Sequence[float] | None
    if probe is None:
        probe_list = None
    elif isinstance(probe, Tensor):
        probe_list = [float(v) for v in probe.reshape(-1)]
    else:
        probe_list = probe
    direction = unit_direction_from_mask(flags, probe_list)
    return torch.as_tensor(direction, dtype=p.dtype, device=p.device)


def block_exact_search(
    loss_fn: ScalarFn,
    params: Tensor,
    *,
    spec: BlockSpec | None = None,
    mask: Tensor | Sequence[bool] | None = None,
    direction: Tensor | None = None,
    config: JetLineSearchConfig | None = None,
    exact_quadratic: bool = False,
    next_derivative_bound: float | None = None,
) -> tuple[Tensor, LineSearchResult]:
    """03-12 line search along one structured block; never-worse by default."""
    p = params.reshape(-1)
    flags = _as_mask(p, spec=spec, mask=mask)
    if direction is None:
        g = grad(loss_fn)(p)
        probe = (-g).reshape(-1)
        d = block_direction(p, spec=BlockSpec(kind="mask"), mask=flags, probe=probe)
    else:
        raw = direction.reshape(-1)
        if int(raw.shape[0]) != int(p.shape[0]):
            raise ValueError("direction must match flat params")
        restricted = torch.where(
            torch.as_tensor(flags, dtype=torch.bool, device=p.device),
            raw,
            torch.zeros_like(raw),
        )
        nrm = float(torch.linalg.vector_norm(restricted))
        if nrm == 0.0:
            d = block_direction(p, spec=BlockSpec(kind="mask"), mask=flags)
        else:
            d = restricted / nrm
    cfg = config if config is not None else default_block_config(
        exact_quadratic=exact_quadratic
    )
    bound = 0.0 if exact_quadratic and next_derivative_bound is None else next_derivative_bound
    result = jet_line_search(
        loss_fn, p, d, config=cfg, next_derivative_bound=bound
    )
    new = apply_block_step(
        [float(v) for v in p], [float(v) for v in d], result.step
    )
    return torch.as_tensor(new, dtype=p.dtype, device=p.device), result


def block_exact_sweep(
    loss_fn: ScalarFn,
    params: Tensor,
    *,
    spec: BlockSpec | None = None,
    mask: Tensor | Sequence[bool] | None = None,
    config: JetLineSearchConfig | None = None,
    exact_quadratic: bool = False,
    next_derivative_bound: float | None = None,
) -> tuple[Tensor, list[LineSearchResult]]:
    """One Gauss–Seidel pass: search each singleton coordinate of the block."""
    p = params.reshape(-1)
    flags = _as_mask(p, spec=spec, mask=mask)
    reports: list[LineSearchResult] = []
    for i, active in enumerate(flags):
        if not active:
            continue
        coord = [False] * len(flags)
        coord[i] = True
        p, report = block_exact_search(
            loss_fn,
            p,
            spec=BlockSpec(kind="mask"),
            mask=coord,
            config=config,
            exact_quadratic=exact_quadratic,
            next_derivative_bound=next_derivative_bound,
        )
        reports.append(report)
    return p, reports


__all__ = [
    "BlockSpec",
    "arrangement_w_block",
    "block_direction",
    "block_exact_search",
    "block_exact_sweep",
    "last_linear_block",
    "ombu_bias_block",
]

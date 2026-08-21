# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Block / coordinate exact search (theory 08-07), JAX twin.

A sparse block direction is handed to the 03-12 jet line search
(``verify=True`` never-worse). Last-layer least squares is exactly
quadratic. This is a coordinate / block sweep, not a global solver
and not CCF stretch.

``loss_fn`` must be jit-compatible. Enable x64 before the first array
when matching the torch twin.
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
from omnibias.jax.line_search import jet_line_search

import jax
import jax.numpy as jnp
from jax import Array

ScalarFn = Callable[[Array], Array]


def _as_mask(
    params: Array,
    *,
    spec: BlockSpec | None,
    mask: Array | Sequence[bool] | None,
) -> tuple[bool, ...]:
    n = int(jnp.reshape(params, (-1,)).shape[0])
    flags: Sequence[bool] | None
    if mask is None:
        flags = None
    else:
        flags = [bool(v) for v in jnp.reshape(jnp.asarray(mask), (-1,))]
    block = spec if spec is not None else BlockSpec(kind="mask")
    return resolve_block_mask(n, block, flags)


def block_direction(
    params: Array,
    *,
    spec: BlockSpec | None = None,
    mask: Array | Sequence[bool] | None = None,
    probe: Array | Sequence[float] | None = None,
) -> Array:
    """Unit direction supported on a named or boolean block."""
    p = jnp.reshape(params, (-1,))
    flags = _as_mask(p, spec=spec, mask=mask)
    probe_list: Sequence[float] | None
    if probe is None:
        probe_list = None
    else:
        probe_list = [float(v) for v in jnp.reshape(jnp.asarray(probe), (-1,))]
    direction = unit_direction_from_mask(flags, probe_list)
    return jnp.asarray(direction, dtype=p.dtype)


def block_exact_search(
    loss_fn: ScalarFn,
    params: Array,
    *,
    spec: BlockSpec | None = None,
    mask: Array | Sequence[bool] | None = None,
    direction: Array | None = None,
    config: JetLineSearchConfig | None = None,
    exact_quadratic: bool = False,
    next_derivative_bound: float | None = None,
) -> tuple[Array, LineSearchResult]:
    """03-12 line search along one structured block; never-worse by default."""
    p = jnp.reshape(params, (-1,))
    flags = _as_mask(p, spec=spec, mask=mask)
    if direction is None:
        g = jax.grad(loss_fn)(p)
        probe = jnp.reshape(-g, (-1,))
        d = block_direction(p, spec=BlockSpec(kind="mask"), mask=flags, probe=probe)
    else:
        raw = jnp.reshape(direction, (-1,))
        if int(raw.shape[0]) != int(p.shape[0]):
            raise ValueError("direction must match flat params")
        flag_arr = jnp.asarray(flags)
        restricted = jnp.where(flag_arr, raw, jnp.zeros_like(raw))
        nrm = float(jnp.linalg.norm(restricted))
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
    return jnp.asarray(new, dtype=p.dtype), result


def block_exact_sweep(
    loss_fn: ScalarFn,
    params: Array,
    *,
    spec: BlockSpec | None = None,
    mask: Array | Sequence[bool] | None = None,
    config: JetLineSearchConfig | None = None,
    exact_quadratic: bool = False,
    next_derivative_bound: float | None = None,
) -> tuple[Array, list[LineSearchResult]]:
    """One Gauss–Seidel pass: search each singleton coordinate of the block."""
    p = jnp.reshape(params, (-1,))
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

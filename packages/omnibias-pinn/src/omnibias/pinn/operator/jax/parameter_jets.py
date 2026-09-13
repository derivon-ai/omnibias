# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Live field/physical-parameter derivatives; legacy heat example stays explicit.

A callable with method="autodiff" uses actual forward-mode autodiff. The
closed-form path requires a provider's mixed_parameter_jet method, or field=None
for the manufactured heat mode. That legacy manufactured entrypoint returns a
shared Python scalar by default; live=True opts into a differentiable array.
Supplied providers and callable autodiff paths always remain live.

The founding bias collapse is delta -> 0; beta -> inf is the separate
temperature/feasibility limit. Do not conflate these with enclosure collapse.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

import jax.numpy as jnp
from jax import Array, jacfwd
from omnibias.core import parameter_jets as core

DISCLAIMER = core.DISCLAIMER
ParameterJetSpec = core.ParameterJetSpec
honesty_payload = core.honesty_payload
FieldFn = Callable[[Array, Array], Array]


@runtime_checkable
class ArrayParameterJetProvider(Protocol):
    def mixed_parameter_jet(
        self, coords: Array, parameters: Array, *, spec: ParameterJetSpec,
    ) -> Array: ...


def _parameter_partial(fn: FieldFn) -> FieldFn:
    def derivative(coords: Array, mu: Array) -> Array:
        return cast(Array, jacfwd(lambda p: fn(coords, p))(mu))
    return derivative


def _coordinate_partial(fn: FieldFn, axis: int) -> FieldFn:
    def derivative(coords: Array, mu: Array) -> Array:
        return cast(Array, jacfwd(lambda c: fn(c, mu))(coords)[..., axis])
    return derivative


def mixed_jet(
    field: object | None,
    coords: Array | tuple[float, float],
    parameters: Array | float,
    *,
    spec: ParameterJetSpec | None = None,
    wave: float = 1.0,
    live: bool = False,
) -> Array | float:
    cfg = core.DEFAULT_SPEC if spec is None else spec
    if field is None and cfg.method == "closed_form" and not live:
        # Compatibility-only host ingestion; traced callers select live=True.
        point_values = coords.reshape(-1).tolist() if isinstance(coords, Array) else list(coords)
        parameter_values = parameters.reshape(-1).tolist() if isinstance(parameters, Array) else [parameters]
        if len(point_values) != 2 or len(parameter_values) != 1:
            raise ValueError("the legacy manufactured heat entrypoint needs (x,t) and one parameter")
        return core.mixed_jet(None, (float(point_values[0]), float(point_values[1])),
                              float(parameter_values[0]), spec=cfg, wave=wave)
    if field is None and not live and not isinstance(coords, Array) and not isinstance(parameters, Array) and cfg.method != "autodiff":
        return core.mixed_jet(None, coords, parameters, spec=cfg, wave=wave)
    reference = parameters if isinstance(parameters, Array) else coords if isinstance(coords, Array) else jnp.empty(())
    point = jnp.asarray(coords, dtype=reference.dtype).reshape(-1)
    mu = jnp.asarray(parameters, dtype=reference.dtype)
    if mu.size != 1:
        raise ValueError("mixed_jet expects one physical parameter; use realization jets for parameter subspaces")
    mu = mu.reshape(())
    spatial = cfg.space_multi_index or (0,) * point.shape[0]
    if len(spatial) != point.shape[0]:
        raise ValueError("space_multi_index must match the coordinate count")
    if cfg.method == "closed_form":
        if not cfg.mu_in_jet_trunk:
            raise ValueError("closed_form requires mu_in_jet_trunk=True")
        if field is not None:
            if not isinstance(field, ArrayParameterJetProvider):
                raise TypeError("closed_form field must implement mixed_parameter_jet")
            result = field.mixed_parameter_jet(point, mu, spec=cfg)
            if not isinstance(result, Array):
                raise TypeError("a tensor jet provider must return a tensor")
            return result
        if point.shape[0] != 2:
            raise ValueError("the manufactured heat example needs (x,t)")
        x, t = point[0], point[1]
        kx, kt = spatial
        n = cfg.param_order
        time_factor = jnp.zeros_like(t)
        for j in range(min(n, kt) + 1):
            time_factor = time_factor + (math.comb(kt, j) * math.factorial(n) / math.factorial(n-j)
                                         * t ** (n-j) * (-mu*wave*wave) ** (kt-j))
        return ((-wave*wave) ** n * time_factor * wave ** kx
                * jnp.exp(-mu*wave*wave*t) * jnp.sin(wave*x+kx*math.pi/2))
    fn: FieldFn
    if field is None:
        if point.shape[0] != 2:
            raise ValueError("the manufactured heat example needs (x,t)")
        def heat(c: Array, p: Array) -> Array:
            return jnp.exp(-p*wave*wave*c[1]) * jnp.sin(wave*c[0])
        fn = heat
    elif callable(field):
        fn = field
    else:
        raise TypeError("field must be callable for autodiff/finite_difference")
    if cfg.method == "finite_difference":
        if cfg.param_order != 1 or any(spatial):
            raise NotImplementedError("finite_difference supports the first parameter derivative only")
        return (fn(point, mu + 1e-6) - fn(point, mu - 1e-6)) / 2e-6
    for _ in range(cfg.param_order):
        fn = _parameter_partial(fn)
    for axis, count in enumerate(spatial):
        for _ in range(count):
            fn = _coordinate_partial(fn, axis)
    return fn(point, mu)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "ArrayParameterJetProvider",
    "DISCLAIMER",
    "ParameterJetSpec",
    "honesty_payload",
    "mixed_jet",
    "worked_example",
]

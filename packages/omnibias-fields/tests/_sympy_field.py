# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sympy-backed analytic field for op tests that need cross-terms or transcendentals.

The separable-polynomial :class:`_analytic.AnalyticField` cannot represent
coupled arguments such as ``(z - t)`` or a Gaussian ``exp(-|v|^2)``. This fixture
evaluates *arbitrary* sympy expressions per component and differentiates them
**exactly** with ``sympy.diff`` + ``lambdify``, so it is the independent analytic
oracle for the MHD (WS3) residuals and the kinetic (WS3) Vlasov/Maxwellian ops.

It implements the ``"spectral"`` state-method dispatch protocol
(``value_component`` / ``derivative`` / ``mixed_partial``) exactly like
``AnalyticField``, so the foundational ops drive it without any backend coupling.
Values are computed in numpy and cast to the backend tensor, which is why this
helper is used only for analytic oracles; genuine torch-vs-jax arithmetic parity
is covered by the polynomial :class:`_analytic.AnalyticField`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import sympy as sp
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState


def axis_symbols(axes: Sequence[str]) -> dict[str, sp.Symbol]:
    """Real sympy symbols, one per axis name."""
    return {a: sp.Symbol(a, real=True) for a in axes}


def _to_columns(coords: object) -> list[np.ndarray]:
    try:
        import torch

        if isinstance(coords, torch.Tensor):
            arr = coords.detach().cpu().numpy()
            return [arr[:, i] for i in range(arr.shape[1])]
    except ImportError:  # pragma: no cover
        pass
    arr = np.asarray(coords)
    return [arr[:, i] for i in range(arr.shape[1])]


def _from_numpy(out: np.ndarray, like: object) -> object:
    try:
        import torch

        if isinstance(like, torch.Tensor):
            return torch.as_tensor(out, dtype=like.dtype, device=like.device)
    except ImportError:  # pragma: no cover
        pass
    import jax.numpy as jnp

    return jnp.asarray(out, dtype=like.dtype)


class SympyField:
    """Analytic field whose components are arbitrary sympy expressions."""

    _omnibias_dispatch = "spectral"

    def __init__(self, coordinate_spec, components, exprs, symbols, ops_module):  # type: ignore[no-untyped-def]
        self.coordinate_spec = coordinate_spec
        self.components = components
        self._exprs = dict(exprs)
        self._symbols = tuple(symbols)
        self._ops = ops_module
        self._lam: dict[tuple[str, tuple[tuple[int, int], ...]], object] = {}

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    def __call__(self, coords):  # type: ignore[no-untyped-def]
        return self.evaluate(coords)

    def _fn(self, name, deriv_key):  # type: ignore[no-untyped-def]
        key = (name, deriv_key)
        if key not in self._lam:
            expr = self._exprs[name]
            for axis_idx, order in deriv_key:
                expr = sp.diff(expr, self._symbols[axis_idx], order)
            self._lam[key] = sp.lambdify(self._symbols, expr, modules="numpy")
        return self._lam[key]

    def _eval(self, coords, name, deriv_key):  # type: ignore[no-untyped-def]
        cols = _to_columns(coords)
        raw = self._fn(name, deriv_key)(*cols)
        out = np.array(np.broadcast_to(np.asarray(raw, dtype=np.float64), (cols[0].shape[0],)))
        return _from_numpy(out, coords)

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        return self._eval(state.coords, name, ())

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        return self._eval(state.coords, name, ((int(axis), int(order)),))

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        key = tuple((int(a), int(o)) for a, o in zip(axes, orders, strict=False))
        return self._eval(state.coords, name, key)


def make_sympy_field(
    backend: str,
    axes: Sequence[str],
    exprs: Mapping[str, sp.Expr],
    symbols: Mapping[str, sp.Symbol],
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
    time_axis: str | None | object = ...,
) -> SympyField:
    """Build a :class:`SympyField` for ``"torch"`` or ``"jax"``."""
    if backend == "torch":
        from omnibias.fields.torch import _ops_dispatch as ops
    elif backend == "jax":
        from omnibias.fields.jax import _ops_dispatch as ops
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown backend {backend!r}")
    coord_spec = CoordinateSpec(tuple(axes), time_axis=time_axis)
    components = ComponentSpec(tuple(exprs), groups=groups)
    symbol_tuple = tuple(symbols[a] for a in axes)
    return SympyField(coord_spec, components, exprs, symbol_tuple, ops)

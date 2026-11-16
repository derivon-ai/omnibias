# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX half of the opt-in planner routing of deep fields (see :mod:`omnibias.fields._core.planned`).

Compiled programs take the layers and the points as runtime inputs, so one program serves every weight value and every
call of the same shapes; it is cached by structure (activations, shapes, dtype, operator, axes), least recently used
first out once :data:`PROGRAM_CACHE_SIZE` programs are held.  The diffusion is a constant of a program (the planner reads
its metric on the host); a device array is read back once per array object, not on every call.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
from omnibias.core.planner import PlanCache
from omnibias.fields._core.planned import request_chain

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

_CACHE = PlanCache()
#: Compiled programs kept between calls; ``0`` keeps none.
PROGRAM_CACHE_SIZE = 64
_PROGRAMS: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_LOCK = threading.Lock()
_DIFFUSIONS: dict[int, tuple[Any, np.ndarray, tuple[Any, ...]]] = {}


def _host_diffusion(diffusion: Any) -> tuple[np.ndarray, tuple[Any, ...]]:
    """The diffusion as a host array and its key; a concrete device array is transferred once while it is alive."""
    import jax

    if isinstance(diffusion, jax.Array) and not isinstance(diffusion, jax.core.Tracer):
        hit = _DIFFUSIONS.get(id(diffusion))
        if hit is not None and hit[0]() is diffusion:
            return hit[1], hit[2]
        host = np.array(diffusion)  # a copy: a zero-copy view would keep the device array (the weakref's referent) alive
        key = (host.shape, str(host.dtype), host.tobytes())
        ident = id(diffusion)
        _DIFFUSIONS[ident] = (weakref.ref(diffusion, lambda _r, k=ident: _DIFFUSIONS.pop(k, None)), host, key)
        return host, key
    host = np.asarray(diffusion)
    return host, (host.shape, str(host.dtype), host.tobytes())


def _structure(layers: list[tuple[Any, Any, Any]], x: Any) -> tuple[Any, ...]:
    return (
        tuple(
            (tuple(W.shape), None if b is None else tuple(b.shape), spec) for W, b, spec in layers
        ),
        tuple(x.shape),
        str(x.dtype),
    )


def routed(state: FieldState, names: tuple[str, ...], operator: str, *, power: int = 1) -> Any:
    """``operator`` of the components ``names`` through the planner, or ``None`` when the field does not opt in."""
    chain = request_chain(state, names)
    if chain is None:
        return None
    import jax.numpy as jnp
    from omnibias.jax.experimental import compile_operator

    x = jnp.asarray(state.coords if chain.coords is None else chain.coords)
    layers = [
        (jnp.asarray(W), None if b is None else jnp.asarray(b), spec) for W, b, spec in chain.layers
    ]
    coordinates = None if chain.coordinates is None else tuple(int(a) for a in chain.coordinates)
    diffusion, diffusion_key = (None, None) if chain.diffusion is None else _host_diffusion(chain.diffusion)
    key = (operator, power, coordinates, diffusion_key, *_structure(layers, x))
    with _LOCK:
        program = _PROGRAMS.get(key)
        if program is not None:
            _PROGRAMS.move_to_end(key)
    if program is None:
        program = compile_operator(
            layers,
            x,
            operator=operator,
            power=power,
            coordinates=coordinates,
            diffusion=diffusion,
            cache=_CACHE,
        )
        with _LOCK:
            _PROGRAMS[key] = program
            _PROGRAMS.move_to_end(key)
            while len(_PROGRAMS) > max(PROGRAM_CACHE_SIZE, 0):
                _PROGRAMS.popitem(last=False)
    value = program(layers, x)
    return (
        value[..., 0] if operator == "laplacian" and len(names) == 1 and value.ndim == 2 else value
    )

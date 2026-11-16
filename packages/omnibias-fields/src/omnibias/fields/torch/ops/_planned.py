# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch half of the opt-in planner routing of deep fields (see :mod:`omnibias.fields._core.planned`).

A compiled operator is kept between calls when the layer tensors the field hands over are the same leaf tensors as on the
call that built it (a training loop reads the same parameters every step): the call then skips planning, the executor
probe and the construction of the program, which cost about as much as evaluating it on a small batch.  The operator
reads its layers by reference, so in-place updates of the parameters reach it.  A field that returns new tensors on every
call (a row selection, a rescaled copy) rebuilds on every call exactly as before and only replaces the one entry of its
structure; tensors that carry an autograd history are never cached.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
from omnibias.core.planner import PlanCache
from omnibias.fields._core.planned import request_chain
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

_CACHE = PlanCache()

#: Operators kept between calls (least recently used first out); ``0`` disables the reuse.
OPERATOR_CACHE_SIZE = 8
_OPERATORS: OrderedDict[tuple[Any, ...], tuple[tuple[weakref.ref[Tensor], ...], Any]] = OrderedDict()
_LOCK = threading.Lock()


def _layer_tensors(layers: Any) -> list[Tensor] | None:
    """The weight and bias tensors of ``layers``, or ``None`` if one is not a leaf tensor (no reuse then)."""
    out: list[Tensor] = []
    for W, b, _ in layers:
        for t in (W, b):
            if t is None:
                continue
            if not isinstance(t, Tensor) or not t.is_leaf:
                return None
            out.append(t)
    return out


def _diffusion_key(diffusion: Any) -> Any:
    if diffusion is None:
        return None
    if isinstance(diffusion, Tensor):
        if diffusion.requires_grad or diffusion.device.type != "cpu":
            return False  # a value that lives on a device (or trains) is not read back to make a key
        diffusion = diffusion.detach().numpy()
    arr = np.ascontiguousarray(diffusion)
    return (arr.shape, str(arr.dtype), arr.tobytes())


def _operator_key(
    chain: Any, tensors: list[Tensor], x: Tensor, operator: str, power: int
) -> tuple[Any, ...] | None:
    diffusion = _diffusion_key(chain.diffusion)
    if diffusion is False:
        return None
    structure = tuple(
        (tuple(W.shape), None if b is None else tuple(b.shape), spec) for W, b, spec in chain.layers
    )
    coordinates = None if chain.coordinates is None else tuple(int(a) for a in chain.coordinates)
    return (
        operator,
        power,
        coordinates,
        diffusion,
        structure,
        tuple((t.dtype, t.device, t.requires_grad) for t in tensors),
        tuple(x.shape),
        x.dtype,
        x.device,
        x.requires_grad,
    )


def _build(chain: Any, x: Tensor, operator: str, power: int) -> Any:
    from omnibias.torch.experimental import compile_planned_operator

    return compile_planned_operator(
        list(chain.layers),
        x,
        operator=operator,
        power=power,
        coordinates=chain.coordinates,
        diffusion=chain.diffusion,
        cache=_CACHE,
    )


def _operator(chain: Any, x: Tensor, operator: str, power: int) -> Any:
    tensors = _layer_tensors(chain.layers) if OPERATOR_CACHE_SIZE > 0 and isinstance(x, Tensor) else None
    key = None if tensors is None else _operator_key(chain, tensors, x, operator, power)
    if key is None:
        return _build(chain, x, operator, power)
    with _LOCK:
        hit = _OPERATORS.get(key)
        if hit is not None:
            refs, op = hit
            if len(refs) == len(tensors) and all(r() is t for r, t in zip(refs, tensors, strict=True)):
                _OPERATORS.move_to_end(key)
                return op
            del _OPERATORS[key]  # the parameters of this structure were replaced
    op = _build(chain, x, operator, power)
    with _LOCK:
        _OPERATORS[key] = (tuple(weakref.ref(t) for t in tensors), op)
        while len(_OPERATORS) > OPERATOR_CACHE_SIZE:
            _OPERATORS.popitem(last=False)
    return op


def routed(
    state: FieldState, names: tuple[str, ...], operator: str, *, power: int = 1
) -> Tensor | None:
    """``operator`` of the components ``names`` through the planner, or ``None`` when the field does not opt in.

    A single scalar component returns shape ``(B,)``; a divergence returns ``(B,)``.
    """
    chain = request_chain(state, names)
    if chain is None:
        return None
    x = state.coords if chain.coords is None else chain.coords
    op = _operator(chain, x, operator, power)
    value = op.evaluate(x).value
    return value[..., 0] if operator == "laplacian" and len(names) == 1 else value

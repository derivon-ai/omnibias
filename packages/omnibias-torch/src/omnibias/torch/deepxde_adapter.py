# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Planned derivative operators for a DeepXDE network, usable inside a ``pde(x, y)`` residual.

DeepXDE (PyTorch backend) forms ``dde.grad.hessian(y, x, ...)`` by differentiating the graph from the network outputs ``y`` to
the inputs ``x``. That cannot be replaced from inside ``pde(x, y)``: by then only the tensors are visible, not the network that
made them. :class:`DeepXDEOperators` therefore takes the network itself, and its methods are the planned operators of
:func:`omnibias.torch.operators.compile_operator` evaluated at ``x``::

    net = dde.nn.FNN([2, 64, 64, 1], "tanh", "Glorot normal")
    ops = DeepXDEOperators(net)

    def pde(x, y):
        return ops.laplacian(x) + source(x)  # instead of dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1)

    data = dde.data.PDE(geometry, pde, bcs, ...)
    model = dde.Model(data, net)

The shapes and meaning follow DeepXDE's: the result is ``(batch, 1)`` and the Laplacian is the sum over *all* input columns of
the chosen output ``component``. Gradients reach the network parameters and ``x``, so the loss trains as before. The plan reads
the network's *live* weights on every call, so optimizer updates are seen.

A plain ``FNN`` (or any module with ``linears`` and one activation, or one per layer) whose activations are known to omnibias
is read as a dense layer chain. A network with an input or output transform (``apply_feature_transform``,
``apply_output_transform``, for instance a hard boundary condition) or an unknown activation is **not** a chain: it runs on the
planner's generic autodiff executors (exact, but no faster than DeepXDE), and ``explain()["fallback_reason"]`` says why. Only
the Laplacian (and its powers) and the divergence are provided; mixed second derivatives such as ``hessian(y, x, i=0, j=1)``
stay with ``dde.grad``.

This module imports nothing from DeepXDE, so it works with any network that has the interface above. It was tested with
DeepXDE's PyTorch backend (``tests/test_interop_adapters.py``); the other DeepXDE backends are not supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from omnibias.core.planner import PlanCache
from omnibias.torch.experimental.planned import compile_planned_operator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DeepXDEOperators"]


_KNOWN = frozenset({"tanh", "sin", "sigmoid", "silu", "gelu", "relu", "elu", "selu"})


def _activation_name(function: Any) -> str | None:
    """The omnibias activation a DeepXDE activation callable stands for, or ``None`` when it is not recognised.

    DeepXDE binds its PyTorch activations as small functions in ``deepxde.backend`` that call ``torch.nn.functional``; a user
    may also pass the ``torch`` function itself. Both are recognised by name and origin, never by guessing at a lambda.
    """
    if function in (torch.tanh, torch.sin, torch.sigmoid, F.silu, F.gelu, F.relu, F.elu, F.selu):
        return str(function.__name__)
    module = getattr(function, "__module__", "") or ""
    name = getattr(function, "__name__", "")
    if module.startswith("deepxde.backend") and name in _KNOWN:
        return str(name)
    return None


class _ChainView:
    """A live ``(W, b, activation)`` list of an ``FNN``, optionally keeping only some output rows."""

    def __init__(self, net: nn.Module, activations: list[str], rows: Sequence[int] | None) -> None:
        self.net, self.activations, self.rows = net, activations, rows

    def _layer_specs(self) -> list[tuple[Tensor, Tensor | None, str | None]]:
        linears = list(self.net.linears)  # type: ignore[arg-type]
        specs: list[tuple[Tensor, Tensor | None, str | None]] = []
        for j, linear in enumerate(linears):
            weight, bias = linear.weight, linear.bias
            last = j == len(linears) - 1
            if last and self.rows is not None:
                weight = weight[list(self.rows)]
                bias = None if bias is None else bias[list(self.rows)]
            specs.append((weight, bias, None if last else self.activations[j]))
        return specs


class _Outputs(nn.Module):
    """``net(x)[..., rows]``: the generic route for a network that is not a plain chain."""

    def __init__(self, net: nn.Module, rows: Sequence[int]) -> None:
        super().__init__()
        self.net, self.rows = net, list(rows)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)[..., self.rows]
        return out[..., 0] if len(self.rows) == 1 else out


class DeepXDEOperators:
    """Planned Laplacian and divergence of the outputs of a DeepXDE network at ``x`` (see the module docstring).

    ``options`` are passed to :func:`omnibias.torch.operators.compile_operator` (``force``, ``mode``, ``precision`` ...).
    """

    def __init__(self, net: nn.Module, **options: Any) -> None:
        self.net = net
        self.options = options
        self._cache = PlanCache()
        self._ops: dict[tuple[Any, ...], Any] = {}
        self.chain_refusal = self._why_not_a_chain(net)

    def _why_not_a_chain(self, net: nn.Module) -> str | None:
        if not hasattr(net, "linears") or not hasattr(net, "activation"):
            return "the network has no `linears` and `activation` (not a DeepXDE FNN)"
        if getattr(net, "_input_transform", None) is not None:
            return "the network applies an input (feature) transform"
        if getattr(net, "_output_transform", None) is not None:
            return "the network applies an output transform (for example a hard boundary condition)"
        activation = net.activation
        funcs = list(activation) if isinstance(activation, list) else [activation]
        unknown = [getattr(f, "__name__", repr(f)) for f in funcs if _activation_name(f) is None]
        if unknown:
            return f"activation(s) {unknown} are not known to omnibias"
        return None

    def _activations(self) -> list[str]:
        n_hidden = len(self.net.linears) - 1  # type: ignore[arg-type]
        activation = self.net.activation
        funcs = list(activation) if isinstance(activation, list) else [activation] * n_hidden
        return [str(_activation_name(f)) for f in funcs[:n_hidden]]

    def _operator(self, kind: str, x: Tensor, rows: tuple[int, ...], power: int) -> Any:
        key = (kind, rows, power, tuple(x.shape[1:]), x.dtype, x.device)
        op = self._ops.get(key)
        if op is None:
            if self.chain_refusal is None:
                model: Any = _ChainView(self.net, self._activations(), rows)
            else:
                model = _Outputs(self.net, rows)
            op = compile_planned_operator(
                model,
                x.detach(),
                operator=kind,
                power=power,
                cache=self._cache,
                **self.options,
            )
            self._ops[key] = op
        return op

    def laplacian(self, x: Tensor, component: int = 0, power: int = 1) -> Tensor:
        """``sum_i d^{2p} y_component / dx_i^{2p}`` at every row of ``x``, shape ``(batch, 1)``."""
        op = self._operator("laplacian", x, (component,), power)
        return cast(Tensor, op(x)).reshape(-1, 1)

    def divergence(self, x: Tensor, components: Sequence[int] | None = None) -> Tensor:
        """``sum_i d y_{components[i]} / d x_i`` for a vector field with as many components as inputs, ``(batch, 1)``."""
        rows = tuple(range(x.shape[-1])) if components is None else tuple(components)
        if len(rows) != x.shape[-1]:
            raise ValueError(
                f"the divergence needs one component per input column, got {len(rows)} for {x.shape[-1]}"
            )
        return cast(Tensor, self._operator("divergence", x, rows, 1)(x)).reshape(-1, 1)

    def explain(self, x: Tensor, component: int = 0, operator: str = "laplacian") -> dict[str, Any]:
        """The plan used for these inputs: the chosen program, the alternatives, refusals and any fallback reason."""
        if operator == "divergence":
            self.divergence(x)
            rows = tuple(range(x.shape[-1]))
        else:
            self.laplacian(x, component)
            rows = (component,)
        key = (operator, rows, 1, tuple(x.shape[1:]), x.dtype, x.device)
        explained: dict[str, Any] = dict(self._ops[key].explain())
        if self.chain_refusal is not None:
            explained["adapter_fallback"] = self.chain_refusal
        return explained

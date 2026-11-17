# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Planned Laplacian for a PINA model, with the interface of ``pina.operator.laplacian``.

``pina.operator.laplacian(output_, input_, components, d)`` differentiates a computed output with respect to its input, so it
needs no model. The planned operators do: they evaluate the derivative of the *network* at ``input_``. :func:`laplacian`
therefore takes the model where PINA takes ``output_`` and returns the same labelled tensor (``dd<component>``)::

    from pina.operator import laplacian as pina_laplacian
    from omnibias.torch.pina_adapter import laplacian

    def poisson(input_, output_):
        lap_u = laplacian(model, input_, components=["u"], d=["x", "y"], output_labels=["u"])
        return lap_u - forcing(input_)  # instead of pina_laplacian(output_, input_, components=["u"], d=["x", "y"])

``model`` is the network a solver wraps (``solver.model``), a ``FeedForward`` or a plain ``nn.Sequential`` of ``nn.Linear`` and
activation modules (``nn.Tanh``, ``nn.SiLU``, ``nn.Sigmoid``, ``nn.Softplus``, ``nn.GELU``, ``nn.ReLU``). It is read as a dense
layer chain when it has that form; any other module runs on the planner's generic autodiff executors (exact, no faster than
PINA's own operator) and ``explain`` says why. ``output_labels`` names the model's output columns, which PINA keeps on the
output tensor and a bare module does not have. ``d`` restricts the Laplacian to a subset of the input labels exactly as in
PINA (the other coordinates are held fixed). Gradients reach the model parameters and ``input_``.

Only the Laplacian is provided; ``grad``, ``div`` and ``advection`` stay with PINA. This module imports nothing from PINA at
module level, so it can be imported without it; calling :func:`laplacian` needs ``pina.LabelTensor``. It was tested with the
PINA release named in ``tests/test_interop_adapters.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from omnibias.core.planner import PlanCache
from omnibias.torch.experimental.planned import chain_from_module, compile_planned_operator

import torch
from torch import Tensor, nn

__all__ = ["explain", "laplacian"]

_CACHE = PlanCache()


def _as_names(value: str | Sequence[str] | None, available: Sequence[str], what: str) -> list[str]:
    if value is None:
        return list(available)
    names = [value] if isinstance(value, str) else list(value)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"{what} {missing} are not among {list(available)}")
    return names


def _network(model: nn.Module) -> nn.Module:
    """The part of a PINA ``FeedForward`` the planner can read; any other module is used as it is."""
    inner = getattr(model, "model", None)
    return inner if isinstance(inner, nn.Sequential) else model


class _ChainColumn:
    """A live layer list of a ``Sequential`` chain that keeps one output row of the last layer."""

    def __init__(self, accessor: Callable[[], list[Any]], column: int) -> None:
        self.accessor, self.column = accessor, column

    def _layer_specs(self) -> list[Any]:
        layers = self.accessor()
        weight, bias, spec = layers[-1]
        layers[-1] = (
            weight[self.column : self.column + 1],
            None if bias is None else bias[self.column : self.column + 1],
            spec,
        )
        return layers


class _Column(nn.Module):
    """``model(x)[..., column]``: the generic route for a module that is not a plain chain."""

    def __init__(self, model: nn.Module, column: int) -> None:
        super().__init__()
        self.model, self.column = model, column

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.model(x)[..., self.column]
        return out


def _planner_model(net: nn.Module, column: int) -> Any:
    accessor = chain_from_module(net) if isinstance(net, nn.Sequential) else None
    return _Column(net, column) if accessor is None else _ChainColumn(accessor, column)


def laplacian(
    model: nn.Module,
    input_: Any,
    components: str | Sequence[str] | None = None,
    d: str | Sequence[str] | None = None,
    *,
    output_labels: Sequence[str],
    **options: Any,
) -> Any:
    """``pina.operator.laplacian`` of ``model`` at ``input_`` (a ``LabelTensor``), computed by the planner.

    ``options`` are passed to :func:`omnibias.torch.operators.compile_operator`.
    """
    from pina import LabelTensor  # type: ignore[import-not-found,unused-ignore]

    labels_in = list(input_.labels)
    components_list = _as_names(components, output_labels, "components")
    d_list = _as_names(d, labels_in, "d")
    coordinates = None if len(d_list) == len(labels_in) else [labels_in.index(n) for n in d_list]
    plain = input_.as_subclass(Tensor)
    net = _network(model)
    results = []
    for name in components_list:
        op = compile_planned_operator(
            _planner_model(net, list(output_labels).index(name)),
            plain.detach(),
            operator="laplacian",
            coordinates=coordinates,
            cache=_CACHE,
            **options,
        )
        results.append(op(plain).reshape(-1, 1))
    return LabelTensor(torch.cat(results, dim=-1), labels=[f"dd{c}" for c in components_list])


def explain(model: nn.Module, input_: Tensor, **options: Any) -> dict[str, Any]:
    """The plan for the full Laplacian of the first output at ``input_``: program, alternatives, refusals, fallback reason."""
    plain = input_.as_subclass(Tensor).detach()
    op = compile_planned_operator(
        _planner_model(_network(model), 0), plain, operator="laplacian", cache=_CACHE, **options
    )
    return dict(op.explain())

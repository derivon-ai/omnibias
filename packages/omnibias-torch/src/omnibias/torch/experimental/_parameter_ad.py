# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parameter differentiation of generated programs, outside spatial AD."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils import _pytree


def _canonical(program: Any) -> nn.Module:
    factory = getattr(program, "_canonical_factory", None)
    if factory is None:
        raise ValueError("this program has no canonical native graph factory")
    canonical = cast(nn.Module, factory())
    # Recomputing callbacks could repeat effects. The retained FX body is pure;
    # callbacks on the outer user-facing wrapper are still ordinary hooks.
    for module in canonical.modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("parameter AD cannot replay hooks inside the canonical graph")
    return canonical


class _FunctionalProgram:
    def __init__(self, program: nn.Module) -> None:
        self.canonical = _canonical(program)
        self.parameter_names = tuple(name for name, _ in self.canonical.named_parameters())
        missing = set(self.parameter_names) - dict(program.named_parameters()).keys()
        if missing:
            raise ValueError(f"canonical parameter names differ from the program: {sorted(missing)}")

    def state(self, program: nn.Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        parameters: dict[str, Tensor] = dict(program.named_parameters())
        live_buffers = dict(program.named_buffers())
        buffers = {name: live_buffers.get(name, value)
                   for name, value in self.canonical.named_buffers()}
        return parameters, buffers

    def __call__(self, parameters: dict[str, Tensor], buffers: dict[str, Tensor],
                 args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        selected = {name: parameters[name] for name in self.parameter_names}
        return torch.func.functional_call(self.canonical, (selected, buffers), args, kwargs,
                                          tie_weights=True, strict=True)


@dataclass
class _Invocation:
    native: _FunctionalProgram
    compiled: Callable[..., Any]
    leaves: list[Any]
    structure: Any
    slots: dict[int, int]
    output_leaves: list[Any] = field(default_factory=list)
    output_structure: Any = None
    output_slots: list[int] = field(default_factory=list)

    def unpack(self, tensors: tuple[Tensor, ...]) -> Any:
        return _pytree.tree_unflatten(
            [tensors[self.slots[i]] if i in self.slots else leaf
             for i, leaf in enumerate(self.leaves)], self.structure)


class _HybridBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, invocation: _Invocation, *tensors: Tensor) -> tuple[Tensor, ...]:
        ctx.invocation = invocation
        ctx.save_for_backward(*tensors)
        ctx.set_materialize_grads(False)
        # The custom bridge owns differentiation. Its pure forward needs only
        # values; fresh detached views also avoid Dynamo inspecting .grad on
        # nonleaf arguments. Originals remain saved for the native pullback.
        forward_values = tuple(t.detach() for t in tensors)
        output = invocation.compiled(*invocation.unpack(forward_values))
        invocation.output_leaves, invocation.output_structure = _pytree.tree_flatten(output)
        invocation.output_slots = [i for i, leaf in enumerate(invocation.output_leaves)
                                   if isinstance(leaf, Tensor)]
        result = tuple(invocation.output_leaves[i] for i in invocation.output_slots)
        ctx.mark_non_differentiable(*(x for x in result if not x.is_floating_point()))
        return result

    @staticmethod
    def backward(ctx: Any, *cotangents: Tensor | None) -> tuple[Any, ...]:
        invocation = ctx.invocation
        higher = torch.is_grad_enabled()
        saved = ctx.saved_tensors  # Includes the usual tensor version checks.
        with torch.enable_grad():
            # Distinct formal arguments may depend on each other. Cloning each
            # unique input isolates their partials without detaching higher AD.
            clones = tuple(t.clone() for t in saved)
            active = [i for i, t in enumerate(clones) if t.requires_grad]
            output = invocation.native(*invocation.unpack(clones))
            leaves, structure = _pytree.tree_flatten(output)
            if structure != invocation.output_structure:
                raise RuntimeError("hybrid recomputation changed the output structure")
            selected = [(leaves[i], bar) for i, bar in zip(invocation.output_slots, cotangents, strict=True)
                        if bar is not None and leaves[i].requires_grad]
            gradients: dict[int, Tensor | None] = {}
            if selected and active:
                values, bars = zip(*selected, strict=True)
                computed = torch.autograd.grad(values, [clones[i] for i in active], bars,
                                               create_graph=higher, allow_unused=True)
                gradients = dict(zip(active, computed, strict=True))
            # Connected zeros allow further gradients through constant partials.
            result = [gradients.get(i) if gradients.get(i) is not None else saved[i] * 0
                      for i in active]
            by_slot = dict(zip(active, result, strict=True))
        return (None, *(by_slot.get(i) for i in range(len(saved))))


class HybridProgram(nn.Module):
    """Compiled pure forward with native, differentiable backward recomputation."""

    def __init__(self, program: nn.Module, **compile_options: Any) -> None:
        super().__init__()
        self.program = program
        self._native = _FunctionalProgram(program)
        compile_options.setdefault("fullgraph", True)
        self._compiled = torch.compile(self._native, **compile_options)

    @torch.compiler.disable(reason="hybrid backward recomputation lives outside outer compilation")  # type: ignore[no-untyped-call,untyped-decorator]
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        parameters, buffers = self._native.state(self.program)
        leaves, structure = _pytree.tree_flatten((parameters, buffers, args, kwargs))
        tensors: list[Tensor] = []
        identities: dict[int, int] = {}
        slots = {}
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, Tensor):
                if id(leaf) not in identities:
                    identities[id(leaf)] = len(tensors)
                    tensors.append(leaf)
                slots[i] = identities[id(leaf)]
        invocation = _Invocation(self._native, self._compiled, leaves, structure, slots)
        outputs = _HybridBridge.apply(invocation, *tensors)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        result = list(invocation.output_leaves)
        for i, output in zip(invocation.output_slots, outputs, strict=True):
            result[i] = output
        return _pytree.tree_unflatten(result, invocation.output_structure)

    def explain(self) -> dict[str, Any]:
        return {**self.program.explain(), "execution": "hybrid",  # type: ignore[operator]
                "outer_fullgraph": False, "backward": "native differentiable recomputation",
                "parameter_higher_gradients": True}


class ParameterHVP(nn.Module):
    """Compile a parameter Hessian-vector product before backend lowering."""

    def __init__(self, program: nn.Module, loss_fn: Callable[..., Tensor], **compile_options: Any) -> None:
        super().__init__()
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable and return a scalar tensor")
        self.program = program
        self._native = _FunctionalProgram(program)

        def loss(parameters: dict[str, Tensor], buffers: dict[str, Tensor],
                 args: tuple[Any, ...], kwargs: dict[str, Any], loss_args: tuple[Any, ...]) -> Tensor:
            value = loss_fn(self._native(parameters, buffers, args, kwargs), *loss_args)
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            return value

        gradient = torch.func.grad(loss)

        def evaluate(parameters: dict[str, Tensor], buffers: dict[str, Tensor],
                     args: tuple[Any, ...], kwargs: dict[str, Any], loss_args: tuple[Any, ...],
                     tangents: dict[str, Tensor]) -> dict[str, Tensor]:
            def g(p: dict[str, Tensor]) -> dict[str, Tensor]:
                return cast(dict[str, Tensor], gradient(p, buffers, args, kwargs, loss_args))
            return cast(dict[str, Tensor], torch.func.jvp(g, (parameters,), (tangents,))[1])

        compile_options.setdefault("fullgraph", True)
        self._compiled = torch.compile(evaluate, **compile_options)

    def forward(self, tangents: Mapping[str, Tensor], *args: Any,
                loss_args: tuple[Any, ...] = (), **kwargs: Any) -> dict[str, Tensor]:
        parameters, buffers = self._native.state(self.program)
        if not isinstance(tangents, Mapping) or tangents.keys() != parameters.keys():
            raise ValueError("HVP tangents must have exactly the program's unique named parameter keys")
        ordered = {}
        for name, parameter in parameters.items():
            tangent = tangents[name]
            if not isinstance(tangent, Tensor) or (tangent.shape, tangent.dtype, tangent.device) != (
                    parameter.shape, parameter.dtype, parameter.device):
                raise ValueError(f"HVP tangent {name!r} must match its parameter shape, dtype and device")
            ordered[name] = tangent
        if not isinstance(loss_args, tuple):
            raise TypeError("loss_args must be a tuple")
        if not parameters:
            return {}
        return self._compiled(parameters, buffers, args, kwargs, loss_args, ordered)

    def explain(self) -> dict[str, Any]:
        return {**self.program.explain(), "execution": "compiled-parameter-hvp",  # type: ignore[operator]
                "canonical_lowering": "inline-native", "parameter_hessian": "inputs and buffers fixed",
                "parameter_keys": tuple(dict(self.program.named_parameters()))}

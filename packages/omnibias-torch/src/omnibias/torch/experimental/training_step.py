# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A captured PyTorch training step: residual blocks, loss, backward and optimizer in one CUDA-graph replay.

The eager loop of a small PINN is launch-bound: the sampler, every residual block, the backward pass and the optimizer
are hundreds of tiny kernels with the host between them.  :class:`TrainingStep` runs ``steps_per_call`` steps inside one
CUDA graph:

* **Static buffers.**  Parameters, gradients, optimizer state and the metric history are fixed tensors the graph reads
  and writes in place; nothing is allocated or synchronised per step.
* **Horizontal grouping.**  Blocks that share one ``evaluate`` callable have their points concatenated and are served
  by one launch, as in the JAX step.
* **Device-side sampling.**  Points are drawn by ``sample`` from the default CUDA generator inside the graph (PyTorch
  advances the generator state per replay), so the host never prepares the next batch.
* **Fused optimizer.**  The default optimizer is ``torch.optim.Adam(fused=True, capturable=True)`` on CUDA.
* **No host syncs.**  Gradient clipping uses a device-side scale; metrics land in a device tensor that is read lazily.

On CPU, or when capture is not possible (an optimizer that is not capturable, a block that synchronises), the same step
runs eagerly and :meth:`TrainingStep.explain` records the reason.  The numbers are identical either way.

Ensembles (``n_models`` in the JAX step) are not offered here: stack models with ``torch.func`` or run one process per
model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["StepBlock", "TrainingStep", "compile_training_step", "planned_evaluator"]


@dataclass(frozen=True)
class StepBlock:
    """One residual term: ``weight * mean(residual(params, x, evaluated, aux) ** 2)``.

    ``sample() -> x`` draws ``(n, D)`` points from the default generator (parameter free; it runs inside the graph).
    ``prepare(x) -> aux`` is optional parameter-free preprocessing.  ``evaluate(params, x) -> pytree`` computes the
    derivative quantities batched over the leading axis; blocks sharing the *same* callable are one launch (``None``:
    the residual receives ``None``).
    """

    name: str
    sample: Callable[[], Tensor]
    residual: Callable[..., Tensor]
    evaluate: Callable[[Any, Tensor], Any] | None = None
    prepare: Callable[[Tensor], Any] | None = None
    weight: float = 1.0
    order: int = 0


def planned_evaluator(
    model: Any,
    params: Any,
    x: Tensor,
    *,
    steps: int,
    operator: str = "laplacian",
    **kwargs: Any,
) -> Callable[[Any, Tensor], Any]:
    """The planned operator of ``model`` as ``evaluate(params, x)``, planned for training.

    The training preset: ``compile_planned_operator`` is asked for ``differentiate="params"``, ``expected_uses=steps``,
    ``mode="find"`` and ``top_k=None`` unless overridden, so every feasible program is timed once on the
    value-and-gradient the step runs and the measured best is kept in the persistent plan cache.  ``model`` is a layer
    list ``[(W, b, spec), ...]`` whose weight tensors require gradients;
    ``params`` is accepted for signature parity with the JAX step and ignored (the layer tensors are the parameters).
    """
    from omnibias.torch.experimental.planned import compile_planned_operator

    kwargs.setdefault("differentiate", "params")
    kwargs.setdefault("expected_uses", int(steps))
    kwargs.setdefault("mode", "find")
    kwargs.setdefault("top_k", None)
    op = compile_planned_operator(model, x, operator=operator, **kwargs)
    return lambda _params, z: op(z)


def _slice_tree(tree: Any, start: int, stop: int) -> Any:
    return torch.utils._pytree.tree_map(lambda a: a[start:stop], tree)


class TrainingStep:
    """A training step compiled into a CUDA graph (or run eagerly); build it with :func:`compile_training_step`."""

    def __init__(
        self,
        blocks: Sequence[StepBlock],
        params: Sequence[Tensor],
        optimizer: torch.optim.Optimizer | None,
        *,
        steps_per_call: int,
        graph: bool | str,
        clip_norm: float | None,
        lr: float,
        warmup: int,
        params_view: Any,
    ) -> None:
        if not blocks:
            raise ValueError("at least one block is required")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique, got {names}")
        if steps_per_call < 1:
            raise ValueError("steps_per_call must be at least 1")
        self.blocks = tuple(blocks)
        self.params = list(params)
        if not self.params or not all(
            isinstance(p, Tensor) and p.requires_grad for p in self.params
        ):
            raise ValueError(
                "params must be a non-empty sequence of leaf tensors with requires_grad=True"
            )
        self._view = self.params if params_view is None else params_view
        self.steps_per_call = int(steps_per_call)
        self._clip = clip_norm
        device = self.params[0].device
        self._device = device
        on_cuda = device.type == "cuda"
        if optimizer is None:
            optimizer = (
                torch.optim.Adam(self.params, lr=lr, fused=True, capturable=True)
                if on_cuda
                else torch.optim.Adam(self.params, lr=lr)
            )
        self.optimizer = optimizer
        self._groups = self._group()
        nb = len(self.blocks)
        self.history = torch.zeros(
            self.steps_per_call, nb + 2, dtype=self.params[0].dtype, device=device
        )
        self._graph: torch.cuda.CUDAGraph | None = None
        self._why_eager: str | None = None
        self._capturable = self._optimizer_capturable()
        want = graph is True or (graph == "auto" and on_cuda)
        if want and not on_cuda:
            self._why_eager = "parameters are on the CPU; CUDA graphs need a CUDA device"
        elif want and not self._capturable:
            self._why_eager = (
                "the optimizer is not capturable (use Adam(capturable=True, fused=True))"
            )
        elif want:
            try:
                self._capture(warmup)
            except Exception as err:  # noqa: BLE001 - recorded; graph=True re-raises
                if graph is True:
                    raise
                self._graph = None
                self._why_eager = f"capture failed: {type(err).__name__}: {str(err)[:200]}"
        else:
            self._why_eager = "graph=False" if graph is False else "parameters are not on CUDA"

    # ------------------------------------------------------------------------------------------ construction
    def _optimizer_capturable(self) -> bool:
        if self.params[0].device.type != "cuda":
            return False
        groups = self.optimizer.param_groups
        if isinstance(self.optimizer, torch.optim.Adam | torch.optim.AdamW):
            return bool(groups[0].get("capturable", False))
        return isinstance(self.optimizer, torch.optim.SGD)

    def _group(self) -> list[tuple[Callable[[Any, Tensor], Any] | None, list[int]]]:
        groups: list[tuple[Callable[[Any, Tensor], Any] | None, list[int]]] = []
        by_fn: dict[int, list[int]] = {}
        for i, b in enumerate(self.blocks):
            if b.evaluate is None:
                groups.append((None, [i]))
                continue
            members = by_fn.get(id(b.evaluate))
            if members is None:
                members = by_fn[id(b.evaluate)] = []
                groups.append((b.evaluate, members))
            members.append(i)
        return groups

    # --------------------------------------------------------------------------------------------- the math
    def _loss_terms(self) -> tuple[Tensor, Tensor]:
        xs = [b.sample() for b in self.blocks]
        aux = [
            None if b.prepare is None else b.prepare(x)
            for b, x in zip(self.blocks, xs, strict=True)
        ]
        terms: list[Tensor | None] = [None] * len(self.blocks)
        for fn, members in self._groups:
            if fn is None:
                (i,) = members
                b = self.blocks[i]
                terms[i] = b.weight * torch.mean(
                    torch.square(b.residual(self._view, xs[i], None, aux[i]))
                )
                continue
            parts = [xs[i] for i in members]
            joined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            out = fn(self._view, joined)
            start = 0
            for i, x in zip(members, parts, strict=True):
                n = int(x.shape[0])
                piece = out if len(parts) == 1 else _slice_tree(out, start, start + n)
                b = self.blocks[i]
                terms[i] = b.weight * torch.mean(
                    torch.square(b.residual(self._view, x, piece, aux[i]))
                )
                start += n
        per = torch.stack([t for t in terms if t is not None])
        return per.sum(), per

    def _one_step(self, k: int) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss, per = self._loss_terms()
        loss.backward()  # type: ignore[no-untyped-call]
        grads = [p.grad for p in self.params if p.grad is not None]
        norms = torch._foreach_norm(grads)
        gnorm = torch.linalg.vector_norm(torch.stack(norms))
        if self._clip is not None:
            scale = torch.clamp(self._clip / (gnorm + 1e-30), max=1.0)
            torch._foreach_mul_(grads, scale)
        self.optimizer.step()
        nb = len(self.blocks)
        self.history[k, 0] = loss.detach()
        self.history[k, 1 : 1 + nb] = per.detach()
        self.history[k, nb + 1] = gnorm.detach()

    def _block_eager(self) -> None:
        for k in range(self.steps_per_call):
            self._one_step(k)

    # ------------------------------------------------------------------------------------------------ capture
    def _snapshot(self) -> dict[str, Any]:
        state = {
            id(p): {
                k: v.clone() for k, v in self.optimizer.state[p].items() if isinstance(v, Tensor)
            }
            for p in self.params
            if p in self.optimizer.state
        }
        return {"params": [p.detach().clone() for p in self.params], "state": state}

    def _restore(self, snap: dict[str, Any]) -> None:
        with torch.no_grad():
            for p, v in zip(self.params, snap["params"], strict=True):
                p.copy_(v)
            for p in self.params:
                if p not in self.optimizer.state:
                    continue
                saved = snap["state"].get(id(p))
                for k, v in self.optimizer.state[p].items():
                    if not isinstance(v, Tensor):
                        continue
                    if saved is not None and k in saved:
                        v.copy_(saved[k])
                    else:
                        v.zero_()
            self.history.zero_()

    def _capture(self, warmup: int) -> None:
        snap = self._snapshot()
        rng = torch.cuda.get_rng_state(self._device)
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                self._block_eager()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                self._block_eager()
            torch.cuda.synchronize()
        except BaseException:
            self._abandon_capture(graph, snap, rng)
            raise
        self._graph = graph
        self._restore(snap)
        torch.cuda.set_rng_state(rng, self._device)
        torch.cuda.synchronize()

    def _abandon_capture(
        self, graph: torch.cuda.CUDAGraph, snap: dict[str, Any], rng: Tensor
    ) -> None:
        """Undo a failed capture so the eager fallback starts from the initial state.

        A capture that fails part way never reaches PyTorch's end-of-capture bookkeeping, so the default CUDA generator
        stays flagged as capturing and every later draw from it, the eager step's sampling included, raises "Offset
        increment outside graph capture".  Resetting the graph does not clear the flag; giving the generator a fresh
        state does.  Graphs captured earlier keep the state they registered and replay as before.  The warm-up steps
        are undone as after a successful capture.
        """
        for undo in (
            torch.cuda.synchronize,
            graph.reset,
            self._renew_generator_if_broken,
            lambda: self._restore(snap),
            lambda: torch.cuda.set_rng_state(rng, self._device),
            torch.cuda.synchronize,
            torch.cuda.empty_cache,
        ):
            try:
                undo()
            except Exception:  # noqa: BLE001, S110 - recovery is best effort; the capture error is the one to report
                pass

    def _renew_generator_if_broken(self) -> None:
        device = torch.device(self._device)
        try:
            torch.rand(1, device=device)
            return
        except RuntimeError:
            pass
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.default_generators[index].graphsafe_set_state(
            torch.Generator(device=device).graphsafe_get_state()
        )

    # --------------------------------------------------------------------------------------------- running
    def __call__(self) -> Tensor:
        """One block of ``steps_per_call`` steps; returns the device-side history ``(K, 2 + n_blocks)``.

        Columns: total loss, each block's weighted mean square, gradient norm.  No host synchronisation.
        """
        if self._graph is not None:
            self._graph.replay()
        else:
            self._block_eager()
        return self.history

    def run(
        self,
        n_blocks: int,
        *,
        on_block: Callable[[int, Tensor], None] | None = None,
        synchronize: bool = False,
    ) -> Tensor:
        """Run ``n_blocks`` blocks back to back (replays are queued asynchronously)."""
        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1")
        for j in range(n_blocks):
            self()
            if on_block is not None:
                on_block(j, self.history)
        if synchronize and self._device.type == "cuda":
            torch.cuda.synchronize()
        return self.history

    def step(self) -> Tensor:
        """One eager step (never graph replay): the cheap path for debugging; returns the total loss."""
        self._one_step(0)
        return self.history[0, 0]

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def explain(self) -> dict[str, Any]:
        return {
            "steps_per_call": self.steps_per_call,
            "captured": self.captured,
            "why_eager": self._why_eager,
            "optimizer": type(self.optimizer).__name__,
            "groups": [
                {
                    "launches": 1,
                    "blocks": [self.blocks[i].name for i in members],
                    "orders": sorted({self.blocks[i].order for i in members}),
                    "shared_evaluate": fn is not None and len(members) > 1,
                }
                for fn, members in self._groups
            ],
        }

    def time_steps(self, n_blocks: int = 5, warmup: int = 1) -> dict[str, float]:
        """Wall seconds per step (synchronised at the ends; compilation and capture excluded)."""
        self.run(warmup, synchronize=True)
        t0 = time.perf_counter()
        self.run(n_blocks, synchronize=True)
        dt = time.perf_counter() - t0
        return {"seconds_per_step": dt / (n_blocks * self.steps_per_call), "seconds": dt}


def compile_training_step(
    blocks: Sequence[StepBlock],
    params: Sequence[Tensor],
    optimizer: torch.optim.Optimizer | None = None,
    *,
    steps_per_call: int = 16,
    graph: bool | str = "auto",
    clip_norm: float | None = None,
    lr: float = 1e-3,
    warmup: int = 3,
    params_view: Any = None,
) -> TrainingStep:
    """Compile ``blocks`` with ``optimizer`` into a CUDA-graph training step.

    Parameters
    ----------
    blocks : the residual terms (see :class:`StepBlock`); blocks sharing an ``evaluate`` are one launch.
    params : the leaf tensors the optimizer updates (``requires_grad=True``).  They are updated in place and keep their
        identity, so a model holding them sees every update.
    optimizer : defaults to ``Adam(fused=True, capturable=True)`` on CUDA and plain ``Adam`` on CPU.
    steps_per_call : steps per graph replay (unrolled in the graph).
    graph : ``"auto"`` captures on CUDA when possible, ``True`` requires capture (errors propagate), ``False`` is eager.
    clip_norm : optional global gradient-norm clip (device side, no sync).
    warmup : eager blocks run on a side stream before capture; parameters, optimizer state and the RNG state are
        restored afterwards, so capture does not train the model.
    params_view : what ``residual`` and ``evaluate`` receive as ``params`` (default: the ``params`` list).
    """
    return TrainingStep(
        blocks,
        params,
        optimizer,
        steps_per_call=steps_per_call,
        graph=graph,
        clip_norm=clip_norm,
        lr=lr,
        warmup=warmup,
        params_view=params_view,
    )

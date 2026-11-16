# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One compiled training step: residual blocks, loss, value-and-gradient and optimizer update in a single program.

An eager PINN loop launches the sampler, one derivative program per residual block, the loss, the backward pass and the
optimizer as separate small programs, and the host sits between them.  On a GPU that leaves the device idle for most of a
small step.  :func:`compile_training_step` removes the gaps in four ways:

* **One program per step.**  Every block's residual, the weighted loss, ``jax.value_and_grad`` and the optimizer update
  are traced together, so XLA sees (and fuses across) the whole step.
* **Horizontal grouping.**  Blocks that share one ``evaluate`` callable (the same derivative program, typically the
  planned operator of one order) have their points concatenated and are served by *one* launch; the outputs are sliced
  back per block.  Blocks of different orders stay separate groups.
* **K steps per launch.**  ``steps_per_call`` steps run in one ``lax.scan`` with donated buffers, so the host launches once
  per K steps.
* **Sampling ahead.**  Points and any parameter-free preprocessing (``prepare``: Fourier features, boundary maps) for the
  *next* K-step block are produced by the same executable that trains the current block, from a key that does not depend
  on the parameters.  :meth:`TrainingStep.run` enqueues block after block without a host synchronisation, so the device
  never waits for the host to draw points.

``n_models`` trains an ensemble (seeds or hyper-parameter grids) with one ``vmap``: parameters, optimizer state and the
sampling keys carry a leading model axis, and the points are independent per model unless ``shared_points=True``.

Planning enters through ``evaluate``: pass the callable returned by ``compile_operator(..., differentiate="params",
expected_uses=steps)`` (see :func:`planned_evaluator`), so the algorithm is chosen for the training objective and the
number of steps that will be taken.

The optimizer is any ``(init, update)`` pair with the optax signature (``update(grads, state, params) -> (updates,
state)``); :func:`adam` is a small dependency-free implementation.  Second-order optimizers that need residual rows
(Gauss-Newton, ENGD) are separate programs; this layer is the first-order, matrix-free step.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import numpy as np
from omnibias.jax.experimental.coldcost import compilation_cache_dir, enable_compilation_cache
from omnibias.jax.experimental.streams import training_layout

import jax
import jax.numpy as jnp

Array = jax.Array

__all__ = [
    "Optimizer",
    "StepBlock",
    "TrainState",
    "TrainingStep",
    "adam",
    "compile_training_step",
    "enable_compile_cache",
    "planned_evaluator",
    "sgd",
]


# ------------------------------------------------------------------------------------------------------- optimizers
class Optimizer(NamedTuple):
    """``init(params) -> state`` and ``update(grads, state, params) -> (updates, state)``; params are ``params + updates``."""

    init: Callable[[Any], Any]
    update: Callable[[Any, Any, Any], tuple[Any, Any]]


class _AdamState(NamedTuple):
    steps: Array
    mu: Any
    nu: Any


def adam(
    learning_rate: float | Callable[[Array], Array] = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Optimizer:
    """Adam with bias correction; ``learning_rate`` may be a schedule ``step -> rate`` (a traced step count)."""

    def init(params: Any) -> _AdamState:
        z = jax.tree.map(jnp.zeros_like, params)
        return _AdamState(jnp.zeros((), jnp.int32), z, jax.tree.map(jnp.zeros_like, params))

    def update(grads: Any, state: _AdamState, params: Any = None) -> tuple[Any, _AdamState]:
        count = state.steps + 1
        mu = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, grads)
        nu = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state.nu, grads)
        t = count.astype(jnp.result_type(float))
        c1 = 1.0 - b1**t
        c2 = 1.0 - b2**t
        lr: Any = learning_rate(count) if callable(learning_rate) else learning_rate
        updates = jax.tree.map(lambda m, v: -lr * (m / c1) / (jnp.sqrt(v / c2) + eps), mu, nu)
        return updates, _AdamState(count, mu, nu)

    return Optimizer(init, update)


def sgd(learning_rate: float | Callable[[Array], Array] = 1e-2, momentum: float = 0.0) -> Optimizer:
    """Plain or heavy-ball gradient descent."""

    def init(params: Any) -> Any:
        return (jnp.zeros((), jnp.int32), jax.tree.map(jnp.zeros_like, params))

    def update(grads: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        count, vel = state
        count = count + 1
        lr: Any = learning_rate(count) if callable(learning_rate) else learning_rate
        vel = jax.tree.map(lambda v, g: momentum * v + g, vel, grads)
        return jax.tree.map(lambda v: -lr * v, vel), (count, vel)

    return Optimizer(init, update)


# ------------------------------------------------------------------------------------------------------------ blocks
@dataclass(frozen=True)
class StepBlock:
    """One residual term of the loss: ``weight * mean(residual(params, x, evaluated) ** 2)``.

    ``sample(key) -> x`` draws the block's points (``(n, D)``); it must not depend on the parameters, so it can run
    ahead.  ``prepare(x) -> aux`` is optional parameter-free preprocessing whose result is passed to ``residual``.
    ``evaluate(params, x) -> pytree`` computes the derivative quantities the residual reads, batched over the leading
    axis of ``x``; blocks that share the *same* callable are one launch.  ``None`` means the residual needs no shared
    evaluation and receives ``None``.  ``residual(params, x, evaluated, aux)`` returns the residual vector for the block.
    ``order`` is a label for grouping and reporting.
    """

    name: str
    sample: Callable[[Array], Array]
    residual: Callable[..., Array]
    evaluate: Callable[[Any, Array], Any] | None = None
    prepare: Callable[[Array], Any] | None = None
    weight: float = 1.0
    order: int = 0


class TrainState(NamedTuple):
    """``params`` and ``opt_state`` carry a leading model axis when ``n_models`` was given; ``key`` drives the sampler."""

    params: Any
    opt_state: Any
    key: Array
    step: Array


@dataclass
class _Group:
    evaluate: Callable[[Any, Array], Any] | None
    members: list[int] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)


def _slice_tree(tree: Any, start: int, stop: int) -> Any:
    return jax.tree.map(lambda a: a[start:stop], tree)


def planned_evaluator(
    model: Any,
    params: Any,
    x: Array,
    *,
    steps: int,
    operator: str = "laplacian",
    **kwargs: Any,
) -> Callable[[Any, Array], Any]:
    """The planned operator for ``model`` as an ``evaluate(params, x)`` callable, planned for training.

    The training preset: ``compile_operator`` is asked for ``differentiate="params"``, ``expected_uses=steps``,
    ``mode="find"`` and ``top_k=None`` unless overridden.  Every feasible program is timed once, on the value-and-gradient
    the step runs, and the measured best is kept in the persistent plan cache, so a run of days does not depend on the
    cost model's ranking of the programs it did not time.  Pass ``mode="auto"`` for the cheaper rule (a timed ``find`` of
    the two leading candidates only from 50 expected uses) or ``mode="heuristic"`` to rank without timing.
    """
    from omnibias.jax.experimental.compiler import compile_operator

    kwargs.setdefault("differentiate", "params")
    kwargs.setdefault("expected_uses", int(steps))
    kwargs.setdefault("mode", "find")
    kwargs.setdefault("top_k", None)
    op = compile_operator(model, (params, x), operator=operator, **kwargs)
    return cast("Callable[[Any, Array], Any]", op)


def enable_compile_cache(path: str, *, min_compile_seconds: float = 0.0) -> dict[str, Any]:
    """Point JAX's persistent compilation cache at ``path`` (process wide, never done on import)."""
    return enable_compilation_cache(path, min_compile_seconds=min_compile_seconds)


# ---------------------------------------------------------------------------------------------------- the step
class TrainingStep:
    """A compiled training step; build it with :func:`compile_training_step`."""

    def __init__(
        self,
        blocks: Sequence[StepBlock],
        optimizer: Optimizer,
        *,
        steps_per_call: int,
        donate: bool | None,
        n_models: int | None,
        shared_points: bool,
        unroll: int,
        project: Callable[[Any], Any] | None,
        clip_norm: float | None,
    ) -> None:
        if not blocks:
            raise ValueError("at least one block is required")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique, got {names}")
        if steps_per_call < 1:
            raise ValueError("steps_per_call must be at least 1")
        if n_models is not None and n_models < 1:
            raise ValueError("n_models must be a positive integer")
        self.blocks = tuple(blocks)
        self.optimizer = optimizer
        self.steps_per_call = int(steps_per_call)
        self.n_models = n_models
        self.shared_points = bool(shared_points)
        self._donate = jax.default_backend() != "cpu" if donate is None else bool(donate)
        self._unroll = int(unroll)
        self._project = project
        self._clip = clip_norm
        self._groups = self._group()
        self._compiled: dict[str, Any] = {}
        self._build()

    # ------------------------------------------------------------------------------------------------ construction
    def _group(self) -> list[_Group]:
        groups: list[_Group] = []
        by_fn: dict[int, _Group] = {}
        for i, b in enumerate(self.blocks):
            if b.evaluate is None:
                groups.append(_Group(None, [i]))
                continue
            g = by_fn.get(id(b.evaluate))
            if g is None:
                g = by_fn[id(b.evaluate)] = _Group(b.evaluate)
                groups.append(g)
            g.members.append(i)
        return groups

    def _loss_terms(
        self, params: Any, points: Sequence[Array], aux: Sequence[Any]
    ) -> tuple[Array, Array]:
        """Weighted total and the per-block weighted mean squares (a ``(n_blocks,)`` array)."""
        terms: list[Array | None] = [None] * len(self.blocks)
        for g in self._groups:
            if g.evaluate is None:
                (i,) = g.members
                b = self.blocks[i]
                r = b.residual(params, points[i], None, aux[i])
                terms[i] = b.weight * jnp.mean(jnp.square(r))
                continue
            xs = [points[i] for i in g.members]
            sizes = [int(x.shape[0]) for x in xs]
            joined = xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=0)
            with training_layout():  # programs traced here are differentiated: the stream engines take their training layout
                out = g.evaluate(params, joined)
            start = 0
            for i, n in zip(g.members, sizes, strict=True):
                b = self.blocks[i]
                piece = out if len(xs) == 1 else _slice_tree(out, start, start + n)
                r = b.residual(params, points[i], piece, aux[i])
                terms[i] = b.weight * jnp.mean(jnp.square(r))
                start += n
        per = jnp.stack([cast(Array, t) for t in terms])
        return jnp.sum(per), per

    def _one_step(
        self, state: TrainState, points: Sequence[Array], aux: Sequence[Any]
    ) -> tuple[TrainState, Any]:
        def loss_fn(p: Any) -> tuple[Array, Array]:
            return self._loss_terms(p, points, aux)

        (loss, per), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
        if self._clip is not None:
            scale = jnp.minimum(1.0, self._clip / (gnorm + 1e-30))
            grads = jax.tree.map(lambda g: g * scale, grads)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = jax.tree.map(lambda p, u: p + u, state.params, updates)
        if self._project is not None:
            params = self._project(params)
        return TrainState(params, opt_state, state.key, state.step + 1), {
            "loss": loss,
            "blocks": per,
            "grad_norm": gnorm,
        }

    def _sample_block(self, key: Array) -> tuple[Sequence[Array], Sequence[Any]]:
        """Points and parameter-free preprocessing for ``steps_per_call`` steps: leading axis K."""
        flat = jax.random.split(key, self.steps_per_call * len(self.blocks))
        keys = flat.reshape((self.steps_per_call, len(self.blocks), *flat.shape[1:]))
        points: list[Array] = []
        aux: list[Any] = []
        for i, b in enumerate(self.blocks):
            x = jax.vmap(b.sample)(keys[:, i])
            points.append(x)
            aux.append(None if b.prepare is None else jax.vmap(b.prepare)(x))
        return tuple(points), tuple(aux)

    def _sample_first(self, key: Array) -> tuple[Sequence[Array], Sequence[Any]]:
        """The first step's points and preprocessing of :meth:`_sample_block` (same keys), without the other ``K - 1``."""
        flat = jax.random.split(key, self.steps_per_call * len(self.blocks))
        keys = flat.reshape((self.steps_per_call, len(self.blocks), *flat.shape[1:]))[0]
        points: list[Array] = []
        aux: list[Any] = []
        for i, b in enumerate(self.blocks):
            x = jax.vmap(b.sample)(keys[i][None])
            points.append(x[0])
            aux.append(None if b.prepare is None else jax.vmap(b.prepare)(x)[0])
        return tuple(points), tuple(aux)

    def _scan_block(
        self, state: TrainState, points: Sequence[Array], aux: Sequence[Any]
    ) -> tuple[TrainState, Any]:
        def body(carry: TrainState, xs: Any) -> tuple[TrainState, Any]:
            pts, ax = xs
            return self._one_step(carry, pts, ax)

        return jax.lax.scan(body, state, (tuple(points), tuple(aux)), unroll=self._unroll)

    def _build(self) -> None:
        single = self.n_models is None
        sample = self._sample_block
        scan = self._scan_block

        def one_model(
            state: TrainState, points: Any, aux: Any, key_next: Array
        ) -> tuple[TrainState, Any, Any, Any]:
            new_state, metrics = scan(state, points, aux)
            nxt_points, nxt_aux = sample(key_next)
            return new_state, nxt_points, nxt_aux, metrics

        if single:
            block_fn = one_model
            first_fn = sample
            single_fn = lambda state, pts, ax: self._one_step(state, pts, ax)  # noqa: E731
            first_one_fn = self._sample_first
        else:
            in_ax = (0, 0, 0, 0)
            block_fn = jax.vmap(one_model, in_axes=in_ax)
            first_fn = jax.vmap(sample)
            single_fn = jax.vmap(lambda state, pts, ax: self._one_step(state, pts, ax))
            first_one_fn = jax.vmap(self._sample_first)
        def keyed_block(
            state: TrainState, points: Any, aux: Any, key: Array
        ) -> tuple[TrainState, Any, Any, Any, Array]:
            key, kn = self._split_keys(key)  # the stream of the host loop: the next block's key, and the key after it
            new_state, nxt_points, nxt_aux, metrics = block_fn(state, points, aux, kn)
            return new_state, nxt_points, nxt_aux, metrics, key

        def keyed_first(key: Array) -> tuple[Any, Any, Array]:
            key, k0 = self._split_keys(key)
            points, aux = first_fn(k0)
            return points, aux, key

        donate = (0, 1, 2) if self._donate else ()
        self._block = jax.jit(block_fn, donate_argnums=donate)
        self._block_keyed = jax.jit(keyed_block, donate_argnums=donate)
        self._first = jax.jit(first_fn)
        self._first_keyed = jax.jit(keyed_first)
        self._first_one = jax.jit(first_one_fn)
        self._single = jax.jit(single_fn, donate_argnums=(0,) if self._donate else ())

    # ----------------------------------------------------------------------------------------------------- state
    def init(self, params: Any, key: Array | int = 0) -> TrainState:
        """Initial state.  With ``n_models``, ``params`` must already carry the leading model axis (stack the seeds)."""
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        if self.n_models is None:
            return TrainState(params, self.optimizer.init(params), key, jnp.zeros((), jnp.int32))
        lead = {int(np.shape(leaf)[0]) for leaf in jax.tree.leaves(params)}
        if lead != {self.n_models}:
            raise ValueError(
                f"with n_models={self.n_models} every parameter needs a leading model axis of that length, saw {sorted(lead)}"
            )
        keys = jax.random.split(key, self.n_models)
        if self.shared_points:
            keys = jnp.broadcast_to(key, keys.shape)
        return TrainState(
            params,
            jax.vmap(self.optimizer.init)(params),
            keys,
            jnp.zeros((self.n_models,), jnp.int32),
        )

    def _split_keys(self, key: Array) -> tuple[Array, Array]:
        if self.n_models is None:
            k = jax.random.split(key)
            return k[0], k[1]
        k = jax.vmap(jax.random.split)(key)
        return k[:, 0], k[:, 1]

    # ----------------------------------------------------------------------------------------------------- running
    def __call__(self, state: TrainState) -> tuple[TrainState, Any]:
        """One block of ``steps_per_call`` steps; the points are drawn here (use :meth:`run` to draw them ahead)."""
        state, metrics, _ = self.run(state, 1)
        return state, metrics

    def run(
        self,
        state: TrainState,
        n_blocks: int,
        *,
        on_block: Callable[[int, TrainState, Any], None] | None = None,
        block_until_ready: bool = False,
    ) -> tuple[TrainState, Any, list[Any]]:
        """Train ``n_blocks * steps_per_call`` steps; returns ``(state, last_metrics, all_metrics)``.

        Block ``j+1``'s points are an output of block ``j``'s executable, so nothing waits on the host between blocks.
        ``on_block(j, state, metrics)`` is called after each block is *enqueued* (read the arrays lazily if it must not
        synchronise).  ``block_until_ready`` waits at the end for timing.
        """
        if n_blocks < 1:
            raise ValueError("n_blocks must be at least 1")
        points, aux, key = self._first_keyed(state.key)
        history: list[Any] = []
        metrics: Any = None
        for j in range(n_blocks):
            state, points, aux, metrics, key = self._block_keyed(state, points, aux, key)
            history.append(metrics)
            if on_block is not None:
                on_block(j, state, metrics)
        state = state._replace(key=key)
        if block_until_ready:
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), (state, metrics))
        return state, metrics, history

    def step(self, state: TrainState, key: Array | None = None) -> tuple[TrainState, Any]:
        """A single eager step (cheap call path): draws one batch and applies one update, no scan."""
        k = state.key if key is None else key
        k, kn = self._split_keys(k)
        pts, ax = self._first_one(kn)  # the first step's points only: the keys are those of the first of K
        new, metrics = self._single(state, pts, ax)
        return new._replace(key=k), metrics

    def lower_text(self, state: TrainState) -> str:
        """The optimized HLO of one block (for kernel counting)."""
        key, k0 = self._split_keys(state.key)
        points, aux = self._first(k0)
        _, kn = self._split_keys(key)
        return str(self._block.lower(state, points, aux, kn).compile().as_text())

    def explain(self) -> dict[str, Any]:
        return {
            "steps_per_call": self.steps_per_call,
            "n_models": self.n_models,
            "donated": self._donate,
            "persistent_cache": compilation_cache_dir(),
            "groups": [
                {
                    "launches": 1,
                    "blocks": [self.blocks[i].name for i in g.members],
                    "orders": sorted({self.blocks[i].order for i in g.members}),
                    "shared_evaluate": g.evaluate is not None and len(g.members) > 1,
                }
                for g in self._groups
            ],
        }

    def time_steps(self, state: TrainState, n_blocks: int = 5, warmup: int = 1) -> dict[str, float]:
        """Wall seconds per step of the compiled block (compilation excluded)."""
        state, _, _ = self.run(state, warmup, block_until_ready=True)
        t0 = time.perf_counter()
        state, _, _ = self.run(state, n_blocks, block_until_ready=True)
        dt = time.perf_counter() - t0
        return {"seconds_per_step": dt / (n_blocks * self.steps_per_call), "seconds": dt}


def compile_training_step(
    blocks: Sequence[StepBlock] | Mapping[str, StepBlock],
    optimizer: Optimizer | None = None,
    *,
    steps_per_call: int = 16,
    donate: bool | None = None,
    n_models: int | None = None,
    shared_points: bool = False,
    unroll: int = 1,
    project: Callable[[Any], Any] | None = None,
    clip_norm: float | None = None,
) -> TrainingStep:
    """Compile ``blocks`` and ``optimizer`` into one training program.

    Parameters
    ----------
    blocks : the residual terms (see :class:`StepBlock`); blocks sharing an ``evaluate`` callable are one launch.
    optimizer : an :class:`Optimizer` (optax signature).  Default :func:`adam` with ``1e-3``.
    steps_per_call : steps per executable launch (a ``lax.scan``); points for the next call are sampled by this one.
    donate : donate state and prefetched points, so parameters and optimizer state are updated in place (default: on
        accelerators, off on CPU where XLA ignores donation).
    n_models : train an ensemble with a leading model axis (``vmap``); see :meth:`TrainingStep.init`.
    shared_points : with ``n_models``, give every model the same sampling key (same points).
    unroll : ``lax.scan`` unroll factor; raise it only when profiling shows loop overhead.
    project : optional ``params -> params`` applied after each update (constraints).
    clip_norm : optional global gradient-norm clip.
    """
    seq = list(blocks.values()) if isinstance(blocks, Mapping) else list(blocks)
    return TrainingStep(
        seq,
        optimizer if optimizer is not None else adam(),
        steps_per_call=steps_per_call,
        donate=donate,
        n_models=n_models,
        shared_points=shared_points,
        unroll=unroll,
        project=project,
        clip_norm=clip_norm,
    )

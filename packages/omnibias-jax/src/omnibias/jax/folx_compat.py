# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A ``folx``-compatible ``forward_laplacian`` whose Laplacian is planned by omnibias.

``folx.forward_laplacian(fn)(*args)`` returns, for every output of ``fn``, a :class:`FwdLaplArray` holding the value, the
Jacobian with respect to all (flattened) arguments and the Laplacian ``sum_i d^2 out / d x_i^2`` over all of them. This module
accepts the same call and returns the same fields, so code that reads ``out.x``, ``out.laplacian``, ``out.dense_jacobian``
or ``out.jacobian.data`` runs unchanged with ``from omnibias.jax.folx_compat import forward_laplacian``.

What is the same: the call signature ``forward_laplacian(fn, sparsity_threshold=0, disable_jit=False)`` and the wrapped call
``wrapped(*args, weights=1.0, **kwargs)``, with ``weights`` a prefix tree of the arguments that scales each coordinate
(the Laplacian is then ``sum_i w_i^2 d^2/dx_i^2``, as in ``folx``); pytree arguments and outputs; the layout of the Jacobian
(``(n_inputs, *output_shape)``).

What differs, and what to expect:

* The Laplacian of a scalar-valued ``fn`` of one vector argument (also under ``vmap`` and ``jit``) is computed by the planner
  (:func:`omnibias.jax.operators.compile_operator`) whenever ``fn``'s weights are concrete arrays: a recognised dense chain
  gets a program chosen by the cost model, any other model the planner's generic executor, and ``wrapped.explain(x)`` shows
  the plan. Under ``vmap`` the plan is made for one point, so the cost model's batch-dependent choices are those of a single
  point. Every other call (several arguments, pytree or array outputs, weights that are tracers of an outer ``grad``/``jit``,
  keyword arguments, non-unit ``weights``) uses nested autodiff, which is exact but does not have ``folx``'s propagation
  rules; it is *not* faster than ``folx`` there and can be slower.
* The Jacobian is always dense. ``sparsity_threshold`` is accepted and ignored; ``folx``'s sparse Jacobians are not produced,
  so ``out.jacobian.x0_idx`` is ``None`` and ``out.is_jacobian_weak`` is ``False``.
* ``folx`` registers custom rules (``register_function``, ``wrap_forward_laplacian``); there is nothing to register here, and
  a function that those rules were written for simply runs through autodiff.
* The returned types are this module's, not ``folx``'s; they are structurally the same but not interchangeable with
  ``folx``'s own wrapped functions.

Whether this is faster than ``folx`` depends on the model, the dimension and the device; see ``docs/performance.md`` for the
measured comparisons and ``docs/interop.md`` for this one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np

import jax
import jax.numpy as jnp
from jax import Array, custom_batching
from jax.flatten_util import ravel_pytree

__all__ = ["FwdJacobian", "FwdLaplArray", "forward_laplacian", "register_folx_engine"]


class FwdJacobian(NamedTuple):
    """The Jacobian of an output with respect to the flattened arguments, shape ``(n_inputs, *output_shape)``.

    Always dense here; ``x0_idx`` exists only for field compatibility with ``folx`` and is ``None``.
    """

    data: Array
    x0_idx: Any = None

    @property
    def weak(self) -> bool:
        return False

    @property
    def dense_array(self) -> Array:
        return self.data

    @property
    def as_dense(self) -> FwdJacobian:
        return self

    @property
    def dense_or_sparse(self) -> Array:
        return self.data

    @property
    def data_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[1:])

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)


class FwdLaplArray(NamedTuple):
    """An output, its Jacobian and its Laplacian with respect to the function's arguments."""

    x: Array
    jacobian: FwdJacobian
    laplacian: Array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.x.shape)

    @property
    def ndim(self) -> int:
        return int(self.x.ndim)

    @property
    def size(self) -> int:
        return int(self.x.size)

    @property
    def dtype(self) -> Any:
        return self.x.dtype

    @property
    def dense_jacobian(self) -> Array:
        return self.jacobian.data

    @property
    def is_jacobian_weak(self) -> bool:
        return False

    @property
    def dense(self) -> FwdLaplArray:
        return self


def _is_scalar_weight(weights: Any) -> bool:
    if weights is None:
        return True
    try:
        return isinstance(weights, int | float | np.number) and float(weights) == 1.0
    except (TypeError, ValueError):
        return False


def forward_laplacian(
    fn: Callable[..., Any], sparsity_threshold: int | float = 0, disable_jit: bool = False
) -> Callable[..., Any]:
    """``folx.forward_laplacian``-compatible wrapper of ``fn``; see the module docstring for what is and is not the same."""
    del sparsity_threshold  # the Jacobian is always dense here
    planned: dict[tuple[Any, ...], Callable[[Array], tuple[Array, Array, Array]] | None] = {}
    plans: dict[tuple[Any, ...], Any] = {}
    thunks: dict[tuple[Any, ...], Callable[[], Any]] = {}

    def _planned_runner(x: Array) -> Callable[[Array], tuple[Array, Array, Array]] | None:
        # ``vmap`` hands the model a ``BatchTracer`` (recognised by name: only an optimisation hint, a renamed class just
        # takes the general path below); calls under ``vmap`` and plain calls keep separate runners
        batch_traced = type(x).__name__ == "BatchTracer"
        key = (tuple(x.shape), str(x.dtype), batch_traced)
        if not disable_jit and key in planned:
            return planned[key]
        from omnibias.core.planner import PlanCache
        from omnibias.jax.experimental.compiler import compile_operator

        def build(shape: tuple[int, ...]) -> Any:
            # only the shape and dtype of ``x`` matter (it may be a tracer under ``vmap`` or ``jit``): the chain is read from
            # ``fn``'s weights and confirmed numerically on a fixed random point of that shape, evaluated eagerly even while
            # an enclosing transform is tracing
            with jax.ensure_compile_time_eval():
                # drawn on the host: ``jax.random.normal`` compiles a program for every new shape (0.3-0.4 s each)
                example = jnp.asarray(np.random.default_rng(0).standard_normal(shape), dtype=x.dtype)
                return compile_operator(
                    lambda batch: jax.vmap(fn)(batch),
                    (example,),
                    operator="laplacian",
                    params_argnum=None,
                    cache=PlanCache(),
                    jit=False,
                )

        closed_over = jax.make_jaxpr(fn)(x).consts
        concrete_weights = not any(isinstance(c, jax.core.Tracer) for c in closed_over)
        # under ``vmap`` with concrete weights the batch plan below serves every call (the vmap rule replaces ``run``), so
        # the plan of one point, about as costly to build as the batch plan, is only made if the batch plan cannot be
        lazy = batch_traced and concrete_weights
        op_box: dict[str, Any] = {}

        def point_plan() -> Any:
            """The plan of one point, built on first use: ``None`` when this model cannot be planned."""
            if "op" not in op_box:
                try:
                    op_box["op"] = build((1, *x.shape))
                except Exception:  # noqa: BLE001 - e.g. weights that are tracers of an outer transform
                    op_box["op"] = None
            return op_box["op"]

        def point_laplacian(point: Array) -> Array:
            op = point_plan()
            if op is None:
                return jnp.trace(jax.hessian(fn)(point))
            return cast(Array, op(point[None, :])[0])

        if not lazy and point_plan() is None:
            return None  # use the generic path; not cached: the same wrapper may later be called with concrete weights

        def run(point: Array) -> tuple[Array, Array, Array]:
            value, grad = jax.value_and_grad(fn)(point)
            if lazy:  # traced for the vmap rule's signature, never executed
                return value, grad, jnp.trace(jax.hessian(fn)(point))
            return value, grad, point_laplacian(point)

        batch_ops: dict[int, Any] = {}

        def batched(points: Array) -> tuple[Array, Array, Array]:
            # one plan for the whole batch of points that an enclosing ``vmap`` carries: the plan is built once per batch size
            # and its shared work (the first-layer projection) is done once per call, not once per point
            n = int(points.shape[0])
            if n not in batch_ops:
                try:
                    batch_ops[n] = build(tuple(points.shape))
                except Exception:  # noqa: BLE001 - keep the per-point plan if a plan for this batch cannot be built
                    batch_ops[n] = None
            batch_op = batch_ops[n]
            if batch_op is None:
                return cast(
                    tuple[Array, Array, Array],
                    jax.vmap(lambda p: (fn(p), jax.grad(fn)(p), point_laplacian(p)))(points),
                )
            return jax.vmap(fn)(points), jax.vmap(jax.grad(fn))(points), batch_op(points)

        point_fn: Callable[[Array], tuple[Array, Array, Array]] = run
        # ``custom_vmap`` cannot be differentiated through a function that closes over tracers (``grad`` of a closed-over
        # parameter): those calls keep the per-point plan, which differentiates
        if concrete_weights:
            vmapped = custom_batching.custom_vmap(run)

            @vmapped.def_vmap  # type: ignore[untyped-decorator,unused-ignore]
            def _vmap_rule(
                axis_size: int, in_batched: list[bool], points: Array
            ) -> tuple[tuple[Array, Array, Array], tuple[bool, bool, bool]]:
                stacked = (
                    points
                    if in_batched[0]
                    else jnp.broadcast_to(points, (axis_size, *points.shape))
                )
                return batched(stacked), (True, True, True)

            point_fn = vmapped

        runner = cast(
            Callable[[Array], tuple[Array, Array, Array]],
            point_fn if disable_jit else jax.jit(point_fn),
        )
        thunks[key] = point_plan
        if "op" in op_box:
            plans[key] = op_box["op"]
        if not disable_jit:
            planned[key] = runner
        return runner

    def _generic(args: tuple[Any, ...], weights: Any, kwargs: dict[str, Any]) -> Any:
        flat, unravel = ravel_pytree(args)
        scale = ravel_pytree(
            jax.tree.map(
                lambda w, tree: jax.tree.map(lambda leaf: jnp.full_like(leaf, w), tree),
                weights,
                args,
            )
        )[0]

        def shifted(z: Array) -> Any:
            return fn(*unravel(flat + scale * z), **kwargs)

        zero = jnp.zeros_like(flat)
        value = shifted(zero)
        out_leaves, structure = jax.tree.flatten(value)
        n_in = int(flat.size)
        use_forward = sum(int(np.prod(leaf.shape)) for leaf in out_leaves) >= n_in
        jac_tree = jax.jacfwd(shifted)(zero) if use_forward else jax.jacrev(shifted)(zero)
        hess_tree = jax.hessian(shifted)(zero)
        results = []
        for leaf, jac, hess in zip(
            out_leaves, jax.tree.leaves(jac_tree), jax.tree.leaves(hess_tree), strict=True
        ):
            results.append(
                FwdLaplArray(
                    leaf,
                    FwdJacobian(jnp.moveaxis(jac, -1, 0)),
                    jnp.trace(hess, axis1=-2, axis2=-1),
                )
            )
        return jax.tree.unflatten(structure, results)

    def wrapped(*args: Any, weights: Any = 1.0, **kwargs: Any) -> Any:
        if weights is None:
            weights = 1.0
        eligible = (
            len(args) == 1
            and not kwargs
            and _is_scalar_weight(weights)
            and hasattr(args[0], "ndim")
            and args[0].ndim == 1
            and jnp.issubdtype(jnp.asarray(args[0]).dtype, jnp.floating)
        )
        if eligible:
            x = jnp.asarray(args[0])
            shape = jax.eval_shape(fn, x)
            if isinstance(shape, jax.ShapeDtypeStruct) and shape.shape == ():
                runner = _planned_runner(x)
                if runner is not None:
                    value, grad, lap = runner(x)
                    return FwdLaplArray(value, FwdJacobian(grad), lap)
        return _generic(args, weights, kwargs)

    def explain(*args: Any) -> dict[str, Any] | None:
        """The plan used for these arguments, or ``None`` when they take the generic autodiff path."""
        if len(args) != 1 or not hasattr(args[0], "ndim") or args[0].ndim != 1:
            return None
        x = jnp.asarray(args[0])
        key = (tuple(x.shape), str(x.dtype), False)
        if key not in plans:
            vmapped = (key[0], key[1], True)
            if vmapped in thunks:
                plans[key] = thunks[vmapped]()
            else:
                wrapped(x)
        op = plans.get(key)
        return None if op is None else dict(op.explain())

    wrapped.explain = explain  # type: ignore[attr-defined]
    wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapped


def register_folx_engine(name: str = "folx", *, replace: bool = False) -> Any:
    """Make the installed ``folx`` an executor the planner can choose, rank, time and name in ``explain()``.

    ``folx.forward_laplacian`` runs on the pointwise model, vectorised over the batch; it serves the second-order Laplacian
    of a scalar field (not powers, metrics or coordinate subsets) and is priced like the built-in ``forward_laplacian``
    program. It is registered ``generic=True``: a model that is not a recognised dense chain can use it by ``force``.
    Raises ``ImportError`` when ``folx`` is not installed.
    """
    import folx  # type: ignore[import-not-found,unused-ignore]
    from omnibias.jax.experimental.compiler import register_engine

    def engine(f: Callable[[Array], Array], x: Array, power: int = 1) -> Array:
        laplacian = folx.forward_laplacian(f)
        return cast(Array, jax.vmap(lambda z: laplacian(z).laplacian)(x))

    return register_engine(
        name, "laplacian", engine, work="forward_laplacian", generic=True, replace=replace
    )

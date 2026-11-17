# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Describe, plan, execute: one entry point for structured derivative operators.

``compile_operator(model, example_args, operator=...)`` takes an ordinary JAX model, finds out whether it is a dense
layer chain, asks the :mod:`omnibias.core.planner` which program is cheapest for that descriptor on this device and
returns a callable.  A model that is not a recognised chain still works: it runs on a generic Taylor-mode or nested
autodiff executor, and :meth:`CompiledOperator.explain` says which pattern failed to match and why.

The chain matcher works on the traced program, not on the Python source: it flattens nested ``jit`` calls and reads the
``dot_general`` / bias / activation structure, then confirms the match numerically against the model on the example
inputs.  Anything it does not understand is refused with a reason; it never guesses.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
from omnibias.core.engines import engine_algorithm
from omnibias.core.planner import (
    FIND_TOP_K,
    Algorithm,
    CostModel,
    Descriptor,
    FindRefusal,
    OperatorRequest,
    Plan,
    PlanCache,
    Refusal,
    Work,
    default_plan_cache_path,
    device_class,
    estimator_probes,
    host_memory_limit,
    measurement_context,
    register_algorithm,
    resolve_planning,
    select,
    unregister_algorithm,
)
from omnibias.core.precision import PrecisionPolicy, request_order, resolve_precision
from omnibias.jax.activations import get_activation
from omnibias.jax.experimental.chain import describe_layers, layers_of
from omnibias.jax.experimental.coldcost import (
    ExportedProgram,
    MemoryBudgetExceeded,
    PolicyChoice,
    StagedProgram,
    bucket_size,
    compilation_cache_dir,
    export_program,
    is_traced,
    pad_rows,
    resolve_compile_policy,
)
from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian
from omnibias.jax.experimental.projected_divergence import projected_deep_divergence
from omnibias.jax.experimental.projected_laplacian import projected_deep_laplacian
from omnibias.jax.experimental.projected_logdet import projected_deep_logdet
from omnibias.jax.experimental.projected_polylaplacian import projected_deep_trilaplacian
from omnibias.jax.experimental.residual_woodbury import residual_solve
from omnibias.jax.experimental.streams import (
    bilaplacian_by_jets,
    directional_jet_streams,
    training_layout,
)
from omnibias.jax.laplacian import deep_field_polylaplacian, deep_field_value_grad_laplacian

import jax
import jax.numpy as jnp
from jax.experimental import jet as _jet

Array = jax.Array
_T = TypeVar("_T")

__all__ = [
    "ChainMatch",
    "CompiledOperator",
    "LayerMatch",
    "OperatorOutput",
    "compile_operator",
    "match_dense_chain",
    "register_engine",
    "unregister_engine",
]


# ------------------------------------------------------------------------------------------- jaxpr flattening

#: ``jit`` names produced by :mod:`jax.nn` and the omnibias activation each stands for (verified numerically).
_JAX_NN = {
    "silu": "silu",
    "swish": "silu",
    "relu": "relu",
    "relu6": "relu6",
    "softplus": "softplus",
    "elu": "elu",
    "selu": "selu",
    "celu": "celu",
    "leaky_relu": "leaky_relu",
    "softsign": "softsign",
    "soft_sign": "softsign",
    "sigmoid": "sigmoid",
    "gelu": "gelu",
    "hard_tanh": "hardtanh",
    "hard_sigmoid": "hardsigmoid",
    "hard_swish": "hardswish",
}
#: Elementwise primitives that are an activation by themselves.
_PRIMITIVE_ACTIVATION = {
    "tanh": "tanh",
    "logistic": "sigmoid",
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "sinh": "sinh",
    "cosh": "cosh",
    "atan": "arctan",
}
_INLINE = frozenset(
    {
        "pjit",
        "jit",
        "closed_call",
        "core_call",
        "remat",
        "checkpoint",
        "custom_jvp_call",
        "custom_vjp_call",
        "custom_vjp_call_jaxpr",
    }
)
_PASS = frozenset({"convert_element_type", "copy", "copy_p"})


class _NoMatch(Exception):
    """The traced program is not a pattern the matcher knows; the message is the reason."""


class _Node(NamedTuple):
    prim: str
    params: Mapping[str, Any]
    ins: tuple[Any, ...]  # int (node id) or ("lit", value)
    out: int
    shape: tuple[int, ...]
    dtype: str


def _block(tree: Any) -> Any:
    """Wait for every array of ``tree`` (timings and the first-call measurement must not include queued work)."""
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), tree)


def _is_literal(atom: Any) -> bool:
    return type(atom).__name__ == "Literal"


def _closed(inner: Any) -> tuple[Any, Sequence[Any]]:
    if hasattr(inner, "jaxpr"):
        return inner.jaxpr, list(inner.consts)
    return inner, []


class _Flattener:
    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self.next_id = 0

    def fresh(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def add(self, prim: str, params: Mapping[str, Any], ins: Sequence[Any], aval: Any) -> int:
        nid = self.fresh()
        self.nodes.append(
            _Node(
                prim,
                params,
                tuple(ins),
                nid,
                tuple(getattr(aval, "shape", ())),
                str(getattr(aval, "dtype", "")),
            )
        )
        return nid

    def walk(self, jaxpr: Any, const_atoms: Sequence[Any], in_atoms: Sequence[Any]) -> list[Any]:
        """Add the equations of ``jaxpr`` (inlining calls); return the atoms of its outputs."""
        env: dict[Any, Any] = {}
        for var, atom in zip(jaxpr.invars, in_atoms, strict=True):
            env[var] = atom
        for var, atom in zip(jaxpr.constvars, const_atoms, strict=True):
            env[var] = atom

        def get(atom: Any) -> Any:
            return ("lit", atom.val) if _is_literal(atom) else env[atom]

        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            ins = [get(a) for a in eqn.invars]
            label = eqn.params.get("name") if name in _INLINE else None
            if label in _JAX_NN and len(ins) == 1 and len(eqn.outvars) == 1:
                env[eqn.outvars[0]] = self.add(
                    "activation", {"activation": _JAX_NN[label]}, ins, eqn.outvars[0].aval
                )
            elif name in _INLINE:
                inner = (
                    eqn.params.get("jaxpr")
                    or eqn.params.get("call_jaxpr")
                    or eqn.params.get("fun_jaxpr")
                )
                if inner is None:
                    raise _NoMatch(f"cannot read the body of {name!r}")
                body, body_consts = _closed(inner)
                outs = self.walk(body, [("lit", c) for c in body_consts], ins)
                for var, atom in zip(eqn.outvars, outs, strict=True):
                    env[var] = atom
            else:
                if len(eqn.outvars) != 1:
                    raise _NoMatch(f"primitive {name!r} has {len(eqn.outvars)} outputs")
                env[eqn.outvars[0]] = self.add(name, eqn.params, ins, eqn.outvars[0].aval)
        return [get(a) for a in jaxpr.outvars]


# --------------------------------------------------------------------------------------------- chain matching


@dataclass(frozen=True)
class LayerMatch:
    """One matched layer: leaf indices into the flattened parameters and how the weight is stored."""

    weight: int
    weight_form: (
        str  # "out_in" (W is (out, in)), "in_out" (W is (in, out)) or "vector" (readout (in,))
    )
    bias: int | None
    activation: str | None
    #: literal scalar factors folded into the weight and the bias (``sin(w0 * (W a + b))`` of a SIREN layer)
    weight_scale: float = 1.0
    bias_scale: float = 1.0


@dataclass(frozen=True)
class ChainMatch:
    """A traced model recognised as ``a -> act(W a + b) -> ... -> W a + b``."""

    layers: tuple[LayerMatch, ...]
    squeeze_output: bool
    batched: bool
    leaf_values: tuple[
        Any, ...
    ] = ()  # for a bound model (weights closed over): the constants, else empty

    def layers_from(self, leaves: Sequence[Any]) -> list[tuple[Array, Array | None, str | None]]:
        out: list[tuple[Array, Array | None, str | None]] = []
        for lm in self.layers:
            W = jnp.asarray(leaves[lm.weight])
            if lm.weight_form == "vector":
                W = W[None, :]
            elif lm.weight_form == "in_out":
                W = W.T
            if lm.weight_scale != 1.0:
                W = W * lm.weight_scale
            b = None if lm.bias is None else jnp.reshape(jnp.asarray(leaves[lm.bias]), (-1,))
            if b is not None and lm.bias_scale != 1.0:
                b = b * lm.bias_scale
            out.append((W, b, lm.activation))
        return out


def _flatten_model(
    fn: Callable[..., Any], example_args: Sequence[Any], params_argnum: int | None
) -> tuple[list[_Node], list[int], list[Any], int, list[Any]]:
    """``(nodes, leaf_ids, leaf_consts, x_id, out_atoms)``.

    ``leaf_ids`` are the flat parameter leaves (``params_argnum``) or, for a bound model, its closed-over constants.
    """
    closed = jax.make_jaxpr(fn)(*example_args)
    flat = _Flattener()
    counts = [len(jax.tree_util.tree_leaves(a)) for a in example_args]
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(int)
    input_ids = [flat.add("input", {}, (), v.aval) for v in closed.jaxpr.invars]
    const_ids = [flat.add("input", {}, (), v.aval) for v in closed.jaxpr.constvars]
    out_atoms = flat.walk(closed.jaxpr, const_ids, input_ids)
    if params_argnum is None:
        leaf_ids, leaf_consts, x_ids = list(const_ids), list(closed.consts), input_ids
    else:
        lo, hi = int(starts[params_argnum]), int(starts[params_argnum + 1])
        leaf_ids, leaf_consts = input_ids[lo:hi], []
        x_ids = [i for k, i in enumerate(input_ids) if not lo <= k < hi]
    if len(x_ids) != 1:
        raise _NoMatch("the model must take exactly one array besides its parameters")
    return flat.nodes, leaf_ids, leaf_consts, x_ids[0], out_atoms


def _match_nodes(
    nodes: Sequence[_Node],
    leaf_ids: Sequence[int],
    x_id: int,
    out_atoms: Sequence[Any],
) -> tuple[tuple[LayerMatch, ...], bool]:
    shapes = {n.out: n.shape for n in nodes}
    leaf_shapes = [shapes[i] for i in leaf_ids]
    if len(shapes[x_id]) not in (1, 2):
        raise _NoMatch("the input must have shape (D,) or (B, D)")
    if len(out_atoms) != 1 or not isinstance(out_atoms[0], int):
        raise _NoMatch("the model must return one array")
    # role of every value: ("act", l) input of layer l; ("lin", l) / ("pre", l) before / after the bias of layer l;
    # ("param", leaf), ("paramT", leaf) and ("bias", leaf) parameters
    role: dict[int, tuple[Any, ...]] = {x_id: ("act", 0)}
    for leaf, nid in enumerate(leaf_ids):
        role[nid] = ("param", leaf)
    info: list[dict[str, Any]] = []
    squeeze = False

    def of(atom: Any) -> tuple[Any, ...]:
        return ("lit",) if not isinstance(atom, int) else role.get(atom, ("unknown",))

    for n in nodes:
        if n.prim == "input":
            continue
        r = [of(a) for a in n.ins]
        p = n.prim
        if p == "transpose":
            if (
                r[0][0] == "param"
                and len(leaf_shapes[r[0][1]]) == 2
                and tuple(n.params["permutation"]) == (1, 0)
            ):
                role[n.out] = ("paramT", r[0][1])
            else:
                raise _NoMatch("transpose of something other than a weight matrix")
        elif (
            p in ("broadcast_in_dim", "reshape")
            and r[0][0] == "param"
            and len(leaf_shapes[r[0][1]]) <= 1
        ):
            role[n.out] = ("bias", r[0][1])
        elif p in _PASS and r[0][0] in ("param", "paramT", "act", "lin", "pre", "bias"):
            if p == "convert_element_type" and str(n.params.get("new_dtype")) != n.dtype:
                raise _NoMatch("dtype conversion inside the model")
            role[n.out] = r[0]
        elif p == "dot_general":
            _match_dot(n, r, shapes, leaf_shapes, info, role)
        elif p == "add":
            lin = next((x for x in r if x[0] == "lin"), None)
            bias = next((x for x in r if x[0] in ("bias", "param")), None)
            if lin is None or bias is None:
                raise _NoMatch("add that is not a bias")
            if len(leaf_shapes[bias[1]]) > 1:
                raise _NoMatch("bias that is not a vector or a scalar")
            if info[lin[1]]["bias"] is not None:
                raise _NoMatch("two bias additions in one layer")
            info[lin[1]]["bias"] = bias[1]
            role[n.out] = ("pre", lin[1])
        elif p == "squeeze":
            if r[0][0] in ("lin", "pre") and tuple(n.params["dimensions"]) == (
                len(shapes[n.ins[0]]) - 1,
            ):
                squeeze = True
                role[n.out] = r[0]
            else:
                raise _NoMatch("squeeze of something other than the readout")
        elif p == "mul" and sum(x[0] == "lit" for x in r) == 1:
            # a literal scalar times a layer output folds into the weight (and the bias once it has been added)
            li = next(i for i, x in enumerate(r) if x[0] == "lit")
            other = r[1 - li]
            factor = np.asarray(n.ins[li][1])
            if other[0] not in ("lin", "pre") or factor.size != 1:
                raise _NoMatch("multiplication that is not a scalar factor of a layer output")
            scale = float(factor.reshape(()))
            info[other[1]]["weight_scale"] *= scale
            if other[0] == "pre":
                info[other[1]]["bias_scale"] *= scale
            role[n.out] = other
        elif p == "activation" or p in _PRIMITIVE_ACTIVATION or _is_zero_literal(n):
            source = next(x for x in r if x[0] != "lit")
            if source[0] not in ("lin", "pre"):
                raise _NoMatch(f"activation {p!r} applied to something other than a layer output")
            info[source[1]]["act"] = (
                n.params["activation"]
                if p == "activation"
                else "relu"
                if p == "max"
                else _PRIMITIVE_ACTIVATION[p]
            )
            role[n.out] = ("act", source[1] + 1)
        else:
            raise _NoMatch(f"unsupported primitive {p!r}")
    if not info:
        raise _NoMatch("no dense layer found")
    out_role = of(out_atoms[0])
    # the output is the last layer's result (an activation on it is allowed; the planner then refuses structured programs)
    last = len(info) - 1
    if not (
        (out_role[0] in ("lin", "pre") and out_role[1] == last)
        or (out_role[0] == "act" and out_role[1] == last + 1)
    ):
        raise _NoMatch("the output is not the result of the last layer")
    layers = tuple(
        LayerMatch(d["weight"], d["form"], d["bias"], d["act"], d["weight_scale"], d["bias_scale"])
        for d in info
    )
    if any(lm.weight_form == "vector" for lm in layers[:-1]):
        raise _NoMatch("a vector weight before the last layer")
    # a vector readout (H,) drops the output axis exactly as a squeeze would
    return layers, squeeze or layers[-1].weight_form == "vector"


def _match_dot(
    n: _Node,
    r: Sequence[tuple[Any, ...]],
    shapes: Mapping[int, tuple[int, ...]],
    leaf_shapes: Sequence[tuple[int, ...]],
    info: list[dict[str, Any]],
    role: dict[int, tuple[Any, ...]],
) -> None:
    (contract, batch) = n.params["dimension_numbers"]
    if batch[0] or batch[1] or len(contract[0]) != 1:
        raise _NoMatch("dot_general with batch or multiple contracting dimensions")
    left_act = r[0][0] == "act"
    if left_act == (r[1][0] == "act"):
        raise _NoMatch("dot_general whose operands are not one activation and one weight")
    act_i, w_i = (0, 1) if left_act else (1, 0)
    act_role, w_role = r[act_i], r[w_i]
    if w_role[0] not in ("param", "paramT"):
        raise _NoMatch("dot_general with a weight that is not a model parameter")
    c_act, c_w = contract[act_i][0], contract[w_i][0]
    act_shape = shapes[n.ins[act_i]]
    if c_act != len(act_shape) - 1:
        raise _NoMatch("the layer must contract the last axis of its input")
    if not left_act and len(act_shape) != 1:
        raise _NoMatch("weight on the left of a batched activation")
    leaf = w_role[1]
    if act_role[1] != len(info):
        raise _NoMatch("layers are not applied in sequence")
    if len(leaf_shapes[leaf]) == 1:
        if w_role[0] != "param":
            raise _NoMatch("transposed vector")
        form = "vector"
    elif len(leaf_shapes[leaf]) == 2:
        # the contracted axis of the stored leaf is the "in" axis
        c_leaf = c_w if w_role[0] == "param" else 1 - c_w
        form = "in_out" if c_leaf == 0 else "out_in"
    else:
        raise _NoMatch("weight with more than two axes")
    info.append(
        {
            "weight": leaf,
            "form": form,
            "bias": None,
            "act": None,
            "weight_scale": 1.0,
            "bias_scale": 1.0,
        }
    )
    role[n.out] = ("lin", act_role[1])


def _is_zero_literal(n: _Node) -> bool:
    if n.prim != "max" or len(n.ins) != 2:
        return False
    other = [a for a in n.ins if not isinstance(a, int)]
    if len(other) != 1:
        return False
    try:
        return float(np.asarray(other[0][1])) == 0.0
    except (TypeError, ValueError):
        return False


def _chain_value(layers: Sequence[tuple[Array, Array | None, Any]], x: Array) -> Array:
    a = x
    for W, b, spec in layers:
        z = a @ W.T
        if b is not None:
            z = z + b
        a = z if spec is None else get_activation(spec).forward(z)
    return a


def match_dense_chain(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    params_argnum: int | None = 0,
) -> tuple[ChainMatch | None, str | None]:
    """Recognise ``fn`` as a dense layer chain.  Returns ``(match, None)`` or ``(None, reason)``.

    ``example_args`` are ``(params, x)`` (``params_argnum=0``) or ``(x,)`` with the weights closed over
    (``params_argnum=None``).  The match is confirmed numerically on the example, so a program that looks like a chain
    but computes something else is refused.
    """
    try:
        nodes, leaf_ids, leaf_consts, x_id, out_atoms = _flatten_model(
            fn, example_args, params_argnum
        )
        x_shape = next(n.shape for n in nodes if n.out == x_id)
        layers, squeeze = _match_nodes(nodes, leaf_ids, x_id, out_atoms)
        for lm in layers:
            if lm.activation is not None:
                get_activation(lm.activation)
    except _NoMatch as err:
        return None, str(err)
    except (KeyError, ValueError, TypeError) as err:  # unknown activation name, odd shapes
        return None, f"{type(err).__name__}: {err}"
    match = ChainMatch(layers, squeeze, len(x_shape) == 2, tuple(leaf_consts))
    # numerical confirmation on the example inputs
    if params_argnum is None:
        leaves: Sequence[Any] = leaf_consts
        x = example_args[0]
    else:
        leaves = jax.tree_util.tree_leaves(example_args[params_argnum])
        x = next(a for k, a in enumerate(example_args) if k != params_argnum)
    try:
        got_shape, want_shape, dtype, scale, diff = _confirm_chain(
            match, fn, example_args, leaves, x, squeeze
        )
    except Exception as err:  # noqa: BLE001 - any failure to evaluate the candidate match is a refusal
        return None, f"matched chain could not be evaluated: {type(err).__name__}: {err}"
    if got_shape != want_shape:
        return None, f"matched chain has output shape {got_shape}, the model {want_shape}"
    tol = 1e-5 if dtype == jnp.float32 else 1e-9 if dtype == jnp.float64 else 1e-2
    if not diff <= tol * scale:
        return (
            None,
            f"matched chain disagrees with the model on the example (max abs difference {diff:.2e})",
        )
    return match, None


def _confirm_chain(
    match: ChainMatch,
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    leaves: Sequence[Any],
    x: Any,
    squeeze: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], Any, float, float]:
    """``(chain shape, model shape, model dtype, 1 + max|model|, max|chain - model|)`` on the example.

    One compiled program evaluates both sides: executed eagerly, each of their dozens of primitives is compiled on its
    own, which was most of the time of a first plan (about 0.9 s of 1.3 s for a three-layer chain).  A program that
    cannot be traced as a whole is evaluated eagerly as before.
    """
    seen: dict[str, Any] = {}

    def sides(args: Sequence[Any], chain_leaves: Sequence[Any], chain_x: Any) -> tuple[Array, Array]:
        got = _chain_value(match.layers_from(chain_leaves), jnp.asarray(chain_x))
        if squeeze:
            got = got[..., 0]
        want = jnp.asarray(fn(*args))
        seen["shapes"] = (tuple(got.shape), tuple(want.shape))
        seen["dtype"] = want.dtype
        return got, want

    def measures(got: Array, want: Array) -> tuple[Array, Array]:
        if got.shape != want.shape:
            return jnp.zeros(()), jnp.zeros(())
        return 1.0 + jnp.max(jnp.abs(want)), jnp.max(jnp.abs(got - want))

    def together(args: Sequence[Any], chain_leaves: Sequence[Any], chain_x: Any) -> tuple[Array, Array]:
        return measures(*sides(args, chain_leaves, chain_x))

    try:
        scale, diff = jax.jit(together)(tuple(example_args), tuple(leaves), x)
    except Exception:  # noqa: BLE001 - not traceable as one program: evaluate it eagerly
        scale, diff = measures(*sides(example_args, leaves, x))
    got_shape, want_shape = seen["shapes"]
    return got_shape, want_shape, seen["dtype"], float(scale), float(diff)


# ------------------------------------------------------------------------------------------------ input products


class InputMap(NamedTuple):
    """The first-layer products a model reads its input through: ``(leaf, stored (in, out), vector)`` per product.

    When ``x`` enters a program only as ``W_1 x, ..., W_n x`` with parameter matrices, every input derivative lies in the
    row space of the stacked ``W``, so an operator whose directions are columns ``V`` needs only an orthonormal basis of
    the row space of ``W V``: ``min(rows, columns)`` directions instead of all of them.
    """

    entries: tuple[tuple[int, bool, bool], ...]
    rows: int


def analyze_input_map(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    params_argnum: int | None = 0,
) -> tuple[InputMap | None, str | None]:
    """Whether ``fn`` reads its input only through products with parameter matrices.  ``(map, None)`` or ``(None, reason)``.

    Any other read of the input (an elementwise use, a slice, a concatenation, a skip of the raw coordinates) refuses with
    the primitive that reads it: the row-space argument needs the input to appear nowhere else.
    """
    try:
        nodes, leaf_ids, _consts, x_id, _out = _flatten_model(fn, example_args, params_argnum)
    except _NoMatch as err:
        return None, str(err)
    shapes = {n.out: n.shape for n in nodes}
    leaf_index = {nid: k for k, nid in enumerate(leaf_ids)}
    transposed_of: dict[int, int] = {}
    alias = {x_id}
    entries: list[tuple[int, bool, bool]] = []
    rows = 0
    for n in nodes:
        if n.prim == "input":
            continue
        ids = [a for a in n.ins if isinstance(a, int)]
        if n.prim in _PASS and ids:
            if ids[0] in alias:
                alias.add(n.out)
                continue
            if ids[0] in leaf_index:
                leaf_index[n.out] = leaf_index[ids[0]]
                continue
        if (
            n.prim == "transpose"
            and ids
            and ids[0] in leaf_index
            and len(shapes[ids[0]]) == 2
            and tuple(n.params["permutation"]) == (1, 0)
        ):
            transposed_of[n.out] = leaf_index[ids[0]]
            continue
        if not any(a in alias for a in ids):
            continue
        if n.prim != "dot_general":
            return None, (
                f"the input is also read by {n.prim!r}; it must enter only through products with parameter matrices"
            )
        (contract, batch) = n.params["dimension_numbers"]
        if batch[0] or batch[1] or len(contract[0]) != 1:
            return None, "an input product with batch or multiple contracting dimensions"
        x_pos = 0 if isinstance(n.ins[0], int) and n.ins[0] in alias else 1
        w_atom = n.ins[1 - x_pos]
        if not isinstance(w_atom, int) or w_atom in alias:
            return None, "an input product whose other factor is not a parameter"
        if contract[x_pos][0] != len(shapes[n.ins[x_pos]]) - 1:
            return None, "an input product that does not contract the last axis of the input"
        if w_atom in leaf_index:
            leaf, direct = leaf_index[w_atom], True
        elif w_atom in transposed_of:
            leaf, direct = transposed_of[w_atom], False
        else:
            return None, "an input product whose other factor is not a parameter"
        shape = shapes[leaf_ids[leaf]]
        if len(shape) == 1:
            if not direct:
                return None, "transposed vector"
            entries.append((leaf, False, True))
            rows += 1
        elif len(shape) == 2:
            c_w = contract[1 - x_pos][0]
            c_leaf = c_w if direct else 1 - c_w
            stored_in_out = c_leaf == 0
            entries.append((leaf, stored_in_out, False))
            rows += shape[1] if stored_in_out else shape[0]
        else:
            return None, "weight with more than two axes"
    if not entries:
        return None, "the input is not read by any weight product"
    return InputMap(tuple(entries), rows), None


# ---------------------------------------------------------------------------------------------------- executors


class OperatorOutput(NamedTuple):
    """Result of :meth:`CompiledOperator.evaluate`: the operator value and the by-products the program formed."""

    value: Array
    aux: dict[str, Array]


@dataclass(frozen=True)
class _Ctx:
    fn: Callable[[Any, Array], Array]
    layers: Callable[[Any], list[tuple[Array, Array | None, Any]]] | None
    squeeze: bool
    batched: bool
    dim: int
    power: int
    coordinates: np.ndarray | None
    diffusion: Any
    directions: Any
    allow_ae: bool
    transpose: bool
    n_probes: int | None
    input_map: InputMap | None = None
    leaves: Callable[[Any], Sequence[Any]] | None = None
    differentiate: str = "none"
    accuracy: float | None = None
    matmul: str = "highest"  # the PrecisionPolicy mode of the call: "highest", "tf32" or "default" (the process setting)

    def input_rows(self, params: Any, dtype: Any) -> Array:
        """The stacked first-layer matrices ``(rows, D)`` of the products the model reads its input through."""
        if self.input_map is None or self.leaves is None:
            raise ValueError(
                "this program needs a model that reads its input through weight products"
            )
        leaves = self.leaves(params)
        blocks = []
        for leaf, stored_in_out, vector in self.input_map.entries:
            W = jnp.asarray(leaves[leaf])
            blocks.append(W[None, :] if vector else (W.T if stored_in_out else W))
        return jnp.concatenate(blocks, axis=0).astype(dtype)

    def reduced_columns(self, params: Any, dtype: Any) -> Array:
        """Directions ``V'`` ``(D, k)``, ``k = min(rows, q)``, with ``W V' V'^T W^T = W V V^T W^T`` for the stacked ``W``.

        They are ``V`` times an orthonormal basis ``Q`` of the row space of ``W V`` (any ``Q`` with ``Q Q^T (W V)^T =
        (W V)^T`` serves, so a thin QR factorization replaces the singular value decomposition) and are held constant
        for differentiation: the operator's first-order parameter derivative is unchanged because ``W`` annihilates
        the directions dropped.  When the metric is the identity on the selected coordinates ``V`` is never formed.
        """
        W = jax.lax.stop_gradient(self.input_rows(params, dtype))
        if self.diffusion is None:
            idx = np.arange(self.dim) if self.coordinates is None else np.asarray(self.coordinates)
            Q = jnp.linalg.qr(W[:, idx].T, mode="reduced")[0]
            if self.coordinates is None:
                return jax.lax.stop_gradient(Q)
            return jax.lax.stop_gradient(jnp.zeros((self.dim, Q.shape[1]), dtype).at[idx].set(Q))
        V = jnp.asarray(self.columns(), dtype)
        Q = jnp.linalg.qr((W @ V).T, mode="reduced")[0]
        return jax.lax.stop_gradient(V @ Q)

    def point(self, params: Any, xi: Array) -> Array:
        return self.fn(params, xi[None, :])[0] if self.batched else self.fn(params, xi)

    def columns(self) -> np.ndarray:
        """``V`` with ``V V^T`` the metric embedded in the ``D`` inputs: the operator is ``sum_c v_c^T H v_c``."""
        idx = np.arange(self.dim) if self.coordinates is None else np.asarray(self.coordinates)
        m = len(idx)
        if self.diffusion is None:
            S = np.eye(m)
        else:
            d = np.asarray(self.diffusion)
            S = np.diag(d) if d.ndim == 1 else d
        V = np.zeros((self.dim, S.shape[1]))
        V[idx] = S
        return V

    def pairing(self, outputs: int) -> np.ndarray:
        if self.coordinates is not None:
            return np.asarray(self.coordinates)
        if outputs != self.dim:
            raise ValueError(
                "a vector field needs as many outputs as inputs (or `coordinates` to pair them)"
            )
        return np.arange(self.dim)


def _squeeze(ctx: _Ctx, tree: _T) -> _T:
    if not ctx.squeeze:
        return tree
    return cast(_T, jax.tree_util.tree_map(lambda a: a[..., 0], tree))


def _points(ctx: _Ctx, g: Callable[[Array], Any], x: Array) -> Any:
    return jax.vmap(g)(x) if ctx.batched else g(x)


def _layers(ctx: _Ctx, params: Any) -> list[tuple[Array, Array | None, Any]]:
    if ctx.layers is None:
        raise ValueError("this program needs a recognised dense chain")
    return ctx.layers(params)


Runner = Callable[[Any, Array, tuple[Any, ...], Any], tuple[Array, dict[str, Array]]]


def _lap_projected(ctx: _Ctx, form: str = "auto") -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        layers = _layers(ctx, params)
        dirs = None if ctx.directions is None else jnp.asarray(ctx.directions, x.dtype)
        if ctx.power == 1:
            u, gs, lap, dv = projected_deep_laplacian(
                x,
                layers,
                diffusion=ctx.diffusion,
                coordinates=ctx.coordinates,
                directions=dirs,
                form=form,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = lap, {"value": u, "grad_sq": gs}
        elif ctx.power == 2:
            u, gs, lap, bil, dv = projected_deep_bilaplacian(
                x,
                layers,
                diffusion=ctx.diffusion,
                coordinates=ctx.coordinates,
                directions=dirs,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = bil, {"value": u, "grad_sq": gs, "laplacian": lap}
        else:
            u, gs, lap, bil, tri, dv = projected_deep_trilaplacian(
                x,
                layers,
                coordinates=ctx.coordinates,
                directions=dirs,
                allow_almost_everywhere=ctx.allow_ae,
            )
            value, aux = tri, {"value": u, "grad_sq": gs, "laplacian": lap, "bilaplacian": bil}
        if dv is not None:
            aux["directional"] = dv
        return _squeeze(ctx, (value, aux))

    return run


def _lap_fused(ctx: _Ctx) -> Runner:
    """One Pallas kernel per forward pass and per backward pass (GPU, first-order parameter derivatives)."""

    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        from omnibias.jax.experimental.fused_streams import fused_second_order

        r = fused_second_order(
            x,
            _layers(ctx, params),
            diffusion=ctx.diffusion,
            coordinates=ctx.coordinates,
            allow_almost_everywhere=ctx.allow_ae,
            precision=ctx.matmul,  # the policy of this operator: TF32 only when it was asked for
        )
        return _squeeze(ctx, (r.laplacian, {"value": r.value, "grad_sq": r.grad_sq}))

    return run


def _lap_taylor_closed(ctx: _Ctx) -> Runner:
    """Closed-form Taylor streams: ``Delta u = sum_c D^2_{v_c} u`` and ``Delta^2 u`` by polarized fourth-order jets."""

    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        layers = _layers(ctx, params)
        if ctx.power == 1:
            V = jnp.asarray(ctx.columns(), x.dtype)
            jet = directional_jet_streams(x, layers, V, 2, allow_almost_everywhere=ctx.allow_ae)
            return _squeeze(
                ctx, (jnp.sum(jet.derivatives[..., 1, :], axis=-2), {"value": jet.value})
            )
        u, bil = bilaplacian_by_jets(
            x, layers, coordinates=ctx.coordinates, allow_almost_everywhere=ctx.allow_ae
        )
        return _squeeze(ctx, (bil, {"value": u}))

    return run


def _lap_forward(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        u, g, lap = deep_field_value_grad_laplacian(
            x, _layers(ctx, params), allow_almost_everywhere=ctx.allow_ae
        )
        return _squeeze(ctx, (lap, {"value": u, "grad": g}))

    return run


def _lap_polylaplacian(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        out = deep_field_polylaplacian(x, _layers(ctx, params), ctx.power, mode="support")
        return _squeeze(ctx, (out, {}))

    return run


def _second_derivative_along(ctx: _Ctx, params: Any, xi: Array, v: Array) -> tuple[Array, Array]:
    """``(f(xi), v^T H(xi) v)`` by one second-order Taylor-mode pass along ``v``."""

    def f(z: Array) -> Array:
        return ctx.point(params, z)

    primal, (_, d2) = _jet.jet(f, (xi,), ((v, jnp.zeros_like(v)),))  # type: ignore[no-untyped-call,unused-ignore]
    return primal, d2


def _lap_jet(ctx: _Ctx, *, reduced: bool = False) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        V = ctx.reduced_columns(params, x.dtype) if reduced else jnp.asarray(ctx.columns(), x.dtype)

        def g(xi: Array) -> tuple[Array, Array]:
            u, d2 = jax.vmap(lambda v: _second_derivative_along(ctx, params, xi, v))(V.T)
            return u[0], jnp.sum(d2, axis=0)

        u, lap = _points(ctx, g, x)
        return u_lap(ctx, u, lap)

    return run


def _lap_subspace_nested(ctx: _Ctx) -> Runner:
    """``Delta^p`` by ``p`` nested forward-over-forward sums along the reduced directions of the input products."""

    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        V = ctx.reduced_columns(params, x.dtype).T

        def second(g: Callable[[Array], Array]) -> Callable[[Array], Array]:
            def h(z: Array) -> Array:
                def along(v: Array) -> Array:
                    return cast(
                        Array,
                        jax.jvp(lambda w: jax.jvp(g, (w,), (v,))[1], (z,), (v,))[1],
                    )

                return jnp.sum(jax.vmap(along)(V), axis=0)

            return h

        g: Callable[[Array], Array] = lambda z: ctx.point(params, z)  # noqa: E731
        for _ in range(ctx.power):
            g = second(g)
        lap = _points(ctx, g, x)
        u = _points(ctx, lambda z: ctx.point(params, z), x)
        return lap, {"value": u}

    return run


def u_lap(ctx: _Ctx, u: Array, lap: Array) -> tuple[Array, dict[str, Array]]:
    del ctx
    return lap, {"value": u}


def _estimator(ctx: _Ctx, kind: str) -> Runner:
    """Stochastic Laplacian: ``n`` second directional derivatives per point, probes independent across points.

    ``hutchinson`` averages Rademacher combinations of the metric columns, ``stde`` Gaussian ones, and ``sdgd`` samples
    ``s`` of the ``r`` columns without replacement and rescales by ``r / s``. The probe count follows the request's
    ``accuracy`` (:func:`omnibias.core.planner.estimator_probes`) unless ``n_probes`` was given. A training objective
    (``differentiate="params"``) needs ``key``: a silent fixed seed would give every step the same probes.
    """

    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        if key is None and ctx.differentiate != "none":
            raise ValueError(
                f"the {kind} estimator draws random probes: pass a PRNG key, op(params, x, key=key), "
                "and a new key every training step"
            )
        k = jax.random.PRNGKey(0) if key is None else key
        V = jnp.asarray(ctx.columns(), x.dtype)
        r = V.shape[1]
        n = ctx.n_probes if ctx.n_probes is not None else estimator_probes(kind, ctx.accuracy, r)
        xb = x if ctx.batched else x[None, :]
        keys = jax.random.split(k, xb.shape[0])

        def draw(kk: Array) -> tuple[Array, float]:
            if kind == "hutchinson":
                return jax.random.rademacher(kk, (n, r), dtype=x.dtype) @ V.T, 1.0 / n
            if kind == "stde":
                return jax.random.normal(kk, (n, r), dtype=x.dtype) @ V.T, 1.0 / n
            s = min(n, r)
            pick = jax.random.permutation(kk, r)[:s]
            return V.T[pick], float(r) / s

        def g(xi: Array, kk: Array) -> tuple[Array, Array]:
            probes, scale = draw(kk)
            u, d2 = jax.vmap(lambda v: _second_derivative_along(ctx, params, xi, v))(probes)
            return u[0], scale * jnp.sum(d2, axis=0)

        u, lap = jax.vmap(g)(xb, keys)
        if not ctx.batched:
            u, lap = u[0], lap[0]
        return u_lap(ctx, u, lap)

    return run


def _lap_hutchinson(ctx: _Ctx) -> Runner:
    return _estimator(ctx, "hutchinson")


def _lap_nested(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        V = ctx.columns()
        A = jnp.asarray(V @ V.T, x.dtype)

        def once(g: Callable[[Array], Array]) -> Callable[[Array], Array]:
            def h(z: Array) -> Array:
                return jnp.einsum("...ij,ij->...", jax.hessian(g)(z), A)

            return h

        g: Callable[[Array], Array] = lambda z: ctx.point(params, z)  # noqa: E731
        for _ in range(ctx.power):
            g = once(g)
        lap = _points(ctx, g, x)
        u = _points(ctx, lambda z: ctx.point(params, z), x)
        return lap, {"value": u}

    return run


def _div_projected(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        u, div = projected_deep_divergence(x, _layers(ctx, params), coordinates=ctx.coordinates)
        return _squeeze(ctx, (div, {"value": u}))

    return run


def _jacobian(ctx: _Ctx, params: Any, xi: Array) -> Array:
    return cast(Array, jax.jacfwd(lambda z: ctx.point(params, z))(xi))


def _div_jacobian(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        def g(xi: Array) -> tuple[Array, Array]:
            J = _jacobian(ctx, params, xi)
            idx = ctx.pairing(J.shape[0])
            return ctx.point(params, xi), jnp.sum(J[jnp.arange(J.shape[0]), idx])

        u, div = _points(ctx, g, x)
        return div, {"value": u}

    return run


def _logdet_projected(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        u, sign, logabs = projected_deep_logdet(
            x, _layers(ctx, params), coordinates=ctx.coordinates
        )
        return logabs, {"value": u, "sign": sign}

    return run


def _logdet_dense(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        def g(xi: Array) -> tuple[Array, Array, Array]:
            J = _jacobian(ctx, params, xi)
            idx = ctx.pairing(J.shape[0])
            M = jnp.eye(J.shape[0], dtype=J.dtype) + J[:, idx]
            sign, logabs = jnp.linalg.slogdet(M)
            return ctx.point(params, xi), sign, logabs

        u, sign, logabs = _points(ctx, g, x)
        return logabs, {"value": u, "sign": sign}

    return run


def _solve_woodbury(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        if not extra:
            raise ValueError("operator='solve' needs the right-hand side: op(params, x, rhs)")
        u, y = residual_solve(
            x, _layers(ctx, params), jnp.asarray(extra[0]), transpose=ctx.transpose
        )
        return y, {"value": u}

    return run


def _solve_dense(ctx: _Ctx) -> Runner:
    def run(
        params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        if not extra:
            raise ValueError("operator='solve' needs the right-hand side: op(params, x, rhs)")
        rhs = jnp.asarray(extra[0])

        def g(xi: Array, ri: Array) -> tuple[Array, Array]:
            J = _jacobian(ctx, params, xi)
            M = jnp.eye(J.shape[0], dtype=J.dtype) + (J.T if ctx.transpose else J)
            return ctx.point(params, xi), jnp.linalg.solve(M, ri)

        u, y = jax.vmap(g)(x, rhs) if ctx.batched else g(x, rhs)
        return y, {"value": u}

    return run


_EXECUTORS: dict[tuple[str, str], Callable[[_Ctx], Runner]] = {
    ("laplacian", "projected"): _lap_projected,
    ("laplacian", "forward_laplacian"): _lap_forward,
    ("laplacian", "forward_polylaplacian"): _lap_polylaplacian,
    ("laplacian", "taylor_jet"): _lap_jet,
    ("laplacian", "subspace_jet"): lambda ctx: _lap_jet(ctx, reduced=True),
    ("laplacian", "subspace_nested"): _lap_subspace_nested,
    ("laplacian", "nested_autodiff"): _lap_nested,
    ("laplacian", "hutchinson"): _lap_hutchinson,
    ("laplacian", "stde"): lambda ctx: _estimator(ctx, "stde"),
    ("laplacian", "sdgd"): lambda ctx: _estimator(ctx, "sdgd"),
    ("laplacian", "fused_streams"): _lap_fused,
    ("laplacian", "projected_cholesky"): lambda ctx: _lap_projected(ctx, "cholesky"),
    ("laplacian", "taylor_closed"): _lap_taylor_closed,
    ("divergence", "projected"): _div_projected,
    ("divergence", "jacobian_trace"): _div_jacobian,
    ("logdet", "projected"): _logdet_projected,
    ("logdet", "dense_jacobian"): _logdet_dense,
    ("solve", "woodbury"): _solve_woodbury,
    ("solve", "dense_jacobian"): _solve_dense,
}

#: Generic executors by operator, in the order the fallback tries them when the model is not a recognised chain.
_GENERIC_ORDER: dict[str, tuple[str, ...]] = {
    "laplacian": ("taylor_jet", "nested_autodiff"),
    "divergence": ("jacobian_trace",),
    "logdet": ("dense_jacobian",),
    "solve": ("dense_jacobian",),
}


# --------------------------------------------------------------------------------------- external engines


def _engine_runner(operator: str, engine: Callable[..., Any]) -> Callable[[_Ctx], Runner]:
    def factory(ctx: _Ctx) -> Runner:
        def run(
            params: Any, x: Array, extra: tuple[Any, ...], key: Any
        ) -> tuple[Array, dict[str, Array]]:
            def f(z: Array) -> Array:
                out = ctx.point(params, z)
                return jnp.reshape(out, ()) if operator == "laplacian" else out

            xb = x if ctx.batched else x[None, :]
            value = jnp.asarray(
                engine(f, xb, power=ctx.power) if operator == "laplacian" else engine(f, xb)
            )
            u = jax.vmap(lambda z: ctx.point(params, z))(xb)  # the model's own output shape
            if operator == "laplacian":
                value = value.reshape(u.shape)
            if not ctx.batched:
                value, u = value[0], u[0]
            return value, {"value": u}

        return run

    return factory


def register_engine(
    name: str,
    operator: str,
    engine: Callable[..., Any],
    *,
    work: Callable[[Descriptor], Work] | str,
    requires: Callable[[Descriptor], str | None] | None = None,
    max_power: int = 1,
    exact: bool = True,
    generic: bool = False,
    replace: bool = False,
) -> Algorithm:
    """Let the planner choose a third-party engine for ``operator`` (``"laplacian"`` or ``"divergence"``).

    ``engine(f, x, power=p)`` (``engine(f, x)`` for the divergence) receives the pointwise model ``f`` (one input point, a
    scalar or, for the divergence, a vector) and a batch ``x`` of shape ``(B, D)`` and returns the operator at every point,
    shape ``(B,)``; it must be traceable by JAX and differentiable if gradients are wanted. ``work`` prices it: a function
    ``descriptor -> Work`` or the name of a registered algorithm to be priced like. The engine is refused, with a reason,
    for requests it cannot serve (see :func:`omnibias.core.engines.engine_algorithm`), competes with the built-in programs
    on the cost model, can be timed by ``mode="find"`` and chosen by ``force=name``. ``generic=True`` also makes it
    available for models that are not a recognised dense chain (after the built-in generic executors, or by ``force``).
    """
    algorithm = engine_algorithm(
        name,
        operator,
        work=work,
        backend="jax",
        requires=requires,
        max_power=max_power,
        exact=exact,
        generic=generic,
    )
    if (operator, name) in _EXECUTORS and not replace:
        raise ValueError(f"an executor named {name!r} already serves {operator!r}")
    register_algorithm(algorithm, replace=replace)
    _EXECUTORS[(operator, name)] = _engine_runner(operator, engine)
    if generic and name not in _GENERIC_ORDER[operator]:
        _GENERIC_ORDER[operator] = (*_GENERIC_ORDER[operator], name)
    return algorithm


def unregister_engine(name: str, operator: str) -> None:
    """Remove an engine added by :func:`register_engine`; the built-in programs cannot be removed."""
    if name in _BUILTIN_EXECUTORS.get(operator, ()):
        raise ValueError(f"{name!r} is a built-in program of {operator!r}, not a registered engine")
    unregister_algorithm(name, frozenset({operator}))
    _EXECUTORS.pop((operator, name), None)
    _GENERIC_ORDER[operator] = tuple(a for a in _GENERIC_ORDER[operator] if a != name)


_BUILTIN_EXECUTORS: dict[str, tuple[str, ...]] = {}
for _op, _name in _EXECUTORS:
    _BUILTIN_EXECUTORS[_op] = (*_BUILTIN_EXECUTORS.get(_op, ()), _name)


# --------------------------------------------------------------------------------- abstract arguments, default pool


def _has_abstract(args: Any) -> bool:
    return any(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree_util.tree_leaves(args))


def _materialize(args: Any) -> Any:
    """Zeros for every :class:`jax.ShapeDtypeStruct` leaf: planning reads shapes and dtypes, never values."""
    return jax.tree_util.tree_map(
        lambda a: jnp.zeros(a.shape, a.dtype) if isinstance(a, jax.ShapeDtypeStruct) else a, args
    )


_PERSISTENT: dict[str, PlanCache] = {}


def _persistent_cache() -> PlanCache:
    """The on-disk plan cache of ``mode="auto"`` (``$OMNIBIAS_PLAN_CACHE=off`` keeps it in process)."""
    path = default_plan_cache_path()
    if path is None:
        return _DEFAULT_CACHE
    key = str(path)
    if key not in _PERSISTENT:
        _PERSISTENT[key] = PlanCache(path)
    return _PERSISTENT[key]


_DEFAULTS_DONE = False


def _ensure_default_engines() -> None:
    """Register ``folx`` as a planner engine when it is installed, so the planner prices it beside its own programs.

    The engine is a candidate like any other: ranked by the cost model (priced like ``forward_laplacian``), timed by
    ``mode="find"``, named in ``explain()``. ``OMNIBIAS_FOLX_ENGINE=off`` opts out; a user's own engine named ``folx`` wins.
    """
    global _DEFAULTS_DONE
    if _DEFAULTS_DONE:
        return
    _DEFAULTS_DONE = True
    if os.environ.get("OMNIBIAS_FOLX_ENGINE", "").lower() in ("off", "0", "false"):
        return
    if importlib.util.find_spec("folx") is None or ("laplacian", "folx") in _EXECUTORS:
        return
    from omnibias.jax.folx_compat import register_folx_engine

    register_folx_engine()


# ------------------------------------------------------------------------------------------------- front end


def _versions() -> dict[str, str]:
    out = {"jax": jax.__version__}
    for pkg in ("omnibias-core", "omnibias-jax"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


_DEFAULT_CACHE = PlanCache()


def _device_description(device: Any | None) -> tuple[str, Any]:
    dev = device if device is not None else jax.devices()[0]
    kind = getattr(dev, "device_kind", "") or getattr(dev, "platform", "cpu")
    cls = device_class(str(kind))
    if cls == "cpu" and os.environ.get("OMNIBIAS_PALLAS_INTERPRET", "") == "1":
        # tests and debugging: accelerator kernel programs stay candidates and run in the Pallas interpreter
        cls = "cpu-interpret"
    return cls, dev


def _default_budget(dev: Any) -> float | None:
    if getattr(dev, "platform", "cpu") == "cpu":
        return None
    try:
        stats = dev.memory_stats() or {}
    except Exception:  # noqa: BLE001 - not every backend reports memory
        return None
    limit = stats.get("bytes_limit")
    return None if limit is None else 0.75 * float(limit)


def _metric_of(diffusion: Any) -> str:
    if diffusion is None:
        return "euclidean"
    return "diagonal" if np.ndim(diffusion) == 1 else "factor"


def _with_policy(
    run: Runner,
    precision: PrecisionPolicy,
    compute_dtype: str | None,
    params: Any,
    x: Array,
    extra: tuple[Any, ...],
    key: Any,
    *,
    training: bool = False,
) -> tuple[Array, dict[str, Array]]:
    """``run`` under the matmul precision and compute dtype of a compiled operator: what its calls execute.

    ``find`` times a candidate through this same function, so it measures the program the operator will run.
    ``training`` traces the program for parameter gradients (``differentiate="params"``): the stream engines take their
    training layout, which on an accelerator took 0.5 to 0.7 of the stack layout's time in the value-and-gradient pass.
    """
    if training:
        with training_layout():
            return _with_policy(run, precision, compute_dtype, params, x, extra, key)
    name = precision.jax_name
    if name is None:
        return _wide(run, compute_dtype, params, x, extra, key)
    with jax.default_matmul_precision(name):
        return _wide(run, compute_dtype, params, x, extra, key)


def _wide(
    run: Runner, compute_dtype: str | None, params: Any, x: Array, extra: tuple[Any, ...], key: Any
) -> tuple[Array, dict[str, Array]]:
    if compute_dtype is None:
        return run(params, x, extra, key)
    wide = jnp.dtype(compute_dtype)
    ref = x.dtype

    def lift(a: Any) -> Any:
        return a.astype(wide) if jnp.issubdtype(jnp.result_type(a), jnp.floating) else a

    value, aux = run(
        jax.tree_util.tree_map(lift, params),
        lift(x),
        jax.tree_util.tree_map(lift, extra),
        key,
    )
    return value.astype(ref), {k: v.astype(ref) for k, v in aux.items()}


class CompiledOperator:
    """A planned derivative operator.  Call it like the model: ``op(params, x)`` (``op(x)`` for a bound model).

    ``__call__`` returns the operator value; :meth:`evaluate` also returns the by-products the chosen program formed
    (the model value, ``|S^T grad|^2``, lower Laplace powers, ...).  :meth:`explain` reports what was matched, what was
    chosen and why, what the alternatives were predicted to cost, and what was refused.
    """

    def __init__(
        self,
        ctx: _Ctx,
        *,
        plan: Plan,
        matched: bool,
        fallback_reason: str | None,
        chain: Mapping[str, Any] | None,
        request: OperatorRequest,
        prepare: Callable[[Any], Any],
        bound: bool,
        jit: bool,
        bucket: str | int | None = None,
        policy: PolicyChoice | None = None,
        alternatives: Sequence[str] = (),
        compute_dtype: str | None = None,
        precision: PrecisionPolicy | None = None,
        precision_notes: Sequence[str] = (),
    ) -> None:
        self._ctx = ctx
        self.compute_dtype = compute_dtype
        self.precision = precision if precision is not None else PrecisionPolicy()
        self.precision_notes = list(precision_notes)
        self.plan = plan
        self._jit_enabled = jit
        self._alternatives = list(alternatives)
        self.matched = matched
        self.fallback_reason = fallback_reason
        self._chain = None if chain is None else dict(chain)
        self.request = request
        self._prepare = prepare
        self._bound = bound
        self.bucket = bucket
        self.policy = (
            policy
            if policy is not None
            else resolve_compile_policy(
                "default",
                platform="cpu",
                predicted_seconds=None,
                cold_seconds=None,
                expected_uses=1,
            )
        )
        self._rebuild()
        self.first_call_seconds: float | None = None
        self.executor_failures: list[Refusal] = []
        self.descriptor: Descriptor | None = (
            None  # the planner's problem description (None for an unmatched model)
        )

    def _rebuild(self) -> None:
        self._run = _EXECUTORS[(self.request.kind, self.plan.algorithm)](self._ctx)
        self._staged: StagedProgram | None = (
            StagedProgram(
                self._call_impl,
                options=self.policy.options,
                memory_budget=self.plan.memory_budget,
                retry=self.policy.requested == "auto",
            )
            if self._jit_enabled
            else None
        )
        self._apply: Callable[..., Any] = self._call_impl
        if self._staged is not None:
            self._apply = self._staged

    def _advance(self, err: MemoryBudgetExceeded) -> bool:
        """Drop the current program because its compiled size exceeds the budget; ``False`` when none is left."""
        self.executor_failures.append(
            Refusal(self.plan.algorithm, f"refused before running: {err}")
        )
        if not self._alternatives:
            return False
        name = self._alternatives.pop(0)
        self.plan = replace(self.plan, algorithm=name, mode=self.plan.mode + "+memory_retry")
        self._rebuild()
        return True

    def _guarded(self, call: Callable[[], Any]) -> Any:
        """Run ``call``; a program refused for its compiled memory is replaced by the next feasible candidate."""
        while True:
            try:
                return call()
            except MemoryBudgetExceeded as err:
                if not self._advance(err):
                    err.add_note(
                        "every candidate was refused: "
                        + "; ".join(f"{f.algorithm}: {f.reason}" for f in self.executor_failures)
                    )
                    raise

    @property
    def algorithm(self) -> str:
        return self.plan.algorithm

    def _call_impl(
        self, params: Any, x: Array, extra: tuple[Any, ...], key: Any
    ) -> tuple[Array, dict[str, Array]]:
        return _with_policy(
            self._run,
            self.precision,
            self.compute_dtype,
            params,
            x,
            extra,
            key,
            training=self._ctx.differentiate != "none",
        )

    def prepared(self, *args: Any) -> tuple[Any, ...]:
        """The arguments of the compiled program for ``op(*args)``: ``(params, x, extra)``, after bucketing.

        A bucketed batch is padded here (the padding repeats the last real row); the first ``len(x)`` rows are real.
        """
        params, x, extra, _, _ = self._arrange(args)
        return params, x, extra

    def _arrange(self, args: tuple[Any, ...]) -> tuple[Any, Array, tuple[Any, ...], int, int]:
        if self._bound:
            params: Any = None
            x, extra = args[0], tuple(args[1:])
        else:
            params, x, extra = self._prepare(args[0]), args[1], tuple(args[2:])
        # converting an array that already is one costs tens of microseconds a call
        if not isinstance(x, jax.Array):
            x = jnp.asarray(x)
        n = int(x.shape[0]) if x.ndim == 2 else 0
        n_run = n
        if self.bucket is not None and x.ndim == 2:
            n_run = bucket_size(n, self.bucket)
            if n_run != n:
                x = pad_rows(x, n_run)
                extra = tuple(
                    pad_rows(e, n_run) if jnp.ndim(e) >= 1 and jnp.shape(e)[0] == n else e
                    for e in extra
                )
        return params, x, extra, n, n_run

    def warmup(self, *args: Any, key: Any = None) -> dict[str, Any]:
        """Build the executable for these arguments now, without evaluating; returns its stage times.

        The first :meth:`evaluate` of those shapes then runs without compiling.  With ``bucket`` set, every batch size
        that rounds to the same bucket shares the executable.
        """
        params, x, extra, _, _ = self._arrange(args)
        if self._staged is None:
            raise ValueError("this operator was built with jit=False; there is nothing to compile")

        def compile_now() -> dict[str, Any]:
            staged = self._staged
            assert staged is not None
            return staged.compile(params, x, extra, key).to_config()

        result: dict[str, Any] = self._guarded(compile_now)
        return result

    def export(self, *args: Any, key: Any = None) -> ExportedProgram:
        """Export the planned program for these argument shapes as StableHLO (see :class:`ExportedProgram`).

        The artifact is called as ``program.call(*op.prepared(*args), key)`` and carries the plan.
        """
        params, x, extra, _, _ = self._arrange(args)
        return export_program(self._call_impl, params, x, extra, key, metadata=self.plan.explain())

    def evaluate(self, *args: Any, key: Any = None) -> OperatorOutput:
        params, x, extra, n, n_run = self._arrange(args)
        first = self.first_call_seconds is None
        t0 = time.perf_counter()
        value, aux = self._guarded(lambda: self._apply(params, x, extra, key))
        if first and not is_traced(value):
            _block(value)
            self.first_call_seconds = time.perf_counter() - t0
        if n_run != n:
            value = value[:n]
            aux = {k: (v[:n] if v.ndim >= 1 and v.shape[0] == n_run else v) for k, v in aux.items()}
        return OperatorOutput(value, aux)

    def __call__(self, *args: Any, key: Any = None) -> Array:
        return self.evaluate(*args, key=key).value

    def explain(self) -> dict[str, Any]:
        out = self.plan.explain()
        out.update(
            {
                "backend": "jax",
                "matched_patterns": ["dense_chain"] if self.matched else [],
                "fallback_reason": self.fallback_reason,
                "input_products": (
                    None
                    if self._ctx.input_map is None
                    else {
                        "products": len(self._ctx.input_map.entries),
                        "rows": self._ctx.input_map.rows,
                    }
                ),
                "chain": self._chain,
                "request": self.request.to_config(),
                "first_call_seconds": self.first_call_seconds,
                "compute_dtype": self.compute_dtype,
                "precision": {**self.precision.to_config(), "notes": self.precision_notes},
                "executor_failures": [
                    {"algorithm": r.algorithm, "reason": r.reason} for r in self.executor_failures
                ],
                "cold_cost": {
                    "predicted_seconds": self.plan.compile_seconds,
                    "bucket": self.bucket,
                    "compile_policy": self.policy.to_config(),
                    "persistent_cache_dir": compilation_cache_dir(),
                    "executables": 0 if self._staged is None else self._staged.signatures(),
                    "stages": [] if self._staged is None else self._staged.report(),
                },
            }
        )
        return out


@dataclass(frozen=True)
class _Setup:
    """What a model contributes to the plan: its function, example parameters and input, and the chain accessor."""

    fn: Callable[[Any, Array], Array]
    params: Any
    x: Array
    layers: Callable[[Any], list[tuple[Array, Array | None, Any]]] | None
    prepare: Callable[[Any], Any]
    bound: bool
    squeeze: bool
    fallback_reason: str | None
    input_map: InputMap | None = None
    input_map_reason: str | None = None
    leaves: Callable[[Any], Sequence[Any]] | None = None

    @property
    def matched(self) -> bool:
        return self.layers is not None


def _is_layer_model(model: Any) -> bool:
    return (
        hasattr(model, "layer_specs")
        or hasattr(model, "_layer_specs")
        or (isinstance(model, Sequence) and not isinstance(model, str))
    )


def _setup_layers(model: Any, example_args: Any) -> _Setup:
    layers0 = layers_of(model)
    first = example_args[0] if isinstance(example_args, tuple | list) else example_args
    specs = [spec for _, _, spec in layers0]
    names = [None if s is None else get_activation(s).name for s in specs]

    def fn(arrays: Any, x: Array) -> Array:
        return _chain_value([(W, b, s) for (W, b), s in zip(arrays, specs, strict=True)], x)

    def prepare(p: Any) -> Any:
        ls = layers_of(p)
        got = [None if s is None else get_activation(s).name for _, _, s in ls]
        if got != names:
            raise ValueError(
                "the layer activations differ from those the operator was compiled for"
            )
        return [(W, b) for W, b, _ in ls]

    def layers(arrays: Any) -> list[tuple[Array, Array | None, Any]]:
        return [(jnp.asarray(W), b, s) for (W, b), s in zip(arrays, specs, strict=True)]

    return _Setup(fn, prepare(layers0), jnp.asarray(first), layers, prepare, False, False, None)


def _without_rhs(model: Callable[..., Any], example_args: Any) -> Any:
    """``operator="solve"`` takes ``(params, x, rhs)`` at call time; accept the right-hand side among the example arguments.

    The last example argument is dropped when the model rejects all of them and accepts all the others, so
    ``compile_operator(model, (params, x, rhs), operator="solve")`` works as it does for the PyTorch front end.
    """
    if not isinstance(example_args, tuple | list) or len(example_args) < 2:
        return example_args
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        return example_args
    args = tuple(example_args)
    try:
        signature.bind(*args)
        return example_args
    except TypeError:
        pass
    try:
        signature.bind(*args[:-1])
    except TypeError:
        return example_args
    return args[:-1]


def _setup_callable(
    model: Callable[..., Any], example_args: Any, params_argnum: int | None
) -> _Setup:
    args = tuple(example_args) if isinstance(example_args, tuple | list) else (example_args,)
    match, reason = match_dense_chain(model, args, params_argnum=params_argnum)
    layers: Callable[[Any], list[tuple[Array, Array | None, Any]]] | None = None
    if match is not None:
        if params_argnum is None:
            consts = match.leaf_values

            def layers(_p: Any) -> list[tuple[Array, Array | None, Any]]:
                assert match is not None
                return match.layers_from(consts)

        else:

            def layers(p: Any) -> list[tuple[Array, Array | None, Any]]:
                assert match is not None
                return match.layers_from(jax.tree_util.tree_leaves(p))

    squeeze = bool(match is not None and match.squeeze_output)
    imap, imap_reason = (
        (None, None)
        if match is not None
        else analyze_input_map(model, args, params_argnum=params_argnum)
    )
    if params_argnum is None:
        bound_leaves = _bound_consts(model, args)
        return _Setup(
            lambda _p, x: model(x),
            None,
            jnp.asarray(args[0]),
            layers,
            lambda p: p,
            True,
            squeeze,
            reason,
            imap,
            imap_reason,
            lambda _p: bound_leaves,
        )
    x = next(a for k, a in enumerate(args) if k != params_argnum)
    return _Setup(
        model,
        args[params_argnum],
        jnp.asarray(x),
        layers,
        lambda p: p,
        False,
        squeeze,
        reason,
        imap,
        imap_reason,
        jax.tree_util.tree_leaves,
    )


def _bound_consts(model: Callable[..., Any], args: Sequence[Any]) -> list[Any]:
    """The constants a bound model closes over, in the order its jaxpr lists them."""
    return list(jax.make_jaxpr(model)(*args).consts)


def _safe_timer(
    model: Any,
    setup: _Setup,
    x0: Array,
    failures: list[Refusal],
    timeout: float,
    kwargs: Mapping[str, Any],
) -> Callable[[str], float]:
    """The ``find`` timer that runs each candidate in its own process (``compile_operator(safe_find=True)``)."""
    from omnibias.jax.experimental._safe_find import SafeFindError, time_candidate_in_subprocess

    if setup.layers is None or not _is_layer_model(model):
        raise ValueError(
            "safe_find times candidates in a fresh process and needs a layer-list model: a callable model closes over "
            "objects that do not cross a process boundary"
        )
    layers: list[tuple[Array, Array | None, str | None]] = []
    for W, b, spec in layers_of(model):
        if spec is None:
            layers.append((W, b, None))
            continue
        name = get_activation(spec).name
        if get_activation(name).name != name:
            raise ValueError(f"safe_find needs activations that resolve by name, not {spec!r}")
        layers.append((W, b, name))
    x64 = bool(jax.config.read("jax_enable_x64"))  # type: ignore[no-untyped-call]
    clean = {k: v for k, v in kwargs.items() if v is not None}

    def timer(name: str) -> float:
        try:
            return time_candidate_in_subprocess(layers, x0, name, clean, x64=x64, timeout=timeout)
        except SafeFindError as err:
            failures.append(Refusal(name, f"find: {err}"))
            if err.returncode is not None:
                raise FindRefusal(str(err)) from err  # its process crashed: it will again, so the plan keeps the refusal
            return float("inf")  # a timeout or a missing timing may be load: the plan is not cached

    return timer


def compile_operator(
    model: Any,
    example_args: Any,
    *,
    operator: str = "laplacian",
    power: int = 1,
    params_argnum: int | None = 0,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Any = None,
    directions: Any = None,
    transpose: bool = False,
    mode: str = "heuristic",
    memory_budget: float | None = None,
    expected_uses: int | str = 1,
    force: str | None = None,
    exact: bool = True,
    allow_almost_everywhere: bool = False,
    n_probes: int | None = None,
    device: Any | None = None,
    cache: PlanCache | str | None = None,
    jit: bool = True,
    cost_model: CostModel | None = None,
    bucket: str | int | None = None,
    compile_policy: str = "default",
    compute_dtype: str | None = None,
    precision: str | PrecisionPolicy | None = None,
    allow_low_precision: bool = False,
    differentiate: str = "none",
    accuracy: float | None = None,
    uses_hint: int | None = None,
    safe_find: bool = False,
    safe_find_timeout: float = 600.0,
    top_k: int | None = FIND_TOP_K,
) -> CompiledOperator:
    r"""Plan ``operator`` for ``model`` on ``example_args`` and return a callable.

    Parameters
    ----------
    model : callable ``model(params, x)``, a layer list ``[(W, b, spec), ...]`` or an object with ``layer_specs()``.
        ``x`` has shape ``(D,)`` or ``(B, D)``; the output ``(C,)``/``(B, C)`` or a scalar/``(B,)``.
    example_args : ``(params, x)`` for a callable (``(x,)`` with ``params_argnum=None`` when the weights are closed
        over); ``x`` or ``(x,)`` for a layer list or layer object, which are then passed again at call time.
    operator : ``"laplacian"`` (``power`` > 1 for ``Delta^p``), ``"divergence"``, ``"logdet"`` (``log|det(I + J)|``) or
        ``"solve"`` (``(I + J) y = rhs`` with ``rhs`` the last call argument, which may also be listed last among ``example_args``).  Residual Gram, Gauss-Newton and NTK
        programs are planned by ``compile_residual``.
    coordinates, diffusion, directions : the operator acts on these inputs only (for example the space coordinates of
        a field that also takes time), with metric ``S S^T`` (``diffusion`` of shape ``(m,)`` or ``(m, r)``) and drift
        directions ``(D, q)``.
    mode : ``"heuristic"`` (rank by the per-device cost model), ``"find"`` (time the leading candidates on the
        example, as ``cudnnFind`` does, and keep the measured best) or ``"auto"`` (``find`` once ``expected_uses`` reaches
        :data:`omnibias.core.planner.AUTO_FIND_MIN_USES`, with the plan cached persistently; ``heuristic`` otherwise, and
        always when the arguments are abstract).
    memory_budget : bytes; candidates predicted to exceed it are refused with the reason.  ``None`` uses 75 % of the
        device memory on an accelerator and no limit on a CPU.
    expected_uses : calls the compile cost is amortised over when ranking, or ``"auto"``: ``uses_hint`` (a training API's
        step count) if given, else 1000 for ``differentiate="params"`` and 1 for a forward evaluation.
    differentiate : what the result is used for: ``"none"`` (a forward evaluation), ``"params"`` (value-and-gradient into
        the parameters: ranking uses forward plus backward work and launches, and ``find`` times that) or ``"params2"``
        (a second derivative into the parameters, as Newton and curvature methods need; the fused kernels are refused).
        A stochastic estimator then needs ``key=`` at call time.
    accuracy : with ``exact=False``, the standard error a stochastic estimator may have as a fraction of the Hessian's
        Frobenius norm (worst case); it sizes the probe count (:func:`omnibias.core.planner.estimator_probes`).
    uses_hint : the number of calls a training API knows it will make (see ``expected_uses="auto"``).
    top_k : how many candidates ``mode="find"`` times, in the cost model's order: ``2`` (default), or ``None`` for every
        feasible candidate, which cannot miss a winner the model ranks low.  Exhaustive find pays one compile and a few
        runs per candidate, once per key; the cut-off at 4x the best time and the host-memory skip still apply.  A
        non-default ``top_k`` is part of the cache key.
    force : name of an algorithm to use regardless of rank (it must be feasible).
    exact : ``False`` admits stochastic estimators (Hutchinson, STDE, SDGD); the default refuses them.
    allow_almost_everywhere : accept activations whose derivative jumps (ReLU family), as in the structured programs.
    cache : a :class:`~omnibias.core.planner.PlanCache` or a path for a persistent one (default: in process; a measured
        choice of ``mode="auto"`` uses the persistent cache at :func:`omnibias.core.planner.default_plan_cache_path`).
    bucket : round a batch up to ``"pow2"``, ``"geometric"`` or a multiple (an ``int``) so that new batch sizes reuse an
        executable; padding rows are discarded, so the result is that of the unpadded call (a stochastic estimator
        draws its probes for the padded batch).
    compute_dtype : ``"float64"`` evaluates a float32 problem in float64 (the derivative contractions then accumulate
        in double precision) and returns the input dtype; needs ``jax_enable_x64``. ``None`` computes in the input dtype.
    precision : ``"highest"`` (the default: float32 matrix products are computed in float32 on every device), ``"tf32"``
        (opt in to TF32 matrix engines for float32, about 1e-3 per product) or ``"default"`` (leave JAX's setting), or a
        :class:`~omnibias.core.precision.PrecisionPolicy`.  Half-precision inputs are refused for derivative order two and
        above unless ``allow_low_precision=True``; ``explain()`` reports the policy and its notes.
    safe_find, safe_find_timeout : with ``safe_find=True`` the candidates of ``mode="find"`` are timed each in a fresh process
        (:mod:`omnibias.jax.experimental._safe_find`), so a compiler crash or a hang ends that candidate (recorded in
        ``explain()["executor_failures"]``, ranked last) instead of the caller; layer models with named activations only, and
        each candidate pays its own interpreter start-up and compile (``safe_find_timeout`` seconds at most).
    compile_policy : ``"default"``, ``"fast"`` (no Triton GEMM), ``"fastest"`` (autotune level 0) or ``"auto"`` (see
        :func:`~omnibias.jax.experimental.coldcost.resolve_compile_policy`); accelerator only, recorded by ``explain()``.

    ``example_args`` may hold :class:`jax.ShapeDtypeStruct` leaves: planning needs shapes and dtypes only, so a program can be
    planned (and its :meth:`CompiledOperator.warmup` compiled) before any data exists. ``mode="find"`` times the example and
    refuses abstract arguments.

    Raises
    ------
    ValueError
        If no program is feasible for the request; the message lists every refusal reason.
    """
    _ensure_default_engines()
    abstract = _has_abstract(example_args)
    if abstract:
        if mode == "find":
            raise ValueError(
                "mode='find' times the program on the example arguments: pass concrete arrays, or use mode='auto'"
            )
        example_args = _materialize(example_args)
    if operator == "solve" and not _is_layer_model(model):
        example_args = _without_rhs(model, example_args)
    setup = (
        _setup_layers(model, example_args)
        if _is_layer_model(model)
        else _setup_callable(model, example_args, params_argnum)
    )
    x0 = setup.x
    if x0.ndim not in (1, 2):
        raise ValueError("x must have shape (D,) or (B, D)")
    batched = x0.ndim == 2
    coords = None if coordinates is None else np.asarray(coordinates, dtype=int)
    dirs = None if directions is None else np.asarray(directions)
    if compute_dtype not in (None, "float64"):
        raise ValueError("compute_dtype must be None or 'float64'")
    if compute_dtype == "float64" and not jax.config.read("jax_enable_x64"):  # type: ignore[no-untyped-call]
        raise ValueError("compute_dtype='float64' needs jax_enable_x64")
    request = OperatorRequest(
        operator,
        power=power,
        selected=None if coords is None else int(len(coords)),
        metric=_metric_of(diffusion),
        directions=0 if dirs is None else int(dirs.shape[-1]),
        exact=exact,
        allow_almost_everywhere=allow_almost_everywhere,
        differentiate=differentiate,
        accuracy=accuracy,
    )
    if differentiate != "none" and setup.params is None:
        raise ValueError(
            "differentiate='params' needs a model with explicit parameters (params_argnum), not one that closes over its weights"
        )
    if request.kind not in _GENERIC_ORDER:
        raise ValueError(
            f"operator {operator!r} is not an operator program; residual Gram, Gauss-Newton and NTK programs "
            "are planned by compile_residual"
        )
    policy = resolve_precision(precision, allow_low_precision)
    precision_notes = policy.check(
        jnp.float64 if compute_dtype == "float64" else x0.dtype,
        request_order(request.kind, request.power),
        what=f"operator {operator!r} power {request.power}",
    )
    ctx = _Ctx(
        fn=setup.fn,
        layers=setup.layers,
        squeeze=setup.squeeze,
        batched=batched,
        dim=int(x0.shape[-1]),
        power=request.power,
        coordinates=coords,
        diffusion=diffusion,
        directions=dirs,
        allow_ae=allow_almost_everywhere,
        transpose=transpose,
        n_probes=n_probes,
        input_map=setup.input_map,
        leaves=setup.leaves,
        differentiate=differentiate,
        accuracy=accuracy,
        matmul=policy.matmul,
    )
    dev_class, dev = _device_description(device)
    budget = memory_budget if memory_budget is not None else _default_budget(dev)
    resolved_mode, uses, uses_source = resolve_planning(
        mode, expected_uses, differentiate=differentiate, uses_hint=uses_hint, can_time=not abstract
    )
    plan_cache = (
        cache
        if isinstance(cache, PlanCache)
        else PlanCache(cache)
        if isinstance(cache, str)
        else _persistent_cache()
        if resolved_mode == "find"
        else _DEFAULT_CACHE
    )
    versions = _versions()
    rhs_probe: tuple[Any, ...] = (jnp.zeros_like(x0),) if request.kind == "solve" else ()
    probe_key = jax.random.PRNGKey(0)

    # What ``find`` times is what the operator runs: the matmul precision and compute dtype of its calls, the options of an
    # explicit compile policy, and the staged executable (a program whose compiled size exceeds the budget is refused
    # before it executes).  ``auto`` is timed with the default options: it is resolved from the winner's prediction.
    find_policy = resolve_compile_policy(
        "default" if compile_policy == "auto" else compile_policy,
        platform=str(getattr(dev, "platform", "cpu")),
        predicted_seconds=None,
        cold_seconds=None,
        expected_uses=uses,
    )

    def timer(name: str, deadline: float | None = None) -> float:
        run = _EXECUTORS[(request.kind, name)](ctx)
        fn: Callable[..., Any]
        if differentiate == "none":

            def fn(p: Any, x: Array, e: tuple[Any, ...]) -> Any:
                return _with_policy(run, policy, compute_dtype, p, x, e, probe_key)

        else:

            def loss(p: Any, x: Array, e: tuple[Any, ...]) -> Array:
                value, aux = _with_policy(run, policy, compute_dtype, p, x, e, probe_key, training=True)
                total = jnp.sum(value * value)
                return total + sum((jnp.sum(a) for a in aux.values() if a.dtype.kind == "f"), 0.0)

            fn = jax.value_and_grad(loss)
        staged = StagedProgram(
            fn,
            options=find_policy.options,
            memory_budget=budget,
            retry=compile_policy == "auto",
        )
        staged.compile(setup.params, x0, rhs_probe)  # MemoryBudgetExceeded here: refused before it runs
        _block(staged(setup.params, x0, rhs_probe))  # warm
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            _block(staged(setup.params, x0, rhs_probe))
            times.append(time.perf_counter() - t0)
            if i == 0 and deadline is not None and times[0] > deadline:
                return times[0]  # slower by the cut factor than the best so far: it cannot be the minimum
        return float(np.median(times))

    find_failures: list[Refusal] = []

    def guarded_timer(name: str, deadline: float | None = None) -> float:
        """A candidate that cannot be compiled or run (out of memory included) is a failed measurement, not a failed plan.

        A compiled size over the budget repeats for the same problem, budget and library versions, so it is raised as a
        refusal the plan keeps (and caches); any other failure is an infinite time, which keeps the plan out of the cache.
        """
        try:
            return timer(name, deadline)
        except Exception as err:  # noqa: BLE001
            what = f"{type(err).__name__}: {str(err)[:200]}"
            find_failures.append(Refusal(name, f"find: {what}"))
            if isinstance(err, MemoryBudgetExceeded):
                raise FindRefusal(what) from err
            return float("inf")

    find_timer: Callable[[str], float] = guarded_timer
    if safe_find:
        find_timer = _safe_timer(
            model,
            setup,
            x0,
            find_failures,
            safe_find_timeout,
            {
                "operator": operator,
                "power": power,
                "coordinates": None if coords is None else coords.tolist(),
                "diffusion": None if diffusion is None else np.asarray(diffusion),
                "directions": dirs,
                "transpose": transpose,
                "allow_almost_everywhere": allow_almost_everywhere,
                "exact": exact,
                "n_probes": n_probes,
                "compute_dtype": compute_dtype,
                "differentiate": differentiate,
                "accuracy": accuracy,
            },
        )

    chain_config: Mapping[str, Any] | None = None
    descriptor: Descriptor | None = None
    if setup.layers is not None:
        chain = describe_layers(setup.layers(setup.params))
        chain_config = chain.to_config()
        descriptor = Descriptor(request, chain, int(x0.shape[0]) if batched else 1)
        plan = select(
            descriptor,
            backend="jax",
            device=dev_class,
            cost_model=cost_model,
            memory_budget=budget,
            mode=mode,
            timer=None if abstract else find_timer,
            top_k=top_k,
            expected_uses=expected_uses,
            uses_hint=uses_hint,
            force=force,
            cache=plan_cache,
            versions=versions,
            key_extra=measurement_context(
                policy.matmul, compute_dtype, None if compile_policy in ("default", "auto") else compile_policy
            ),
            find_memory_limit=None if budget is not None else host_memory_limit(),
        )
        order = [plan.algorithm] + [
            c["name"] for c in plan.candidates if c["name"] != plan.algorithm
        ]
    else:
        plan, order = _fallback_plan(
            request,
            exact,
            force,
            dev_class,
            budget,
            uses,
            versions,
            ctx=ctx,
            reason=setup.input_map_reason,
            uses_source=uses_source,
        )
    if force is not None:
        # a forced program that cannot run is an error, never a silent substitution of another program
        order = [plan.algorithm]
    failures: list[Refusal] = list(find_failures)
    chosen = None
    for name in order:
        try:
            run = _EXECUTORS[(request.kind, name)](ctx)
            jax.eval_shape(
                lambda p, x, e, run=run: run(p, x, e, probe_key), setup.params, x0, rhs_probe
            )
        except Exception as err:  # noqa: BLE001 - an executor that cannot trace the model is a refusal, not a crash
            failures.append(
                Refusal(
                    name,
                    f"executor could not trace this model: {type(err).__name__}: {str(err)[:200]}",
                )
            )
            continue
        chosen = name
        break
    if chosen is None:
        raise ValueError(
            "no executor could run this model: "
            + "; ".join(f"{f.algorithm}: {f.reason}" for f in failures)
        )
    if chosen != plan.algorithm:
        plan = replace(plan, algorithm=chosen, mode=plan.mode + "+retry")
    op = CompiledOperator(
        ctx,
        plan=plan,
        matched=setup.matched,
        fallback_reason=setup.fallback_reason,
        chain=chain_config,
        request=request,
        prepare=setup.prepare,
        bound=setup.bound,
        jit=jit,
        bucket=bucket,
        compute_dtype=compute_dtype,
        precision=policy,
        precision_notes=precision_notes,
        alternatives=order[order.index(chosen) + 1 :],
        policy=resolve_compile_policy(
            compile_policy,
            platform=str(getattr(dev, "platform", "cpu")),
            predicted_seconds=plan.predicted_seconds,
            cold_seconds=plan.compile_seconds,
            expected_uses=uses,
        ),
    )
    op.executor_failures = failures
    op.descriptor = descriptor
    return op


def _fallback_plan(
    request: OperatorRequest,
    exact: bool,
    force: str | None,
    device: str,
    budget: float | None,
    expected_uses: int,
    versions: Mapping[str, str],
    *,
    ctx: _Ctx | None = None,
    reason: str | None = None,
    uses_source: str = "explicit",
) -> tuple[Plan, list[str]]:
    """Plan for a model that is not a recognised chain: only the generic executors, in a fixed order.

    A model that reads its input only through weight products is served first by the subspace programs, which
    differentiate along ``min(rows, columns)`` directions instead of every input coordinate.
    """
    generic = [
        a for a in _GENERIC_ORDER[request.kind] if not (a == "taylor_jet" and request.power > 1)
    ]
    refusals: list[Refusal] = []
    notes: dict[str, str] = {}
    if request.kind == "laplacian":
        if ctx is not None and ctx.input_map is not None:
            q = int(ctx.columns().shape[1])
            k = min(ctx.input_map.rows, q)
            if k < q:
                names = (
                    ["subspace_jet", "subspace_nested"]
                    if request.power == 1
                    else ["subspace_nested"]
                )
                generic = [*names, *generic]
                for name in names:
                    notes[name] = (
                        f"the input enters only through weight products of {ctx.input_map.rows} rows: "
                        f"{k} directions replace {q}"
                    )
            else:
                refusals.append(
                    Refusal(
                        "subspace",
                        f"no reduction: the input products have {ctx.input_map.rows} rows for {q} direction columns",
                    )
                )
        elif reason is not None:
            refusals.append(Refusal("subspace", reason))
    if not exact and request.kind == "laplacian" and request.power == 1:
        generic = ["hutchinson", "stde", "sdgd", *generic]
    if force is not None:
        if force not in generic:
            raise ValueError(
                f"cannot force {force!r} for a model that is not a dense chain; use one of {generic}"
            )
        generic = [force]
    plan = Plan(
        descriptor="unmatched",
        operator=request.kind,
        algorithm=generic[0],
        mode="forced" if force else "fallback",
        device=device,
        candidates=tuple(
            {
                "name": a,
                "seconds": None,
                "peak_bytes": None,
                "compile_seconds": None,
                "generic": True,
                **({"note": notes[a]} if a in notes else {}),
            }
            for a in generic
        ),
        refusals=tuple(refusals),
        memory_budget=budget,
        expected_uses=max(1, int(expected_uses)),
        versions={
            **versions,
            "cost_model": "none: unmatched model, generic executors are not priced",
        },
        objective="forward" if request.differentiate == "none" else request.differentiate,
        uses_source=uses_source,
    )
    return plan, generic

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The stable, planned derivative-operator API of the JAX back end.

``compile_operator`` describes a derivative request (Laplacian and powers, divergence, ``log|det(I + J)|``, a Jacobian solve) for
a model, lists the programs that can serve it, ranks them with a per-device cost model and returns an ordinary JAX callable with
``explain()``. ``compile_residual`` does the same for a PDE residual: the residual rows, their Gram matrix, the damped
Gauss-Newton step or the NTK diagonal. Plans are persisted in a versioned format (:data:`PLAN_FORMAT`); precision and conditioning
are policies, not accidents (``docs/precision.md``); ``docs/stability.md`` states what is stable and how names are retired.

These names were developed under ``omnibias.jax.experimental``, which keeps them as aliases on the schedule recorded in
:mod:`omnibias.core.deprecation`: silent until 0.6.0, a :class:`DeprecationWarning` from 0.6.0, removed in 0.7.0. The programs
themselves (closed-form ridge operators, first-layer projections, fused kernels) remain under ``experimental``: they are
reached through the planner, and their own names carry measured but scoped evidence (``docs/performance.md``).

:func:`conformance_engine` adapts the front end to :mod:`omnibias.core.conformance`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from omnibias.core.deprecation import Alias, register
from omnibias.core.planner import PLAN_FORMAT, Plan, PlanCache
from omnibias.core.precision import PrecisionPolicy
from omnibias.jax.experimental.chain import describe_layers, layers_of
from omnibias.jax.experimental.coldcost import (
    ExportedProgram,
    StagedProgram,
    bucket_size,
    enable_compilation_cache,
)
from omnibias.jax.experimental.compiler import (
    ChainMatch,
    CompiledOperator,
    LayerMatch,
    OperatorOutput,
    compile_operator,
    match_dense_chain,
    register_engine,
    unregister_engine,
)
from omnibias.jax.experimental.conditioning import ConditioningPolicy
from omnibias.jax.experimental.demand import Quantities, Residual
from omnibias.jax.experimental.distributed import (
    distributed_residual_gauss_newton,
    distributed_residual_gram,
    distributed_residual_ntk_diagonal,
    shard_points,
)
from omnibias.jax.experimental.residual_compiler import CompiledResidual, compile_residual

__all__ = [
    "ChainMatch",
    "CompiledOperator",
    "CompiledResidual",
    "ConditioningPolicy",
    "ExportedProgram",
    "LayerMatch",
    "OperatorOutput",
    "PLAN_FORMAT",
    "Plan",
    "PlanCache",
    "PrecisionPolicy",
    "Quantities",
    "Residual",
    "StagedProgram",
    "bucket_size",
    "compile_operator",
    "compile_residual",
    "conformance_engine",
    "describe_layers",
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "enable_compilation_cache",
    "layers_of",
    "match_dense_chain",
    "register_engine",
    "shard_points",
    "unregister_engine",
]

#: Names promoted from ``omnibias.jax.experimental``; the old module resolves them through ``omnibias.core.deprecation``.
PROMOTED = (
    "ChainMatch",
    "CompiledOperator",
    "CompiledResidual",
    "ConditioningPolicy",
    "ExportedProgram",
    "LayerMatch",
    "OperatorOutput",
    "Quantities",
    "Residual",
    "StagedProgram",
    "bucket_size",
    "compile_operator",
    "compile_residual",
    "describe_layers",
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "enable_compilation_cache",
    "layers_of",
    "match_dense_chain",
    "shard_points",
)

for _name in PROMOTED:
    register(
        Alias(
            f"omnibias.jax.experimental.{_name}",
            f"omnibias.jax.operators.{_name}",
            since="0.5.0",
            warn_from="0.6.0",
            remove_in="0.7.0",
        )
    )


def conformance_engine(**options: Any) -> Callable[[str, Any], np.ndarray]:
    """An engine for :func:`omnibias.core.conformance.run_correctness` that plans every request with ``compile_operator``.

    ``options`` are passed to ``compile_operator`` (``force``, ``mode``, ``precision`` ...). The fixtures are float64: enable
    ``jax_enable_x64`` first.
    """
    import jax
    import jax.numpy as jnp

    names = {"lap": "laplacian", "divergence": "divergence", "logdet": "logdet", "solve": "solve"}

    def engine(quantity: str, case: Any) -> np.ndarray:
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("the conformance fixtures are float64: enable jax_enable_x64")
        act = {"tanh": jnp.tanh, "silu": jax.nn.silu}[case.activation]
        params = [(jnp.asarray(W.T), jnp.asarray(b)) for W, b in case.weights]

        def model(p: Any, x: Any) -> Any:
            a = x
            for W, b in p[:-1]:
                a = act(a @ W + b)
            W, b = p[-1]
            out = a @ W + b
            return out[..., 0] if out.shape[-1] == 1 else out

        x = jnp.asarray(case.x)
        if quantity == "solve":
            rhs = jnp.asarray(case.rhs)
            op = compile_operator(
                model, (params, x, rhs), operator="solve", cache=PlanCache(), **options
            )
            return np.asarray(op(params, x, rhs))
        op = compile_operator(
            model, (params, x), operator=names[quantity], cache=PlanCache(), **options
        )
        return np.asarray(op(params, x))

    return engine

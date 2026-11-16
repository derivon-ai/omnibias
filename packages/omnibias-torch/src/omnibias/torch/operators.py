# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The stable, planned derivative-operator API of the PyTorch back end.

``compile_operator`` here is the planner's front end: it describes a derivative request (Laplacian and powers, divergence,
``log|det(I + J)|``, a Jacobian solve) for a model, ranks the programs that can serve it with a per-device cost model and returns
a callable with ``explain()``; ``compile_residual`` does the same for PDE residuals, their Gram matrix, the damped Gauss-Newton
step and the NTK diagonal. Plans use the versioned format shared with the JAX back end (:data:`PLAN_FORMAT`); precision and
conditioning are policies (``docs/precision.md``); ``docs/stability.md`` states what is stable and how names are retired.

The planner front end was developed as ``omnibias.torch.experimental.compile_planned_operator``. That name stays an alias on the
schedule of :mod:`omnibias.core.deprecation` (silent until 0.6.0, a :class:`DeprecationWarning` from 0.6.0, removed in 0.7.0).
``omnibias.torch.experimental.compile_operator`` is a different object: it keeps compiling an ``OperatorSpec`` with the structural
compiler, and plans only when given an operator name, as this function does. New code should call this one.

:func:`conformance_engine` adapts the front end to :mod:`omnibias.core.conformance`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from omnibias.core.deprecation import Alias, register
from omnibias.core.planner import PLAN_FORMAT, Plan, PlanCache
from omnibias.core.precision import PrecisionPolicy
from omnibias.torch.experimental.conditioning import ConditioningPolicy
from omnibias.torch.experimental.distributed import (
    distributed_residual_gauss_newton,
    distributed_residual_gram,
    distributed_residual_ntk_diagonal,
)
from omnibias.torch.experimental.planned import (
    CompiledOperator,
    OperatorOutput,
    compile_planned_operator,
    register_engine,
    unregister_engine,
)
from omnibias.torch.experimental.residual_compiler import CompiledResidual, compile_residual

#: The planner front end under its stable name.
compile_operator = compile_planned_operator

__all__ = [
    "CompiledOperator",
    "CompiledResidual",
    "ConditioningPolicy",
    "OperatorOutput",
    "PLAN_FORMAT",
    "Plan",
    "PlanCache",
    "PrecisionPolicy",
    "compile_operator",
    "compile_residual",
    "conformance_engine",
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "register_engine",
    "unregister_engine",
]

#: ``experimental`` name -> stable name. ``compile_planned_operator`` becomes ``compile_operator``.
PROMOTED = {
    "CompiledOperator": "CompiledOperator",
    "CompiledResidual": "CompiledResidual",
    "compile_planned_operator": "compile_operator",
    "compile_residual": "compile_residual",
    "distributed_residual_gauss_newton": "distributed_residual_gauss_newton",
    "distributed_residual_gram": "distributed_residual_gram",
    "distributed_residual_ntk_diagonal": "distributed_residual_ntk_diagonal",
}

for _old, _new in PROMOTED.items():
    register(
        Alias(
            f"omnibias.torch.experimental.{_old}",
            f"omnibias.torch.operators.{_new}",
            since="0.5.0",
            warn_from="0.6.0",
            remove_in="0.7.0",
        )
    )


def conformance_engine(**options: Any) -> Callable[[str, Any], np.ndarray]:
    """An engine for :func:`omnibias.core.conformance.run_correctness` that plans every request with ``compile_operator``.

    ``options`` are passed to ``compile_operator`` (``force``, ``mode``, ``precision`` ...).
    """
    import torch

    names = {"lap": "laplacian", "divergence": "divergence", "logdet": "logdet", "solve": "solve"}

    def engine(quantity: str, case: Any) -> np.ndarray:
        n = len(case.weights)
        layers = [
            (
                torch.tensor(W, dtype=torch.float64),
                torch.tensor(b, dtype=torch.float64),
                case.activation if k < n - 1 else None,
            )
            for k, (W, b) in enumerate(case.weights)
        ]
        x = torch.tensor(case.x, dtype=torch.float64)
        if quantity == "solve":
            rhs = torch.tensor(case.rhs, dtype=torch.float64)
            op = compile_operator(layers, (x, rhs), operator="solve", cache=PlanCache(), **options)
            return np.asarray(op(x, rhs).detach().numpy())
        op = compile_operator(layers, x, operator=names[quantity], cache=PlanCache(), **options)
        return np.asarray(op.evaluate(x).value.detach().numpy())

    return engine

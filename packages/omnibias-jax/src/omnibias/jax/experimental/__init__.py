# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in experimental programs.  Names here carry measured, scoped evidence only; see ``docs/performance.md``."""

from omnibias.jax.experimental._chain import Skip, fourier_feature_layer
from omnibias.jax.experimental.chain import as_batch, ridge_layers, separable_axis_layers
from omnibias.jax.experimental.closed_ridge import (
    closed_ridge_operator,
    closed_ridge_residual_loss,
    derivative_closed_system,
)
from omnibias.jax.experimental.compensated import exact_gram, exact_matmul
from omnibias.jax.experimental.demand import (
    QuantitySpec,
    legacy_full_gradient,
    legacy_projected,
    streams_quantities,
    streams_residual,
)
from omnibias.jax.experimental.fused_closed_ridge import (
    fused_closed_ridge_residual_loss,
    fused_closed_ridge_value_and_grad,
)
from omnibias.jax.experimental.grid_gauss_newton import closed_ridge_grid_gauss_newton
from omnibias.jax.experimental.grid_ridge import closed_ridge_grid_loss
from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian
from omnibias.jax.experimental.projected_divergence import projected_deep_divergence
from omnibias.jax.experimental.projected_gram import (
    projected_residual_gauss_newton,
    projected_residual_gram,
    projected_residual_ntk_diagonal,
)
from omnibias.jax.experimental.projected_laplacian import input_factor, projected_deep_laplacian
from omnibias.jax.experimental.projected_logdet import projected_deep_logdet
from omnibias.jax.experimental.projected_polylaplacian import projected_deep_trilaplacian
from omnibias.jax.experimental.residual_gram import (
    deep_residual_gauss_newton,
    deep_residual_gram,
    deep_residual_ntk_diagonal,
    masked_residual,
    pad_block,
)
from omnibias.jax.experimental.residual_woodbury import (
    residual_inverse,
    residual_inverse_newton,
    residual_solve,
    residual_solve_transpose,
)
from omnibias.jax.experimental.separable_grid import (
    SeparableTerm,
    axis_derivative_tables,
    mlp_axis_tables,
    separable_grid_gauss_newton,
    separable_grid_loss,
)
from omnibias.jax.experimental.tanh_ridge import (
    RidgeTerm,
    tanh_ridge_operator,
    tanh_ridge_residual_loss,
)

__all__ = [
    "ChainMatch",
    "CompiledOperator",
    "CompiledResidual",
    "ConditioningPolicy",
    "ExportedProgram",
    "LayerMatch",
    "OperatorOutput",
    "Quantities",
    "QuantitySpec",
    "Residual",
    "RidgeTerm",
    "SeparableTerm",
    "Skip",
    "StagedProgram",
    "as_batch",
    "axis_derivative_tables",
    "bucket_size",
    "closed_ridge_grid_gauss_newton",
    "closed_ridge_grid_loss",
    "closed_ridge_operator",
    "closed_ridge_residual_loss",
    "compile_operator",
    "compile_residual",
    "deep_residual_gauss_newton",
    "deep_residual_gram",
    "deep_residual_ntk_diagonal",
    "derivative_closed_system",
    "describe_layers",
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
    "enable_compilation_cache",
    "exact_gram",
    "exact_matmul",
    "fourier_feature_layer",
    "fused_closed_ridge_residual_loss",
    "fused_closed_ridge_value_and_grad",
    "input_factor",
    "layers_of",
    "legacy_full_gradient",
    "legacy_projected",
    "masked_residual",
    "match_dense_chain",
    "mlp_axis_tables",
    "pad_block",
    "projected_deep_bilaplacian",
    "projected_deep_divergence",
    "projected_deep_laplacian",
    "projected_deep_logdet",
    "projected_deep_trilaplacian",
    "projected_residual_gauss_newton",
    "projected_residual_gram",
    "projected_residual_ntk_diagonal",
    "residual_inverse",
    "residual_inverse_newton",
    "residual_solve",
    "residual_solve_transpose",
    "ridge_layers",
    "separable_axis_layers",
    "separable_grid_gauss_newton",
    "separable_grid_loss",
    "shard_points",
    "streams_quantities",
    "streams_residual",
    "tanh_ridge_operator",
    "tanh_ridge_residual_loss",
]


def __getattr__(name: str) -> object:
    """Names promoted to :mod:`omnibias.jax.operators` still resolve here, on the schedule of ``omnibias.core.deprecation``."""
    import importlib

    from omnibias.core.deprecation import lookup, resolve
    from omnibias.jax import __version__

    importlib.import_module("omnibias.jax.operators")  # registers the aliases
    old = f"{__name__}.{name}"
    if lookup(old) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolve(old, __version__)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

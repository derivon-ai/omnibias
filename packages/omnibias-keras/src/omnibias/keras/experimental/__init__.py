# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Experimental, backend-portable Taylor-jet transformation of Keras models.

Select ``KERAS_BACKEND`` before importing Keras. Packed model inputs and outputs
have shape ``(batch, coefficient, ...)``; coefficients are factorial-normalized.
"""

from omnibias.core.jet_algebra import JetBudget, JetSpec
from omnibias.keras.experimental.model import (
    JetModel,
    extract,
    jet_model,
    register_jet_rule,
    seed,
)
from omnibias.keras.experimental.operator import OperatorModel, OperatorSpec, operator_model
from omnibias.keras.experimental.precision import PrecisionMeanSquaredError

__all__ = [
    "JetBudget", "JetModel", "JetSpec", "OperatorModel", "OperatorSpec", "PrecisionMeanSquaredError",
    "extract", "jet_model", "operator_model", "register_jet_rule", "seed",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing PDE transforms (theory 02-13, gated).

Cole-Hopf / Miura / Bäcklund / Darboux only. Integrability search
(spec 03-11) is not claimed. Exactness is to jet truncation order N.
"""

from __future__ import annotations

from omnibias.core.transforms_pde import (
    LinearizingTransform,
    TransformKind,
    cole_hopf_from_heat_phi,
    cole_hopf_jet,
    cole_hopf_u,
    darboux_dress,
    factorial_jet_multiply,
    factorial_jet_reciprocal,
    miura_v,
    named_cole_hopf,
    permutability,
    verify_cole_hopf_burgers_jet,
    verify_transform,
)

__all__ = [
    "ColeHopfField",
    "LinearizingTransform",
    "MiuraLift",
    "TransformKind",
    "cole_hopf_from_heat_phi",
    "cole_hopf_jet",
    "cole_hopf_u",
    "darboux_dress",
    "factorial_jet_multiply",
    "factorial_jet_reciprocal",
    "miura_v",
    "named_cole_hopf",
    "permutability",
    "verify_cole_hopf_burgers_jet",
    "verify_transform",
]


def __getattr__(name: str) -> object:
    if name in {"ColeHopfField", "MiuraLift"}:
        from omnibias.pinn.transform import torch as _torch

        return getattr(_torch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

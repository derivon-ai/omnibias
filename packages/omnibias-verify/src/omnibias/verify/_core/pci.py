# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Proof-carrying forward (theory 09-24).

A shallow TM (09-05) forward that returns ``(y_mid, box)``. Distinct
from 08-09, which filters a parameter step. The founding bias collapse
(``delta -> 0``) supplies the polynomial part. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

``theorem_prover_verified`` and ``mathlib_verified`` stay false unless
a later PR attaches a genuine ``lake build``. This module never forges
them. Not ImageNet. Not CCF stretch. Not Navier–Stokes regularity.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.core.verified.tm_neuron import (
    TMNeuronSpec,
    contains_grid_and_sample,
    input_tm,
    nest_layers,
    tm_dense,
)

DISCLAIMER = (
    "PCI box on a TM forward; not an 08-09 step filter, not ImageNet, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "is_08_09_step_filter": False,
        "imagenet_claim": False,
        "stretch_claim": False,
        "ns_regularity": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


@dataclass(frozen=True)
class PCIResult:
    y_mid: float
    box_lo: float
    box_hi: float
    vacuous: bool
    theorem_prover_verified: bool = False
    mathlib_verified: bool = False
    is_08_09_step_filter: bool = False

    def __post_init__(self) -> None:
        if self.theorem_prover_verified or self.mathlib_verified:
            raise ValueError(
                "theorem_prover_verified and mathlib_verified require a genuine lake build"
            )
        if self.is_08_09_step_filter:
            raise ValueError("PCI is not the 08-09 step filter")

    @property
    def width(self) -> float:
        return self.box_hi - self.box_lo


def _from_tm(tm: TaylorModel) -> PCIResult:
    box = tm.bound()
    vacuous = box.width >= 1e6 or (box.lo <= -1e6 and box.hi >= 1e6)
    return PCIResult(
        y_mid=box.mid,
        box_lo=box.lo,
        box_hi=box.hi,
        vacuous=vacuous,
    )


def proof_carrying_forward(
    tm: TaylorModel,
    *,
    width_cap: float = 0.1,
) -> PCIResult:
    """Return ``(mid, cert)``. Reject deploy if ``width`` exceeds ``width_cap``."""
    result = _from_tm(tm)
    if result.width > width_cap:
        return PCIResult(
            y_mid=result.y_mid,
            box_lo=result.box_lo,
            box_hi=result.box_hi,
            vacuous=True,
        )
    return result


def worked_sigmoid_box() -> tuple[PCIResult, TaylorModel]:
    spec = TMNeuronSpec(order=1, activation="sigmoid")
    tm = tm_dense(input_tm(0.0, 0.2, spec.order), 1.0, 0.0, spec)
    return proof_carrying_forward(tm, width_cap=0.1), tm


def two_layer_box() -> PCIResult:
    spec = TMNeuronSpec(order=1, activation="sigmoid")
    tm = nest_layers((1.0, 1.0), (0.0, 0.0), spec=spec)
    return proof_carrying_forward(tm, width_cap=0.1)


def pci_sound(result: PCIResult, tm: TaylorModel) -> bool:
    if result.vacuous:
        return False
    return contains_grid_and_sample(tm, "sigmoid", 0.0, 0.2)


__all__ = [
    "DISCLAIMER",
    "PCIResult",
    "honesty_payload",
    "pci_sound",
    "proof_carrying_forward",
    "two_layer_box",
    "worked_sigmoid_box",
]

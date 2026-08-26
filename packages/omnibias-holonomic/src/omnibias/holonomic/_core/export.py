# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Net-to-annihilator export (theory 09-26).

Emit an Ore / jet description plus an optional *finite rational*
Lean obligation. Infinite analytic claims stay out of Lean.

A trained holonomic layer may have used founding bias collapse
(``delta -> 0``) for its data jet. The export itself is not that
collapse. Temperature collapse (``beta -> inf``, feasibility) does
not appear. do not conflate the two.

``theorem_prover_verified`` and ``mathlib_verified`` stay false
until a genuine ``lake build``. Not a continuum PDE. Not NS, YM,
RH, or P vs NP. Not CCF stretch. This module imports neither torch
nor jax.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from omnibias.holonomic._core.layer import d2_plus_1, d_minus_1
from omnibias.holonomic._core.ore import OrePolynomial, diff_algebra, shift_algebra
from omnibias.holonomic._core.rational_poly import to_poly

DISCLAIMER = (
    "Ore annihilator export; finite rational Lean only; verified flags "
    "future-earned; not a continuum PDE, not CCF stretch"
)

def source_imports_no_backend() -> bool:
    """G4: the export module has no backend imports."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    banned = {"torch", "jax"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned:
                    return False
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in banned:
                return False
    return True


FORBIDDEN_LEAN = (
    "sorry",
    "continuum",
    "navier",
    "yang-mills",
    "riemann",
    "p=np",
    "p vs np",
    "ccf stretch",
    "global regularity",
)


def honesty_payload() -> dict[str, bool]:
    return {
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "continuum_claim": False,
        "navier_stokes_proof_claim": False,
        "stretch_claim": False,
    }


@dataclass(frozen=True)
class AnnihilatorExport:
    ore: dict[str, Any]
    lean_path: str | None
    theorem_prover_verified: bool = False
    mathlib_verified: bool = False

    def __post_init__(self) -> None:
        if self.theorem_prover_verified or self.mathlib_verified:
            raise ValueError(
                "verified flags are future-earned; do not assert them without "
                "a genuine lake build"
            )


def ore_to_dict(op: OrePolynomial) -> dict[str, Any]:
    """JSON-ready Ore payload (rational coefficient tuples)."""
    coeffs = [
        [(int(c.numerator), int(c.denominator)) for c in poly] for poly in op.coeffs
    ]
    constants: list[list[int]] = []
    for poly in op.coeffs:
        if len(poly) == 1:
            constants.append([int(poly[0].numerator), int(poly[0].denominator)])
        elif len(poly) == 0:
            constants.append([0, 1])
        else:
            constants.append([])
    return {
        "algebra": op.algebra.name,
        "order": op.order,
        "coeffs": coeffs,
        "constant_coeffs": constants,
    }


def ore_from_dict(payload: dict[str, Any]) -> OrePolynomial:
    name = str(payload["algebra"])
    if name == "differential":
        algebra = diff_algebra()
    elif name == "shift":
        algebra = shift_algebra()
    else:
        raise ValueError(f"unknown Ore algebra {name!r}")
    coeffs = []
    for poly in payload["coeffs"]:
        coeffs.append(to_poly([Fraction(int(n), int(d)) for n, d in poly]))
    return algebra.operator(coeffs)


def assert_lean_scope(text: str) -> str:
    """Raise if the Lean text smuggles a continuum / sorry claim."""
    low = text.lower()
    for word in FORBIDDEN_LEAN:
        if word in low:
            raise ValueError(f"refusing to emit Lean with forbidden claim {word!r}")
    return text


def render_finite_lean(op: OrePolynomial) -> str:
    """Finite rational readout only. No sorry. No continuum sentence."""
    lead = op.coeffs[op.order] if op.order >= 0 and op.order < len(op.coeffs) else ()
    lead0 = lead[0] if lead else Fraction(0)
    text = (
        f"-- finite rational: order = {op.order}\n"
        f"-- leading constant numerator = {int(lead0.numerator)}\n"
        f"-- leading constant denominator = {int(lead0.denominator)}\n"
        "-- theorem_prover_verified stays false until lake build\n"
    )
    return assert_lean_scope(text)


def export_annihilator(
    layer: OrePolynomial | None,
    *,
    emit_lean: bool = False,
    lean_claim: str | None = None,
) -> AnnihilatorExport:
    """Must not set verified flags without a genuine lake build."""
    op = d_minus_1() if layer is None else layer
    if lean_claim is not None:
        assert_lean_scope(lean_claim)
    lean = render_finite_lean(op) if emit_lean else None
    if lean is not None and lean_claim is not None:
        lean = assert_lean_scope(lean + lean_claim)
    return AnnihilatorExport(
        ore=ore_to_dict(op),
        lean_path=lean,
        theorem_prover_verified=False,
        mathlib_verified=False,
    )


def worked_example() -> dict[str, object]:
    """G1: ``D-1`` serialize/parse identity."""
    op = d_minus_1()
    payload = ore_to_dict(op)
    back = ore_from_dict(payload)
    return {
        "order": back.order,
        "match": ore_to_dict(back) == payload,
        "constant_coeffs": payload["constant_coeffs"],
        "algebra": payload["algebra"],
    }


def export_skill() -> dict[str, object]:
    """G2/G3: flags stay false; continuum Lean is refused."""
    exported = export_annihilator(d_minus_1(), emit_lean=True)
    sin_op = d2_plus_1()
    sin_ex = export_annihilator(sin_op)
    raised = False
    try:
        export_annihilator(d_minus_1(), emit_lean=True, lean_claim="continuum PDE theorem")
    except ValueError:
        raised = True
    return {
        "flags_false": (
            exported.theorem_prover_verified is False
            and exported.mathlib_verified is False
        ),
        "sin_order": sin_ex.ore["order"],
        "continuum_raised": raised,
        "g2_earned": (
            exported.theorem_prover_verified is False
            and exported.mathlib_verified is False
            and raised
        ),
    }


__all__ = [
    "AnnihilatorExport",
    "DISCLAIMER",
    "FORBIDDEN_LEAN",
    "assert_lean_scope",
    "export_annihilator",
    "export_skill",
    "honesty_payload",
    "ore_from_dict",
    "ore_to_dict",
    "render_finite_lean",
    "source_imports_no_backend",
    "worked_example",
]

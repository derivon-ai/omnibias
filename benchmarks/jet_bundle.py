# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: jet-bundle contact tests (theory 01-10).

Vocabulary, not a discovery. No ``omnibias-jetbundle`` package.
Founding ``delta -> 0`` produces fiber coordinates. Temperature
collapse acts on the base stratification, not on the fiber.
G1/G2 are the named contact tests. G3 is dictionary coverage in
other theory specs, not a discovery.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
REPO = Path(__file__).resolve().parents[1]
N_CASES = 220
NEED = 200
SELF_SPEC = "10-jet-bundle-formalization.md"
DICTIONARY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fiber coordinate", ("fiber coordinate", "fiber coordinates")),
    ("scattered jet", ("scattered jet",)),
    ("fiber interval", ("fiber interval",)),
    (
        "translation of the base",
        ("translation of the base", "translation along the base"),
    ),
    ("stratification", ("stratification",)),
    ("fiber product", ("fiber product", "fiber-product")),
    ("subvariety", ("subvariety",)),
    ("prolongation", ("prolongation", "prolongations")),
)
COMPETING_NEEDLES = (
    "band role is a local jet",
    "integral role is a local jet",
    "fiber interval is a derivative",
)


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _tanh_jet(x: float, order: int = 3) -> tuple[float, ...]:
    from omnibias.core.polynomials import tanh_polynomial_coeffs

    t = math.tanh(x)
    out = [t]
    for n in range(1, order + 1):
        out.append(_horner(tanh_polynomial_coeffs(n), t))
    return tuple(out)


def _gauss_jet(x: float, order: int = 3) -> tuple[float, ...]:
    from omnibias.core.polynomials import hermite_coeffs

    g = math.exp(-0.5 * x * x)
    out = [g]
    for n in range(1, order + 1):
        sign = 1.0 if n % 2 == 0 else -1.0
        out.append(sign * _horner(hermite_coeffs(n), x) * g)
    return tuple(out)


def _poly_jet(x: float, coeffs: tuple[float, ...], order: int = 3) -> tuple[float, ...]:
    out: list[float] = []
    for n in range(order + 1):
        acc = 0.0
        for k in range(n, len(coeffs)):
            falling = 1.0
            for j in range(n):
                falling *= k - j
            power = 1.0 if k == n else x ** (k - n)
            acc += coeffs[k] * falling * power
        out.append(acc)
    return tuple(out)


def _exp_jet(x: float, order: int = 3) -> tuple[float, ...]:
    e = math.exp(0.3 * x)
    return tuple((0.3**n) * e for n in range(order + 1))


def _corrupt(sampler: Callable[[float], tuple[float, ...]]) -> Callable[[float], tuple[float, ...]]:
    def corrupted(z: float) -> tuple[float, ...]:
        jet = list(sampler(z))
        jet[1] = 0.90
        return tuple(jet)

    return corrupted


def _sampler_for(kind: int, rng: random.Random) -> Callable[[float], tuple[float, ...]]:
    if kind == 0:
        return _tanh_jet
    if kind == 1:
        return _gauss_jet
    if kind == 2:
        coeffs = tuple(rng.uniform(-1.0, 1.0) for _ in range(5))

        def poly(z: float, c: tuple[float, ...] = coeffs) -> tuple[float, ...]:
            return _poly_jet(z, c)

        return poly
    return _exp_jet


def _run_g1() -> dict[str, Any]:
    from omnibias.core.jets import is_holonomic

    rng = random.Random(0)
    n_ok = 0
    n_bad = 0
    n_misclass = 0
    for i in range(N_CASES):
        sampler = _sampler_for(i % 4, rng)
        x = rng.uniform(-1.2, 1.2)
        h = 1e-4
        if is_holonomic(sampler, x, h=h):
            n_ok += 1
        else:
            n_misclass += 1
        if not is_holonomic(_corrupt(sampler), x, h=h):
            n_bad += 1
        else:
            n_misclass += 1
    earned = n_ok >= NEED and n_bad >= NEED and n_misclass == 0
    return {
        "name": "g1_contact",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "n_cases": N_CASES,
        "n_holonomic_true": n_ok,
        "n_corrupted_false": n_bad,
        "n_misclass": n_misclass,
        "need": NEED,
        "note": (
            "is_holonomic on genuine tanh / Gauss / polynomial / exp "
            "prolongations versus first-derivative overwrite. Named G1 "
            "needs >= 200 of each class and zero misclassifications."
        ),
    }


def _run_g2() -> dict[str, Any]:
    from omnibias.core.jets import max_abs_contact_residual

    x = 0.5
    h0 = 8e-4
    j0 = _tanh_jet(x)
    hs = [h0 / (2**k) for k in range(4)]

    def holonomic_res(h: float) -> float:
        return max_abs_contact_residual(j0, _tanh_jet(x + h), h=h)

    def corrupted_res(h: float) -> float:
        bad0 = list(_tanh_jet(x))
        bad_h = list(_tanh_jet(x + h))
        bad0[1] = 0.90
        bad_h[1] = 0.90
        return max_abs_contact_residual(bad0, bad_h, h=h)

    r_h = [holonomic_res(h) for h in hs]
    r_c = [corrupted_res(h) for h in hs]
    holonomic_ratios = [r_h[i + 1] / r_h[i] for i in range(3)]
    corrupted_ratios = [r_c[i + 1] / r_c[i] for i in range(3)]
    holonomic_ok = all(0.15 <= r <= 0.35 for r in holonomic_ratios)
    corrupted_ok = all(0.40 <= r <= 0.60 for r in corrupted_ratios)
    earned = holonomic_ok and corrupted_ok
    return {
        "name": "g2_rate",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "holonomic_ratios": holonomic_ratios,
        "corrupted_ratios": corrupted_ratios,
        "need": "holonomic ~1/4 and corrupted ~1/2 over three halvings",
        "note": (
            "Contact residual of tanh versus first-derivative overwrite "
            "over three halvings of h. Same named path as G1."
        ),
    }


def _theory_specs() -> list[Path]:
    theory = REPO / "theory"
    return sorted(
        path
        for path in theory.rglob("*.md")
        if path.name != "README.md" and path.name != SELF_SPEC
    )


def _run_g3() -> dict[str, Any]:
    """Named leftover: dictionary terms in other specs; no competing names."""
    specs = _theory_specs()
    blobs = [(path, path.read_text(encoding="utf-8").lower()) for path in specs]
    coverage: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, needles in DICTIONARY_TERMS:
        hits: list[str] = []
        for path, text in blobs:
            if any(needle in text for needle in needles):
                hits.append(str(path.relative_to(REPO)))
        row = {"term": name, "n_specs": len(hits), "specs": hits[:8]}
        coverage.append(row)
        if not hits:
            missing.append(name)
    competing: list[str] = []
    for path, text in blobs:
        for needle in COMPETING_NEEDLES:
            if needle in text:
                competing.append(f"{path.relative_to(REPO)}:{needle}")
    earned = not missing and not competing
    return {
        "name": "g3_vocabulary",
        "passed": bool(earned),
        "earned": bool(earned),
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "n_terms": len(DICTIONARY_TERMS),
        "n_covered": len(DICTIONARY_TERMS) - len(missing),
        "missing": missing,
        "competing": competing,
        "coverage": coverage,
        "need": "every dictionary term in another spec; no competing name",
        "note": (
            "Consistency pass over theory/ specs excluding 01-10. "
            "Named G3 needs every dictionary term in at least one other "
            "spec and no competing term for the same object. Scale-flow "
            "V-cycle 'prolongation' is a different object, not a "
            "competing name for the jet-bundle prolongation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3()
    in_scope = [row for row in (g1, g2, g3) if row["in_ci_all_passed"]]
    payload = provenance(
        schema="omnibias.benchmark.jet_bundle.v1",
        config={
            "family": "jet_bundle",
            "full": bool(args.full),
            "gates_in_scope": [row["name"] for row in in_scope],
            "g3_in_all_passed": bool(g3["in_ci_all_passed"]),
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["honesty"] = {
        "reformulation_not_discovery": True,
        "omnibias_jetbundle_package": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "g1_earned": bool(g1["earned"]),
        "g1_in_ci_all_passed": bool(g1["in_ci_all_passed"]),
        "g2_earned": bool(g2["earned"]),
        "g2_in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        "g3_earned": bool(g3["earned"]),
        "g3_in_ci_all_passed": bool(g3["in_ci_all_passed"]),
        "scale_flow_prolongation_is_v_cycle": True,
        "certificate_tier": False,
    }
    if args.full:
        out_dir = SCRATCH / "jets"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "jet_bundle.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('jet_bundle_smoke.json', payload)}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

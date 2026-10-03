#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Reproduce finite Hilbert XVI program checks and record exact source hashes.

This runs actual event/zero/topology checks. It does not set a parent-problem
proof flag. Optional Lean builds check the named source modules, whose analytic
premises are explicitly limited. Normal and optimized Python use the same
non-assertion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, is_dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.core.verified.asymptotic_jet import (
    fixed_product_jet,
    power_compensator,
    signed_root_primitive,
    verify_fixed_product_derivative,
    weighted_scale_derivative,
)
from omnibias.core.verified.interval import Interval
from omnibias.difference._core.singularity import certified_singularity_annulus
from omnibias.dynamics.cyclicity import (
    ConfluentExponentialPolynomial,
    certify_exponential_cyclicity,
    certify_planar_return_cyclicity,
    certify_polynomial_cyclicity,
    verify_cyclicity_certificate,
    verify_exponential_cyclicity,
    verify_return_cyclicity,
)
from omnibias.dynamics.hilbert16 import (
    CyclicityLeaf,
    CyclicitySplit,
    certify_polynomial_cover,
    verify_polynomial_cover,
)
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
)
from omnibias.geometry.algebraic import (
    HomogeneousPlaneCurve,
    RationalRectangle,
    RectangularAnnulus,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
)
from omnibias.geometry.algebraic_surfaces import (
    certify_separable_quartic_surface,
    replay_separable_quartic_surface,
    separable_quartic_polynomial,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_MODULES = (
    "Hilbert16Rolle", "Hilbert16Parabola", "Hilbert16Resonance",
    "Hilbert16ReturnMap", "Hilbert16Scale", "Hilbert16ChiScale",
    "Hilbert16SaddleNode", "Hilbert16TwoBlowup", "Hilbert16ScaleDichotomy",
    "Hilbert16WeightedSection", "Hilbert16QuasihomogeneousDichotomy",
    "Hilbert16LNFormatBarrier", "Hilbert16AbelianReturnTransfer",
    "Hilbert16LNCell", "Hilbert16EntryExit", "Hilbert16StageB", "Hilbert16StageA", "Hilbert16ChiB", "Hilbert16DxELeading", "Hilbert16DxEUnif", "Hilbert16StageC", "Hilbert16StageCExit", "Hilbert16StageCTh", "Hilbert16StageCGap", "Hilbert16StageCEnv", "Hilbert16StageCIf", "Hilbert16StageCInt", "Hilbert16StageCLo", "Hilbert16StageCK", "Hilbert16StageCBoot", "Hilbert16StageCRect", "Hilbert16StageCHit", "Hilbert16StageCSec", "Hilbert16StageCOneshot", "Hilbert16StageCOneshotEps", "Hilbert16StageCEpsSpan", "Hilbert16StageCOrigin", "Hilbert16StageCOriginSpan", "Hilbert16StageCOriginIface", "Hilbert16StageCOriginNear", "Hilbert16StageCOriginX32", "Hilbert16StageCCompare", "Hilbert16StageCUniform", "Hilbert16StageCInterface", "Hilbert16SepSpre", "Hilbert16DxEOff", "Hilbert16DxERay", "Hilbert16DxENear", "Hilbert16DxEOpen", "Hilbert16FoldLeading",
    "Hilbert16ShrinkingRoot", "Hilbert16CanonicalZeta", "Hilbert16FoldZeta",
    "Hilbert16PhysicalC2", "Hilbert16ZXGap", "Hilbert16ZVBound", "Hilbert16ZSlowV", "Hilbert16FoldZX", "Hilbert16OutgoingCorridor",
    "Hilbert16PostCorridor", "Hilbert16HeightEnvelope",
    "Hilbert16QRatioC2", "Hilbert16KZetaRemainder",
    "Hilbert16KillZeta", "Hilbert16CancelledN", "Hilbert16HeightMix",
    "Hilbert16OrbitTh", "Hilbert16ThIntegral", "Hilbert16VhOrbit",
    "Hilbert16EOutSection", "Hilbert16EOutEps", "Hilbert16EOutSpeed",
    "Hilbert16ESigmaSpeed", "Hilbert16ESigmaIn", "Hilbert16ESigmaHit",
    "Hilbert16ESigmaFrom0", "Hilbert16ESigmaUnif", "Hilbert16ESigmaWall",
    "Hilbert16ESigmaBox", "Hilbert16ESigmaSpan", "Hilbert16ESigmaPack",
    "Hilbert16ESigmaEps", "Hilbert16ESigmaOneshot",
    "Hilbert16ESigmaOneshotEps", "Hilbert16ESigmaEpsSpan",
    "Hilbert16ESigmaEpsLo",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    return value


def _square(a: int, b: int, radius: Fraction) -> RationalRectangle:
    return RationalRectangle(a - radius, a + radius, b - radius, b + radius)


def run(*, lean: bool = False) -> dict[str, object]:
    checks: dict[str, object] = {}
    radius = certified_singularity_annulus((1.0, 1.0), tail_bound=Interval.point(0), tail_ratio=0.5)
    require(radius.lo == radius.hi == float("inf"), "entire-polynomial radius regression")
    checks["entire_polynomial_radius"] = radius

    omega, u = P.variable(2, 0), P.variable(2, 1)
    product = omega * u
    exact = weighted_scale_derivative(product, order=2)
    require(not exact.terms, "fixed-product acceleration")
    require(not verify_fixed_product_derivative(product, -2 * product, order=2),
            "missing-acceleration rejection")
    checks["fixed_product_jet"] = fixed_product_jet(product, Fraction(1, 16), Fraction(1, 32))
    checks["confluent_diagonal"] = power_compensator(0, 0, 2)
    checks["signed_root_across_zero"] = signed_root_primitive(Interval(-0.01, 0.01), 1)

    h, a = P.variable(2, 0), P.variable(2, 1)
    polynomial = a * (h**3 - h)
    center = certify_polynomial_cyclicity(polynomial, ((-2, 2), (0, 0)))
    require(center.identity is True and center.exact_count == 0, "identity fiber isolation")
    tree = CyclicitySplit(
        1, 0,
        CyclicityLeaf(certify_polynomial_cyclicity(polynomial, ((-2, 2), (-1, 0)))),
        CyclicityLeaf(certify_polynomial_cyclicity(polynomial, ((-2, 2), (0, 1)))),
    )
    cover = certify_polynomial_cover(polynomial, ((-2, 2), (-1, 1)), tree)
    require(cover.upper_bound == 3 and verify_polynomial_cover(polynomial, cover), "finite cover")
    require(not verify_cyclicity_certificate(polynomial + 1, center), "source substitution")
    checks["polynomial_center"] = center
    checks["finite_polynomial_cover"] = cover
    exponential = ConfluentExponentialPolynomial([(0, (1,)), (1, (-2,)), (2, (1,))])
    exp_count = certify_exponential_cyclicity(exponential)
    require(exp_count.upper_bound == 2 and verify_exponential_cyclicity(exponential, exp_count),
            "exponential derivative-division")
    checks["exponential_count"] = exp_count

    x, y, initial_h = P.variable(4, 0), P.variable(4, 1), P.variable(1, 0)
    radial = Fraction(1, 10) * (P.constant(4, 1) - x*x - y*y)
    flow = PolynomialFlow((y + radial*x, -x + radial*y), 1)
    request = StoppedEventRequest(
        flow=flow, initial=(P.constant(1, 0), initial_h),
        parameters=(Interval(0.99999, 1.00001),),
        target=PolynomialEvent(x, direction=1, guard=y),
        step=0.03125, max_steps=210, order=8,
    )
    event = certify_stopped_event(request)
    require(event.certified, event.reason)
    count = certify_planar_return_cyclicity(event)
    require(count.upper_bound == 1, "actual Hopf first-return Rolle count")
    require(verify_return_cyclicity(count, expected_source_fingerprint=request.fingerprint),
            "physical source replay")
    require(not verify_return_cyclicity(replace(count, upper_bound=0),
                                       expected_source_fingerprint=request.fingerprint),
            "physical zero-count tampering")
    require(not (x*flow.components[0] + y*flow.components[1] - radial*(x*x + y*y)).terms,
            "exact unit-circle invariant identity")
    checks["hopf_return"] = {
        "request": request, "source_fingerprint": request.fingerprint,
        "time_bracket": event.time_bracket, "upper_bound": count.upper_bound,
        "displacement_range": count.displacement_range,
        "first_derivative": count.first_derivative,
        "scope": "first eligible positive return of this exact field on this initial-height interval",
    }

    xx, yy, zz = (P.variable(3, i) for i in range(3))
    px, py = (xx**2 - zz**2)*(xx**2 - 9*zz**2), (yy**2 - zz**2)*(yy**2 - 9*zz**2)
    curve = HomogeneousPlaneCurve(px**2 + py**2 - Fraction(1, 16)*zz**8)
    smooth = find_smoothness_witness(curve, max_multiplier_degree=12)
    require(smooth is not None, "octic complex nonsingularity witness")
    if smooth is None:  # narrowing; require above also runs under optimized Python
        raise RuntimeError("missing smoothness witness")
    annuli = [RectangularAnnulus(_square(i, j, Fraction(1, 1024)), _square(i, j, Fraction(1, 32)))
              for i in (-3, -1, 1, 3) for j in (-3, -1, 1, 3)]
    curve_cert = certify_curve(curve, smooth, annuli)
    require(curve_cert.component_lower_bound == 16 and replay_curve_certificate(curve, curve_cert),
            "sixteen separated octic ovals")
    require(not curve_cert.complete_real_scheme, "open maximal target not inferred")
    checks["octic"] = {"polynomial": curve.polynomial.to_payload(), "certificate": curve_cert}
    surface = separable_quartic_polynomial(Fraction(1, 16))
    surface_cert = certify_separable_quartic_surface(surface)
    require(replay_separable_quartic_surface(surface, surface_cert), "quartic surface replay")
    checks["surface"] = {"polynomial": surface.to_payload(), "certificate": surface_cert}

    formal: list[dict[str, object]] = []
    if lean:
        project = ROOT / "formal" / "omnibias-analytic"
        for module in FORMAL_MODULES:
            build = subprocess.run(["lake", "build", f"OmnibiasAnalytic.Dynamics.{module}"],
                                   cwd=project, capture_output=True, text=True, timeout=180)
            require(build.returncode == 0, f"Lean build {module}: {build.stdout}\n{build.stderr}")
            source = project / "OmnibiasAnalytic" / "Dynamics" / f"{module}.lean"
            names = re.findall(r"^theorem\s+(\w+)", source.read_text(), flags=re.MULTILINE)
            commands = f"import OmnibiasAnalytic.Dynamics.{module}\n" + "\n".join(
                f"#print axioms OmnibiasAnalytic.Dynamics.{module}.{name}" for name in names
            )
            audit = subprocess.run(["lake", "env", "lean", "--stdin"], cwd=project,
                                   input=commands, capture_output=True, text=True, timeout=60)
            require(audit.returncode == 0, f"Lean axiom audit {module}: {audit.stdout}\n{audit.stderr}")
            used = re.findall(r"\[([^]]*)\]", audit.stdout)
            require(len(used) == len(names), f"every theorem audited in {module}")
            allowed = {"propext", "Classical.choice", "Quot.sound"}
            for entry in used:
                require(set(filter(None, (a.strip() for a in entry.split(",")))) <= allowed,
                        f"unexpected theorem axiom in {module}")
            formal.append({"module": module, "build_passed": True,
                           "theorems": names, "axiom_audit": audit.stdout.strip()})

    paths = [Path(__file__).relative_to(ROOT)]
    paths += [p.relative_to(ROOT) for folder, pattern in (
        (ROOT / "packages/omnibias-dynamics", "HILBERT16*.md"),
        (ROOT / "formal/omnibias-analytic/OmnibiasAnalytic/Dynamics", "Hilbert16*.lean"),
    ) for p in folder.glob(pattern)]
    paths += [Path(p) for p in (
        "packages/omnibias-dynamics/src/omnibias/dynamics/return_maps.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/cyclicity.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/hilbert16.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/hilbert16_identities.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/chart_cells.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/scale_dichotomy.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/entry_exit_leading.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/fold_leading.py",
        "packages/omnibias-dynamics/src/omnibias/dynamics/shrinking_root_leading.py",
        "benchmarks/hilbert16_two_blowup.py",
        "benchmarks/hilbert16_next_atlas.py",
        "benchmarks/hilbert16_entry_exit_leading.py",
        "benchmarks/hilbert16_fold_leading.py",
        "benchmarks/hilbert16_shrinking_root_leading.py",
        "packages/omnibias-core/src/omnibias/core/verified/asymptotic_jet.py",
        "packages/omnibias-core/src/omnibias/core/verified/ode.py",
        "packages/omnibias-core/src/omnibias/core/verified/interval.py",
        "packages/omnibias-core/src/omnibias/core/verified/transcend.py",
        "packages/omnibias-core/src/omnibias/core/realization/polynomial.py",
        "packages/omnibias-core/src/omnibias/core/realization/algebraic.py",
        "packages/omnibias-difference/src/omnibias/difference/_core/singularity.py",
        "packages/omnibias-geometry/src/omnibias/geometry/algebraic.py",
        "packages/omnibias-geometry/src/omnibias/geometry/algebraic_surfaces.py",
        "packages/omnibias-dynamics/hilbert16-literature.json",
    )]
    hashes = {p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(set(paths))}
    return {"schema": "hilbert16-program-evidence-v1", "checks": _safe(checks),
            "formal": formal, "source_sha256": hashes,
            "remaining": ["uniform singular passage", "complete I2/I4 graphic capture",
                          "all quadratic cases", "degree-only bound", "maximal octic target",
                          "curve and surface embedded classification"],
            "scope": "Finite executable checks and named conditional analytic lemmas; full Hilbert XVI open."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts")) / "hilbert16/program.json")
    args = parser.parse_args()
    report = run(lean=args.lean)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": len(cast(dict[str, object], report["checks"])),
                      "formal_modules": len(cast(list[object], report["formal"])),
                      "status": "finite checks passed; full Hilbert XVI remains open"}))


if __name__ == "__main__":
    main()

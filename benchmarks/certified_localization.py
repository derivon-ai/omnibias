# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: certified scan localization (theory 03-08).

Smoke earns G1 (soundness on a dense grid and a random sample;
``--full`` raises the random sample to 10 000), G2 (no false
uniqueness on two peaks), G3 (Krawczyk 100x tighter than B&B),
G4 (quadratic contraction), G5 (flat / inflected is
Inconclusive), and G6 (seal + tamper). The scan template is
founding bias collapse, not temperature collapse. Scope is
``local_box``. Not ``theorem_prover_verified``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.verify.localization import (
    Inconclusive,
    ScanResponse,
    branch_and_bound_peak,
    certify_peak,
    honesty_payload,
    seal,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1(*, n_random: int) -> dict[str, Any]:
    grid = np.linspace(-0.8, 0.8, 41)
    rng = np.random.default_rng(1)
    random = rng.uniform(-0.8, 0.8, size=int(n_random))
    alphas = rng.uniform(3.0, 8.0, size=int(n_random))
    violations = 0
    n = 0
    for tau_star in grid:
        resp = ScanResponse.sech2_peak(float(tau_star), alpha=5.0)
        box = Interval(float(tau_star) - 0.10, float(tau_star) + 0.10)
        got = certify_peak(resp, box=box, max_iter=6)
        n += 1
        if isinstance(got, Inconclusive) or not got.offset_enclosure.contains(float(tau_star)):
            violations += 1
    for tau_star, alpha in zip(random, alphas, strict=True):
        half = 0.50 / float(alpha)
        resp = ScanResponse.sech2_peak(float(tau_star), alpha=float(alpha))
        box = Interval(float(tau_star) - half, float(tau_star) + half)
        got = certify_peak(resp, box=box, max_iter=6)
        n += 1
        if isinstance(got, Inconclusive) or not got.offset_enclosure.contains(float(tau_star)):
            violations += 1
    return {
        "name": "g1_soundness",
        "passed": violations == 0,
        "n": n,
        "violations": violations,
        "detail": "grid + random; enclosure contains the analytic peak",
    }


def _run_g2() -> dict[str, Any]:
    two = ScanResponse.two_peaks(-0.3, 0.3, alpha=6.0)
    got = certify_peak(two, box=Interval(-0.6, 0.6))
    one = certify_peak(ScanResponse.sech2_peak(-0.3, alpha=5.0), box=Interval(-0.4, -0.2))
    ok = isinstance(got, Inconclusive) and not isinstance(one, Inconclusive)
    return {"name": "g2_uniqueness", "passed": ok, "two_peak": getattr(got, "reason", None), "detail": "never a false unique peak"}


def _run_g3() -> dict[str, Any]:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    box = Interval(-0.4, -0.2)
    cert = certify_peak(resp, box=box, max_iter=4)
    bab = branch_and_bound_peak(resp, box, n_iters=4)
    if isinstance(cert, Inconclusive):
        return {"name": "g3_tightness", "passed": False, "detail": cert.reason}
    ratio = bab.width / max(cert.offset_enclosure.width, 1e-18)
    return {"name": "g3_tightness", "passed": ratio >= 100.0 and cert.route == "krawczyk", "ratio": ratio, "k_width": cert.offset_enclosure.width, "bab_width": bab.width}


def _run_g4() -> dict[str, Any]:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    cert = certify_peak(resp, box=Interval(-0.4, -0.2), max_iter=4)
    if isinstance(cert, Inconclusive) or len(cert.widths) < 4:
        return {"name": "g4_quadratic", "passed": False}
    widths = cert.widths
    ok = all(nxt <= 40.0 * prev * prev for prev, nxt in zip(widths[1:-1], widths[2:], strict=True))
    return {"name": "g4_quadratic", "passed": ok, "widths": list(widths), "detail": "width squares over three contractions"}


def _run_g5() -> dict[str, Any]:
    box = Interval(-0.2, 0.2)
    flat = certify_peak(ScanResponse.constant(1.0), box=box)
    inflect = certify_peak(ScanResponse.flat_max(), box=box)
    ok = isinstance(flat, Inconclusive) and isinstance(inflect, Inconclusive)
    return {"name": "g5_degenerate", "passed": ok, "flat": getattr(flat, "reason", None), "inflect": getattr(inflect, "reason", None)}


def _run_g6() -> dict[str, Any]:
    cert = certify_peak(ScanResponse.sech2_peak(-0.3, alpha=5.0), box=Interval(-0.4, -0.2))
    if isinstance(cert, Inconclusive):
        return {"name": "g6_seal", "passed": False}
    sealed = seal(cert)
    tampered = dict(sealed)
    payload = dict(tampered["payload"])
    payload["offset"] = [0.0, 1.0]
    tampered["payload"] = payload
    ok = verify_certificate_digest(sealed) and not verify_certificate_digest(tampered)
    ok = ok and honesty_payload()["theorem_prover_verified"] is False
    return {"name": "g6_seal", "passed": ok, "detail": "digest + tamper; no theorem_prover_verified"}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "certified_localization.json" if full else "certified_localization_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(n_random=10_000 if full else 256), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.certified_localization.v1",
            config={"family": "certified_localization", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "localization"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

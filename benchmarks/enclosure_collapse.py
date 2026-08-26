# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-1: Enclosure Collapse and the Width Law (theory 01-14).

Smoke earns G1 (Width Law ratio), G2 (Lemma Floor), G3 (Helmholtz
decrease), G4 (six callables), G5 (Inconclusive / empty ball), and
G6 (homes / honesty). Enclosure Collapse is ``width -> 0`` of a
sound enclosure: a point plus a proof. Not bias collapse and not
temperature collapse.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.remainder_train import exp_jet
from omnibias.core.verified.enclosure_collapse import (
    measured_mean_value_width,
    rounding_floor_witness,
    two_ulp,
    width_law,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    lohner_flow_jet,
    named_tower_fields,
    tower_field,
    tower_jacobian,
)
from omnibias.core.verified.kantorovich import polynomial_sqrt2_maps
from omnibias.core.verified.pde_certificate import helmholtz
from omnibias.verify import MLPArchitecture
from omnibias.verify.enclosure_collapse import (
    squeeze,
    squeeze_existence,
    squeeze_flow,
    squeeze_identifiability,
    squeeze_peak,
    squeeze_remainder,
    squeeze_residual,
)
from omnibias.verify.localization import Inconclusive, ScanResponse

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    law = width_law("sigmoid", 0, 0.5, kind="mean_value")
    enc = measured_mean_value_width("sigmoid", 0, 0.5, 1e-4)
    ratio = enc.width / (2.0 * 1e-4)
    err = abs(ratio - law.predicted_leading.mid)
    return {"name": "g1_width_law", "passed": err < 5e-3, "error": err}


def _run_g2() -> dict[str, Any]:
    witness = rounding_floor_witness()
    ok = witness.width >= two_ulp(0.0) and witness.lo < witness.hi
    return {"name": "g2_floor", "passed": ok, "width": witness.width}


def _run_g3() -> dict[str, Any]:
    w1, w2 = 1.3, -0.7
    layers = [([[w1, w2]], [0.0], "cos")]
    pde = helmholtz(2, math.hypot(w1, w2))
    widths = [
        squeeze_residual(layers, [(0.0, 1.0), (0.0, 1.0)], pde, splits=s).width
        for s in (1, 4, 16, 64)
    ]
    ok = widths[0] is not None and widths[-1] is not None and widths[0] > widths[-1]
    return {"name": "g3_residual", "passed": ok, "widths": widths}


def _run_g4() -> dict[str, Any]:
    drivers = {
        "peak": squeeze_peak,
        "residual": squeeze_residual,
        "identifiability": squeeze_identifiability,
        "existence": squeeze_existence,
        "remainder": squeeze_remainder,
        "flow": squeeze_flow,
    }
    ok = all(callable(fn) for fn in drivers.values())
    report = squeeze("peak", response=ScanResponse.sech2_peak(-0.3, alpha=5.0), box=Interval(-0.4, -0.2))
    return {"name": "g4_callables", "passed": ok and report.kind == "peak", "status": report.status}


def _run_g5() -> dict[str, Any]:
    flat = squeeze_peak(ScanResponse.flat_max(), box=Interval(-0.5, 0.5))
    empty = squeeze_existence(y0=1.0, z0=1.0, z1=1.0, z2=1.0)
    ok = flat.status == "inconclusive" and isinstance(flat.inner, Inconclusive)
    ok = ok and empty.status == "rejected"
    return {"name": "g5_inconclusive", "passed": ok}


def _run_g6() -> dict[str, Any]:
    arch = MLPArchitecture(dims=(1, 1), activation="tanh")
    ident = squeeze_identifiability(
        arch, [((-1.0,), (1.0,)), ((1.0,), (-1.0,))], [(0.5, 1.5)] * arch.n_params, tol=1e-2, max_boxes=4000
    )
    rem = squeeze_remainder([math.exp(x) for x in (-0.2, 0.0, 0.2)], exp_jet(2), [-0.2, 0.0, 0.2])
    spec = named_tower_fields()[0]
    flow = squeeze_flow(
        lohner_flow_jet(tower_field(spec), tower_jacobian(spec), [Interval(0.2, 0.3)], 0.05, 2, order=4)
    )
    func, jac, lip = polynomial_sqrt2_maps()
    exist = squeeze_existence(
        func=func, jacobian=jac, a_inv=[[1.0 / 3.0]], trial_params=[1.5], lipschitz_df=lip, r_max=0.2
    )
    honesty = ident.honesty["p_vs_np_claim"] is False and flow.honesty["navier_stokes_regularity_claim"] is False
    ok = ident.scope == "parameter_box" and rem.budget is not None and exist.status == "certified" and honesty
    return {"name": "g6_honesty", "passed": ok, "ident": ident.status}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "enclosure_collapse.json" if full else "enclosure_collapse_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.enclosure_collapse.v1",
            config={
                "family": "enclosure_collapse",
                "full": full,
                "honesty": {
                    "enclosure_collapse": True,
                    "width_to_zero": True,
                    "theorem_prover_verified": False,
                },
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "enclosure_collapse"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

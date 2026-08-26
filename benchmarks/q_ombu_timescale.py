# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-15: q-OMBU / timescale hybrid.

G1 matches D_q z^2 = 4.02. G2 is monotone as q -> 1.
G3 records no continuum claim from the named limit. G4 is
torch/jax bit-identity on G1.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "omnibias-qcalculus" / "src"))
from omnibias.qcalculus._core.hybrid import (  # noqa: E402
    DISCLAIMER,
    honesty_payload,
    q_limit_skill,
    worked_example,
)
from omnibias.qcalculus.jax.hybrid import q_ombu_forward as jax_fwd  # noqa: E402
from omnibias.qcalculus.jax.hybrid import worked_example as jax_ex  # noqa: E402
from omnibias.qcalculus.torch.hybrid import q_ombu_forward as torch_fwd  # noqa: E402
from omnibias.qcalculus.torch.hybrid import worked_example as torch_ex  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["y_err"] < 1e-12 and ex["limit_err"] < 1e-12)
    skill = q_limit_skill()
    g2 = bool(skill["monotone"])
    hon = honesty_payload()
    g3 = hon["continuum_claimed_from_q_limit"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t_y, t_r = torch_fwd(torch.tensor(2.0))
    j_y, j_r = jax_fwd(jnp.asarray(2.0))
    g4 = bool(
        torch_ex()["y"] == jax_ex()["y"]
        and float(t_y) == float(j_y)
        and float(t_r) == float(j_r)
    )
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "y": ex["y"],
            "y_err": ex["y_err"],
            "limit": ex["limit"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "errs": skill["errs"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "continuum_claimed_from_q_limit": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_y": float(t_y),
            "jax_y": float(j_y),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.q_ombu_timescale.v1",
            config={"family": "q_ombu_timescale", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "q_ombu_timescale.json" if full else "q_ombu_timescale_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "q_ombu"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

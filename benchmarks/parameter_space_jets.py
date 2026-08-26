# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-27: parameter-space mixed jets.

G1 is the Fourier heat identity. G2 beats h=1e-3 FD. G3/G4
refuse a ParamPINN / autodiff closed_form label. G5 is
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
from omnibias.core.parameter_jets import (
    DISCLAIMER,
    ParameterJetSpec,
    honesty_payload,
    mixed_jet,
    parameter_jet_skill,
    worked_example,
)
from omnibias.pinn.operator.jax.parameter_jets import mixed_jet as jax_jet
from omnibias.pinn.operator.jax.parameter_jets import worked_example as jax_ex
from omnibias.pinn.operator.torch.parameter_jets import mixed_jet as torch_jet
from omnibias.pinn.operator.torch.parameter_jets import worked_example as torch_ex

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
    g1 = ex["abs_sum"] < 1e-8
    skill = parameter_jet_skill()
    g2 = bool(skill["g2_earned"])
    closed_refused = False
    try:
        mixed_jet(
            None,
            (1.0, 1.0),
            1.0,
            spec=ParameterJetSpec(method="closed_form", mu_in_jet_trunk=False),
        )
    except ValueError:
        closed_refused = True
    hon = honesty_payload()
    g3 = closed_refused
    g4 = hon["parampinn_package"] is False and hon["ns_claim"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t_du = torch_jet(None, torch.tensor([ex["x"], ex["t"]]), torch.tensor([ex["mu"]]))
    j_du = jax_jet(None, jnp.asarray([ex["x"], ex["t"]]), jnp.asarray([ex["mu"]]))
    g5 = bool(torch_ex() == jax_ex() and t_du == j_du)
    entries = [
        {"name": "g1_cell", "passed": g1, "abs_sum": ex["abs_sum"], "u": ex["u"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "jet_mae": skill["jet_mae"],
            "fd_mae": skill["fd_mae"],
            "g2_earned": skill["g2_earned"],
        },
        {"name": "g3_method", "passed": g3, "closed_form_refused_without_trunk": True},
        {
            "name": "g4_honesty",
            "passed": g4,
            "parampinn_package": False,
            "ns_claim": False,
        },
        {
            "name": "g5_parity",
            "passed": g5,
            "torch_du": t_du,
            "jax_du": j_du,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.parameter_space_jets.v1",
            config={"family": "parameter_space_jets", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "parameter_space_jets.json" if full else "parameter_space_jets_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "parameter_space_jets"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

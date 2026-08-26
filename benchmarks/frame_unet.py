# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-04: Frame-UNet (order encoder + integral decoder).

G1 splits the band skip from the collapse head. G2 denoises a named
1-D sine against a same-width Scan-Net. G3 keeps sigma' non-admissible.
G4 is torch/jax bit-identity on the G1 worked example and forward.
Not ImageNet. Not CCF stretch. Jets are founding bias collapse, not
temperature collapse.
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
from omnibias.core.frame_unet import (
    DISCLAIMER,
    denoise_skill,
    honesty_payload,
    worked_example,
)
from omnibias.jax.architectures.frame_unet import (
    frame_unet_forward as jax_fwd,
)
from omnibias.jax.architectures.frame_unet import (
    worked_example as jax_ex,
)
from omnibias.torch.architectures.frame_unet import (
    frame_unet_forward as torch_fwd,
)
from omnibias.torch.architectures.frame_unet import (
    worked_example as torch_ex,
)

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
    g1 = bool(ex["band_err"] < 1e-12 and ex["collapse_err"] < 1e-12 and ex["skip_gap"] > 1e-3)
    skill = denoise_skill()
    g2 = bool(skill["below_zero"] and skill["not_worse_than_scan"])
    hon = honesty_payload()
    g3 = hon["sigma_prime_admissible"] is False
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    t_ex = torch_ex()
    j_ex = jax_ex()
    ty, ts = torch_fwd(torch.tensor(0.0))
    jy, js = jax_fwd(jnp.asarray(0.0))
    g4 = bool(
        t_ex["band"] == j_ex["band"]
        and t_ex["collapse"] == j_ex["collapse"]
        and float(ty) == float(jy)
        and ts["band"] == js["band"]
        and ts["collapse"] == js["collapse"]
    )
    entries = [
        {
            "name": "g1_skip_split",
            "passed": g1,
            "band": ex["band"],
            "collapse": ex["collapse"],
            "skip_gap": ex["skip_gap"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "unet_median": skill["unet_median"],
            "scan_median": skill["scan_median"],
            "zero_mse": skill["zero_mse"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "sigma_prime_admissible": False,
        },
        {
            "name": "g4_parity",
            "passed": g4,
            "torch_y": float(ty),
            "jax_y": float(jy),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.frame_unet.v1",
            config={"family": "frame_unet", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "frame_unet.json" if full else "frame_unet_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "frame_unet"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()

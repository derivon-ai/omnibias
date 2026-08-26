# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 architecture: multi-interface transmission PINN (theory 02-05).

``alpha -> inf`` is interface sharpening, neither collapse. Parallel
interfaces only. Conditions hold to a stated smoothing tolerance.
G3 versus PartitionedField / FBPINN / MLP is leftover-recorded
(leftover #37): smoke is a linear stand-in, not a training bake-off.
G4 hard versus penalized is reported unearned: smoke is zero-coeff
soft, not an equal-budget train.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block, rel_l2, skill_score  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _affine(a: float, b: float):
    import torch

    def fn(x: torch.Tensor) -> torch.Tensor:
        return a + b * x[..., 0]

    return fn


def _exact_two_layer(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0, 0.8 * (x + 1.0), 0.8 + 0.2 * x)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.pinn.interface import Interface
    from omnibias.pinn.interface.torch import MultiInterfaceField

    torch.set_default_dtype(torch.float64)
    alpha = 1.0e8
    c = -0.3
    a = 0.8 - abs(c) * math.log(2.0) / alpha
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=alpha, jump=-0.6)
    field = MultiInterfaceField(_affine(a, 0.5), [iface], hard=True, dtype=torch.float64)
    xs = torch.linspace(-1.0, 1.0, 81, dtype=torch.float64).reshape(-1, 1)
    pred = field(xs).detach().numpy().reshape(-1)
    x = xs.numpy().reshape(-1)
    exact = _exact_two_layer(x)
    far = np.abs(x) > 1.0e-3
    err = rel_l2(pred[far], exact[far])
    return {"name": "g1_sharp_limit", "passed": bool(err <= 1e-10), "rel_l2": err}


def _run_g2() -> dict[str, Any]:
    import torch
    from omnibias.pinn.interface import Interface, smoothing_error_bound
    from omnibias.pinn.interface.torch import MultiInterfaceField

    torch.set_default_dtype(torch.float64)
    c = -0.3
    alphas = (20.0, 40.0, 80.0, 160.0)
    rows = []
    ok = True
    for alpha in alphas:
        a = 0.8 - abs(c) * math.log(2.0) / alpha
        iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=alpha, jump=-0.6)
        field = MultiInterfaceField(_affine(a, 0.5), [iface], hard=True, dtype=torch.float64)
        u0 = float(field(torch.zeros(1, 1, dtype=torch.float64)))
        err = abs(u0 - 0.8)
        bound = smoothing_error_bound(iface, coeff=c)
        covered = err <= float(bound.hi) + 1e-15
        ok = ok and covered
        rows.append({"alpha": alpha, "err": err, "bound_hi": float(bound.hi), "covered": covered})
    scaled = [r["err"] * r["alpha"] for r in rows]
    rate_ok = max(scaled) / min(scaled) < 1.2
    return {
        "name": "g2_smoothing_rate",
        "passed": bool(ok and rate_ok),
        "rows": rows,
        "err_times_alpha": scaled,
    }


def _run_g3() -> dict[str, Any]:
    """Named leftover: mixed-condition field vs linear stand-in; bake-off stays --full."""
    import torch
    from omnibias.pinn.interface import Interface
    from omnibias.pinn.interface.torch import MultiInterfaceField
    from omnibias.pinn.partition.torch.field import PartitionedField
    from omnibias.pinn.torch.fields.fbpinn import FBPINNField

    torch.set_default_dtype(torch.float64)
    alpha = 1.0e6
    c = -0.3
    a = 0.8 - abs(c) * math.log(2.0) / alpha
    # Mixed-condition three-layer: flux at 0, plus inert value/curvature packs at ±0.5
    # with jump 0 so they do not change the two-layer solution.
    ifaces = [
        Interface(normal=(1.0,), offset=0.5, condition="value", sharpness=alpha, jump=0.0),
        Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=alpha, jump=-0.6),
        Interface(normal=(1.0,), offset=-0.5, condition="curvature", sharpness=alpha, jump=0.0),
    ]
    field = MultiInterfaceField(_affine(a, 0.5), ifaces, hard=True, dtype=torch.float64)
    xs = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64).reshape(-1, 1)
    pred = field(xs).detach().numpy()
    exact = _exact_two_layer(xs.numpy().reshape(-1))
    err = rel_l2(pred, exact)
    skill = skill_score(pred, exact)
    x = xs.numpy().reshape(-1)
    poly = 0.5 * (x + 1.0)
    err_poly = rel_l2(poly, exact)
    return {
        "name": "g3_mixed_vs_baselines",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "rel_l2": err,
        "rel_l2_linear_standin": err_poly,
        "skill": skill,
        "n_seeds": 1,
        "partitioned_field_exported": callable(PartitionedField),
        "fbpinn_field_exported": callable(FBPINNField),
        "training_loop": False,
        "stays_full": True,
        "leftover_recorded": True,
        "leftover_id": 37,
        "leftover_tick": 59,
        "need": "rel L2 <= 1e-6, skill > 0, beat PartitionedField + FBPINN + MLP, five seeds",
        "note": (
            "Leftover #37 leftover-recorded: MultiInterfaceField on a "
            "mixed-condition three-layer versus a linear stand-in. Named "
            "G3 needs a matched-parameter bake-off against "
            "PartitionedField, FBPINNField, and a plain MLP over five "
            "seeds. That training loop is not wired. Previous "
            "g3_mixed_vs_baselines passed=True stand-in stub withdrawn. "
            "Not in CI all_passed."
        ),
    }


def _run_g4() -> dict[str, Any]:
    """Named leftover: hard residual vs zero-coeff soft; equal budget stays --full."""
    import torch
    from omnibias.pinn.interface import Interface
    from omnibias.pinn.interface.torch import MultiInterfaceField

    torch.set_default_dtype(torch.float64)
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=50.0, jump=-0.6)
    hard = MultiInterfaceField(_affine(0.8, 0.5), [iface], hard=True, dtype=torch.float64)
    soft = MultiInterfaceField(_affine(0.8, 0.5), [iface], hard=False, dtype=torch.float64)
    with torch.no_grad():
        soft.coeffs.fill_(0.0)
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).reshape(-1, 1)
    hard_r = abs(float(hard.interface_residuals(x)[0].detach()))
    soft_r = abs(float(soft.interface_residuals(x)[0].detach()))
    return {
        "name": "g4_hard_vs_penalized",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "hard_residual": hard_r,
        "zero_coeff_residual": soft_r,
        "hard_below_1e_6": bool(hard_r < 1e-6),
        "zero_coeff_worse": bool(soft_r > hard_r),
        "training_loop": False,
        "equal_budget": False,
        "stays_full": True,
        "need": "hard residual at smoothing floor; soft worse at equal training budget",
        "note": (
            "Hard MultiInterfaceField residual versus a zero-coeff soft "
            "field. Named G4 needs hard residuals at the smoothing-error "
            "floor with no interface loss term, and hard=False strictly "
            "worse at equal training budget. That training loop is not "
            "wired. Previous g4_hard_vs_penalized passed=True zero-coeff "
            "stub withdrawn. Not in CI all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.pinn.interface import Interface
    from omnibias.pinn.interface.jax import hard_coeffs, multi_interface_apply
    from omnibias.pinn.interface.torch import MultiInterfaceField

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    iface = Interface(normal=(1.0,), offset=0.0, condition="flux", sharpness=40.0, jump=-0.6)
    field = MultiInterfaceField(_affine(0.79, 0.5), [iface], hard=True, dtype=torch.float64)
    x_np = np.linspace(-1.0, 1.0, 15, dtype=np.float64).reshape(-1, 1)
    t = field(torch.as_tensor(x_np)).detach().numpy()

    def base(x: jnp.ndarray) -> jnp.ndarray:
        return 0.79 + 0.5 * x[..., 0]

    j = np.asarray(
        multi_interface_apply(
            jnp.asarray(x_np),
            base=base,
            interfaces=[iface],
            coeffs=hard_coeffs([iface]),
        )
    )
    worst = float(np.max(np.abs(t - j)))
    return {"name": "g5_parity", "passed": bool(worst <= 1e-12), "max_abs": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3()
    g4 = _run_g4()
    g5 = _run_g5()
    in_scope = [g1, g2, g5]
    payload = provenance(
        schema="multi-interface-pinn-v1",
        config={
            "family": "multi_interface_pinn",
            "full": bool(args.full),
            "gates_in_scope": ["g1", "g2", "g5"],
            "g3_in_all_passed": False,
            "g4_in_all_passed": False,
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["g5"] = g5
    payload["honesty"] = {
        "alpha_inf_is_collapse": False,
        "alpha_inf_is_interface_sharpening": True,
        "exact_transmission_at_finite_alpha": False,
        "parallel_interfaces_only": True,
        "g3_earned": False,
        "g3_reported": True,
        "g3_leftover_recorded": True,
        "g3_leftover_id": 37,
        "g3_leftover_tick": 59,
        "g3_in_ci_all_passed": False,
        "g3_training_loop": False,
        "g4_earned": False,
        "g4_reported": True,
        "g4_in_ci_all_passed": False,
        "g4_training_loop": False,
        "g4_equal_budget": False,
        "temperature_collapse": False,
        "founding_bias_collapse": False,
    }
    if args.full:
        out_dir = SCRATCH / "interface"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "multi_interface_pinn.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("multi_interface_pinn_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

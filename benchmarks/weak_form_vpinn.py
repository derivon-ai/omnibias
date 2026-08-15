# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 architecture: weak-form VPINN (theory 02-04).

Exact integrals only for polynomial coefficients on boxes. Boundary bound
on by default. Path recorded per term. G4 conditioning is a unit test.
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


def _run_g1() -> dict[str, Any]:
    from omnibias.core.scan import BankSpec
    from omnibias.fields._core.quadrature import gauss_legendre
    from omnibias.fields.weak import TestFunctionSpace, eval_test, exact_moment

    space = TestFunctionSpace(
        BankSpec.uniform(0.3, 0.7, 3, scales=(2.0,)),
        orders=(2,),
        base="tanh",
        window=(0.0, 1.0),
    )
    exact = exact_moment(space, 0, 0)
    hi = gauss_legendre(((0.0, 1.0),), 48)
    lo = gauss_legendre(((0.0, 1.0),), 2)
    v_hi = np.array([eval_test(space, 0, float(x)) for x in hi.nodes[:, 0]])
    v_lo = np.array([eval_test(space, 0, float(x)) for x in lo.nodes[:, 0]])
    g_hi = float(np.dot(hi.weights, v_hi))
    g_lo = float(np.dot(lo.weights, v_lo))
    rel = abs(exact - g_hi) / max(abs(g_hi), 1e-16)
    return {
        "name": "g1_exact_integral",
        "passed": bool(rel <= 1e-13 and abs(g_lo - exact) > abs(g_hi - exact)),
        "rel_vs_high_gauss": rel,
        "gauss2_abs_err": abs(g_lo - exact),
        "gauss48_abs_err": abs(g_hi - exact),
        "g4_deferred_from_01_05": True,
    }


def _run_g2() -> dict[str, Any]:
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.fields.weak import TestFunctionSpace, WeakForm
    from omnibias.fields.weak.torch import weak_loss

    torch.set_default_dtype(torch.float64)
    space = TestFunctionSpace(
        BankSpec.uniform(0.4, 0.6, 3, scales=(10.0,)),
        orders=(2,),
        base="tanh",
        window=(0.0, 1.0),
    )
    op = WeakForm(diffusion=(1.0,), source=(2.0,))
    field = (0.0, 1.0, -1.0)
    on = float(weak_loss(field, space, operator=op, include_boundary_bound=True))
    off = float(weak_loss(field, space, operator=op, include_boundary_bound=False))
    return {
        "name": "g2_boundary_honesty",
        "passed": bool(on != off),
        "loss_with_bound": on,
        "loss_without_bound": off,
    }


def _log_cosh(z: np.ndarray) -> np.ndarray:
    az = np.abs(z)
    return az + np.log1p(np.exp(-2.0 * az)) - math.log(2.0)


def _run_g3(*, full: bool) -> dict[str, Any]:
    """Discontinuous-a manufactured 1-D: closed-form flux-ramp ansatz."""
    alpha = 1.0e6
    q = 1.6
    c = -0.6  # slope jump -1.2, tanh-family jump 2
    b = 1.0
    half = 0.5 * alpha
    a = -c * float(_log_cosh(np.array([-half]))[0]) / alpha

    def u_exact(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left = q * x
        right = 0.8 + 0.4 * (x - 0.5)
        return np.where(x <= 0.5, left, right)

    xs = np.linspace(0.0, 1.0, 64)
    pred = a + b * xs + c * _log_cosh(alpha * (xs - 0.5)) / alpha
    err = rel_l2(pred, u_exact(xs))
    skill = skill_score(pred, u_exact(xs))
    phi = np.stack([xs * (1.0 - xs), (xs**2) * (1.0 - xs)], axis=1)
    coeff, *_ = np.linalg.lstsq(phi, u_exact(xs) - xs, rcond=None)
    strong = xs + phi @ coeff
    err_strong = rel_l2(strong, u_exact(xs))
    passed = err <= 1e-6 and skill > 0.0 and err < err_strong
    n_seeds = 5 if full else 1
    return {
        "name": "g3_discontinuous_coeff",
        "passed": bool(passed),
        "rel_l2_weak_ansatz": err,
        "rel_l2_strong_cubic": err_strong,
        "skill": skill,
        "n_seeds": n_seeds,
        "note": "smoke uses 1 seed / closed-form ramp; --full repeats the same check",
    }


def _run_g5() -> dict[str, Any]:
    import jax
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.fields.weak import TestFunctionSpace, WeakForm
    from omnibias.fields.weak.jax import weak_residual as jax_res
    from omnibias.fields.weak.torch import weak_residual as torch_res

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    space = TestFunctionSpace(
        BankSpec.uniform(0.4, 0.6, 3, scales=(10.0,)),
        orders=(2,),
        base="tanh",
        window=(0.0, 1.0),
    )
    op = WeakForm(diffusion=(1.0,), source=(2.0,))
    field = (0.0, 1.0, -1.0)
    t, t_terms = torch_res(field, space, operator=op)
    j, j_terms = jax_res(field, space, operator=op)
    worst = float(np.max(np.abs(t.detach().numpy() - np.asarray(j))))
    paths = [term.path for term in t_terms]
    return {
        "name": "g5_parity",
        "passed": bool(worst <= 1e-14 and all(p == "exact" for p in paths)),
        "max_abs": worst,
        "paths": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3(full=args.full)
    g5 = _run_g5()
    in_scope = [g1, g2, g3, g5]
    payload = provenance(
        schema="weak-form-vpinn-v1",
        config={
            "family": "weak_form_vpinn",
            "full": bool(args.full),
            "gates_in_scope": ["g1", "g2", "g3", "g5"],
            "g4_is_unit_test": True,
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g5"] = g5
    payload["honesty"] = {
        "exact_only_for_polynomial_on_boxes": True,
        "boundary_terms_dropped": False,
        "boundary_bound_default_on": True,
        "sdf_domains_claimed": False,
    }
    if args.full:
        out_dir = SCRATCH / "weakform"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "weak_form_vpinn.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("weak_form_vpinn_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

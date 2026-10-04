# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Probe published consumer features against historical and prepared primitives.

Expected new-wheel incompatibilities are recorded explicitly, never called passes.
Use --require-compatible for the future stable-promotion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "ferminet": (
        "0.2.0",
        "",
        "from omnibias.ferminet.hermite import qho_ground_energy\nassert abs(qho_ground_energy() - .5) < 1e-12",
    ),
    "pinn": (
        "0.1.0",
        "",
        'from omnibias.pinn.train._core.depth_residual import hard_bc_mask_tower\nimport numpy as np\nnp.testing.assert_allclose(hard_bc_mask_tower(.3, "unit_interval", 2), [.21,.4,-2.])',
    ),
    "geometry": (
        "0.2.0",
        "torch",
        "from omnibias.geometry.torch.ops.fisher import exponential_family_fisher_metric\nimport torch\nmetric=exponential_family_fisher_metric(dim=2)\ntorch.testing.assert_close(metric.g_point(torch.zeros(2)), torch.eye(2)*.25)",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/published-compatibility")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--require-compatible", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheels = args.wheelhouse.resolve()
    constraints = output / "new-constraints.txt"
    constraints.write_text(
        "".join(
            f"{name} @ {next(wheels.glob(name.replace('-', '_') + '-*.whl')).as_uri()}\n"
            for name in (
                "omnibias-core",
                "omnibias-torch",
                "omnibias-jax",
                "omnibias-fields",
                "omnibias-keras",
            )
        )
    )
    env = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}
    }
    env.update(
        UV_TORCH_BACKEND="cpu", JAX_ENABLE_X64="true", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="2"
    )
    rows = []
    for name, (version, extra, code) in PROFILES.items():
        for mode, pins in [
            ("historical", ROOT / "scripts/constraints-published.txt"),
            ("prepared", constraints),
        ]:
            with tempfile.TemporaryDirectory(prefix="omnibias-published-") as temp:
                work = Path(temp)
                python = work / "env/bin/python"
                subprocess.run(
                    ["uv", "venv", "--python", args.python, str(work / "env")],
                    env=env,
                    check=True,
                    capture_output=True,
                )
                req = f"omnibias-{name}" + (f"[{extra}]" if extra else "") + f"=={version}"
                install = subprocess.run(
                    ["uv", "pip", "install", "--python", str(python), "-c", str(pins), req],
                    env=env,
                    cwd=work,
                    capture_output=True,
                    text=True,
                )
                if install.returncode:
                    raise RuntimeError(install.stderr)
                shutil.copy2(ROOT / "scripts/import_smoke.py", work / "import_smoke.py")
                # Run the feature first, then inspect every module it loaded.
                code_with_origin_check = code + "\n" + (
                    "import runpy, sys\n"
                    f"sys.argv = ['import_smoke.py', '--distribution', 'omnibias-{name}']\n"
                    "runpy.run_path('import_smoke.py', run_name='__main__')\n"
                )
                probe = subprocess.run(
                    [str(python), "-I", "-c", code_with_origin_check],
                    env=env,
                    cwd=work,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                (output / f"{name}-{mode}.log").write_text(
                    install.stdout + install.stderr + probe.stdout + probe.stderr
                )
                rows.append(
                    {
                        "package": name,
                        "version": version,
                        "mode": mode,
                        "compatible": probe.returncode == 0,
                        "detail": probe.stderr.splitlines()[-1:],
                    }
                )
                print(rows[-1], flush=True)
    (output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    return int(
        any(
            not r["compatible"] and (r["mode"] == "historical" or args.require_compatible)
            for r in rows
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

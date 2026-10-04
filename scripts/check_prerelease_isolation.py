# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verify normal resolution keeps historical primitives when prerelease wheels exist."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    env = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}
    }
    env["UV_TORCH_BACKEND"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="omnibias-prerelease-isolation-") as temp:
        work = Path(temp)
        links = work / "candidates"
        links.mkdir()
        for name in ["core", "torch", "jax", "fields", "keras"]:
            wheel = next(args.wheelhouse.glob(f"omnibias_{name}-*.whl"))
            shutil.copy2(wheel, links / wheel.name)
        python = work / "env/bin/python"
        subprocess.run(["uv", "venv", "--python", "3.12", str(work / "env")], check=True, env=env)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--find-links",
                str(links),
                "omnibias-ferminet==0.2.0",
                "omnibias-pinn==0.1.0",
                "omnibias-geometry==0.2.0",
            ],
            cwd=work,
            env=env,
            check=True,
        )
        probe = 'from importlib.metadata import version; import json; print(json.dumps({n:version(n) for n in ["omnibias-core","omnibias-jax","omnibias-fields"]}))'
        versions = json.loads(
            subprocess.check_output([str(python), "-I", "-c", probe], cwd=work, env=env, text=True)
        )
        assert versions == {
            "omnibias-core": "0.4.0",
            "omnibias-jax": "0.4.0",
            "omnibias-fields": "0.1.0",
        }, versions
        print(json.dumps({"ordinary_install": versions, "prerelease_wheels_available": True}))


if __name__ == "__main__":
    main()

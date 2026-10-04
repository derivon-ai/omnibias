#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Build an ecosystem wheelhouse and validate isolated consumer installations.

Run with --projects-root pointing at the extracted project inventory. Source
checkout paths are used only to build wheels and copy test inputs, never to
resolve imports in the validation subprocesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def project_data(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads((path / "pyproject.toml").read_text())["project"])


def discover(projects_root: Path | None) -> tuple[list[Path], list[Path]]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    primitives = [ROOT / p for p in config["tool"]["uv"]["workspace"]["members"]]
    consumers = []
    if projects_root:
        inventory = json.loads((projects_root / "migration.json").read_text())
        consumers = [
            (projects_root / row["destination"]).resolve() for row in inventory["consumers"]
        ]
        support = projects_root / "omnibias-research"
        consumers.append(support)
        consumers.extend(p.parent for p in sorted((support / "packages").glob("*/pyproject.toml")))
    return primitives, consumers


def wheel_index(wheelhouse: Path) -> dict[str, Path]:
    index = {}
    owners: dict[str, str] = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            meta_path = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            meta = BytesParser().parsebytes(archive.read(meta_path))
            name = meta["Name"]
            if name in index:
                raise ValueError(f"Multiple wheels for {name}; use a clean wheelhouse")
            index[name] = wheel
            for member in archive.namelist():
                if member.endswith("/") or ".dist-info/" in member:
                    continue
                if member in owners:
                    raise ValueError(
                        f"Overlapping wheel file {member}: {owners[member]} and {name}"
                    )
                owners[member] = name
    return index


def checked_wheel_metadata(wheel: Path, expected: dict[str, Any]) -> Any:
    """Reject artifacts built from different version or license metadata."""
    with zipfile.ZipFile(wheel) as archive:
        paths = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(paths) != 1:
            raise ValueError(f"{wheel}: expected exactly one distribution metadata record")
        metadata = BytesParser().parsebytes(archive.read(paths[0]))
    for field, key in [("Name", "name"), ("Version", "version"), ("License-Expression", "license")]:
        if metadata[field] != expected[key]:
            raise ValueError(f"{wheel}: {field} differs from declared project metadata")
    return metadata


def wheel_readme(metadata: Any) -> str:
    """Core metadata is UTF-8 even without an email Content-Type charset."""
    payload = metadata.get_payload(decode=True)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Missing wheel long description")
    return payload.decode("utf-8")


def run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str]) -> None:
    with log.open("a") as output:
        output.write("\n$ " + " ".join(command) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def build_wheel(project: Path, wheelhouse: Path, *, log: Path, env: dict[str, str]) -> None:
    """Build from a source archive so deleted modules cannot survive in build/lib."""
    with tempfile.TemporaryDirectory(prefix="omnibias-sdist-") as scratch:
        source = Path(scratch)
        run(
            ["uv", "build", "--sdist", "--no-sources", str(project), "--out-dir", str(source)],
            cwd=source,
            log=log,
            env=env,
        )
        archives = list(source.glob("*.tar.gz"))
        if len(archives) != 1:
            raise ValueError(f"Expected one source archive for {project.name}, got {len(archives)}")
        run(
            [
                "uv",
                "build",
                "--wheel",
                "--no-sources",
                str(archives[0]),
                "--out-dir",
                str(wheelhouse),
            ],
            cwd=source,
            log=log,
            env=env,
        )


def validate(
    project: Path,
    wheel: Path,
    constraints: Path,
    output: Path,
    env: dict[str, str],
    *,
    numerical: bool,
    readme: bool = False,
    python_version: str = "3.12",
) -> dict[str, Any]:
    data = project_data(project)
    name = data["name"]
    artifact_metadata = checked_wheel_metadata(wheel, data)
    profile_path = project / "wheel-tests.toml"
    profile = tomllib.loads(profile_path.read_text()) if profile_path.exists() else {}
    extras = profile.get(
        "readme_extras" if readme else "extras" if numerical else "base_extras", []
    )
    if readme and "readme_extras" not in profile:
        raise ValueError(f"{name}: missing explicit README profile")
    modules = profile.get("modules" if numerical else "base_modules", []) or [
        "omnibias." + name.removeprefix("omnibias-").replace("-", "_")
    ]
    log = output / f"{name}.log"
    log.write_text("")
    with tempfile.TemporaryDirectory(prefix="omnibias-wheel-") as scratch:
        work = Path(scratch)
        python = work / "env/bin/python"
        run(
            ["uv", "venv", "--python", python_version, str(work / "env")],
            cwd=work,
            log=log,
            env=env,
        )
        requested = name + ("[" + ",".join(extras) + "]" if extras else "") + " @ " + wheel.as_uri()
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--constraint",
                str(constraints),
                requested,
            ],
            cwd=work,
            log=log,
            env=env,
        )
        probe = work / "import_smoke.py"
        shutil.copy2(ROOT / "scripts/import_smoke.py", probe)
        command = [str(python), "-I", str(probe), "--distribution", name]
        for module in modules:
            command.extend(["--module", module])
        if data["license"] == "Apache-2.0":
            command.append("--permissive")
        run(command, cwd=work, log=log, env=env)
        if readme:
            source = wheel_readme(artifact_metadata)
            blocks = re.findall(r"^```python[^\n]*\n(.*?)^```", source, re.MULTILINE | re.DOTALL)
            runner = work / "readme_example.py"
            runner.write_text(
                "\n".join(blocks) + "\nimport sys, import_smoke\n"
                "for name, module in tuple(sys.modules.items()):\n"
                "    if name == 'omnibias' or name.startswith('omnibias.'):\n"
                "        import_smoke.check_module(module, import_smoke.installed_roots())\n"
            )
            run([str(python), str(runner)], cwd=work, log=log, env=env)
            for command in profile.get("readme_commands", []):
                run([str(python), *command], cwd=work, log=log, env=env)
        tests = profile.get("tests", []) if numerical else []
        if tests:
            shutil.copytree(
                project / "tests",
                work / "tests",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            # Keep only test helper paths, never inherit project/source pytest settings.
            (work / "pytest.ini").write_text(
                "[pytest]\naddopts = --import-mode=importlib -m 'not slow'\n"
                "pythonpath = tests tests/solver\nmarkers =\n"
                "    slow: expensive optional check\n    timeout: runtime budget\n"
                "    needs_research: external data\n    gpu: accelerator required\n"
                "    slow_gpu: expensive accelerator check\n"
            )
            # Run import-origin checks after the numerical tests in the same process.
            (work / "wheel_guard.py").write_text(
                "import sys\nimport import_smoke\n"
                "def pytest_sessionfinish(session, exitstatus):\n"
                "    for name, module in tuple(sys.modules.items()):\n"
                "        if name == 'omnibias' or name.startswith('omnibias.'):\n"
                "            import_smoke.check_module(module, import_smoke.installed_roots())\n"
            )
            runner = work / "run_tests.py"
            runner.write_text(
                "import pytest, wheel_guard\nraise SystemExit(pytest.main("
                + repr(["-c", "pytest.ini", "-q", *tests])
                + ", plugins=[wheel_guard]))\n"
            )
            run([str(python), str(runner)], cwd=work, log=log, env=env)
        return {
            "name": name,
            "version": data["version"],
            "extras": extras,
            "modules": modules,
            "test_paths": tests,
            "status": "passed",
            "readme_blocks": len(blocks) if readme else None,
            "readme_commands": profile.get("readme_commands", []) if readme else [],
            "python": python_version,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/wheel-validation")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument(
        "--readme",
        action="store_true",
        help="Execute README code using its declared feature profile",
    )
    parser.add_argument("--python", default="3.12", dest="python_version")
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Validate existing wheels without rebuilding or resolving source overrides",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheelhouse = output / "wheels"
    wheelhouse.mkdir(exist_ok=True)
    primitives, consumers = discover(args.projects_root.resolve() if args.projects_root else None)
    projects = primitives + consumers
    names = {project_data(p)["name"] for p in projects}
    if args.only and not set(args.only) <= names:
        parser.error("--only must name discovered distributions")
    env = {
        k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}
    }
    env.update(
        KERAS_BACKEND="torch",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        OMP_NUM_THREADS="2",
        UV_TORCH_BACKEND="cpu",
    )
    if not (args.no_build or args.artifact_only):
        for project in projects:
            name = project_data(project)["name"]
            print("build", name, flush=True)
            build_wheel(project, wheelhouse, log=output / "build.log", env=env)
    wheels = wheel_index(wheelhouse)
    expected = set(args.only) if args.artifact_only and args.only else names
    if not expected <= set(wheels) or not set(wheels) <= names:
        raise ValueError(f"Wheelhouse coverage differs: {set(wheels) ^ expected}")
    constraints = output / "constraints.txt"
    constraints.write_text(
        "".join(f"{name} @ {wheel.as_uri()}\n" for name, wheel in sorted(wheels.items()))
    )
    targets = projects if args.readme else consumers or primitives
    if args.only:
        targets = [p for p in projects if project_data(p)["name"] in args.only]
    report: dict[str, Any] = {"schema": 1, "wheel_count": len(wheels), "results": []}
    for project in targets:
        name = project_data(project)["name"]
        print("validate", name, flush=True)
        try:
            result = validate(
                project,
                wheels[name],
                constraints,
                output,
                env,
                numerical=args.numerical,
                readme=args.readme,
                python_version=args.python_version,
            )
        except Exception as error:
            result = {"name": name, "status": "failed", "error": str(error)}
        report["results"].append(result)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(result["status"], name, flush=True)
    return int(any(row["status"] != "passed" for row in report["results"]))


if __name__ == "__main__":
    sys.exit(main())

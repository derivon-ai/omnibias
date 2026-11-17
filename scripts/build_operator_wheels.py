#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Build and verify the three operator wheels without network or dependency changes.

Run only after the implementation is stable:
    .venv/bin/python scripts/build_operator_wheels.py --output-dir artifacts/operator-wheels-v2

The build backend and build frontend must already be installed. Setuptools 84
ships its wheel command, so a separate wheel distribution is unnecessary; the
frontend dependency check is skipped and no installer is invoked while building.
Fresh package copies exclude generated files. Installation is confined to a new
temporary /tmp target and uses only the completed local wheels, without deps.
The output directory must be empty; prior evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PACKAGES = ("omnibias-core", "omnibias-torch", "omnibias-keras")
EXCLUDED_NAMES = {
    "build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".git", ".venv", ".tox", ".coverage", "htmlcov",
}
REQUIRED_SHIPPED_FILES = {
    "omnibias-core": {
        "omnibias/core/apolar.py", "omnibias/core/jet_algebra.py",
        "omnibias/core/jet_series.py", "omnibias/core/operator_spec.py",
        "omnibias/core/operator_polynomial.py", "omnibias/core/operator_quotient.py",
    },
    "omnibias-torch": {
        "omnibias/torch/experimental/__init__.py",
        "omnibias/torch/experimental/compiler.py",
        "omnibias/torch/experimental/operator.py",
        "omnibias/torch/experimental/_operator_apolar.py",
        "omnibias/torch/experimental/_operator_ir.py",
        "omnibias/torch/experimental/_operator_lowering.py",
        "omnibias/torch/experimental/_operator_metric.py",
        "omnibias/torch/experimental/_operator_separable.py",
        "omnibias/torch/experimental/_kernel_lowering.py",
        "omnibias/torch/experimental/_parameter_ad.py",
    },
    "omnibias-keras": {
        "omnibias/keras/experimental/__init__.py",
        "omnibias/keras/experimental/model.py",
        "omnibias/keras/experimental/operator.py",
        "omnibias/keras/experimental/precision.py",
    },
}

PROBE = r'''
import importlib
import json
from pathlib import Path
import sys
import tempfile

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import numpy as np
import torch
import keras
from keras import ops
from omnibias.torch.experimental import JetSpec, OperatorSpec, compile_jet, compile_operator
from omnibias.keras.experimental import PrecisionMeanSquaredError, operator_model

torch.set_num_threads(1)

for name in (
    "omnibias.core.apolar", "omnibias.core.jet_algebra", "omnibias.core.jet_series",
    "omnibias.core.operator_spec", "omnibias.core.operator_polynomial",
    "omnibias.core.operator_quotient", "omnibias.torch.experimental",
    "omnibias.keras.experimental", "omnibias.keras.experimental.precision",
    "omnibias.torch.experimental._kernel_lowering", "omnibias.torch.experimental._parameter_ad",
):
    importlib.import_module(name)

class TorchSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.7, -.2], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
    def forward(self, x):
        return torch.exp((x * self.weight).sum(-1) + self.bias)

source = TorchSource()
points = torch.tensor([.125, -.5], dtype=torch.float64)
program = compile_operator(source, (points,), operator=OperatorSpec.mixed((2, 1)))
assert {id(p) for p in program.parameters()} == {id(p) for p in source.parameters()}
actual = program(points)
expected = source.weight[0]**2 * source.weight[1] * source(points)
torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
expected_gradients = torch.autograd.grad(expected, tuple(source.parameters()))
actual.backward()
for parameter, expected_gradient in zip(source.parameters(), expected_gradients, strict=True):
    torch.testing.assert_close(parameter.grad, expected_gradient, rtol=2e-12, atol=2e-14)

names = dict(program.named_parameters())
parameters = tuple(names.values())
def double_gradient(value):
    first = torch.autograd.grad(value, parameters, create_graph=True)
    return torch.autograd.grad(sum(g.sum() for g in first), parameters)
hybrid = program.hybrid(backend="aot_eager")
hybrid_second = double_gradient(hybrid(points).square().sum())
expected_second = double_gradient((source.weight[0]**2 * source.weight[1] * source(points)).square().sum())
for observed, wanted in zip(hybrid_second, expected_second, strict=True):
    torch.testing.assert_close(observed, wanted, rtol=2e-12, atol=2e-14)
hvp = program.compile_parameter_hvp(lambda value: value.square().sum(), backend="aot_eager")
observed_hvp = hvp({name: torch.ones_like(value) for name, value in names.items()}, points)
for name, wanted in zip(names, expected_second, strict=True):
    torch.testing.assert_close(observed_hvp[name], wanted, rtol=2e-12, atol=2e-14)

spec = JetSpec.directional(4)
packed = torch.tensor([.2, .3, .04, -.01, .002], dtype=torch.float64)
lowerings = ("auto", "inline", "packed", "regional")
kernels = {mode: compile_jet(torch.nn.Tanh(), (packed[0],), spec=spec, kernel_lowering=mode)(packed)
           for mode in lowerings}
for value in kernels.values():
    torch.testing.assert_close(value, kernels["inline"], rtol=2e-12, atol=2e-14)

inputs = keras.Input((2,), dtype="float64")
layer = keras.layers.Dense(1, dtype="float64")
keras_source = keras.Model(inputs, ops.exp(layer(inputs)))
keras_source.set_weights([np.array([[.5], [-.25]]), np.array([.125])])
keras_program = operator_model(keras_source, operator=OperatorSpec.mixed((1, 1)), precision="strict")
assert {id(v) for v in keras_program.trainable_variables} == {id(v) for v in keras_source.trainable_variables}
keras_points = np.array([[.125, -.5], [.25, .125]], dtype=np.float64)
keras_actual = keras_program(keras_points)
keras_expected = -.125 * np.exp(.5 * keras_points[:, :1] - .25 * keras_points[:, 1:] + .125)
np.testing.assert_allclose(ops.convert_to_numpy(keras_actual), keras_expected, rtol=2e-12, atol=2e-14)
keras_parameters = tuple(v.value for v in keras_source.trainable_variables)
keras_gradients = torch.autograd.grad(keras_actual.sum(), keras_parameters)
assert all(bool(torch.isfinite(v).all()) for v in keras_gradients)
assert all(bool(torch.count_nonzero(v)) for v in keras_gradients)

precision_loss = PrecisionMeanSquaredError(dtype="float64")
truth = np.zeros_like(keras_expected)
objective = precision_loss(truth, keras_actual)
assert str(objective.dtype) == "torch.float64"
np.testing.assert_allclose(ops.convert_to_numpy(objective), np.mean(keras_expected**2), rtol=2e-12, atol=2e-14)
keras_program.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3), loss=precision_loss,
                      run_eagerly=True)
before = [np.array(ops.convert_to_numpy(v)) for v in keras_source.trainable_variables]
keras_program.train_on_batch(keras_points, truth)
assert any(not np.array_equal(a, ops.convert_to_numpy(v))
           for a, v in zip(before, keras_source.trainable_variables, strict=True))
with tempfile.TemporaryDirectory(prefix="operator-wheel-roundtrip-") as directory:
    path = Path(directory) / "strict.keras"
    keras_program.save(path)
    restored = keras.models.load_model(path)
    assert restored.precision == "strict"
    assert int(ops.convert_to_numpy(restored.optimizer.iterations)) == int(ops.convert_to_numpy(keras_program.optimizer.iterations))
    np.testing.assert_allclose(ops.convert_to_numpy(restored(keras_points)),
                               ops.convert_to_numpy(keras_program(keras_points)), rtol=2e-12, atol=2e-14)
    restored.train_on_batch(keras_points, truth)
    keras_program.train_on_batch(keras_points, truth)
    for a, b in zip(restored.trainable_variables, keras_program.trainable_variables, strict=True):
        np.testing.assert_allclose(ops.convert_to_numpy(a), ops.convert_to_numpy(b), rtol=2e-12, atol=2e-14)

imported_files = {}
for name, module in tuple(sys.modules.items()):
    if name.startswith(("omnibias.core", "omnibias.torch", "omnibias.keras")):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            assert path.is_relative_to(target), (name, str(path), "not from installed wheels")
            imported_files[name] = str(path.relative_to(target))
print(json.dumps({
    "status": "passed",
    "backend": keras.backend.backend(),
    "torch_version": torch.__version__,
    "keras_version": keras.__version__,
    "imports_from_isolated_target": imported_files,
    "torch_derivative_and_parameter_backward": "passed",
    "torch_parameter_identity": "passed",
    "torch_hybrid_double_backward": "passed",
    "torch_compiled_parameter_hvp": "passed (aot_eager)",
    "torch_activation_lowerings": list(lowerings),
    "keras_derivative_and_parameter_backward": "passed",
    "keras_variable_identity": "passed",
    "keras_strict_loss_dtype_and_value": "passed",
    "keras_strict_save_load_training_continuation": "passed (Torch backend, eager optimizer smoke)",
    "torch_strategy": program.explain()["selected_strategy"],
    "keras_strategy": keras_program.explain()["selected_strategy"],
}, indent=2))
'''


def excluded(name: str) -> bool:
    return name in EXCLUDED_NAMES or name.endswith((".egg-info", ".pyc", ".pyo"))


def ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if excluded(name)]


def file_manifest(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and not any(excluded(part) for part in relative.parts):
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tree_hash(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    log.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new or empty evidence directory; existing evidence is never replaced")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Output directory must be new or empty; choose a new evidence path.")
    output.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("The already installed uv executable is required.")
    started = time.monotonic()
    env = dict(os.environ, SOURCE_DATE_EPOCH="315532800", PYTHONHASHSEED="0",
               KERAS_BACKEND="torch", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               MPLCONFIGDIR="/tmp/omnibias-wheel-mpl", UV_NO_PROGRESS="1")
    env.pop("PYTHONPATH", None)
    script_bytes = Path(__file__).read_bytes()
    (output / "build_operator_wheels.source.py").write_bytes(script_bytes)
    report = {
        "schema": "omnibias/operator-wheels/v2",
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "clean_source_build": True,
        "network_access_required": False,
        "installed_dependencies_changed": False,
        "source_date_epoch": int(env["SOURCE_DATE_EPOCH"]),
        "exclusions": sorted(EXCLUDED_NAMES) + ["*.egg-info", "*.pyc", "*.pyo"],
        "build_tools": {name: importlib.metadata.version(name) for name in ("build", "setuptools")},
        "wheels": [],
        "isolated_target_install_probe": "not_run",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="omnibias-operator-wheels-", dir="/tmp") as temporary:
            staging = Path(temporary)
            built = staging / "wheels"
            built.mkdir()
            manifests = {}
            for package in PACKAGES:
                original = repo / "packages" / package
                snapshot = staging / package
                before = file_manifest(original)
                shutil.copytree(original, snapshot, ignore=ignore)
                if file_manifest(snapshot) != before:
                    raise RuntimeError(f"source changed during snapshot: {package}")
                manifests[package] = before
                command = [sys.executable, "-m", "build", "--wheel", "--no-isolation",
                           "--skip-dependency-check", "--outdir", str(built), str(snapshot)]
                run(command, cwd=staging, env=env, log=output / f"{package}.build.log")
                matches = list(built.glob(package.replace("-", "_") + "-*.whl"))
                if len(matches) != 1:
                    raise RuntimeError(f"expected one wheel for {package}, found {matches}")
                wheel = matches[0]
                shipped = {}
                with zipfile.ZipFile(wheel) as archive:
                    names = set(archive.namelist())
                    if not REQUIRED_SHIPPED_FILES[package].issubset(names):
                        raise RuntimeError(f"missing operator files: {REQUIRED_SHIPPED_FILES[package] - names}")
                    # The inventory is derived from the frozen source tree so
                    # newly introduced implementation modules cannot be missed.
                    expected_sources = {name.removeprefix("src/") for name in before
                                        if name.startswith("src/omnibias/") and name.endswith((".py", ".pyi", "py.typed"))}
                    if not expected_sources.issubset(names):
                        raise RuntimeError(f"missing source modules in {package}: {expected_sources - names}")
                    for name in sorted(names):
                        if name.startswith("omnibias/") and not name.endswith("/"):
                            digest = hashlib.sha256(archive.read(name)).hexdigest()
                            if before.get("src/" + name) != digest:
                                raise RuntimeError(f"wheel source differs from snapshot: {name}")
                            shipped[name] = digest
                operator_files = [name for name in shipped if "operator" in name or "/experimental/" in name]
                record = {
                    "package": package,
                    "wheel": str((output / wheel.name).relative_to(repo)) if output.is_relative_to(repo) else str(output / wheel.name),
                    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    "source_tree_sha256": tree_hash(before),
                    "source_file_count": len(before),
                    "operator_files": operator_files,
                    "operator_source_sha256": {name: shipped[name] for name in operator_files},
                    "shipped_source_sha256": shipped,
                    "build_command": command,
                }
                report["wheels"].append(record)
                print(f"Built and verified {wheel.name}", flush=True)
            for package, before in manifests.items():
                if file_manifest(repo / "packages" / package) != before:
                    raise RuntimeError(f"source changed during wheel build: {package}; rebuild after freeze")
            target = staging / "installed"
            install_command = [uv, "--offline", "--no-cache", "pip", "install", "--no-deps", "--no-index",
                               "--python", sys.executable, "--target", str(target), *map(str, sorted(built.glob("*.whl")))]
            run(install_command, cwd=staging, env=env, log=output / "install.log")
            probe_output = run([sys.executable, "-I", "-c", PROBE, str(target)], cwd=staging,
                               env=env, log=output / "probe.log")
            probe = json.loads(probe_output)
            if probe["status"] != "passed":
                raise RuntimeError("isolated wheel probe did not pass")
            (output / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
            report["isolated_target_install_probe"] = "passed"
            report["probe"] = probe
            report["temporary_target_removed_after_probe"] = True
            for wheel in sorted(built.glob("*.whl")):
                shutil.copyfile(wheel, output / wheel.name)
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified all wheels; manifest: {output / 'build.json'}", flush=True)


if __name__ == "__main__":
    main()

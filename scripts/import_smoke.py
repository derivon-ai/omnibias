#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verify installed distributions and imports; reject source and editable leakage."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def installed_roots() -> tuple[Path, ...]:
    return tuple(Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib"))


def check_module(module: ModuleType, roots: tuple[Path, ...]) -> None:
    locations = list(getattr(module, "__path__", []))
    if getattr(module, "__file__", None):
        locations.append(module.__file__)
    if not locations:
        raise ValueError(f"No installed location for {module.__name__}")
    for location in locations:
        resolved = Path(location).resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"Source leakage: {module.__name__} resolved outside site-packages")


def check_distribution(dist: metadata.Distribution, *, permissive: bool) -> None:
    name = dist.metadata["Name"]
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise ValueError(f"Editable installation: {name}")
    if not dist.files:
        raise ValueError(f"Missing wheel RECORD: {name}")
    license_expression = dist.metadata.get("License-Expression", "")
    if permissive and license_expression != "Apache-2.0":
        raise ValueError(f"Non-permissive dependency: {name} ({license_expression})")
    roots = installed_roots()
    for file in dist.files:
        if str(file).startswith("omnibias/"):
            resolved = Path(str(dist.locate_file(file))).resolve()
            if not any(resolved.is_relative_to(root) for root in roots):
                raise ValueError(f"Distribution source leakage: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--distribution", action="append", default=[])
    parser.add_argument("--permissive", action="store_true")
    args = parser.parse_args()
    distributions = {
        dist.metadata["Name"]: dist for dist in metadata.distributions()
        if dist.metadata["Name"].startswith("omnibias-")
    }
    for name in args.distribution:
        if name not in distributions:
            raise ValueError(f"Missing requested distribution: {name}")
    if not distributions:
        raise ValueError("No installed omnibias distributions")
    for dist in distributions.values():
        check_distribution(dist, permissive=args.permissive)
    names = args.module or [
        "omnibias." + name.removeprefix("omnibias-").replace("-", "_")
        for name in sorted(distributions)
    ]
    for name in names:
        importlib.import_module(name)
    roots = installed_roots()
    for name, module in tuple(sys.modules.items()):
        if name == "omnibias" or name.startswith("omnibias."):
            check_module(module, roots)
    print(json.dumps({"imports": names, "distributions": sorted(distributions),
                      "origins": "installed wheels", "permissive": args.permissive}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

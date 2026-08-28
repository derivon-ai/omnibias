#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validate the one-to-one omnibias package skill catalog."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
SKILLS = ROOT / ".cursor" / "skills"
REQUIRED_HEADINGS = (
    "## Why nested AD fails",
    "## What only this tower unlocks",
    "## Use",
    "## Extend",
    "## Next invention",
)
MAX_LINES = 500
#: Cross-cutting stems that are not workspace distributions.
EXTRA_STEMS = frozenset(
    {
        "backends",
        "certificate-lean",
        "control-research",
        "core-concepts",
        "deepmind-campaign",
        "derivative-tower",
        "discovery-engine",
        "discrete-consumer",
        "empirical-validation",
        "field-op",
        "formal-agent",
        "frontier",
        "new-package",
        "pinn-research",
        "verified-primitive",
    }
)


def package_names() -> set[str]:
    """Return every workspace distribution short name."""
    names: set[str] = set()
    for pyproject in PACKAGES.glob("omnibias-*/pyproject.toml"):
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        name = project["name"]
        if isinstance(name, str) and name.startswith("omnibias-"):
            names.add(name.removeprefix("omnibias-"))
    return names


def validate() -> list[str]:
    """Return actionable catalog violations."""
    errors: list[str] = []
    expected = package_names()
    found = {
        path.parent.name.removeprefix("omnibias-")
        for path in SKILLS.glob("omnibias-*/SKILL.md")
        if path.parent.name.startswith("omnibias-")
    }
    for name in sorted(expected - found):
        errors.append(f"missing package skill: omnibias-{name}")
    for name in sorted(found - expected):
        if name not in EXTRA_STEMS:
            errors.append(f"orphan skill: omnibias-{name}")
    for name in sorted(expected):
        path = SKILLS / f"omnibias-{name}" / "SKILL.md"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n") + 1
        if lines > MAX_LINES:
            errors.append(f"{path.relative_to(ROOT)} exceeds {MAX_LINES} lines")
        if not re.search(rf"^name: omnibias-{re.escape(name)}$", text, re.MULTILINE):
            errors.append(f"{path.relative_to(ROOT)} has incorrect skill metadata")
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                errors.append(f"{path.relative_to(ROOT)} missing {heading!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        print("\n".join(errors))
        return 1
    print(f"package skills: {len(package_names())} valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

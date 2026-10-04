# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validate a retained-package release, including its optional dependency closure."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def projects(root: Path) -> dict[str, dict]:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    return {
        Path(member).name.removeprefix("omnibias-"): tomllib.loads(
            (root / member / "pyproject.toml").read_text()
        )["project"]
        for member in config["tool"]["uv"]["workspace"]["members"]
    }


def select(root: Path, requested: str, tag: str, event: str) -> list[str]:
    available = projects(root)
    names = requested.split() or sorted(available)
    tagged_version = None
    if event == "push":
        match = re.fullmatch(r"omnibias-([a-z0-9-]+)-v(.+)", tag)
        if not match:
            raise ValueError("Use an independent omnibias-<package>-v<version> tag")
        names, tagged_version = [match[1]], Version(match[2])
    if not names or any(name not in available for name in names):
        raise ValueError("Release selections must name retained workspace packages")
    names = list(dict.fromkeys(names))
    for name in names:
        if tagged_version is not None and Version(available[name]["version"]) != tagged_version:
            raise ValueError(f"Tag does not match {available[name]['name']} metadata version")
    return names


def index_project(name: str, repository: str) -> dict | None:
    host = {"pypi": "pypi.org", "testpypi": "test.pypi.org"}[repository]
    try:
        with urlopen(f"https://{host}/pypi/{name}/json", timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def ensure_unpublished(name: str, version: str, repository: str) -> None:
    host = {"pypi": "pypi.org", "testpypi": "test.pypi.org"}[repository]
    try:
        with urlopen(f"https://{host}/pypi/{name}/{version}/json", timeout=30):
            pass
    except HTTPError as error:
        if error.code == 404:
            return
        raise
    raise ValueError(
        f"{name} {version} already exists on {repository}; inspect any partial upload before retrying"
    )


def validate_closure(root: Path, names: list[str], repository: str | None = None) -> None:
    """Every omnibias dependency, including extras, must be selected or available."""
    available = projects(root)
    selected = {available[n]["name"]: Version(available[n]["version"]) for n in names}
    cache = {}
    for name in names:
        project = available[name]
        dependencies = list(project.get("dependencies", []))
        dependencies += [
            r for values in project.get("optional-dependencies", {}).values() for r in values
        ]
        for raw in dependencies:
            requirement = Requirement(raw)
            if not requirement.name.startswith("omnibias-"):
                continue
            if requirement.name in selected:
                if requirement.specifier.contains(selected[requirement.name], prereleases=True):
                    continue
                raise ValueError(f"{name}: selected {requirement.name} does not satisfy {raw}")
            if repository is not None:
                if requirement.name not in cache:
                    cache[requirement.name] = index_project(requirement.name, repository)
                metadata = cache[requirement.name]
                if metadata and any(
                    requirement.specifier.contains(v, prereleases=True)
                    and any(not f.get("yanked", False) for f in files)
                    for v, files in metadata.get("releases", {}).items()
                ):
                    continue
            raise ValueError(
                f"{name}: dependency {raw} must be selected or available on the target index"
            )


def require_prerelease(root: Path, names: list[str]) -> None:
    """Stable promotion needs a separately reviewed compatibility-policy change."""
    config = tomllib.loads((root / "pyproject.toml").read_text())
    if config["tool"]["omnibias"]["release"]["prerelease-only"]:
        for name, project in projects(root).items():
            if name in names and not Version(project["version"]).is_prerelease:
                raise ValueError(
                    "Stable promotion is blocked pending published-consumer compatibility"
                )


def publisher_environment(name: str, repository: str) -> str:
    # Existing production projects share a publisher; initial projects need distinct
    # pending identities. Binary is the first pending project on the shared identity.
    shared = {"core", "torch", "jax", "keras", "fields", "binary"}
    return repository if repository == "pypi" and name in shared else repository + "-" + name


def main() -> None:
    names = select(
        ROOT,
        os.getenv("REQUESTED_PACKAGES", ""),
        os.getenv("RELEASE_TAG", ""),
        os.getenv("RELEASE_EVENT", ""),
    )
    repository = os.getenv("RELEASE_REPOSITORY") or "prepare"
    target = None if repository == "prepare" else repository
    require_prerelease(ROOT, names)
    validate_closure(ROOT, list(projects(ROOT)) if target is None else names, target)
    if target:
        for name in names:
            project = projects(ROOT)[name]
            ensure_unpublished(project["name"], project["version"], target)
    print(json.dumps({"packages": names, "repository": repository}))
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a") as output:
            print("pkgs=" + " ".join(names), file=output)
            matrix = [
                {"package": name, "environment": publisher_environment(name, repository)}
                for name in names
            ]
            print("matrix=" + json.dumps({"include": matrix}), file=output)


if __name__ == "__main__":
    main()

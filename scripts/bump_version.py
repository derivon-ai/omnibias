#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bump the version of one or more omnibias distributions.

The 42 packages are versioned **independently**, and each version lives in
exactly one place: the ``version`` field of that package's ``pyproject.toml``.
No module hardcodes it -- ``__version__`` is read from installed distribution
metadata at import time -- so this script is the whole mechanism.

    # inspect
    python scripts/bump_version.py --list
    python scripts/bump_version.py core torch --bump minor --dry-run

    # apply
    python scripts/bump_version.py core --bump patch
    python scripts/bump_version.py verify --set 0.2.0
    python scripts/bump_version.py --all --bump patch

Packages are named by their short name (``core``) or in full
(``omnibias-core``). ``--all`` is accepted but rarely right: bumping 42
versions in lockstep discards the per-package release cadence that independent
versioning exists to provide.

Pre-release suffixes are understood and preserved in the sense PEP 440 defines:

    0.1.0a1 --bump pre    -> 0.1.0a2
    0.1.0a1 --bump patch  -> 0.1.0      (a release of the version being previewed)
    0.1.0   --bump minor  -> 0.2.0

The script refuses to bump backwards, prints the tag to push
(``omnibias-core-v0.4.1``), and reminds you to add a ``CHANGELOG.md`` entry --
it deliberately does not write one, because a generated changelog entry is
worse than none.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"

#: ``N.N.N`` with an optional PEP 440 pre-release segment (``a1``/``b2``/``rc3``).
_VERSION = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:(?P<pre_kind>a|b|rc)(?P<pre_num>\d+))?$"
)

#: Matches the ``version = "..."`` line inside the ``[project]`` table only.
_VERSION_LINE = re.compile(r'^(?P<prefix>version\s*=\s*")(?P<version>[^"]+)(?P<suffix>")$', re.M)

BUMPS = ("major", "minor", "patch", "pre")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    pre_kind: str | None = None
    pre_num: int | None = None

    @classmethod
    def parse(cls, text: str) -> Version:
        m = _VERSION.match(text.strip())
        if m is None:
            raise ValueError(
                f"{text!r} is not a supported version; expected N.N.N with an "
                "optional a/b/rc pre-release suffix"
            )
        pre_kind = m.group("pre_kind")
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            pre_kind=pre_kind,
            pre_num=int(m.group("pre_num")) if pre_kind else None,
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}{self.pre_kind}{self.pre_num}" if self.pre_kind else base

    @property
    def _key(self) -> tuple[int, int, int, int, int]:
        # A release sorts above any pre-release of the same base version.
        rank = {"a": 0, "b": 1, "rc": 2}
        pre = (rank[self.pre_kind], self.pre_num or 0) if self.pre_kind else (3, 0)
        return (self.major, self.minor, self.patch, *pre)

    def __lt__(self, other: Version) -> bool:
        return self._key < other._key

    def bump(self, part: str) -> Version:
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            # Releasing the version a pre-release was previewing: 0.1.0a3 -> 0.1.0.
            if self.pre_kind:
                return Version(self.major, self.minor, self.patch)
            return Version(self.major, self.minor, self.patch + 1)
        if part == "pre":
            if not self.pre_kind:
                raise ValueError(
                    f"{self} is not a pre-release; use --set to start one "
                    f"(e.g. --set {self.major}.{self.minor + 1}.0a1)"
                )
            return Version(self.major, self.minor, self.patch, self.pre_kind, (self.pre_num or 0) + 1)
        raise ValueError(f"unknown bump part {part!r}")


def package_dirs() -> dict[str, Path]:
    """Map distribution name -> package directory."""
    out: dict[str, Path] = {}
    for pkg_dir in sorted(PACKAGES.glob("omnibias-*")):
        pyproject = pkg_dir / "pyproject.toml"
        if not pyproject.is_file():
            continue
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        out[str(data["project"]["name"])] = pkg_dir
    return out


def current_version(pkg_dir: Path) -> str:
    data = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def resolve(names: list[str], known: dict[str, Path]) -> list[str]:
    resolved: list[str] = []
    unknown: list[str] = []
    for name in names:
        full = name if name.startswith("omnibias-") else f"omnibias-{name}"
        (resolved if full in known else unknown).append(full)
    if unknown:
        raise SystemExit(
            f"unknown package(s): {', '.join(unknown)}\n"
            f"known: {', '.join(sorted(n.removeprefix('omnibias-') for n in known))}"
        )
    return resolved


def write_version(pkg_dir: Path, new: Version) -> None:
    pyproject = pkg_dir / "pyproject.toml"
    src = pyproject.read_text(encoding="utf-8")
    updated, count = _VERSION_LINE.subn(
        lambda m: f"{m.group('prefix')}{new}{m.group('suffix')}", src, count=1
    )
    if count != 1:
        raise SystemExit(f"could not locate a single `version = \"...\"` line in {pyproject}")
    pyproject.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("packages", nargs="*", help="short or full package names")
    parser.add_argument("--all", action="store_true", help="every package (rarely what you want)")
    parser.add_argument("--bump", choices=BUMPS, help="which component to increment")
    parser.add_argument("--set", dest="exact", help="set an explicit version")
    parser.add_argument("--list", action="store_true", help="print current versions and exit")
    parser.add_argument("--dry-run", action="store_true", help="show the change without writing")
    args = parser.parse_args()

    known = package_dirs()

    if args.list:
        width = max(len(n) for n in known)
        for name in sorted(known):
            print(f"  {name:<{width}}  {current_version(known[name])}")
        return 0

    if args.all:
        targets = sorted(known)
    elif args.packages:
        targets = resolve(args.packages, known)
    else:
        parser.error("name at least one package, or pass --all or --list")

    if bool(args.bump) == bool(args.exact):
        parser.error("pass exactly one of --bump or --set")

    if args.exact:
        # Validate once, up front, rather than 42 times.
        Version.parse(args.exact)

    changes: list[tuple[str, Version, Version]] = []
    for name in targets:
        old = Version.parse(current_version(known[name]))
        try:
            new = Version.parse(args.exact) if args.exact else old.bump(args.bump)
        except ValueError as exc:
            raise SystemExit(f"{name}: {exc}") from None
        if not old < new:
            raise SystemExit(f"{name}: refusing to move {old} -> {new} (not an increase)")
        changes.append((name, old, new))

    width = max(len(n) for n, _, _ in changes)
    for name, old, new in changes:
        print(f"  {name:<{width}}  {old} -> {new}")
        if not args.dry_run:
            write_version(known[name], new)

    if args.dry_run:
        print(f"\ndry run: {len(changes)} package(s) would change")
        return 0

    print(f"\nbumped {len(changes)} package(s). Next:")
    print("  1. add a CHANGELOG.md entry describing the change (not generated on purpose)")
    print("  2. commit, then tag:")
    for name, _, new in changes:
        print(f"       git tag {name}-v{new}")
    print("  3. push the tag to trigger .github/workflows/release.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

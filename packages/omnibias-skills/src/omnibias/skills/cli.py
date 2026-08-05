# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Command-line entry point for ``omnibias-skills``.

Subcommands: ``install`` / ``update`` / ``uninstall`` / ``list``. Placement is
always explicit -- installing this package never writes skills on its own.
"""

from __future__ import annotations

import argparse

from omnibias.skills._installer import (
    TOOLS,
    InstallResult,
    bundled_skills,
    check_skills,
    install_skills,
    uninstall_skills,
)


def _tools_from_arg(tool: str) -> tuple[str, ...]:
    return TOOLS if tool == "all" else (tool,)


def _print_result(kind: str, result: InstallResult) -> None:
    for path in result.written:
        print(f"  {kind}: {path}")
    for path in result.removed:
        print(f"  removed: {path}")
    for path in result.drifted:
        print(f"  DRIFT: {path}")
    print(
        f"{kind}: {len(result.written)} written, {len(result.skipped)} up-to-date, "
        f"{len(result.removed)} removed, {len(result.drifted)} drifted"
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tool",
        choices=[*TOOLS, "all"],
        default="all",
        help="which assistant to target (default: all)",
    )
    parser.add_argument(
        "--dest",
        default=".",
        help="project root to install into (default: current directory)",
    )
    parser.add_argument(
        "--global",
        dest="global_",
        action="store_true",
        help="install into the user home (~/.cursor, ~/.claude) instead of --dest",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnibias-skills",
        description="Install the omnibias consumer agent-skill library.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install", help="write the bundled skills")
    _add_common(p_install)
    p_install.add_argument("--force", action="store_true", help="rewrite even up-to-date files")
    p_install.add_argument("--dry-run", action="store_true", help="show what would change")
    p_install.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any installed skill drifts from the bundle (implies no writes)",
    )

    p_update = sub.add_parser("update", help="refresh installed skills to this version")
    _add_common(p_update)

    p_uninstall = sub.add_parser("uninstall", help="remove installed omnibias-* skills")
    _add_common(p_uninstall)
    p_uninstall.add_argument("--dry-run", action="store_true", help="show what would be removed")

    sub.add_parser("list", help="list the bundled skills")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command: str = args.command

    if command == "list":
        for info in bundled_skills():
            print(f"{info.name}\t{info.description}")
        return 0

    tools = _tools_from_arg(args.tool)

    if command == "install":
        if args.check:
            result = check_skills(args.dest, tools, global_=args.global_)
            _print_result("check", result)
            return 0 if result.ok else 1
        result = install_skills(
            args.dest, tools, global_=args.global_, force=args.force, dry_run=args.dry_run
        )
        _print_result("dry-run" if args.dry_run else "install", result)
        if "claude" in tools and not args.dry_run:
            print(
                "  tip: point CLAUDE.md at AGENTS.md so Claude Code loads project guidance."
            )
        return 0

    if command == "update":
        result = install_skills(args.dest, tools, global_=args.global_, force=True)
        _print_result("update", result)
        return 0

    if command == "uninstall":
        result = uninstall_skills(args.dest, tools, global_=args.global_, dry_run=args.dry_run)
        _print_result("uninstall", result)
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

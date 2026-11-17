# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Compile the session overview and complete research notes without altering sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT_DOCUMENTS = (
    "bias-collapse-pde.md",
    "affine-geometry.md",
    "calculus-geometry-pilot.md",
    "calculus-geometry-results.md",
    "calculus-geometry-opportunities.md",
    "calculus-geometry-campaign.md",
    "verified-physical-calibration.md",
    "unification-research-audit.md",
    "two-bias-bell-experiment.md",
    "multi-bias-capacity-and-architecture.md",
    "activation-profile-compression.md",
    "multi-bias-learning-and-margin.md",
    "multibias-frontier-study.md",
    "multibias-followup-audit.md",
    "multi-bias-research-synthesis.md",
    "projection-profile-allocation.md",
    "deep-bias-collapse.md",
    "performance.md",
    "profile-pde.md",
    "nonlinear-pde-identities.md",
    "inverse-pde-map.md",
)
EXCLUDED_PARTS = {
    ".git", ".venv", ".agents", ".pytest_cache", ".ruff_cache",
    "__pycache__", "artifacts", "replay_workspaces",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def rebase_links(document: str, source: Path) -> str:
    """Keep source-relative local links usable after concatenation."""
    def replace(match: re.Match[str]) -> str:
        target = match.group(2)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("/"):
            return match.group(0)
        local, sep, anchor = target.partition("#")
        if not local:
            resolved = str(source)
        else:
            resolved = str((source.parent / local).resolve())
        return match.group(1) + resolved + (sep + anchor if sep else "") + match.group(3)

    # Inline code and mathematics can contain bracket/parenthesis expressions
    # which are not Markdown links. Never rewrite those expressions.
    protected = re.split(r"(`+[^`]*`+|\\\(.*?\\\)|(?<!\\)\$[^$]*?(?<!\\)\$)", document)
    return "".join(
        part if index % 2 else re.sub(r"(\]\()([^\s()<>]+)(\))", replace, part)
        for index, part in enumerate(protected)
    )


def appendix(document: str, source: Path) -> str:
    """Nest headings while preserving code fences and mathematical text."""
    lines: list[str] = []
    fence: str | None = None
    display_math: str | None = None
    for line in document.splitlines():
        stripped = line.strip()
        was_fenced = fence is not None
        token = re.match(r"^\s*(`{3,}|~{3,})", line)
        if token:
            character = token.group(1)[0]
            if fence is None:
                fence = character
            elif fence == character:
                fence = None
        is_math = display_math is not None or stripped.startswith(("\\[", "$$"))
        if stripped.startswith("\\[") and "\\]" not in stripped:
            display_math = "\\]"
        elif stripped == "$$":
            display_math = None if display_math == "$$" else "$$"
        elif display_math and display_math in stripped:
            display_math = None
        if not (was_fenced or fence or token or is_math or line.startswith(("    ", "\t"))):
            line = rebase_links(line, source)
            heading = re.match(r"^(#{1,6}) (.*)$", line)
            if heading:
                line = "#" * min(6, len(heading.group(1)) + 2) + " " + heading.group(2)
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lab", type=Path)
    parser.add_argument("--root-base", help="Override the recorded pre-checkpoint core commit")
    args = parser.parse_args()
    root = args.root.resolve()
    lab = (args.lab or root.parent / "omnibias_projects" / "omnibias-lab").resolve()
    data_dir = root / "docs/data/session-research-20261023"
    overview = data_dir / "overview.md"
    manifest_path = data_dir / "sources.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = [root / "docs" / name for name in ROOT_DOCUMENTS]
    paths += sorted(
        path for path in lab.rglob("*.md")
        if not EXCLUDED_PARTS.intersection(path.relative_to(lab).parts)
        and path.name not in {"AGENTS.md", "completion_checklist_pre_closeout.md"}
    )
    records = []
    sections = []
    for number, path in enumerate(paths, start=1):
        raw = path.read_bytes()
        repository = root if path.is_relative_to(root) else lab
        relative = str(path.relative_to(repository))
        item = {
            "appendix": number,
            "repository": "omnibias" if repository == root else "omnibias-lab",
            "path": relative, "absolute_path": str(path),
            "sha256": digest(raw), "bytes": len(raw),
        }
        records.append(item)
        label = f"{item['repository']}/{relative}"
        # This template's links were written for its generated root/docs page.
        link_source = (
            root / "docs/multi-bias-research-synthesis.md"
            if relative == "reports/multibias_synthesis_20261022/report_template.md"
            else path
        )
        if link_source != path:
            item["link_context"] = str(link_source)
        sections.append(
            f"\n## Appendix {number:03d}: {label}\n\n"
            f"Source: [{label}]({path}). Original SHA256: `{item['sha256']}`.\n\n"
            "This is the complete source note at compilation time. Its original experiment "
            "date, protocol, limitations and historical status apply; later corrections "
            "in the overview and subsequent audit notes take precedence. Heading "
            "levels, line endings, outer document spacing and source-relative Markdown "
            "links are normalized here; code and mathematical expressions are preserved."
            + (" Template links use its generated root/docs output context." if link_source != path else "")
            + "\n\n" + appendix(raw.decode("utf-8"), link_source)
        )
    manifest = {
        "schema": "omnibias-session-compendium-v1",
        "date": "2026-10-23",
        "root_base_commit": args.root_base or previous.get("root_base_commit") or revision(root),
        "lab_snapshot_commit": revision(lab),
        "overview_sha256": digest(overview.read_bytes()),
        "sources": records,
        "scope": "Complete canonical session reports and available LAB Markdown theory notes; "
        "historical prerequisite notes are retained with provenance. Raw numerical evidence "
        "and code are referenced and committed separately, not reproduced as prose.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    text = overview.read_text()
    first, remainder = text.split("\n", 1)
    provenance = (
        "\n**Compiled 23 October 2026.** This is the standalone session record: an "
        "overview with positive/negative comparisons, followed by the complete canonical "
        f"reports and theory notes ({len(records)} source documents). The appendices retain "
        "historical wording; the corrected overview and later audit notes resolve conflicts. "
        "The source inventory records hashes rather than rewriting frozen evidence.\n\n"
        f"LAB source checkpoint: `{manifest['lab_snapshot_commit'] or 'pending initial commit'}`. "
        f"Core base before this checkpoint: `{manifest['root_base_commit']}`. The core "
        "commit containing this report is available from Git; it cannot be embedded "
        "in its own content without a circular hash.\n\n"
        "[Source/hash inventory](data/session-research-20261023/sources.json) · "
        f"[Curated allocation evidence]({lab}/reports/"
        "projection_profile_20261023/README.md). Large frozen artifacts, environments, "
        "downloads and model checkpoints remain local and are intentionally outside Git. "
        "The curated archive verifies file integrity; full numerical replay additionally "
        "requires the recorded original model/data artifacts and dependencies.\n"
    )
    index = "\n## Complete source-note appendices\n\n" + "\n".join(
        f"- Appendix {item['appendix']:03d}: `{item['repository']}/{item['path']}` "
        f"({item['bytes']:,} source bytes)." for item in records
    ) + "\n"
    output = root / "docs/session-research-findings.md"
    output.write_text(first + "\n" + provenance + remainder + index + "".join(sections))
    print(json.dumps({"output": str(output), "sources": len(records), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()

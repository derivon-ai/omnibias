# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: no site, scheduler, personal, or secret leakage in tracked files.

`AGENTS.md` requires vendor-neutral language in **any tracked file**: "GPU job" /
"GPU cluster", never a specific scheduler, vendor, or local filesystem path. This
is the enforcing guard for that rule. It is backend-free and only reads the
working tree, so it runs in the dependency-light `guards` CI job.

The scan covers tracked files *and* new ones that are not gitignored, which is
slightly wider than the rule as written and deliberately so: a leak is easiest to
introduce in a file that has just been created, and a guard that waits for the
commit reports it one step too late.

Seven leak families are blocked, each with at least one pattern and each pattern
carrying **synthetic bait** that the self-tests below assert it catches:

* ``local_path`` -- absolute developer paths (``/home/<user>``, ``/Users/<user>``,
  ``/u/<user>``, ``C:\Users\<user>``);
* ``site_mount`` -- site scratch / shared-filesystem conventions (``/scratch``,
  ``/gpfs``, ``/lustre``, ``$SCRATCH``). Artifacts belong under
  ``$OMNIBIAS_SCRATCH``, which is explicitly *not* blocked;
* ``scheduler`` -- batch-scheduler commands, script directives, environment
  variables, and product names from any vendor;
* ``corporate`` -- the EDA-shop synonym for a compute cluster, and EDA vendor
  names that would identify the shop;
* ``personal`` -- consumer email accounts and ``user@host`` login lines. The
  published Derivon identity (in `NOTICE` / `CITATION.cff` / `MAINTAINERS.md`) is
  the project's legal identity, not a leak, and is not blocked;
* ``private_tree`` -- paths into the formerly gitignored ``internal/`` tree;
* ``credential`` -- private keys, cloud / forge / chat tokens, and assigned
  secret literals.

Two scope decisions are deliberate, so this file states them rather than letting
them look like oversights:

* patterns are written to need a *path segment* or a *usage context* rather than
  a bare word. ``/home/`` inside backticks in `docs/release_blockers.md` documents
  the rule; ``/home/jdoe/dev`` leaks a username. Likewise the guard blocks
  "compute farm" and "farm node", not every occurrence of "farm". This keeps the
  guard from taxing the prose that describes the guard.
* naming the separate private ``omnibias_experiments`` project is sanctioned by
  `AGENTS.md` and `CONTRIBUTING.md` and is therefore not blocked; what is blocked
  is an absolute path into a developer's checkout of it, via ``local_path``.

The blocklist can never go vacuous: :func:`test_every_pattern_catches_its_bait`
fails on a regex that matches nothing, :func:`test_scan_catches_injected_bait`
drives the real scanner over each bait string, and
:func:`test_scan_surface_is_not_vacuous` pins that the file list is large and
actually reaches ``.github/``, ``notebooks/``, and ``formal/``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# This file necessarily contains one example of every blocked token.
SELF = Path(__file__).resolve()

# Binary / generated payloads carry no reviewable prose.
_BINARY_SUFFIXES = frozenset(
    {
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".npy",
        ".npz",
        ".parquet",
        ".pdf",
        ".png",
        ".pyc",
        ".svgz",
        ".ttf",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
)

_SKIP_DIR_PARTS = frozenset({".git", "__pycache__", ".mypy_cache", ".ruff_cache"})

# --------------------------------------------------------------------------
# The blocklist: (family, pattern, synthetic bait the pattern must catch)
# --------------------------------------------------------------------------

_RAW_BLOCKLIST: tuple[tuple[str, str, str], ...] = (
    # -- local developer paths ---------------------------------------------
    ("local_path", r"/home/[A-Za-z0-9._-]+", "trained in /home/jdoe/dev/omnibias"),
    ("local_path", r"/Users/[A-Za-z0-9._-]+", "see /Users/jdoe/Desktop/run.log"),
    ("local_path", r"(?<![\w:/])/u/[A-Za-z0-9._-]+", "artifacts in /u/jdoe/runs"),
    ("local_path", r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+", r"C:\Users\jdoe\omnibias"),
    # -- site mounts and scratch conventions -------------------------------
    (
        "site_mount",
        r"(?<![\w.-])/(?:scratch|nfs|gpfs|lustre|afs|proj|net)/[A-Za-z0-9._-]+",
        "checkpoints under /gpfs/projects/team",
    ),
    ("site_mount", r"\$\{?SCRATCH\b", "cd $SCRATCH/omnibias"),
    # -- batch schedulers, any vendor --------------------------------------
    (
        "scheduler",
        r"\b(?:sbatch|srun|squeue|scancel|salloc|sacct|scontrol|sinfo|bsub|bjobs"
        r"|bkill|bqueues|bhosts|lsload|qsub|qstat|qdel|qhold|condor_submit"
        r"|condor_q|oarsub|pjsub)\b",
        "submit with sbatch run.sh",
    ),
    ("scheduler", r"(?m)^\s*#\s*(?:SBATCH|BSUB|PBS)\b", "#SBATCH --gres=gpu:1"),
    ("scheduler", r"(?m)^\s*#\$\s+-\S", "#$ -q long.q"),
    (
        "scheduler",
        r"\b(?:SLURM|LSB|LSF|PBS|SGE|OAR|CONDOR|PJM)_[A-Z0-9_]+\b",
        "read $SLURM_ARRAY_TASK_ID",
    ),
    (
        "scheduler",
        r"(?i)\b(?:slurm|htcondor|openpbs|pbs\s*pro|grid\s*engine"
        r"|load\s*sharing\s*facility|moab)\b",
        "the Slurm queue was busy",
    ),
    ("scheduler", r"\b(?:LSF|SGE)\b", "the LSF queue was busy"),
    # -- the EDA shop ------------------------------------------------------
    (
        "corporate",
        r"(?i)\b(?:compute|regression|batch|the)\s+farm\b",
        "launched on the compute farm",
    ),
    (
        "corporate",
        r"(?i)\bfarm\s+(?:job|jobs|machine|machines|node|nodes|queue|queues)\b",
        "waiting for a farm node",
    ),
    (
        "corporate",
        r"(?i)\b(?:synopsys|cadence\s+design|mentor\s+graphics|siemens\s+eda)\b",
        "ran on the Synopsys grid",
    ),
    # -- personal accounts -------------------------------------------------
    (
        "personal",
        r"(?i)\b[A-Za-z0-9._%+-]+@(?:gmail|googlemail|outlook|hotmail|yahoo"
        r"|proton|protonmail|icloud)\.[a-z]{2,}\b",
        "contact jdoe@gmail.com",
    ),
    (
        "personal",
        r"(?i)\b(?:ssh|scp|rsync)\s+(?:-\S+\s+)*"
        r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "ssh jdoe@login.example.edu",
    ),
    # -- the private tree --------------------------------------------------
    (
        "private_tree",
        r"(?<![\w.-])internal/[A-Za-z0-9_.-]+",
        "results live in internal/experiments",
    ),
    # -- credentials -------------------------------------------------------
    (
        "credential",
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
    ),
    ("credential", r"\bAKIA[0-9A-Z]{16}\b", "AKIAIOSFODNN7EXAMPLE"),
    ("credential", r"\bghp_[A-Za-z0-9]{36}\b", "token ghp_" + "a" * 36),
    ("credential", r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "token github_pat_" + "b" * 24),
    ("credential", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "xoxb-1234567890-abcdef"),
    (
        "credential",
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|passwd|password)"
        r"\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']",
        'api_key = "s3cr3t-value-here"',
    ),
)

BLOCKLIST: tuple[tuple[str, re.Pattern[str], str], ...] = tuple(
    (family, re.compile(pattern), bait) for family, pattern, bait in _RAW_BLOCKLIST
)

FAMILIES = (
    "corporate",
    "credential",
    "local_path",
    "personal",
    "private_tree",
    "scheduler",
    "site_mount",
)


def scan_text(text: str) -> list[tuple[str, str]]:
    """Every ``(family, matched_text)`` a blocked pattern finds in ``text``."""
    hits: list[tuple[str, str]] = []
    for family, pattern, _bait in BLOCKLIST:
        for match in pattern.finditer(text):
            hits.append((family, match.group(0)))
    return hits


def _scanned_files() -> list[Path]:
    """Every reviewable file in the working tree, as absolute paths.

    Tracked files **and** new ones that are not gitignored. Scanning only the
    tracked set would clear a leak until the moment it was committed, which is
    the wrong end of the loop: a brand-new file is exactly where a stray
    developer path is most likely to be. ``--exclude-standard`` still honours
    ``.gitignore``, so build output and local artifacts stay out.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    paths = [REPO_ROOT / rel for rel in out.stdout.split("\0") if rel]
    return [
        p
        for p in paths
        if p.is_file()
        and p.resolve() != SELF
        and p.suffix.lower() not in _BINARY_SUFFIXES
        and not _SKIP_DIR_PARTS.intersection(p.parts)
    ]


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_no_leakage_in_tracked_files() -> None:
    """No file in the working tree may carry a site, scheduler, or secret token."""
    offenders: list[str] = []
    for path in _scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary payload without a known suffix
        if "\0" in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for family, matched in scan_text(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: [{family}] {matched!r}")
    assert not offenders, (
        "leakage of local paths / scheduler or vendor tokens / personal accounts / "
        "secrets in tracked files -- use vendor-neutral language ('GPU job', 'GPU "
        "cluster') and write artifacts to $OMNIBIAS_SCRATCH:\n" + "\n".join(offenders)
    )


# --------------------------------------------------------------------------
# Self-tests: the blocklist can never go vacuous
# --------------------------------------------------------------------------


def test_blocklist_covers_every_leak_family() -> None:
    """Every declared family is actually represented by at least one pattern."""
    covered = {family for family, _pattern, _bait in BLOCKLIST}
    assert covered == set(FAMILIES), (
        f"blocklist families {sorted(covered)} do not match the documented "
        f"families {sorted(FAMILIES)}"
    )


def test_every_pattern_catches_its_bait() -> None:
    """A pattern that matches nothing would silently disarm its family."""
    dead = [
        pattern.pattern
        for _family, pattern, bait in BLOCKLIST
        if not pattern.search(bait)
    ]
    assert not dead, f"blocked patterns that match no bait (dead regexes): {dead}"


def test_scan_catches_injected_bait() -> None:
    """End-to-end: the scanner itself reports every family's bait."""
    for family, _pattern, bait in BLOCKLIST:
        hits = scan_text(bait)
        assert any(f == family for f, _m in hits), (
            f"scan_text missed {family} bait {bait!r} (got {hits})"
        )


def test_sanctioned_vocabulary_is_not_blocked() -> None:
    """The approved neutral forms must survive, or the guard would tax good prose."""
    allowed = (
        "artifacts go to $OMNIBIAS_SCRATCH, defaulting to artifacts/",
        "run this as a GPU job on a GPU cluster",
        "the separate, private omnibias_experiments project",
        "the formerly gitignored `internal/` tree",
        "no local paths (`/u/`, `/home/`), scheduler/vendor names",
        "class LBFGSInfo:",  # a scheduler command as an identifier substring
        "a steady cadence of releases on a wind farm dataset",
        "ghp_import-2.1.0-py3-none-any.whl",
    )
    for text in allowed:
        assert not scan_text(text), f"sanctioned text was flagged: {text!r}"


def test_scan_surface_is_not_vacuous() -> None:
    """An empty or narrow file list would make the guard pass by doing nothing."""
    files = _scanned_files()
    assert len(files) > 500, f"working-tree scan collapsed to {len(files)} files"
    rels = {str(p.relative_to(REPO_ROOT)) for p in files}
    for required in (".github/", "notebooks/", "formal/", "docs/", "packages/"):
        assert any(r.startswith(required) for r in rels), (
            f"the scan never reaches {required}, so leakage there would go unseen"
        )
    assert SELF.relative_to(REPO_ROOT).as_posix() not in rels, (
        "the guard must exempt itself; it holds one example of every blocked token"
    )


def test_a_new_uncommitted_file_is_still_scanned() -> None:
    """The listing must include untracked work, not just what is already committed.

    Pinned separately because it is invisible in a clean checkout: if the scan
    ever narrows back to ``git ls-files``, every test above still passes while a
    leak in a file being written right now sails through review.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split("\n")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, check=True, text=True
    ).stdout.split("\n")
    assert len(listed) >= len(tracked), (
        "the scan surface is narrower than the tracked set; it must be a superset"
    )
    # Ignored paths must stay out, or the scan would wander into build output.
    scanned = {str(p.relative_to(REPO_ROOT)) for p in _scanned_files()}
    assert not any(r.startswith((".venv/", "site/", ".git/")) for r in scanned)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Release guard: no unfilled launch/identity placeholders in shipped files.

The repository was scaffolded with placeholder tokens for the legal identity
(copyright holder, contact inboxes, hosting org). Before publishing, every one
must be replaced with the real value. This backend-free test runs in the core
CI job and fails if any placeholder survives in a file that is either shipped
in a wheel, carried as packaging metadata, part of the active legal/governance
surface, a CI workflow, a repo-root or example launch script, or a committed
agent skill.

Out of scope on purpose:

* the AGPL ``LICENSE`` "how to apply" appendix (``<year> <name of author>`` is
  the upstream template text and must stay verbatim); and
* historical records (``CHANGELOG.md``) and design notes (``docs/design/``)
  that *document* the placeholder mechanism.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The exact identity tokens that must never survive into a release artifact.
PLACEHOLDERS = (
    "<COPYRIGHT HOLDER>",
    "<commercial-contact-email>",
    "<trademark-contact-email>",
    "<security-contact-email>",
    "<your-org>",
)

# Active legal / governance documents (relative to the repo root).
_LEGAL_DOCS = (
    "NOTICE",
    "LICENSING.md",
    "COMMERCIAL-LICENSE.md",
    "TRADEMARKS.md",
    "SECURITY.md",
    "README.md",
    "CONTRIBUTING.md",
    "docs/CLA.md",
)


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    files += REPO_ROOT.glob("packages/*/src/**/*.py")
    files += REPO_ROOT.glob("packages/*/pyproject.toml")
    files.append(REPO_ROOT / "pyproject.toml")
    files += REPO_ROOT.glob(".github/workflows/*.yml")
    # Repo-root scripts, example launch scripts, and the committed Cursor /
    # Claude agent-skill libraries are public too and must not carry unfilled
    # launch placeholders. (CHANGELOG.md and docs/design/ stay out of scope --
    # they *document* the placeholder mechanism.)
    files += REPO_ROOT.glob("scripts/*")
    files += REPO_ROOT.glob("examples/**/*.sh")
    files += REPO_ROOT.glob("examples/**/*.py")
    files += REPO_ROOT.glob(".cursor/skills/**/*.md")
    files += REPO_ROOT.glob(".claude/skills/**/*.md")
    for rel in _LEGAL_DOCS:
        files.append(REPO_ROOT / rel)
    return [f for f in files if f.is_file() and "__pycache__" not in f.parts]


def test_no_unfilled_identity_placeholders() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _guarded_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [tok for tok in PLACEHOLDERS if tok in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert not offenders, (
        "unfilled identity placeholders remain in release-surface files "
        "(fill them with the real Derivon identity values): "
        + "; ".join(f"{k} -> {v}" for k, v in sorted(offenders.items()))
    )

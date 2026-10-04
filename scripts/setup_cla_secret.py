# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Configure CLA authentication without displaying or copying the credential.

Run with the docs environment (PyYAML) and an authenticated GitHub CLI.
The local YAML contains a token-file path, never the token itself.
Default mode checks access; --apply uploads PERSONAL_ACCESS_TOKEN via stdin.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = "PERSONAL_ACCESS_TOKEN"


class SetupError(Exception):
    """A fixed, credential-free diagnostic that is safe to print."""


@dataclass(frozen=True)
class Settings:
    token_file: Path
    repository: str
    signatures_repository: str
    branch: str


def load_settings(path: Path) -> Settings:
    """Require ignored local configuration and reject token-valued fields."""
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        raise SetupError("Place the configuration inside .local/github/.") from None
    if relative.parts[:2] != (".local", "github"):
        raise SetupError("Place the configuration inside .local/github/.")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=ROOT, capture_output=True, check=False,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(relative)],
        cwd=ROOT, capture_output=True, check=False,
    )
    if tracked.returncode == 0 or ignored.returncode != 0:
        raise SetupError("Local configuration must be untracked and git-ignored.")
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        raise SetupError("Could not read the local setup configuration.") from None
    fields = {"token_file", "repository", "signatures_repository", "branch"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise SetupError("Use only token_file, repository, signatures_repository and branch.")
    if not all(isinstance(value, str) and value for value in raw.values()):
        raise SetupError("All configuration values must be nonempty strings.")
    if any(re.fullmatch(r"[\w.-]+/[\w.-]+", raw[key]) is None
           for key in ("repository", "signatures_repository")):
        raise SetupError("Repository names must have the form owner/repository.")
    token_file = Path(raw["token_file"]).expanduser()
    if not token_file.is_absolute():
        token_file = path.parent / token_file
    token_file = token_file.resolve()
    if token_file.is_relative_to(ROOT):
        raise SetupError("Keep the credential file outside the repository checkout.")
    return Settings(token_file, raw["repository"], raw["signatures_repository"], raw["branch"])


def read_token(path: Path) -> str:
    """Read internally; never put file contents in diagnostics or exceptions."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(4097)
        token = raw.decode("ascii").strip()
    except (OSError, UnicodeError):
        raise SetupError("Could not read a token from the configured credential file.") from None
    if len(raw) > 4096 or re.fullmatch(r"(?:github_pat_|ghp_)[A-Za-z0-9_]+", token) is None:
        raise SetupError("Credential file must contain exactly one GitHub personal access token.")
    return token


def github_get(endpoint: str, token: str) -> dict[str, Any]:
    """Read GitHub metadata; discard response bodies on failure."""
    request = Request(
        "https://api.github.com/" + endpoint,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
    except HTTPError as exc:
        hints = {
            401: "Credential rejected; check expiration or revocation.",
            403: "Access denied; check organization approval, token permissions and repository policy.",
            404: "Resource is absent or inaccessible to this token; check repository selection and branch.",
        }
        raise SetupError(hints.get(exc.code, f"GitHub metadata check failed (HTTP {exc.code}).")) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise SetupError("GitHub metadata check could not complete.") from None
    if not isinstance(result, dict):
        raise SetupError("GitHub returned an unexpected metadata response.")
    return result


def check_access(settings: Settings, token: str) -> None:
    repo = github_get("repos/" + settings.signatures_repository, token)
    if repo.get("private") is not True:
        raise SetupError("The configured CLA signature repository must be private.")
    github_get("repos/" + settings.signatures_repository + "/branches/"
               + quote(settings.branch, safe=""), token)


def upload_secret(settings: Settings, token: str) -> None:
    """Use the current CLI login to encrypt/upload; the PAT travels on stdin."""
    environment = os.environ.copy()
    environment.pop("GH_DEBUG", None)
    environment["GH_HOST"] = "github.com"
    try:
        result = subprocess.run(
            ["gh", "secret", "set", SECRET_NAME, "--repo", settings.repository],
            input=token.encode(), capture_output=True, env=environment, check=False, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise SetupError("Secret upload could not complete; verify the GitHub CLI login.") from None
    if result.returncode:
        raise SetupError("Secret upload failed; the CLI login needs repository secret-management access.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".local/github/cla.yml")
    parser.add_argument("--apply", action="store_true", help="Upload the secret after access checks pass.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        token = read_token(settings.token_file)
        check_access(settings, token)
        print("Private signature repository and configured branch are accessible.")
        print("The token requires Contents: write; the CLA workflow will exercise signature persistence.")
        if args.apply:
            upload_secret(settings, token)
            print("PERSONAL_ACCESS_TOKEN uploaded. Credential contents were not displayed or copied.")
        else:
            print("Check complete. Use --apply to upload the repository secret.")
        return 0
    except SetupError as exc:
        print(f"CLA setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

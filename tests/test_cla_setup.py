# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CLA setup must keep credentials off argv, logs and tracked configuration."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_cla_secret", ROOT / "scripts/setup_cla_secret.py")
assert spec and spec.loader
setup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = setup
spec.loader.exec_module(setup)
TOKEN = "github_pat_dummy_credential_for_offline_tests"


def settings(tmp_path):
    return setup.Settings(tmp_path / "credential", "example/project", "example/signatures", "main")


def test_upload_sends_credential_only_on_stdin(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setenv("GH_DEBUG", "api")
    setup.upload_secret(settings(tmp_path), TOKEN)
    command, options = calls[0]
    assert TOKEN not in str(command)
    assert options["input"] == TOKEN.encode()
    assert "GH_DEBUG" not in options["env"]
    assert options["capture_output"] is True


@pytest.mark.parametrize("failure", ["exit", "timeout", "os"])
def test_upload_errors_cannot_echo_credential(monkeypatch, tmp_path, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60, output=TOKEN, stderr=TOKEN)
        if failure == "os":
            raise OSError(TOKEN)
        return subprocess.CompletedProcess(command, 1, TOKEN.encode(), TOKEN.encode())

    monkeypatch.setattr(setup.subprocess, "run", run)
    with pytest.raises(setup.SetupError) as error:
        setup.upload_secret(settings(tmp_path), TOKEN)
    assert TOKEN not in str(error.value)


def test_http_errors_discard_sensitive_response(monkeypatch):
    def open_request(*args, **kwargs):
        raise HTTPError("https://api.github.com/", 404, TOKEN, {}, io.BytesIO(TOKEN.encode()))

    monkeypatch.setattr(setup, "urlopen", open_request)
    with pytest.raises(setup.SetupError, match="absent or inaccessible") as error:
        setup.github_get("repos/example/signatures", TOKEN)
    assert TOKEN not in str(error.value)


def test_access_requires_private_repository_and_requested_branch(monkeypatch, tmp_path):
    endpoints = []

    def get(endpoint, token):
        endpoints.append(endpoint)
        assert token == TOKEN
        return {"private": True}

    monkeypatch.setattr(setup, "github_get", get)
    setup.check_access(settings(tmp_path), TOKEN)
    assert endpoints == ["repos/example/signatures", "repos/example/signatures/branches/main"]
    monkeypatch.setattr(setup, "github_get", lambda *_: {"private": False})
    with pytest.raises(setup.SetupError, match="must be private"):
        setup.check_access(settings(tmp_path), TOKEN)


def test_local_settings_require_gitignore_and_reject_secret_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    path = tmp_path / ".local/github/cla.yml"
    path.parent.mkdir(parents=True)
    path.write_text("token_file: ../../../credential\nrepository: example/project\n"
                    "signatures_repository: example/signatures\nbranch: main\n")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, int(command[1] == "ls-files"))

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.load_settings(path).token_file == tmp_path.parent / "credential"
    external = path.read_text()
    path.write_text(external.replace("../../../credential", "./credential"))
    with pytest.raises(setup.SetupError, match="outside the repository"):
        setup.load_settings(path)
    path.write_text(external)
    path.write_text(path.read_text() + "token: " + TOKEN)
    with pytest.raises(setup.SetupError) as error:
        setup.load_settings(path)
    assert TOKEN not in str(error.value)
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **_: subprocess.CompletedProcess(command, 0))
    with pytest.raises(setup.SetupError, match="untracked and git-ignored"):
        setup.load_settings(path)


def test_main_check_mode_never_uploads(monkeypatch, tmp_path, capsys):
    local = settings(tmp_path)
    local.token_file.write_text(TOKEN + "\n")
    monkeypatch.setattr(setup, "load_settings", lambda _: local)
    monkeypatch.setattr(setup, "check_access", lambda *_: None)
    monkeypatch.setattr(setup, "upload_secret", lambda *_: pytest.fail("check-only mode uploaded"))
    assert setup.main([]) == 0
    assert TOKEN not in capsys.readouterr().out


def test_main_failure_is_sanitized_and_prevents_upload(monkeypatch, tmp_path, capsys):
    local = settings(tmp_path)
    local.token_file.write_text(TOKEN + "\n" + TOKEN)
    monkeypatch.setattr(setup, "load_settings", lambda _: local)
    monkeypatch.setattr(setup, "upload_secret", lambda *_: pytest.fail("invalid credential uploaded"))
    assert setup.main(["--apply"]) == 1
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_local_config_directory_is_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".local/github/cla.yml"], cwd=ROOT, check=False,
    )
    assert result.returncode == 0

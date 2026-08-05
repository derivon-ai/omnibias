# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Execute every fenced ``python`` block in the documentation.

Docs rot silently: a renamed symbol or a changed signature keeps rendering
perfectly in mkdocs while every reader who copies the snippet hits a traceback.
This module closes that hole by *running* the documented code -- the README, the
root guides, the whole ``docs/`` tree (cookbook + handbook + API pages), and every
package README.

**Blocks are executed by default.** A new fenced ``python`` block is therefore
covered the moment it is written, and every opt-out is an explicit, reviewable
line in the markdown. Opt out with an HTML comment directly above the fence (it
renders as nothing):

.. code-block:: markdown

    <!-- docs-test: signature -->      an API signature, not runnable code
    <!-- docs-test: skip reason="..." -->   not executable; reason is mandatory
    <!-- docs-test: slow -->           real code, but too heavy for per-PR CI
    <!-- docs-test: raises=ValueError -->   the block is *meant* to raise

A whole document opts out with a single directive placed anywhere in it:

.. code-block:: markdown

    <!-- docs-test: file-skip reason="design doc describing a planned API" -->

Execution model
---------------

One test per document, with the document's blocks executed **in order sharing a
single namespace**, because the docs are written as narratives: the handbook
fits a ``field`` in one block and reads jets off it three blocks later. A failing
block therefore stops that document (the chain is broken anyway) and is reported
with its ``path:line``.

``slow`` is a *document*-level property for the same reason -- skipping one block
of a chain would only produce cascading ``NameError``s in the rest -- so any
``slow`` block marks its whole document ``slow``, which the repo-wide
``addopts = -m 'not slow'`` then excludes from the default run.

A missing **third-party** import (matplotlib, pandas, ...) skips the document:
the snippet was not verified, and the skip says so. A missing **omnibias**
module is a hard failure, since all 42 distributions are installed in CI -- that
is exactly the "documented import no longer exists" bug this module exists to
catch.
"""

from __future__ import annotations

import gc
import os
import re
import signal
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

# Snippets that plot must not try to open a window, and the Keras snippets need a
# backend chosen before ``keras`` is first imported.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KERAS_BACKEND", "torch")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Docs legitimately import the repo's own ``examples`` package (see the RSA and
# boolean-limits cookbooks), and snippets run from a scratch directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Import roots that ship *in this repository*. A missing module under one of
#: these is a documentation bug (the symbol was renamed or removed), never an
#: uninstalled optional dependency -- but only where every distribution is
#: actually installed. See :data:`STRICT_IMPORTS`.
REPO_IMPORT_ROOTS = frozenset({"omnibias", "examples"})

#: Whether a missing ``omnibias.*`` module is a failure rather than a skip.
#:
#: The dedicated CI job installs all 42 distributions and sets this, so a
#: documented import that no longer exists is caught. A plain ``uv run pytest``
#: installs only the four workspace members, so there the same import is an
#: honest "not verified here" skip instead of a false alarm.
STRICT_IMPORTS = os.environ.get("OMNIBIAS_DOCS_SNIPPETS_STRICT", "") == "1"

#: Where documented Python lives. ``docs/**`` covers cookbook, handbook, api.
DOC_GLOBS = ("*.md", "docs/**/*.md", "packages/*/README.md")

#: Built output and virtualenvs are not source documentation.
EXCLUDED_PARTS = frozenset({"site", ".venv", "node_modules", "build", ".tox"})

#: The changelog is a historical record: its snippets describe the API as it was
#: at each past release, so executing them against today's code is meaningless.
EXCLUDED_FILES = frozenset({"CHANGELOG.md"})

_DIRECTIVE_RE = re.compile(r"<!--\s*docs-test:\s*(?P<body>.*?)\s*-->")
_FENCE_OPEN_RE = re.compile(r"^(?P<indent>\s*)```+\s*(?:\{\.)?python\b")
_FENCE_CLOSE_RE = re.compile(r"^\s*```+\s*$")
_REASON_RE = re.compile(r'reason="(?P<reason>[^"]*)"')
_RAISES_RE = re.compile(r"raises=(?P<exc>[A-Za-z_][A-Za-z0-9_.]*)")

MODES = frozenset({"exec", "signature", "skip", "file-skip"})
KEYWORDS = MODES | {"slow"}

#: Per-block wall-clock budget. An unmarked block that blows this budget is a
#: *failure*, not a hang: documentation is meant to be copy-pasteable in seconds,
#: and anything heavier belongs behind ``<!-- docs-test: slow -->``. Without this
#: a single 8^3x16 lattice Monte-Carlo snippet would stall the whole CI job.
BLOCK_TIMEOUT_SECONDS = float(os.environ.get("OMNIBIAS_DOCS_SNIPPET_TIMEOUT", "120"))


class SnippetTimeout(Exception):
    """Raised when a block outruns :data:`BLOCK_TIMEOUT_SECONDS`."""


_WARMED_UP = False
_REGISTRY_SNAPSHOTS: dict[str, dict[str, object]] = {}
_TORCH_DEFAULT_DTYPE = None


def _snapshot_activation_registries() -> None:
    """Capture built-in activation registries so docs snippets cannot leak."""
    global _TORCH_DEFAULT_DTYPE
    try:
        import torch
        from omnibias.torch.activations import registry as torch_reg

        _REGISTRY_SNAPSHOTS["torch"] = dict(torch_reg._REGISTRY)
        _TORCH_DEFAULT_DTYPE = torch.get_default_dtype()
    except Exception:  # noqa: BLE001 - optional backend
        pass
    try:
        from omnibias.jax import activations as jax_act

        _REGISTRY_SNAPSHOTS["jax"] = dict(jax_act._REGISTRY)
    except Exception:  # noqa: BLE001
        pass
    try:
        from omnibias.keras.activations import registry as keras_reg

        _REGISTRY_SNAPSHOTS["keras"] = dict(keras_reg._REGISTRY)
    except Exception:  # noqa: BLE001
        pass


def _restore_activation_registries() -> None:
    """Undo user registrations and default-dtype changes left by a document."""
    if "torch" in _REGISTRY_SNAPSHOTS:
        try:
            import torch
            from omnibias.torch.activations import registry as torch_reg

            torch_reg._REGISTRY.clear()
            torch_reg._REGISTRY.update(_REGISTRY_SNAPSHOTS["torch"])
            if _TORCH_DEFAULT_DTYPE is not None:
                torch.set_default_dtype(_TORCH_DEFAULT_DTYPE)
        except Exception:  # noqa: BLE001
            pass
    if "jax" in _REGISTRY_SNAPSHOTS:
        try:
            from omnibias.jax import activations as jax_act

            jax_act._REGISTRY.clear()
            jax_act._REGISTRY.update(_REGISTRY_SNAPSHOTS["jax"])
        except Exception:  # noqa: BLE001
            pass
    if "keras" in _REGISTRY_SNAPSHOTS:
        try:
            from omnibias.keras.activations import registry as keras_reg

            keras_reg._REGISTRY.clear()
            keras_reg._REGISTRY.update(_REGISTRY_SNAPSHOTS["keras"])
        except Exception:  # noqa: BLE001
            pass


def _warm_up() -> None:
    """Import the heavy backends once, *outside* any block time limit.

    A cold ``import torch`` can take tens of seconds on a network filesystem. If
    the timeout fires part-way through it, ``sys.modules["torch"]`` is left
    half-built and every later snippet dies with a bogus "partially initialized
    module" error. Paying the import cost up front keeps the per-block budget a
    measure of the *snippet*, not of the machine's disk.
    """
    global _WARMED_UP
    if _WARMED_UP:
        return
    _WARMED_UP = True
    for module in ("numpy", "torch", "jax", "jax.numpy", "keras"):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - a missing backend is handled per block
            pass
    # Force activation packages to register builtins, then snapshot.
    for module in (
        "omnibias.torch.activations",
        "omnibias.jax.activations",
        "omnibias.keras.activations",
    ):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            pass
    _snapshot_activation_registries()


def _release(namespace: dict[str, object]) -> None:
    """Drop a finished document's objects.

    The suite runs ~90 documents in one process, and a document's namespace pins
    whole networks, autograd graphs, and jitted JAX executables. Without this the
    resident set grows monotonically until the run is killed part-way, which looks
    like a hang rather than an out-of-memory.
    """
    namespace.clear()
    _restore_activation_registries()
    jax = sys.modules.get("jax")
    if jax is not None:  # jitted executables live in a module-level cache
        try:
            jax.clear_caches()
        except Exception:  # noqa: BLE001 - best effort; never fail a doc over it
            pass
    gc.collect()


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Bound a block's runtime with SIGALRM where the platform allows it."""
    usable = (
        seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def _fire(signum: int, frame: object) -> None:
        raise SnippetTimeout(f"exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class Directive:
    """A parsed ``docs-test`` comment."""

    mode: str
    slow: bool
    reason: str
    raises: str | None
    line: int
    raw: str


@dataclass(frozen=True)
class Block:
    """One fenced ``python`` block plus the directive that governs it."""

    path: Path
    line: int
    code: str
    mode: str
    slow: bool
    reason: str
    raises: str | None

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT).as_posix()}:{self.line}"


def _parse_directive(body: str, line: int) -> Directive:
    tokens = [t for t in body.split() if "=" not in t]
    mode = "exec"
    for token in tokens:
        if token in MODES:
            mode = token
            break
    reason_match = _REASON_RE.search(body)
    raises_match = _RAISES_RE.search(body)
    return Directive(
        mode=mode,
        slow="slow" in tokens,
        reason=reason_match.group("reason") if reason_match else "",
        raises=raises_match.group("exc") if raises_match else None,
        line=line,
        raw=body,
    )


def _parse_document(path: Path) -> tuple[list[Block], list[Directive]]:
    """Return the document's python blocks and every directive it declares."""
    lines = path.read_text(encoding="utf-8").splitlines()
    directives: list[Directive] = []
    blocks: list[Block] = []
    pending: Directive | None = None
    file_skip: Directive | None = None

    index = 0
    while index < len(lines):
        line = lines[index]

        if match := _DIRECTIVE_RE.search(line):
            directive = _parse_directive(match.group("body"), index + 1)
            directives.append(directive)
            if directive.mode == "file-skip":
                file_skip = directive
            else:
                pending = directive
            index += 1
            continue

        if open_match := _FENCE_OPEN_RE.match(line):
            indent = len(open_match.group("indent"))
            start = index + 1
            body: list[str] = []
            index += 1
            while index < len(lines) and not _FENCE_CLOSE_RE.match(lines[index]):
                body.append(lines[index][indent:] if lines[index][:indent].isspace() else lines[index])
                index += 1
            index += 1  # step over the closing fence
            directive = pending or Directive("exec", False, "", None, start, "")
            blocks.append(
                Block(
                    path=path,
                    line=start,
                    code="\n".join(body) + "\n",
                    mode=directive.mode,
                    slow=directive.slow,
                    reason=directive.reason,
                    raises=directive.raises,
                )
            )
            pending = None
            continue

        # Any other non-blank line ends a directive's reach, so a stray comment
        # far above a fence cannot silently disable it.
        if line.strip():
            pending = None
        index += 1

    if file_skip is not None:
        blocks = [
            Block(
                path=b.path,
                line=b.line,
                code=b.code,
                mode="file-skip",
                slow=b.slow,
                reason=file_skip.reason,
                raises=b.raises,
            )
            for b in blocks
        ]
    return blocks, directives


def _doc_files() -> list[Path]:
    found: set[Path] = set()
    for pattern in DOC_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            relative = path.relative_to(REPO_ROOT)
            if EXCLUDED_PARTS & set(relative.parts) or path.name in EXCLUDED_FILES:
                continue
            found.add(path)
    return sorted(found)


def _collect() -> list[tuple[Path, list[Block], list[Directive]]]:
    documents = []
    for path in _doc_files():
        blocks, directives = _parse_document(path)
        if blocks or directives:
            documents.append((path, blocks, directives))
    return documents


DOCUMENTS = _collect()
ALL_BLOCKS = [block for _, blocks, _ in DOCUMENTS for block in blocks]
ALL_DIRECTIVES = [d for _, _, directives in DOCUMENTS for d in directives]


def _execute(block: Block, namespace: dict[str, object]) -> tuple[str, str]:
    """Run one block in ``namespace``; return ``(outcome, detail)``."""
    try:
        code = compile(block.code, block.where, "exec")
    except SyntaxError as exc:
        return "failed", (
            f"{block.where} is not valid Python ({exc.msg}); if it is an API "
            "signature or pseudo-code, mark it <!-- docs-test: signature -->"
        )
    try:
        with warnings.catch_warnings(), _time_limit(BLOCK_TIMEOUT_SECONDS):
            # The repo turns warnings into errors; a snippet emitting a
            # DeprecationWarning is still a working snippet.
            warnings.simplefilter("ignore")
            exec(code, namespace, namespace)  # noqa: S102 - executing docs is the point
    except SnippetTimeout as exc:
        return "failed", (
            f"{block.where} {exc}; mark it <!-- docs-test: slow --> if the cost is intended"
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "?"
        if missing.split(".")[0] in REPO_IMPORT_ROOTS:
            if STRICT_IMPORTS:
                return "failed", (
                    f"{block.where} imports missing in-repo module {missing!r}"
                )
            return "skipped", (
                f"{block.where} needs {missing!r}, which this environment does not "
                "have installed"
            )
        return "skipped", f"{block.where} needs optional dependency {missing!r}"
    except SystemExit as exc:
        if exc.code not in (None, 0):
            return "failed", f"{block.where} exited with code {exc.code}"
    except BaseException as exc:  # noqa: BLE001 - report, never mask
        if block.raises and _matches(exc, block.raises):
            return "passed", block.where
        return "failed", f"{block.where} raised {type(exc).__name__}: {exc}"
    if block.raises:
        return "failed", f"{block.where} was expected to raise {block.raises}"
    return "passed", block.where


def run_blocks(blocks: list[Block]) -> tuple[str, str]:
    """Execute a document's blocks in one shared namespace, stopping at the first
    problem (later blocks depend on earlier ones, so the chain is broken anyway).

    Returns ``(outcome, detail)`` with outcome ``"passed"`` / ``"skipped"`` /
    ``"failed"``. Free of pytest calls so the triage report can reuse it.
    """
    runnable = [b for b in blocks if b.mode == "exec"]
    if not runnable:
        return "skipped", "no executable blocks"

    _warm_up()
    namespace: dict[str, object] = {"__name__": "__main__"}
    try:
        for block in runnable:
            outcome, detail = _execute(block, namespace)
            if outcome != "passed":
                return outcome, detail
        return "passed", f"{len(runnable)} block(s)"
    finally:
        _release(namespace)


def run_blocks_verbose(blocks: list[Block]) -> list[tuple[Block, str, str]]:
    """Triage variant: keep going after a failure to surface every problem at once.

    Later results in a broken chain can be cascades of the first failure, so read
    the first failure per document as the authoritative one.
    """
    _warm_up()
    namespace: dict[str, object] = {"__name__": "__main__"}
    results = []
    try:
        for block in [b for b in blocks if b.mode == "exec"]:
            outcome, detail = _execute(block, namespace)
            results.append((block, outcome, detail))
        return results
    finally:
        _release(namespace)


def _matches(exc: BaseException, expected: str) -> bool:
    wanted = expected.rsplit(".", 1)[-1]
    return any(klass.__name__ == wanted for klass in type(exc).__mro__)


def _params() -> list[pytest.ParameterSet]:
    params = []
    for path, blocks, _ in DOCUMENTS:
        if not any(b.mode == "exec" for b in blocks):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        marks = [pytest.mark.slow] if any(b.slow for b in blocks) else []
        params.append(pytest.param(blocks, marks=marks, id=relative))
    return params


@pytest.mark.parametrize("blocks", _params())
def test_documented_snippets_run(
    blocks: list[Block], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every documented ``python`` block executes, or says why it does not."""
    # Snippets that write artifacts must not litter the repository.
    monkeypatch.chdir(tmp_path)
    outcome, detail = run_blocks(blocks)
    if outcome == "skipped":
        pytest.skip(detail)
    assert outcome == "passed", detail


def test_every_opt_out_states_a_reason() -> None:
    """``skip`` / ``file-skip`` must justify themselves; ``signature`` is self-evident."""
    unexplained = [
        f"{d.mode} at {path.relative_to(REPO_ROOT).as_posix()}:{d.line}"
        for path, _, directives in DOCUMENTS
        for d in directives
        if d.mode in {"skip", "file-skip"} and not d.reason.strip()
    ]
    assert not unexplained, (
        "every docs-test skip must carry reason=\"...\" so the opt-out is "
        f"auditable in review: {unexplained}"
    )


def test_no_unknown_directive_keywords() -> None:
    """A typo like ``docs-test: skipp`` must not silently enable a block."""
    unknown = []
    for path, _, directives in DOCUMENTS:
        for directive in directives:
            # The prose inside reason="..." is free text, so drop it (and any
            # other key=value pair) before looking at the bare keywords.
            body = _REASON_RE.sub("", directive.raw)
            bare = [t for t in body.split() if "=" not in t]
            strays = sorted(set(bare) - KEYWORDS)
            if strays:
                where = f"{path.relative_to(REPO_ROOT).as_posix()}:{directive.line}"
                unknown.append(f"{where} -> {strays}")
    assert not unknown, (
        f"unrecognised docs-test keywords (expected a subset of {sorted(KEYWORDS)}): {unknown}"
    )


def main() -> int:
    """Print a full inventory + per-block outcome (triage aid, not a test).

    ``--all`` keeps going after a failing block so one pass surfaces every
    problem; ``--only PREFIX`` restricts the run to matching documents;
    ``--slow`` includes the ``slow``-marked documents the default run skips.
    """
    import sys
    import tempfile

    verbose = "--all" in sys.argv
    include_slow = "--slow" in sys.argv
    only = ""
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    modes: dict[str, int] = {}
    for block in ALL_BLOCKS:
        modes[block.mode] = modes.get(block.mode, 0) + 1
    print(f"documents: {len(DOCUMENTS)}   python blocks: {len(ALL_BLOCKS)}")
    for mode in sorted(modes):
        print(f"  {mode:<10} {modes[mode]}")
    print(f"per-block timeout: {BLOCK_TIMEOUT_SECONDS:g}s\n")

    failures: list[str] = []
    skips: list[str] = []
    origin = Path.cwd()
    for path, blocks, _ in DOCUMENTS:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if only and not relative.startswith(only):
            continue
        runnable = [b for b in blocks if b.mode == "exec"]
        if not runnable:
            continue
        slow = any(b.slow for b in blocks)
        if slow and not include_slow:
            print(f"slow {relative} (skipped; pass --slow to run)", flush=True)
            continue
        with tempfile.TemporaryDirectory() as sandbox:
            os.chdir(sandbox)
            try:
                if verbose:
                    outcomes = run_blocks_verbose(blocks)
                else:
                    outcome, detail = run_blocks(blocks)
                    outcomes = [(runnable[0], outcome, detail)]
            finally:
                os.chdir(origin)

        tag = " [slow]" if slow else ""
        bad = [(b, o, d) for b, o, d in outcomes if o == "failed"]
        missing = [(b, o, d) for b, o, d in outcomes if o == "skipped"]
        if bad:
            for _, _, detail in bad:
                failures.append(detail)
                print(f"FAIL{tag} {detail}", flush=True)
        elif missing:
            for _, _, detail in missing:
                skips.append(detail)
                print(f"skip{tag} {detail}", flush=True)
        else:
            print(f"ok  {tag} {relative} ({len(runnable)} block(s))", flush=True)

    print(f"\n{len(failures)} failing block(s), {len(skips)} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

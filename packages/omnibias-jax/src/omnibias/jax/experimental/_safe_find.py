# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Time one planner candidate in a fresh process, so a compiler crash cannot take the caller down.

``compile_operator(..., safe_find=True)`` times the leading candidates of ``mode="find"`` here instead of in process.  XLA
has crashed the interpreter (segfault, sticky CUDA error) on shapes that other programs of the same pool compile; the
child absorbs that, the parent records the failure and ranks the candidate last.

The child receives plain data over a pipe (layer arrays as NumPy, activation *names*, the request keywords) and rebuilds the
operator with ``force=<candidate>``.  Only layer-sequence models with named activations travel this way; a callable model
closes over objects that do not cross a process boundary.  The first call pays interpreter start-up, import and compile in
the child, so this is for ``find`` (a few candidates, once), not for per-step use.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["SafeFindError", "time_candidate_in_subprocess"]


class SafeFindError(RuntimeError):
    """The candidate crashed, timed out or raised in its own process; ``returncode`` is that process's exit status."""

    def __init__(self, algorithm: str, reason: str, returncode: int | None = None) -> None:
        self.algorithm = algorithm
        self.returncode = returncode
        super().__init__(f"{algorithm}: {reason}")


def time_candidate_in_subprocess(
    layers: Sequence[tuple[Any, Any, str | None]],
    x: Any,
    algorithm: str,
    kwargs: Mapping[str, Any],
    *,
    x64: bool,
    timeout: float = 600.0,
    repeats: int = 5,
    python: str | None = None,
) -> float:
    """Median seconds of one call of ``algorithm`` (forward, or value-and-gradient when ``kwargs`` say so) in a child."""
    payload = {
        "layers": [
            (np.asarray(W), None if b is None else np.asarray(b), spec) for W, b, spec in layers
        ],
        "x": np.asarray(x),
        "algorithm": algorithm,
        "kwargs": dict(kwargs),
        "x64": bool(x64),
        "repeats": int(repeats),
    }
    cmd = [python or sys.executable, "-m", "omnibias.jax.experimental._safe_find"]
    try:
        done = subprocess.run(  # noqa: S603 - the interpreter running this process, a module of this package
            cmd,
            input=pickle.dumps(payload),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        raise SafeFindError(algorithm, f"timed out after {timeout:g} s in its own process") from err
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise SafeFindError(
            algorithm,
            f"its process exited with status {done.returncode}: {' | '.join(tail)[:300]}",
            done.returncode,
        )
    try:
        return float(done.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError) as err:
        raise SafeFindError(algorithm, "its process returned no timing") from err


def _ready(tree: Any) -> Any:  # pragma: no cover - runs in the subprocess
    import jax

    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), tree)


def _child() -> None:  # pragma: no cover - runs in the subprocess
    import time

    payload = pickle.loads(sys.stdin.buffer.read())  # noqa: S301 - written by the parent just above
    import jax

    if payload["x64"]:
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    from omnibias.jax.experimental.compiler import compile_operator

    import jax.numpy as jnp

    layers = [
        (jnp.asarray(W), None if b is None else jnp.asarray(b), spec)
        for W, b, spec in payload["layers"]
    ]
    x = jnp.asarray(payload["x"])
    kwargs = dict(payload["kwargs"])
    op = compile_operator(
        layers, x, force=payload["algorithm"], mode="heuristic", safe_find=False, **kwargs
    )
    key = jax.random.PRNGKey(0)
    specs = [spec for _, _, spec in layers]
    if kwargs.get("differentiate", "none") == "none":

        def run(arrays: Any) -> Any:
            return op([(W, b, s) for (W, b), s in zip(arrays, specs, strict=True)], x, key=key)

        compiled = jax.jit(run)
    else:

        def loss(arrays: Any) -> Any:
            value = run_value(arrays)
            return jnp.sum(value * value)

        def run_value(arrays: Any) -> Any:
            return op([(W, b, s) for (W, b), s in zip(arrays, specs, strict=True)], x, key=key)

        compiled = jax.jit(jax.value_and_grad(loss))
    arrays = [(W, b) for W, b, _ in layers]
    _ready(compiled(arrays))
    times = []
    for _ in range(payload["repeats"]):
        t0 = time.perf_counter()
        _ready(compiled(arrays))
        times.append(time.perf_counter() - t0)
    print(float(np.median(times)))


if __name__ == "__main__":  # pragma: no cover
    _child()

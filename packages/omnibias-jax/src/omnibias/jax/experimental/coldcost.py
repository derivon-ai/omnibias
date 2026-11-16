# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The cold cost of a planned program: what it is made of and the levers that move it.

A planned program pays once, in every fresh process, for tracing, lowering and XLA compilation before its first
evaluation.  Steady-state speed-ups are measured without it (``docs/performance.md``), so a program that is faster per
call can still lose on a short run.  This module makes the cost visible and gives the caller bounded levers; none of
them is on by default and every one is recorded in ``explain()``.

* :class:`StagedProgram` runs ``trace -> lower -> compile`` as separate, timed steps and keeps one executable per
  argument signature, so :meth:`~StagedProgram.compile` can pay the cost ahead of time and the time spent in each
  stage is reported rather than lumped into the first call.
* :func:`bucket_size` / :func:`pad_rows` round a changing batch size up to a small set of shapes so that a new batch
  size reuses an executable instead of compiling a new one.  Operators act pointwise, so padding rows are discarded and
  the result is exactly that of the unpadded call.  Residual blocks use :func:`~omnibias.jax.experimental.pad_block`
  with a mask.
* ``COMPILE_POLICIES`` are named sets of XLA compiler options (accelerator only).  They trade compile time against
  steady-state speed by amounts measured on one device class (``docs/performance.md``); ``"auto"`` applies the
  break-even rule below and nothing guarantees it is right for a particular program, which is why ``explain()`` lists
  the policy and the measured stage times.
* :func:`enable_compilation_cache` points JAX's persistent compilation cache at a directory.  It is a process-wide
  setting, so it is an explicit call and never made by the planner.
* :func:`export_program` stores the traced program (StableHLO) so that a later process can skip tracing; XLA
  compilation still happens on first use there unless the persistent cache holds the executable.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

Array = jax.Array

__all__ = [
    "BUCKET_POLICIES",
    "COMPILE_POLICIES",
    "ExportedProgram",
    "MemoryBudgetExceeded",
    "PolicyChoice",
    "RETRY_LADDER",
    "StageTimes",
    "StagedProgram",
    "bucket_size",
    "compilation_cache_dir",
    "enable_compilation_cache",
    "export_program",
    "is_retryable_xla_error",
    "is_traced",
    "pad_rows",
    "resolve_compile_policy",
]

# ------------------------------------------------------------------------------------------------- batch buckets

#: Named bucket policies; an ``int`` ``m`` means "round up to a multiple of ``m``".
BUCKET_POLICIES = ("none", "pow2", "geometric")


def bucket_size(n: int, policy: str | int | None) -> int:
    """The bucket that holds ``n`` rows under ``policy``.

    ``None``/``"none"`` keeps ``n``; ``"pow2"`` rounds up to a power of two; ``"geometric"`` to the next of
    ``1, 2, 3, 4, 6, 8, 12, 16, 24, ...`` (successive buckets grow by at most 1.5, so padding stays below 50 %); an
    integer ``m`` rounds up to a multiple of ``m``.
    """
    if n < 1:
        raise ValueError("a bucket holds at least one row")
    if policy is None or policy == "none":
        return int(n)
    if isinstance(policy, bool):
        raise TypeError("bucket policy must be a name or an integer")
    if isinstance(policy, int):
        if policy < 1:
            raise ValueError("an integer bucket must be a positive multiple")
        return int(-(-n // policy) * policy)
    if policy == "pow2":
        return 1 << (int(n) - 1).bit_length()
    if policy == "geometric":
        p = 1 << (int(n) - 1).bit_length()  # smallest power of two >= n
        q = p * 3 // 4  # the member of the set between p / 2 and p
        return q if n <= q else p
    raise ValueError(
        f"unknown bucket policy {policy!r}; use one of {BUCKET_POLICIES} or an integer"
    )


def pad_rows(a: Array, rows: int) -> Array:
    """Grow the leading axis of ``a`` to ``rows`` by repeating its last row (finite wherever the real rows are)."""
    a = jnp.asarray(a)
    n = a.shape[0]
    if rows < n:
        raise ValueError("cannot pad to fewer rows than given")
    if rows == n:
        return a
    return jnp.concatenate([a, jnp.repeat(a[-1:], rows - n, axis=0)])


# ----------------------------------------------------------------------------------------------- compile policies

#: Compiler options by policy name.  Accelerator options only; on a CPU backend every policy resolves to ``{}``.
#: ``fast``: no Triton GEMM (measured cold compile 0.2-0.7x for shipped programs, steady state up to 1.15x).
#: ``fastest``: autotune level 0 (cold compile 0.2-0.9x, steady state 1.1-3x; the level to use for forward evaluation
#: at very small D and very large batches, where the default compile took minutes).
COMPILE_POLICIES: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {"xla_gpu_enable_triton_gemm": False},
    "fastest": {"xla_gpu_autotune_level": 0},
}

#: Options tried in order, after the requested ones, when ``retry`` is on and XLA's GPU autotuner or a Triton GEMM
#: fails (``compile_policy="auto"``): no autotuning first, then no Triton GEMM either.
RETRY_LADDER: tuple[Mapping[str, Any], ...] = (
    {"xla_gpu_autotune_level": 0},
    {"xla_gpu_autotune_level": 0, "xla_gpu_enable_triton_gemm": False},
)

_RETRYABLE = re.compile(
    r"Failed to get configs|autotun|misaligned address|illegal memory access|CUDA_ERROR_(ILLEGAL|MISALIGNED)|"
    r"triton.{0,40}(gemm|fusion)|gemm_fusion",
    re.IGNORECASE,
)


def is_retryable_xla_error(err: BaseException) -> bool:
    """Whether ``err`` looks like an XLA autotune/Triton-GEMM failure that other compiler options may avoid."""
    return bool(_RETRYABLE.search(f"{type(err).__name__}: {err}"))


# Ratios used by ``auto`` (population medians/upper ends of the ``fast`` policy; see ``docs/performance.md``).
_FAST_COMPILE_RATIO = 0.28
_FAST_STEADY_RATIO = 1.15


@dataclass(frozen=True)
class PolicyChoice:
    """The compile policy in force and why."""

    requested: str
    name: str
    options: Mapping[str, Any]
    applied: bool
    reason: str

    def to_config(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "name": self.name,
            "options": dict(self.options),
            "applied": self.applied,
            "reason": self.reason,
        }


def resolve_compile_policy(
    policy: str,
    *,
    platform: str,
    predicted_seconds: float | None,
    cold_seconds: float | None,
    expected_uses: int,
) -> PolicyChoice:
    """Resolve ``policy`` (``default``, ``fast``, ``fastest`` or ``auto``) for ``platform``.

    ``auto`` picks ``fast`` when the compile time it is expected to save exceeds the steady-state time it is expected
    to cost over ``expected_uses`` calls: ``(1 - 0.28) * cold > 0.15 * uses * seconds`` (the measured median compile
    ratio and the upper end of the measured steady-state ratio of the ``fast`` policy).  Without a prediction it
    keeps ``default``.
    """
    if policy not in (*COMPILE_POLICIES, "auto"):
        raise ValueError(
            f"unknown compile policy {policy!r}; use one of {(*COMPILE_POLICIES, 'auto')}"
        )
    on_accelerator = platform not in ("cpu", "interpreter")
    name = policy
    why = f"policy {policy!r}"
    if policy == "auto":
        if not on_accelerator:
            name, why = "default", "auto: no accelerator options on this backend"
        elif predicted_seconds is None or cold_seconds is None:
            name, why = "default", "auto: no cost prediction for this program"
        else:
            saved = (1.0 - _FAST_COMPILE_RATIO) * cold_seconds
            cost = (_FAST_STEADY_RATIO - 1.0) * max(1, expected_uses) * predicted_seconds
            name = "fast" if saved > cost else "default"
            why = (
                f"auto: compile saved {saved:.3g} s vs steady-state cost {cost:.3g} s over "
                f"{max(1, expected_uses)} uses"
            )
    options = COMPILE_POLICIES[name]
    if options and not on_accelerator:
        return PolicyChoice(
            policy, name, {}, False, f"{why}; accelerator options are not applied on {platform}"
        )
    return PolicyChoice(policy, name, dict(options), bool(options), why)


# --------------------------------------------------------------------------------------------- persistent cache


def enable_compilation_cache(path: str, *, min_compile_seconds: float = 0.0) -> dict[str, Any]:
    """Point JAX's persistent compilation cache at ``path`` for this process; returns the previous settings.

    Process-wide and explicit: the planner never calls it.  A second process that compiles the same program with the
    same JAX and device reuses the stored executable (``docs/performance.md`` measures the saving).  Writing costs
    ``0.95-1.60x`` of an uncached compile.
    """
    config: Any = (
        jax.config
    )  # the flag attributes are created at import, which a type checker cannot see
    names = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    previous = {name: getattr(config, name) for name in names}
    config.update("jax_compilation_cache_dir", path)
    config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_seconds))
    config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return previous


def compilation_cache_dir() -> str | None:
    """The persistent compilation cache directory in force, if any."""
    config: Any = jax.config
    d = config.jax_compilation_cache_dir
    return None if not d else str(d)


# ---------------------------------------------------------------------------------------------- staged programs


class MemoryBudgetExceeded(ValueError):
    """The compiled program needs more device memory than the budget: refused before it runs, not an out-of-memory.

    ``peak_bytes`` is XLA's buffer-assignment peak for the program (arguments, outputs and temporaries), known after
    compilation and before the first execution.
    """

    def __init__(self, signature: str, peak_bytes: int, budget: float) -> None:
        self.signature = signature
        self.peak_bytes = int(peak_bytes)
        self.budget = float(budget)
        super().__init__(
            f"the compiled program for arguments {signature} needs {peak_bytes / 2**30:.3g} GiB of device memory "
            f"(arguments, outputs and temporaries), above the budget of {budget / 2**30:.3g} GiB"
        )


def _memory_of(executable: Any) -> dict[str, int] | None:
    """Buffer sizes of a compiled executable (bytes), or ``None`` where the backend does not report them."""
    try:
        stats = executable.memory_analysis()
    except Exception:  # noqa: BLE001 - not every backend implements it
        return None
    if stats is None:
        return None
    arg = int(getattr(stats, "argument_size_in_bytes", 0))
    out = int(getattr(stats, "output_size_in_bytes", 0))
    tmp = int(getattr(stats, "temp_size_in_bytes", 0))
    peak = int(getattr(stats, "peak_memory_in_bytes", 0) or 0)
    return {
        "argument_bytes": arg,
        "output_bytes": out,
        "temp_bytes": tmp,
        "peak_bytes": max(peak, arg + out + tmp),
    }


@dataclass
class StageTimes:
    """Seconds spent in each stage of building one executable (``first_run_seconds`` is the first execution)."""

    signature: str
    trace_seconds: float
    lower_seconds: float
    compile_seconds: float
    first_run_seconds: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    options_error: str | None = None
    memory: Mapping[str, int] | None = None
    retried: int = 0  # rungs of RETRY_LADDER used after the requested options failed (0: none)

    @property
    def cold_seconds(self) -> float:
        return self.trace_seconds + self.lower_seconds + self.compile_seconds

    def to_config(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "trace_seconds": self.trace_seconds,
            "lower_seconds": self.lower_seconds,
            "compile_seconds": self.compile_seconds,
            "cold_seconds": self.cold_seconds,
            "first_run_seconds": self.first_run_seconds,
            "options": dict(self.options),
            "options_error": self.options_error,
            "memory": None if self.memory is None else dict(self.memory),
            "retried": self.retried,
        }


def _block(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), tree)


def is_traced(*trees: Any) -> bool:
    """Whether any leaf of ``trees`` is a JAX tracer (the call happens inside ``grad``, ``vmap`` or an outer ``jit``)."""
    return any(
        isinstance(leaf, jax.core.Tracer) for t in trees for leaf in jax.tree_util.tree_leaves(t)
    )


def _weak(leaf: Any) -> bool:
    """Whether JAX treats ``leaf`` as weakly typed (Python scalars and arrays made from them)."""
    if isinstance(leaf, int | float | complex):
        return True
    return bool(getattr(getattr(leaf, "aval", None), "weak_type", False))


def _leaf_signature(leaf: Any) -> tuple[tuple[int, ...], Any, bool]:
    # A concrete device array already carries its canonical shape, dtype and weak type; reading them is a few
    # attribute lookups, whereas ``jnp.result_type`` and ``str(dtype)`` cost about 15 microseconds a leaf, which is
    # more than the whole program at launch-bound sizes.
    if isinstance(leaf, jax.Array):
        return leaf.shape, leaf.dtype, bool(leaf.aval.weak_type)
    return tuple(np.shape(leaf)), jnp.result_type(leaf), _weak(leaf)


def _signature(args: tuple[Any, ...]) -> tuple[Any, tuple[tuple[tuple[int, ...], Any, bool], ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(args)
    return treedef, tuple(_leaf_signature(leaf) for leaf in leaves)


def _describe(sig: tuple[tuple[tuple[int, ...], Any, bool], ...]) -> str:
    return ",".join(f"{dtype}{list(shape)}" + ("~" if weak else "") for shape, dtype, weak in sig)


class StagedProgram:
    """``jax.jit(fn)`` split into timed ``trace -> lower -> compile`` stages, one executable per signature.

    Calling it with arguments it has not seen builds the executable (and records its :class:`StageTimes`); calling it
    with seen shapes and dtypes goes straight to the stored executable.  With a ``memory_budget`` (bytes) an executable
    whose buffer-assignment peak exceeds it is refused with :class:`MemoryBudgetExceeded` before it runs.  ``options`` are XLA compiler options applied
    at ``compile``; if the backend rejects them the program is compiled without them and the reason is recorded, so a
    policy that does not apply to a device degrades instead of failing.

    With ``retry=True`` an XLA autotune or Triton-GEMM failure (:func:`is_retryable_xla_error`) at compile, or at the first
    execution of an executable, rebuilds it with the next options of :data:`RETRY_LADDER` (``xla_gpu_autotune_level=0``,
    then also no Triton GEMM); the rung used is recorded in :class:`StageTimes` (``retried``).  A sticky CUDA error
    (``misaligned address``) can poison the device context, which an in-process retry cannot repair: isolate such a
    compile in a subprocess (``find(safe=True)``).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        options: Mapping[str, Any] | None = None,
        memory_budget: float | None = None,
        retry: bool = False,
    ) -> None:
        self.memory_budget = memory_budget
        self.retry = retry
        self._jit = jax.jit(fn)
        self.options: dict[str, Any] = dict(options or {})
        self._lowered: dict[Any, Any] = {}
        self._rung: dict[Any, int] = {}
        self._table: dict[Any, Callable[..., Any]] = {}
        self._times: dict[Any, StageTimes] = {}
        self.stages: list[StageTimes] = []

    def signatures(self) -> int:
        """Number of executables built so far."""
        return len(self._table)

    def _ladder(self) -> list[dict[str, Any]]:
        """The option sets in the order they are tried: the requested ones, then (with ``retry``) the fallback ladder."""
        ladder = [dict(self.options)]
        if self.retry:
            ladder += [dict(r) for r in RETRY_LADDER if dict(r) != dict(self.options)]
        return ladder

    @staticmethod
    def _compile_lowered(lowered: Any, options: Mapping[str, Any]) -> Any:
        return lowered.compile(compiler_options=dict(options)) if options else lowered.compile()

    def compile(self, *args: Any) -> StageTimes:
        """Build the executable for ``args`` (arrays or ``ShapeDtypeStruct``) now; return its stage times."""
        key = _signature(args)
        if key in self._times:
            return self._times[key]
        t0 = time.perf_counter()
        traced = self._jit.trace(*args)
        t1 = time.perf_counter()
        lowered = traced.lower()
        t2 = time.perf_counter()
        error: str | None = None
        applied: dict[str, Any] = {}
        ladder = self._ladder()
        executable: Any = None
        rung = 0
        for rung, opts in enumerate(ladder):
            try:
                executable = self._compile_lowered(lowered, opts)
                applied = dict(opts)
                break
            except Exception as err:  # noqa: BLE001 - an option the backend rejects must not block the program
                error = f"{type(err).__name__}: {str(err)[:200]}"
                if self.retry and is_retryable_xla_error(err) and rung + 1 < len(ladder):
                    continue
                if not opts:
                    raise
                t2 = time.perf_counter()
                executable = self._compile_lowered(lowered, {})
                applied = {}
                break
        landed = applied == dict(ladder[rung])  # False where every option was dropped
        t3 = time.perf_counter()
        memory = _memory_of(executable)
        st = StageTimes(
            _describe(key[1]),
            t1 - t0,
            t2 - t1,
            t3 - t2,
            None,
            applied,
            error,
            memory,
            rung if landed else 0,
        )
        if self.retry:  # only a retry reads the lowered module again; holding it otherwise keeps the whole IR alive
            self._lowered[key] = lowered
        self._rung[key] = rung if landed else len(ladder)
        if (
            self.memory_budget is not None
            and memory is not None
            and memory["peak_bytes"] > self.memory_budget
        ):
            raise MemoryBudgetExceeded(st.signature, memory["peak_bytes"], self.memory_budget)
        self._table[key] = executable
        self._times[key] = st
        self.stages.append(st)
        return st

    def __call__(self, *args: Any) -> Any:
        if is_traced(args):
            # A compiled executable cannot be differentiated or batched; inside a transformation the program is traced
            # into the caller's computation, which then owns the compile (and its options).
            return self._jit(*args)
        key = _signature(args)
        if key not in self._table:
            self.compile(*args)
        executable = self._table[key]
        t0 = time.perf_counter()
        st = self._times[key]
        try:
            out = executable(*args)
            if st.first_run_seconds is None:
                _block(out)
                st.first_run_seconds = time.perf_counter() - t0
        except Exception as err:  # noqa: BLE001 - only an XLA autotune/Triton failure is retried
            if not (self.retry and is_retryable_xla_error(err)):
                raise
            out = self._rerun(key, args, err)
        return out

    def _rerun(self, key: Any, args: tuple[Any, ...], first: BaseException) -> Any:
        """Rebuild the executable of ``key`` with the next fallback options and run it again."""
        ladder = self._ladder()
        st = self._times[key]
        last: BaseException = first
        for rung in range(self._rung[key] + 1, len(ladder)):
            t0 = time.perf_counter()
            try:
                executable = self._compile_lowered(self._lowered[key], ladder[rung])
                out = _block(executable(*args))
            except Exception as err:  # noqa: BLE001
                if not is_retryable_xla_error(err):
                    raise
                last = err
                continue
            self._table[key] = executable
            self._rung[key] = rung
            st.options = dict(ladder[rung])
            st.retried = rung
            st.options_error = f"{type(first).__name__}: {str(first)[:200]}"
            st.first_run_seconds = time.perf_counter() - t0
            return out
        raise last

    def report(self) -> list[dict[str, Any]]:
        return [s.to_config() for s in self.stages]


# --------------------------------------------------------------------------------------------------- export


class ExportedProgram:
    """A traced program kept as StableHLO: callable in this process, serialisable for another.

    ``call`` skips tracing and lowering to StableHLO; XLA compilation still runs on first use (use the persistent
    compilation cache to skip that too).  ``serialize`` needs the optional ``flatbuffers`` package, as ``jax.export``
    does.
    """

    def __init__(self, exported: Any, metadata: Mapping[str, Any]) -> None:
        self._exported = exported
        self.metadata = dict(metadata)

    def call(self, *args: Any) -> Any:
        return self._exported.call(*args)

    def serialize(self) -> bytes:
        try:
            return bytes(self._exported.serialize())
        except ImportError as err:  # pragma: no cover - depends on the optional dependency
            raise ImportError(
                "serialising an exported program needs the optional 'flatbuffers' package, as jax.export does"
            ) from err

    @classmethod
    def deserialize(cls, blob: bytes, metadata: Mapping[str, Any] | None = None) -> ExportedProgram:
        from jax import export as jax_export

        return cls(jax_export.deserialize(bytearray(blob)), metadata or {})


def export_program(
    fn: Callable[..., Any], *args: Any, metadata: Mapping[str, Any] | None = None
) -> ExportedProgram:
    """Export ``jax.jit(fn)`` for the shapes of ``args`` (arrays or ``ShapeDtypeStruct``)."""
    from jax import export as jax_export

    return ExportedProgram(jax_export.export(jax.jit(fn))(*args), metadata or {})

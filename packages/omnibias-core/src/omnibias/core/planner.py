# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-neutral planning of structured derivative programs, in the manner of a convolution library.

A *problem descriptor* (:class:`Descriptor`) states what is wanted: an operator request on a dense chain, a batch size, a
dtype and, for residual operators, which derivative quantities are read. An *algorithm* (:class:`Algorithm`) is a way to
compute it: a name, the operators it serves, a refusal rule that says why it does not apply, and a work model (flops,
traffic, peak memory, launches). A :class:`CostModel` turns work into predicted seconds on one device class from a small
calibration (rates and one efficiency factor per algorithm). :func:`select` lists the feasible algorithms, drops those
that exceed a memory budget with the reason recorded, ranks the rest, and optionally *finds* the winner by timing the
leading candidates through a caller-supplied timer, returning a :class:`Plan` that :class:`PlanCache` persists. This
module imports no tensor framework: JAX and PyTorch executors register under the algorithm names.

The cost model is a ranking instrument calibrated per device, not a performance claim. ``docs/performance.md`` owns
measured claims.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnibias.core.chain import ChainDescriptor
from omnibias.core.demand import Demand

__all__ = [
    "AUTO_FIND_MIN_USES",
    "AUTO_TRAINING_USES",
    "Algorithm",
    "CostModel",
    "Descriptor",
    "FIND_CUT_FACTOR",
    "FIND_TOP_K",
    "FindRefusal",
    "OperatorRequest",
    "PLAN_FORMAT",
    "PLAN_MIGRATIONS",
    "Plan",
    "PlanCache",
    "Prediction",
    "Refusal",
    "Work",
    "algorithms_for",
    "default_plan_cache_path",
    "device_class",
    "estimator_probes",
    "host_memory_limit",
    "measurement_context",
    "migrate_plan_config",
    "register_algorithm",
    "registry_digest",
    "resolve_planning",
    "select",
    "unregister_algorithm",
]

#: Version of the persisted plan format. A cache written by another version is ignored, never misread.
#: The integer changes whenever ``Plan.to_config`` changes incompatibly (a key removed or renamed, a meaning changed);
#: adding an optional key does not change it. A configuration with an older version is read through ``PLAN_MIGRATIONS``,
#: each entry mapping format ``n`` to format ``n + 1``; one without a complete chain, or from a newer version, is refused.
#: The frozen sample ``tests/data/plan_format_1.json`` must keep loading.
#:
#: Format 2 records what the plan optimizes (``objective``: a forward pass or value-and-gradient into the parameters) and
#: where its use count came from (``uses_source``); a format-1 plan was ranked on forward work with an explicit count.
PLAN_FORMAT = 2


def _migrate_1_to_2(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("objective", "forward")
    config.setdefault("uses_source", "explicit")
    return config


PLAN_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {1: _migrate_1_to_2}


def migrate_plan_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` brought to :data:`PLAN_FORMAT`, or ``ValueError`` when no migration chain reaches it."""
    out = dict(config)
    version = out.get("format")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"plan format {version!r} is not an integer")
    if version > PLAN_FORMAT:
        raise ValueError(f"plan format {version} is newer than this library's {PLAN_FORMAT}")
    while version < PLAN_FORMAT:
        step = PLAN_MIGRATIONS.get(version)
        if step is None:
            raise ValueError(f"no migration from plan format {version} to {version + 1}")
        out = step(dict(out))
        version += 1
        out["format"] = version
    return out


_KINDS = (
    "laplacian",
    "bilaplacian",
    "polylaplacian",
    "divergence",
    "logdet",
    "solve",
    "gram",
    "gauss_newton",
    "ntk_diagonal",
)
_METRICS = ("euclidean", "diagonal", "point_diagonal", "factor")
_DIFFERENTIATE = ("none", "params", "params2")


# ------------------------------------------------------------------------------------------------ descriptors


@dataclass(frozen=True)
class OperatorRequest:
    """What derivative quantity is wanted, independent of how it is computed.

    ``kind`` is one of ``laplacian`` (``power`` = 1 is the Laplacian, 2 the bi-Laplacian, ...), ``divergence``,
    ``logdet`` (``log |det(I + J)|`` of a vector field), ``solve`` (a Woodbury solve with ``I + J``) or a residual
    operator: ``gram``, ``gauss_newton`` and ``ntk_diagonal``. ``selected`` counts the inputs the operator acts on
    (``None`` is all of them: a time coordinate is excluded by selecting the space coordinates); ``metric`` says how
    the second-order form is weighted (a runtime diagonal, a per-point diagonal, or a factor); ``directions`` counts
    first-order drift directions read alongside.

    ``differentiate`` says what the caller does with the result: ``"none"`` (a forward evaluation), ``"params"``
    (value-and-gradient into the model parameters, as in training) or ``"params2"`` (a second derivative into the
    parameters: Newton and curvature methods; the fused kernels refuse it); the planner ranks, and ``find`` times, that
    objective. ``accuracy`` is the standard error a stochastic estimator may have, as a fraction of the Frobenius norm of
    the Hessian (a worst-case bound, see :func:`estimator_probes`); it needs ``exact=False`` and sizes the estimator.
    """

    kind: str
    power: int = 1
    selected: int | None = None
    metric: str = "euclidean"
    directions: int = 0
    exact: bool = True
    allow_almost_everywhere: bool = False
    differentiate: str = "none"
    accuracy: float | None = None

    def __post_init__(self) -> None:
        kind = "laplacian" if self.kind in ("bilaplacian", "polylaplacian") else self.kind
        if kind not in _KINDS:
            raise ValueError(f"unknown operator kind {self.kind!r}")
        if self.kind == "bilaplacian":
            object.__setattr__(self, "power", 2)
        if self.kind in ("bilaplacian", "polylaplacian"):
            object.__setattr__(self, "kind", "laplacian")
        if self.power < 1:
            raise ValueError("power must be >= 1")
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if self.selected is not None and self.selected < 1:
            raise ValueError("selected must be >= 1 or None")
        if self.directions < 0:
            raise ValueError("directions must be >= 0")
        if self.differentiate not in _DIFFERENTIATE:
            raise ValueError(f"differentiate must be one of {_DIFFERENTIATE}")
        if self.accuracy is not None:
            if self.exact:
                raise ValueError("accuracy sizes a stochastic estimator: pass exact=False with it")
            if not 0.0 < float(self.accuracy) < 10.0:
                raise ValueError("accuracy must be in (0, 10)")

    @property
    def order(self) -> int:
        """Highest derivative order read (``2 * power`` for Laplace powers, 1 for first-order quantities)."""
        return 2 * self.power if self.kind == "laplacian" else 1

    def to_config(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "power": self.power,
            "selected": self.selected,
            "metric": self.metric,
            "directions": self.directions,
            "exact": self.exact,
            "allow_almost_everywhere": self.allow_almost_everywhere,
            "differentiate": self.differentiate,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OperatorRequest:
        return cls(**dict(config))


@dataclass(frozen=True)
class Descriptor:
    """The full problem a plan is keyed on: request, model chain, batch, and (for residuals) the demand."""

    request: OperatorRequest
    chain: ChainDescriptor
    batch: int
    demand: Demand = field(default_factory=Demand.of)
    rows: int | None = None  # residual rows per block batch (defaults to ``batch``)

    def __post_init__(self) -> None:
        if isinstance(self.batch, bool) or not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError("batch must be a positive integer")

    @property
    def D(self) -> int:
        return self.chain.input_dim

    @property
    def selected(self) -> int:
        return self.request.selected or self.chain.input_dim

    @property
    def n_rows(self) -> int:
        return self.rows if self.rows is not None else self.batch

    def to_config(self) -> dict[str, Any]:
        return {
            "request": self.request.to_config(),
            "chain": self.chain.to_config(),
            "batch": self.batch,
            "demand": self.demand.to_config(),
            "rows": self.rows,
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_config(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------------------------- work and cost


@dataclass(frozen=True)
class Work:
    """Machine-independent work of one evaluation: GEMM and elementwise flops, memory traffic, peak bytes, launches."""

    gemm_flops: float = 0.0
    elem_flops: float = 0.0
    traffic_bytes: float = 0.0
    peak_bytes: float = 0.0
    launches: float = 0.0
    compile_units: float = 1.0  # relative size of the program the compiler must build
    #: Kernel launches the backward pass adds (from the lowering probe's per-template counts).
    launches_backward: float = 0.0
    #: Dense factorization / solver library calls (each pays a launch-and-latency penalty the flop count does not show).
    solver_calls: float = 0.0
    #: Host synchronizations (a device-to-host read blocks asynchronous execution and CUDA-graph capture).
    syncs: float = 0.0
    #: Flops one parallel unit (a tile of points on one multiprocessor) executes in sequence: the latency floor of a fused
    #: kernel that cannot spread a small batch over the device.
    serial_flops: float = 0.0

    def to_config(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Work:
        return cls(**{f: float(config[f]) for f in cls.__dataclass_fields__ if f in config})

    def second_derivative(self) -> Work:
        """A second derivative into the parameters of a training pass: the backward pass differentiated again, about 3x."""
        k = 3.0
        return Work(
            self.gemm_flops * k,
            self.elem_flops * k,
            self.traffic_bytes * k,
            self.peak_bytes * 1.5,
            self.launches * k,
            self.compile_units * 1.5,
            self.launches_backward * k,
            self.solver_calls * k,
            self.syncs,
            self.serial_flops * k,
        )

    def training(
        self,
        *,
        gemm: float = 2.0,
        elem: float = 2.0,
        traffic: float = 2.0,
        serial: float = 2.0,
        solver: float = 2.0,
        peak: float = 2.0,
    ) -> Work:
        """Forward plus backward into the parameters: each factor is the backward pass's work as a multiple of the forward's.

        Launches add :attr:`launches_backward`; the backward pass also keeps the saved streams (``peak`` times the forward
        peak, as a fraction of the forward's).
        """
        return Work(
            self.gemm_flops * (1.0 + gemm),
            self.elem_flops * (1.0 + elem),
            self.traffic_bytes * (1.0 + traffic),
            self.peak_bytes * (1.0 + 0.5 * (peak - 1.0)) if peak > 1.0 else self.peak_bytes,
            self.launches + self.launches_backward,
            self.compile_units * 1.8,
            0.0,
            self.solver_calls * (1.0 + solver),
            self.syncs,
            self.serial_flops * (1.0 + serial),
        )


@dataclass(frozen=True)
class Prediction:
    seconds: float
    peak_bytes: float
    compile_seconds: float


@dataclass(frozen=True)
class Refusal:
    algorithm: str
    reason: str


@dataclass(frozen=True)
class Algorithm:
    """A way to compute some operators.

    ``requires(descriptor)`` returns ``None`` when the algorithm applies and a reason otherwise. ``work(descriptor)``
    returns :class:`Work`. ``exact=False`` marks stochastic estimators, which are never offered for an ``exact`` request.
    """

    name: str
    operators: frozenset[str]
    requires: Callable[[Descriptor], str | None]
    work: Callable[[Descriptor], Work]
    backend: str = "any"  # "any", or the executors that implement it: "jax", "torch", "jax|torch"
    exact: bool = True
    generic: bool = False  # always-applicable fallback of last resort
    #: Extra refusal rules of one executor: ``{"torch": rule}`` where ``rule(descriptor)`` returns a reason or ``None``
    #: (for an executor that implements only part of what the algorithm covers elsewhere).
    backend_requires: Mapping[str, Callable[[Descriptor], str | None]] = field(default_factory=dict)
    #: Backward-pass work as multiples of the forward's, for :meth:`Work.training` (empty: its defaults).
    backward: Mapping[str, float] = field(default_factory=dict)
    #: A kernel program that exists only on an accelerator (refused on a CPU device class).
    accelerator_only: bool = False
    #: ``rule(descriptor, backend, device)`` returns a reason when the algorithm is measured to lose to another pool member
    #: here and so runs only when named with ``force`` (it stays visible in the refusals); ``None`` admits it to the ranking.
    forced_only: Callable[[Descriptor, str, str], str | None] | None = None
    #: ``rule(descriptor, backend, device)`` returns a reason when the kernel cannot run on this device at all (a hardware
    #: resource such as shared memory); unlike ``forced_only`` it also refuses a ``force`` request.
    hardware_requires: Callable[[Descriptor, str, str], str | None] | None = None

    def serves(self, backend: str) -> bool:
        """Whether the executors of ``backend`` implement this algorithm (``"any"`` asks for no restriction)."""
        return self.backend == "any" or backend in ("", "any") or backend in self.backend.split("|")

    def refusal(self, descriptor: Descriptor, backend: str = "any") -> str | None:
        """The reason this algorithm does not apply to ``descriptor`` on ``backend``, or ``None``."""
        reason = self.requires(descriptor)
        if reason is None and backend in self.backend_requires:
            reason = self.backend_requires[backend](descriptor)
        return reason


_REGISTRY: dict[tuple[str, str], Algorithm] = {}


def register_algorithm(algorithm: Algorithm, *, replace: bool = False) -> Algorithm:
    """Register ``algorithm``; a name may be reused for different operator sets (``projected`` serves several)."""
    key = (algorithm.name, "|".join(sorted(algorithm.operators)))
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"algorithm {algorithm.name!r} is already registered for {sorted(algorithm.operators)}"
        )
    _REGISTRY[key] = algorithm
    return algorithm


def unregister_algorithm(name: str, operators: Iterable[str] | frozenset[str]) -> Algorithm:
    """Remove and return the algorithm registered as ``name`` for exactly ``operators`` (``KeyError`` if there is none)."""
    key = (name, "|".join(sorted(operators)))
    try:
        return _REGISTRY.pop(key)
    except KeyError:
        raise KeyError(f"no algorithm {name!r} is registered for {sorted(operators)}") from None


def algorithms_for(operator: str, backend: str = "any") -> list[Algorithm]:
    """Registered algorithms for ``operator`` (kind) usable from ``backend``, in registration order."""
    return [a for a in _REGISTRY.values() if operator in a.operators and a.serves(backend)]


def registry_digest() -> str:
    names = sorted("/".join(key) for key in _REGISTRY)
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()[:12]


_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


@dataclass(frozen=True)
class CostModel:
    """Per-device ranking model: ``seconds = eff[a] * (max(gemm/rate_gemm, traffic/bandwidth) + elem/rate_elem) + launches * launch_s``.

    ``efficiency`` is one multiplicative factor per algorithm (default 1) fitted from measurements on the device
    class; ``compile_s_per_unit`` converts a work model's relative program size into predicted compile seconds.
    Rates may depend on the dtype (``rate_gemm`` for float64 and ``rate_gemm_f32`` for lower precision).
    """

    device: str = "generic"
    rate_gemm: float = 5e12
    rate_gemm_f32: float = 1e13
    rate_elem: float = 5e11
    bandwidth: float = 1e12
    launch_s: float = 5e-6
    compile_s_per_unit: float = 2.0
    efficiency: Mapping[str, float] = field(default_factory=dict)
    source: str = "defaults"
    #: Latency of one dense factorization or solver library call (cuSOLVER-class: tens of microseconds, not a launch).
    solver_s: float = 5e-5
    #: Cost of one device-to-host synchronization.
    sync_s: float = 1e-4
    #: Flop rate of one parallel unit (one multiprocessor of ``rate_gemm / units``), for ``Work.serial_flops``.
    units: float = 100.0
    #: Device launches of one algorithm as a multiple of the registry's count (default 1): the registry counts the kernels
    #: of the JAX/XLA programs, whose elementwise chains fuse; an executor that dispatches each operation as its own
    #: kernel (PyTorch eager) launches several times more, and a calibration file records the counted ratio. A key is the
    #: algorithm name, or ``"<operator><power>/<name>"`` (``"laplacian1/projected"``) for one operator and Laplace power, which
    #: takes precedence: the same name serves several operators and the counts differ.
    launch_scale: Mapping[str, float] = field(default_factory=dict)
    #: One-off compile seconds of an algorithm, replacing ``compile_s_per_unit * compile_units`` where the executor's compile stage
    #: is not the XLA one (keys as for ``launch_scale``; ``".../<differentiate>"`` takes precedence for a differentiated request).
    compile_seconds: Mapping[str, float] = field(default_factory=dict)

    def predict(self, algorithm: Algorithm, descriptor: Descriptor) -> Prediction:
        w = algorithm.work(descriptor)
        mode = descriptor.request.differentiate
        if mode != "none":
            w = w.training(**dict(algorithm.backward))
            if mode == "params2":
                # a second derivative into the parameters differentiates the backward pass again (about 3x its work)
                w = w.second_derivative()
        request = descriptor.request
        key = f"{request.kind}{request.power}/{algorithm.name}"
        compile_s = self.compile_seconds.get(f"{key}/{mode}", self.compile_seconds.get(key))
        return self.predict_work(
            algorithm.name,
            w,
            descriptor.chain.dtype,
            launch_scale=self.launch_scale.get(key),
            compile_seconds=compile_s,
        )

    def predict_work(
        self,
        name: str,
        w: Work,
        dtype: str = "float64",
        *,
        launch_scale: float | None = None,
        compile_seconds: float | None = None,
    ) -> Prediction:
        """The prediction for a work model already evaluated (what a stored measurement carries).

        ``seconds = eff * (max(gemm / rate, traffic / bandwidth, serial / (rate / units)) + elem / rate_elem)
        + launches * launch_s + solver_calls * solver_s + syncs * sync_s``: throughput terms scale with the efficiency
        factor, while launch, solver and synchronization latencies do not.
        """
        gemm = self.rate_gemm if dtype == "float64" else self.rate_gemm_f32
        core = (
            max(
                w.gemm_flops / gemm,
                w.traffic_bytes / self.bandwidth,
                w.serial_flops / (gemm / max(self.units, 1.0)),
            )
            + w.elem_flops / self.rate_elem
        )
        seconds = (
            self.efficiency.get(name, 1.0) * core
            + w.launches * (self.launch_scale.get(name, 1.0) if launch_scale is None else launch_scale) * self.launch_s
            + w.solver_calls * self.solver_s
            + w.syncs * self.sync_s
        )
        compile_s = w.compile_units * self.compile_s_per_unit if compile_seconds is None else compile_seconds
        return Prediction(seconds, w.peak_bytes, compile_s)

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "device": self.device,
            "rate_gemm": self.rate_gemm,
            "rate_gemm_f32": self.rate_gemm_f32,
            "rate_elem": self.rate_elem,
            "bandwidth": self.bandwidth,
            "launch_s": self.launch_s,
            "compile_s_per_unit": self.compile_s_per_unit,
            "efficiency": dict(self.efficiency),
            "source": self.source,
            "solver_s": self.solver_s,
            "sync_s": self.sync_s,
            "units": self.units,
        }
        if self.launch_scale:  # absent when empty, so the digest of a model without it is unchanged
            config["launch_scale"] = dict(self.launch_scale)
        if self.compile_seconds:
            config["compile_seconds"] = dict(self.compile_seconds)
        return config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CostModel:
        return cls(**dict(config))

    def digest(self) -> str:
        blob = json.dumps(self.to_config(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def for_device(cls, device: str, backend: str | None = None) -> CostModel:
        """Calibration shipped with the package for ``device`` (a device-class key), else the generic defaults.

        The shipped files were measured through the JAX executors, so ``<device>.json`` serves ``backend`` ``None`` or
        ``"jax"``; another backend reads ``<device>_<backend>.json`` and otherwise ranks with the generic defaults
        rather than with another framework's measured constants.
        """
        stem = _device_key(device) + ("" if backend in (None, "", "any", "jax") else f"_{backend}")
        path = Path(__file__).with_name("calibration") / f"{stem}.json"
        if path.exists():
            return cls.from_config(json.loads(path.read_text()))
        on_cpu = _device_key(device).startswith("cpu")
        if backend == "torch" and not on_cpu:
            return cls(
                device=device,
                launch_scale=TORCH_LAUNCH_SCALE,
                compile_seconds=TORCH_COMPILE_SECONDS,
                source="defaults+counted torch launches",
            )
        if on_cpu and backend in (None, "", "any", "jax"):
            return cls(device=device, **CPU_CLASS_PRIOR, source="defaults+cpu class prior")
        return cls(device=device)


#: A prior for one core of a server-class x86 CPU, set from the hardware and not fitted to any measurement: AVX2 FMA at 8 float64
#: flops per cycle on a 2.5 GHz core with about half of peak achieved, twice that in float32, one core's share of memory bandwidth, the
#: cost of an XLA:CPU thunk, a LAPACK call through a custom call, and no device to synchronize with. The generic defaults are A100-scale
#: (5e12 flop/s, 5 us launches, 50 us solver calls), which on a CPU rank by the launch and solver terms where the flops decide. A
#: calibration file, where one ships, replaces it; a CPU with many threads has higher rates and the same ordering of the terms.
CPU_CLASS_PRIOR: Mapping[str, float] = {
    "rate_gemm": 2e10,
    "rate_gemm_f32": 4e10,
    "rate_elem": 2e9,
    "bandwidth": 1e10,
    "launch_s": 1e-6,
    "solver_s": 2e-5,
    "sync_s": 0.0,
    "units": 1.0,
}


#: Counted, not fitted: the CUDA kernels one PyTorch call launches over the kernels the registry counts for the same algorithm
#: (the registry counts the fused XLA programs), the median over the cells ``D = 2..12, H = 32..64, L = 2..4, B = 256`` of one
#: ``torch.profiler`` window on an A100 (``tf32``, ``float32`` and ``float64`` launch the same kernels).  The Laplacian
#: (``laplacian1``): the stream engine, the closed-form tower and the jet tower dispatch each elementwise operation of the
#: tower on its own, the fused Triton kernel is one program per forward call.  A calibration file, where one ships, replaces it.
TORCH_LAUNCH_SCALE: Mapping[str, float] = {
    "laplacian1/projected": 5.25,
    "laplacian1/forward_laplacian": 5.25,
    "laplacian1/taylor_closed": 4.3,
    "laplacian1/taylor_jet": 5.4,
    "laplacian1/nested_autodiff": 3.5,
    "laplacian1/fused_streams": 0.9,
}


#: Measured, not fitted: the fused Triton kernel compiles on its first call with an empty Triton cache, 4.3 to 7.0 s forward and
#: 3.2 to 3.9 s more for the gradient kernel (A100, three fresh processes); the compiled kernels stay in Triton's on-disk cache, so a
#: later process with the same shapes pays milliseconds. The other candidates keep the generic per-unit compile model, which orders
#: them as before.
TORCH_COMPILE_SECONDS: Mapping[str, float] = {
    "laplacian1/fused_streams": 5.5,
    "laplacian1/fused_streams/params": 9.0,
}


def device_class(kind: str) -> str:
    """Calibration class of a device description (``"NVIDIA A100-SXM4-80GB"`` -> ``"a100"``, a CPU -> ``"cpu"``)."""
    low = kind.lower()
    for tag in ("h200", "h100", "a100", "a30", "a10", "l40", "v100", "t4", "p100"):
        if tag in low:
            return tag
    return "cpu" if "cpu" in low or "host" in low else "generic"


def _device_key(device: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in device.lower()).strip("_")


# ------------------------------------------------------------------------------------------------------ plans


@dataclass(frozen=True)
class Plan:
    """A decision: the chosen algorithm, every candidate with its prediction or refusal, and how it was chosen."""

    descriptor: str  # fingerprint
    operator: str
    algorithm: str
    mode: str  # "heuristic" | "find" | "forced"
    device: str
    candidates: tuple[dict[str, Any], ...]
    refusals: tuple[Refusal, ...]
    memory_budget: float | None
    expected_uses: int
    versions: Mapping[str, str]
    measured: Mapping[str, float] = field(default_factory=dict)
    objective: str = "forward"  # "forward" | "params" (value-and-gradient into the parameters)
    uses_source: str = "explicit"  # "explicit" | "hint" (a training API's step count) | "auto" (the default for the objective)

    @property
    def predicted_seconds(self) -> float | None:
        for c in self.candidates:
            if c["name"] == self.algorithm:
                return None if c["seconds"] is None else float(c["seconds"])
        return None

    @property
    def compile_seconds(self) -> float | None:
        for c in self.candidates:
            if c["name"] == self.algorithm:
                return None if c["compile_seconds"] is None else float(c["compile_seconds"])
        return None

    def explain(self) -> dict[str, Any]:
        return {
            "selected_strategy": self.algorithm,
            "mode": self.mode,
            "device": self.device,
            "operator": self.operator,
            "descriptor": self.descriptor,
            "predicted_seconds": self.predicted_seconds,
            "cold_cost_seconds": self.compile_seconds,
            "expected_uses": self.expected_uses,
            "uses_source": self.uses_source,
            "objective": self.objective,
            "candidates": [dict(c) for c in self.candidates],
            "refused": [{"algorithm": r.algorithm, "reason": r.reason} for r in self.refusals],
            "measured_seconds": dict(self.measured),
            "memory_budget_bytes": self.memory_budget,
            "cost_model": self.versions.get("cost_model"),
            "versions": dict(self.versions),
        }

    def to_config(self) -> dict[str, Any]:
        return {
            "format": PLAN_FORMAT,
            "descriptor": self.descriptor,
            "operator": self.operator,
            "algorithm": self.algorithm,
            "mode": self.mode,
            "device": self.device,
            "candidates": [dict(c) for c in self.candidates],
            "refusals": [{"algorithm": r.algorithm, "reason": r.reason} for r in self.refusals],
            "memory_budget": self.memory_budget,
            "expected_uses": self.expected_uses,
            "versions": dict(self.versions),
            "measured": dict(self.measured),
            "objective": self.objective,
            "uses_source": self.uses_source,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Plan:
        config = migrate_plan_config(config)
        return cls(
            descriptor=config["descriptor"],
            operator=config["operator"],
            algorithm=config["algorithm"],
            mode=config["mode"],
            device=config["device"],
            candidates=tuple(dict(c) for c in config["candidates"]),
            refusals=tuple(Refusal(**r) for r in config["refusals"]),
            memory_budget=config["memory_budget"],
            expected_uses=int(config["expected_uses"]),
            versions=dict(config["versions"]),
            measured=dict(config.get("measured", {})),
            objective=str(config["objective"]),
            uses_source=str(config["uses_source"]),
        )


class PlanCache:
    """Persistent plan store keyed by descriptor, device, format and model digest.

    ``path=None`` keeps the cache in memory. Entries written by another :data:`PLAN_FORMAT` or for another calibration
    are never returned (the key changes), so an upgrade re-plans instead of reusing a stale decision.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path is not None and self.path.exists():
            try:
                blob = json.loads(self.path.read_text())
                if blob.get("format") == PLAN_FORMAT:
                    self._entries = dict(blob.get("plans", {}))
            except (OSError, ValueError):
                self._entries = {}

    @staticmethod
    def key(
        descriptor: Descriptor,
        device: str,
        model: CostModel,
        mode: str,
        budget: float | None,
        uses: int,
        backend: str = "any",
        extra: str = "",
    ) -> str:
        """The cache key.  ``extra`` names what else changes the program or its timing (precision, compute dtype, compile
        mode); it is appended only when non-empty, so the keys of plans made under the defaults are unchanged."""
        parts = [
            descriptor.fingerprint(),
            backend,
            _device_key(device),
            f"f{PLAN_FORMAT}",
            model.digest(),
            registry_digest(),
            mode,
            "-" if budget is None else f"{budget:.0f}",
            str(uses),
        ]
        if extra:
            parts.append(extra)
        return "|".join(parts)

    def get(self, key: str) -> Plan | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        try:
            plan = Plan.from_config(entry)
        except (KeyError, ValueError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return plan

    def put(self, key: str, plan: Plan) -> None:
        self._entries[key] = plan.to_config()
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump({"format": PLAN_FORMAT, "plans": self._entries}, handle, sort_keys=True)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._entries)

    # -- the find journal: one measured candidate per line, so a killed ``find`` resumes where it stopped -------------------

    @property
    def journal_path(self) -> Path | None:
        return None if self.path is None else self.path.with_name(self.path.name + ".journal")

    def journal_get(self, key: str, stamp: str) -> dict[str, float]:
        """Seconds measured for the candidates of ``key`` by an earlier, interrupted find (same ``stamp``)."""
        path = self.journal_path
        if path is None or not path.exists():
            return {}
        out: dict[str, float] = {}
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return {}
        for line in lines:
            try:
                entry = json.loads(line)
                if entry["key"] != key:
                    continue
                if entry.get("done"):
                    out = {}  # that find finished and its plan was kept: what came before is history, not progress
                elif entry["stamp"] == stamp and math.isfinite(float(entry["seconds"])):
                    out[str(entry["name"])] = float(entry["seconds"])
            except (ValueError, KeyError, TypeError):
                continue  # a torn or foreign line is skipped, never trusted
        return out

    def _journal_write(self, entry: Mapping[str, Any]) -> None:
        path = self.journal_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            return  # a journal that cannot be written costs the resume, not the plan

    def journal_add(self, key: str, stamp: str, name: str, seconds: float) -> None:
        """Record one successful measurement.  A failed candidate is never recorded: its failure may be transient."""
        if math.isfinite(seconds):
            self._journal_write({"key": key, "stamp": stamp, "name": name, "seconds": float(seconds)})

    def journal_done(self, key: str) -> None:
        """Mark the find of ``key`` finished: its entries no longer count as progress (a fresh find measures again)."""
        self._journal_write({"key": key, "done": True})


# ----------------------------------------------------------------------------------------------------- select

#: Candidates ``find`` times by default, taken in predicted order.  ``top_k=None`` times every feasible candidate.
FIND_TOP_K = 2


class FindRefusal(Exception):
    """Raised by a ``find`` timer for a candidate that cannot run *and would fail the same way again*.

    A compiled size over the memory budget and a candidate that crashes its own process are such failures: they are
    decided by the problem, the budget, the device and the library versions, all of which are in the cache key.
    :func:`select` records the candidate as a refusal in the plan (reason prefixed ``find:``) and keeps the plan, so the
    next call does not pay for the failure again.  A timer reports any other failure (a timeout, a transient error) as an
    infinite time, which keeps the plan out of the cache.
    """


def _find_key_part(versions: Mapping[str, str] | None, top_k: int | None) -> str:
    """What a measured choice depends on beyond the descriptor: the library versions, and how many candidates were timed.

    A find made with another JAX/PyTorch or omnibias release, or by timing every candidate, is another measurement.  The
    default ``top_k`` adds nothing, so keys that differ only by it are unchanged; no ``versions`` adds nothing either.
    """
    parts: list[str] = []
    if versions:
        digest = hashlib.sha256(json.dumps(sorted((str(k), str(v)) for k, v in versions.items())).encode()).hexdigest()
        parts.append("v" + digest[:10])
    if top_k is None:
        parts.append("top_k=all")
    elif int(top_k) != FIND_TOP_K:
        parts.append(f"top_k={max(1, int(top_k))}")
    return "|".join(parts)


def select(
    descriptor: Descriptor,
    *,
    backend: str = "any",
    device: str = "generic",
    cost_model: CostModel | None = None,
    memory_budget: float | None = None,
    mode: str = "heuristic",
    timer: Callable[[str], float] | None = None,
    top_k: int | None = FIND_TOP_K,
    expected_uses: int | str = 1,
    force: str | None = None,
    cache: PlanCache | None = None,
    versions: Mapping[str, str] | None = None,
    candidates: Iterable[Algorithm] | None = None,
    uses_hint: int | None = None,
    key_extra: str = "",
    find_memory_limit: float | None = None,
) -> Plan:
    """Choose an algorithm for ``descriptor``.

    ``mode="heuristic"`` ranks feasible algorithms by predicted ``seconds + compile_seconds / expected_uses``, where the
    seconds are those of the request's objective (forward work, or forward plus backward when
    ``request.differentiate == "params"``). ``mode="find"`` times the ``top_k`` leading candidates through
    ``timer(name) -> seconds`` (steady state, supplied by the backend, which times the same objective) and picks the
    measured best, keeping the predictions for comparison; ``top_k=None`` times every feasible candidate, which is the
    only setting that cannot miss a winner the cost model ranks low (the default of ``2`` trusts the ranking for the
    rest). ``mode="auto"`` is ``find`` once the expected number of uses
    reaches :data:`AUTO_FIND_MIN_USES` and a timer is available, ``heuristic`` otherwise; ``expected_uses="auto"`` takes
    ``uses_hint`` (a training API's step count) or :data:`AUTO_TRAINING_USES` for a training objective (see
    :func:`resolve_planning`). ``force`` names an algorithm to use regardless of rank (it must still be feasible).
    Refusals (not applicable, over the memory budget, inexact for an exact request, an accelerator program on a CPU) are
    recorded with their reason; if nothing is feasible a ``ValueError`` lists every reason.

    ``find`` times the leading candidate first and the others against what it measured: a timer that takes a ``deadline``
    argument (seconds per run) may stop a candidate after one steady-state run slower than ``FIND_CUT_FACTOR`` times the best
    time so far, which cannot change the minimum; a candidate after the leader whose *predicted* peak exceeds
    ``find_memory_limit`` (bytes) is not run at all (it stays ranked and is marked in its ``candidates`` row).  The seconds of
    every successful measurement are journaled beside a persistent ``cache`` so an interrupted find resumes; a failed
    candidate is never journaled.  A timer that raises :class:`FindRefusal` for a failure that repeats (a compiled size over
    the budget, a crash of the candidate's own process) has that candidate recorded as a refusal and the plan is cached; a
    timer that returns an infinite time for a transient failure keeps the plan out of the cache.  ``key_extra`` extends
    the cache key with what else changes the measurement (precision, compute dtype, compile mode); a ``find`` key also
    carries the library ``versions`` and a non-default ``top_k``, and the use count of a ``find`` key is rounded to one
    significant digit, since a measured choice does not depend on it more finely.
    """
    if mode not in ("heuristic", "find", "auto"):
        raise ValueError("mode must be 'heuristic', 'find' or 'auto'")
    if mode == "find" and timer is None and force is None:
        raise ValueError("mode='find' needs a timer")
    mode, expected_uses, uses_source = resolve_planning(
        mode,
        expected_uses,
        differentiate=descriptor.request.differentiate,
        uses_hint=uses_hint,
        can_time=timer is not None,
    )
    model = cost_model or CostModel.for_device(device, backend)
    key = None
    if cache is not None and candidates is None:
        find_part = _find_key_part(versions, top_k) if mode == "find" and not force else ""
        key = PlanCache.key(
            descriptor,
            device,
            model,
            "forced:" + force if force else mode,
            memory_budget,
            _bucket_uses(expected_uses) if mode == "find" else expected_uses,
            backend,
            "|".join(p for p in (key_extra, find_part) if p),
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
    pool = (
        list(candidates)
        if candidates is not None
        else algorithms_for(descriptor.request.kind, backend)
    )
    if not pool:
        raise ValueError(
            f"no algorithm is registered for operator {descriptor.request.kind!r} on backend {backend!r}"
        )
    feasible: list[tuple[Algorithm, Prediction]] = []
    refusals: list[Refusal] = []
    for algo in pool:
        if descriptor.request.exact and not algo.exact:
            refusals.append(
                Refusal(algo.name, "stochastic estimator; the request asks for an exact result")
            )
            continue
        if algo.accelerator_only and device == "cpu":
            refusals.append(
                Refusal(algo.name, "a kernel program for accelerators; the device class is a CPU")
            )
            continue
        reason = algo.refusal(descriptor, backend)
        if reason is None and algo.hardware_requires is not None:
            reason = algo.hardware_requires(descriptor, backend, device)
        if reason is None and algo.forced_only is not None and force != algo.name:
            reason = algo.forced_only(descriptor, backend, device)
        if reason is not None:
            refusals.append(Refusal(algo.name, reason))
            continue
        pred = model.predict(algo, descriptor)
        if memory_budget is not None and pred.peak_bytes > memory_budget:
            refusals.append(
                Refusal(
                    algo.name,
                    f"memory_budget: predicted peak {pred.peak_bytes / 2**20:.0f} MiB exceeds {memory_budget / 2**20:.0f} MiB",
                )
            )
            continue
        feasible.append((algo, pred))
    if not feasible:
        raise ValueError(
            "no algorithm is feasible for this descriptor: "
            + "; ".join(f"{r.algorithm}: {r.reason}" for r in refusals)
        )
    uses = max(1, int(expected_uses))
    feasible.sort(key=lambda ap: ap[1].seconds + ap[1].compile_seconds / uses)
    rows = [
        {
            "name": a.name,
            "seconds": p.seconds,
            "peak_bytes": p.peak_bytes,
            "compile_seconds": p.compile_seconds,
            "generic": a.generic,
        }
        for a, p in feasible
    ]
    measured: dict[str, float] = {}
    find_notes: dict[str, str] = {}
    chosen = feasible[0][0].name
    used_mode = mode
    if force is not None:
        names = {a.name for a, _ in feasible}
        if force not in names:
            why = next(
                (r.reason for r in refusals if r.algorithm == force), "unknown or not feasible"
            )
            raise ValueError(f"cannot force {force!r}: {why}")
        chosen, used_mode = force, "forced"
    elif mode == "find":
        assert timer is not None
        stamp = json.dumps(sorted((versions or {}).items()))
        done = cache.journal_get(key, stamp) if cache is not None and key is not None else {}
        bounded = _accepts_deadline(timer)
        timed = feasible if top_k is None else feasible[: max(1, int(top_k))]
        find_refused: dict[str, str] = {}
        for rank, (algo, pred) in enumerate(timed):
            finite = [t for t in measured.values() if math.isfinite(t)]
            best = min(finite) if finite else None
            if algo.name in done:
                measured[algo.name] = done[algo.name]
                find_notes[algo.name] = "measured by an earlier, interrupted find (journal)"
                continue
            if rank > 0 and best is not None and find_memory_limit is not None and pred.peak_bytes > find_memory_limit:
                find_notes[algo.name] = (
                    f"not timed: predicted peak {pred.peak_bytes / 2**30:.3g} GiB exceeds the "
                    f"{find_memory_limit / 2**30:.3g} GiB find may use"
                )
                continue
            deadline = None if best is None or not bounded else FIND_CUT_FACTOR * best
            try:
                seconds = float(timer(algo.name, deadline=deadline) if deadline is not None else timer(algo.name))
            except FindRefusal as err:
                find_refused[algo.name] = str(err)
                refusals.append(Refusal(algo.name, f"find: {err}"))
                continue
            measured[algo.name] = seconds
            if deadline is not None and seconds > deadline:
                find_notes[algo.name] = f"stopped after one run above {FIND_CUT_FACTOR:g}x the best time: it cannot win"
            elif cache is not None and key is not None:
                cache.journal_add(key, stamp, algo.name, seconds)
        if not any(math.isfinite(t) for t in measured.values()):
            raise ValueError(
                "find: every timed candidate failed to run ("
                + ", ".join(sorted({*measured, *find_refused}))
                + "); a timer reports a failure as an infinite time"
                + "".join(f"; {name}: {why}" for name, why in find_refused.items())
            )
        chosen = min(measured, key=lambda n: measured[n])
        for name, why in find_refused.items():
            find_notes[name] = f"refused when timed: {why}"
        for row in rows:
            if row["name"] in find_notes:
                row["find"] = find_notes[row["name"]]
    plan = Plan(
        descriptor=descriptor.fingerprint(),
        operator=descriptor.request.kind,
        algorithm=chosen,
        mode=used_mode,
        device=device,
        candidates=tuple(rows),
        refusals=tuple(refusals),
        memory_budget=memory_budget,
        expected_uses=uses,
        versions={**(versions or {}), "cost_model": model.source},
        measured=measured,
        objective=descriptor.request.differentiate
        if descriptor.request.differentiate != "none"
        else "forward",
        uses_source=uses_source,
    )
    if cache is not None and key is not None and all(math.isfinite(t) for t in measured.values()):
        # a plan with a candidate that failed *transiently* (an infinite time) is not kept; a failure that repeats was
        # raised as FindRefusal, is a refusal of the plan, and does not stop it from being kept
        cache.put(key, plan)
        if mode == "find":
            cache.journal_done(key)
    return plan


#: ``find`` stops a candidate after one steady-state run slower than this many times the best time measured so far: its
#: median over the remaining runs would have to be faster by that factor than a run just taken, which noise does not do.
FIND_CUT_FACTOR = 4.0


def _accepts_deadline(timer: Callable[..., float]) -> bool:
    try:
        params = inspect.signature(timer).parameters
    except (TypeError, ValueError):
        return False
    return "deadline" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _bucket_uses(uses: int | str) -> int:
    """``uses`` rounded to one significant digit (1000 stays 1000, 1234 becomes 1000, 1600 becomes 2000)."""
    n = max(1, int(uses))
    if n < 10:
        return n
    scale = 10 ** (len(str(n)) - 1)
    return max(1, int(round(n / scale)) * scale)


def measurement_context(matmul: str = "highest", compute_dtype: str | None = None, compile: str | None = None) -> str:
    """The part of a ``find`` cache key that names how the program is run: the float32 matmul mode, the compute dtype and
    the compile mode.  Empty under the defaults, so a plan measured under them keeps the key it always had."""
    parts = []
    if matmul != "highest":
        parts.append(f"matmul={matmul}")
    if compute_dtype is not None:
        parts.append(f"compute={compute_dtype}")
    if compile is not None:
        parts.append(f"compile={compile}")
    return ";".join(parts)


def host_memory_limit(fraction: float = 0.5) -> float | None:
    """Bytes of host memory a timed candidate may be predicted to need: ``fraction`` of the smaller of the physical memory
    and the container limit, or ``None`` where neither can be read.  A scheduler's per-job limit is not visible here."""
    limits: list[float] = []
    try:
        limits.append(float(os.sysconf("SC_PHYS_PAGES")) * float(os.sysconf("SC_PAGE_SIZE")))
    except (ValueError, OSError, AttributeError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2**60:
            limits.append(float(text))
    return fraction * min(limits) if limits else None


#: Calls from which a measured choice is worth its timing cost: ``mode="auto"`` finds at or above this many uses.
AUTO_FIND_MIN_USES = 50
#: Calls assumed for a training objective when no step count is known (``expected_uses="auto"``).
AUTO_TRAINING_USES = 1000


def resolve_planning(
    mode: str,
    expected_uses: int | str,
    *,
    differentiate: str = "none",
    uses_hint: int | None = None,
    can_time: bool = True,
) -> tuple[str, int, str]:
    """``(mode, uses, uses_source)`` after resolving ``"auto"``.

    ``expected_uses="auto"`` is ``uses_hint`` when a training API passes its step count (source ``"hint"``), else
    :data:`AUTO_TRAINING_USES` for ``differentiate="params"`` and 1 for a forward evaluation (source ``"auto"``);
    an integer is kept (source ``"explicit"``). ``mode="auto"`` is ``"find"`` when ``uses >= AUTO_FIND_MIN_USES`` and
    ``can_time``, else ``"heuristic"``; the other modes are kept.
    """
    if isinstance(expected_uses, str):
        if expected_uses != "auto":
            raise ValueError("expected_uses must be a positive integer or 'auto'")
        if uses_hint is not None:
            uses, source = max(1, int(uses_hint)), "hint"
        else:
            uses = AUTO_TRAINING_USES if differentiate == "params" else 1
            source = "auto"
    else:
        if isinstance(expected_uses, bool) or int(expected_uses) < 1:
            raise ValueError("expected_uses must be a positive integer or 'auto'")
        uses, source = int(expected_uses), "explicit"
    if mode == "auto":
        mode = "find" if uses >= AUTO_FIND_MIN_USES and can_time else "heuristic"
    elif mode not in ("heuristic", "find"):
        raise ValueError("mode must be 'heuristic', 'find' or 'auto'")
    return mode, uses, source


def default_plan_cache_path() -> Path | None:
    """The persistent plan cache ``mode="auto"`` uses once it finds: ``$OMNIBIAS_PLAN_CACHE`` (``off`` disables it),
    else ``plans.json`` under ``$OMNIBIAS_CACHE_DIR`` or ``~/.cache/omnibias``."""
    env = os.environ.get("OMNIBIAS_PLAN_CACHE")
    if env is not None:
        return None if env.strip().lower() in ("", "off", "0", "none") else Path(env)
    root = os.environ.get("OMNIBIAS_CACHE_DIR")
    return (Path(root) if root else Path.home() / ".cache" / "omnibias") / "plans.json"


def estimator_probes(
    name: str, accuracy: float | None, m: int, *, default: int = 16, cap: int = 4096
) -> int:
    """Samples a stochastic Laplacian estimator draws to reach a worst-case standard error ``accuracy``.

    The error is a fraction of the Hessian's Frobenius norm. ``hutchinson`` (Rademacher) and ``stde`` (Gaussian directions)
    average ``n`` quadratic forms ``v^T H v``, whose variance is at most ``2 ||H||_F^2 / n``: ``n = ceil(2 / accuracy^2)``.
    ``sdgd`` samples ``s`` of the ``m`` coordinates without replacement and rescales by ``m / s``, with standard error at
    most ``sqrt(m / s - 1) ||H||_F``: ``s = ceil(m / (1 + accuracy^2))``. Without a target the count is ``default``
    (at most ``m`` for ``sdgd``). The result is at least 1 and at most ``cap``.
    """
    if name == "sdgd":
        s = default if accuracy is None else math.ceil(m / (1.0 + float(accuracy) ** 2))
        return max(1, min(int(s), m, cap))
    if name in ("hutchinson", "stde"):
        n = default if accuracy is None else math.ceil(2.0 / float(accuracy) ** 2)
        return max(1, min(int(n), cap))
    raise ValueError(f"unknown estimator {name!r}")


def bytes_of(dtype: str) -> int:
    return _DTYPE_BYTES[dtype]


def predicted_regret(plan: Plan, measured: Mapping[str, float]) -> float:
    """Ratio of the measured time of the chosen algorithm to the best measured time among ``measured`` (>= 1)."""
    best = min(measured.values())
    return measured[plan.algorithm] / best if best > 0 and math.isfinite(best) else float("nan")


def dispatch_candidates(
    operator: str, backend: str, sequence: Sequence[str] | None = None
) -> list[Algorithm]:
    """Registered algorithms of ``operator`` restricted to ``sequence`` names (in that order) when given."""
    pool = algorithms_for(operator, backend)
    if sequence is None:
        return pool
    by_name = {a.name: a for a in pool}
    return [by_name[n] for n in sequence if n in by_name]


# The built-in algorithms register themselves on import (kept in their own module: formulas, not mechanism).
from omnibias.core import plan_algorithms as _plan_algorithms  # noqa: E402,F401

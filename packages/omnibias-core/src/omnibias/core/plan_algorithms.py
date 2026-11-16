# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Built-in algorithms of the planner: refusal rules and work models, one entry per executor name.

An executor (JAX, PyTorch) implements an algorithm under the same name. The work model counts the dominant GEMM flops,
elementwise flops, memory traffic and peak activations of one evaluation of an MLP chain; constants (rates, one efficiency
factor per algorithm) live in the per-device calibration, not here. The models state the *growth* of each algorithm with
the dimension ``D``, the width ``H``, the batch and the stream count, which is what decides the ranking.

Stream counts used below (``k`` seed columns after the first layer, ``q`` drift directions):

* second order: ``1 + (k + q) + 1`` (value, tangents, Laplacian);
* fourth order: ``1 + k + q + k(k+1)/2 + 1 + k + 1`` (value, tangents, packed Hessian, Laplacian, gradient of the Laplacian,
  bi-Laplacian), as in ``omnibias.jax.experimental.projected_bilaplacian``;
* sixth order (tri-Laplacian by composition): ``k/2 + 1`` times the fourth-order count.

The stacked XLA stream engine (``projected``, Laplacian) carries ``r + 2 + q`` channels in diagonal form (``r`` metric
columns, at most ``2 H_1``) and ``2 H_1 + 2 + q`` in Gram form; the Cholesky-basis variant keeps ``min(r, H_1) + 2 + q``.

Kernel launches per template come from the lowering probe (``omnibias_experiments/percall_*/DIAGNOSIS.md``: compiled HLO of
each program at three dense layers on an A100): ``(forward base, forward per dense layer, backward base, backward per
layer)``. They are latency terms the flop counts do not show, and what separates the programs at low ``D``, ``H`` and ``B``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import comb

from omnibias.core.planner import (
    Algorithm,
    Descriptor,
    Work,
    bytes_of,
    estimator_probes,
    register_algorithm,
)

#: Activations with closed-form derivative kernels of every order in the JAX/PyTorch registries.
KNOWN_ACTIVATIONS = frozenset(
    {
        "abs", "arctan", "atan", "celu", "cos", "cosh", "cot", "coth", "ctan", "ctg", "elu", "exp", "gaussian", "gelu",
        "hardshrink", "hardsigmoid", "hardswish", "hardtanh", "heaviside", "huber", "leaky_relu", "log1pu2", "log_cosh",
        "logcosh", "logcosh_dual", "logistic", "mish", "prelu", "rbf", "relu", "relu6", "sech", "selu", "sigmoid", "sign",
        "silu", "sin", "sinh", "smooth_sign", "soft_abs", "soft_heaviside", "soft_relu", "soft_sign", "soft_step",
        "softabs", "softplus", "softshrink", "softsign", "step", "swish", "tan", "tanh", "threshold",
    }
)  # fmt: skip
#: Activations whose second derivative vanishes almost everywhere (a closed-form Laplacian would be a confident zero).
AE_ZERO_SECOND = frozenset(
    {"relu", "leaky_relu", "prelu", "relu6", "hardtanh", "hardsigmoid", "hardshrink", "threshold", "abs", "sign", "step",
     "softshrink"}
)  # fmt: skip
#: Activations whose derivatives up to order three jump (the bi-Laplacian would drop a singular part).
NOT_C3 = AE_ZERO_SECOND | frozenset({"celu", "elu", "hardswish", "huber", "selu", "softsign"})

_PROJECTED_SECOND = {"laplacian": (1, 2, 3)}

#: ``(forward base, forward per dense layer, extra backward base, extra backward per layer)`` launches per program template.
_LAUNCH: dict[str, tuple[float, float, float, float]] = {
    "projected": (2.0, 2.0, 0.0, 6.0),
    "projected_cholesky": (6.0, 2.0, 6.0, 7.0),
    "forward_laplacian": (2.0, 2.0, 0.0, 6.7),
    "forward_polylaplacian": (4.0, 2.5, 0.0, 7.0),
    "taylor_jet": (5.0, 3.0, 0.0, 11.0),
    "nested_autodiff": (4.0, 3.0, 0.0, 11.0),
    "taylor_closed": (3.0, 2.0, 0.0, 6.0),
    "fused_streams": (9.0, 0.0, 2.0, 3.0),
    "engine": (1.0, 3.0, 0.0, 9.0),
    "estimator": (5.0, 3.0, 0.0, 11.0),
    "first_order": (2.0, 2.0, 0.0, 6.0),
    "residual": (14.0, 4.0, 10.0, 8.0),
}


def _launches(template: str, d: Descriptor) -> tuple[float, float]:
    base, per, bbase, bper = _LAUNCH[template]
    n = len(d.chain.layers)
    return base + per * n, bbase + bper * n


#: Activations of the fused kernels' towers (``omnibias.jax.experimental.fused_streams.FUSED_ACTIVATIONS``).
FUSED_ACTIVATIONS = frozenset({"tanh", "sigmoid", "logistic", "sin", "cos", "softplus"})
FUSED_MAX_WIDTH = 128
FUSED_MAX_CHANNELS = 24
FUSED_TILE = 16  # points per program
#: Where the Triton kernel on PyTorch (float32, A100) was measured faster than the stream engine in the forward pass and in
#: eager value-and-gradient; the planner ranks it only there (``force`` still runs it anywhere it can).
FUSED_TORCH_MAX_WIDTH = 64
FUSED_TORCH_MAX_D = 16
#: Per-block shared memory of an sm80-class GPU (A100/A30) as the Triton lowering reports it; larger parts only gain headroom.
FUSED_SHARED_MEMORY_LIMIT = 166912
#: The support-grouped polylaplacian traces one local jet per support set in Python (measured: about 0.15 s each) and unrolls
#: them into one program; past this many the trace alone takes many minutes and the program size no longer fits a compiler.
POLYLAPLACIAN_MAX_SUPPORTS = 2000
#: The closed-form Taylor streams of ``Delta^2`` stack ``4 m^2 + 1`` channels for ``m`` selected inputs.  Measured: 16,385
#: channels (``m = 64``) compile and run; 262,145 (``m = 256``) held more than 98 GiB of host memory for 40 minutes without
#: finishing a compile, on JAX and on PyTorch.  The cap is the largest size measured to compile; above it the planner uses
#: ``projected``.
TAYLOR_CLOSED_MAX_CHANNELS = 16_385
#: Safety factor on the calibrated bytes (JAX/Pallas: measured within 3% on 35 of 36 compile outcomes; Torch/Triton: two
#: failure points only, 11% above the Pallas model, so a wider margin until it is calibrated).
_FUSED_SMEM_SLACK = {"jax": 1.05, "torch": 1.15}


# ----------------------------------------------------------------------------------------------- helpers


def _pairs(d: Descriptor) -> list[tuple[int, int]]:
    return [(layer.in_features, layer.out_features) for layer in d.chain.layers]


def _flops_forward(d: Descriptor) -> float:
    return 2.0 * sum(i * o for i, o in _pairs(d))


def _h1(d: Descriptor) -> int:
    return d.chain.layers[0].out_features


def _seed(d: Descriptor) -> int:
    """Columns the first-layer projection keeps: ``min(selected inputs, H_1)``."""
    return min(d.selected, _h1(d))


def _streams2(s: int, q: int) -> int:
    return 1 + s + q + 1


def _streams4(k: int, q: int) -> int:
    return 1 + k + q + k * (k + 1) // 2 + 1 + k + 1


def _streams6(k: int, q: int) -> int:
    return int(_streams4(k, q) * (k / 2.0 + 1.0))


def _stream_count(d: Descriptor, k: int) -> int:
    q = d.request.directions
    power = d.request.power
    if power == 1:
        return _streams2(k, q)
    if power == 2:
        return _streams4(k, q)
    return _streams6(k, q)


def _streamed_work(
    d: Descriptor,
    k: int,
    *,
    setup: float = 0.0,
    first_gemm: bool = True,
    template: str = "projected",
    solver_calls: float = 0.0,
    streams: int | None = None,
) -> Work:
    """GEMM/traffic of pushing ``S`` streams through the layers after the first (the first is a single GEMM on ``x``)."""
    B, bts = d.batch, bytes_of(d.chain.dtype)
    pairs = _pairs(d)
    S = _stream_count(d, k) if streams is None else streams
    gemm = (2.0 * B * pairs[0][0] * pairs[0][1] if first_gemm else 0.0) + setup
    elem = traffic = 0.0
    widest = max(o for _, o in pairs)
    for idx, (i, o) in enumerate(pairs):
        if idx > 0:
            gemm += 2.0 * B * i * o * (S - 1)
        traffic += 2.0 * B * o * S * bts
        elem += B * o * (S + 4) * (6.0 if d.request.power == 1 else 8.0)
    peak = 3.0 * B * widest * S * bts + d.D * _h1(d) * bts
    fwd, bwd = _launches(template, d)
    return Work(
        gemm,
        elem,
        traffic,
        peak,
        launches=fwd,
        compile_units=1.0 + 0.1 * len(pairs),
        launches_backward=bwd,
        solver_calls=solver_calls,
    )


def _engine_channels(d: Descriptor) -> int:
    """Channels of the stacked XLA stream engine: ``r + 2 + q`` in diagonal form (``r <= 2 H_1``), ``2 H_1 + 2 + q`` in Gram form."""
    r, H, q = d.selected, _h1(d), d.request.directions
    return r + 2 + q if r <= 2 * H else 2 * H + 2 + q


def _hidden_activations(d: Descriptor) -> list[str | None]:
    return [layer.activation for layer in d.chain.layers[:-1]]


def _chain_ok(d: Descriptor) -> str | None:
    layers = d.chain.layers
    if len(layers) < 2:
        return "the chain needs at least one hidden layer and an affine readout"
    if layers[-1].activation is not None:
        return "the last layer must be affine"
    unknown = sorted(
        {str(a) for a in _hidden_activations(d) if a is None or a not in KNOWN_ACTIVATIONS}
    )
    if unknown:
        return f"no closed-form derivative kernel for activation(s) {unknown}"
    return None


def _smooth_ok(d: Descriptor, power: int) -> str | None:
    if d.request.allow_almost_everywhere:
        return None
    names = set(_hidden_activations(d))
    bad = names & (AE_ZERO_SECOND if power == 1 else NOT_C3 if power >= 2 else set())
    if bad:
        what = (
            "an a.e.-zero second derivative"
            if power == 1
            else "derivatives that jump below order four"
        )
        return f"activation(s) {sorted(str(b) for b in bad)} have {what}; pass allow_almost_everywhere=True to opt in"
    return None


def _power_in(d: Descriptor, powers: tuple[int, ...]) -> str | None:
    return (
        None
        if d.request.power in powers
        else f"serves Laplace powers {powers}, not {d.request.power}"
    )


# ------------------------------------------------------------------------------------------------ Laplacians


def _torch_no_projected_streams(d: Descriptor) -> str | None:
    return (
        "the PyTorch backend does not port the first-layer projected stream program; "
        "full_streams carries the input dimension as tangent columns and autodiff_rows is the generic route"
    )


def _req_projected(d: Descriptor) -> str | None:
    if d.request.power == 3 and d.request.metric != "euclidean":
        return "the tri-Laplacian program takes the Euclidean metric on the selected inputs only"
    return _chain_ok(d) or _power_in(d, (1, 2, 3)) or _smooth_ok(d, d.request.power)


def _work_projected(d: Descriptor) -> Work:
    if d.request.power == 1:
        # the stacked engine: no factorization, no conditional; the metric enters only through W_1 S
        m, H1 = d.selected, _h1(d)
        return _streamed_work(
            d, 0, streams=_engine_channels(d), setup=2.0 * m * H1 * d.selected, template="projected"
        )
    m, H1 = d.selected, _h1(d)
    k = _seed(d)
    setup = (
        4.0 * m * H1 * H1
    )  # Gram, Cholesky/QR of the first-layer factor: once per call, independent of the batch
    return _streamed_work(d, k, setup=setup, solver_calls=3.0)


def _req_cholesky(d: Descriptor) -> str | None:
    if d.request.power != 1:
        return "the Cholesky-basis program serves the Laplacian (power 1); higher powers use `projected`"
    return _chain_ok(d) or _smooth_ok(d, 1)


def _work_cholesky(d: Descriptor) -> Work:
    """First-layer Cholesky basis: ``min(r, H_1) + 2 + q`` channels, but a Gram GEMM, a factorization and a conditional per call."""
    m, H1 = d.selected, _h1(d)
    k = _seed(d)
    return _streamed_work(
        d, k, setup=4.0 * m * H1 * H1, template="projected_cholesky", solver_calls=3.0
    )


def _req_fused(d: Descriptor) -> str | None:
    r = d.request
    if r.power != 1:
        return "the fused kernels serve the Laplacian (power 1)"
    if r.directions:
        return "point drift streams are not fused"
    if r.metric == "point_diagonal":
        return "a per-point metric is not fused"
    if r.differentiate == "params2":
        return (
            "the fused kernels differentiate once into the parameters; a second derivative needs "
            "the XLA stream engine"
        )
    reason = _chain_ok(d) or _smooth_ok(d, 1)
    if reason is not None:
        return reason
    names = {str(a) for a in _hidden_activations(d)}
    bad = sorted(names - FUSED_ACTIVATIONS)
    if bad:
        return f"no fused kernel tower for activation(s) {bad}"
    widths = [layer.out_features for layer in d.chain.layers[:-1]]
    if max(widths) > FUSED_MAX_WIDTH:
        return f"hidden width {max(widths)} exceeds the fused limit {FUSED_MAX_WIDTH}"
    channels = d.selected + 2
    if channels > FUSED_MAX_CHANNELS:
        return f"{channels} stream channels exceed the fused limit {FUSED_MAX_CHANNELS}"
    if d.chain.dtype not in ("float32", "float64"):
        return f"the fused kernels run float32 or float64, not {d.chain.dtype}"
    return None


def _pad_p2(n: int, floor: int = FUSED_TILE) -> int:
    p = floor
    while p < n:
        p *= 2
    return p


def fused_shared_memory_bytes(
    *,
    dtype: str,
    width: int,
    channels: int,
    block: int = FUSED_TILE,
    backward: bool = False,
    backend: str = "jax",
) -> int:
    """Static bound on the shared memory one fused-kernel program needs (A100 calibration, two hidden layers).

    ``P = pad(width)`` is the padded hidden width, ``S`` the stream channels, ``BB`` the points per program.  The measured
    compile outcomes (``docs/benchmarks/fused_shared_memory.json``) fit ``esize * (P^2 + BB * P * c(S))`` with
    ``c = S`` for the float32 forward, ``2S + 1`` for the float32 backward and ``S + 1`` for the float64 backward; the
    float64 forward never exceeded the limit (its dots stay out of shared memory, no tensor-core path).  A tile larger
    than the default can need more than the model says (one float64 cell at ``BB = 32``, ``P = 32`` under-predicts), so
    the planner and the kernels evaluate it at the smallest tile and the kernels shrink an unnamed block to fit.
    """
    if dtype == "float64" and not backward:
        return 0
    esize = 4 if dtype == "float32" else 8
    P = _pad_p2(width)
    if not backward:
        c = channels
    else:
        c = 2 * channels + 1 if dtype == "float32" else channels + 1
    return int(_FUSED_SMEM_SLACK.get(backend, 1.15) * esize * (P * P + block * P * c))


def _fused_hardware(d: Descriptor, backend: str, device: str) -> str | None:
    """Refuse a configuration whose smallest tile still exceeds the shared memory the lowering allows."""
    if device == "cpu-interpret":
        return None
    widths = [layer.out_features for layer in d.chain.layers[:-1]]
    need = fused_shared_memory_bytes(
        dtype=d.chain.dtype,
        width=max(widths),
        channels=d.selected + 2,
        block=FUSED_TILE,
        backward=d.request.differentiate != "none",
        backend=backend,
    )
    if need > FUSED_SHARED_MEMORY_LIMIT:
        phase = "backward" if d.request.differentiate != "none" else "forward"
        return (
            f"the fused {phase} kernel needs about {need} B of shared memory at the smallest tile "
            f"({FUSED_TILE} points) against the {FUSED_SHARED_MEMORY_LIMIT} B the lowering allows (width "
            f"{max(widths)}, {d.selected + 2} channels, {d.chain.dtype}); the stream engine has no such limit"
        )
    return None


def _fused_forced_only(d: Descriptor, backend: str, device: str) -> str | None:
    """Where the measured grid says the fused kernel loses, it runs only when forced (still listed in the refusals)."""
    if (
        device == "cpu-interpret"
    ):  # the Pallas/Triton interpreter: tests and debugging, no performance claim
        return None
    if d.chain.dtype == "float64":
        return (
            "float64 has no tensor-core path in the Triton dot: the fused kernel measured 3-8x slower than the "
            "stacked-stream program on an A100; name it with force to use it"
        )
    if backend == "jax":
        return (
            "opt-in on JAX: the Pallas kernel measured slower than the XLA stream engine in every cell of the A100 "
            "exploration (forward 25-42 us against 12-14 us, value-and-gradient 100-250 us against 66-160 us); "
            "name it with force to use it"
        )
    if backend == "torch":
        width = max(layer.out_features for layer in d.chain.layers[:-1])
        if width > FUSED_TORCH_MAX_WIDTH or d.D > FUSED_TORCH_MAX_D:
            return (
                f"the Triton kernel is measured against the stream engine for hidden width <= {FUSED_TORCH_MAX_WIDTH} and "
                f"D <= {FUSED_TORCH_MAX_D} (float32, A100); width {width}, D = {d.D} is outside that region (at width 128 "
                "it ran 3 to 26 times slower in the forward pass); name it with force to use it"
            )
    return None


def _work_fused(d: Descriptor) -> Work:
    """One kernel: weights and streams stay in registers, so traffic is the inputs, outputs and weights once.

    A tile of ``FUSED_TILE`` points is one program on one multiprocessor, so a small batch cannot spread: the program's own
    sequential work (``serial_flops``) is the latency floor the measurements show at small ``B``.
    """
    B, bts = d.batch, bytes_of(d.chain.dtype)
    pairs = _pairs(d)
    S = d.selected + 2
    P = max(16, max(o for _, o in pairs[:-1]))
    gemm = 2.0 * B * pairs[0][0] * pairs[0][1]
    serial = 0.0
    elem = 0.0
    for idx, (i, o) in enumerate(pairs):
        if idx > 0:
            gemm += 2.0 * B * i * o * (S - 1)
            serial += 2.0 * FUSED_TILE * i * o * S
        elem += B * o * (S + 4) * 6.0
        serial += (
            FUSED_TILE * max(o, 16) * (S + 4) * 6.0 * 8.0
        )  # elementwise tower: ~8x the flop count in latency
    params = sum(i * o for i, o in pairs) * bts
    traffic = B * (d.D + S * pairs[-1][1]) * bts + params * max(B // FUSED_TILE, 1) * 0.1
    peak = B * (d.D + S * pairs[-1][1]) * bts + params + S * FUSED_TILE * P * bts
    fwd, bwd = _launches("fused_streams", d)
    return Work(
        gemm,
        elem,
        traffic,
        peak,
        launches=fwd,
        compile_units=1.5 + 0.1 * len(pairs),
        launches_backward=bwd,
        serial_flops=serial,
    )


def _req_forward(d: Descriptor) -> str | None:
    r = d.request
    if r.selected is not None or r.metric != "euclidean" or r.directions:
        return "computes the full-input Euclidean Laplacian only (no coordinate subset, metric or drift)"
    return _chain_ok(d) or _power_in(d, (1,)) or _smooth_ok(d, 1)


def _work_forward(d: Descriptor) -> Work:
    return _streamed_work(d, d.selected, template="forward_laplacian")


def _req_polylaplacian(d: Descriptor) -> str | None:
    r = d.request
    if r.selected is not None or r.metric != "euclidean" or r.directions:
        return "computes the full-input Euclidean Laplace power only (no coordinate subset, metric or drift)"
    reason = _chain_ok(d) or _power_in(d, (2, 3)) or _smooth_ok(d, d.request.power)
    if reason is not None:
        return reason
    supports = _polylaplacian_supports(d.selected, d.request.power)
    if supports > POLYLAPLACIAN_MAX_SUPPORTS:
        return (
            f"unrolls {supports} support jets (their number grows like D^{d.request.power}) against the "
            f"{POLYLAPLACIAN_MAX_SUPPORTS} the program can trace in minutes; use `projected`"
        )
    return None


def _polylaplacian_supports(m: int, p: int) -> int:
    """Support sets of ``Delta^p`` over ``m`` axes: ``sum_{s <= p} C(m, s)`` (one local jet each)."""
    return sum(comb(m, s) for s in range(1, min(p, m) + 1))


def _work_polylaplacian(d: Descriptor) -> Work:
    """Support-grouped local jets: ``sum_{s <= p} C(m, s)`` jets of order ``2p``, each in ``s`` variables.

    ``Delta^p`` groups its ``|a| = p`` terms by support (``omnibias.core.contraction.polylaplacian_support_terms``): one
    multivariate jet of dimension ``s <= p`` per support, ``C(s + 2p, 2p)`` coefficients each, composed through every
    activation by truncated series products (about ``2p`` products of ``N^2`` multiplications per unit). The number of
    supports grows like ``m^p``, and the program unrolls them, so trace and compile size grow with it as well: that is what
    makes this program lose to the projected ones as soon as ``m`` is more than a few, however few the streams per support.
    """
    B, bts = d.batch, bytes_of(d.chain.dtype)
    m, p = d.selected, d.request.power
    order = 2 * p
    pairs = _pairs(d)
    gemm = elem = traffic = 0.0
    supports = _polylaplacian_supports(m, p)
    widest_jet = 1
    for s in range(1, min(p, m) + 1):
        count = comb(m, s)
        n_coef = comb(s + order, order)
        widest_jet = max(widest_jet, n_coef)
        for idx, (i, o) in enumerate(pairs):
            if idx == 0:
                gemm += count * 2.0 * B * s * o  # the restricted first layer reads s columns
                continue
            gemm += count * 2.0 * B * i * o * n_coef
            traffic += count * 2.0 * B * o * n_coef * bts
            elem += count * B * o * order * n_coef * n_coef
    widest = max(o for _, o in pairs)
    fwd, bwd = _launches("forward_polylaplacian", d)
    return Work(
        gemm,
        elem,
        traffic,
        3.0 * B * widest * widest_jet * bts + d.D * _h1(d) * bts,
        launches=fwd * supports,
        compile_units=1.0 + 0.05 * len(pairs) * supports,
        launches_backward=bwd * supports,
    )


def _req_generic(d: Descriptor) -> str | None:
    return None


def _req_jet(d: Descriptor) -> str | None:
    if d.request.power > 1:
        return "the generic jet executor serves the Laplacian; higher Laplace powers use nested autodiff or a structured program"
    return None


def _jet_work(d: Descriptor, directions: float, template: str) -> Work:
    B, bts, p = d.batch, bytes_of(d.chain.dtype), d.request.power
    cols = 1.0 + 2.0 * p * directions
    pairs = _pairs(d)
    gemm = sum(2.0 * B * i * o * cols for i, o in pairs)
    elem = sum(B * o * cols * 12.0 for _, o in pairs)
    traffic = sum(2.0 * B * o * cols * bts for _, o in pairs)
    peak = 3.0 * B * max(o for _, o in pairs) * cols * bts
    fwd, bwd = _launches(template, d)
    return Work(
        gemm,
        elem,
        traffic,
        peak,
        launches=fwd,
        compile_units=2.0 + 0.2 * len(pairs),
        launches_backward=bwd,
    )


def _work_jet(d: Descriptor) -> Work:
    """One order-``2p`` jet per input direction: ``(1 + 2p m)`` columns through every layer."""
    return _jet_work(d, d.selected, "taylor_jet")


def _work_nested(d: Descriptor) -> Work:
    """Nested ``hessian`` traces: every Laplace power multiplies the work by ``m`` HVPs of about four forward passes."""
    B, m, bts, p = d.batch, d.selected, bytes_of(d.chain.dtype), d.request.power
    F = _flops_forward(d)
    gemm = B * F * (4.0 * m) ** p
    peak = 4.0 * B * m * max(o for _, o in _pairs(d)) * bts * (m ** max(p - 1, 0))
    fwd, bwd = _launches("nested_autodiff", d)
    return Work(
        gemm,
        0.1 * gemm,
        0.5 * gemm / 8 * bts,
        peak,
        launches=fwd * p,
        compile_units=2.0 * p,
        launches_backward=bwd * p,
    )


# --------------------------------------------------------------------------------------- closed-form Taylor streams


def _req_taylor_closed(d: Descriptor) -> str | None:
    r = d.request
    if r.power > 2:
        return (
            "the directional-jet streams carry derivative order four at most (Delta^2); "
            "Delta^3 needs order six: use `projected` or `forward_polylaplacian`"
        )
    if r.metric == "point_diagonal" or r.directions:
        return "a per-point metric or drift is not served by the directional-jet streams"
    if r.power == 2 and (r.metric != "euclidean"):
        return "the bi-Laplacian by jets takes the Euclidean metric on the selected inputs only"
    reason = _chain_ok(d) or _smooth_ok(d, r.power)
    if reason is not None:
        return reason
    if r.power == 2:
        channels = 4 * d.selected * d.selected + 1
        if channels > TAYLOR_CLOSED_MAX_CHANNELS:
            return (
                f"the bi-Laplacian streams stack {channels} channels for {d.selected} inputs, more than the "
                f"{TAYLOR_CLOSED_MAX_CHANNELS} (64 inputs) a compile has been measured to survive; use `projected`"
            )
    return None


def _work_taylor_closed(d: Descriptor) -> Work:
    """One closed-form Taylor stream per direction and order.

    ``Delta`` takes ``m`` directions of order two.  ``Delta^2`` stacks ``m^2`` directions of order four (``4 m^2 + 1``
    channels, as :func:`_req_taylor_closed` counts them, not the ``m(m+1)/2`` an unsymmetrized polarization would need)
    and contracts them to the bi-Laplacian: one gather and multiply-add per direction and output.
    """
    m, p = d.selected, d.request.power
    if p == 1:
        return _jet_work(d, m, "taylor_closed")
    directions = float(m * m)
    w = _jet_work(d, directions, "taylor_closed")
    contraction = d.batch * directions * d.chain.layers[-1].out_features * 8.0
    return replace(w, elem_flops=w.elem_flops + contraction)


# ----------------------------------------------------------------------------------------------- estimators


def _req_estimator(name: str) -> Callable[[Descriptor], str | None]:
    def requires(d: Descriptor) -> str | None:
        if d.request.power != 1:
            return "the estimators serve the Laplacian (power 1)"
        if d.request.directions:
            return "drift directions are not estimated"
        if name == "sdgd" and d.request.metric not in ("euclidean", "diagonal"):
            return "coordinate sampling needs a diagonal metric on the selected inputs"
        acc = d.request.accuracy
        if acc is not None and name != "sdgd" and 2.0 / float(acc) ** 2 > 4096:
            return f"accuracy {acc} needs more than 4096 probes"
        if name == "sdgd":
            m = d.selected
            s = estimator_probes("sdgd", acc, m)
            if acc is not None and s >= m:
                return f"accuracy {acc} needs every one of the {m} coordinates: the exact program is cheaper"
        return None

    return requires


def _estimator_work(name: str) -> Callable[[Descriptor], Work]:
    def work(d: Descriptor) -> Work:
        n = estimator_probes(name, d.request.accuracy, d.selected)
        return _jet_work(d, n, "estimator")

    return work


# ------------------------------------------------------------------------------ vector-field quantities


def _req_vector_field(d: Descriptor) -> str | None:
    ok = _chain_ok(d)
    if ok:
        return ok
    if d.chain.output_dim != d.chain.input_dim and d.request.selected is None:
        return "a vector field maps D inputs to D outputs (pass `selected` to pair outputs with inputs)"
    return None


def _hidden_widths(d: Descriptor) -> list[int]:
    return [layer.out_features for layer in d.chain.layers[:-1]]


def _first_order_work(
    d: Descriptor, gemm: float, elem: float, peak: float, *, solver: float = 0.0, moved: float = 0.0
) -> Work:
    """A first-order program: the forward pass plus ``gemm`` extra GEMM flops, with launch counts from the template.

    ``moved`` is the bytes the extra matrix products read and write beyond the forward pass: the carried columns of every
    layer they cross, counted from their shapes (a matrix product of width ``w`` reuses each byte ``w / bytes`` times, so its
    traffic does not grow with its flops).
    """
    B, bts = d.batch, bytes_of(d.chain.dtype)
    pairs = _pairs(d)
    base = B * _flops_forward(d)
    traffic = sum(2.0 * B * o * bts for _, o in pairs) + moved
    fwd, bwd = _launches("first_order", d)
    return Work(
        base + gemm,
        elem + sum(B * o * 4.0 for _, o in pairs),
        traffic,
        peak + 3.0 * B * max(o for _, o in pairs) * bts + d.D * pairs[0][1] * bts,
        launches=fwd + 2.0 * solver,
        compile_units=1.0 + 0.1 * len(pairs),
        launches_backward=bwd + 4.0 * solver,
        solver_calls=solver,
    )


def _carried_columns(d: Descriptor) -> int:
    """Columns the cyclic-trace programs carry: the pairing size ``m`` or the last hidden width, whichever is smaller."""
    return max(1, min(d.selected, _hidden_widths(d)[-1]))


def _work_div_projected(d: Descriptor) -> Work:
    """``tr(S_n W_n ... S_1 W_1 W_L)`` (cyclic): one hidden layer is a dot product, two a single matrix-vector product per
    point, more carry ``c`` columns through the layers between the first and the last; the Jacobian is never formed."""
    B, bts = d.batch, bytes_of(d.chain.dtype)
    widths = _hidden_widths(d)
    n = len(widths)
    c = _carried_columns(d)
    once = (
        2.0 * d.selected * widths[0] * widths[-1]
    )  # the parameter product C = W_1 W_L, once per call
    if n <= 2:
        extra = once + (2.0 * B * widths[0] * widths[1] if n == 2 else 0.0)
        carried = 0.0
    else:
        carried = sum(2.0 * B * widths[i - 1] * widths[i] * c for i in range(1, n))
        extra = once + carried
    moved = 3.0 * B * c * sum(widths[1:]) * bts + (widths[0] * (d.selected + widths[-1]) + d.selected * widths[-1]) * bts
    elem = 4.0 * B * c * sum(widths[1:]) if n > 2 else 0.0  # the activation derivative times each carried column
    return _first_order_work(
        d, extra, elem, 3.0 * B * c * max(widths) * bts if n > 2 else 0.0, moved=moved
    )


def _work_logdet_projected(d: Descriptor) -> Work:
    """Sylvester: ``det(I + J) = det(I_c + A C)``; ``c`` columns through the hidden layers, one ``c x c`` LU per point."""
    B, bts = d.batch, bytes_of(d.chain.dtype)
    widths = _hidden_widths(d)
    c = _carried_columns(d)
    once = 2.0 * d.selected * widths[0] * widths[-1]
    carried = sum(2.0 * B * widths[i - 1] * widths[i] * c for i in range(1, len(widths)))
    gemm = once + carried + 2.0 * B * d.selected * widths[-1] * c + B * (2.0 / 3.0) * c**3
    moved = (
        3.0 * B * c * sum(widths[1:]) * bts  # the carried columns through every layer after the first, read, written, re-read
        + B * c * (widths[-1] + d.selected) * bts  # the product with the last layer and the c x c block
        + B * c * c * bts * 3.0  # the c x c LU, read, written, re-read
        + (widths[0] * (d.selected + widths[-1]) + d.selected * widths[-1]) * bts  # the parameter product, once
    )
    return _first_order_work(
        d,
        gemm,
        4.0 * B * c * sum(widths[1:]) + B * c * c * 4.0,  # the activation derivative times each carried column, the LU
        B * c * c * bts * 3.0 + 3.0 * B * c * max(widths) * bts,
        solver=1.0,
        moved=moved,
    )


def _work_solve_woodbury(d: Descriptor) -> Work:
    """The Woodbury solve shares the logdet's ``c``-column carry and replaces the determinant by a ``c x c`` solve."""
    return _work_logdet_projected(d)


def _work_first_order(d: Descriptor, k: int, det_power: float = 0.0) -> Work:
    """The dense Jacobian programs: ``k`` forward-mode columns through every layer, then (``det_power`` > 0) a ``k^3`` LU."""
    B, bts = d.batch, bytes_of(d.chain.dtype)
    pairs = _pairs(d)
    S = k + 1
    gemm = (
        sum(2.0 * B * i * o * (S - 1) for i, o in pairs[1:]) + 2.0 * B * pairs[0][0] * pairs[0][1]
    )
    gemm += B * (2.0 / 3.0) * (k**det_power) if det_power else 0.0
    elem = sum(B * o * S * 4.0 for _, o in pairs)
    traffic = sum(2.0 * B * o * S * bts for _, o in pairs)
    peak = (
        3.0 * B * max(o for _, o in pairs) * S * bts
        + d.D * pairs[0][1] * bts
        + B * k * d.chain.output_dim * bts
    )
    fwd, bwd = _launches("first_order", d)
    solver = 1.0 if det_power else 0.0
    return Work(
        gemm,
        elem,
        traffic,
        peak,
        launches=fwd + 2.0 * solver,
        compile_units=1.0 + 0.1 * len(pairs),
        launches_backward=bwd + 4.0 * solver,
        solver_calls=solver,
    )


# ----------------------------------------------------------------------------------- residual operators


_PROJECTED_DEMAND = frozenset({"value", "grad_sq", "lap", "bilap", "directional"})
_FULL_DEMAND = frozenset({"value", "grad", "grad_sq", "lap"})


def _residual_streams(d: Descriptor, k: int) -> int:
    q = d.request.directions
    order = d.demand.order
    if order >= 4:
        return _streams4(k, q)
    if order >= 2:
        return _streams2(k, q)
    return 1 + k + q


def _req_projected_residual(d: Descriptor) -> str | None:
    bad = d.demand.quantities - _PROJECTED_DEMAND
    if bad:
        return (
            f"the projected engine cannot provide {sorted(bad)} (the full gradient is not formed)"
        )
    if d.demand.order not in (0, 1, 2, 4):
        return "the projected engine serves blocks of order 0, 1, 2 or 4"
    return _chain_ok(d) or _smooth_ok(d, 2 if d.demand.order < 4 else 4)


def _req_full_residual(d: Descriptor) -> str | None:
    bad = d.demand.quantities - _FULL_DEMAND
    if bad:
        return f"the full-gradient engine cannot provide {sorted(bad)}"
    if d.demand.order > 2:
        return "the full-gradient engine serves blocks of order 0, 1 or 2"
    return _chain_ok(d) or _smooth_ok(d, 2)


def _gram_work(d: Descriptor, k: int, *, projected: bool = False) -> Work:
    """Work of a residual-Gram program with ``k`` first-layer tangent columns.

    ``projected`` marks the first-layer projection program: when it reduces the columns (``m > H_1``) it also pays the QR of the
    first-layer metric matrix, which the full-gradient program does not have.
    """
    n, B, bts = d.n_rows, d.batch, bytes_of(d.chain.dtype)
    pairs = _pairs(d)
    S = _residual_streams(d, k)
    rows = max(n, B)
    kind = d.request.kind
    c = 1.2 if d.demand.order >= 4 else 2.2
    gemm = 2.0 * rows * pairs[0][0] * pairs[0][1]
    traffic = elem = 0.0
    for idx, (i, o) in enumerate(pairs):
        if idx > 0:
            gemm += 2.0 * rows * i * o * (S - 1)
        stream_gram = c * rows * rows * (i + o) * 0.5 * S
        explicit = 2.0 * rows * rows * i * o + 2.0 * rows * i * o * S
        if kind == "ntk_diagonal":
            gemm += 1.0 * rows * (i + o) * S
        else:
            gemm += min(stream_gram, explicit)
        traffic += 2.0 * rows * o * S * bts
        elem += rows * o * (S + 4) * 6.0
    if kind == "gauss_newton":
        gemm += rows**3 / 3.0 + 2.0 * rows * sum(i * o for i, o in pairs)
    solver = 0.0
    if projected and d.selected > _h1(d):
        # ``_First`` of the projected engine: a QR of the ``H_1 x m`` first-layer metric matrix (one ``geqrf`` and one ``orgqr``
        # library call; ``4 m H_1^2`` flops, the convention of ``_work_projected``) and the products ``x Q``, ``M Q`` and
        # ``Q^T (Q s^2)``. Both library calls were counted in the compiled program of a CPU lowering.
        m, H1 = d.selected, _h1(d)
        gemm += 4.0 * m * H1 * H1 + 2.0 * rows * m * k + 2.0 * H1 * m * k + 2.0 * k * m * k
        solver = 2.0
    peak = 3.0 * rows * max(o for _, o in pairs) * S * bts + rows * rows * bts
    return Work(
        gemm,
        elem,
        traffic,
        peak,
        launches=14.0 * len(pairs),
        compile_units=1.5 + 0.2 * len(pairs),
        solver_calls=solver,
    )


def _work_projected_residual(d: Descriptor) -> Work:
    return _gram_work(d, _seed(d), projected=True)


def _work_full_residual(d: Descriptor) -> Work:
    return _gram_work(d, d.selected)


def _work_autodiff_rows(d: Descriptor) -> Work:
    """Explicit Jacobian rows by reverse mode through a nested derivative program, then ``K = J J^T``."""
    n, bts = max(d.n_rows, d.batch), bytes_of(d.chain.dtype)
    P = float(sum(i * o for i, o in _pairs(d)))
    F = _flops_forward(d)
    order = max(d.demand.order, 1)
    per_point = F * (4.0 * d.selected) ** max(order // 2, 1) if order >= 2 else 3.0 * F
    gemm = n * 3.0 * per_point + 2.0 * n * n * P
    if d.request.kind == "ntk_diagonal":
        gemm = n * 3.0 * per_point + n * P
    if d.request.kind == "gauss_newton":
        gemm += n**3 / 3.0 + 2.0 * n * P
    peak = (n * P + n * n + 4.0 * n * max(o for _, o in _pairs(d)) * d.selected) * bts
    return Work(gemm, 0.1 * gemm, 0.25 * gemm / 8.0 * bts, peak, launches=60.0, compile_units=3.0)


# --------------------------------------------------------------------------------------------- register


#: Backward-pass work of the fused kernels: the backward kernel recomputes the forward (it saves no streams) and applies the
#: transposed dots, so its flops are about three forward passes' and its traffic is only the cotangents and weight gradients.
_FUSED_BACKWARD = {
    "gemm": 3.0,
    "elem": 3.0,
    "traffic": 0.5,
    "serial": 3.0,
    "solver": 0.0,
    "peak": 1.0,
}


def _register() -> None:
    reg = register_algorithm
    lap = frozenset({"laplacian"})
    reg(
        Algorithm(
            "projected",
            lap,
            _req_projected,
            _work_projected,
            backend="jax|torch",
        )
    )
    reg(
        Algorithm(
            "fused_streams",
            lap,
            _req_fused,
            _work_fused,
            backend="jax|torch",
            backward=_FUSED_BACKWARD,
            accelerator_only=True,
            forced_only=_fused_forced_only,
            hardware_requires=_fused_hardware,
        )
    )
    reg(Algorithm("projected_cholesky", lap, _req_cholesky, _work_cholesky, backend="jax"))
    reg(Algorithm("forward_laplacian", lap, _req_forward, _work_forward, backend="any"))
    reg(
        Algorithm(
            "forward_polylaplacian",
            lap,
            _req_polylaplacian,
            _work_polylaplacian,
            backend="jax|torch",
        )
    )
    reg(
        Algorithm(
            "taylor_closed", lap, _req_taylor_closed, _work_taylor_closed, backend="jax|torch"
        )
    )
    reg(Algorithm("taylor_jet", lap, _req_jet, _work_jet, backend="any", generic=True))
    reg(Algorithm("nested_autodiff", lap, _req_generic, _work_nested, backend="any", generic=True))
    for estimator in ("hutchinson", "stde", "sdgd"):
        reg(
            Algorithm(
                estimator,
                lap,
                _req_estimator(estimator),
                _estimator_work(estimator),
                backend="any",
                exact=False,
            )
        )

    div = frozenset({"divergence"})
    reg(
        Algorithm(
            "projected",
            div,
            _req_vector_field,
            _work_div_projected,
            backend="jax|torch",
        )
    )
    reg(
        Algorithm(
            "jacobian_trace",
            div,
            _req_generic,
            lambda d: _work_first_order(d, d.selected),
            backend="any",
            generic=True,
        )
    )

    ld = frozenset({"logdet"})
    reg(
        Algorithm(
            "projected",
            ld,
            _req_vector_field,
            _work_logdet_projected,
            backend="jax|torch",
        )
    )
    reg(
        Algorithm(
            "dense_jacobian",
            ld,
            _req_generic,
            lambda d: _work_first_order(d, d.selected, 3.0),
            backend="any",
            generic=True,
        )
    )

    solve = frozenset({"solve"})
    reg(
        Algorithm(
            "woodbury",
            solve,
            _req_vector_field,
            _work_solve_woodbury,
            backend="jax|torch",
        )
    )
    reg(
        Algorithm(
            "dense_jacobian",
            solve,
            _req_generic,
            lambda d: _work_first_order(d, d.selected, 3.0),
            backend="any",
            generic=True,
        )
    )

    res = frozenset({"gram", "gauss_newton", "ntk_diagonal"})
    reg(
        Algorithm(
            "projected_streams",
            res,
            _req_projected_residual,
            _work_projected_residual,
            backend="jax|torch",
            backend_requires={"torch": _torch_no_projected_streams},
        )
    )
    reg(
        Algorithm("full_streams", res, _req_full_residual, _work_full_residual, backend="jax|torch")
    )
    reg(
        Algorithm(
            "autodiff_rows", res, _req_generic, _work_autodiff_rows, backend="any", generic=True
        )
    )


_register()

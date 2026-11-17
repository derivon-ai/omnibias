# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unexecuted private fixed-bit Adam trace candidate.

The supplied VG is an INTERNAL TRUSTED GENERATED objective/gradient boundary,
not a public arbitrary-gradient certificate API. This candidate changes the
binary64 power circuit; it preserves classical exact-real bias correction,
not bitwise equality to native pow. Full trace/error analysis may refuse.
"""

import math
from dataclasses import asdict, dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import bounds, device_training, planner


class TraceStep(NamedTuple):
    accepted: device_training.TrainingState
    proposed: device_training.TrainingState
    delta: object
    loss: object
    gradient: object
    status: object
    evaluated: object


class TraceBoxes(NamedTuple):
    parameter_lower: object
    parameter_upper: object
    input_lower: object
    input_upper: object
    projection_lower: object
    projection_upper: object

    def ordinary(self):
        return device_training.Boxes(*self[:4])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, maximum, name):
    _require(
        type(value) is int and 1 <= value <= maximum, name + " outside declared positive bound"
    )
    return value


def _power_nonnegative(beta, exponent, bits):
    """INTERNAL precondition: exact integer0<=exponent<2**bits, beta in[0,1).

    The public-to-this-component step checks counter/headroom before entering
    the VG/update branch. Never call this helper to truncate an unchecked count.
    """
    one = jnp.asarray(1, dtype=exponent.dtype)
    two = jnp.asarray(2, dtype=exponent.dtype)

    def body(carry, _):
        remaining, base, accumulator = carry
        odd = jax.lax.bitwise_and(remaining, one) != 0
        accumulator = jax.lax.select(odd, accumulator * base, accumulator)
        remaining = jax.lax.div(remaining, two)  # nonnegative integer floor
        return (remaining, base * base, accumulator), None

    initial = (exponent, jnp.asarray(beta, jnp.float64), jnp.asarray(1.0, jnp.float64))
    final, _ = jax.lax.scan(body, initial, None, length=bits)
    return final[2]


def _strict_slots(closed, source):
    result = bounds.ConstantBindings()

    def visit(value, path=()):
        graph = getattr(value, "jaxpr", value)
        for index, constant in enumerate(getattr(value, "consts", ())):
            bound = source.resolve(constant, path, index)
            if bound is None:
                raise NotImplementedError("unknown optimizer/VG constant origin")
            result.bind_slot(path, index, bound)
        for index, eq in enumerate(graph.eqns):
            name, params = eq.primitive.name, eq.params
            if name in (
                "jit",
                "pjit",
                "custom_jvp_call",
                "custom_vjp_call",
                "closed_call",
                "remat2",
            ):
                visit(params.get("jaxpr", params.get("call_jaxpr")), path + (f"{index}:{name}",))
            elif name == "scan":
                visit(params["jaxpr"], path + (f"{index}:scan",))
            elif name == "cond":
                for branch, sub in enumerate(params["branches"]):
                    visit(sub, path + (f"{index}:cond:{branch}",))

    visit(closed)
    return result


@dataclass(frozen=True)
class BoundedTraceAdam:
    value_gradient: object
    max_accepted_steps: int = (1 << 31) - 1
    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        _require(callable(self.value_gradient), "trusted generated VG required")
        _positive(self.max_accepted_steps, (1 << 31) - 1, "maximum accepted steps")
        for name in ("learning_rate", "beta1", "beta2", "epsilon"):
            value = getattr(self, name)
            _require(type(value) in (int, float), "fixed Python binary64 coefficient required")
            try:
                represented = float(value)
            except OverflowError as error:
                raise ValueError("optimizer coefficient conversion overflow") from error
            _require(math.isfinite(represented), "fixed finite optimizer coefficient required")
            _require(
                type(value) is not int or int(represented) == value,
                "optimizer integer coefficient not exactly binary64 representable",
            )
            object.__setattr__(self, name, represented)
        _require(
            self.learning_rate > 0
            and self.epsilon > 0
            and 0 <= self.beta1 < 1
            and 0 <= self.beta2 < 1,
            "ordinary positive Adam scales/betas required",
        )

    @property
    def bits(self):
        return self.max_accepted_steps.bit_length()

    @property
    def analysis_contract(self):
        """A conditional arithmetic report is not a runnable certificate."""
        return {
            "runtime_input_containment_checked": False,
            "certificate_issued": False,
            "analysis_input_scope": "conditional on supplied initial state/X/Z/threshold bounds; caller must bind and check actual parameters, BOTH moments, counter, X/Z and thresholds before using this report",
        }

    def counter_valid(self, counter, *, headroom=1):
        _positive(headroom, self.max_accepted_steps, "headroom")
        _require(
            hasattr(counter, "shape") and hasattr(counter, "dtype"),
            "array counter metadata required",
        )
        _require(
            counter.shape == ()
            and np.dtype(counter.dtype).kind in "iu"
            and np.dtype(counter.dtype).itemsize in (4, 8),
            "scalar32/64-bit integer counter required",
        )
        maximum = min(self.max_accepted_steps, int(np.iinfo(counter.dtype).max))
        return (counter >= 0) & (counter <= maximum - headroom)

    def _metadata(self, state, points, projection, boxes):
        _require(jax.config.x64_enabled, "binary64 configuration required")
        _require(
            isinstance(state, device_training.TrainingState) and isinstance(boxes, TraceBoxes),
            "TrainingState and explicit six-field TraceBoxes required",
        )
        shape_tree = jax.tree.structure(state.params)
        p = jax.tree.leaves(state.params)
        _require(
            p
            and hasattr(points, "dtype")
            and getattr(points, "ndim", None) == 2
            and points.shape[0] > 0
            and np.dtype(points.dtype) == np.dtype("float64"),
            "nonempty real binary64 parameter tree/points required",
        )
        _require(
            hasattr(projection, "dtype")
            and getattr(projection, "ndim", None) == 2
            and projection.shape[0] == points.shape[0]
            and projection.shape[1] > 0
            and np.dtype(projection.dtype) == np.dtype("float64"),
            "explicit binary64 runtime projection required",
        )
        _require(
            all(
                hasattr(v, "shape")
                and hasattr(v, "dtype")
                and v.shape == projection.shape
                and np.dtype(v.dtype) == np.dtype("float64")
                for v in (boxes.projection_lower, boxes.projection_upper)
            ),
            "exact projection threshold metadata required",
        )
        for tree in (
            state.params,
            state.first_moment,
            state.second_moment,
            boxes.parameter_lower,
            boxes.parameter_upper,
        ):
            leaves = jax.tree.leaves(tree)
            _require(
                jax.tree.structure(tree) == shape_tree and len(leaves) == len(p),
                "complete state/bounds pytrees required",
            )
            _require(
                all(
                    hasattr(v, "shape")
                    and hasattr(v, "dtype")
                    and hasattr(original, "shape")
                    and v.shape == original.shape
                    and np.dtype(v.dtype) == np.dtype("float64")
                    for v, original in zip(leaves, p, strict=True)
                ),
                "state/bound leaf shape/dtype mismatch",
            )
        _require(
            all(
                hasattr(v, "dtype")
                and hasattr(v, "shape")
                and np.dtype(v.dtype) == np.dtype("float64")
                and np.broadcast_shapes(v.shape, points.shape) == points.shape
                for v in (boxes.input_lower, boxes.input_upper)
            ),
            "broadcastable binary64 input boxes required",
        )

    def step(self, state, points, projection, boxes, *, headroom=1):
        """One actual VG evaluation, only on a valid guarded state/input/count."""
        self._metadata(state, points, projection, boxes)
        cvalid = self.counter_valid(state.accepted_steps, headroom=headroom)
        pvalid = device_training.inside(state.params, boxes.parameter_lower, boxes.parameter_upper)
        xvalid = jnp.all(
            jnp.isfinite(points) & (points >= boxes.input_lower) & (points <= boxes.input_upper)
        )
        mvalid = device_training.finite((state.first_moment, state.second_moment))
        zvalid = jnp.all(
            jnp.isfinite(projection)
            & (projection >= boxes.projection_lower)
            & (projection <= boxes.projection_upper)
        )
        valid = cvalid & pvalid & xvalid & zvalid & mvalid
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def evaluate(_):
            value, gradient = self.value_gradient(state.params, points, projection)
            _require(
                value.shape == ()
                and np.dtype(value.dtype) == np.dtype("float64")
                and jax.tree.structure(gradient) == jax.tree.structure(state.params),
                "generated scalar/complete gradient required",
            )
            _require(
                all(
                    g.shape == p.shape and np.dtype(g.dtype) == np.dtype("float64")
                    for g, p in zip(
                        jax.tree.leaves(gradient), jax.tree.leaves(state.params), strict=True
                    )
                ),
                "complete binary64 gradient leaves required",
            )
            count = state.accepted_steps + jnp.asarray(1, dtype=state.accepted_steps.dtype)
            beta1, beta2 = (
                jnp.asarray(self.beta1, jnp.float64),
                jnp.asarray(self.beta2, jnp.float64),
            )
            decay1, decay2 = (
                jnp.asarray(1.0, jnp.float64) - beta1,
                jnp.asarray(1.0, jnp.float64) - beta2,
            )
            first = jax.tree.map(lambda m, g: beta1 * m + decay1 * g, state.first_moment, gradient)
            second = jax.tree.map(
                lambda v, g: beta2 * v + decay2 * g * g, state.second_moment, gradient
            )
            b1 = _power_nonnegative(self.beta1, count, self.bits)
            b2 = _power_nonnegative(self.beta2, count, self.bits)
            delta = jax.tree.map(
                lambda m, v: (
                    -self.learning_rate * (m / (1 - b1)) / (jnp.sqrt(v / (1 - b2)) + self.epsilon)
                ),
                first,
                second,
            )
            proposed = device_training.TrainingState(
                jax.tree.map(jnp.add, state.params, delta), first, second, count
            )
            finite = device_training.finite((value, gradient, proposed, delta))
            in_domain = device_training.inside(
                proposed.params, boxes.parameter_lower, boxes.parameter_upper
            )
            status = jnp.where(
                finite,
                jnp.where(in_domain, device_training.OK, device_training.DOMAIN_EXIT),
                device_training.NONFINITE,
            ).astype(jnp.int32)
            accepted = jax.lax.cond(
                status == device_training.OK, lambda _: proposed, lambda _: state, None
            )
            return TraceStep(accepted, proposed, delta, value, gradient, status, jnp.asarray(True))

        def reject(_):
            status = jnp.where(
                cvalid & pvalid & xvalid & zvalid,
                device_training.NONFINITE,
                device_training.INVALID_INPUT,
            ).astype(jnp.int32)
            return TraceStep(
                state, state, zero, jnp.asarray(0.0, jnp.float64), zero, status, jnp.asarray(False)
            )

        return jax.lax.cond(valid, evaluate, reject, None)

    def analyze_trace(
        self,
        state,
        points,
        projection,
        boxes,
        state_bounds,
        point_bound,
        projection_bound,
        box_bounds,
        *,
        headroom=1,
        extra_bindings=None,
        limits=None,
        tolerances=None,
    ):
        """Analyze the actual emitted trace. Refusals/inconclusive are preserved.

        This conditional arithmetic check does NOT prove that an arbitrary
        supplied callback is the derivative of an objective. Known context
        builders must establish that relationship and originating constants.
        Initial bounds are explicit CPU data; no point-reset of carried errors.
        Actual runtime input containment (including both moments/counter and
        thresholds) is NOT checked here; no runnable certificate is issued.
        """
        limits = planner.Limits(cells=1) if limits is None else limits
        tolerances = planner.Tolerances() if tolerances is None else tolerances
        _require(isinstance(limits, planner.Limits), "typed planner Limits required")
        _require(isinstance(tolerances, planner.Tolerances), "typed planner Tolerances required")
        _require(
            type(limits.equations) is int and 1 <= limits.equations <= 100000,
            "analysis equation limit exceeds hard cap100000",
        )
        _require(
            type(limits.state_bytes) is int and 1 <= limits.state_bytes <= 3 * 1024**3,
            "analysis state limit exceeds hard cap3GiB",
        )
        _require(
            asdict(tolerances) == asdict(planner.Tolerances()), "unchanged tolerances required"
        )
        self._metadata(state, points, projection, boxes)
        _require(hasattr(state.accepted_steps, "shape"), "array optimizer counter required")
        args, declared = (
            (state, points, projection, boxes),
            (state_bounds, point_bound, projection_bound, box_bounds),
        )
        actual_leaves, input_tree = jax.tree.flatten(args)
        declared_leaves, declared_tree = jax.tree.flatten(declared)
        _require(
            input_tree == declared_tree and len(actual_leaves) == len(declared_leaves),
            "complete actual-step input bounds required",
        )
        _require(
            sum(24 * math.prod(value.shape) for value in actual_leaves) <= limits.state_bytes,
            "initial full trace-bound payload exceeds state budget",
        )
        owned = []
        for value, bound in zip(actual_leaves, declared_leaves, strict=True):
            _require(
                isinstance(bound, bounds.Bound) and bound.lo.shape == value.shape,
                "exact input-bound leaf shapes required",
            )
            owned.append(bounds.immutable_bound(bound))
        result = {
            "admissible": False,
            **self.analysis_contract,
            "nonvacuous_evaluated_case": False,
            "scope": "actual full same-call optimizer trace; trusted generated VG semantic boundary",
            "tolerances": asdict(tolerances),
            "limits": asdict(limits),
            "bits": self.bits,
            "max_accepted_steps": self.max_accepted_steps,
            "state_error_contract": "forward tolerance also required for loss/delta/float accepted/proposed state; gradient tolerance for every actual gradient leaf; discrete controls/counters exact",
            "native_pow": False,
            "projection_error_max": float(np.max(projection_bound.error)),
            "public_integration": False,
            "performance_evidence": False,
            "assumptions": "binary64 RN, gradual underflow, finite intermediates, compatible compiler transformations, existing VG native-transcendental contracts; exact integer div/bit operations within declared range",
        }
        try:
            function = partial(self.step, headroom=headroom)
            source = bounds.ConstantBindings().extend(
                getattr(self.value_gradient, "constant_bounds", None)
            )
            source.extend(extra_bindings)
            function.constant_bounds = source
            frozen = planner.freeze_callable(function, args)
            graph = frozen.closed_jaxpr
            # Keep actual emitted evidence even if later shape/bounds analysis refuses.
            result["traced_graph"] = str(graph)
            identity = planner._emitted_graph_identity(graph)
            result["emitted_graph_identity"] = identity
            result["emitted_graph_id"] = planner.digest_record(identity)
            shaped = jax.eval_shape(frozen, *args)
            values, meta = bounds.interpret(
                graph,
                owned,
                limits.equations,
                _strict_slots(graph, frozen.constant_bounds),
                max_state_bytes=limits.state_bytes,
            )
            _require(len(values) == len(jax.tree.leaves(shaped)), "incomplete interpreted trace")
            traced = jax.tree.unflatten(jax.tree.structure(shaped), values)
            summaries, discrete = [], []
            for name in TraceStep._fields:
                physical, expected = (
                    jax.tree.leaves(getattr(traced, name)),
                    jax.tree.leaves(getattr(shaped, name)),
                )
                for i, (value, target) in enumerate(zip(physical, expected, strict=True)):
                    label = f"{name}/{i}"
                    if np.dtype(target.dtype).kind in "biu":
                        exact = bool(
                            np.array_equal(value.lo, value.hi)
                            and not np.any(value.error)
                            and np.all(value.lo == np.floor(value.lo))
                        )
                        discrete.append({"leaf": label, "exact": exact})
                    else:
                        atol, rtol = (
                            (tolerances.gradient_atol, tolerances.gradient_rtol)
                            if name == "gradient"
                            else (tolerances.forward_atol, tolerances.forward_rtol)
                        )
                        summaries.append({"leaf": label, **value.summary(atol, rtol)})
            admissible = all(v["admissible"] for v in summaries) and all(
                v["exact"] for v in discrete
            )
            evaluated = bool(
                np.array_equal(traced.evaluated.lo, traced.evaluated.hi)
                and not np.any(traced.evaluated.error)
                and np.all(traced.evaluated.lo == 1)
            )
            result.update(
                admissible=admissible,
                nonvacuous_evaluated_case=admissible and evaluated,
                outputs=summaries,
                discrete=discrete,
                graph=meta,
            )
            if not admissible:
                result["reason"] = "full trace conditional error contract inconclusive"
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            result.update(reason=str(error), exception=type(error).__name__)
        return result

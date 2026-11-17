# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unvalidated private ridge-only diagonal geometry proposal.

Install only in a separately manifested R9 candidate. A three-argument raw
call has the precondition that X,Z0 come from this owner's producer. Only the
five supplied parameter leaves have a physical first-gradient contract.
Full LowRankAdapter.geometry is deliberately unrelated and unchanged.
"""

import json
import math
import time
import weakref
from dataclasses import asdict, dataclass
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from omnibias.core.operator_spec import OperatorSpec

from . import bounds, planner, prefetch, providers
from .certificates import matrix_product

_BASES = weakref.WeakValueDictionary()
_PROJECTIONS = weakref.WeakValueDictionary()


def _record(cls, registry, **fields):
    value = object.__new__(cls)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    registry[id(value)] = value
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _device_signature(value):
    require(isinstance(value, jax.Array) and not value.is_deleted(), "live owned device array required")
    require(value.is_fully_addressable and len(value.devices()) == 1, "one addressable device required")
    return (id(value), value.shape, value.dtype.str, tuple(map(str, value.devices())))


def _exact(value):
    try:
        result = float(value)
    except OverflowError as error:
        raise NotImplementedError("operator coefficient exceeds binary64 conversion range") from error
    if not math.isfinite(result) or Fraction(result) != value:
        raise NotImplementedError("operator coefficient is not exactly binary64 representable")
    return result


def _requests(operators, dimension, max_work):
    require(type(operators) is tuple and 1 <= len(operators) <= 256, "bounded nonempty OperatorSpec tuple required")
    rows, orders, axes, powers = [], set(), set(), set()
    work = 0
    for op in operators:
        require(isinstance(op, OperatorSpec), "explicit OperatorSpec required")
        op.check_dimension(dimension)
        if op.mode == "directional" or not 0 <= op.order <= 16:
            raise NotImplementedError("constant-coordinate spatial orders through16 required")
        orders.add(op.order)
        if op.mode == "laplacian":
            rows.append(("laplacian", op.power))
            if op.power:
                powers.add(op.power)
        else:
            terms = []
            for alpha, coefficient in op.resolve_terms(dimension):
                value = _exact(coefficient)
                active = tuple((i, n) for i, n in enumerate(alpha) if n)
                axes.update(active)
                terms.append((sum(alpha), active, value))
                orders.add(sum(alpha))
                work += 1 + len(active)
            rows.append(("polynomial", tuple(terms)))
        work += 1
        if work > max_work:
            raise MemoryError("operator symbol work budget exhausted before tracing")
    return tuple(rows), tuple(sorted(orders)), tuple(sorted(axes)), tuple(sorted(powers)), work


@dataclass(frozen=True, init=False, eq=False)
class RidgeBase:
    """Owned A0 plus optional n0 vector; never a general metric adapter."""
    frozen: object
    metadata: object
    base: object
    norms: object
    rows: tuple
    orders: tuple
    coordinate_powers: tuple
    norm_powers: tuple
    symbol_work: int
    max_symbol_work: int
    max_host_bytes: int
    setup_s: float
    _signatures: tuple

    @property
    def need_norms(self):
        return bool(self.norm_powers)

    def validate(self):
        require(_BASES.get(id(self)) is self, "base was not issued by the owned norm producer")
        require(isinstance(self.frozen, prefetch.FrozenDependencies)
                and isinstance(self.metadata, prefetch.FrozenDependencies), "owned frozen dependencies required")
        self.frozen.validate_ownership()
        self.metadata.validate_ownership()
        values = (self.base, self.norms) if self.need_norms else (self.base,)
        require(tuple(_device_signature(v) for v in values) == self._signatures, "owned base/norm device identity changed")
        require(tuple(self.frozen.values) == ("base",), "one fixed base dependency required")
        require(self.base.shape == self.frozen.values["base"].shape, "frozen base shape changed")
        expected = ("error", "hi", "lo", "norm") if self.need_norms else ()
        require(tuple(self.metadata.values) == expected and (self.norms is not None) == self.need_norms,
                "declared operator/norm storage contract changed")
        return self

    def norm_bound(self):
        self.validate()
        if not self.need_norms:
            raise ValueError("coordinate-only operator set has no norm quantity")
        m = self.metadata.values
        return bounds.immutable_bound(bounds.Bound(m["lo"], m["hi"], m["error"]))

    def constant_bounds(self):
        self.validate()
        result = bounds.ConstantBindings().bind_owned_transfer(
            self.base, self.frozen.values["base"], bounds.Bound.point(self.frozen.values["base"]))
        if self.need_norms:
            result.bind_owned_transfer(self.norms, self.metadata.values["norm"], self.norm_bound())
        return result

    def storage_report(self):
        self.validate()
        return {"base_shape": list(self.base.shape), "need_norms": self.need_norms,
                "norm_shape": list(self.norms.shape) if self.need_norms else None,
                "owned_host_payload_bytes": self.frozen.nbytes + self.metadata.nbytes,
                "owned_device_payload_bytes": self.base.nbytes + (self.norms.nbytes if self.need_norms else 0),
                "host_limit_bytes": self.max_host_bytes, "setup_s": self.setup_s,
                "symbol_work": self.symbol_work, "symbol_work_limit": self.max_symbol_work,
                "setup_scratch_payload_allowance_bytes": 512 * self.base.size + 128 * self.base.shape[0] if self.need_norms else 0,
                "setup_scope": "factory checks/enclosure/device transfer/synchronization; supplied FrozenDependencies construction, compilation and per-batch producer separate",
                "norm_precompute_work": "O(H*D) products/reduction" if self.need_norms else "omitted",
                "metric_storage": "none; no H-by-H Gram or diagonal embedding",
                "memory_scope": "numeric retained payload only; setup scratch estimates, Python ownership metadata, interpreter/compiler/allocator/process peaks separate",
                "cache_scope": "no new plan/certificate/executable cache; these owners and caller-retained reports are outside existing LRU accounting",
                "performance_scope": "setup/producer/reuse costs separate; no measured speed or overlap claim"}


def prepare_ridge_base(frozen, operators, *, device=None, max_host_bytes=3 * 1024**3, max_symbol_work=100000):
    """Prepare only requested ridge invariants from an owned CPU A0 snapshot."""
    started = time.perf_counter()
    require(jax.config.x64_enabled, "binary64 configuration required")
    require(isinstance(frozen, prefetch.FrozenDependencies) and tuple(frozen.values) == ("base",),
            "explicit FrozenDependencies({'base': A0}) required")
    a = frozen.values["base"]
    require(a.ndim == 2 and min(a.shape) > 0 and a.dtype == np.float64 and np.isfinite(a).all(),
            "finite positive-shape binary64 base(H,D) required")
    require(type(max_host_bytes) is int and max_host_bytes > 0 and type(max_symbol_work) is int
            and 1 <= max_symbol_work <= 100000, "positive host/bounded symbol budgets required")
    rows, orders, coordinate, powers, symbol_work = _requests(operators, a.shape[1], max_symbol_work)
    # Conservative explicit payload allowance; not a proof of overall NumPy RSS.
    estimate = frozen.nbytes + (512 * a.size + 128 * a.shape[0] if powers else 0)
    if estimate > max_host_bytes:
        raise MemoryError("base/norm preparation scratch payload allowance exceeds host budget")
    if powers:
        with np.errstate(over="raise", invalid="raise"):
            squared = a * a
            norm = np.sum(squared, axis=1, dtype=np.float64)
        require(np.isfinite(norm).all(), "saved row norm overflow")
        # SAME NumPy product then row-reduction arithmetic; never an exact cache.
        enclosure = bounds.reduction(bounds.binary(bounds.Bound.point(a), bounds.Bound.point(a), "mul"), (1,))
        # Ideal n0 is sum_i A0_hi**2 >=0. No runtime clipping is performed.
        enclosure = bounds.Bound(np.maximum(enclosure.lo, 0.), enclosure.hi, enclosure.error)
        low, high = enclosure.enlarged()
        require(np.all((norm >= low) & (norm <= high)), "saved norm outside its construction enclosure")
        metadata = prefetch.FrozenDependencies({"norm": norm, "lo": enclosure.lo,
                                                 "hi": enclosure.hi, "error": enclosure.error})
        placed_base, placed_norm = jax.device_put((a, metadata.values["norm"]), device)
        values = placed_base, placed_norm
    else:
        metadata = prefetch.FrozenDependencies({})
        placed_base, placed_norm = jax.device_put(a, device), None
        values = (placed_base,)
    jax.block_until_ready(values)
    owner = _record(RidgeBase, _BASES, frozen=frozen, metadata=metadata, base=placed_base,
                    norms=placed_norm, rows=rows, orders=orders, coordinate_powers=coordinate,
                    norm_powers=powers, symbol_work=symbol_work, max_symbol_work=max_symbol_work,
                    max_host_bytes=max_host_bytes, setup_s=0.,
                    _signatures=tuple(_device_signature(v) for v in values))
    owner.validate()
    object.__setattr__(owner, "setup_s", time.perf_counter() - started)
    return owner


@dataclass(frozen=True, init=False, eq=False)
class RidgeProjection:
    owner: RidgeBase
    metadata: object
    points: object
    projection: object
    preparation_s: float
    _signatures: tuple

    def validate(self):
        require(_PROJECTIONS.get(id(self)) is self, "projection was not issued by the owned dot producer")
        self.owner.validate()
        self.metadata.validate_ownership()
        require(tuple(_device_signature(v) for v in (self.points, self.projection)) == self._signatures,
                "owned input/projection device identity changed")
        m = self.metadata.values
        require(m["points"].shape == self.points.shape and m["projection"].shape == self.projection.shape
                and self.projection.shape == (self.points.shape[0], self.owner.base.shape[0]),
                "owned producer shape mismatch")
        return self

    def construction_bound(self):
        self.validate()
        m = self.metadata.values
        return bounds.immutable_bound(bounds.Bound(m["lo"], m["hi"], m["error"]))

    def storage_report(self):
        self.validate()
        return {"owned_host_payload_bytes": self.metadata.nbytes,
                "device_payload_bytes": self.points.nbytes + self.projection.nbytes,
                "producer_scratch_payload_allowance_bytes": 512 * (self.metadata.values["points"].size + self.projection.size),
                "producer_s": self.preparation_s, "base_shared": True,
                "scope": "point-X/saved-Z0 producer only; no trainable projection preparation or overlap claim"}


def prepare_projection(owner, points):
    """Owned point-X producer. Z0's dot construction error is never reset."""
    started = time.perf_counter()
    require(isinstance(owner, RidgeBase), "RidgeBase required")
    owner.validate()
    require(isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[0] > 0
            and points.shape[1] == owner.base.shape[1] and points.dtype == np.float64
            and np.isfinite(points).all(), "finite binary64 CPU points(B,D) required")
    h = owner.base.shape[0]
    estimate = owner.frozen.nbytes + owner.metadata.nbytes + 512 * (points.size + points.shape[0] * h)
    if estimate > owner.max_host_bytes:
        raise MemoryError("producer scratch payload allowance exceeds host budget")
    incoming = prefetch.FrozenDependencies({"points": points})
    x, a = incoming.values["points"], owner.frozen.values["base"]
    with np.errstate(over="raise", invalid="raise"):
        projection = x @ a.T
    require(np.isfinite(projection).all(), "saved projection overflow")
    enclosure = matrix_product(x, a.T)
    lower, upper = enclosure.enlarged()
    require(np.all((projection >= lower) & (projection <= upper)), "saved dot outside its construction enclosure")
    metadata = prefetch.FrozenDependencies({"points": x, "projection": projection,
                                          "lo": enclosure.lo, "hi": enclosure.hi, "error": enclosure.error})
    payload = jax.device_put((metadata.values["points"], metadata.values["projection"]), owner.base.device)
    jax.block_until_ready(payload)
    result = _record(RidgeProjection, _PROJECTIONS, owner=owner, metadata=metadata,
                     points=payload[0], projection=payload[1], preparation_s=0.,
                     _signatures=tuple(_device_signature(v) for v in payload))
    result.validate()
    object.__setattr__(result, "preparation_s", time.perf_counter() - started)
    return result


def _parameter_contract(params, points, projection, h, d):
    require(type(params) is tuple and len(params) == 5, "live (U,V,bias,readout,offset) tuple required")
    shapes = tuple(p.shape for p in params)
    rank = shapes[0][1] if len(shapes[0]) == 2 else 0
    require(rank > 0 and shapes == ((h, rank), (d, rank), (h,), (h,), ()), "complete low-rank parameter shapes required")
    require(points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == d
            and projection.shape == (points.shape[0], h), "runtime X,Z0 shapes differ")
    require(all(np.dtype(v.dtype) == np.dtype("float64") for v in (*params, points, projection)),
            "all runtime arguments must be binary64")


def _slot_bindings(closed, source):
    """Bind captured exact transfers; never read unknown GPU constants as exact."""
    result = bounds.ConstantBindings()
    def visit(value, path=()):
        graph = getattr(value, "jaxpr", value)
        if not hasattr(graph, "eqns"):
            raise NotImplementedError("nested graph has no supported constant provenance path")
        for i, constant in enumerate(getattr(value, "consts", ())):
            item = source.objects.get(id(constant))
            if item is not None and item[0] is constant:
                source.owned_host_snapshot(constant)
                bound = item[1]
            elif isinstance(constant, jax.Array) or not np.isscalar(constant):
                raise NotImplementedError("unowned graph constant: explicit conversion/provenance obligation")
            else:
                bound = bounds.Bound.point(constant)
            result.bind_slot(path, i, bound)
        for i, equation in enumerate(graph.eqns):
            name, params = equation.primitive.name, equation.params
            if name in ("jit", "pjit", "custom_jvp_call", "custom_vjp_call", "closed_call", "remat2"):
                visit(params.get("jaxpr", params.get("call_jaxpr")), path + (f"{i}:{name}",))
            elif name == "scan":
                visit(params["jaxpr"], path + (f"{i}:scan",))
            elif name == "cond":
                for branch, sub in enumerate(params["branches"]):
                    visit(sub, path + (f"{i}:cond:{branch}",))
    visit(closed)
    return result


@dataclass(frozen=True, init=False, eq=False)
class RidgeProgram:
    """Generated circuits only; no arbitrary supplied gradient callback API."""
    def __init__(self, owner, *, provider="parity", targets=None, weights=None, gradient_scope="parameters"):
        require(isinstance(owner, RidgeBase), "typed ridge owner required")
        owner.validate()
        if gradient_scope != "parameters":
            raise NotImplementedError("fixed base/input physical gradients are unsupported")
        require(provider in {"parity", "polynomial", "sech", "recurrence", "jax_native", "jax_analytic"}, "selected provider required")
        count = len(owner.rows)
        targets = (0.,) * count if targets is None else targets
        weights = (1.,) * count if weights is None else weights
        require(type(targets) is tuple and type(weights) is tuple and len(targets) == len(weights) == count,
                "one scalar target and weight per requested output required")
        require(all(type(v) is float and math.isfinite(v) for v in (*targets, *weights))
                and all(w >= 0 for w in weights), "finite declared binary64 scalar targets/nonnegative weights required")
        for name, value in (("owner", owner), ("provider", provider), ("targets", targets), ("weights", weights)):
            object.__setattr__(self, name, value)
        a0, n0 = owner.base, owner.norms
        h, d = a0.shape
        tower = providers.tower(owner.orders, provider)

        def forward(params, points, projection):
            _parameter_contract(params, points, projection, h, d)
            u, v, bias, readout, offset = params
            raw = projection + points @ v @ u.T
            z = jax.lax.optimization_barrier(raw) + bias
            values = dict(zip(owner.orders, tower(z), strict=True))
            columns = {axis: a0[:, axis] + u @ v[axis] for axis in sorted({i for i, _ in owner.coordinate_powers})}
            coordinate = {(i, n): columns[i] ** n for i, n in owner.coordinate_powers}
            if owner.need_norms:
                av = a0 @ v
                norm = n0 + 2 * jnp.sum(av * u, axis=1) + jnp.sum((u @ (v.T @ v)) * u, axis=1)
                norms = {n: norm ** n for n in owner.norm_powers}
            answers = []
            for kind, specification in owner.rows:
                if kind == "laplacian":
                    answer = values[2 * specification] @ (readout * norms[specification]) if specification else values[0] @ readout + offset
                else:
                    grouped, zero = {}, 0.
                    for order, active, coefficient in specification:
                        factor = jnp.ones_like(readout) * coefficient
                        for key in active:
                            factor = factor * coordinate[key]
                        grouped[order] = grouped.get(order, 0) + factor
                        if order == 0:
                            zero = coefficient  # Unique canonical zero multi-index.
                    answer = sum(values[n] @ (readout * factor) for n, factor in grouped.items()) + zero * offset
                answers.append(answer)
            return tuple(answers)

        def objective(params, points, projection):
            outputs = forward(params, points, projection)
            return sum(.5 * weight * jnp.mean((value - target) ** 2)
                       for value, target, weight in zip(outputs, targets, weights, strict=True))

        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "value_gradient", jax.value_and_grad(objective, argnums=0))
        demands = {"activation_orders": list(owner.orders),
                        "first_gradient_orders": sorted(set(owner.orders) | {n + 1 for n in owner.orders}),
                        "coordinate_powers": [list(v) for v in owner.coordinate_powers],
                        "norm_powers": list(owner.norm_powers), "n0_cached_directly": owner.need_norms,
                        "coefficient_conversion": "exact binary64 only; nonexact operator/provider coefficients refuse",
                        "provider_dependencies": "demand list does not remove the selected provider's genuine recurrence/Taylor intermediate orders",
                        "gradients": "complete live U,V,bias,readout,offset only; no base/input physical gradients"}
        object.__setattr__(self, "_demands_json", json.dumps(demands, sort_keys=True))

    @property
    def demands(self):
        return json.loads(self._demands_json)

    def qualify(self, params, packet, domain, *, limits=None, tolerances=None):
        """One bounded actual-graph analysis; inconclusive is a valid outcome."""
        started = time.perf_counter()
        limits = planner.Limits(cells=1) if limits is None else limits
        tolerances = planner.Tolerances() if tolerances is None else tolerances
        require(isinstance(limits, planner.Limits) and isinstance(tolerances, planner.Tolerances),
                "typed bounded limits/tolerances required")
        self.owner.validate()
        require(isinstance(packet, RidgeProjection) and packet.owner is self.owner, "owned producer packet from this base required")
        packet.validate()
        require(asdict(tolerances) == asdict(planner.Tolerances()), "unchanged tolerances required")
        _parameter_contract(params, packet.points, packet.projection, *self.owner.base.shape)
        require(isinstance(domain, planner.Domain) and type(domain.parameters) is tuple
                and len(domain.parameters) == 5, "complete explicit parameter/point-X Domain required")
        initial_payload = 3 * sum(v.nbytes for v in (*params, packet.points, packet.projection))
        if initial_payload > limits.state_bytes:
            raise MemoryError("initial three-array bound payload exceeds interpretation budget")
        owned = planner.Domain(tuple(bounds.immutable_bound(v) for v in domain.parameters), bounds.immutable_bound(domain.inputs))
        require(all(v.lo.shape == p.shape for v, p in zip(owned.parameters, params, strict=True))
                and all(not np.any(v.error) for v in (*owned.parameters, owned.inputs)), "real zero-error exact-shape parameter/input domains required")
        x = packet.metadata.values["points"]
        require(owned.inputs.lo.shape == x.shape and np.array_equal(owned.inputs.lo, x)
                and np.array_equal(owned.inputs.hi, x), "initial qualifier requires the exact owned real-X point domain")
        require(bool(planner.runtime_guard(params, packet.points, tuple(v.lo for v in owned.parameters),
                                           tuple(v.hi for v in owned.parameters), owned.inputs.lo, owned.inputs.hi)),
                "actual parameters lie outside the declared domain")
        args = params, packet.points, packet.projection
        construction = packet.construction_bound()
        inputs = [*owned.parameters, owned.inputs, construction]  # Z0.error deliberately retained.
        result = {"admissible": False, "scope": "generated actual3arg outputs/scalar/complete first parameter gradient; fixed point X and parameter box",
                  "construction_max_error": float(np.max(construction.error)), "construction_error_preserved": True,
                  "norm_construction_max_error": float(np.max(self.owner.norm_bound().error)) if self.owner.need_norms else None,
                  "tolerances": asdict(tolerances), "limits": asdict(limits), "analysis_cells": 1,
                  "budget_scope": "initial preflight covers parameter/X/Z0 bounds only; captured base/n0 binding storage is checked by the interpreter; neither is a whole-process cap",
                  "demands": self.demands, "base_storage": self.owner.storage_report(), "producer_storage": packet.storage_report(),
                  "assumptions": "binary64 RN, gradual underflow, finite intermediates, compatible NumPy reduction/dot and compiler transformations, declared native transcendental bounds, exact owned transfers",
                  "public_promotion": False, "new_final_source_confirmation_required": True}
        try:
            shaped = jax.eval_shape(self.forward, *args)
            vg = jax.eval_shape(self.value_gradient, *args)
            require(type(shaped) is tuple and len(shaped) == len(self.owner.rows)
                    and all(v.shape == (x.shape[0],) and v.dtype == jnp.float64 for v in shaped), "complete forward shape contract required")
            require(type(vg) is tuple and len(vg) == 2 and vg[0].shape == () and vg[0].dtype == jnp.float64
                    and jax.tree.structure(vg[1]) == jax.tree.structure(params), "generated scalar/complete gradient contract required")
            require(all(v.shape == p.shape and v.dtype == jnp.float64 for v, p in zip(vg[1], params, strict=True)),
                    "all five gradient leaf shapes required")
            forward = jax.make_jaxpr(self.forward)(*args)
            gradient = jax.make_jaxpr(self.value_gradient)(*args)
            require(len(inputs) == len(forward.jaxpr.invars) == len(gradient.jaxpr.invars), "actual3arg flattened input arity differs")
            source = self.owner.constant_bounds()
            values, fmeta = bounds.interpret(forward, inputs, limits.equations, _slot_bindings(forward, source), max_state_bytes=limits.state_bytes)
            joint, gmeta = bounds.interpret(gradient, inputs, limits.equations, _slot_bindings(gradient, source), max_state_bytes=limits.state_bytes)
            require(len(values) == len(self.owner.rows) and len(joint) == 6, "interpreter omitted outputs or a parameter gradient")
            outputs = [v.summary(tolerances.forward_atol, tolerances.forward_rtol) for v in values]
            scalar = joint[0].summary(tolerances.forward_atol, tolerances.forward_rtol)
            gradients = [v.summary(tolerances.gradient_atol, tolerances.gradient_rtol) for v in joint[1:]]
            result.update(outputs=outputs, objective=scalar, gradients=gradients, forward_graph=fmeta,
                          gradient_graph=gmeta, traced_forward=str(forward), traced_value_gradient=str(gradient),
                          admissible=all(v["admissible"] for v in [*outputs, scalar, *gradients]))
            if not result["admissible"]:
                result["reason"] = "conditional error bounds inconclusive; tolerances unchanged"
        except (ArithmeticError, ValueError, TypeError, NotImplementedError, MemoryError) as error:
            result.update(reason=str(error), exception=type(error).__name__)
        result["qualification_s"] = time.perf_counter() - started
        return result

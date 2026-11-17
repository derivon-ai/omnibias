# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private low-rank ridge objective-only streaming proposal, unvalidated.

No numerical imports occur at module import. The private builder consumes an
already verified runtime and an owned RidgeBase. It grants no runtime authority.
Only source-bound, after-idle numerical tests may qualify the actual graphs.
"""

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType


CONTRACT = "lowrank-ridge-objective-streaming/source-proposal-v1"
MAX_BYTES = 3 * 1024**3
MAX_EQUATIONS = 100000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, limit, label):
    require(type(value) is int and 1 <= value <= limit, label + " outside hard cap")
    return value


@dataclass(frozen=True)
class LeastSquares:
    """Explicit sum_k weights[k] mean((output_k-target_k(X))**2).

    Each target is intercepts[k] + input_scale * X[:,input_axis]. The supplied
    coefficients are ideal binary64 constants, with no extra factor one-half.
    Other global objectives are unsupported by this descriptor.
    """

    weights: tuple
    intercepts: tuple
    input_scale: float = 0.1
    input_axis: int = 0

    def validate(self, count, dimension):
        require(
            type(self.weights) is tuple
            and type(self.intercepts) is tuple
            and len(self.weights) == len(self.intercepts) == count,
            "one explicit scalar weight/intercept per output required",
        )
        require(
            all(
                type(value) is float and math.isfinite(value)
                for value in (*self.weights, *self.intercepts, self.input_scale)
            )
            and all(value >= 0 for value in self.weights),
            "finite binary64 constants and nonnegative weights required",
        )
        require(
            type(self.input_axis) is int and 0 <= self.input_axis < dimension,
            "declared target input axis out of range",
        )
        return self


def layout(batch, chunk):
    """Static row layout only; neither input validity nor proof is established."""
    positive(batch, MAX_BYTES // 8, "batch rows")
    positive(chunk, batch, "chunk rows")
    count = (batch + chunk - 1) // chunk
    return {
        "batch": batch,
        "chunk": chunk,
        "chunks": count,
        "pad_rows": count * chunk - batch,
    }


def storage_estimate(owner, shapes, batch, chunk, counter_bytes=0):
    """Conservative logical buffers, not allocator/compiler/process peak memory.

    Counts original and padded runtime inputs, both adjoint carries, one chunk's
    principal intermediates and prepared vectors. AD/compiler scratch and bounds
    interpreter storage are additional. No alias, donation, or overlap credit.
    """
    parts = layout(batch, chunk)
    h, d = owner.base.shape
    r = shapes[0][1]
    p = sum(math.prod(shape) for shape in shapes)
    prepared_vectors = sum(
        1 if kind == "laplacian" else len({term[0] for term in spec})
        for kind, spec in owner.rows
    )
    prepared = h * prepared_vectors + len(owner.rows)
    padded = parts["chunks"] * chunk * (d + h)
    return {
        "original_inputs_bytes": 8 * batch * (d + h),
        "padded_inputs_bytes": 8 * padded,
        "mask_bytes": parts["chunks"] * chunk,
        "parameters_bytes": 8 * p,
        "owner_device_constants_bytes": owner.base.nbytes
        + (owner.norms.nbytes if owner.need_norms else 0),
        "prepared_and_adjoint_carries_bytes": 8 * (2 * prepared + 2 * p + 1),
        "chunk_principal_intermediates_bytes": 8 * chunk * (d + 3 * h + r),
        "counter_bytes": counter_bytes,
        "scope": "logical retained buffers only; no full collocation outputs or scan history; compiler/AD/interpreter/allocator/process peaks separate",
    }


@dataclass(frozen=True, init=False, eq=False)
class StreamingProgram:
    """Generated scalar objective and complete five-leaf first gradient.

    Arbitrary callables are not accepted. Full-output storage, input/base
    derivatives, higher parameter derivatives, device training, packet ownership
    and containing-domain proof reuse are outside this source proposal.
    """

    _runtime: object
    owner: object
    objective_spec: LeastSquares
    provider: str
    layout: object
    max_state_bytes: int
    objective: object
    value_gradient: object
    _prepare: object
    _loss_chunk: object
    _row_orders: tuple
    _np: object

    def __init__(
        self,
        runtime,
        owner,
        *,
        batch,
        chunk,
        provider,
        objective,
        max_state_bytes=MAX_BYTES,
    ):
        required = {"jax", "jnp", "np", "modules"}
        require(
            type(runtime) is dict and required <= set(runtime),
            "already verified runtime required",
        )
        modules = runtime["modules"]
        require(
            {"ridge_diagonal_geometry", "providers", "bounds", "planner"}
            <= set(modules),
            "complete ridge/provider/bounds/planner modules required",
        )
        ridge = modules["ridge_diagonal_geometry"]
        require(isinstance(owner, ridge.RidgeBase), "owned selected RidgeBase required")
        owner.validate()
        jax, jnp, np = (runtime[key] for key in ("jax", "jnp", "np"))
        require(jax.config.x64_enabled, "binary64 configuration required")
        require(
            provider
            in {
                "parity",
                "polynomial",
                "sech",
                "recurrence",
                "jax_native",
                "jax_analytic",
            },
            "fixed existing provider required",
        )
        require(
            isinstance(objective, LeastSquares),
            "explicit streaming LeastSquares descriptor required",
        )
        h, d = owner.base.shape
        objective.validate(len(owner.rows), d)
        immutable_runtime = MappingProxyType(
            {
                "jax": jax,
                "jnp": jnp,
                "np": np,
                "modules": MappingProxyType(
                    {
                        key: modules[key]
                        for key in (
                            "ridge_diagonal_geometry",
                            "providers",
                            "bounds",
                            "planner",
                        )
                    }
                ),
            }
        )
        for key, value in (
            ("layout", MappingProxyType(layout(batch, chunk))),
            ("max_state_bytes", positive(max_state_bytes, MAX_BYTES, "state bytes")),
            ("_runtime", immutable_runtime),
            ("owner", owner),
            ("objective_spec", objective),
            ("provider", provider),
        ):
            object.__setattr__(self, key, value)
        a0, n0 = owner.base, owner.norms
        tower = modules["providers"].tower(owner.orders, provider)

        def prepare(params):
            u, v, _, readout, offset = params
            columns = {
                axis: a0[:, axis] + u @ v[axis]
                for axis in sorted({axis for axis, _ in owner.coordinate_powers})
            }
            coordinate = {
                (axis, power): columns[axis] ** power
                for axis, power in owner.coordinate_powers
            }
            norms = {}
            if owner.need_norms:
                av = a0 @ v
                # n0 carries its producer's nonzero construction error.
                norm = (
                    n0
                    + 2 * jnp.sum(av * u, axis=1)
                    + jnp.sum((u @ (v.T @ v)) * u, axis=1)
                )
                norms = {power: norm**power for power in owner.norm_powers}
            prepared = []
            for kind, specification in owner.rows:
                if kind == "laplacian":
                    vectors = (
                        (readout * norms[specification],)
                        if specification
                        else (readout,)
                    )
                    constant = (
                        jnp.asarray(0.0, jnp.float64) if specification else offset
                    )
                else:
                    grouped, zero = {}, 0.0
                    for order, active, coefficient in specification:
                        factor = jnp.ones_like(readout) * coefficient
                        for key in active:
                            factor = factor * coordinate[key]
                        grouped[order] = grouped.get(order, 0) + factor
                        if order == 0:
                            zero = coefficient
                    vectors = tuple(
                        readout * grouped[order] for order in sorted(grouped)
                    )
                    constant = zero * offset
                prepared.append((vectors, constant))
            return tuple(prepared)

        row_orders = tuple(
            (2 * specification,)
            if kind == "laplacian"
            else tuple(sorted({term[0] for term in specification}))
            for kind, specification in owner.rows
        )

        def loss_chunk(direct, prepared, points, projection, mask):
            u, v, bias = direct
            raw = projection + points @ v @ u.T
            z = jax.lax.optimization_barrier(raw) + bias
            values = dict(zip(owner.orders, tower(z), strict=True))
            target_input = objective.input_scale * points[:, objective.input_axis]
            result = jnp.asarray(0.0, jnp.float64)
            for orders, (vectors, constant), weight, intercept in zip(
                row_orders,
                prepared,
                objective.weights,
                objective.intercepts,
                strict=True,
            ):
                prediction = (
                    sum(
                        values[order] @ vector
                        for order, vector in zip(orders, vectors, strict=True)
                    )
                    + constant
                )
                residual = jnp.where(mask, prediction - (intercept + target_input), 0.0)
                # Divide by the ORIGINAL batch, never by chunk length/count.
                result = result + weight * jnp.sum(residual * residual) / batch
            return result

        def chunks(points, projection):
            require(
                points.shape == (batch, d) and projection.shape == (batch, h),
                "fixed runtime X/Z0 shapes required",
            )
            # Repeat actual row0: zero padding need not be inside declared boxes.
            padding = self.layout["pad_rows"]
            if padding:
                points = jnp.concatenate(
                    (points, jnp.broadcast_to(points[0], (padding, d))), axis=0
                )
                projection = jnp.concatenate(
                    (projection, jnp.broadcast_to(projection[0], (padding, h))), axis=0
                )
            shape = self.layout["chunks"], chunk
            masks = (jnp.arange(shape[0] * chunk) < batch).reshape(shape)
            return points.reshape((*shape, d)), projection.reshape((*shape, h)), masks

        local_vg = jax.value_and_grad(loss_chunk, argnums=(0, 1))

        def preflight(params, points, projection):
            ridge._parameter_contract(params, points, projection, h, d)
            estimate = storage_estimate(
                owner, tuple(value.shape for value in params), batch, chunk
            )
            required_bytes = sum(
                value for key, value in estimate.items() if key.endswith("_bytes")
            )
            require(
                required_bytes <= self.max_state_bytes,
                "logical streaming payload exceeds state budget",
            )

        def objective_only(params, points, projection):
            preflight(params, points, projection)
            prepared = prepare(params)
            direct = params[:3]

            def body(total, inputs):
                return total + loss_chunk(direct, prepared, *inputs), None

            return jax.lax.scan(
                body, jnp.asarray(0.0, jnp.float64), chunks(points, projection)
            )[0]

        def value_gradient(params, points, projection):
            preflight(params, points, projection)
            # This VJP's primal and tape are outside the collocation scan.
            prepared, pullback = jax.vjp(prepare, params)
            direct = params[:3]
            initial = (
                jnp.asarray(0.0, jnp.float64),
                jax.tree.map(jnp.zeros_like, direct),
                jax.tree.map(jnp.zeros_like, prepared),
            )

            def body(carry, inputs):
                total, adjoint_direct, adjoint_prepared = carry
                value, (local_direct, local_prepared) = local_vg(
                    direct, prepared, *inputs
                )
                return (
                    total + value,
                    jax.tree.map(jnp.add, adjoint_direct, local_direct),
                    jax.tree.map(jnp.add, adjoint_prepared, local_prepared),
                ), None

            value, adjoint_direct, adjoint_prepared = jax.lax.scan(
                body, initial, chunks(points, projection)
            )[0]
            geometry_gradient = pullback(adjoint_prepared)[0]
            direct_gradient = (
                *adjoint_direct,
                jnp.zeros_like(params[3]),
                jnp.zeros_like(params[4]),
            )
            return value, jax.tree.map(jnp.add, direct_gradient, geometry_gradient)

        # These bindings carry the actual fixed A0/n0 producer enclosures. An
        # exact constant-slot table is built separately for each emitted graph.
        objective_only.constant_bounds = owner.constant_bounds()
        value_gradient.constant_bounds = owner.constant_bounds()
        for key, value in (
            ("objective", objective_only),
            ("value_gradient", value_gradient),
            ("_prepare", prepare),
            ("_loss_chunk", loss_chunk),
            ("_row_orders", row_orders),
            ("_np", np),
        ):
            object.__setattr__(self, key, value)

    def report(self, params):
        self.owner.validate()
        shapes = tuple(value.shape for value in params)
        estimate = storage_estimate(
            self.owner, shapes, self.layout["batch"], self.layout["chunk"]
        )
        return {
            "contract": CONTRACT,
            "layout": dict(self.layout),
            "provider": self.provider,
            "objective": asdict(self.objective_spec),
            "physical_gradients": ["U", "V", "bias", "readout", "offset"],
            "activation_orders": list(self.owner.orders),
            "potential_first_gradient_orders": sorted(
                set(self.owner.orders) | {order + 1 for order in self.owner.orders}
            ),
            "dependencies": "actual provider recurrence and AD circuit retained; demand list is not an accuracy proof",
            "preparation_pullbacks_per_evaluation": 1,
            "full_collocation_outputs_stored": False,
            "scan_history_stored": False,
            "input_storage": "full original X/Z0 plus padded input views/copies; no O(1) total-space claim",
            "storage": estimate,
            "max_state_bytes": self.max_state_bytes,
            "numerical_qualification": False,
            "runtime_certificate_issued": False,
            "performance_claim": False,
            "public_promotion": False,
        }

    def qualify(self, params, packet, domain, *, limits=None, tolerances=None):
        """Conditional bounds on ACTUAL scalar and emitted full-gradient graph.

        This method grants no execution certificate. Physical X/A0 are fixed;
        Z0 and n0 retain producer errors. Per-call operand validity and supplied
        graph arithmetic assumptions remain explicit obligations.
        """
        runtime = self._runtime
        jax, np = runtime["jax"], runtime["np"]
        ridge, bounds, planner = (
            runtime["modules"][key]
            for key in ("ridge_diagonal_geometry", "bounds", "planner")
        )
        limits = planner.Limits(cells=1) if limits is None else limits
        tolerances = planner.Tolerances() if tolerances is None else tolerances
        require(isinstance(limits, planner.Limits), "typed bounded limits required")
        require(
            isinstance(tolerances, planner.Tolerances),
            "typed unchanged tolerances required",
        )
        require(
            asdict(tolerances) == asdict(planner.Tolerances()),
            "unchanged tolerances required",
        )
        positive(limits.equations, MAX_EQUATIONS, "equations")
        positive(limits.state_bytes, self.max_state_bytes, "interpreter state bytes")
        require(
            limits.cells == 1, "one explicit fixed-X cell supported in this proposal"
        )
        self.owner.validate()
        require(
            isinstance(packet, ridge.RidgeProjection) and packet.owner is self.owner,
            "actual owned ridge producer packet required",
        )
        packet.validate()
        require(isinstance(domain, planner.Domain), "complete typed Domain required")
        require(
            type(domain.parameters) is tuple and len(domain.parameters) == 5,
            "all five parameter domains required",
        )
        ridge._parameter_contract(
            params, packet.points, packet.projection, *self.owner.base.shape
        )
        owned = planner.Domain(
            tuple(bounds.immutable_bound(value) for value in domain.parameters),
            bounds.immutable_bound(domain.inputs),
        )
        x = packet.metadata.values["points"]
        require(
            x.shape[0] == self.layout["batch"]
            and owned.inputs.lo.shape == x.shape
            and np.array_equal(owned.inputs.lo, x)
            and np.array_equal(owned.inputs.hi, x),
            "exact owned real-X point domain required",
        )
        require(
            all(
                bound.lo.shape == value.shape
                for bound, value in zip(owned.parameters, params, strict=True)
            )
            and all(
                not np.any(value.error) for value in (*owned.parameters, owned.inputs)
            ),
            "exact-shape zero-initial-error real parameter/X domains required",
        )
        require(
            bool(
                planner.runtime_guard(
                    params,
                    packet.points,
                    tuple(value.lo for value in owned.parameters),
                    tuple(value.hi for value in owned.parameters),
                    owned.inputs.lo,
                    owned.inputs.hi,
                )
            ),
            "actual parameters outside supplied domain",
        )
        construction = packet.construction_bound()
        inputs = [*owned.parameters, owned.inputs, construction]
        estimate = self.report(params)
        estimated_bytes = sum(
            value
            for key, value in estimate["storage"].items()
            if key.endswith("_bytes")
        )
        require(
            estimated_bytes <= self.max_state_bytes,
            "logical streaming payload exceeds state budget",
        )
        require(
            3
            * sum(value.nbytes for value in (*params, packet.points, packet.projection))
            <= limits.state_bytes,
            "initial bound payload exceeds interpreter state budget",
        )
        result = {
            **estimate,
            "admissible": False,
            "scope": "conditional emitted objective scalar and complete first parameter gradient; no full-output or higher-gradient certificate",
            "tolerances": asdict(tolerances),
            "limits": asdict(limits),
            "construction_error_preserved": True,
            "assumptions": "binary64 RN, gradual underflow, finite intermediates, compatible compiler transformations/reductions, declared native-transcendental bounds, exact owned transfers",
        }
        try:
            args = params, packet.points, packet.projection
            emitted = [
                jax.make_jaxpr(function)(*args)
                for function in (self.objective, self.value_gradient)
            ]
            source = self.owner.constant_bounds()
            interpreted = []
            metadata = []
            for graph, expected in zip(emitted, (1, 6), strict=True):
                require(
                    len(graph.jaxpr.invars) == 7
                    and len(graph.jaxpr.outvars) == expected,
                    "actual scalar/full5 graph inventory differs",
                )
                values, meta = bounds.interpret(
                    graph,
                    inputs,
                    limits.equations,
                    ridge._slot_bindings(graph, source),
                    max_state_bytes=limits.state_bytes,
                )
                require(
                    len(values) == expected, "interpreter omitted a requested output"
                )
                interpreted.append(values)
                metadata.append(meta)
            scalar = interpreted[0][0].summary(
                tolerances.forward_atol, tolerances.forward_rtol
            )
            vg_scalar = interpreted[1][0].summary(
                tolerances.forward_atol, tolerances.forward_rtol
            )
            gradient = [
                value.summary(tolerances.gradient_atol, tolerances.gradient_rtol)
                for value in interpreted[1][1:]
            ]
            result.update(
                objective=scalar,
                value_gradient_objective=vg_scalar,
                gradients=gradient,
                traced_objective=str(emitted[0]),
                traced_value_gradient=str(emitted[1]),
                interpreter_metadata=metadata,
                admissible=all(
                    value["admissible"] for value in (scalar, vg_scalar, *gradient)
                ),
            )
            if not result["admissible"]:
                result["reason"] = (
                    "conditional error bounds inconclusive; tolerances unchanged"
                )
        except (
            ArithmeticError,
            ValueError,
            TypeError,
            NotImplementedError,
            MemoryError,
        ) as error:
            result.update(reason=str(error), exception=type(error).__name__)
        result["conditional_bounds_satisfy_contract"] = result["admissible"]
        return result
